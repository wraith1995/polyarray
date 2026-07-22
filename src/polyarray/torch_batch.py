"""Torch backend for batched execution — lower a :class:`~polyarray.ir.Program` to torch via ``pyab`` and
run it batched with ``torch.vmap``.

This is the torch counterpart of :func:`polyarray.batch.batched_run` (numpy). Rather than a hand-written
torch evaluator, it uses the sanctioned lowering path: ``pyab.compile_torch(program)`` compiles the IR to a
per-element torch function, and ``torch.vmap`` batches it over the leading axis — so the batching stays an
IR concept and torch is reached only through ``pyab`` (torch/pyarraybackend are OPTIONAL deps, imported
lazily; the core stack never imports torch).

On a GPU or large batch this can beat the numpy path; on tiny CPU arrays (the FEEC per-point residual) it is
roughly on par with :func:`batched_run`. ``pyab``'s torch backend bootstraps a ``torch.distributed`` process
group from mpi4py at module load; :func:`ensure_torch_pg` pre-initializes a 1-rank ``gloo`` group so that
bootstrap short-circuits (no mpi4py / MPI dependency for single-process use).

Requires ``torch`` and ``pyarraybackend``; raises :class:`RuntimeError` with a clear message if absent. Not
every op has a ``pyab`` torch lowering — front-end (grassmann) ops may need ``pyab`` ``op_lowerings``; a
missing lowering surfaces as the ``pyab`` ``NotImplementedError`` (a caller can fall back to ``batched_run``).
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from .ir import Program

__all__ = ["batched_torch", "ensure_torch_pg", "torch_available"]


def torch_available() -> bool:
    """True iff both ``torch`` and ``pyarraybackend`` import — the torch backend is usable."""
    import importlib.util as u
    return u.find_spec("torch") is not None and u.find_spec("pyarraybackend") is not None


# --- pyab op-lowerings for the front-end (grassmann-origin) ops the FEEC residual carries ------------
# These render each op's PER-ELEMENT semantics to pyab AST (``torch.vmap`` supplies the batch axis, so no
# batch bookkeeping here). Keyed by op CLASS NAME and supplied to pyab via ``LowerOpts.op_lowerings`` — so
# the knowledge lives in polyarray (this module), pyab stays generic, and grassmann is never imported.
# A renderer has signature ``(op, builder, args, lowerer) -> list[expr]`` (see pyab `_render_op`).

def _reshape_low(fn, _b, args, low):                         # _ReshapeOp(shape): A.reshape(shape)
    c = low.core
    return [c.ReshapeExpr(a=args[0], shape=tuple(c.IntLit(value=int(d)) for d in fn.shape))]


def _axislen_low(fn, _b, args, low):                         # _AxisLenOp(axis): x.shape[axis] (0-d int)
    c = low.core
    return [c.AccessExpr(base=c.ShapeExpr(x=args[0]), index=(c.IntLit(value=int(fn.axis)),))]


def _firstcols_low(_fn, _b, args, low):                      # _FirstColsOp: A[:, :int(rank)]
    c = low.core
    rank = c.CallExpr(fn=c.Var(name="int"), args=(args[1],))
    return [c.AccessExpr(base=args[0], index=(c.Slice(), c.Slice(stop=rank)))]


def _project_low(_fn, _b, args, low):                        # _ProjectOp: P.T @ v.reshape(-1)
    c = low.core
    p_t = c.TransposeExpr(a=args[0], axes=None)
    v_flat = c.ReshapeExpr(a=args[1], shape=(c.IntLit(value=-1),))
    return [c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=p_t, rhs=v_flat)]


def _embed_low(fn, _b, args, low):                           # _EmbedOp(shape): (P @ vsub).reshape(shape)
    c = low.core
    mm = c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=args[0], rhs=args[1])
    if fn.shape:
        return [c.ReshapeExpr(a=mm, shape=tuple(c.IntLit(value=int(d)) for d in fn.shape))]
    return [mm]


def _add_low(_fn, _b, args, low):                            # _AddOp: a + b
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.ADD, lhs=args[0], rhs=args[1])]


def _scale_low(fn, _b, args, low):                           # _ScaleOp(factor): factor * a
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=c.FloatLit(value=float(fn.factor)), rhs=args[0])]


def _assert_low(_fn, _b, args, low):                         # AssertOp: runtime check → value passthrough
    return [args[0]]


def _const_low(fn, _b, _args, low):                          # _ConstOp: rematerialize the frozen constant
    arr = np.frombuffer(fn.data_bytes, dtype=fn.dtype).reshape(fn.shape)
    return [low._const_expr(np.asarray(arr, dtype=float))]


def feec_op_lowerings() -> dict:
    """pyab ``op_lowerings`` for the grassmann-origin ops in the FEEC residual (by class name)."""
    return {
        "_ReshapeOp": _reshape_low,
        "_AxisLenOp": _axislen_low,
        "_FirstColsOp": _firstcols_low,
        "_ProjectOp": _project_low,
        "_EmbedOp": _embed_low,
        "_AddOp": _add_low,
        "_ScaleOp": _scale_low,
        "AssertOp": _assert_low,
        "_ConstOp": _const_low,
    }


def ensure_torch_pg(port: int = 29591) -> None:
    """Pre-initialize a 1-rank ``gloo`` ``torch.distributed`` process group.

    ``pyab``'s torch backend calls ``ensure_pg()`` at generated-module load, which bootstraps the process
    group from mpi4py. That check short-circuits when ``torch.distributed`` is already initialized (BEFORE
    importing mpi4py), so initializing a 1-rank group here avoids the mpi4py / MPI dependency for
    single-process use. Idempotent."""
    import torch.distributed as td
    if td.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    td.init_process_group(backend="gloo", rank=0, world_size=1)


def batched_torch(program: Program, values: Mapping[str, Any]) -> np.ndarray:
    """Evaluate ``program`` for a whole batch on torch: each ``values[name]`` carries a leading batch axis
    ``(B, *decl_shape)``. Returns the ``result`` output as a numpy array ``(B, *out_shape)``.

    Lowers ``program`` to a per-element torch function via ``pyab.compile_torch`` and batches it with
    ``torch.vmap`` over axis 0 of every input (in program-input order). Numerically matches the per-element
    loop to ~machine epsilon (torch's contraction kernels differ from numpy's by ~1 ULP)."""
    if not torch_available():
        raise RuntimeError(
            "batched_torch requires `torch` and `pyarraybackend`; install them (optional deps) to use the "
            "torch backend, or use polyarray.batch.batched_run for the numpy path."
        )
    import torch

    from . import pyab
    ensure_torch_pg()
    opts = pyab.LowerOpts(op_lowerings=feec_op_lowerings())    # lower the FEEC front-end ops (by name)
    cm = pyab.compile_torch(program, name="f", opts=opts)
    fn = cm.module.f
    args = [torch.as_tensor(np.asarray(values[inp.name], dtype=float)) for inp in program.inputs]
    out = torch.vmap(fn)(*args)
    result = out[0] if isinstance(out, tuple) else out
    return np.asarray(result.detach())
