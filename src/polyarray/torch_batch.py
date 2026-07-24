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
every op has a ``pyab`` torch lowering — a front-end op supplies its own by carrying a ``__pyab_lower__``
hook on its class (discovered by ``pyab``'s ``_render_op``); a missing lowering surfaces as the ``pyab``
``NotImplementedError`` (a caller can fall back to ``batched_run``).
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from .ir import Program

__all__ = [
    "batched_torch",
    "ensure_torch_pg",
    "feec_op_lowerings",
    "switch_vmap_op_lowerings",
    "torch_available",
]


def torch_available() -> bool:
    """True iff both ``torch`` and ``pyarraybackend`` import — the torch backend is usable."""
    import importlib.util as u
    return u.find_spec("torch") is not None and u.find_spec("pyarraybackend") is not None


# --- pyab op-lowerings for the front-end ops — now the ONE canonical op-carried-hook path ------------
# The grassmann-origin FEEC ops (``_ReshapeOp``/``_AxisLenOp``/``_FirstColsOp``/``_ProjectOp``/
# ``_EmbedOp``/``_AddOp``/``_ScaleOp``/``_ConstOp``) each now carry their OWN pyab lowering as a
# ``__pyab_lower__(self, builder, args, low)`` method on the op class (grassmann ``lower/represent.py`` +
# ``lower/space_basis.py``) — the sanctioned twin of ``numpy_source``'s ``__numpy_source__`` hook, which
# pyab's ``_render_op`` discovers by ``getattr``. The orientation ``SwitchOp`` and ``AssertOp`` are native
# pyab builtins (``pyab._switch_expr`` is now the canonical vmap-safe one-hot; ``AssertOp`` passes through).
# So NO ``op_lowerings`` dict is needed anymore: plain ``LowerOpts()`` lowers everything.
#
# The two functions below stay EXPORTED for back-compat (older callers still write
# ``LowerOpts(op_lowerings=feec_op_lowerings())``) but are now empty SHIMS — an empty dict means "let the
# hooks + native builtins do the work". They add nothing and override nothing.

def switch_vmap_op_lowerings() -> dict:
    """Back-compat shim — returns ``{}``.

    The vmap-safe ``SwitchOp`` lowering (one-hot · stacked branches) is now the CANONICAL pyab
    builtin (``pyab._Lowerer._switch_expr``): for a dynamic/batched scrutinee it emits the pure-
    arithmetic one-hot that lowers under ``torch.vmap`` (and a concrete-``Const`` scrutinee takes a
    direct branch-pick fast path). No ``op_lowerings`` override is needed; kept only so callers that
    still merge this dict keep working."""
    return {}


def feec_op_lowerings() -> dict:
    """Back-compat shim — returns ``{}``.

    The grassmann-origin FEEC ops now carry their own ``__pyab_lower__`` hooks (discovered by pyab's
    ``_render_op``) and ``SwitchOp``/``AssertOp`` are native builtins, so plain ``LowerOpts()`` lowers
    the whole FEEC residual. Kept exported (and empty) so callers that still write
    ``LowerOpts(op_lowerings=feec_op_lowerings())`` keep working with no behaviour change."""
    return {}


def _free_tcp_port() -> int:
    """A currently-free localhost TCP port (bind to :0, read the OS-assigned port, release).

    There is a tiny window between release and ``init_process_group`` rebinding it, but the
    port is per-process and ephemeral, so parallel xdist workers / concurrent runs each get a
    distinct one — unlike a fixed port, which they all collide on (hang / silent dead PG)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ensure_torch_pg(port: int | None = None) -> None:
    """Pre-initialize a 1-rank ``gloo`` ``torch.distributed`` process group.

    ``pyab``'s torch backend calls ``ensure_pg()`` at generated-module load, which bootstraps the process
    group from mpi4py. That check short-circuits when ``torch.distributed`` is already initialized (BEFORE
    importing mpi4py), so initializing a 1-rank group here avoids the mpi4py / MPI dependency for
    single-process use. Idempotent.

    ``port`` defaults to a FREE ephemeral port (was a fixed 29591, which collided under
    ``pytest -n auto`` / concurrent agents). An already-set ``MASTER_PORT`` env still wins
    (``setdefault``), so a hand-pinned port is honoured."""
    import torch.distributed as td
    if td.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port if port is not None else _free_tcp_port()))
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
    # Plain opts — the front-end ops carry their own ``__pyab_lower__`` hooks and SwitchOp/AssertOp are
    # native pyab builtins, so no ``op_lowerings`` dict is needed (the ONE canonical lowering path).
    opts = pyab.LowerOpts()
    cm = pyab.compile_torch(program, name="f", opts=opts)
    fn = cm.module.f
    args = [torch.as_tensor(np.asarray(values[inp.name], dtype=float)) for inp in program.inputs]
    out = torch.vmap(fn)(*args)
    result = out[0] if isinstance(out, tuple) else out
    return np.asarray(result.detach())
