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
    cm = pyab.compile_torch(program, name="f")
    fn = cm.module.f
    args = [torch.as_tensor(np.asarray(values[inp.name], dtype=float)) for inp in program.inputs]
    out = torch.vmap(fn)(*args)
    result = out[0] if isinstance(out, tuple) else out
    return np.asarray(result.detach())
