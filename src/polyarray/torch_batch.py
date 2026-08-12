"""Torch backend for batched execution of a :class:`~polyarray.ir.Program`.

:func:`batched_torch` is the torch counterpart of :func:`polyarray.batch.batched_run`: it
lowers a program to torch through ``pyab`` and runs it batched with ``torch.vmap``, so the
batching stays an IR concept and torch is reached only through the one lowering path::

    program ──pyab.compile_torch──▶ per-element torch fn ──torch.vmap──▶ batched over axis 0

``torch`` and ``pyarraybackend`` are optional dependencies, imported lazily; the core
stack never imports torch. On a GPU or a large batch this path can beat the numpy path,
and on tiny CPU arrays it runs roughly on par with :func:`batched_run`.

``pyab``'s torch backend bootstraps a ``torch.distributed`` process group from mpi4py at
module load. :func:`ensure_torch_pg` pre-initializes a 1-rank ``gloo`` group so that
bootstrap short-circuits, removing the mpi4py and MPI dependency for single-process use.

The backend requires ``torch`` and ``pyarraybackend`` and raises :class:`RuntimeError`
with a clear message when either is absent. Not every op has a ``pyab`` torch lowering: a
front-end op supplies its own by carrying a ``__pyab_lower__`` hook on its class, which
``pyab``'s ``_render_op`` discovers, and a missing lowering surfaces as the ``pyab``
``NotImplementedError``, which a caller can catch to fall back to :func:`batched_run`.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from .ir import Program
from .pyab import OpLowering

__all__ = [
    "batched_torch",
    "ensure_torch_pg",
    "feec_op_lowerings",
    "switch_vmap_op_lowerings",
    "torch_available",
]


def torch_available() -> bool:
    """Report whether both ``torch`` and ``pyarraybackend`` import, i.e. the backend is usable."""
    import importlib.util as u
    return u.find_spec("torch") is not None and u.find_spec("pyarraybackend") is not None


# --- pyab op lowerings ------------------------------------------------------------------
# A front-end op carries its own lowering as a ``__pyab_lower__(self, builder, args, low)``
# method on the op class, which pyab's ``_render_op`` discovers by ``getattr`` — the twin of
# ``numpy_source``'s ``__numpy_source__`` hook. ``SwitchOp`` and ``AssertOp`` are native pyab
# builtins. Plain ``LowerOpts()`` therefore lowers everything, and no ``op_lowerings`` dict is
# needed; the two accessors below stay exported, and empty, for callers that still pass one.

def switch_vmap_op_lowerings() -> dict[str, OpLowering]:
    """Return an empty op-lowering table for ``SwitchOp``.

    The vmap-safe ``SwitchOp`` lowering — a one-hot over stacked branches — is a pyab builtin
    (``pyab._Lowerer._switch_expr``): a dynamic or batched scrutinee emits the pure-arithmetic
    one-hot that lowers under ``torch.vmap``, and a concrete ``Const`` scrutinee takes a direct
    branch-pick fast path. No override is needed.
    """
    return {}


def feec_op_lowerings() -> dict[str, OpLowering]:
    """Return an empty op-lowering table for the front-end ops.

    Those ops carry their own ``__pyab_lower__`` hooks, so plain ``LowerOpts()`` lowers the whole
    residual with no override.
    """
    return {}


def _free_tcp_port() -> int:
    """Return a currently-free localhost TCP port.

    Binds to port 0, reads the port the OS assigned, and releases it. There is a tiny window
    between release and ``init_process_group`` rebinding it, but each port is ephemeral and
    per-process, so parallel xdist workers and concurrent runs each get a distinct one. A fixed
    port instead collides across them, hanging or leaving a dead process group.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ensure_torch_pg(port: int | None = None) -> None:
    """Pre-initialize a 1-rank ``gloo`` ``torch.distributed`` process group.

    ``pyab``'s torch backend calls ``ensure_pg()`` at generated-module load, which bootstraps the process
    group from mpi4py. That check short-circuits when ``torch.distributed`` is already initialized, before
    importing mpi4py, so initializing a 1-rank group here avoids the mpi4py / MPI dependency for
    single-process use. It is idempotent.

    Parameters
    ----------
    port
        Port for the group's rendezvous. Defaults to a free ephemeral port, so concurrent
        processes do not collide. An already-set ``MASTER_PORT`` environment variable wins,
        so a hand-pinned port is honoured.
    """
    import torch.distributed as td
    if td.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port if port is not None else _free_tcp_port()))
    td.init_process_group(backend="gloo", rank=0, world_size=1)


def batched_torch(program: Program, values: Mapping[str, npt.ArrayLike]) -> np.ndarray:
    """Evaluate ``program`` over a whole batch on torch.

    Lowers ``program`` to a per-element torch function via ``pyab.compile_torch`` and batches it
    with ``torch.vmap`` over axis 0 of every input, in program-input order. Matches the
    per-element loop to about machine epsilon; torch's contraction kernels differ from numpy's
    by roughly one ulp.

    Parameters
    ----------
    program
        The program to evaluate.
    values
        Input name to a ``(B, *decl_shape)`` array carrying the batch on axis 0.

    Returns
    -------
    numpy.ndarray
        The ``result`` output, shaped ``(B, *out_shape)``.

    Raises
    ------
    RuntimeError
        If ``torch`` or ``pyarraybackend`` is not installed.
    """
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
