"""Torch reliability for the polyarray test suite.

Two chronic problems made the ``[torch]`` tests unrunnable / flaky:

1. ``pyab.compile_torch`` emits ``ensure_pg()`` at generated-module load, which imports
   mpi4py — absent in the ``[torch]`` venv, so every direct-``compile_torch`` test died with
   ``ModuleNotFoundError: No module named 'mpi4py'``.
2. The 1-rank ``gloo`` group used a FIXED ``MASTER_PORT`` (29591), so ``pytest -n auto``
   workers and concurrent agents collided on it (hang / silently-dead process).

Both are fixed by pre-initializing ONE 1-rank ``gloo`` group per test process on a FREE
ephemeral port: ``ensure_pg`` then short-circuits on ``is_initialized()`` BEFORE importing
mpi4py, and each xdist worker (a separate process) binds its own distinct port. No-op when
torch is not installed (the ``[torch]`` extra), so the numpy-only venv is unaffected.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _torch_single_rank_pg() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        return  # torch not installed (numpy-only venv) — nothing to pre-init
    from polyarray.torch_batch import ensure_torch_pg
    try:
        ensure_torch_pg()  # free ephemeral port, mpi4py-free, idempotent
    except Exception:
        # torch present but the process group could not be stood up — leave it to the
        # individual torch tests to skip/fail with their own diagnostics, don't abort the run.
        pass
