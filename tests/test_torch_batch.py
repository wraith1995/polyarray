"""``polyarray.torch_batch.batched_torch`` — the torch backend (pyab → torch.vmap) matches batched_run.

Skipped unless BOTH ``torch`` and ``pyarraybackend`` are importable (optional deps; the core stack never
imports torch). torch's contraction kernels differ from numpy's by ~1 ULP, so the tolerance is loose."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarraybackend")

from polyarray import Program
from polyarray.ir import SymInput, Provenance, SymbolicBudget
from polyarray.batch import batched_run
from polyarray.torch_batch import batched_torch, torch_available


def _prov(label="t"):
    return Provenance(kind="coeff", origin="t", index=(), label=label)


@pytest.mark.skipif(not torch_available(), reason="torch/pyarraybackend not available")
def test_batched_torch_matches_batched_run():
    # pinv (Stmt lane) composed with an einsum: result = pinv(A) @ x
    p = Program("f", inputs=[SymInput("A", (4, 4), _prov("A")), SymInput("x", (4,), _prov("x"))],
                budget=SymbolicBudget.force_stmts())
    p.add_output("result", p.input("A").pinv().einsum("ij,j->i", p.input("x")).cells)
    B = 32
    rng = np.random.default_rng(3)
    As = np.stack([rng.standard_normal((4, 4)) + 4 * np.eye(4) for _ in range(B)])
    xs = rng.standard_normal((B, 4))
    got = batched_torch(p, {"A": As, "x": xs})
    ref = batched_run(p, {"A": As, "x": xs})
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)
