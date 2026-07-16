"""``polyarray.batch.batched_run`` — batched execution equals the per-element loop."""
import numpy as np
import pytest

from polyarray import Program
from polyarray.ir import SymInput, Provenance
from polyarray.batch import batched_run, BatchUnsupported


def _prov(label="t"):
    return Provenance(kind="coeff", origin="t", index=(), label=label)


def _loop(prog, batched):
    B = next(iter(batched.values())).shape[0]
    return np.stack([np.asarray(prog.run({k: v[b] for k, v in batched.items()})["result"], float)
                     for b in range(B)])


def _check(prog, batched, rtol=1e-12, atol=1e-14):
    got = np.asarray(batched_run(prog, batched), float)
    ref = _loop(prog, batched)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
    return got


B = 7
_rng = np.random.default_rng(0)


def test_einsum_cell_lane():
    # small einsum stays in the CELL lane (symbolic RFs over per-element atoms)
    p = Program("f", inputs=[SymInput("A", (3, 3), _prov("A")), SymInput("x", (3,), _prov("x"))])
    p.add_output("result", p.input("A").einsum("ij,j->i", p.input("x")).cells)
    assert not p.statements
    _check(p, {"A": _rng.standard_normal((B, 3, 3)), "x": _rng.standard_normal((B, 3))})


def test_inverse_rational_cell_lane():
    p = Program("g", inputs=[SymInput("A", (2, 2), _prov("A"))])
    p.add_output("result", p.input("A").inverse().cells)
    As = np.stack([_rng.standard_normal((2, 2)) + 3 * np.eye(2) for _ in range(B)])
    _check(p, {"A": As})


def test_pinv_stmt_lane():
    from polyarray.ir import SymbolicBudget
    p = Program("m", inputs=[SymInput("A", (4, 4), _prov())], budget=SymbolicBudget.force_stmts())
    p.add_output("result", p.input("A").pinv().cells)
    assert p.statements                                   # forced into the Stmt lane
    As = np.stack([_rng.standard_normal((4, 4)) + 4 * np.eye(4) for _ in range(B)])
    _check(p, {"A": As}, rtol=1e-8, atol=1e-10)


def test_batch_unsupported_is_notimplemented():
    # the caller (pointwise `_assemble`) catches this to fall back to the per-element loop
    assert issubclass(BatchUnsupported, NotImplementedError)
