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


# ── the rules that shipped DEAD (keyed `_ScaleOp` … `_EmbedOp`, classes that never existed) ──────
# Each is structural or elementwise, so batched == per-element BYTE-for-byte (atol=rtol=0), not
# merely close. See `test_batch_rules_named.py` for the guard that keeps the keys honest.

def _stmt_prog(name, fn, ins, out_shape, budget_force=True):
    """A one-Stmt program: inputs `ins` = [(name, shape)], output = `fn(*inputs)`."""
    from polyarray.ir import OutSpec, SymbolicBudget
    p = Program(name, inputs=[SymInput(n, s, _prov(n)) for n, s in ins],
                budget=SymbolicBudget.force_stmts() if budget_force else None)
    (out,) = p.emit_stmt(fn, [p.input(n) for n, _ in ins], [OutSpec("r", out_shape)])
    p.add_output("result", out.cells)
    return p


def _exact(prog, batched):
    """Batched result must equal the per-element loop EXACTLY."""
    got = np.asarray(batched_run(prog, batched), float)
    ref = _loop(prog, batched)
    assert got.shape == ref.shape, (got.shape, ref.shape)
    np.testing.assert_array_equal(got, ref)


def test_reshape_op_batched():
    from polyarray.ir import ReshapeOp
    p = _stmt_prog("rs", ReshapeOp((2, 3)), [("A", (6,))], (2, 3))
    _exact(p, {"A": _rng.standard_normal((B, 6))})


def test_scale_op_batched():
    from polyarray.ir import ScaleOp
    p = _stmt_prog("sc", ScaleOp(2.5), [("A", (3,))], (3,))
    _exact(p, {"A": _rng.standard_normal((B, 3))})


def test_scale_by_op_batched_scalar():
    """`ScaleByOp` had NO rule at all. The runtime scalar is itself batched here — the case that
    needs the trailing-axis reshape, not a bare broadcast."""
    from polyarray.ir import ScaleByOp
    p = _stmt_prog("sb", ScaleByOp(), [("A", (3,)), ("s", ())], (3,))
    _exact(p, {"A": _rng.standard_normal((B, 3)), "s": _rng.standard_normal((B,))})


def test_add_op_three_operands():
    """The dead `_AddOp` body summed only TWO operands; `AddOp` is a left-fold over `n`. Had the
    key been right without this fix, a 3-way sum would have silently dropped a term."""
    from polyarray.ir import AddOp
    p = _stmt_prog("ad", AddOp(3), [("A", (3,)), ("C", (3,)), ("D", (3,))], (3,))
    _exact(p, {"A": _rng.standard_normal((B, 3)), "C": _rng.standard_normal((B, 3)),
               "D": _rng.standard_normal((B, 3))})


def test_axis_len_op_is_batch_invariant():
    from polyarray.ir import AxisLenOp, ScaleByOp, OutSpec, SymbolicBudget
    p = Program("al", inputs=[SymInput("A", (4, 3), _prov("A"))],
                budget=SymbolicBudget.force_stmts())
    (n,) = p.emit_stmt(AxisLenOp(0), [p.input("A")], [OutSpec("n", ())])
    (out,) = p.emit_stmt(ScaleByOp(), [p.input("A"), n], [OutSpec("r", (4, 3))])
    p.add_output("result", out.cells)
    _exact(p, {"A": _rng.standard_normal((B, 4, 3))})       # axis 0 stays 4 under the batch axis


def test_first_and_last_cols_batched():
    from polyarray.ir import FirstColsOp, LastColsOp, ConstOp
    for op, shape in ((FirstColsOp(), (4, 2)), (LastColsOp(), (4, 3))):
        from polyarray.ir import OutSpec, SymbolicBudget, Const
        p = Program("fc", inputs=[SymInput("A", (4, 5), _prov("A"))],
                    budget=SymbolicBudget.force_stmts())
        (out,) = p.emit_stmt(op, [p.input("A"), Const(np.asarray(2))], [OutSpec("r", shape)])
        p.add_output("result", out.cells)
        _exact(p, {"A": _rng.standard_normal((B, 4, 5))})


def test_project_and_embed_batched():
    from polyarray.ir import ProjectOp, EmbedOp, OutSpec, SymbolicBudget, Const
    P = np.linalg.qr(_rng.standard_normal((4, 4)))[0][:, :2]          # a 4→2 orthonormal projector
    p = Program("pj", inputs=[SymInput("v", (4,), _prov("v"))], budget=SymbolicBudget.force_stmts())
    (sub,) = p.emit_stmt(ProjectOp(), [Const(P), p.input("v")], [OutSpec("s", (2,))])
    (amb,) = p.emit_stmt(EmbedOp((4,)), [Const(P), sub], [OutSpec("a", (4,))])
    p.add_output("result", amb.cells)
    _exact(p, {"v": _rng.standard_normal((B, 4))})
