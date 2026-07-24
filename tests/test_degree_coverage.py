"""Every polyarray Stmt.fn op must carry a degree category.

Guards the fragility that shipped when the grassmann->polyarray relocation renamed ~40
ops and a class-NAME-string degree table (then in pointwise) went stale: an op that fell
out of every category degraded to `inf`, which the quadrature-order chooser turned into a
wrong order (or an OverflowError). `degree.DEFAULT_DEGREE_KINDS` is now TYPE-keyed, and
this test makes the coverage total: a new/renamed builtin with no category fails HERE,
loudly, instead of silently at a consumer's integral.
"""
from __future__ import annotations

from polyarray import degree, ir
from polyarray.pyab import _ARRAY_OP_LOWERINGS

# The Stmt.fn op vocabulary = the relocated array/linalg ops (the type-keyed pyab table,
# enumerated dynamically so a NEW relocated op is auto-checked) + the native pyab-ladder
# builtins (the isinstance vocabulary of _Lowerer._render_op).
_LADDER_BUILTINS: frozenset[type] = frozenset({
    ir.DetOp, ir.InvOp, ir.PinvOp, ir.SolveOp, ir.SqrtOp, ir.AbsOp, ir.SignOp,
    ir.MoveaxisOp, ir.TensordotOp, ir.EinsumStmtOp, ir.EinsumOp, ir.IdentityOp,
    ir.AssertOp, ir.SwitchOp, ir.QrOp, ir.SvdOp, ir.GSvdOp, ir.WhileOp, ir.CallOp,
})


def test_every_builtin_op_has_a_degree_category() -> None:
    ops = set(_ARRAY_OP_LOWERINGS) | set(_LADDER_BUILTINS)
    missing = sorted(t.__name__ for t in ops if t not in degree.DEFAULT_DEGREE_KINDS)
    assert not missing, (
        "polyarray builtin ops with NO degree category in "
        f"degree.DEFAULT_DEGREE_KINDS: {missing}. Add each as zero / passthrough / "
        "multilinear / rational (or special for DetOp/CallOp-style handling) -- an "
        "uncategorized op silently degrees to inf, giving the wrong quadrature order."
    )


def test_degree_kinds_reference_real_ops() -> None:
    # No typos / stale entries: every key is an actual class object (type-keyed, so a
    # rename would be an AttributeError at import -- this just asserts they are types).
    assert all(isinstance(t, type) for t in degree.DEFAULT_DEGREE_KINDS)
    assert set(degree.DEFAULT_DEGREE_KINDS.values()) <= {
        degree.DEG_ZERO, degree.DEG_PASS, degree.DEG_MULT, degree.DEG_RAT, degree.DEG_SPECIAL}
