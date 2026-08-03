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


def test_every_builtin_op_has_a_degree_category() -> None:
    # The vocabulary is `ir.StmtFn` — the authoritative union — not a hand-kept list
    # reconstructed from the render tables (which was itself a mirror that could go
    # stale). `tests/test_op_union.py` pins the union against the ops `ir` defines, so
    # this reads coverage off the real thing.
    ops = set(ir.STMT_FN_OPS)
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
