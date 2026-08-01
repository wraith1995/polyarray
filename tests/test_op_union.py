"""``StmtFn`` is the authoritative op vocabulary — nothing may drift from it.

The static half of the discipline is `assert_never`: a pass that `match`es over
:data:`polyarray.ir.StmtFn` and forgets an op is a *mypy* error. That covers
`exact_fold._sym_apply_builtin` and `sparsity._apply_builtin_op`.

What mypy cannot check is the places the vocabulary is mirrored into a **dict or a
hand-written list** — the degree categories, the two render tables, the package's
public export list. Those are exactly where the drift happened before: a NAME-string
degree table went stale across the grassmann→polyarray relocation, ~40 ops fell out of
every category, and an uncategorised op silently degraded to `inf` degree ⇒ the wrong
quadrature order. These tests close each mirror against `STMT_FN_OPS`.
"""
from __future__ import annotations

import dataclasses
import inspect

import polyarray
from polyarray import degree, ir
from polyarray.numpy_source import _builtin_renderers
from polyarray.pyab import _ARRAY_OP_LOWERINGS

UNION: frozenset[type] = frozenset(ir.STMT_FN_OPS)


def _declared_ops() -> set[type]:
    """Every op dataclass actually defined in ``ir`` — discovered, not listed.

    An op is a frozen dataclass in ``polyarray.ir`` defining its own ``__call__``
    (the ``Stmt.fn`` contract: it runs the modeled operation on bound numeric
    operands at ``Program.run`` time).
    """
    return {
        obj
        for obj in vars(ir).values()
        if inspect.isclass(obj)
        and dataclasses.is_dataclass(obj)
        and obj.__module__ == "polyarray.ir"
        and "__call__" in obj.__dict__
    }


def test_union_is_exactly_the_ops_defined_in_ir() -> None:
    declared = _declared_ops()
    missing = sorted(t.__name__ for t in declared - UNION)
    extra = sorted(t.__name__ for t in UNION - declared)
    assert not missing, (
        f"op dataclasses in polyarray.ir MISSING from ir.StmtFn: {missing}. Add each to "
        "the union -- until you do, every exhaustive `match` over StmtFn silently "
        "ignores it instead of failing mypy, which is the exact failure mode StmtFn "
        "exists to end (Kron/KronFree, SwitchOp)."
    )
    assert not extra, f"ir.StmtFn names types that are not ops in polyarray.ir: {extra}"


def test_runtime_tuple_matches_the_static_union() -> None:
    # STMT_FN_OPS is derived from StmtFn via get_args, so this pins the derivation
    # (and that `is_builtin_op` therefore accepts exactly the union).
    assert len(ir.STMT_FN_OPS) == len(UNION)
    assert all(ir.is_builtin_op(t()) for t in UNION if not dataclasses.fields(t))


def test_every_op_is_exported_from_the_package() -> None:
    exported = {getattr(polyarray, n) for n in polyarray.__all__ if n.endswith("Op")}
    missing = sorted(t.__name__ for t in UNION - exported)
    stale = sorted(t.__name__ for t in exported - UNION)
    assert not missing, f"ops in ir.StmtFn not re-exported by polyarray/__init__: {missing}"
    assert not stale, f"polyarray/__init__ exports ops not in ir.StmtFn: {stale}"


def test_every_op_has_a_degree_category() -> None:
    missing = sorted(t.__name__ for t in UNION if t not in degree.DEFAULT_DEGREE_KINDS)
    assert not missing, (
        f"ops with NO category in degree.DEFAULT_DEGREE_KINDS: {missing}. Add each as "
        "zero / passthrough / multilinear / rational (or special for DetOp/CallOp-style "
        "handling) -- an uncategorised op silently degrees to inf, i.e. the wrong "
        "quadrature order."
    )


# `pyab._Lowerer._render_op` dispatches the type-keyed `_ARRAY_OP_LOWERINGS` table
# first, then an isinstance ladder for these. Listed here (not in the ladder) so the
# union stays the single source of truth: the assertion below is what keeps the two in
# step -- an op in NEITHER place raises `NotImplementedError` at codegen time.
_PYAB_LADDER: frozenset[type] = frozenset({
    ir.DetOp, ir.InvOp, ir.PinvOp, ir.SolveOp, ir.SqrtOp, ir.AbsOp, ir.SignOp,
    ir.MoveaxisOp, ir.TensordotOp, ir.EinsumStmtOp, ir.EinsumOp, ir.IdentityOp,
    ir.AssertOp, ir.SwitchOp, ir.QrOp, ir.SvdOp, ir.GSvdOp, ir.WhileOp, ir.CallOp,
})


def test_pyab_lowers_every_op() -> None:
    covered = set(_ARRAY_OP_LOWERINGS) | _PYAB_LADDER
    missing = sorted(t.__name__ for t in UNION - covered)
    assert not missing, (
        f"ops pyab cannot lower: {missing}. Add a `_ARRAY_OP_LOWERINGS` entry, an arm in "
        "`_Lowerer._render_op`, or an op-carried `__pyab_lower__` hook (and list it in "
        "`_PYAB_LADDER` here)."
    )
    stale = sorted(t.__name__ for t in covered - UNION)
    assert not stale, f"pyab lowers types that are not in ir.StmtFn: {stale}"


# `to_numpy_source` renders from a type-keyed table and RAISES NotImplementedError on a
# miss -- loud, not silent -- so these are a documented gap, not a hazard. Kept as an
# exact set so a NEW op cannot join them by accident.
#   CallOp  -- unwrapped to its inner fn by `_render_op`, never rendered itself.
#   GSvdOp  -- multi-output (U, UI, V, VI, S, rank); needs a multi-statement emission
#              the single-expression renderer signature cannot express.
#   WhileOp -- control flow; needs an emitted loop, not an expression.
_NO_NUMPY_SOURCE_RENDERER: frozenset[type] = frozenset({ir.CallOp, ir.GSvdOp, ir.WhileOp})


def test_numpy_source_renders_every_op_but_the_documented_three() -> None:
    unrendered = UNION - set(_builtin_renderers())
    assert unrendered == _NO_NUMPY_SOURCE_RENDERER, (
        "to_numpy_source coverage changed. Missing renderers: "
        f"{sorted(t.__name__ for t in unrendered - _NO_NUMPY_SOURCE_RENDERER)}; "
        "newly rendered (drop from _NO_NUMPY_SOURCE_RENDERER): "
        f"{sorted(t.__name__ for t in _NO_NUMPY_SOURCE_RENDERER - unrendered)}."
    )
