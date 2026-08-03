"""Every op name `batch._apply` dispatches on must name a REAL op class.

`_apply` dispatches by ``type(fn).__name__`` deliberately — so this module carries no dependency on
the front end (grassmann) that defines some ops. The cost of that choice is that a misspelled name is
INVISIBLE: the rule simply never fires and `batched_run` silently falls back to the per-element loop,
which is correct but slow. That is exactly what happened — `batch.py` shipped with seven rules keyed
`_ScaleOp` / `_AddOp` / `_AxisLenOp` / `_ReshapeOp` / `_FirstColsOp` / `_ProjectOp` / `_EmbedOp`, none
of which is a class that exists, so 80% of FEEC batch attempts fell back for a year.

This test re-couples the names to reality without re-coupling the MODULE to it.
"""
from __future__ import annotations

import ast
import inspect

import polyarray.batch as B
import polyarray.ir as ir


def _dispatched_names() -> set[str]:
    """The op-class names `_apply` compares `name` against, read out of its source."""
    tree = ast.parse(inspect.getsource(B._apply))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id != "name":
            continue
        for op, cmp in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(cmp, ast.Constant):
                names.add(cmp.value)
            elif isinstance(op, ast.In) and isinstance(cmp, (ast.Tuple, ast.List, ast.Set)):
                names.update(e.value for e in cmp.elts if isinstance(e, ast.Constant))
    return names


def test_every_batch_rule_names_a_real_op_class():
    dispatched = _dispatched_names()
    assert dispatched, "found no dispatch names — the source-reading heuristic broke, not the rules"
    missing = sorted(n for n in dispatched if not isinstance(getattr(ir, n, None), type))
    assert not missing, (
        f"batch._apply dispatches on {missing}, which are not op classes in polyarray.ir. "
        "A rule keyed on a non-existent name never fires — batched_run silently falls back."
    )


def test_the_seven_regressed_rules_are_reachable():
    """The specific names that were dead. Pinned so a rename cannot quietly re-break them."""
    dispatched = _dispatched_names()
    for n in ("ScaleOp", "ScaleByOp", "AddOp", "AxisLenOp", "ReshapeOp",
              "FirstColsOp", "ProjectOp", "EmbedOp"):
        assert n in dispatched, f"{n} lost its batch rule"
        assert isinstance(getattr(ir, n, None), type), f"{n} is not an op class"
