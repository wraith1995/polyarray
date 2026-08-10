"""Every op `batch._apply` dispatches on must be one of polyarray's own op classes.

`_apply` dispatches by ``isinstance`` over the :data:`polyarray.ir.StmtFn` vocabulary. That
choice is what makes a dead rule impossible: a misspelled class name is an ImportError at
module load, not a rule that silently never fires. Under the earlier name-string dispatch it
was invisible — `batch.py` shipped with seven rules keyed `_ScaleOp` / `_AddOp` /
`_AxisLenOp` / `_ReshapeOp` / `_FirstColsOp` / `_ProjectOp` / `_EmbedOp`, none of which is a
class that exists, so most batch attempts fell back to the per-element loop, correct and slow,
for a year.

These tests keep the rule set coupled to the op vocabulary: every class `_apply` tests against
must still be an op polyarray owns, and the rules that were once dead must stay reachable.
"""

from __future__ import annotations

import ast
import inspect

import polyarray.batch as B
import polyarray.ir as ir


def _dispatched_classes() -> set[str]:
    """The op-class names `_apply` tests against, read out of its source.

    Covers both spellings the body uses: ``isinstance(fn, X)`` and ``isinstance(fn, X | Y)``,
    plus the type-keyed lookup dicts that pick a numpy function per op.
    """
    tree = ast.parse(inspect.getsource(B._apply))
    names: set[str] = set()

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            collect(node.left)
            collect(node.right)

    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance" and len(node.args) == 2):
            collect(node.args[1])
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Name):
                    names.add(key.id)
    return {n for n in names if n.endswith("Op")}


def test_every_batch_rule_names_a_real_op_class():
    """Each class `_apply` dispatches on is an op in polyarray's own closed vocabulary."""
    dispatched = _dispatched_classes()
    assert dispatched, "found no dispatch classes — the source-reading heuristic broke, not the rules"
    own = {t.__name__ for t in ir.STMT_FN_OPS}
    foreign = sorted(n for n in dispatched if n not in own)
    assert not foreign, (
        f"batch._apply dispatches on {foreign}, which are not members of polyarray.ir.StmtFn. "
        "A rule for an op this layer does not own cannot be relied on to fire."
    )


def test_the_seven_regressed_rules_are_reachable():
    """The rules that were once dead. Pinned so a rename cannot quietly re-break them."""
    dispatched = _dispatched_classes()
    for n in ("ScaleOp", "ScaleByOp", "AddOp", "AxisLenOp", "ReshapeOp",
              "FirstColsOp", "ProjectOp", "EmbedOp"):
        assert n in dispatched, f"{n} lost its batch rule"
        assert isinstance(getattr(ir, n, None), type), f"{n} is not an op class"


def test_dispatch_is_not_by_class_name():
    """No rule may compare ``type(fn).__name__`` against a string.

    A name comparison reintroduces the dead-rule failure this module was fixed for, and is the
    string-typed dispatch the stack forbids outright.
    """
    tree = ast.parse(inspect.getsource(B._apply))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        assert node.left.id != "name" or not any(
            isinstance(c, ast.Constant) and isinstance(c.value, str) for c in node.comparators
        ), "batch._apply compares an op's class NAME against a string; dispatch by isinstance"
