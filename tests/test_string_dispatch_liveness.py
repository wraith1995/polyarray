"""Every string-keyed dispatch site must name something REAL.

THE BUG CLASS. polyarray dispatches on ``type(fn).__name__`` in a few places, deliberately: it
lets a module route front-end (grassmann) ops without importing the front end. The cost is that
a misspelled key is INVISIBLE — the branch simply never matches, and the fallback is *correct*,
just slower or coarser. Nothing fails, nothing warns, and the only symptom is a performance or
precision loss nobody attributes to a typo. `batch.py` shipped seven such dead keys and they
survived a year (see `test_batch_rules_named.py`, the detector written for that incident).

That detector covers exactly one site. This module covers the others, because the defect is a
property of the *technique*, not of `batch.py`:

* `batch.py`'s inner ``{op-name: numpy-function-name}`` dicts — the OUTER ladder is checked by
  `test_batch_rules_named`, but the dict values are `getattr(np.linalg, ...)` at call time and
  a typo there raises only when that op is finally batched;
* `degree.py`'s legacy NAME-string category sets. **This is the dangerous one**: a miss there
  does not fall back to a slow path, it falls through to ``_INF`` (`degree.py`'s "unknown"
  answer), which is a *safe over-estimate* of polynomial degree and therefore silent — you get
  a needlessly high quadrature order, not an error. Same silent-degradation shape as the batch
  bug, with a numerical consequence instead of a performance one;
* `AssertOp.kind` — one string vocabulary written out in two modules (`ir` executes it,
  `exact_fold` mirrors it exactly). Drift is only discovered if the drifted kind is exercised.

The technique here is `test_batch_rules_named`'s: read the source with `ast`, harvest the string
constants a site dispatches on, and resolve each against `polyarray.ir`. Reading the source
(rather than probing behaviour) is what makes a *never-taken* branch visible at all.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import warnings

import numpy as np
import pytest

import polyarray.batch as B
import polyarray.degree as D
import polyarray.exact_fold as EF
import polyarray.ir as ir


def _string_constants(fn) -> set[str]:
    """Every plain string constant in ``fn``'s source — the candidate dispatch keys."""
    tree = ast.parse(inspect.getsource(fn))
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _op_names() -> set[str]:
    """The names of polyarray's own op classes."""
    return {n for n in dir(ir) if isinstance(getattr(ir, n), type) and n.endswith("Op")}


# --------------------------------------------------------------------------------------------
# batch.py — the inner name→numpy-function dicts
# --------------------------------------------------------------------------------------------

def test_batch_inner_dispatch_dicts_resolve_on_both_sides():
    """`_apply`'s inner dicts pick a numpy function per op CLASS. Both halves must resolve.

    The keys are op classes, so a wrong key cannot survive import — but the values are the
    numpy callables, and nothing else checks that the attribute a value names still exists.
    """
    tree = ast.parse(inspect.getsource(B._apply))
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.id for k in node.keys if isinstance(k, ast.Name)]
        if not keys or not all(k.endswith("Op") for k in keys):
            continue                                   # not an op-keyed dict
        for k in keys:
            assert isinstance(getattr(ir, k, None), type), \
                f"batch._apply dispatches on {k!r}, which is not an op class in polyarray.ir"
            checked += 1
        # Values name a numpy callable: `np.sqrt`, `np.linalg.pinv`.
        for v in node.values:
            if isinstance(v, ast.Attribute):
                mod = np.linalg if getattr(v.value, "attr", None) == "linalg" else np
                assert hasattr(mod, v.attr), \
                    f"batch._apply resolves {ast.unparse(v)} — which does not exist"
    assert checked, "found no op-keyed dict in batch._apply — the heuristic broke"


# --------------------------------------------------------------------------------------------
# degree.py — the legacy NAME-string category sets (silent fall-through to _INF)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("setname", ["DEFAULT_ZERO_OPS", "DEFAULT_PASSTHROUGH_OPS",
                                     "DEFAULT_MULTILINEAR_OPS"])
def test_degree_name_sets_name_real_op_classes(setname: str):
    """A name here that matches no class is not an error — it is a silently WRONG degree.

    `_op_degree` resolves a category by (1) the type-keyed native map, (2) an op-carried
    `_DEGREE_KIND`, then (3) these name sets; anything unmatched returns `_INF`. `_INF` is a
    safe over-estimate, so a typo costs quadrature order and says nothing. These sets are for
    FRONT-END op names polyarray must not import — but every entry that IS one of polyarray's
    own must still resolve, or it is dead weight claiming to do something."""
    names = getattr(D, setname)
    ours = _op_names()
    # An entry that looks like one of ours (`…Op`) must actually be one of ours.
    bogus = sorted(n for n in names if n.endswith("Op") and n not in ours)
    assert not bogus, (
        f"degree.{setname} lists {bogus}, which are not op classes in polyarray.ir. "
        "A name that matches nothing falls through to _INF — a silently over-estimated degree, "
        "not a failure.")


def test_degree_name_sets_agree_with_the_type_keyed_map():
    """The type-keyed map is authoritative for polyarray's own ops; a name set that DISAGREES
    with it is a latent contradiction — whichever is consulted first wins, and they are
    consulted in different orders on different paths."""
    kinds = D.DEFAULT_DEGREE_KINDS
    by_name = {cls.__name__: kind for cls, kind in kinds.items()}
    for setname, expected in (("DEFAULT_ZERO_OPS", D.DEG_ZERO),
                              ("DEFAULT_PASSTHROUGH_OPS", D.DEG_PASS),
                              ("DEFAULT_MULTILINEAR_OPS", D.DEG_MULT)):
        for n in getattr(D, setname):
            if n in by_name:
                assert by_name[n] == expected, (
                    f"degree.{setname} says {n} is {expected}, but DEFAULT_DEGREE_KINDS says "
                    f"{by_name[n]} — two answers for one op")


# --------------------------------------------------------------------------------------------
# AssertOp.kind — one vocabulary, two modules
# --------------------------------------------------------------------------------------------

def _assert_kinds(fn) -> set[str]:
    """The `AssertOp` kind strings a function compares against.

    ``dedent`` because `AssertOp.__call__` is a METHOD — `inspect.getsource` hands back its
    source still indented inside the class body, which `ast.parse` rejects outright."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, cmp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.In)):
                    if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
                        kinds.add(cmp.value)
                    elif isinstance(cmp, (ast.Tuple, ast.List, ast.Set)):
                        kinds.update(e.value for e in cmp.elts
                                     if isinstance(e, ast.Constant)
                                     and isinstance(e.value, str))
    return kinds


def test_the_exact_twin_covers_every_assert_kind_the_op_executes():
    """`ir.AssertOp.__call__` and `exact_fold._exact_assert` enumerate the SAME kind vocabulary
    in two places, each ending in `raise ValueError("unknown kind")`. Drift is loud only if the
    drifted kind is exercised — and `in_span` reached production before the twin knew it
    (polyarray `14776a3`). Pin the two vocabularies together."""
    executed = _assert_kinds(ir.AssertOp.__call__)
    twinned = _assert_kinds(EF._exact_assert)
    assert executed, "found no kind strings in AssertOp.__call__ — the heuristic broke"
    missing = sorted(k for k in executed if k not in twinned)
    assert not missing, (
        f"AssertOp kinds {missing} are executed by ir but unknown to exact_fold's twin — "
        "the exact lane would raise 'unknown kind' the first time one is certified")


# --------------------------------------------------------------------------------------------
# The two PUBLIC string-keyed extension points, checked where the keys arrive
# --------------------------------------------------------------------------------------------

def test_a_private_spelling_of_one_of_our_ops_is_reported_at_the_entry_point():
    """`op_renderers` / `op_lowerings` take FRONT-END names, so polyarray cannot validate them
    in general — it deliberately cannot see that namespace. It can catch the one spelling that
    caused every instance so far: `_Name` for an op that now lives here as `Name`.

    These ops were RELOCATED into polyarray from grassmann's lowering layer, so keys like
    `_AxisLenOp` were correct before the move and silently dead after it — and because the
    built-in type-keyed renderer takes precedence and handles the op anyway, the results stayed
    right. That is exactly why it goes unnoticed."""
    from polyarray import Program, Provenance, SymInput, to_numpy_source
    from polyarray.numpy_source import DeadOpKeyWarning

    prog = Program("t", inputs=[SymInput("x", (2,), Provenance("vertex", "x", (), "x"))])
    prog.add_output("y", prog.input("x").cells)

    with pytest.warns(DeadOpKeyWarning, match=r"_AxisLenOp.*PRIVATE spelling"):
        to_numpy_source(prog, op_renderers={"_AxisLenOp": lambda op, a: "0"})

    # A genuine front-end name resolves to nothing here and must NOT be flagged — a false
    # alarm on a legitimate key is how a warning gets trained away.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        to_numpy_source(prog, op_renderers={"_QrSignFixOp": lambda op, a: "0"})
    assert not [w for w in rec if issubclass(w.category, DeadOpKeyWarning)]

    # And the correct spelling is silent.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        to_numpy_source(prog, op_renderers={"AxisLenOp": lambda op, a: "0"})
    assert not [w for w in rec if issubclass(w.category, DeadOpKeyWarning)]
