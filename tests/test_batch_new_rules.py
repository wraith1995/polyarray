"""New batch rules, each A/B'd against the per-element loop they replace.

WHY THIS SHAPE. `batched_run`'s fallback is *correct* — when an op has no rule, the caller
loops per element and gets the right answer, slowly. So a batch rule buys performance and can
only cost correctness: a wrong rule silently returns wrong numbers where there were right ones.
That makes "agrees with the loop" the whole specification, and it is what every test here
asserts, on shapes chosen to expose the mistakes the obvious implementation would make:

* `TransposeOp` is ``A.T`` — a FULL-REVERSE transpose. On ndim > 2 the plausible-looking
  ``swapaxes(-2, -1)`` is a different operation, and both are "a transpose", so a 3-D case is
  the only thing that separates them.
* `ConcatOp` and `ColStackOp` FLATTEN each operand before combining, which their names do not
  say; a rule that concatenated on an axis instead would agree on 1-D inputs and diverge on 2-D.
* The batched-projector `ProjectOp`/`EmbedOp` arms previously raised `BatchUnsupported`.
"""
from __future__ import annotations

import numpy as np
import pytest

import polyarray.batch as B
from polyarray import ColStackOp, ConcatOp, EmbedOp, MoveaxisOp, ProjectOp, TransposeOp


def _loop(fn, arrays, batched):
    """The per-element fallback: run the real op on each slice and stack."""
    n = next(a.shape[0] for a, b in zip(arrays, batched) if b)
    return np.stack([fn(*[a[i] if b else a for a, b in zip(arrays, batched)]) for i in range(n)])


def _check(fn, arrays, batched):
    """`_apply`'s batched answer must equal the loop's, elementwise."""
    got, is_b = B._apply(fn, list(zip(arrays, batched)))
    assert is_b is True, f"{type(fn).__name__} lost its batch flag"
    np.testing.assert_allclose(np.asarray(got), _loop(fn, arrays, batched), rtol=1e-12, atol=0.0)


def test_transpose_batch_rule_is_a_full_reverse_not_a_last_two_swap():
    rng = np.random.default_rng(0)
    _check(TransposeOp(), [rng.standard_normal((5, 3, 4))], [True])        # 2-D per element
    x = rng.standard_normal((5, 2, 3, 4))                                   # 3-D per element
    _check(TransposeOp(), [x], [True])
    # And it really is the full reverse — the swap would give a different array here.
    got, _ = B._apply(TransposeOp(), [(x, True)])
    assert np.asarray(got).shape == (5, 4, 3, 2)
    assert not np.allclose(np.asarray(got), np.swapaxes(x, -2, -1).reshape(np.asarray(got).shape),
                           atol=0.0) or x.shape[1] == x.shape[3]


def test_moveaxis_batch_rule_shifts_past_the_batch_axis():
    rng = np.random.default_rng(1)
    _check(MoveaxisOp(0, 2), [rng.standard_normal((6, 2, 3, 4))], [True])
    _check(MoveaxisOp(-1, 0), [rng.standard_normal((6, 2, 3, 4))], [True])  # negative: no shift


@pytest.mark.parametrize("fn", [ConcatOp(), ColStackOp()])
def test_flattening_combiners_agree_with_the_loop_on_2d_operands(fn):
    """2-D per element is the discriminating case: these ops flatten first."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 2, 3))
    b = rng.standard_normal((4, 2, 3))
    _check(fn, [a, b], [True, True])


def test_batched_projector_project_and_embed_agree_with_the_loop():
    """Previously `BatchUnsupported` — one P per batch element is now handled."""
    rng = np.random.default_rng(3)
    P = rng.standard_normal((7, 4, 2))                       # ambient 4 -> sub 2, per element
    v_amb = rng.standard_normal((7, 4))
    v_sub = rng.standard_normal((7, 2))
    _check(ProjectOp(), [P, v_amb], [True, True])
    _check(EmbedOp((4,)), [P, v_sub], [True, True])


def test_an_unbatched_operand_still_takes_the_unbatched_path():
    """The rules must not change what they do when nothing is batched."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 4))
    for fn in (TransposeOp(), ConcatOp(), ColStackOp()):
        args = [x, x] if isinstance(fn, (ConcatOp, ColStackOp)) else [x]
        got, is_b = B._apply(fn, [(a, False) for a in args])
        assert is_b is False
        np.testing.assert_allclose(np.asarray(got), np.asarray(fn(*args)), atol=0.0)
