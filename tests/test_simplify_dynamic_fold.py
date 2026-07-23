"""``specialize`` / ``fold_numeric`` folding a *dynamic-dim*-creating Stmt.

When a Stmt that produces a runtime dimension δ (a :class:`DimAtom`, e.g. the
rank output of :class:`SvdOp`) has *all-numeric* inputs, it is a value-invariant
map whose rank is statically knowable.  The fold now:

* executes it (like any numeric fold), dropping the Stmt;
* resolves every δ it *created* to the concrete rank from the folded output
  shape;
* substitutes that δ (→ concrete int) across EVERY remaining shape, so a
  downstream Stmt whose output was δ-sized becomes STATIC.

A Stmt with a genuinely symbolic input is NOT folded — its δ survives unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyarray import (
    DimAtom,
    OutSpec,
    Program,
    RationalFunction,
    SymArray,
    SvdOp,
)
from polyarray.ir import is_dynamic
from polyarray.simplify import fold_numeric


@dataclass(frozen=True)
class _TakeColsOp:
    """First ``rank`` columns of ``A`` (runtime-dim output)."""

    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:
        return np.asarray(A)[:, : int(rank)]


@dataclass(frozen=True)
class _ScaleOp:
    """``A * s`` — keeps ``A``'s (dynamic) shape; ``s`` may be symbolic."""

    def __call__(self, A: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.asarray(A) * float(s)


def _const_rank2_4x4() -> np.ndarray:
    rng = np.random.default_rng(7)
    u = rng.standard_normal((4, 2))
    v = rng.standard_normal((2, 4))
    return u @ v  # exact numerical rank 2


def _program() -> Program:
    """SVD a *constant* 4x4 rank-2 matrix, take its δ columns, scale by a
    symbolic input ``s`` — a δ-sized output threaded past a symbolic Stmt."""
    from polyarray import Provenance, SymInput

    prog = Program(
        "dyn_fold",
        inputs=[SymInput("s", (), Provenance(kind="scalar", origin="s",
                                             index=(), label="s"))],
    )
    Mconst = SymArray(_const_rank2_4x4(), program=prog)  # numeric, constant
    U, S, Vh, rank = prog.emit_stmt(
        SvdOp(full_matrices=True),
        [Mconst],
        [OutSpec("U", (4, 4)), OutSpec("S", (4,)), OutSpec("Vh", (4, 4)),
         OutSpec("rank", ())],
        note="svd",
    )
    svd_idx = len(prog.statements) - 1
    delta = DimAtom(name="rank:M", source=(svd_idx, 3))
    [cols] = prog.emit_stmt(
        _TakeColsOp(), [U, rank], [OutSpec("cols", (4, delta))], note="take_cols",
    )
    s_arr = prog.input_arrays["s"]
    [scaled] = prog.emit_stmt(
        _ScaleOp(), [cols, s_arr], [OutSpec("scaled", (4, delta))], note="scale",
    )
    prog.add_output("scaled", scaled)
    return prog


def _all_shapes_static(prog: Program) -> bool:
    for inp in prog.inputs:
        if is_dynamic(inp.shape):
            return False
    for stmt in prog.statements:
        for o in stmt.out:
            if o._bulk is not None and is_dynamic(o._bulk.shape):
                return False
    for sa in prog.outputs.values():
        if sa._bulk is not None and is_dynamic(sa._bulk.shape):
            return False
    return True


def test_constant_svd_folds_and_resolves_delta() -> None:
    prog = _program()
    # Before: the SVD Stmt is present and the downstream shapes are dynamic.
    assert any(isinstance(st.fn, SvdOp) for st in prog.statements)
    assert not _all_shapes_static(prog)

    folded = fold_numeric(prog)

    # The constant SVD (and the δ-taking Stmt) folded away entirely...
    assert not any(isinstance(st.fn, SvdOp) for st in folded.statements)
    assert not any(isinstance(st.fn, _TakeColsOp) for st in folded.statements)
    # ...leaving only the genuinely-symbolic scale Stmt.
    assert any(isinstance(st.fn, _ScaleOp) for st in folded.statements)

    # The δ was resolved: every remaining shape is STATIC (no lingering DimAtom).
    assert _all_shapes_static(folded)
    [scale_stmt] = [st for st in folded.statements if isinstance(st.fn, _ScaleOp)]
    assert scale_stmt.out[0]._bulk is not None
    assert scale_stmt.out[0]._bulk.shape == (4, 2)  # δ -> 2, concrete

    # And it still computes the right answer for a bound ``s``.
    ref = prog.run({"s": 3.0})["scaled"]
    got = folded.run({"s": 3.0})["scaled"]
    assert got.shape == (4, 2)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_symbolic_svd_not_folded_delta_survives() -> None:
    """A SYMBOLIC-input SVD must NOT fold: its δ stays dynamic (byte-identical
    to the pre-fold conservative behaviour)."""
    prog = Program("dyn_sym", inputs=[])
    cells = np.empty((4, 4), dtype=object)
    for idx in np.ndindex(4, 4):
        cells[idx] = RationalFunction.atom("M" + "x".join(map(str, idx)))
    M = SymArray(cells, program=prog)
    U, S, Vh, rank = prog.emit_stmt(
        SvdOp(full_matrices=True),
        [M],
        [OutSpec("U", (4, 4)), OutSpec("S", (4,)), OutSpec("Vh", (4, 4)),
         OutSpec("rank", ())],
        note="svd",
    )
    svd_idx = len(prog.statements) - 1
    delta = DimAtom(name="rank:M", source=(svd_idx, 3))
    [cols] = prog.emit_stmt(
        _TakeColsOp(), [U, rank], [OutSpec("cols", (4, delta))], note="take_cols",
    )
    prog.add_output("cols", cols)

    folded = fold_numeric(prog)
    # SVD survives; the δ-sized output stays dynamic.
    assert any(isinstance(st.fn, SvdOp) for st in folded.statements)
    assert not _all_shapes_static(folded)
