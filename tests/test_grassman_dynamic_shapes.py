"""Stage-B additions for Grassman lowering: dynamic (runtime-dim) shapes.

An array axis may be a runtime integer produced by a Stmt (the SVD rank
δ_f), so image / projected / FFS arrays are sized at run time.  The key
test runs the SAME Program on two inputs of different numerical rank and
gets two different output shapes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from polyarray import (
    DimAtom,
    EinsumStmtOp,
    OutSpec,
    Program,
    Provenance,
    RationalFunction,
    SymArray,
    SvdOp,
)
from polyarray.ir import is_dynamic


def _prov(label: str) -> Provenance:
    return Provenance(kind="vertex", origin=label, index=(), label=label)


def _sym_matrix(program, Mnum: np.ndarray, prefix: str):
    cells = np.empty(Mnum.shape, dtype=object)
    values: dict[str, float] = {}
    for idx in np.ndindex(*Mnum.shape):
        name = prefix + "x".join(str(i) for i in idx)
        cells[idx] = RationalFunction.atom(name)
        values[name] = float(Mnum[idx])
    return SymArray(cells, program=program), values


@dataclass(frozen=True)
class _TakeColsOp:
    """Select the first ``rank`` columns of a matrix (runtime-dim output)."""

    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:
        return np.asarray(A)[:, : int(rank)]


# ---------------------------------------------------------------------------
# DimAtom / is_dynamic basics
# ---------------------------------------------------------------------------

def test_dim_atom_is_frozen_value_type() -> None:
    a = DimAtom(name="rank:f", source=(0, 3))
    b = DimAtom(name="rank:f", source=(0, 3))
    assert a == b and hash(a) == hash(b)
    # usable as a dict key
    d = {a: 5}
    assert d[b] == 5


def test_is_dynamic_helper() -> None:
    assert not is_dynamic(())
    assert not is_dynamic((3, 4))
    assert is_dynamic((3, DimAtom("d", (0, 0))))
    assert is_dynamic((DimAtom("d", (0, 0)),))


# ---------------------------------------------------------------------------
# emit_stmt forces bulk for dynamic shapes
# ---------------------------------------------------------------------------

def test_emit_stmt_dynamic_shape_forces_bulk() -> None:
    rng = np.random.default_rng(0)
    An = rng.standard_normal((4, 4))
    prog = Program("dyn")
    A, vals = _sym_matrix(prog, An, "a")
    U, S, Vh, rank = prog.emit_stmt(
        SvdOp(full_matrices=True),
        [A],
        [OutSpec("U", (4, 4)), OutSpec("S", (4,)), OutSpec("Vh", (4, 4)),
         OutSpec("rank", ())],
        note="svd",
        bulk=True,
    )
    svd_idx = len(prog.statements) - 1
    delta = DimAtom(name="rank:A", source=(svd_idx, 3))

    # Dynamic output (m, δ); pass bulk=False but expect it forced to bulk.
    [cols] = prog.emit_stmt(
        _TakeColsOp(),
        [U, rank],
        [OutSpec("cols", (4, delta))],
        note="take_cols",
        bulk=False,  # should be overridden because the shape is dynamic
    )
    assert cols._bulk is not None  # forced bulk
    assert is_dynamic(cols._bulk.shape)
    assert cols.shape == (4, delta)  # build-time symbolic shape


# ---------------------------------------------------------------------------
# THE KEY TEST: same Program, two ranks -> two output shapes
# ---------------------------------------------------------------------------

def _rank_sized_program() -> Program:
    """SVD an input, take its first-δ columns -> a δ-sized output."""
    prog = Program("rank_sized", inputs=[])
    # use a symbolic input matrix
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
        _TakeColsOp(),
        [U, rank],
        [OutSpec("cols", (4, delta))],
        note="take_cols",
    )
    prog.add_output("cols", cols)
    prog.add_output("rank", rank)
    return prog


def _vals_for(M: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx in np.ndindex(4, 4):
        out["M" + "x".join(map(str, idx))] = float(M[idx])
    return out


def test_same_program_two_ranks_two_shapes() -> None:
    prog = _rank_sized_program()
    rng = np.random.default_rng(1)

    # Rank-2 input.
    u2 = rng.standard_normal((4, 2))
    v2 = rng.standard_normal((2, 4))
    M2 = u2 @ v2
    res2 = prog.run(_vals_for(M2))
    assert int(round(float(res2["rank"]))) == 2
    assert res2["cols"].shape == (4, 2)

    # Rank-3 input — SAME program.
    u3 = rng.standard_normal((4, 3))
    v3 = rng.standard_normal((3, 4))
    M3 = u3 @ v3
    res3 = prog.run(_vals_for(M3))
    assert int(round(float(res3["rank"]))) == 3
    assert res3["cols"].shape == (4, 3)

    # Different ranks -> different output shapes from the SAME Program.
    assert res2["cols"].shape != res3["cols"].shape


# ---------------------------------------------------------------------------
# einsum contracting a δ-sized axis runs on both ranks
# ---------------------------------------------------------------------------

def test_einsum_over_dynamic_axis() -> None:
    """Project onto the δ-dim subspace: P = U_δ @ U_δ^T (4x4), via einsum."""
    prog = Program("proj", inputs=[])
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
        _TakeColsOp(), [U, rank], [OutSpec("Ud", (4, delta))], note="take_cols",
    )
    # Contract the δ-sized axis: P[i,j] = sum_r Ud[i,r] Ud[j,r].
    [P] = prog.emit_stmt(
        EinsumStmtOp("ir,jr->ij"),
        [cols, cols],
        [OutSpec("P", (4, 4))],
        note="proj",
    )
    prog.add_output("P", P)
    prog.add_output("rank", rank)

    rng = np.random.default_rng(2)
    for r in (1, 2, 3):
        u = rng.standard_normal((4, r))
        v = rng.standard_normal((r, 4))
        M_in = u @ v
        res = prog.run(_vals_for(M_in))
        assert int(round(float(res["rank"]))) == r
        P_out = res["P"]
        assert P_out.shape == (4, 4)
        # P is a projector: symmetric, idempotent, trace == rank.
        np.testing.assert_allclose(P_out, P_out.T, atol=1e-9)
        np.testing.assert_allclose(P_out @ P_out, P_out, atol=1e-8)
        np.testing.assert_allclose(np.trace(P_out), r, atol=1e-7)


# ---------------------------------------------------------------------------
# dynamic output shape mismatch is caught
# ---------------------------------------------------------------------------

def test_dynamic_shape_mismatch_raises() -> None:
    @dataclass(frozen=True)
    class _WrongShapeOp:
        def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:
            # deliberately return the WRONG number of columns
            return np.asarray(A)[:, : int(rank) + 1]

    prog = Program("bad", inputs=[])
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
        _WrongShapeOp(), [U, rank], [OutSpec("cols", (4, delta))], note="bad",
    )
    prog.add_output("cols", cols)
    rng = np.random.default_rng(3)
    u = rng.standard_normal((4, 2))
    v = rng.standard_normal((2, 4))
    with pytest.raises(ValueError, match="expected shape"):
        prog.run(_vals_for(u @ v))
