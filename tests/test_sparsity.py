"""Structural-zero sparsity propagation tests (P4 of the `simplify` plan).

Correctness invariant under test: the propagated mask is a *subset* of the true
zero pattern — never a false positive.  Concretely, for every modeled-op program
and every cell the pass marks structurally zero, that cell must evaluate to 0.0
for *all* random numeric inputs.  False-negatives (a true zero left unmarked) are
allowed; false-positives are a correctness bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import (
    DetOp,
    EinsumStmtOp,
    InvOp,
    MoveaxisOp,
    OutSpec,
    Program,
    Provenance,
    SymArray,
    SymInput,
    TensordotOp,
    block_zero_mask,
    propagate_sparsity,
)
from polyarray.ir import cells_sparsity
from polyarray.rational import RationalFunction
from polyarray.sparsity import add_mask, mul_mask


def _prov(name: str):
    return Provenance("vertex", name, (), name)


def _zero_rows(sa: SymArray, rows) -> SymArray:
    """Copy ``sa``'s cells and overwrite whole rows with literal 0.0.

    The surviving atom cells stay the *same* RF objects, so the array runs with
    the original input's bindings; the zeroed rows are structural (literal 0).
    """
    cells = np.array(sa.cells, dtype=object)
    for r in rows:
        cells[r, :] = 0.0
    return SymArray(cells, program=sa.program)


def _true_zero_pattern(prog: Program, value_fn, *, seeds=range(8)) -> dict:
    """Cells that are exactly 0.0 across many random inputs, per output name."""
    acc: dict[str, np.ndarray] = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        out = prog.run(value_fn(rng))
        for name, arr in out.items():
            z = np.asarray(arr) == 0.0
            acc[name] = z if name not in acc else (acc[name] & z)
    return acc


# ---------------------------------------------------------------------------
# + / * combinator rules (small explicit cases)
# ---------------------------------------------------------------------------

def test_add_mask_zero_where_both() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(add_mask(a, b), [True, False, False, False])


def test_mul_mask_zero_where_either() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(mul_mask(a, b), [True, True, True, False])


# ---------------------------------------------------------------------------
# matmul / tensordot: a structural-zero row propagates; subset-safe vs numeric
# ---------------------------------------------------------------------------

def test_tensordot_zero_row_propagates_and_is_subset() -> None:
    prog = Program(
        "td",
        inputs=[SymInput("A", (3, 4), _prov("A")), SymInput("B", (4, 2), _prov("B"))],
    )
    A0 = _zero_rows(prog.input("A"), rows=[0])  # row 0 of A is structurally zero
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [A0, prog.input("B")],
        [OutSpec("C", (3, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    rep = propagate_sparsity(prog)
    mask = rep.output_mask("C")
    # Row 0 of the product is forced zero; nothing else is claimed.
    expect = np.zeros((3, 2), dtype=bool)
    expect[0, :] = True
    np.testing.assert_array_equal(mask, expect)

    # Subset-safety: every marked cell is 0.0 for all random inputs.
    def vals(rng):
        return {"A": rng.standard_normal((3, 4)), "B": rng.standard_normal((4, 2))}

    truez = _true_zero_pattern(prog, vals)["C"]
    assert np.all(truez[mask])  # mask ⊆ true-zero pattern


def test_einsum_zero_column_propagates_and_is_subset() -> None:
    # out[i,j] = sum_k A[i,k] B[k,j]; zero a column of B -> that out column zero.
    prog = Program(
        "es",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 2), _prov("B"))],
    )
    # Build B with column 1 structurally zero.
    Bc = np.array(prog.input("B").cells, dtype=object)
    Bc[:, 1] = 0.0
    Bz = SymArray(Bc, program=prog)
    [C] = prog.emit_stmt(
        EinsumStmtOp("ik,kj->ij"),
        [prog.input("A"), Bz],
        [OutSpec("C", (2, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    mask = propagate_sparsity(prog).output_mask("C")
    expect = np.zeros((2, 2), dtype=bool)
    expect[:, 1] = True
    np.testing.assert_array_equal(mask, expect)

    def vals(rng):
        return {"A": rng.standard_normal((2, 3)), "B": rng.standard_normal((3, 2))}

    truez = _true_zero_pattern(prog, vals)["C"]
    assert np.all(truez[mask])


# ---------------------------------------------------------------------------
# moveaxis permutes the mask
# ---------------------------------------------------------------------------

def test_moveaxis_permutes_mask() -> None:
    prog = Program("mv", inputs=[SymInput("A", (2, 3), _prov("A"))])
    A0 = _zero_rows(prog.input("A"), rows=[1])  # row 1 zero
    [C] = prog.emit_stmt(
        MoveaxisOp.from_spec(0, 1),
        [A0],
        [OutSpec("C", (3, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    mask = propagate_sparsity(prog).output_mask("C")
    expect = np.zeros((3, 2), dtype=bool)
    expect[:, 1] = True  # the zero row became a zero column
    np.testing.assert_array_equal(mask, expect)


# ---------------------------------------------------------------------------
# block_zero_mask: block-lower-triangular symbolic matrix -> upper-tri zero block
# ---------------------------------------------------------------------------

def test_block_zero_mask_lower_triangular() -> None:
    n = 4
    cells = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            cells[i, j] = RationalFunction.atom(f"v_{i}_{j}") if j <= i else 0.0
    mask = block_zero_mask(cells)
    expect = np.triu(np.ones((n, n), dtype=bool), k=1)  # strictly-upper = zero
    np.testing.assert_array_equal(mask, expect)

    # Same answer through a SymArray wrapper.
    sa = SymArray(cells)
    np.testing.assert_array_equal(block_zero_mask(sa), expect)


# ---------------------------------------------------------------------------
# opaque ops reset to all-False even when inputs are sparse
# ---------------------------------------------------------------------------

def test_opaque_det_resets_to_unknown() -> None:
    prog = Program("det", inputs=[SymInput("A", (2, 2), _prov("A"))])
    Ac = np.array(prog.input("A").cells, dtype=object)
    Ac[0, 1] = 0.0  # a structural zero in the input
    Az = SymArray(Ac, program=prog)
    [d] = prog.emit_stmt(DetOp(), [Az], [OutSpec("d", ())], bulk=False)
    prog.add_output("d", d.cells)

    mask = propagate_sparsity(prog).output_mask("d")
    assert mask.shape == ()
    assert bool(mask) is False  # opaque: unknown, never claimed zero


def test_opaque_inv_resets_to_unknown() -> None:
    prog = Program("inv", inputs=[SymInput("A", (2, 2), _prov("A"))])
    Ac = np.array(prog.input("A").cells, dtype=object)
    Ac[0, 1] = 0.0  # lower-triangular -> invertible, but one structural zero
    Az = SymArray(Ac, program=prog)
    [Inv] = prog.emit_stmt(InvOp(), [Az], [OutSpec("Inv", (2, 2))], bulk=False)
    prog.add_output("Inv", Inv.cells)

    rep = propagate_sparsity(prog)
    mask = rep.output_mask("Inv")
    assert not mask.any()  # output zero pattern of an opaque op is never claimed

    # And the input's own mask DID see the structural zero (sanity).
    in_mask = rep.mask_for(Az)
    assert bool(in_mask[0, 1]) is True


# ---------------------------------------------------------------------------
# a chained modeled-op program: subset-safety over several random seeds
# ---------------------------------------------------------------------------

def test_chain_subset_safety_random_seeds() -> None:
    # C = A0 @ B ; D = C @ E  where A0 has a zero row -> C row0 zero -> D row0 zero.
    prog = Program(
        "chain",
        inputs=[
            SymInput("A", (3, 3), _prov("A")),
            SymInput("B", (3, 3), _prov("B")),
            SymInput("E", (3, 2), _prov("E")),
        ],
    )
    A0 = _zero_rows(prog.input("A"), rows=[0])
    td = TensordotOp.from_axes(([1], [0]))
    [C] = prog.emit_stmt(td, [A0, prog.input("B")], [OutSpec("C", (3, 3))], bulk=False)
    [D] = prog.emit_stmt(td, [C, prog.input("E")], [OutSpec("D", (3, 2))], bulk=False)
    prog.add_output("C", C.cells)
    prog.add_output("D", D.cells)

    rep = propagate_sparsity(prog)
    cmask, dmask = rep.output_mask("C"), rep.output_mask("D")
    # Row 0 propagates through both contractions.
    assert np.all(cmask[0, :]) and not cmask[1:, :].any()
    assert np.all(dmask[0, :]) and not dmask[1:, :].any()

    def vals(rng):
        return {
            "A": rng.standard_normal((3, 3)),
            "B": rng.standard_normal((3, 3)),
            "E": rng.standard_normal((3, 2)),
        }

    truez = _true_zero_pattern(prog, vals, seeds=range(20))
    assert np.all(truez["C"][cmask])
    assert np.all(truez["D"][dmask])


# ---------------------------------------------------------------------------
# inputs are never structurally zero (fresh atoms); report exposes their masks
# ---------------------------------------------------------------------------

def test_report_input_and_mask_for() -> None:
    prog = Program("io", inputs=[SymInput("A", (2, 2), _prov("A"))])
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("A")],
        [OutSpec("C", (2, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    rep = propagate_sparsity(prog)
    # A fresh input has no structural zeros.
    assert not rep.input_mask("A").any()
    # mask_for accepts a name and a SymArray, returning the same array.
    np.testing.assert_array_equal(rep.mask_for("C"), rep.output_mask("C"))
    np.testing.assert_array_equal(rep.mask_for(prog.input("A")), rep.input_mask("A"))


# ---------------------------------------------------------------------------
# the pass is read-only: the program is untouched
# ---------------------------------------------------------------------------

def test_pass_is_read_only() -> None:
    prog = Program("ro", inputs=[SymInput("A", (2, 2), _prov("A"))])
    A0 = _zero_rows(prog.input("A"), rows=[0])
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [A0, prog.input("A")],
        [OutSpec("C", (2, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    before = cells_sparsity(np.asarray(prog.input("A").cells))
    n_stmts = len(prog.statements)
    propagate_sparsity(prog)
    after = cells_sparsity(np.asarray(prog.input("A").cells))
    np.testing.assert_array_equal(before, after)
    assert len(prog.statements) == n_stmts


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
