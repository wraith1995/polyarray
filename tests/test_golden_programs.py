"""Golden programs built by hand and executed via ``Program.run`` (plan §7).

Each test constructs a small :class:`Program` directly (no sampler /
interpreter) and asserts ``program.run`` matches an independent
reference. Mirrors the construction idioms in chartlib's
``tests/module/`` (typed_ops, select_x, forward).
"""
from __future__ import annotations

import numpy as np

from polyarray import (
    DetOp,
    IdentityOp,
    IntAtom,
    OutSpec,
    Program,
    Provenance,
    RationalFunction,
    SwitchOp,
    SymArray,
    SymInput,
    SymbolicBudget,
    TensordotOp,
    analyze,
    partial_eval,
)
from polyarray.ir import EinsumStmtOp, IntAtomRef


def _prov(label: str) -> Provenance:
    return Provenance(kind="vertex", origin=label, index=(), label=label)


def _sym_matrix(program, Mnum: np.ndarray, prefix: str):
    """Object-dtype SymArray of fresh atoms mirroring ``Mnum`` (+ values dict)."""
    cells = np.empty(Mnum.shape, dtype=object)
    values: dict[str, float] = {}
    for idx in np.ndindex(*Mnum.shape):
        name = prefix + "x".join(str(i) for i in idx)
        cells[idx] = RationalFunction.atom(name)
        values[name] = float(Mnum[idx])
    return SymArray(cells, program=program), values


# ---------------------------------------------------------------------------
# numeric lane: matmul + det + solve match numpy
# ---------------------------------------------------------------------------

def test_numeric_lane_matmul_det_solve() -> None:
    rng = np.random.default_rng(0)
    An = rng.standard_normal((3, 3)) + 3.0 * np.eye(3)
    Bn = rng.standard_normal((3, 3))
    bn = rng.standard_normal(3)

    prog = Program(
        "numeric",
        inputs=[
            SymInput("A", (3, 3), _prov("A")),
            SymInput("B", (3, 3), _prov("B")),
            SymInput("b", (3,), _prov("b")),
        ],
    )
    A = prog.input("A")
    B = prog.input("B")
    bvec = prog.input("b")

    C = A.matmul(B)                 # small -> stays rational (matmul is cell arith)
    d = A.det()                     # 3x3 <= budget -> closed-form det
    x = A.solve(bvec)               # solve A x = b

    prog.add_output("C", C.cells)
    prog.add_output("det", d.cells)
    prog.add_output("x", x.cells)

    res = prog.run({"A": An, "B": Bn, "b": bn})
    np.testing.assert_allclose(res["C"], An @ Bn, atol=1e-9)
    np.testing.assert_allclose(res["det"], np.linalg.det(An), atol=1e-9)
    np.testing.assert_allclose(res["x"], np.linalg.solve(An, bn), atol=1e-9)


# ---------------------------------------------------------------------------
# rational lane: symbolic 2x2 inverse -> RationalFunction entries by hand
# ---------------------------------------------------------------------------

def test_rational_lane_2x2_inverse() -> None:
    # Symbolic matrix [[a, b], [c, d]]; inverse = 1/(ad-bc) [[d, -b], [-c, a]].
    prog = Program("rat", inputs=[SymInput("M", (2, 2), _prov("M"))])
    M = prog.input("M")
    inv = M.inverse()                # 2x2 <= budget -> closed-form cofactor inverse
    prog.add_output("inv", inv.cells)

    # Each entry must be a RationalFunction (the rational lane).
    for idx in np.ndindex(2, 2):
        assert isinstance(inv.cells[idx], RationalFunction)

    # Evaluate at a concrete matrix and compare to the hand formula.
    a, b, c, d = 2.0, 1.0, 1.0, 3.0
    # Program input atoms are named "M_<i>_<j>" by allocate_input.
    vals = {"M": np.array([[a, b], [c, d]], dtype=float)}
    res = prog.run(vals)
    det = a * d - b * c
    expected = np.array([[d, -b], [-c, a]], dtype=float) / det
    np.testing.assert_allclose(res["inv"], expected, atol=1e-12)
    np.testing.assert_allclose(res["inv"], np.linalg.inv([[a, b], [c, d]]), atol=1e-12)


# ---------------------------------------------------------------------------
# deferred / imperative lane: over-budget det emits Stmt(DetOp) and evaluates
# ---------------------------------------------------------------------------

def test_deferred_lane_over_budget_det_emits_detop() -> None:
    rng = np.random.default_rng(1)
    n = 7  # > SymbolicBudget.naive_inverse_max_size (6) -> deferred Stmt
    Mn = rng.standard_normal((n, n)) + n * np.eye(n)

    prog = Program("deferred")
    A, vals = _sym_matrix(prog, Mn, "d")
    out = A.det()  # emits a Stmt whose fn is a DetOp

    assert len(prog.statements) >= 1
    assert isinstance(prog.statements[-1].fn, DetOp)

    prog.add_output("det", out.cells)
    res = prog.run(vals)
    np.testing.assert_allclose(res["det"], np.linalg.det(Mn), rtol=1e-7)


# ---------------------------------------------------------------------------
# tensordot contraction round-trip
# ---------------------------------------------------------------------------

def test_tensordot_op_roundtrip() -> None:
    prog = Program(
        "td",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))],
    )
    op = TensordotOp.from_axes(([1], [0]))
    [C] = prog.emit_stmt(
        op, [prog.input("A"), prog.input("B")], [OutSpec("C", (2, 4))], bulk=False,
    )
    prog.add_output("C", C.cells)
    rng = np.random.default_rng(4)
    Av = rng.standard_normal((2, 3))
    Bv = rng.standard_normal((3, 4))
    res = prog.run({"A": Av, "B": Bv})
    np.testing.assert_allclose(res["C"], np.tensordot(Av, Bv, axes=([1], [0])))


# ---------------------------------------------------------------------------
# einsum-stmt contraction round-trip (>= 2 symbolic operands)
# ---------------------------------------------------------------------------

def test_einsum_stmt_op_roundtrip() -> None:
    prog = Program(
        "es",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))],
    )
    op = EinsumStmtOp("ij,jk->ik")
    [C] = prog.emit_stmt(
        op, [prog.input("A"), prog.input("B")], [OutSpec("C", (2, 4))], bulk=False,
    )
    prog.add_output("C", C.cells)
    rng = np.random.default_rng(11)
    Av = rng.standard_normal((2, 3))
    Bv = rng.standard_normal((3, 4))
    res = prog.run({"A": Av, "B": Bv})
    np.testing.assert_allclose(res["C"], np.einsum("ij,jk->ik", Av, Bv))


# ---------------------------------------------------------------------------
# SwitchOp selects a branch by IntAtom
# ---------------------------------------------------------------------------

def _build_switch_program():
    prog = Program("switch", inputs=[])
    prog.declare_int_atom("o", range(3))
    branches = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
    ]
    branch_syms = [SymArray(b, program=prog) for b in branches]
    op = SwitchOp(n_branches=3)
    [out] = prog.emit_stmt(
        op,
        [IntAtomRef("o"), *branch_syms],
        [OutSpec("chosen", (2,))],
        note="select_x(o)",
    )
    prog.add_output("chosen", out.cells)
    return prog, branches


def test_switch_op_selects_branch_by_int_atom() -> None:
    prog, branches = _build_switch_program()
    assert isinstance(prog.statements[0].fn, SwitchOp)
    assert isinstance(prog.statements[0].in_[0], IntAtomRef)
    for k in range(3):
        res = prog.run({"o": k})
        np.testing.assert_array_equal(res["chosen"], branches[k])


def test_int_atom_is_frozen_value_type() -> None:
    a = IntAtom(name="o", domain=range(3))
    b = IntAtom(name="o", domain=range(3))
    assert a == b and hash(a) == hash(b)


# ---------------------------------------------------------------------------
# partial_eval round-trip: heavy symbolic output collapses exactly
# ---------------------------------------------------------------------------

def _heavy_program():
    prog = Program(
        "heavy",
        inputs=[SymInput("V", (2,), _prov("V"))],
    )
    v = prog.input("V")
    a, b = v[0], v[1]
    big = a * a * a + a * a * b + b * b
    small = a + b
    prog.add_output("big", np.array([big], dtype=object))
    prog.add_output("small", np.array([small], dtype=object))
    return prog


def test_partial_eval_collapse_all_matches_uncollapsed() -> None:
    prog = _heavy_program()
    analyze(prog)  # smoke: report builds on the un-collapsed program

    pe = partial_eval(prog, max_cell_size=0)  # collapse all symbolic outputs
    assert len(prog.statements) == 0          # original untouched
    assert len(pe.statements) == 2            # both outputs collapsed to atoms

    # Every collapsed cell is an IdentityOp stmt.
    for st in pe.statements:
        assert isinstance(st.fn, IdentityOp)

    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.standard_normal(2)
        a = prog.run({"V": x})
        b = pe.run({"V": x})
        np.testing.assert_allclose(b["big"], a["big"], atol=1e-12)
        np.testing.assert_allclose(b["small"], a["small"], atol=1e-12)


def test_partial_eval_default_budget_is_available() -> None:
    # SymbolicBudget is part of the public surface; confirm it constructs.
    bud = SymbolicBudget()
    assert bud.naive_inverse_max_size >= 1
    assert isinstance(SymbolicBudget.legacy(), SymbolicBudget)
