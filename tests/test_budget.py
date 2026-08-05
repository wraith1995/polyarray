"""P5 tests: ``SimplifyBudget`` + the moderation procedure.

Covers the budget-moderation properties:
exactness over presets/seeds, collapse subsuming ``partial_eval``, monotone
collapse, idempotence, ``keep_provenance`` protection, and ``den_degree_max``
intermediate extraction.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import (
    Program,
    Provenance,
    SimplifyBudget,
    SymInput,
    partial_eval,
    specialize,
)
from polyarray.ir import _cell_size
from polyarray.rational import RationalFunction


def _prov(name: str, kind: str = "vertex"):
    return Provenance(kind, name, (), name)


def _symbolic_cells(program: Program, out_name: str = "out") -> list:
    cells = np.asarray(program.outputs[out_name].cells).reshape(-1)
    return [c for c in cells if isinstance(c, RationalFunction)]


def _n_symbolic(program: Program, out_name: str = "out") -> int:
    """Number of output cells still symbolic over a NON-captured generator."""
    n = 0
    for c in _symbolic_cells(program, out_name):
        # A bare stmt_out atom counts as collapsed (captured), not symbolic.
        from polyarray.budget import _is_captured_atom

        if not _is_captured_atom(c, program):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Fixtures: a program with a heavy + a light symbolic output cell
# ---------------------------------------------------------------------------

def _poly_program() -> Program:
    """out = [heavy(mass 7), light(mass 3)] over a 3-vector input ``a``."""
    p = Program("poly", inputs=[SymInput("a", (3,), _prov("a"))])
    a = p.input("a").cells
    heavy = a[0] * a[1] + a[0] * a[2] + a[1] * a[2] + a[0] + a[1] + a[2]
    light = a[0] + a[1]
    cells = np.empty((2,), dtype=object)
    cells[0] = heavy
    cells[1] = light
    p.add_output("out", cells)
    return p


def _den_program() -> Program:
    """out = [a0/(a1*a2)] — a single cell with a degree-2 denominator."""
    p = Program("den", inputs=[SymInput("b", (3,), _prov("b"))])
    b = p.input("b").cells
    cell = b[0] / (b[1] * b[2])
    cells = np.empty((1,), dtype=object)
    cells[0] = cell
    p.add_output("out", cells)
    return p


# ---------------------------------------------------------------------------
# collapse: per-cell mass ceiling
# ---------------------------------------------------------------------------

def test_collapse_above_mass_is_captured() -> None:
    p = _poly_program()
    heavy_mass = _cell_size(_symbolic_cells(p)[0])
    # ceiling below the heavy cell's mass but above the light one's.
    out = specialize(p, budget=SimplifyBudget(max_cell_mass=heavy_mass - 1))
    # exactly the heavy cell collapses -> one IdentityOp capture Stmt appears.
    assert len(out.statements) == 1
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


def test_collapse_above_budget_is_noop() -> None:
    p = _poly_program()
    big = 10_000
    out = specialize(p, budget=SimplifyBudget(max_cell_mass=big))
    assert len(out.statements) == 0  # nothing over budget
    x = {"a": np.array([3.0, -1.0, 2.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


# ---------------------------------------------------------------------------
# numeric_only() collapses all symbolic cells AND == partial_eval(.., 0)
# ---------------------------------------------------------------------------

def test_numeric_only_collapses_all_symbolic() -> None:
    p = _poly_program()
    out = specialize(p, budget=SimplifyBudget.numeric_only())
    assert _n_symbolic(out) == 0  # every symbolic cell captured
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


def test_numeric_only_equals_partial_eval() -> None:
    p = _poly_program()
    a = specialize(p, budget=SimplifyBudget.numeric_only())
    b = partial_eval(p, max_cell_size=0)
    # Same number of capture Stmts, and identical resulting output-cell atoms.
    assert len(a.statements) == len(b.statements)
    acells = [str(c) for c in np.asarray(a.outputs["out"].cells).reshape(-1)]
    bcells = [str(c) for c in np.asarray(b.outputs["out"].cells).reshape(-1)]
    assert acells == bcells
    # And same as the general balanced collapse at the same ceiling.
    for m in (0, 1, 3, 5):
        sa = specialize(p, budget=SimplifyBudget(max_cell_mass=m))
        pe = partial_eval(p, max_cell_size=m)
        assert len(sa.statements) == len(pe.statements)
        x = {"a": np.array([2.0, -3.0, 1.5])}
        np.testing.assert_allclose(sa.run(x)["out"], pe.run(x)["out"])


# ---------------------------------------------------------------------------
# monotone collapse: smaller max_cell_mass => fewer/equal symbolic cells
# ---------------------------------------------------------------------------

def test_monotone_collapse() -> None:
    p = _poly_program()
    masses = [0, 2, 4, 6, 100]
    counts = [_n_symbolic(specialize(p, budget=SimplifyBudget(max_cell_mass=m)))
              for m in masses]
    # non-decreasing in the ceiling (looser budget keeps >= symbolic cells).
    assert counts == sorted(counts)
    x = {"a": np.array([1.0, 2.0, 3.0])}
    for m in masses:
        out = specialize(p, budget=SimplifyBudget(max_cell_mass=m))
        np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


# ---------------------------------------------------------------------------
# idempotence: specialize(specialize(p,B),B) == specialize(p,B)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "budget",
    [
        SimplifyBudget.none(),
        SimplifyBudget.numeric_only(),
        SimplifyBudget.balanced(3),
        SimplifyBudget(max_cell_mass=2, total_mass=4),
    ],
)
def test_idempotence(budget) -> None:
    p = _poly_program()
    once = specialize(p, budget=budget)
    twice = specialize(once, budget=budget)
    assert len(once.statements) == len(twice.statements)
    assert _n_symbolic(once) == _n_symbolic(twice)
    x = {"a": np.array([1.5, -2.0, 0.5])}
    np.testing.assert_allclose(once.run(x)["out"], twice.run(x)["out"])
    np.testing.assert_allclose(once.run(x)["out"], p.run(x)["out"])


# ---------------------------------------------------------------------------
# exactness over presets x random seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "budget",
    [
        SimplifyBudget.none(),
        SimplifyBudget.legacy(),
        SimplifyBudget.numeric_only(),
        SimplifyBudget.balanced(2),
        SimplifyBudget.balanced(5),
        SimplifyBudget.expose_symbols(keep=frozenset({"coeff"})),
        SimplifyBudget(total_mass=4),
        SimplifyBudget(max_cell_mass=1, total_mass=3),
    ],
)
def test_exactness_random_seeds(budget) -> None:
    p = _poly_program()
    out = specialize(p, budget=budget)
    rng = np.random.default_rng(0)
    for _ in range(8):
        x = {"a": rng.standard_normal(3)}
        np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"], rtol=1e-9)


# ---------------------------------------------------------------------------
# numeric-fold floor is independent of budget
# ---------------------------------------------------------------------------

def test_numeric_fold_floor_independent_of_budget() -> None:
    # A program with a fully-numeric subtree folds the SAME regardless of budget.
    p = Program("mixed", inputs=[SymInput("a", (2,), _prov("a"))])
    a = p.input("a").cells
    # one symbolic cell + one numeric-constant cell.
    num = RationalFunction.constant(7.0)
    cells = np.empty((2,), dtype=object)
    cells[0] = a[0] + a[1]
    cells[1] = num
    p.add_output("out", cells)
    x = {"a": np.array([1.0, 2.0])}
    for budget in (SimplifyBudget.none(), SimplifyBudget.numeric_only(),
                   SimplifyBudget.balanced(1)):
        out = specialize(p, budget=budget)
        got = out.run(x)["out"]
        np.testing.assert_allclose(got, p.run(x)["out"])
        # the constant cell folded to a float regardless of budget.
        assert float(got[1]) == 7.0


# ---------------------------------------------------------------------------
# keep_provenance: a vertex cell is NOT collapsed even when over budget
# ---------------------------------------------------------------------------

def test_keep_provenance_protects_vertex_cells() -> None:
    p = _poly_program()  # all cells are over "vertex" atoms
    out = specialize(
        p,
        budget=SimplifyBudget(max_cell_mass=0, keep_provenance=frozenset({"vertex"})),
    )
    # nothing collapsed -> no capture Stmts, all cells stay symbolic.
    assert len(out.statements) == 0
    assert _n_symbolic(out) == 2
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


def test_keep_provenance_other_kind_does_not_protect() -> None:
    # keeping a DIFFERENT kind does not protect a vertex cell -> it collapses.
    p = _poly_program()
    out = specialize(
        p,
        budget=SimplifyBudget(max_cell_mass=0, keep_provenance=frozenset({"coeff"})),
    )
    assert _n_symbolic(out) == 0
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


# ---------------------------------------------------------------------------
# den_degree_max: a degree-2 denominator cell is extracted; exactness holds
# ---------------------------------------------------------------------------

def test_den_degree_extraction() -> None:
    p = _den_program()
    # default den_degree_max=1; the cell's denominator degree is 2 -> extracted.
    out = specialize(p, budget=SimplifyBudget())
    assert len(out.statements) == 1  # one named-intermediate capture
    x = {"b": np.array([4.0, 2.0, 5.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


def test_den_degree_high_threshold_is_noop() -> None:
    p = _den_program()
    out = specialize(p, budget=SimplifyBudget(den_degree_max=5))
    assert len(out.statements) == 0  # degree 2 <= 5 -> left symbolic
    x = {"b": np.array([4.0, 2.0, 5.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


# ---------------------------------------------------------------------------
# total_mass: greedy global collapse until under the whole-program ceiling
# ---------------------------------------------------------------------------

def test_total_mass_greedy_collapse() -> None:
    p = _poly_program()  # masses 7 + 3 = 10 total
    out = specialize(p, budget=SimplifyBudget(total_mass=5))
    # heaviest (mass 7) collapses first; 3 + atom(2) = 5 <= 5 -> stop.
    assert len(out.statements) == 1
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


def test_total_mass_high_is_noop() -> None:
    p = _poly_program()
    out = specialize(p, budget=SimplifyBudget(total_mass=10_000))
    assert len(out.statements) == 0
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


# ---------------------------------------------------------------------------
# presets + passthrough params
# ---------------------------------------------------------------------------

def test_sparsity_passthrough_noop() -> None:
    p = _poly_program()
    x = {"a": np.array([1.0, 2.0, 3.0])}
    # ``sparsity`` is accepted for API parity but is a separate pass (P4) — it has
    # no effect on the program specialize() returns.  (``subs`` is now a real
    # transform post-merge; see tests/test_substitute.py.)
    out = specialize(p, sparsity=True)
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


def test_budget_none_is_floor_only() -> None:
    p = _poly_program()
    out = specialize(p, budget=None)
    assert len(out.statements) == 0  # no collapse, just the fold floor (no-op here)
    x = {"a": np.array([1.0, 2.0, 3.0])}
    np.testing.assert_allclose(out.run(x)["out"], p.run(x)["out"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
