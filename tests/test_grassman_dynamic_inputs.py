"""Stage-C additions for Grassman lowering: dynamic (runtime-dim) INPUTS.

A ``Program`` *input* may itself be runtime-sized (an FFS-typed Grassmann
input like a ``Λᵏ`` vector whose dimension ``δ`` is known only at run time).
Stage B sized *Stmt outputs* dynamically; Stage C does the same for *inputs*
and adds runtime ``AssertOp``s that make a runtime-sized input safe.

The key test runs the SAME Program on two differently-sized arrays and gets
two different downstream shapes; a mis-sized input trips an ``AssertOp``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from polyarray import (
    AssertOp,
    DimAtom,
    OutSpec,
    Program,
    Provenance,
    SymInput,
)
from polyarray.ir import is_dynamic


def _prov(label: str) -> Provenance:
    return Provenance(kind="vertex", origin=label, index=(), label=label)


@dataclass(frozen=True)
class _ScaleOp:
    """Multiply a (runtime-sized) vector by a constant — shape-preserving."""

    factor: float = 2.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.factor * np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# DimAtom input-axis source (C2)
# ---------------------------------------------------------------------------

def test_dim_atom_input_source_tagged() -> None:
    d = DimAtom.from_input("delta:a", "a", 0)
    assert d.source == ("in", "a", 0)
    # hashable / usable as a dim_bindings key
    assert {d: 7}[DimAtom(name="delta:a", source=("in", "a", 0))] == 7


def test_dim_atom_stmt_source_compat_shim() -> None:
    # Stage-B 2-tuple form still works and normalises to the tagged form.
    a = DimAtom(name="rank:f", source=(0, 3))
    assert a.source == ("stmt", 0, 3)
    assert a == DimAtom(name="rank:f", source=("stmt", 0, 3))


def test_dynamic_input_shape_is_dynamic() -> None:
    delta = DimAtom.from_input("delta:a", "a", 0)
    assert is_dynamic((delta,))
    assert is_dynamic((delta, 3))
    assert not is_dynamic((3, 4))


# ---------------------------------------------------------------------------
# A dynamic bulk input allocates no per-cell atoms (C1)
# ---------------------------------------------------------------------------

def test_dynamic_input_is_bulk() -> None:
    delta = DimAtom.from_input("delta:a", "a", 0)
    prog = Program(
        "dyn_in",
        inputs=[SymInput("a", (delta,), _prov("a"))],
    )
    sa = prog.input("a")
    assert sa._bulk is not None
    assert sa.shape == (delta,)
    assert prog._has_dynamic_shape()


def test_static_input_unchanged() -> None:
    prog = Program("static", inputs=[SymInput("a", (3,), _prov("a"))])
    sa = prog.input("a")
    assert sa._bulk is None
    assert not prog._has_dynamic_shape()


# ---------------------------------------------------------------------------
# THE KEY TEST: same Program, two input sizes -> two downstream shapes (C3/C5)
# ---------------------------------------------------------------------------

def _scale_program() -> Program:
    delta = DimAtom.from_input("delta:a", "a", 0)
    prog = Program("scale", inputs=[SymInput("a", (delta,), _prov("a"))])
    a = prog.input("a")
    [out] = prog.emit_stmt(
        _ScaleOp(3.0),
        [a],
        [OutSpec("scaled", (delta,))],
        note="scale",
    )
    prog.add_output("scaled", out)
    return prog


def test_dynamic_input_two_sizes() -> None:
    prog = _scale_program()

    r3 = prog.run({"a": np.array([1.0, 2.0, 3.0])})
    assert r3["scaled"].shape == (3,)
    np.testing.assert_allclose(r3["scaled"], [3.0, 6.0, 9.0])

    r5 = prog.run({"a": np.arange(5.0)})
    assert r5["scaled"].shape == (5,)
    np.testing.assert_allclose(r5["scaled"], 3.0 * np.arange(5.0))


# ---------------------------------------------------------------------------
# sibling agreement: two inputs that must share a δ (C4)
# ---------------------------------------------------------------------------

def _sibling_program() -> Program:
    delta = DimAtom.from_input("delta:a", "a", 0)
    prog = Program(
        "siblings",
        inputs=[
            SymInput("a", (delta,), _prov("a")),
            SymInput("b", (delta,), _prov("b")),
        ],
    )
    a = prog.input("a")
    b = prog.input("b")
    # shape_eq assert: a and b must share their δ axis.
    [a_checked] = prog.emit_stmt(
        AssertOp("shape_eq", msg="a,b: Λᵏ siblings"),
        [a, b],
        [OutSpec("a_checked", (delta,))],
        note="assert sibling shape_eq a,b",
    )
    [out] = prog.emit_stmt(
        _ScaleOp(1.0),
        [a_checked],
        [OutSpec("out", (delta,))],
        note="passthrough",
    )
    prog.add_output("out", out)
    return prog


def test_sibling_agreement_ok() -> None:
    prog = _sibling_program()
    r = prog.run({"a": np.ones(4), "b": np.zeros(4)})
    assert r["out"].shape == (4,)


def test_sibling_mismatch_trips_assert() -> None:
    prog = _sibling_program()
    with pytest.raises(AssertionError):
        prog.run({"a": np.ones(4), "b": np.zeros(5)})


# ---------------------------------------------------------------------------
# mis-sized input trips a rank-consistency AssertOp (C4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RankOp:
    """Return the (asserted) rank a defining map would produce."""

    rank: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.rank)


@dataclass(frozen=True)
class _LenOp:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(int(np.asarray(x).shape[0]))


def _rank_consistency_program(expected_rank: int) -> Program:
    delta = DimAtom.from_input("delta:a", "a", 0)
    prog = Program(
        "rank_consistency",
        inputs=[SymInput("a", (delta,), _prov("a"))],
    )
    a = prog.input("a")
    [computed] = prog.emit_stmt(
        _RankOp(expected_rank), [a], [OutSpec("computed_rank", ())],
        note="svd_rank(Alt)",
    )
    [actual] = prog.emit_stmt(
        _LenOp(), [a], [OutSpec("actual_len", ())],
        note="len(a)=δ_input",
    )
    # rank_eq: computed map rank must equal the input's bound δ.
    [checked] = prog.emit_stmt(
        AssertOp("rank_eq", msg="FFS input δ vs svd_rank(Alt)"),
        [a, computed, actual],
        [OutSpec("a_checked", (delta,))],
        note="assert rank_eq",
    )
    prog.add_output("out", checked)
    return prog


def test_rank_consistency_ok() -> None:
    prog = _rank_consistency_program(expected_rank=4)
    r = prog.run({"a": np.ones(4)})
    assert r["out"].shape == (4,)


def test_rank_consistency_mismatch_trips_assert() -> None:
    prog = _rank_consistency_program(expected_rank=4)
    with pytest.raises(AssertionError):
        prog.run({"a": np.ones(5)})


# ---------------------------------------------------------------------------
# a concrete-axis mismatch on a partially-dynamic input is caught (C3)
# ---------------------------------------------------------------------------

def test_dynamic_input_concrete_axis_mismatch() -> None:
    delta = DimAtom.from_input("delta:a", "a", 0)
    prog = Program(
        "mixed",
        inputs=[SymInput("a", (delta, 3), _prov("a"))],
    )
    a = prog.input("a")
    [out] = prog.emit_stmt(
        _ScaleOp(1.0), [a], [OutSpec("out", (delta, 3))], note="id",
    )
    prog.add_output("out", out)
    # ok: dynamic axis 0 = 4, concrete axis 1 = 3
    assert prog.run({"a": np.ones((4, 3))})["out"].shape == (4, 3)
    # concrete axis 1 mismatched
    with pytest.raises(ValueError):
        prog.run({"a": np.ones((4, 5))})
