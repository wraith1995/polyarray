"""Exactness + reduction tests for the `simplify` numeric-fold pass."""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import (
    DetOp,
    OutSpec,
    Program,
    Provenance,
    SymInput,
    TensordotOp,
    bind_inputs,
    fold_numeric,
    strip_asserts,
)
from polyarray.ir import AssertOp, IdentityOp, OutputRef, Stmt


def _prov(name: str):
    return Provenance("vertex", name, (), name)


# ---------------------------------------------------------------------------
# fold_numeric on a fully-symbolic program is an exact no-op (degrades to copy)
# ---------------------------------------------------------------------------

def test_strip_asserts_replaces_asserts_preserving_value() -> None:
    prog = Program("a", inputs=[SymInput("A", (3, 3), _prov("A"))])
    [Ac] = prog.emit_stmt(
        AssertOp(kind="shape_eq", msg="A vs A"),
        [prog.input("A"), prog.input("A")],
        [OutSpec("Ac", (3, 3))],
        bulk=False,
    )
    prog.add_output("Ac", Ac.cells)
    assert any(isinstance(s.fn, AssertOp) for s in prog.statements)

    stripped = strip_asserts(prog)
    assert not any(isinstance(s.fn, AssertOp) for s in stripped.statements)
    assert any(isinstance(s.fn, IdentityOp) for s in stripped.statements)

    rng = np.random.default_rng(0)
    vals = {"A": rng.standard_normal((3, 3))}
    np.testing.assert_allclose(stripped.run(vals)["Ac"], prog.run(vals)["Ac"])


def test_strip_asserts_noop_returns_same_object() -> None:
    prog = Program(
        "td",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))],
    )
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("B")],
        [OutSpec("C", (2, 4))],
        bulk=False,
    )
    prog.add_output("C", C.cells)
    assert strip_asserts(prog) is prog


def test_fold_numeric_symbolic_is_noop() -> None:
    prog = Program(
        "td",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))],
    )
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("B")],
        [OutSpec("C", (2, 4))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    folded = fold_numeric(prog)
    # Nothing was numeric, so the Stmt survives untouched.
    assert len(folded.statements) == 1

    rng = np.random.default_rng(0)
    vals = {"A": rng.standard_normal((2, 3)), "B": rng.standard_normal((3, 4))}
    np.testing.assert_allclose(folded.run(vals)["C"], prog.run(vals)["C"])


# ---------------------------------------------------------------------------
# bind every input -> all Stmts fold away, output is pure numeric
# ---------------------------------------------------------------------------

def test_bind_full_collapses_to_numeric() -> None:
    prog = Program("det", inputs=[SymInput("A", (3, 3), _prov("A"))])
    [out] = prog.emit_stmt(
        DetOp(), [prog.input("A")], [OutSpec("det", ())], bulk=False,
    )
    prog.add_output("det", out.cells)

    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))

    folded = bind_inputs(prog, {"A": A})
    # The whole computation reduced: no Stmts, no remaining inputs.
    assert len(folded.statements) == 0
    assert folded.inputs == ()
    # Exact: runs with NO inputs and matches the original.
    np.testing.assert_allclose(folded.run({})["det"], np.linalg.det(A), rtol=1e-9)
    np.testing.assert_allclose(folded.run({})["det"], prog.run({"A": A})["det"], rtol=1e-9)


# ---------------------------------------------------------------------------
# bind ONE of two inputs -> the bound stmt folds, residual stays symbolic
# (the "leave residual symbols" / partial case)
# ---------------------------------------------------------------------------

def test_bind_partial_leaves_residual_symbols() -> None:
    # out = (A @ B) @ C ; bind A,B numeric, leave C symbolic.
    prog = Program(
        "chain",
        inputs=[
            SymInput("A", (2, 2), _prov("A")),
            SymInput("B", (2, 2), _prov("B")),
            SymInput("C", (2, 2), _prov("C")),
        ],
    )
    [AB] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("B")],
        [OutSpec("AB", (2, 2))],
        bulk=False,
    )
    [ABC] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [AB, prog.input("C")],
        [OutSpec("ABC", (2, 2))],
        bulk=False,
    )
    prog.add_output("ABC", ABC.cells)

    rng = np.random.default_rng(2)
    A = rng.standard_normal((2, 2))
    B = rng.standard_normal((2, 2))
    Cv = rng.standard_normal((2, 2))

    folded = bind_inputs(prog, {"A": A, "B": B})
    # The A@B stmt folded into a constant; only the @C stmt remains.
    assert len(folded.statements) == 1
    assert "C" in {inp.name for inp in folded.inputs}
    assert "A" not in {inp.name for inp in folded.inputs}

    got = folded.run({"C": Cv})["ABC"]
    want = prog.run({"A": A, "B": B, "C": Cv})["ABC"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# the pass never disturbs the original program (copy / shared-cell safety)
# ---------------------------------------------------------------------------

def test_original_program_untouched() -> None:
    prog = Program("det", inputs=[SymInput("A", (2, 2), _prov("A"))])
    [out] = prog.emit_stmt(
        DetOp(), [prog.input("A")], [OutSpec("det", ())], bulk=False,
    )
    prog.add_output("det", out.cells)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    _ = bind_inputs(prog, {"A": A})

    assert len(prog.statements) == 1
    assert {inp.name for inp in prog.inputs} == {"A"}
    np.testing.assert_allclose(prog.run({"A": A})["det"], np.linalg.det(A), rtol=1e-9)


# ---------------------------------------------------------------------------
# bulk-output Stmt folds when its inputs are bound numeric
# ---------------------------------------------------------------------------

def test_bind_bulk_output_folds() -> None:
    prog = Program(
        "td",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))],
    )
    # bulk=True is the emit_stmt default; the whole tensor is one runtime handle.
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("B")],
        [OutSpec("C", (2, 4))],
        bulk=True,
    )
    assert C._bulk is not None
    prog.add_output("C", C)  # keep the bulk handle (do not unpack)

    rng = np.random.default_rng(3)
    Av = rng.standard_normal((2, 3))
    Bv = rng.standard_normal((3, 4))

    folded = bind_inputs(prog, {"A": Av, "B": Bv})
    assert len(folded.statements) == 0
    assert folded.inputs == ()
    np.testing.assert_allclose(
        folded.run({})["C"], np.tensordot(Av, Bv, axes=([1], [0])), rtol=1e-9
    )


# ---------------------------------------------------------------------------
# a sub-Program Stmt folds (executed via .run) when all operands are numeric
# ---------------------------------------------------------------------------

def test_bind_subprogram_folds() -> None:
    sub = Program("sub_det", inputs=[SymInput("x", (2, 2), _prov("x"))])
    [d] = sub.emit_stmt(DetOp(), [sub.input("x")], [OutSpec("d", ())], bulk=False)
    sub.add_output("d", d.cells)

    top = Program("top", inputs=[SymInput("M", (2, 2), _prov("M"))])
    [out] = top.emit_stmt(sub, [top.input("M")], [OutSpec("d", ())], bulk=False)
    top.add_output("d", out.cells)

    M = np.array([[2.0, 1.0], [0.0, 3.0]])
    folded = bind_inputs(top, {"M": M})
    assert len(folded.statements) == 0
    np.testing.assert_allclose(folded.run({})["d"], np.linalg.det(M), rtol=1e-9)
    np.testing.assert_allclose(
        folded.run({})["d"], top.run({"M": M})["d"], rtol=1e-9
    )


# ---------------------------------------------------------------------------
# three-stmt chain, bind the head: middle folds, tail survives (idx remap)
# ---------------------------------------------------------------------------

def test_three_stmt_chain_partial_fold() -> None:
    prog = Program(
        "chain3",
        inputs=[
            SymInput("A", (2, 2), _prov("A")),
            SymInput("B", (2, 2), _prov("B")),
            SymInput("C", (2, 2), _prov("C")),
            SymInput("D", (2, 2), _prov("D")),
        ],
    )
    td = TensordotOp.from_axes(([1], [0]))
    [AB] = prog.emit_stmt(td, [prog.input("A"), prog.input("B")], [OutSpec("AB", (2, 2))], bulk=False)
    [ABC] = prog.emit_stmt(td, [AB, prog.input("C")], [OutSpec("ABC", (2, 2))], bulk=False)
    [ABCD] = prog.emit_stmt(td, [ABC, prog.input("D")], [OutSpec("ABCD", (2, 2))], bulk=False)
    prog.add_output("ABCD", ABCD.cells)

    rng = np.random.default_rng(5)
    A, B = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
    Cv, Dv = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))

    folded = bind_inputs(prog, {"A": A, "B": B})
    # A@B and (A@B)@... no: only A@B is fully numeric; @C still needs C symbolic.
    # So stmt0 folds; stmt1 (@C) and stmt2 (@D) survive.
    assert len(folded.statements) == 2
    got = folded.run({"C": Cv, "D": Dv})["ABCD"]
    want = prog.run({"A": A, "B": B, "C": Cv, "D": Dv})["ABCD"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# OutputRef stmt-indices are remapped when an earlier Stmt folds away
# ---------------------------------------------------------------------------

def test_outputref_reindex_after_fold() -> None:
    prog = Program(
        "orr",
        inputs=[SymInput("X", (2, 2), _prov("X")), SymInput("C", (2, 2), _prov("C"))],
    )
    td = TensordotOp.from_axes(([1], [0]))
    # stmt0: X@X — folds once X is bound (its output is unused, just removed).
    prog.emit_stmt(td, [prog.input("X"), prog.input("X")], [OutSpec("XX", (2, 2))], bulk=False)
    # stmt1: C@C — symbolic, survives (becomes new index 0).
    [CC] = prog.emit_stmt(td, [prog.input("C"), prog.input("C")], [OutSpec("CC", (2, 2))], bulk=False)
    # stmt2: references stmt1 via OutputRef(1,0) — survives (new index 1); the
    # OutputRef's stmt_idx must remap 1 -> 0 after stmt0 folds away.
    [CCC] = prog.emit_stmt(td, [CC, prog.input("C")], [OutSpec("CCC", (2, 2))], bulk=False)
    s2 = prog.statements[2]
    prog.statements[2] = Stmt(
        fn=s2.fn, in_=(OutputRef(1, 0, ()), s2.in_[1]), out=s2.out,
        note=s2.note, provenance=s2.provenance, inline=s2.inline,
    )
    prog.add_output("CCC", CCC.cells)

    rng = np.random.default_rng(7)
    Xv, Cv = rng.standard_normal((2, 2)), rng.standard_normal((2, 2))
    base = prog.run({"X": Xv, "C": Cv})["CCC"]  # sanity: original runs w/ OutputRef

    folded = bind_inputs(prog, {"X": Xv})
    assert len(folded.statements) == 2  # stmt0 (X@X) folded away
    assert isinstance(folded.statements[1].in_[0], OutputRef)
    assert folded.statements[1].in_[0].stmt_idx == 0  # remapped 1 -> 0
    np.testing.assert_allclose(folded.run({"C": Cv})["CCC"], base, rtol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# partial_eval_numeric: probe-and-freeze — folds INVARIANT (not just numeric-
# closed) subcomputations. Canonical case: A·inv(A) ≡ I for every A.
# ---------------------------------------------------------------------------

def test_partial_eval_numeric_freezes_invariant_chain() -> None:
    from polyarray import InvOp, partial_eval_numeric

    prog = Program("painv", inputs=[SymInput("A", (2, 2), _prov("A"))])
    [B] = prog.emit_stmt(InvOp(), [prog.input("A")], [OutSpec("B", (2, 2))], bulk=False)
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), OutputRef(0, 0)],
        [OutSpec("C", (2, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    # fold_numeric cannot touch it (A is symbolic everywhere) …
    assert len(fold_numeric(prog).statements) == 2
    # … probing discovers C ≡ I and freezes BOTH stmts (inv(A) varies, so it
    # survives only if referenced — here its sole consumer folded too, but we
    # keep statement-level granularity: the inv stmt itself VARIES per probe).
    folded = partial_eval_numeric(prog)
    rng = np.random.default_rng(7)
    A = rng.uniform(0.5, 1.5, (2, 2)) + np.eye(2)
    np.testing.assert_allclose(folded.run({"A": A})["C"], np.eye(2), atol=1e-9)
    # the matmul stmt folded away; C's cells are numeric constants
    assert len(folded.statements) < 2 or all(
        not isinstance(c, type(prog.outputs["C"].cells.ravel()[0]))
        for c in np.asarray(folded.outputs["C"].cells).ravel()
    ) or np.allclose(np.asarray(folded.outputs["C"].cells, dtype=float), np.eye(2))


def test_partial_eval_numeric_keeps_genuinely_variant_outputs() -> None:
    from polyarray import partial_eval_numeric

    prog = Program(
        "var",
        inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))],
    )
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("B")],
        [OutSpec("C", (2, 4))],
        bulk=False,
    )
    prog.add_output("C", C.cells)
    folded = partial_eval_numeric(prog)
    assert len(folded.statements) == 1              # A·B genuinely varies — untouched
    rng = np.random.default_rng(1)
    vals = {"A": rng.standard_normal((2, 3)), "B": rng.standard_normal((3, 4))}
    np.testing.assert_allclose(folded.run(vals)["C"], prog.run(vals)["C"])
