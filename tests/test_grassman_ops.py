"""Stage-A additions for Grassman lowering: SvdOp / QrOp / AssertOp + force_stmts.

Mirrors the construction idioms in ``test_golden_programs.py``: small
:class:`Program`s built by hand and executed via ``Program.run``, plus
direct op ``__call__`` checks.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import (
    AssertOp,
    DetOp,
    EinsumStmtOp,
    OutSpec,
    Program,
    Provenance,
    QrOp,
    RationalFunction,
    SymArray,
    SymInput,
    SymbolicBudget,
    SvdOp,
)


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


# ---------------------------------------------------------------------------
# SvdOp — rank on a known-rank matrix + reconstruction
# ---------------------------------------------------------------------------

def test_svdop_rank_and_reconstruction() -> None:
    rng = np.random.default_rng(0)
    # Rank-2 4x4: outer product of two independent column/row pairs.
    u = rng.standard_normal((4, 2))
    v = rng.standard_normal((2, 4))
    A = u @ v
    op = SvdOp()
    U, S, Vh, rank = op(A)
    assert int(rank) == 2
    recon = U @ np.diag(S) @ Vh
    np.testing.assert_allclose(recon, A, atol=1e-9)


def test_svdop_full_rank() -> None:
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3)) + 3.0 * np.eye(3)
    U, S, Vh, rank = SvdOp()(A)
    assert int(rank) == 3


def test_svdop_in_program_multi_output() -> None:
    rng = np.random.default_rng(2)
    u = rng.standard_normal((5, 3))
    v = rng.standard_normal((3, 5))
    An = u @ v  # rank 3

    prog = Program("svd")
    A, vals = _sym_matrix(prog, An, "a")
    U, S, Vh, rank = prog.emit_stmt(
        SvdOp(),
        [A],
        [OutSpec("U", (5, 5)), OutSpec("S", (5,)), OutSpec("Vh", (5, 5)),
         OutSpec("rank", ())],
        note="svd",
        bulk=True,
    )
    assert isinstance(prog.statements[-1].fn, SvdOp)
    assert len(prog.statements[-1].out) == 4
    prog.add_output("rank", rank)
    prog.add_output("S", S)
    res = prog.run(vals)
    assert int(round(float(res["rank"]))) == 3
    # Singular values match numpy's.
    np.testing.assert_allclose(np.sort(res["S"]), np.sort(np.linalg.svd(An, compute_uv=False)), atol=1e-9)


# ---------------------------------------------------------------------------
# QrOp — Q orthonormal & Q @ R ≈ A
# ---------------------------------------------------------------------------

def test_qrop_orthonormal_and_reconstruction() -> None:
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 3))
    Q, R = QrOp()(A)
    # Q columns orthonormal.
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-9)
    np.testing.assert_allclose(Q @ R, A, atol=1e-9)


def test_qrop_in_program() -> None:
    rng = np.random.default_rng(4)
    An = rng.standard_normal((4, 4))
    prog = Program("qr")
    A, vals = _sym_matrix(prog, An, "q")
    Q, R = prog.emit_stmt(
        QrOp(), [A], [OutSpec("Q", (4, 4)), OutSpec("R", (4, 4))], note="qr", bulk=True,
    )
    assert isinstance(prog.statements[-1].fn, QrOp)
    prog.add_output("Q", Q)
    prog.add_output("R", R)
    res = prog.run(vals)
    np.testing.assert_allclose(res["Q"] @ res["R"], An, atol=1e-9)
    np.testing.assert_allclose(res["Q"].T @ res["Q"], np.eye(4), atol=1e-9)


# ---------------------------------------------------------------------------
# AssertOp — passes/raises per kind + passthrough
# ---------------------------------------------------------------------------

def test_assertop_shape_eq() -> None:
    op = AssertOp("shape_eq", msg="shapes: ")
    x = np.zeros((2, 3))
    out = op(x, np.ones((2, 3)))
    np.testing.assert_array_equal(out, x)  # passthrough
    with pytest.raises(AssertionError, match="shapes: "):
        op(x, np.ones((2, 4)))


def test_assertop_rank_eq() -> None:
    op = AssertOp("rank_eq")
    out = op(np.array(7.0), np.asarray(3), np.asarray(3))
    np.testing.assert_array_equal(out, np.array(7.0))  # passthrough first input
    with pytest.raises(AssertionError):
        op(np.array(7.0), np.asarray(3), np.asarray(4))


def test_assertop_spd() -> None:
    op = AssertOp("spd")
    spd = np.array([[2.0, 0.0], [0.0, 3.0]])
    np.testing.assert_array_equal(op(spd), spd)
    with pytest.raises(AssertionError):
        op(np.array([[0.0, 1.0], [1.0, 0.0]]))  # symmetric but indefinite
    with pytest.raises(AssertionError):
        op(np.array([[1.0, 2.0], [0.0, 1.0]]))  # not symmetric


def test_assertop_square_full_rank() -> None:
    op = AssertOp("square_full_rank")
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_array_equal(op(A), A)
    with pytest.raises(AssertionError):
        op(np.array([[1.0, 1.0], [1.0, 1.0]]))  # rank-deficient
    with pytest.raises(AssertionError):
        op(np.zeros((2, 3)))  # not square


def test_assertop_unknown_kind() -> None:
    with pytest.raises(ValueError):
        AssertOp("nonsense")(np.array(1.0))


def test_assertop_in_program_passthrough() -> None:
    prog = Program("assert", inputs=[SymInput("x", (2, 2), _prov("x"))])
    x = prog.input("x")
    [out] = prog.emit_stmt(
        AssertOp("square_full_rank", msg="x must be invertible: "),
        [x],
        [OutSpec("checked", (2, 2))],
        note="assert",
        bulk=True,
    )
    assert isinstance(prog.statements[-1].fn, AssertOp)
    prog.add_output("checked", out)
    xn = np.array([[2.0, 1.0], [1.0, 3.0]])
    res = prog.run({"x": xn})
    np.testing.assert_allclose(res["checked"], xn)
    with pytest.raises(AssertionError):
        prog.run({"x": np.ones((2, 2))})  # rank-deficient


# ---------------------------------------------------------------------------
# force_stmts — every modeled op emits a Stmt
# ---------------------------------------------------------------------------

def test_force_stmts_det_emits_stmt() -> None:
    rng = np.random.default_rng(5)
    Mn = rng.standard_normal((2, 2)) + 2.0 * np.eye(2)
    prog = Program("fs_det", budget=SymbolicBudget.force_stmts())
    A, vals = _sym_matrix(prog, Mn, "fd")
    out = A.det()
    # 2x2 is normally closed-form; force_stmts (naive_inverse_max_size=0) defers.
    assert len(prog.statements) >= 1
    assert isinstance(prog.statements[-1].fn, DetOp)
    prog.add_output("det", out)
    res = prog.run(vals)
    np.testing.assert_allclose(res["det"], np.linalg.det(Mn), rtol=1e-9)


def test_force_stmts_inverse_emits_stmt() -> None:
    from polyarray import InvOp
    rng = np.random.default_rng(6)
    Mn = rng.standard_normal((2, 2)) + 2.0 * np.eye(2)
    prog = Program("fs_inv", budget=SymbolicBudget.force_stmts())
    A, vals = _sym_matrix(prog, Mn, "fi")
    out = A.inverse()
    assert isinstance(prog.statements[-1].fn, InvOp)
    prog.add_output("inv", out)
    res = prog.run(vals)
    np.testing.assert_allclose(res["inv"], np.linalg.inv(Mn), rtol=1e-7)


def test_force_stmts_multi_einsum_emits_stmt() -> None:
    from polyarray.ir import runtime_einsum_multi
    rng = np.random.default_rng(7)
    An = rng.standard_normal((2, 3))
    Bn = rng.standard_normal((3, 4))
    prog = Program("fs_es", budget=SymbolicBudget.force_stmts())
    A, va = _sym_matrix(prog, An, "ea")
    B, vb = _sym_matrix(prog, Bn, "eb")
    n_before = len(prog.statements)
    out = runtime_einsum_multi(
        "ij,jk->ik",
        np.asarray(A.cells),
        np.asarray(B.cells),
        program=prog,
        out_shape=(2, 4),
    )
    assert len(prog.statements) > n_before
    assert isinstance(prog.statements[-1].fn, EinsumStmtOp)
    out_sa = SymArray(np.asarray(out), program=prog)
    prog.add_output("C", out_sa)
    res = prog.run({**va, **vb})
    np.testing.assert_allclose(res["C"], An @ Bn, atol=1e-9)


def test_force_stmts_is_symbolic_budget() -> None:
    bud = SymbolicBudget.force_stmts()
    assert isinstance(bud, SymbolicBudget)
    assert bud.naive_inverse_max_size == 0
    assert bud.inverse_max_degree == 0
    assert bud.einsum_bag_threshold == 1
    assert bud.freeze is True
    # overrides flow through
    bud2 = SymbolicBudget.force_stmts(iszero_tol=1e-9)
    assert bud2.iszero_tol == 1e-9
