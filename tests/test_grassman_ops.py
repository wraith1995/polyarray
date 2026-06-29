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
    GSvdOp,
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


def _rand_spd(rng: np.random.Generator, k: int) -> np.ndarray:
    X = rng.standard_normal((k, k))
    return X @ X.T + k * np.eye(k)


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
# GSvdOp — metric-aware generalised SVD
# ---------------------------------------------------------------------------

def test_gsvdop_identity_metrics_matches_svd() -> None:
    rng = np.random.default_rng(10)
    A = rng.standard_normal((5, 4))
    Iv, Iw = np.eye(4), np.eye(5)
    U, UI, V, VI, S, rank = GSvdOp()(A, Iv, Iw)

    # rank + singular values agree with the plain SVD.
    Us, Ss, Vhs = np.linalg.svd(A, full_matrices=True)
    assert int(rank) == int((Ss > Ss.max() * max(A.shape) * np.finfo(float).eps).sum())
    np.testing.assert_allclose(S, Ss, atol=1e-9)

    # Full factors are (plain) orthonormal and reconstruct A = [U|UI] Sfull [V|VI]^T.
    Ufull = np.concatenate([U, UI], axis=1)
    Vfull = np.concatenate([V, VI], axis=1)
    np.testing.assert_allclose(Ufull.T @ Ufull, np.eye(5), atol=1e-9)
    np.testing.assert_allclose(Vfull.T @ Vfull, np.eye(4), atol=1e-9)
    Sfull = np.zeros((5, 4))
    Sfull[: S.size, : S.size] = np.diag(S)
    # With identity metrics the documented identity reduces to U Sfull V^T,
    # i.e. Vfull^T plays SvdOp's Vh role (matching factorisation, sign/subspace aside).
    np.testing.assert_allclose(Ufull @ Sfull @ Vfull.T, A, atol=1e-9)


def test_gsvdop_metric_orthonormality_and_reconstruction() -> None:
    rng = np.random.default_rng(11)
    A = rng.standard_normal((5, 4))
    M_V = _rand_spd(rng, 4)
    M_W = _rand_spd(rng, 5)
    U, UI, V, VI, S, rank = GSvdOp()(A, M_V, M_W)

    Ufull = np.concatenate([U, UI], axis=1)
    Vfull = np.concatenate([V, VI], axis=1)
    # Orthonormal in the respective metrics.
    np.testing.assert_allclose(Ufull.T @ M_W @ Ufull, np.eye(5), atol=1e-9)
    np.testing.assert_allclose(Vfull.T @ M_V @ Vfull, np.eye(4), atol=1e-9)

    # Documented reconstruction identity: A = [U|UI] Sfull [V|VI]^T M_V.
    Sfull = np.zeros((5, 4))
    Sfull[: S.size, : S.size] = np.diag(S)
    np.testing.assert_allclose(Ufull @ Sfull @ Vfull.T @ M_V, A, atol=1e-9)


def test_gsvdop_block_split_known_rank() -> None:
    rng = np.random.default_rng(12)
    # rank-2 map V(4) -> W(5)
    A = rng.standard_normal((5, 2)) @ rng.standard_normal((2, 4))
    M_V = _rand_spd(rng, 4)
    M_W = _rand_spd(rng, 5)
    U, UI, V, VI, S, rank = GSvdOp()(A, M_V, M_W)

    assert int(rank) == 2
    assert U.shape == (5, 2) and UI.shape == (5, 3)
    assert V.shape == (4, 2) and VI.shape == (4, 2)

    # VI is the kernel: A · VI ≈ 0.
    np.testing.assert_allclose(A @ VI, 0.0, atol=1e-9)
    # UI is the coker: the metric adjoint A* = M_V^{-1} A^T M_W kills it.
    Astar = np.linalg.inv(M_V) @ A.T @ M_W
    np.testing.assert_allclose(Astar @ UI, 0.0, atol=1e-9)
    # Reduced reconstruction from the image / coimg blocks only.
    np.testing.assert_allclose(U @ np.diag(S[:2]) @ V.T @ M_V, A, atol=1e-9)


def test_gsvdop_in_program_multi_output() -> None:
    rng = np.random.default_rng(13)
    An = rng.standard_normal((5, 4))
    M_Vn = _rand_spd(rng, 4)
    M_Wn = _rand_spd(rng, 5)

    prog = Program("gsvd")
    A, va = _sym_matrix(prog, An, "ga")
    MV, vmv = _sym_matrix(prog, M_Vn, "gv")
    MW, vmw = _sym_matrix(prog, M_Wn, "gw")
    U, UI, V, VI, S, rank = prog.emit_stmt(
        GSvdOp(),
        [A, MV, MW],
        [
            OutSpec("U", (5, 4)),
            OutSpec("UI", (5, 1)),
            OutSpec("V", (4, 4)),
            OutSpec("VI", (4, 0)),
            OutSpec("S", (4,)),
            OutSpec("rank", ()),
        ],
        note="gsvd",
        bulk=True,
    )
    assert isinstance(prog.statements[-1].fn, GSvdOp)
    assert len(prog.statements[-1].out) == 6  # 6-output scatter
    prog.add_output("U", U)
    prog.add_output("V", V)
    prog.add_output("S", S)
    prog.add_output("rank", rank)
    res = prog.run({**va, **vmv, **vmw})

    assert int(round(float(res["rank"]))) == 4  # full rank 5x4
    np.testing.assert_allclose(res["U"].T @ M_Wn @ res["U"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(res["V"].T @ M_Vn @ res["V"], np.eye(4), atol=1e-9)


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
