"""``symbolic_inverse`` — the block-triangular Schur inverse over ``SymArray`` (sparsity-aware sibling of
``SymArray.inverse``). Correctness = evaluates to ``np.linalg.inv`` on concrete inputs, across the numeric,
syntactic-mask (no program), and probed-mask (program-carrying) lanes and above the dense ``BASE`` cutoff.
"""
import numpy as np
import pytest
from polyarray import Program, Provenance, SymArray, SymInput, symbolic_inverse
from polyarray.rational import RationalFunction as RF


def _num(cells, sub=None):
    out = np.empty(cells.shape, float)
    for idx in np.ndindex(cells.shape):
        c = cells[idx]
        out[idx] = float(c.eval(sub)) if hasattr(c, "eval") else float(c)
    return out


def test_numeric_matches_linalg_inv():
    A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [4.0, 5.0, 2.0]])
    inv = symbolic_inverse(A)
    assert isinstance(inv, SymArray)
    assert np.allclose(np.asarray(inv.cells, float), np.linalg.inv(A))


def test_symbolic_block_triangular_no_program():
    # [[x+1, 0], [x, 2]] — syntactic mask (no program); Schur must see the structural zero top-right.
    x, one, zero = RF.atom("x"), RF.constant(1.0), RF.constant(0.0)
    M = SymArray(np.array([[x + one, zero], [x, one + one]], dtype=object))
    inv = symbolic_inverse(M)
    got = _num(inv.cells, {"x": 3.0})
    assert np.allclose(got, np.linalg.inv([[4.0, 0.0], [3.0, 2.0]]))


def _block_lower(prog_name, n, kzero):
    """A program-carrying ``n×n`` SymArray whose top-right ``kzero×(n−kzero)`` block is structurally zero
    (entries = distinct rational functions of a single input vector so the probe mask is meaningful)."""
    prog = Program(prog_name, inputs=[SymInput("v", (n * n,), Provenance("vertex", "v", (), "v"))])
    v = prog.input("v")                                     # (n*n,) SymArray of atoms
    zero, diag = RF.constant(0.0), RF.constant(float(4 * n))
    cells = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i < kzero and j >= kzero:
                cells[i, j] = zero                          # the structurally-zero top-right block
            elif i == j:
                cells[i, j] = v.cells[i * n + j] + diag     # diagonally dominant → well-conditioned blocks
            else:
                cells[i, j] = v.cells[i * n + j]
    return prog, SymArray(cells, program=v.program)


def test_program_carrying_probe_mask_small():
    # 4×4, top-right 2×2 structurally zero → the deterministic probe mask must find the block-triangular
    # split; inverse evaluates to np.linalg.inv on a concrete cell.
    prog, M = _block_lower("blk4", 4, 2)
    inv = symbolic_inverse(M)
    rng = np.arange(1, 17, dtype=float) * 0.5 + 0.3
    Mn = _num(M.cells, None) if M.is_numeric else np.asarray(M.evaluate({"v": rng}), float)
    got = np.asarray(inv.evaluate({"v": rng}), float)
    assert np.allclose(got @ Mn, np.eye(4), atol=1e-9)


def test_block_triangular_above_base_cutoff():
    # 8×8 (> BASE=6) with a zero top-right 4×4 → must Schur-split rather than fall to a numeric InvOp;
    # correctness on evaluation is the gate.
    prog, M = _block_lower("blk8", 8, 4)
    inv = symbolic_inverse(M)
    rng = np.sqrt(np.arange(2, 66, dtype=float)) + 0.11
    Mn = np.asarray(M.evaluate({"v": rng}), float)
    got = np.asarray(inv.evaluate({"v": rng}), float)
    assert np.allclose(got @ Mn, np.eye(8), atol=1e-8)


def test_explicit_mask_is_honored():
    # A caller-supplied mask steers the split; a fully-dense (conservative) mask must still be correct.
    prog, M = _block_lower("blkmask", 4, 2)
    dense = np.ones((4, 4), dtype=bool)
    inv = symbolic_inverse(M, mask=dense)
    rng = np.arange(1, 17, dtype=float) * 0.5 + 0.3
    Mn = np.asarray(M.evaluate({"v": rng}), float)
    assert np.allclose(np.asarray(inv.evaluate({"v": rng}), float) @ Mn, np.eye(4), atol=1e-9)


def test_general_split_pivots_singular_midpoint_block():
    # An 8×8 (> BASE) block ANTI-diagonal `[[0, X],[Y, 0]]` — non-singular (det = ±detX·detY) but whose
    # MIDPOINT principal block A (top-left 4×4) is ZERO ⇒ singular. A DENSE mask forces it onto the general
    # Schur split, which must PIVOT columns so A is non-singular; without pivoting A⁻¹ blows up and the
    # inverse is WRONG even though M is invertible. Regression (2026-07-24): the plate-element P(T) hit this
    # via a conservative sparse mask (Hermite symbolic P(T) → 2.4e30 off inv(C)).
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 2.0, (4, 4)) + 5.0 * np.eye(4)
    Y = rng.uniform(1.0, 2.0, (4, 4)) + 5.0 * np.eye(4)
    M = np.zeros((8, 8))
    M[:4, 4:] = X
    M[4:, :4] = Y
    for mask in (np.ones((8, 8), dtype=bool), None):   # dense (conservative) AND resolved — both correct
        inv = symbolic_inverse(SymArray(M), mask=mask)
        got = np.asarray(inv.cells, dtype=float)
        assert np.allclose(got @ M, np.eye(8), atol=1e-10), f"mask_dense={mask is not None}"
