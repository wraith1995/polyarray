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
    # inverse is WRONG even though M is invertible. Regression: the plate-element P(T) hit this
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


def _sparse_block_lower(name: str) -> SymArray:
    """A 6×6 program-carrying ``[[A, 0], [C, D]]`` with A, D DIAGONAL and C carrying 2 nonzeros per row.

    The structural truth of its inverse is ``[[A⁻¹, 0], [−D⁻¹·C·A⁻¹, D⁻¹]]`` — and with A, D diagonal the
    product has C's OWN pattern, so 3 + 6 + 3 = 12 nonzeros, never 3 + 9 + 3 = 15.

    Parameters
    ----------
    name
        Name of the carried program.

    Returns
    -------
    SymArray
        The block-lower-triangular matrix, riding a program with one ``vertex`` feed input.
    """
    from polyarray import Program, Provenance, SymInput
    prog = Program(name, inputs=[SymInput("v", (12,), Provenance("vertex", "v", (), "v"))])
    v = prog.input("v")
    zero = RF.constant(0.0)
    cells = np.empty((6, 6), dtype=object)
    cells[:] = zero
    for i in range(6):
        cells[i, i] = v.cells[i] + RF.constant(8.0)          # diagonally dominant A and D
    c_cols = {3: (0, 1), 4: (0, 2), 5: (1, 2)}               # 2 nonzeros per row of C
    for i, cols in c_cols.items():
        for j in cols:
            cells[i, j] = v.cells[6 + i - 3 + 3 * (j % 2)] + RF.constant(1.0)
    return SymArray(cells, program=prog)


def test_schur_combine_keeps_the_products_structural_zeros() -> None:
    """The Schur combine's ``−D⁻¹·C·A⁻¹`` must come back with C's SPARSITY, not dense.

    That product is DEFERRED to a numeric matmul Stmt whose outputs are fresh atoms — NAMES for a
    pending computation — so without the factor masks threaded through ``_deferred_matmul`` every
    entry reads as nonzero and the inverse silently loses exactly the sparsity the split was chosen
    for: one spurious nonzero per row, 15 against a numeric truth of 12 here.  Both halves are
    asserted: the STRUCTURE, and that the structure is not bought with a wrong value."""
    from polyarray.schur import _approx_zero
    M = _sparse_block_lower("schur_sparse6")
    inv = symbolic_inverse(M)
    cells = np.asarray(inv.cells)
    nnz = sum(not _approx_zero(cells[idx]) for idx in np.ndindex(cells.shape))
    assert nnz == 12, f"expected the 12-nonzero structure of [[A⁻¹,0],[−D⁻¹CA⁻¹,D⁻¹]], got {nnz}"
    rng = np.sqrt(np.arange(2, 14, dtype=float)) + 0.17
    Mn = np.asarray(M.evaluate({"v": rng}), float)
    got = np.asarray(inv.evaluate({"v": rng}), float)
    assert np.allclose(got @ Mn, np.eye(6), atol=1e-9)


def _dense_block(name: str, n: int) -> SymArray:
    """A program-carrying DENSE ``n×n`` symbolic block — no structural zero anywhere, so ``_invert``
    exhausts both splits and falls through to ``_base_inverse``.

    Parameters
    ----------
    name
        Name of the carried program.
    n
        The block size.

    Returns
    -------
    SymArray
        The dense block, riding a program with one ``vertex`` feed input.
    """
    prog = Program(name, inputs=[SymInput("v", (n * n,), Provenance("vertex", "v", (), "v"))])
    v = prog.input("v")
    cells = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            cells[i, j] = v.cells[i * n + j] + RF.constant(float(4 * n) if i == j else 1.0)
    return SymArray(cells, program=prog)


@pytest.mark.parametrize("n,defers", [(2, False), (3, False), (4, True)])
def test_small_dense_blocks_invert_inline_large_ones_defer(n: int, defers: bool) -> None:
    """``_DEFER_INVERSE = 4``: a dense symbolic block of 2 or 3 is inverted EXACTLY (closed-form
    ``cofactor_inverse``, cells stay ``RationalFunction``s), 4 and above defer to a numeric ``InvOp``.

    Why the boundary sits there: a deferred inverse's output cells are FRESH ATOMS, opaque to the exact
    fold and to ``_approx_zero``, so an inverse built over one is not closed-form. ≤3×3 buys the whole
    thing back for a bounded cost — no ``InvOp`` left at all, monomial mass up by a tenth to a half,
    sparsity and wall time unchanged. Both halves are asserted — the MECHANISM (did an ``InvOp`` get
    emitted?) and that it still inverts."""
    from polyarray import InvOp
    M = _dense_block(f"dense{n}", n)
    inv = symbolic_inverse(M)
    emitted = [st for st in M.program.statements if isinstance(st.fn, InvOp)]
    assert bool(emitted) is defers, f"{n}×{n}: InvOp emitted={bool(emitted)}, expected {defers}"
    if not defers:                                   # inline ⇒ exact rational cells, nothing opaque
        assert all(isinstance(c, RF) for c in np.asarray(inv.cells).reshape(-1))
    rng = np.sqrt(np.arange(2, 2 + n * n, dtype=float)) + 0.23
    Mn = np.asarray(M.evaluate({"v": rng}), float)
    got = np.asarray(inv.evaluate({"v": rng}), float)
    assert np.allclose(got @ Mn, np.eye(n), atol=1e-9)
