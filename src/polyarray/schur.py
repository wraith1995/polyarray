"""``symbolic_inverse`` — the recursive block-triangular Schur inverse of a :class:`SymArray`.

The dense :meth:`SymArray.inverse` (cofactor ≤ ``naive_inverse_max_size``, else a numeric ``InvOp`` Stmt)
blows up on a *structurally sparse* symbolic matrix — a 6×6 whose off-diagonal block is structurally zero
still pays the 720-term cofactor determinant, and a larger one falls to a numeric Stmt instead of a
``RationalFunction`` inverse. This module exploits the sparsity: when the matrix is (after a row reordering)
block lower-triangular, the Schur recursion

    [A 0; C D]⁻¹ = [A⁻¹ 0; −D⁻¹·C·A⁻¹  D⁻¹]

keeps the symbolic entries small. It is the sparsity-aware sibling of :meth:`SymArray.inverse`, reusing
:func:`~polyarray.rational.cofactor_inverse` (adjugate / Bareiss det) as the ≤ :data:`BASE` base case and the
pre-reserved ``SymbolicBudget.schur_{inverse,matmul}_stmt_size`` knobs to defer the heavy leaf inverses /
Schur-combine products to numeric Stmts.

The general Schur formula subsumes the block-lower-triangular case (a structurally-zero top-right block ``B``
collapses ``S = D − C·A⁻¹·B`` to ``D`` and the top-right inverse block to ``0``), so ``B == 0`` is
special-cased only to *skip* the wasted symbolic work, not for correctness.

SymArray-native: blocks are SymArray slices, so the owning ``Program`` rides on the carrier — Stmt deferrals
emit into it, numerics mixed into symbolic arithmetic are coerced by ``RationalFunction`` itself, and a
float-cell (numeric) block short-circuits to numpy arithmetic.

Driver structure:
1. diagonal / ≤ ``BASE`` → direct (reciprocal / ``cofactor_inverse``);
2. else reorder rows by support (``_by_row_zeros``), choose the split maximizing the zero block
   (``_choose_split``), recurse on the two diagonal blocks, combine, and undo the reordering.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from .ir import (Cell, DimAtom, EinsumStmtOp, InvOp, OutSpec, Program, SymArray, SymInput,
                 current_budget_override, is_dynamic, probe_direct_eval)
from .rational import RationalFunction, cofactor_inverse, simple_zero


def _dim(x: int | DimAtom) -> int:
    """Narrow a matrix dimension to a static ``int``.

    ``SymArray.shape`` is typed ``int | DimAtom``, but this module only handles statically
    shaped square matrices — dynamic ranks are filtered upstream by ``is_dynamic`` — so the
    runtime value is always a plain ``int``.
    """
    return cast(int, x)


# Largest matrix we invert directly by cofactor expansion; above this we split via Schur.
BASE = 6

# Budget for DEFERRING the heavy symbolic arithmetic to a NUMERIC Stmt (evaluated at the concrete inputs)
# instead of inline RationalFunction blowup. The block split (driven by the exact mask) keeps the *structure*
# symbolic; the leaf inverses and Schur-combine matrix products are where the rational explodes. At/above
# these sizes we emit `np.linalg.inv` / `@` as a deferred Stmt; smaller blocks stay exact-rational. DEFAULTS:
# a caller dials exact-vs-fast per call via the ambient `SymbolicBudget` (see `_defer_thresholds`).
_DEFER_INVERSE = 2   # a base-case inverse this size or larger → numeric InvOp Stmt (only 1×1 stays inline)
_DEFER_MATMUL = 2    # a Schur-combine product with any dim ≥ this → numeric matmul Stmt


def _defer_thresholds() -> tuple[int, int]:
    """Return the ``(matmul_size, inverse_size)`` deferral thresholds.

    Taken from the ambient :class:`~polyarray.ir.SymbolicBudget` override
    (``schur_matmul_stmt_size`` / ``schur_inverse_stmt_size``) when set, else the module
    defaults. This is how a caller dials the Schur inverse per call:
    ``budget_override(SymbolicBudget.build_big_symbols())`` never defers, giving a fully
    symbolic exact inverse; ``force_stmts()`` always defers, giving the numeric extreme.
    """
    b = current_budget_override()
    mm = getattr(b, "schur_matmul_stmt_size", None) if b is not None else None
    inv = getattr(b, "schur_inverse_stmt_size", None) if b is not None else None
    return (_DEFER_MATMUL if mm is None else mm, _DEFER_INVERSE if inv is None else inv)


def _deferred_matmul(*arrs: SymArray) -> SymArray:
    """Multiply SymArray blocks left to right, deferring the large steps to numeric statements.

    A two-factor step whose largest dimension reaches the matmul threshold emits a numeric
    matmul statement into the carried program: the blocks evaluate at the concrete inputs and
    the result is fresh atom cells. Smaller steps stay inline exact-rational. Float-cell
    operands never defer, since :meth:`SymArray.matmul` short-circuits them to numpy.
    """
    result = arrs[0]
    mm_thresh = _defer_thresholds()[0]
    for nxt in arrs[1:]:
        program = result.program if result.program is not None else nxt.program
        rows, inner, cols = _dim(result.shape[0]), _dim(result.shape[1]), _dim(nxt.shape[1])
        big = max(rows, inner, cols) >= mm_thresh
        if program is not None and big and not (result.is_numeric and nxt.is_numeric):
            # A TYPED matmul op (2-D einsum `ij,jk->ik`), not an opaque ``lambda a, b: a @ b``:
            # only a typed op lowers through every backend — ``Program.run``, ``to_numpy_source``
            # and ``pyab``/torch alike. An opaque python callable raises at lowering time.
            (out,) = program.emit_stmt(
                EinsumStmtOp(spec="ij,jk->ik"),
                [result, nxt],
                [OutSpec("schur_mm", (rows, cols))],
                note="schur_matmul", bulk=False,
            )
            result = out
        else:
            result = result.matmul(nxt)
    return result


def _base_inverse(arr: SymArray) -> SymArray:
    """Invert a base-case block of size at most :data:`BASE`.

    Emits a numeric ``InvOp`` statement into the carried program when the block is symbolic and
    at least as large as the inverse-deferral threshold, where the symbolic cofactor determinant
    blows up. Smaller blocks take the exact :func:`~polyarray.rational.cofactor_inverse`, which
    itself short-circuits float cells.
    """
    n = _dim(arr.shape[0])
    program = arr.program
    if program is not None and not arr.is_numeric and n >= _defer_thresholds()[1]:
        # A TYPED ``InvOp`` — the same op :meth:`SymArray.inverse` defers to — rather than an
        # opaque ``lambda a: np.linalg.inv(a)``, so it lowers through Program.run,
        # to_numpy_source and pyab/torch alike. An opaque callable lowers through none of them.
        (out,) = program.emit_stmt(
            InvOp(),
            [arr],
            [OutSpec("schur_inv", (n, n))],
            note="schur_inverse", bulk=False,
        )
        return out
    return SymArray(cofactor_inverse(arr.cells), program=program)


# Structural-zero detection: number of DETERMINISTIC generic cells to probe a SymArray at, and the magnitude
# below which a probed cell counts as a structural zero. Tolerance (not exact 0) is needed because a
# normalized basis may be irrational, so a true zero lands as float roundoff.
_N_PROBES = 3
_MASK_TOL = 1e-9


def _zero_cells(rows: int, cols: int, symbolic: bool) -> np.ndarray:
    """Build a zero block in the matching lane.

    Ring-less constant :class:`RationalFunction` cells for the symbolic lane, which join rings
    on contact; plain float zeros for the numeric lane.
    """
    if symbolic:
        return np.full((rows, cols), RationalFunction.constant(0), dtype=object)
    return np.zeros((rows, cols))


# --- structural sparsity mask ----------------------------------------------------------------
# The cells of a symbolic ``C`` are messy ``RationalFunction``s that are *mathematically* zero/identity but
# do NOT syntactically simplify, so per-cell ``simple_zero`` cannot see the true block structure. We detect
# the structural nonzero pattern by probing ``C`` at a small set of DETERMINISTIC GENERIC inputs (no
# randomness): a structural zero is zero at every probe, while a coincidental zero (measure zero) does not
# survive the union. This drives the block-triangular split on the TRUE zeros. The cells stay messy —
# correctness comes from ``.evaluate(inputs)``; the mask only steers the recursion.


def _probe_binding(inputs: Sequence[SymInput], k: int) -> dict[str, np.ndarray]:
    """Build the ``k``-th deterministic generic binding for a program's inputs.

    Coordinates are irrational-spaced and distinct per probe and per slot, so the probed cells
    are generic rather than degenerate, and no RNG is involved.
    """
    binding: dict[str, np.ndarray] = {}
    for ii, inp in enumerate(inputs):
        n = int(np.prod(inp.shape))
        vec = np.array(
            [np.sqrt(2.0 + ((ii * 5 + j * 3 + k * 7) % 11)) + 0.3 * (k + 1) for j in range(n)]
        )
        binding[inp.name] = vec.reshape(inp.shape)
    return binding


def _structural_mask(matrix: SymArray) -> np.ndarray | None:
    """Compute a boolean nonzero mask by deterministic probing.

    Evaluates ``matrix`` at :data:`_N_PROBES` fixed generic inputs and ORs the patterns above
    :data:`_MASK_TOL`.

    Returns
    -------
    numpy.ndarray or None
        ``None`` when ``matrix`` carries no program, or has a dynamic or bulk-shaped input, so
        the caller falls back to the syntactic mask.
    """
    if matrix.program is None:
        return None
    inputs = matrix.program.inputs
    if any(is_dynamic(inp.shape) for inp in inputs):
        return None
    mask: np.ndarray | None = None
    # Direct term-sum evaluation (``probe_direct_eval`` in the program runner, ``compiled=False``
    # in the output-cell loop) is much cheaper than codegen for the handful of probe points, and
    # gives byte-identical values.
    with probe_direct_eval():
        for k in range(_N_PROBES):
            try:
                vals = np.abs(np.asarray(
                    matrix.evaluate(_probe_binding(inputs, k), compiled=False), dtype=float))
            except (KeyError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                return None
            m = np.isfinite(vals) & (vals > _MASK_TOL)
            mask = m if mask is None else (mask | m)
    return mask


def _approx_zero(cell: Cell) -> bool:
    """Test a cell for zero, accepting roundoff only where that is sound.

    A cell counts as zero when it is a syntactic (coefficient) zero, or when it is a constant —
    a number, or a total-degree-0 :class:`RationalFunction` — of magnitude below
    :data:`_MASK_TOL`. Inverting over an irrational normalized basis lands true zeros at
    roundoff scale rather than exactly zero, so the tolerance is load-bearing.

    Applying the tolerance only to constants is what keeps it sound: a constant does not vary,
    so a small magnitude really is a rounded zero. Reading a *symbolic* cell's magnitude at
    sample points instead would drop a tiny-but-nonzero cell and yield a wrong inverse, so a
    vertex-dependent cell stays exact.
    """
    if simple_zero(cell):
        return True
    if isinstance(cell, (int, float, np.floating, np.integer)):
        return abs(float(cell)) < _MASK_TOL
    if isinstance(cell, RationalFunction) and cell.is_constant():
        return abs(float(cell.to_constant())) < _MASK_TOL
    return False


def _syntactic_mask(cells: np.ndarray) -> np.ndarray:
    """Build the conservative mask: a cell is nonzero unless :func:`_approx_zero` proves it zero."""
    out = np.empty(cells.shape, dtype=bool)
    for idx in np.ndindex(cells.shape):
        out[idx] = not _approx_zero(cells[idx])
    return out


def _resolve_mask(matrix: SymArray, mask: np.ndarray | None) -> np.ndarray:
    """Resolve the nonzero mask steering the recursion. Sound by default.

    An explicitly threaded sub-mask wins. Otherwise the syntactic mask applies: it marks a cell
    zero only when its numerator polynomial is coefficient-zero, so it can never false-zero a
    truly non-zero cell. At worst it is conservative — dense where it cannot prove a cell zero
    — which only makes the block split less aggressive, never wrong.

    :func:`_structural_mask`, the numeric probe, is unsound and not the default: it marks a
    cell zero when it is merely small at a few sample points, so a tiny-but-nonzero cell is
    dropped and the Schur split silently returns a wrong inverse. Adding sample points does not
    help, because such cells are small everywhere. It is reachable only through the explicit,
    self-labelled ``POLYARRAY_SCHUR_UNSOUND_PROBE_MASK`` opt-in, for a large element whose
    cancellation sparsity a sound exact fold cannot recover, and only under an external
    correctness backstop.
    """
    if mask is not None:
        return mask
    if os.environ.get("POLYARRAY_SCHUR_UNSOUND_PROBE_MASK", "") not in ("", "0"):
        probed = _structural_mask(matrix)
        if probed is not None:
            import warnings
            warnings.warn(
                "schur: POLYARRAY_SCHUR_UNSOUND_PROBE_MASK is set — the numeric sparsity probe is UNSOUND "
                "(it drops tiny-but-nonzero cells, giving a WRONG inverse). Use "
                "only with the numeric-vs-symbolic P(T) backstop.", stacklevel=2)
            return probed
    return _syntactic_mask(matrix.cells)


def _all_zero(mask_block: np.ndarray) -> bool:
    """Report whether every entry of a mask block is structurally zero."""
    return not mask_block.any()


def _is_diagonal(mask: np.ndarray) -> bool:
    """Report whether the mask has no non-zero off-diagonal entry."""
    n = mask.shape[0]
    return not any(mask[i, j] for i in range(n) for j in range(n) if i != j)


def _diag_inverse(arr: SymArray) -> np.ndarray:
    """Invert a diagonal matrix by taking reciprocals down the diagonal."""
    cells = arr.cells
    n = cells.shape[0]
    out = _zero_cells(n, n, symbolic=not arr.is_numeric)
    for i in range(n):
        out[i, i] = 1.0 / cells[i, i]        # float, or RationalFunction.__rtruediv__
    return out


# --- block-lower-triangular detection --------------------------------------------------------


def _rightmost_nonzero(mask: np.ndarray) -> list[int]:
    """Per row, the index of the rightmost non-zero column (-1 if the row is all-zero)."""
    n = mask.shape[1]
    out = []
    for i in range(mask.shape[0]):
        r = -1
        for j in range(n - 1, -1, -1):
            if mask[i, j]:
                r = j
                break
        out.append(r)
    return out


def _by_row_zeros(arr: SymArray, mask: np.ndarray) -> tuple[list[int], list[int], SymArray, np.ndarray]:
    """Reorder rows ascending by their rightmost non-zero column.

    This clusters rows whose support is confined to early columns at the top, exposing
    block-lower-triangular structure. The mask is reordered in lockstep.

    Returns
    -------
    tuple
        ``(new_order, sizes, reordered_arr, reordered_mask)``, where ``sizes[i]`` is the
        rightmost non-zero column of the row now at position ``i``.
    """
    rightmost = _rightmost_nonzero(mask)
    new_order = sorted(range(len(rightmost)), key=lambda k: rightmost[k])
    sizes = [rightmost[k] for k in new_order]
    return new_order, sizes, arr[new_order], mask[new_order]


def _components(mask: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Find the connected components of the bipartite row-column graph, ``row i ~ col j`` iff ``mask[i, j]``.

    A matrix that is block-diagonal under a possibly asymmetric row/column permutation splits
    into one component per block, each inverting independently. This catches structure the
    block-*triangular* split cannot: a 6x6 of three disjoint 2x2 blocks is block-diagonal but
    not triangular.

    Returns
    -------
    list of (list of int, list of int)
        ``(rows, cols)`` index lists per component. For a nonsingular matrix every component is
        square.
    """
    n = mask.shape[0]
    parent = list(range(2 * n))                                  # rows 0..n-1, cols n..2n-1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                parent[find(i)] = find(n + j)
    from collections import defaultdict

    rows_of: dict[int, list[int]] = defaultdict(list)
    cols_of: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        rows_of[find(i)].append(i)
    for j in range(n):
        cols_of[find(n + j)].append(j)
    return [(rows_of[r], cols_of.get(r, [])) for r in rows_of]


def _choose_split(n: int, sizes: list[int]) -> int | None:
    """Choose the split ``p`` whose top-right ``p x (n-p)`` block is structurally zero.

    Valid when the first ``p`` rows' support stays within the first ``p`` columns, i.e.
    ``sizes[p - 1] < p`` under the ascending sort. Among valid splits the most balanced one
    wins.

    Returns
    -------
    int or None
        The chosen split, or ``None`` if no zero-block split exists.
    """
    valid = [p for p in range(1, n) if sizes[p - 1] < p]
    if not valid:
        return None
    return min(valid, key=lambda p: abs(2 * p - n))


# --- the recursion ---------------------------------------------------------------------------


def _schur_combine(M: SymArray, k: int, mask: np.ndarray) -> SymArray:
    """Invert ``M`` split at ``k`` via Schur, recursing on the diagonal blocks.

    Covers both the block-lower-triangular fast path, where the mask says the top-right block
    is zero, and the general case. The mask is sliced in lockstep with ``M`` so the
    diagonal-block recursions see the true sparsity; the Schur complement is a fresh arithmetic
    product with no probed mask of its own, so it takes the syntactic one. Matrix products go
    through :func:`_deferred_matmul`, so a large combine defers to a numeric statement.
    """
    n = _dim(M.shape[0])
    symbolic = not M.is_numeric
    A, B, C, D = M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]
    mA, mB, mD = mask[:k, :k], mask[:k, k:], mask[k:, k:]
    A_inv = _invert(A, mask=mA)
    out = np.empty((n, n), dtype=object) if symbolic else np.empty((n, n))
    if _all_zero(mB):
        D_inv = _invert(D, mask=mD)
        out[:k, :k] = A_inv.cells
        out[:k, k:] = _zero_cells(k, n - k, symbolic)
        out[k:, :k] = (-_deferred_matmul(D_inv, C, A_inv)).cells
        out[k:, k:] = D_inv.cells
    else:
        S = D - _deferred_matmul(C, A_inv, B)
        S_inv = _invert(S, mask=_syntactic_mask(S.cells))
        AB_Sinv = _deferred_matmul(A_inv, B, S_inv)
        CA = _deferred_matmul(C, A_inv)
        out[:k, :k] = (A_inv + _deferred_matmul(AB_Sinv, CA)).cells
        out[:k, k:] = (-AB_Sinv).cells
        out[k:, :k] = (-_deferred_matmul(S_inv, CA)).cells
        out[k:, k:] = S_inv.cells
    return SymArray(out, program=M.program)


def symbolic_inverse(matrix: SymArray | np.ndarray, *, mask: np.ndarray | None = None,
                     program: Program | None = None) -> SymArray:
    """Invert a square matrix via the block-triangular Schur recursion.

    The sparsity-aware sibling of :meth:`SymArray.inverse`. :class:`RationalFunction`
    arithmetic coerces numerics and joins rings on contact, so symbolic and numeric blocks mix
    freely.

    Parameters
    ----------
    matrix
        The matrix to invert, as a :class:`SymArray` or a raw cell ndarray.
    mask
        Boolean nonzero mask steering the block split. Pass it when the caller can compute the
        sparsity cheaply and exactly; when omitted it is resolved from ``matrix`` by
        deterministic probing, or by the syntactic fallback. A denser mask only makes the split
        less aggressive, never wrong.
    program
        The shared program the inverse should be grounded onto, so the emitted statements lower
        through a value kernel compiled from it rather than being stranded on a by-product
        program. When ``matrix`` already rides ``program``, or ``matrix`` is numeric or
        program-less, this is a no-op and statements emit into ``matrix``'s own program.

        Otherwise the block-split mask is resolved on ``matrix``'s own program **first** — its
        inputs are the clean generic-cell generators the deterministic probe needs — and only
        then is ``matrix`` grafted onto ``program``. A graft rather than a relabel is
        load-bearing: ``matrix``'s cells may reference its program's own producing statements,
        not just shared input atoms, and a relabel would strand those. The graft emits
        ``matrix``'s program as a sub-program statement with deduplicated outputs, so several
        inverses grounded on one shared program do not collide.

    Returns
    -------
    SymArray
        The inverse, carrying the owning program so downstream simplify, sparsity and codegen
        passes apply to it.
    """
    M = matrix if isinstance(matrix, SymArray) else SymArray(matrix)
    if program is not None and M.program is not None and M.program is not program:
        mask = _resolve_mask(M, mask)                       # probe on M's OWN (clean-input) program first
        M = program.graft(M)                                # bring M's producing Stmts onto `program`
    return _invert(M, mask=mask)


def _invert(M: SymArray, mask: np.ndarray | None = None) -> SymArray:
    """Run the recursion proper on the SymArray carrier.

    ``mask`` steers the block-triangular split, and is resolved from ``M`` when omitted.
    Sub-recursions thread the corresponding sub-mask so the sparsity stays aligned through the
    row reordering and block splits.
    """
    mask = _resolve_mask(M, mask)
    n = _dim(M.shape[0])
    if n == 0:
        return M
    if n == 1 or _is_diagonal(mask):
        return SymArray(_diag_inverse(M), program=M.program)

    # Block-DIAGONAL split (disjoint bipartite components) BEFORE the triangular split: a matrix block-
    # diagonal under a row/col permutation is not block-triangular, so `_choose_split` misses it and it would
    # fall to a dense cofactor determinant. Invert each component independently: the inverse of the block at
    # (rows, cols) lands at (cols, rows).
    comps = _components(mask)
    if len(comps) > 1 and all(len(r) == len(c) for r, c in comps):
        out = _zero_cells(n, n, symbolic=not M.is_numeric)
        for rows, cols in comps:
            sub = _invert(M[np.ix_(rows, cols)], mask=mask[np.ix_(rows, cols)])
            out[np.ix_(cols, rows)] = sub.cells
        return SymArray(out, program=M.program)

    # Try the block-triangular split FIRST (the mask exposes the true zeros): an n×n with a structurally-zero
    # off-diagonal block must be Schur-split even at n ≤ BASE, else the dense cofactor blows up despite the
    # trivial sparsity.
    new_order, sizes, reordered, reordered_mask = _by_row_zeros(M, mask)
    split = _choose_split(n, sizes)
    if split is not None:
        # Block-lower-triangular split (top-right block structurally zero): the diagonal blocks A, D are
        # non-singular (det M = det A · det D ≠ 0 with B = 0), so no pivot is needed. `reordered = M[new_
        # order]` (row perm P) ⇒ reordered⁻¹ = M⁻¹Pᵀ, so M⁻¹ = a column permutation of the result.
        rinv = _schur_combine(reordered, split, reordered_mask)
        return rinv[:, np.argsort(new_order)]
    # No beneficial block structure REMAINS here (no triangular split, single bipartite component): the
    # block is effectively DENSE, so recursing a general midpoint split would only churn symbolic
    # RationalFunction arithmetic on a dense high-degree block — the degree-5 blow-up.
    # Instead DEFER THE WHOLE BLOCK (any size) to a NUMERIC `InvOp` Stmt that evaluates per cell (§14:
    # stop recursing once the block zeros stop being useful, and defer everywhere). This also sidesteps the
    # un-pivoted general-Schur singular-pivot bug — `np.linalg.inv` pivots internally. `_base_inverse` emits
    # the numeric InvOp for a symbolic block riding a program (`_DEFER_INVERSE`), else the exact cofactor.
    return _base_inverse(M)
