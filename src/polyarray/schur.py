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
float-cell (numeric) block short-circuits to numpy arithmetic. (Ported from oracle ``vandermonde/schur.py`` —
pure SymArray algebra, so its home is polyarray; oracle's FEEC driver can consume this.)

Driver structure:
1. diagonal / ≤ ``BASE`` → direct (reciprocal / ``cofactor_inverse``);
2. else reorder rows by support (``_by_row_zeros``), choose the split maximizing the zero block
   (``_choose_split``), recurse on the two diagonal blocks, combine, and undo the reordering.
"""

from typing import cast

import numpy as np

from .ir import (EinsumStmtOp, InvOp, OutSpec, Program, SymArray, current_budget_override,
                 is_dynamic, probe_direct_eval)
from .rational import RationalFunction, cofactor_inverse, simple_zero


def _dim(x: object) -> int:
    """A static matrix dimension as an ``int``. ``SymArray.shape`` is typed ``int | DimAtom``, but this
    routine requires statically-shaped square matrices (dynamic ``DimAtom`` ranks are filtered upstream by
    ``is_dynamic``), so the runtime value is always a plain ``int``."""
    return cast(int, x)


# Largest matrix we invert directly by cofactor expansion; above this we split via Schur.
BASE = 6

# Budget for DEFERRING the heavy symbolic arithmetic to a NUMERIC Stmt (evaluated at the concrete inputs)
# instead of inline RationalFunction blowup. The block split (driven by the exact mask) keeps the *structure*
# symbolic; the leaf inverses and Schur-combine matrix products are where the rational explodes. At/above
# these sizes we emit `np.linalg.inv` / `@` as a deferred Stmt; smaller blocks stay exact-rational. DEFAULTS:
# a caller dials exact-vs-fast per call via the ambient `SymbolicBudget` (see `_defer_thresholds`).
_DEFER_INVERSE = 4   # a base-case inverse this size or larger → numeric InvOp Stmt (2×2/3×3 stay rational)
_DEFER_MATMUL = 4    # a Schur-combine product with any dim ≥ this → numeric matmul Stmt


def _defer_thresholds() -> tuple[int, int]:
    """``(matmul_size, inverse_size)`` deferral thresholds — from the ambient ``SymbolicBudget`` override
    (``schur_matmul_stmt_size`` / ``schur_inverse_stmt_size``) when set, else the module defaults. Lets a
    caller dial the Schur inverse per call: ``budget_override(SymbolicBudget.build_big_symbols())`` ⇒ never
    defer (fully symbolic exact inverse); ``... force_stmts()`` ⇒ always defer (max numeric)."""
    b = current_budget_override()
    mm = getattr(b, "schur_matmul_stmt_size", None) if b is not None else None
    inv = getattr(b, "schur_inverse_stmt_size", None) if b is not None else None
    return (_DEFER_MATMUL if mm is None else mm, _DEFER_INVERSE if inv is None else inv)


def _deferred_matmul(*arrs: SymArray) -> SymArray:
    """Left-to-right product of SymArray blocks. Each 2-factor step emits a numeric matmul Stmt into the
    carried program when an operand is large (`_DEFER_MATMUL`) — the blocks evaluate numerically at the
    concrete inputs and the result is fresh atom cells — else inline exact-rational matmul. Float-cell
    (numeric) operands never defer: `SymArray.matmul` short-circuits them to numpy."""
    result = arrs[0]
    mm_thresh = _defer_thresholds()[0]
    for nxt in arrs[1:]:
        program = result.program if result.program is not None else nxt.program
        rows, inner, cols = _dim(result.shape[0]), _dim(result.shape[1]), _dim(nxt.shape[1])
        big = max(rows, inner, cols) >= mm_thresh
        if program is not None and big and not (result.is_numeric and nxt.is_numeric):
            # A TYPED matmul op (2-D einsum `ij,jk->ik`), NOT an opaque ``lambda a, b: a @ b``: the typed op
            # lowers through EVERY backend — ``Program.run`` (numeric), ``to_numpy_source``, AND
            # ``pyab``/torch (a grounded-symbolic ``P(T)`` compiled into savo's vmapped value kernel) — where
            # an opaque python callable raises "no lowering for op 'function'".
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
    """The ≤BASE base-case inverse: emit a numeric `np.linalg.inv` Stmt into the carried program when the
    block is symbolic and large enough that the symbolic cofactor determinant blows up (`_DEFER_INVERSE`);
    else exact `cofactor_inverse` (which itself short-circuits float cells)."""
    n = _dim(arr.shape[0])
    program = arr.program
    if program is not None and not arr.is_numeric and n >= _defer_thresholds()[1]:
        # A TYPED ``InvOp`` (the SAME op :meth:`SymArray.inverse` defers to), NOT an opaque
        # ``lambda a: np.linalg.inv(a)``: lowers through Program.run / to_numpy_source / pyab-torch alike,
        # so a grounded-symbolic ``P(T)`` compiles into savo's value kernel (an opaque callable cannot).
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
    """A zero block in the matching lane: ring-less constant ``RationalFunction`` cells for the symbolic
    lane (``RationalFunction`` joins rings on contact), float zeros for the numeric."""
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


def _probe_binding(inputs, k: int) -> dict:
    """The ``k``-th DETERMINISTIC generic binding for the program ``inputs`` — irrational-spaced coordinates,
    distinct per probe and per slot, so the probed cells are generic (non-degenerate) and no RNG is
    involved."""
    binding: dict = {}
    for ii, inp in enumerate(inputs):
        n = int(np.prod(inp.shape))
        vec = np.array(
            [np.sqrt(2.0 + ((ii * 5 + j * 3 + k * 7) % 11)) + 0.3 * (k + 1) for j in range(n)]
        )
        binding[inp.name] = vec.reshape(inp.shape)
    return binding


def _structural_mask(matrix: SymArray) -> np.ndarray | None:
    """Boolean nonzero mask for a ``SymArray`` carrying a program, by DETERMINISTIC probing.

    Evaluates ``matrix`` at ``_N_PROBES`` fixed generic inputs (no RNG) and ORs the ``> _MASK_TOL`` patterns.
    Returns ``None`` when ``matrix`` carries no program (or a dynamic/bulk-shaped input) so the caller falls
    back to the syntactic ``simple_zero`` mask."""
    if matrix.program is None:
        return None
    inputs = matrix.program.inputs
    if any(is_dynamic(inp.shape) for inp in inputs):
        return None
    mask: np.ndarray | None = None
    # A 3-point probe runs the C program only 3× — no-``compile`` RF evaluation (direct term
    # sum, both in the program runner via ``probe_direct_eval`` and in the output-cell loop
    # via ``compiled=False``) is ~31 s cheaper than codegen on a degree-5 C (the Argyris/Bell
    # floor), and byte-identical, so the probed mask is unchanged.
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


def _syntactic_mask(cells: np.ndarray) -> np.ndarray:
    """Conservative fallback mask: a cell is nonzero unless it is *syntactically* zero."""
    out = np.empty(cells.shape, dtype=bool)
    for idx in np.ndindex(cells.shape):
        out[idx] = not simple_zero(cells[idx])
    return out


def _resolve_mask(matrix: SymArray, mask: np.ndarray | None) -> np.ndarray:
    """The nonzero mask steering the recursion: an explicitly-threaded sub-mask if given, else a
    deterministically-probed structural mask of ``matrix``, else the syntactic ``simple_zero`` fallback."""
    if mask is not None:
        return mask
    probed = _structural_mask(matrix)
    if probed is not None:
        return probed
    return _syntactic_mask(matrix.cells)


def _all_zero(mask_block: np.ndarray) -> bool:
    return not mask_block.any()


def _is_diagonal(mask: np.ndarray) -> bool:
    n = mask.shape[0]
    return not any(mask[i, j] for i in range(n) for j in range(n) if i != j)


def _diag_inverse(arr: SymArray) -> np.ndarray:
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
    """Reorder rows ascending by their rightmost non-zero column — clustering rows whose support is confined
    to early columns at the top (exposing block-lower-triangular structure). The mask is reordered in
    lockstep. Returns ``(new_order, sizes, reordered_arr, reordered_mask)``."""
    rightmost = _rightmost_nonzero(mask)
    new_order = sorted(range(len(rightmost)), key=lambda k: rightmost[k])
    sizes = [rightmost[k] for k in new_order]
    return new_order, sizes, arr[new_order], mask[new_order]


def _components(mask: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Connected components of the bipartite row↔column graph (``row i ~ col j`` iff ``mask[i,j]``).

    A matrix block-DIAGONAL under a (possibly asymmetric) row/column permutation splits into one component
    per block, each inverting independently — catching structure the block-*triangular* split cannot (e.g. a
    6×6 that is three disjoint 2×2 blocks, block-diagonal but not triangular). Returns ``[(rows, cols), …]``
    with sorted index lists; for a nonsingular matrix every component is square."""
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
    """The split ``p`` (square blocks ``p`` and ``n−p``) whose top-right ``p×(n−p)`` block is structurally
    zero — i.e. the first ``p`` rows' support stays within the first ``p`` columns (``sizes[p-1] < p``, using
    the ascending sort). Among valid splits pick the most balanced (closest to ``n/2``). ``None`` if no
    zero-block split exists."""
    valid = [p for p in range(1, n) if sizes[p - 1] < p]
    if not valid:
        return None
    return min(valid, key=lambda p: abs(2 * p - n))


# --- the recursion ---------------------------------------------------------------------------


def _schur_combine(M: SymArray, k: int, mask: np.ndarray) -> SymArray:
    """Invert ``M`` split at ``k`` via Schur, recursing on the diagonal blocks. Handles the block-lower-
    triangular fast path (``B == 0`` per the mask) and the general case. The mask is sliced in lockstep with
    ``M`` so the diagonal-block recursions see the true sparsity; the Schur complement ``S`` is a fresh
    arithmetic product with no sampled mask, so it gets the syntactic ``simple_zero`` mask. Matrix products
    go through ``_deferred_matmul`` so a large combine defers to a numeric Stmt instead of blowing up."""
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
                     program: object = None) -> SymArray:
    """Invert a (square) matrix via the block-triangular Schur recursion — the sparsity-aware sibling of
    :meth:`SymArray.inverse`.

    Accepts a ``SymArray`` or a raw cell ndarray; returns a ``SymArray`` of ``RationalFunction``/numeric
    cells, propagating the input's owning ``Program`` so downstream simplify/sparsity/codegen passes apply
    and Stmt deferrals emit into it. ``RationalFunction`` arithmetic coerces numerics and joins rings on
    contact.

    ``mask`` is the boolean nonzero mask steering the block split. Pass it when the caller can compute the
    sparsity cheaply/exactly; when omitted it is resolved from ``matrix`` by DETERMINISTIC probing (a
    program-carrying SymArray) or the syntactic ``simple_zero`` fallback. A conservative (denser) mask only
    makes the split less aggressive, never wrong.

    ``program`` (GROUNDED SYMBOLIC): the SHARED ``Program`` the block-triangular inverse should be GROUNDED
    onto — so a symbolic ``P(T)`` lowers through a value kernel compiled from that shared program (savo's
    block program) rather than leaving Stmts stranded on an ephemeral by-product program. When ``matrix``
    already rides ``program`` (or ``program`` is ``None``, or ``matrix`` is numeric / program-less) this is a
    no-op and the Stmts emit into ``matrix``'s own program as before (backward compatible). Otherwise the
    block-split mask is resolved on ``matrix``'s OWN program FIRST — its inputs are the clean generic-cell
    generators the deterministic structural probe needs — and only then is ``matrix`` GRAFTED onto
    ``program`` (:meth:`Program.graft`): ``matrix``'s cells may reference not only shared input atoms but
    also *its program's own producing Stmts* (e.g. the world-Vandermonde's grassmann-lowered derivative-DOF
    ``grass_dof`` Stmts), so a bare relabel would strand those; the graft emits ``matrix``'s program as a
    sub-Program Stmt of ``program`` (fresh dedup'd outputs). The recursion's own deferred leaf-inverse /
    Schur-combine Stmts (``schur_inverse``/``schur_matmul``) then emit natively onto ``program`` (mask
    threaded so it never re-probes on the shared program), and several elements' ``P(T)``s grounded on one
    shared program do not collide."""
    M = matrix if isinstance(matrix, SymArray) else SymArray(matrix)
    if program is not None and M.program is not None and M.program is not program:
        mask = _resolve_mask(M, mask)                       # probe on M's OWN (clean-input) program first
        M = cast(Program, program).graft(M)                 # bring M's producing Stmts onto `program`
    return _invert(M, mask=mask)


def _invert(M: SymArray, mask: np.ndarray | None = None) -> SymArray:
    """The recursion proper, on the SymArray carrier. ``mask`` steers the block-triangular split; when
    omitted it is resolved from ``M``. Sub-recursions thread the corresponding sub-mask so the sparsity
    stays aligned through the row reordering and block splits."""
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
    if split is None:
        if n <= BASE:
            return _base_inverse(M)
        # No beneficial block structure — split in place at the midpoint (general Schur).
        reordered, reordered_mask, new_order, split = M, mask, list(range(n)), n // 2

    rinv = _schur_combine(reordered, split, reordered_mask)
    # reordered = M[new_order] (row permutation P); reordered⁻¹ = M⁻¹ Pᵀ, so M⁻¹ = reordered⁻¹ P = a COLUMN
    # permutation of rinv by the inverse order.
    inv_order = np.argsort(new_order)
    return rinv[:, inv_order]
