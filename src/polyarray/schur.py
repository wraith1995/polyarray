"""``symbolic_inverse`` — the recursive block-triangular Schur inverse of a :class:`SymArray`.

The dense :meth:`SymArray.inverse` (cofactor ≤ ``naive_inverse_max_size``, else a numeric ``InvOp`` Stmt)
blows up on a *structurally sparse* symbolic matrix — a 6×6 whose off-diagonal block is structurally zero
still pays the 720-term cofactor determinant, and a larger one falls to a numeric Stmt instead of a
``RationalFunction`` inverse. This module exploits the sparsity: when the matrix is (after a row reordering)
block lower-triangular, the Schur recursion::

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

import os
from typing import cast

from collections.abc import Sequence

import numpy as np

from .ir import (Cell, DimAtom, EinsumStmtOp, InvOp, OutSpec, Program, SymArray, SymInput,
                 current_budget_override, is_dynamic, probe_direct_eval)
from .rational import RationalFunction, cofactor_inverse, simple_zero


def _dim(x: int | DimAtom) -> int:
    """Narrow a matrix dimension to a static ``int``.

    ``SymArray.shape`` is typed ``int | DimAtom``, but this routine requires statically-shaped square matrices
    (dynamic ``DimAtom`` ranks are filtered upstream by ``is_dynamic``), so the runtime value is always a
    plain ``int``.
    """
    return cast(int, x)


# Largest matrix we invert directly by cofactor expansion; above this we split via Schur.
BASE = 6

# Budget for DEFERRING the heavy symbolic arithmetic to a NUMERIC Stmt (evaluated at the concrete inputs)
# instead of inline RationalFunction blowup. The block split (driven by the exact mask) keeps the *structure*
# symbolic; the leaf inverses and Schur-combine matrix products are where the rational explodes. At/above
# these sizes we emit `np.linalg.inv` / `@` as a deferred Stmt; smaller blocks stay exact-rational. DEFAULTS:
# a caller dials exact-vs-fast per call via the ambient `SymbolicBudget` (see `_defer_thresholds`).
#
# ⚠ A DEFERRED INVERSE IS OPAQUE. A Stmt's output cells are FRESH ATOMS — names for a pending numeric
# computation, carrying no arithmetic — so every `InvOp` the recursion emits is a hole the exact fold
# cannot see through, and an inverse built over one is only partly closed-form. Deferring is therefore
# a COST, paid to avoid a worse one (the cofactor determinant's degree growth: det ~ n·d, cofactors
# ~ (n−1)·d). The threshold is where that trade turns:
#
#   ≤ 3×3 is free where it matters. A matrix that is structurally diagonal never reaches
#   `_base_inverse` at all — `_invert` takes the `_is_diagonal` arm — so the threshold is inert for it
#   at any setting, and build time, monomial mass, degree and nnz are identical at 2 and at 4. The lane
#   the threshold actually governs is the one whose blocks are still dense when the splits run out, and
#   there the recursion bottoms out at blocks of size 2 and 3. Inverting those inline leaves NO `InvOp`
#   in the result, for a small bounded symbolic cost: monomial mass up by roughly a tenth to a half,
#   max cell degree 1 → 3, wall time and sparsity unchanged.
#
# So: 4 = "invert a dense block of 2 or 3 exactly, defer anything larger". 4×4 is where the cofactor's
# 24-term determinant starts to bite, so raising it further wants its own measurement.
_DEFER_INVERSE = 4   # a base-case inverse this size or larger → numeric InvOp Stmt (≤3×3 stays inline)
_DEFER_MATMUL = 2    # a Schur-combine product with any dim ≥ this → numeric matmul Stmt


def _defer_thresholds() -> tuple[int, int]:
    """Return the ``(matmul_size, inverse_size)`` deferral thresholds.

    Taken from the ambient ``SymbolicBudget`` override (``schur_matmul_stmt_size`` /
    ``schur_inverse_stmt_size``) when set, else the module defaults. This is how a caller dials the
    Schur inverse per call: ``budget_override(SymbolicBudget.build_big_symbols())`` never defers,
    giving a fully symbolic exact inverse; ``force_stmts()`` always defers, giving the numeric
    extreme.
    """
    b = current_budget_override()
    mm = getattr(b, "schur_matmul_stmt_size", None) if b is not None else None
    inv = getattr(b, "schur_inverse_stmt_size", None) if b is not None else None
    return (_DEFER_MATMUL if mm is None else mm, _DEFER_INVERSE if inv is None else inv)


def _deferred_matmul(*arrs: SymArray, masks: tuple[np.ndarray, ...] | None = None) -> SymArray:
    """Left-to-right product of ``SymArray`` blocks, deferring the large steps to numeric Stmts.

    Each 2-factor step emits a numeric matmul Stmt into the carried program when an operand is large
    (`_DEFER_MATMUL`) — the blocks evaluate numerically at the concrete inputs and the result is fresh
    atom cells — else it is an inline exact-rational matmul. Float-cell (numeric) operands never defer:
    `SymArray.matmul` short-circuits them to numpy.

    Parameters
    ----------
    *arrs
        The factors, multiplied left to right.
    masks
        One boolean nonzero mask per factor, sound in this module's sense (a ``False`` entry marks a
        cell :func:`_approx_zero` proves zero), or ``None`` to carry no mask. When supplied, the boolean
        product mask is carried along the chain and every result cell it certifies zero is written as an
        EXACT zero (:func:`mask_zeros`) instead of the Stmt's fresh output atom; a step whose whole mask
        is zero emits no Stmt at all.

    Returns
    -------
    SymArray
        The product, riding the program the operands carry.

    Notes
    -----
    The mask is what keeps a structural zero VISIBLE across a deferral. A deferred matmul's outputs are
    fresh atoms — NAMES for a pending computation, carrying no arithmetic — so without it the Schur
    combine's ``−D⁻¹·C·A⁻¹`` comes back structurally DENSE even when ``C`` is sparse and the two inverses
    are diagonal (one spurious nonzero per row), and the inverse loses exactly the sparsity the mask just
    established. Sound because ``(A·B)[i,j] = Σₖ A[i,k]·B[k,j]``: when no ``k`` has both factors nonzero,
    every term of that sum carries a proven-zero factor.
    """
    result = arrs[0]
    rmask = masks[0] if masks is not None else None
    mm_thresh = _defer_thresholds()[0]
    for pos, nxt in enumerate(arrs[1:], start=1):
        program = result.program if result.program is not None else nxt.program
        rows, inner, cols = _dim(result.shape[0]), _dim(result.shape[1]), _dim(nxt.shape[1])
        numeric = result.is_numeric and nxt.is_numeric
        nmask = None
        if rmask is not None and masks is not None:
            # int8 product then `> 0` — the boolean OR-of-ANDs, spelled so no dtype-dependent
            # interpretation of `@` on bool arrays is involved.
            nmask = (rmask.astype(np.int8) @ masks[pos].astype(np.int8)) > 0
        if nmask is not None and not nmask.any():
            result = SymArray(_zero_cells(rows, cols, symbolic=not numeric), program=program)
            rmask = nmask
            continue
        big = max(rows, inner, cols) >= mm_thresh
        if program is not None and big and not numeric:
            # A TYPED matmul op (2-D einsum `ij,jk->ik`), NOT an opaque ``lambda a, b: a @ b``: the typed op
            # lowers through EVERY backend — ``Program.run`` (numeric), ``to_numpy_source``, AND
            # ``pyab``/torch (a grounded-symbolic matrix compiled into a vmapped value kernel) — where
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
        if nmask is not None:
            result = mask_zeros(result, nmask)
        rmask = nmask
    return result


def _base_inverse(arr: SymArray) -> SymArray:
    """Invert a base-case block of size at most :data:`BASE`.

    Emits a numeric `np.linalg.inv` Stmt into the carried program when the
    block is symbolic and large enough that the symbolic cofactor determinant blows up (`_DEFER_INVERSE`);
    else exact `cofactor_inverse` (which itself short-circuits float cells).

    The inline lane is the one to PREFER when it is affordable: `cofactor_inverse` is exact rational
    arithmetic, so the result stays readable to the exact fold and to `_approx_zero`, whereas the Stmt's
    output cells are fresh atoms the fold cannot enter. `_DEFER_INVERSE` (see its comment) is where
    "affordable" stops.
    """
    n = _dim(arr.shape[0])
    program = arr.program
    if program is not None and not arr.is_numeric and n >= _defer_thresholds()[1]:
        # A TYPED ``InvOp`` (the SAME op :meth:`SymArray.inverse` defers to), NOT an opaque
        # ``lambda a: np.linalg.inv(a)``: lowers through Program.run / to_numpy_source / pyab-torch alike,
        # so a grounded-symbolic matrix compiles into a vmapped value kernel (an opaque callable cannot).
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

    Ring-less constant ``RationalFunction`` cells for the symbolic
    lane (``RationalFunction`` joins rings on contact), float zeros for the numeric.
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

    Coordinates are irrational-spaced,
    distinct per probe and per slot, so the probed cells are generic (non-degenerate) and no RNG is
    involved.
    """
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
    back to the syntactic ``simple_zero`` mask.
    """
    if matrix.program is None:
        return None
    inputs = matrix.program.inputs
    if any(is_dynamic(inp.shape) for inp in inputs):
        return None
    mask: np.ndarray | None = None
    # A 3-point probe runs the C program only 3× — no-``compile`` RF evaluation (direct term
    # sum, both in the program runner via ``probe_direct_eval`` and in the output-cell loop
    # via ``compiled=False``) is much cheaper than codegen on a degree-5 C, and
    # gives identical values, so the probed mask is unchanged.
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

    A cell is zero iff it is a SYNTACTIC (coefficient) zero, OR it is a CONSTANT (a number, or a
    total-degree-0 ``RationalFunction``) whose magnitude is ``< _MASK_TOL`` — a true zero landed as float
    ROUNDOFF (a numeric inverse over an irrational normalized basis makes some value rows ``~1e-17``,
    not exact ``0``).

    Roundoff is applied ONLY to CONSTANTS: a constant does not vary, so ``|const| < tol`` is genuinely a
    rounded zero. Reading a SYMBOLIC cell's magnitude at a few generic points would be unsound — a
    tiny-but-nonzero cell would test as zero and give a WRONG inverse — so a parameter-dependent
    ``RationalFunction`` stays EXACT (``simple_zero``, no roundoff, no sampling). Guarded end-to-end by the
    numeric-vs-symbolic backstop.
    """
    if simple_zero(cell):
        return True
    if isinstance(cell, (int, float, np.floating, np.integer)):
        return abs(float(cell)) < _MASK_TOL
    if isinstance(cell, RationalFunction) and cell.is_constant():
        return abs(float(cell.to_constant())) < _MASK_TOL
    return False


#: WORK UNITS (:class:`~polyarray.exact_fold._Meter`, roughly one monomial touched — NOT seconds)
#: the mask may spend EXACTLY folding the matrix's program (:func:`_exactly_folded_cells`).
#: Bounded like every other use of the exact lane; over budget ⇒ the raw, unfolded cells.
#:
#: The unit matters here for the same reason it matters to the certificate: the mask decides the
#: block split, hence the sparsity of the inverse this module returns, and a bound in seconds makes
#: that structure a property of the machine. Work units are a function of the program alone, so the
#: same matrix yields the same mask everywhere.
#:
#: SIZED FROM MEASUREMENT, at ``exact_fold._DEFAULT_WORK_BUDGET`` — this is the same lane, bounded
#: the same way. The heaviest mask folds observed spend several hundred thousand units. At the lane's
#: calibrated ~128 000 units/second the ceiling this replaced (10 seconds) was worth only ~1.28 M
#: units, which the heaviest runs were already spending more than half of: the mask was one loaded
#: box away from silently coarsening, which is exactly the machine dependence this currency removes.
_MASK_FOLD_WORK_BUDGET = 4_000_000


def _exactly_folded_cells(matrix: SymArray) -> np.ndarray:
    """``matrix``'s cells with every EXACTLY-CERTIFIED constant substituted in.

    This is the input the roundoff-accepting mask (:func:`_approx_zero`) actually needs. A cell of
    a lowered matrix is typically ``1.0·<atom>``: a NAME for a pending Stmt, carrying no arithmetic
    at all. ``_approx_zero`` can say nothing about such a cell, so a cell whose VALUE is zero is
    kept as a nonzero and the block split never happens. The exact lane
    (:func:`~polyarray.exact_fold.exact_partial_eval` +
    :func:`~polyarray.exact_fold.exact_fold_cells`) resolves exactly those names, over ℚ(feed
    atoms), and replaces a cell whose normal form has total degree zero by its exact constant.

    Parameters
    ----------
    matrix
        The matrix whose cells to fold. One with no program, or with float cells, is returned as-is.

    Returns
    -------
    np.ndarray
        The cell array, with every certified-constant cell replaced by its exact constant.

    Notes
    -----
    SOUND — it is the same rule, better informed. The fold is exact-by-construction (rational
    normal form, no sampling), so a cell it certifies constant IS constant, which is precisely the
    condition ``_approx_zero`` requires before it applies roundoff. A cell the fold cannot resolve
    is returned UNCHANGED, so the mask can only get sparser where a constant was proved, never
    where one was guessed. Any failure (no program, over budget, a raise) falls back to the raw
    cells.
    """
    cells = np.asarray(matrix.cells)
    program = getattr(matrix, "program", None)
    if program is None or cells.dtype.kind == "f":
        return cells
    from .exact_fold import exact_fold_cells, exact_partial_eval
    try:
        state = exact_partial_eval(program, work_budget=_MASK_FOLD_WORK_BUDGET)
        return np.asarray(exact_fold_cells(cells, state, program,
                                           work_budget=_MASK_FOLD_WORK_BUDGET))
    except Exception:  # noqa: BLE001 — a failing fold is a non-fold, never a crash
        return cells


def _syntactic_mask(cells: np.ndarray) -> np.ndarray:
    """Build the conservative mask: a cell is nonzero unless :func:`_approx_zero` proves it zero."""
    out = np.empty(cells.shape, dtype=bool)
    for idx in np.ndindex(cells.shape):
        out[idx] = not _approx_zero(cells[idx])
    return out


def _result_mask(arr: SymArray) -> np.ndarray:
    """Return the nonzero mask of a block the recursion has already materialized.

    A diagonal inverse, a component assembly, a Schur combine: the mask is read straight off its cells
    by :func:`_syntactic_mask`. Sound in the same sense as every other mask here — a ``False`` entry is
    a cell :func:`_approx_zero` proves zero — and it needs no re-derivation because the recursion writes
    its structural zeros as EXACT zeros. It is deliberately NOT :func:`_resolve_mask`: this array is an
    intermediate of the inverse, not the user's matrix, so there is nothing to fold and no probe to run.

    Parameters
    ----------
    arr
        A block the recursion produced.

    Returns
    -------
    np.ndarray
        The boolean nonzero mask.
    """
    return _syntactic_mask(np.asarray(arr.cells))


def mask_zeros(arr: SymArray, mask: np.ndarray) -> SymArray:
    """``arr`` with every cell ``mask`` marks zero replaced by an EXACT zero of the matching lane.

    The point is downstream VISIBILITY: a proven-zero cell must be a syntactic zero, not a fresh
    Stmt-output atom, or the next :func:`_syntactic_mask`/:func:`_approx_zero` reader cannot see it.

    Parameters
    ----------
    arr
        The block to rewrite.
    mask
        Its boolean nonzero mask, sound in this module's sense (a ``False`` entry is a cell proved
        zero — see :func:`sound_sparsity_mask`). An all-``True`` mask returns ``arr`` unchanged.

    Returns
    -------
    SymArray
        The block with its proven zeros made syntactic, riding the same program.

    Notes
    -----
    The companion of :func:`sound_sparsity_mask`, and the other half of the graft problem it
    describes: a mask taken where the zeros are still visible can either be CARRIED to the one
    reader that accepts a ``mask=`` argument, or WRITTEN BACK into the matrix here — after which
    every reader sees the sparsity, from the exact fold and the degree walk to codegen and to
    whatever arithmetic the consumer does with the matrix itself. Writing it back is the stronger
    move for a matrix with more than one consumer.

    It is the same commitment the recursion already makes internally: :func:`_deferred_matmul`
    writes the product mask's zeros into a deferred matmul's fresh atoms, and ``_schur_combine``
    sets a whole inverse block to exact zero on the mask's say-so. The mask decides values here as
    it does there; this only makes the treatment available to a caller.
    """
    if mask.all():
        return arr
    cells = np.array(arr.cells)
    cells[~mask] = _zero_cells(1, 1, symbolic=not arr.is_numeric)[0, 0]
    return SymArray(cells, program=arr.program)


def _resolve_mask(matrix: SymArray, mask: np.ndarray | None) -> np.ndarray:
    """Resolve the nonzero mask steering the recursion. Sound by default.

    An explicitly-threaded sub-mask wins. Otherwise the syntactic ``simple_zero`` mask: it marks a cell
    zero ONLY when its numerator polynomial is coefficient-zero, so it can NEVER false-zero a truly-
    nonzero cell (at worst it is conservative — dense — when it cannot prove a cell zero, which is always
    safe: a denser mask only makes the block split less aggressive, never wrong).

    The numeric probe ``_structural_mask`` is UNSOUND and is NOT the default: it marks a cell zero when
    it is merely SMALL (< ``_MASK_TOL``) at a few sample points, so a *tiny-but-nonzero* cell is dropped
    and the Schur split silently returns a WRONG inverse (adding sample points does not help — the cells
    are tiny at every point). It is available only behind an explicit, self-labelled-unsound opt-in
    (``POLYARRAY_SCHUR_UNSOUND_PROBE_MASK``) for a large matrix whose cancellation-sparsity a sound exact
    fold cannot yet recover — and only ever with the numeric-vs-symbolic backstop watching it.
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
    return _syntactic_mask(_exactly_folded_cells(matrix))


def sound_sparsity_mask(matrix: SymArray) -> np.ndarray:
    """Return the sound nonzero mask of ``matrix``, as a value a caller can hold and carry.

    The same mask :func:`symbolic_inverse` resolves for itself, exposed so a caller can hand it back
    via that function's ``mask=`` argument.

    Parameters
    ----------
    matrix
        The matrix to read.

    Returns
    -------
    np.ndarray
        A boolean mask. A ``False`` entry is a cell proved zero (syntactically, or a constant within
        roundoff after the exact fold); a ``True`` entry claims nothing. Conservative by construction:
        a denser mask makes the block split less aggressive, never wrong.

    Notes
    -----
    WHY A CONSUMER NEEDS THIS. A mask must sometimes be taken at a point where the zeros are still
    VISIBLE and used at a point where they are not. :meth:`~polyarray.ir.Program.graft` re-homes a
    matrix by emitting its producing program as one Stmt whose outputs are FRESH ATOMS, one per cell —
    values preserved, but a provably-zero cell becomes an opaque atom that no later reader can prove
    zero, so a sparse block's mask comes back fully dense.
    Grafting preserves values, so a mask proved before it is still a valid mask of the same matrix
    after; reading it early and passing it forward is how the sparsity survives.
    """
    return _resolve_mask(matrix, None)


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
    """Reorder rows ascending by their rightmost non-zero column.

    This clusters rows whose support is confined to early columns at the top, exposing
    block-lower-triangular structure. The mask is reordered in lockstep.

    Returns
    -------
    tuple
        ``(new_order, sizes, reordered_arr, reordered_mask)``, where ``sizes[i]`` is the rightmost
        non-zero column of the row now at position ``i``.
    """
    rightmost = _rightmost_nonzero(mask)
    new_order = sorted(range(len(rightmost)), key=lambda k: rightmost[k])
    sizes = [rightmost[k] for k in new_order]
    return new_order, sizes, arr[new_order], mask[new_order]


def _components(mask: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Find the connected components of the bipartite row-column graph, ``row i ~ col j`` iff ``mask[i, j]``.

    A matrix block-DIAGONAL under a (possibly asymmetric) row/column permutation splits into one component
    per block, each inverting independently — catching structure the block-*triangular* split cannot (e.g. a
    6×6 that is three disjoint 2×2 blocks, block-diagonal but not triangular). Returns ``[(rows, cols), …]``
    with sorted index lists; for a nonsingular matrix every component is square.
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

    Valid when the first ``p`` rows' support stays
    zero — i.e. the first ``p`` rows' support stays within the first ``p`` columns (``sizes[p-1] < p``, using
    the ascending sort). Among valid splits pick the most balanced (closest to ``n/2``). ``None`` if no
    zero-block split exists.
    """
    valid = [p for p in range(1, n) if sizes[p - 1] < p]
    if not valid:
        return None
    return min(valid, key=lambda p: abs(2 * p - n))


# --- the recursion ---------------------------------------------------------------------------


def _schur_combine(M: SymArray, k: int, mask: np.ndarray) -> SymArray:
    """Invert ``M`` split at ``k`` via Schur, recursing on the diagonal blocks.

    Handles the block-lower- triangular fast path (``B == 0`` per the mask) and the general case. The mask is
    sliced in lockstep with ``M`` so the diagonal-block recursions see the true sparsity; the Schur complement
    ``S`` is a fresh arithmetic product with no sampled mask, so it gets the syntactic ``simple_zero`` mask.
    Matrix products go through ``_deferred_matmul`` so a large combine defers to a numeric Stmt instead of
    blowing up.

    Every product is given the masks of its FACTORS (the caller's mask for the ``M`` sub-blocks,
    :func:`_result_mask` for the already-materialized inverses), so the structural zeros survive the
    deferral — see :func:`_deferred_matmul`. Without them the combine returns a dense block whatever the
    mask says, and the sparsity the split was chosen for is lost in the result.
    """
    n = _dim(M.shape[0])
    symbolic = not M.is_numeric
    A, B, C, D = M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]
    mA, mB, mC, mD = mask[:k, :k], mask[:k, k:], mask[k:, :k], mask[k:, k:]
    A_inv = _invert(A, mask=mA)
    mA_inv = _result_mask(A_inv)
    out = np.empty((n, n), dtype=object) if symbolic else np.empty((n, n))
    if _all_zero(mB):
        D_inv = _invert(D, mask=mD)
        out[:k, :k] = A_inv.cells
        out[:k, k:] = _zero_cells(k, n - k, symbolic)
        out[k:, :k] = (-_deferred_matmul(D_inv, C, A_inv,
                                         masks=(_result_mask(D_inv), mC, mA_inv))).cells
        out[k:, k:] = D_inv.cells
    else:
        S = D - _deferred_matmul(C, A_inv, B, masks=(mC, mA_inv, mB))
        S_inv = _invert(S, mask=_syntactic_mask(S.cells))
        mS_inv = _result_mask(S_inv)
        AB_Sinv = _deferred_matmul(A_inv, B, S_inv, masks=(mA_inv, mB, mS_inv))
        CA = _deferred_matmul(C, A_inv, masks=(mC, mA_inv))
        mAB_Sinv, mCA = _result_mask(AB_Sinv), _result_mask(CA)
        out[:k, :k] = (A_inv + _deferred_matmul(AB_Sinv, CA, masks=(mAB_Sinv, mCA))).cells
        out[:k, k:] = (-AB_Sinv).cells
        out[k:, :k] = (-_deferred_matmul(S_inv, CA, masks=(mS_inv, mCA))).cells
        out[k:, k:] = S_inv.cells
    return SymArray(out, program=M.program)


def symbolic_inverse(matrix: SymArray | np.ndarray, *, mask: np.ndarray | None = None,
                     program: Program | None = None) -> SymArray:
    """Invert a square matrix via the block-triangular Schur recursion.

    The sparsity-aware sibling of :meth:`SymArray.inverse`.

    Accepts a ``SymArray`` or a raw cell ndarray; returns a ``SymArray`` of ``RationalFunction``/numeric
    cells, propagating the input's owning ``Program`` so downstream simplify/sparsity/codegen passes apply
    and Stmt deferrals emit into it. ``RationalFunction`` arithmetic coerces numerics and joins rings on
    contact.

    ``mask`` is the boolean nonzero mask steering the block split. Pass it when the caller can compute the
    sparsity cheaply/exactly; when omitted it is resolved from ``matrix`` by DETERMINISTIC probing (a
    program-carrying SymArray) or the syntactic ``simple_zero`` fallback. A conservative (denser) mask only
    makes the split less aggressive, never wrong.

    ``program`` (GROUNDED SYMBOLIC): the SHARED ``Program`` the block-triangular inverse should be GROUNDED
    onto — so a symbolic inverse lowers through a value kernel compiled from that shared program
    rather than leaving Stmts stranded on an ephemeral by-product program. When ``matrix``
    already rides ``program`` (or ``program`` is ``None``, or ``matrix`` is numeric / program-less) this is a
    no-op and the Stmts emit into ``matrix``'s own program. Otherwise the
    block-split mask is resolved on ``matrix``'s OWN program FIRST — its inputs are the clean generic-cell
    generators the deterministic structural probe needs — and only then is ``matrix`` GRAFTED onto
    ``program`` (:meth:`Program.graft`): ``matrix``'s cells may reference not only shared input atoms but
    also *its program's own producing Stmts* (e.g. Stmts produced upstream in that program), so a bare
    relabel would strand those; the graft emits ``matrix``'s program as a
    sub-Program Stmt of ``program`` (fresh dedup'd outputs). The recursion's own deferred leaf-inverse /
    Schur-combine Stmts (``schur_inverse``/``schur_matmul``) then emit natively onto ``program`` (mask
    threaded so it never re-probes on the shared program), and several matrices grounded
    on one shared program do not collide.
    """
    M = matrix if isinstance(matrix, SymArray) else SymArray(matrix)
    if program is not None and M.program is not None and M.program is not program:
        mask = _resolve_mask(M, mask)                       # probe on M's OWN (clean-input) program first
        M = cast(Program, program).graft(M)                 # bring M's producing Stmts onto `program`
    return _invert(M, mask=mask)


def _invert(M: SymArray, mask: np.ndarray | None = None) -> SymArray:
    """Run the recursion proper on the SymArray carrier.

    ``mask`` steers the block-triangular split; when omitted it is resolved from ``M``. Sub-recursions thread
    the corresponding sub-mask so the sparsity stays aligned through the row reordering and block splits.
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
