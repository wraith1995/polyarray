"""Structural-zero sparsity propagation over a Program.

This is the *additive* sparsity pass.  It threads a structural-zero boolean mask (``True``
= structurally zero) through ``program.statements`` in run order, computing a
mask for every :class:`SymArray` — program inputs, each :class:`Stmt` output,
and each program output.

Hard rules (obeyed exactly):

* **Structural, never numeric.**  A mask bit is set only when the *algebra*
  guarantees zero — ``simple_zero`` (rational lane) or a literal ``0.0`` (float
  lane), via :func:`~polyarray.ir.cells_sparsity`.  False-*negatives* (a true
  zero left marked unknown) are safe; false-*positives* (a non-zero marked zero)
  are a correctness bug, so anything not modeled defaults to ``False`` (unknown).
* **Never cross an opaque Stmt boundary.**  ``DetOp`` / ``InvOp`` / ``PinvOp`` /
  ``SolveOp`` / ``SvdOp`` / ``GSvdOp`` / ``QrOp`` / ``SqrtOp`` / ``AbsOp`` /
  ``SignOp`` / ``SwitchOp`` / ``CallOp`` / ``WhileOp`` / a sub-``Program`` /
  any unmodeled op ⇒ the output mask **resets to all-False** (unknown).
* **Bulk outputs** (``SymArray._bulk is not None``) carry no per-cell symbols;
  their mask is all-False (unknown) unless derived from a modeled op's inputs.
  We never call ``.cells`` on a bulk array (it would materialise / unpack).

The pass is read-only on the program: it never mutates inputs, statements, cells
or outputs.

Modeled propagation rules
-------------------------

* ``A + B`` / ``A - B`` (:func:`add_mask`): zero where **both** zero.
* ``A * B`` / Hadamard (:func:`mul_mask`): zero where **either** zero.
* ``A @ B`` / ``TensordotOp`` (:func:`tensordot_mask`),
  ``EinsumOp`` / ``EinsumStmtOp`` (:func:`einsum_mask`): ``out`` zero iff
  **every** contributing product term has a zero operand — a boolean contraction
  mirroring the op's axes / einsum subscripts.
* ``MoveaxisOp`` (:func:`moveaxis_mask`): permute the mask.
* ``IdentityOp`` / ``AssertOp``: passthrough (copy the input mask).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, assert_never

import numpy as np

from .ir import (
    AbsOp,
    AddOp,
    AssertOp,
    AxisLenOp,
    BlockDiagOp,
    BlockRepeatOp,
    CallOp,
    ColStackOp,
    CompRankOp,
    ComposeViaStdOp,
    ConcatOp,
    Const,
    ConstOp,
    DetOp,
    DynBlockRepeatOp,
    DynEyeOp,
    DynEyeTensorOp,
    DynZerosOp,
    EinsumOp,
    EinsumStmtOp,
    EmbedOp,
    EyeOp,
    FirstColsOp,
    GSvdFullOp,
    GSvdOp,
    HStackOp,
    IdentityOp,
    InputRef,
    IntAtomRef,
    InvOp,
    InvTransposeOp,
    KronFreeOp,
    KronOp,
    LastColsOp,
    MetricOrthonormalOp,
    MoveaxisOp,
    MulAxisDimOp,
    OutputRef,
    PinvOp,
    ProdDimOp,
    ProdShapeOp,
    Program,
    ProjectOp,
    QrOp,
    RankOp,
    RationalRef,
    ReshapeOp,
    ScaleAxisDimOp,
    ScaleByOp,
    ScaleOp,
    SignOp,
    SinvFullOp,
    SolveOp,
    SqrtOp,
    SqrtSpdOp,
    Stmt,
    StmtFn,
    SumDimOp,
    SumShapeOp,
    SvdOp,
    SwitchOp,
    SymArray,
    Ref,
    SymArrayRef,
    TensordotOp,
    TransposeOp,
    WhileOp,
    cells_sparsity,
    is_builtin_op,
    is_dynamic,
)
from .rational import simple_zero

# A mask is an ``ndarray`` of bool (True = structurally zero), or ``None`` when
# the shape is dynamic / unknown (treated as all-False / unknown by consumers).
Mask: TypeAlias = np.ndarray | None

#: One axis index, or a tuple/list of them — what ``np.moveaxis`` accepts.
Axes: TypeAlias = int | Sequence[int]

#: The ``axes`` argument of ``np.tensordot``: a contraction count, or the two
#: index lists to contract over.
TensordotAxes: TypeAlias = int | Sequence[Sequence[int]]


# ---------------------------------------------------------------------------
# Mask combinators (the modeled propagation rules) — reusable building blocks
# ---------------------------------------------------------------------------

def add_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combine masks under ``A + B`` / ``A - B``: zero where **both** are zero."""
    return np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)


def mul_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combine masks under ``A * B`` / Hadamard: zero where **either** is zero."""
    return np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)


def tensordot_mask(a: np.ndarray, b: np.ndarray, axes: TensordotAxes) -> np.ndarray:
    """Boolean ``np.tensordot`` contraction of two structural-zero masks.

    ``out`` is structurally zero iff **every** contributing product pair has a
    zero operand.  Equivalently ``out`` is *possibly* non-zero iff some
    contracted index makes both operands non-zero — counted by a real-valued
    ``tensordot`` over the non-zero indicators ``~mask``.
    """
    a_nz = (~np.asarray(a, dtype=bool)).astype(float)
    b_nz = (~np.asarray(b, dtype=bool)).astype(float)
    count = np.tensordot(a_nz, b_nz, axes=axes)
    return np.asarray(count) == 0.0


def einsum_mask(spec: str, *masks: np.ndarray, optimize: bool = True) -> np.ndarray:
    """Boolean ``np.einsum`` contraction of several structural-zero masks.

    Mirrors the einsum: a contributing term is non-zero iff *all* its operand
    factors are non-zero (product of ``~mask`` indicators is 1); the output cell
    is structurally zero iff **no** contributing term is non-zero (count == 0).
    """
    nz = [(~np.asarray(m, dtype=bool)).astype(float) for m in masks]
    count = np.einsum(spec, *nz, optimize=optimize)
    return np.asarray(count) == 0.0


def moveaxis_mask(mask: np.ndarray, source: Axes, destination: Axes) -> np.ndarray:
    """Permute a mask exactly as ``np.moveaxis`` permutes its array."""
    return np.moveaxis(np.asarray(mask, dtype=bool), source, destination)


# ---------------------------------------------------------------------------
# Shape / array helpers
# ---------------------------------------------------------------------------

def _static_shape(sa: SymArray) -> tuple[int, ...] | None:
    """Return the concrete shape of ``sa``, or ``None`` if it is ``DimAtom``-sized."""
    shp = sa._bulk.shape if sa._bulk is not None else sa._cells.shape
    if is_dynamic(shp):
        return None
    return tuple(int(d) for d in shp)


def _false_mask(sa: SymArray) -> Mask:
    """Return an all-False (unknown) mask shaped like ``sa``, or ``None`` if dynamic."""
    shp = _static_shape(sa)
    return None if shp is None else np.zeros(shp, dtype=bool)


def _array_mask(sa: SymArray) -> Mask:
    """Return the structural-zero mask of a leaf SymArray, never unpacking bulk."""
    if sa._bulk is not None:
        return _false_mask(sa)
    return cells_sparsity(np.asarray(sa._cells))


def block_zero_mask(symarray_or_cells: SymArray | np.ndarray) -> np.ndarray:
    """Read the structural-zero pattern of a matrix.

    The convenience a Schur-inverse caller uses to choose the ``(p, q)``
    split that maximises the zero block: ``True`` where the cell is structurally
    zero (``simple_zero`` / literal ``0.0``).

    Accepts a :class:`SymArray` or a raw cell ``ndarray``.  A *bulk* SymArray has
    no per-cell symbols, so its pattern is all-False (unknown) — we never unpack
    it to read a pattern that isn't there.
    """
    if isinstance(symarray_or_cells, SymArray):
        if symarray_or_cells._bulk is not None:
            m = _false_mask(symarray_or_cells)
            if m is None:
                raise ValueError(
                    "block_zero_mask: dynamic bulk SymArray has no static "
                    "zero pattern"
                )
            return m
        cells = np.asarray(symarray_or_cells._cells)
    else:
        cells = np.asarray(symarray_or_cells)
    return cells_sparsity(cells)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class SparsityReport:
    """Structural-zero masks for every SymArray in a Program.

    ``True`` = structurally zero.  A mask is ``None`` when its array's shape is
    dynamic (unknown until run time) — consumers treat ``None`` as all-unknown.
    """

    input_masks: dict[str, Mask]
    output_masks: dict[str, Mask]
    stmt_out_masks: dict[tuple[int, int], Mask]
    _cells_id_to_mask: dict[int, Mask] = field(default_factory=dict, repr=False)
    _bulk_name_to_mask: dict[str, Mask] = field(default_factory=dict, repr=False)

    def output_mask(self, name: str) -> Mask:
        """Mask for the named program output."""
        return self.output_masks[name]

    def input_mask(self, name: str) -> Mask:
        """Mask for the named program input."""
        return self.input_masks[name]

    def mask_for(self, key: str | SymArray) -> Mask:
        """Fetch the mask for a program output / input *name* or a ``SymArray``.

        A ``SymArray`` is matched first by identity of its computed cells (so a
        Stmt output / output that *shares* those cells gets the precise
        propagated mask), then — for a leaf array we never propagated through —
        falls back to its own structural-zero pattern.
        """
        if isinstance(key, str):
            if key in self.output_masks:
                return self.output_masks[key]
            if key in self.input_masks:
                return self.input_masks[key]
            raise KeyError(f"no mask for name {key!r}")
        if isinstance(key, SymArray):
            if key._bulk is not None:
                return self._bulk_name_to_mask.get(
                    key._bulk.name, _false_mask(key)
                )
            m = self._cells_id_to_mask.get(id(key._cells))
            if m is not None:
                return m
            return cells_sparsity(np.asarray(key._cells))
        raise TypeError(f"mask_for: expected str or SymArray, got {type(key)}")


# ---------------------------------------------------------------------------
# Ref resolution
# ---------------------------------------------------------------------------

def _index(mask: Mask, indices: tuple[int, ...]) -> Mask:
    """Index into a mask the way a ref's ``indices`` index into its array."""
    if mask is None or not indices:
        return mask
    return np.asarray(mask[indices])


def _resolve_ref_mask(
    ref: Ref,
    input_masks: dict[str, Mask],
    stmt_out_masks: dict[tuple[int, int], Mask],
    cells_id_to_mask: dict[int, Mask],
    bulk_name_to_mask: dict[str, Mask],
) -> Mask:
    """Return the structural-zero mask flowing in through a statement input ref.

    A :class:`SymArrayRef` to a prior statement output or to an input carries that
    array's *fresh-atom* cells, which have no structural zeros of their own, so the
    propagated mask is looked up by cells identity — the producing op's modeled rule —
    before falling back to the cells' own pattern.

    Returns
    -------
    Mask
        ``None`` for a ref that carries no array mask (an ``IntAtomRef`` selector) or
        whose shape is dynamic. Callers read ``None`` as "unknown", which keeps the pass
        conservative.
    """
    if isinstance(ref, InputRef):
        return _index(input_masks.get(ref.name), ref.indices)
    if isinstance(ref, OutputRef):
        return _index(stmt_out_masks.get((ref.stmt_idx, ref.out_idx)), ref.indices)
    if isinstance(ref, Const):
        return cells_sparsity(np.asarray(ref.value))
    if isinstance(ref, RationalRef):
        return np.asarray(simple_zero(ref.rf), dtype=bool)
    if isinstance(ref, SymArrayRef):
        if ref._bulk is not None:
            if ref._bulk.name in bulk_name_to_mask:
                return bulk_name_to_mask[ref._bulk.name]
            shp = ref._bulk.shape
            if is_dynamic(shp):
                return None
            return np.zeros(tuple(int(d) for d in shp), dtype=bool)
        m = cells_id_to_mask.get(id(ref._cells))
        if m is not None:
            return m
        return cells_sparsity(np.asarray(ref.cells))
    if isinstance(ref, IntAtomRef):
        return None
    return None


# ---------------------------------------------------------------------------
# Per-op propagation
# ---------------------------------------------------------------------------

def _checked(m: Mask, osa: SymArray) -> Mask:
    """Accept a modeled output mask only if its shape matches ``osa`` exactly.

    A shape mismatch would risk a *false positive* (a structural zero in the
    wrong cell), so we conservatively fall back to all-False (unknown).
    """
    declared = _static_shape(osa)
    if m is None or declared is None or tuple(np.asarray(m).shape) != declared:
        return _false_mask(osa)
    return np.asarray(m, dtype=bool)


def _apply_op(stmt: Stmt, in_masks: list[Mask]) -> list[Mask]:
    """Apply the modeled propagation rule for ``stmt.fn`` to its input masks.

    ``stmt.fn`` is only half closed: a sub-:class:`~polyarray.ir.Program`, a ``vmap``
    closure and a front-end op class are all legal and have no modeled rule here, so they
    reset to unknown.  polyarray's OWN vocabulary is dispatched exhaustively by
    :func:`_apply_builtin_op`.
    """
    outs = stmt.out
    unknown = [_false_mask(o) for o in outs]
    if not is_builtin_op(stmt.fn):
        return unknown                    # sub-Program / vmap closure / front-end op
    return _apply_builtin_op(stmt.fn, in_masks, outs, unknown)


def _apply_builtin_op(
    fn: StmtFn, in_masks: list[Mask], outs: tuple[SymArray, ...], unknown: list[Mask],
) -> list[Mask]:
    """Apply the modeled mask rule for one polyarray-owned op.

    Exhaustive over :data:`~polyarray.ir.StmtFn`; ``assert_never`` makes a newly added op
    a mypy error rather than a silent fall-through.

    Unlike the exact-fold lane, an omission here is not a soundness bug: ``unknown``
    (all-False) is the conservative answer for every op, since false negatives are safe
    while false positives are a correctness bug. The grouped arm below is therefore a
    precision ledger — but a ledger nonetheless, so that "no rule yet" is written down
    per op instead of being the silent default.
    """
    match fn:
      # --- contraction ops -------------------------------------------------
      case TensordotOp():
        if len(in_masks) != 2 or len(outs) != 1:
            return unknown
        a, b = in_masks
        if a is None or b is None:
            return [_false_mask(outs[0])]
        try:
            m = tensordot_mask(a, b, fn.axes)
        except Exception:
            return [_false_mask(outs[0])]
        return [_checked(m, outs[0])]

      case EinsumOp():
        if len(in_masks) != 1 or len(outs) != 1:
            return unknown
        lhs = in_masks[0]
        if lhs is None:
            return [_false_mask(outs[0])]
        rhs = cells_sparsity(np.asarray(fn._rhs))
        try:
            m = einsum_mask(fn.spec, lhs, rhs, optimize=False)
        except Exception:
            return [_false_mask(outs[0])]
        return [_checked(m, outs[0])]

      case EinsumStmtOp():
        if len(outs) != 1:
            return unknown
        if any(m is None for m in in_masks):
            return [_false_mask(outs[0])]
        try:
            m = einsum_mask(fn.spec, *in_masks, optimize=fn.optimize)
        except Exception:
            return [_false_mask(outs[0])]
        return [_checked(m, outs[0])]

      # --- shape / passthrough ops ----------------------------------------
      case MoveaxisOp():
        if len(in_masks) != 1 or len(outs) != 1:
            return unknown
        x = in_masks[0]
        if x is None:
            return [_false_mask(outs[0])]
        try:
            m = moveaxis_mask(x, fn.source, fn.destination)
        except Exception:
            return [_false_mask(outs[0])]
        return [_checked(m, outs[0])]

      # ``IdentityOp`` freezes a heavy cell as a fresh atom — value-preserving.
      # ``AssertOp`` returns its first input unchanged.  Both pass the (first)
      # input's mask straight through.
      case IdentityOp() | AssertOp():
        if not in_masks or len(outs) != 1:
            return unknown
        x = in_masks[0]
        if x is None:
            return [_false_mask(outs[0])]
        return [_checked(x, outs[0])]

      # --- genuinely opaque: a structural zero cannot survive --------------
      #
      # A zero entry of the operand says nothing about a zero entry of the result: an
      # inverse / factorization / square root / sign mixes every entry, a rank or shape
      # read is not entrywise at all, and control flow picks a branch this pass does not
      # follow.  All-False (unknown) is the RIGHT answer here, not merely a safe one.
      case DetOp() | InvOp() | InvTransposeOp() | PinvOp() | SolveOp() | ComposeViaStdOp() \
              | SqrtOp() | SqrtSpdOp() | QrOp() | SvdOp() | GSvdOp() | GSvdFullOp() \
              | SinvFullOp() | MetricOrthonormalOp() | AbsOp() | SignOp() | RankOp() \
              | CompRankOp() | SwitchOp() | CallOp() | WhileOp():
        return unknown

      # --- constants: a mask could be read off the value, but is not -------
      #
      # `ConstOp` / `EyeOp` / `DynEyeOp` / `DynZerosOp` / `DynEyeTensorOp` produce
      # KNOWN arrays (an identity is zero off the diagonal; `DynZerosOp` is zero
      # everywhere).  Modeling them would be sound and would seed real structure into a
      # downstream contraction — the `DynZerosOp` case especially, since it is *entirely*
      # zero and currently propagates as fully unknown.  Not implemented: a precision gap.
      case ConstOp() | EyeOp() | DynEyeOp() | DynZerosOp() | DynEyeTensorOp():
        return unknown

      # --- shape-derived 0-d ints: no cells to mask ------------------------
      case AxisLenOp() | ProdShapeOp() | SumShapeOp() | SumDimOp() | ProdDimOp() \
              | ScaleAxisDimOp() | MulAxisDimOp():
        return unknown

      # --- structural rearrangement: modelable, NOT modeled ----------------
      #
      # Every op here permutes / slices / tiles / adds / scales entries, so its output mask
      # is an exact function of its input masks (the same class of rule as `MoveaxisOp`
      # above, which IS modeled).  These are the real precision gaps in this pass:
      # `ReshapeOp` and `TransposeOp` in particular sit on ordinary lowering paths and
      # currently erase a mask that was fully known one statement earlier.
      case TransposeOp() | ReshapeOp() | HStackOp() | ColStackOp() | ConcatOp() \
              | AddOp() | ScaleOp() | ScaleByOp() | BlockDiagOp() | BlockRepeatOp() \
              | DynBlockRepeatOp() | KronOp() | KronFreeOp() | FirstColsOp() \
              | LastColsOp() | ProjectOp() | EmbedOp():
        return unknown

      case _ as unreachable:
        assert_never(unreachable)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def propagate_sparsity(program: Program) -> SparsityReport:
    """Compute structural-zero masks for every SymArray in ``program``.

    Walks ``program.statements`` in run order, seeding masks from the inputs
    (each cell zero iff ``simple_zero`` / literal ``0.0``), then applying the
    modeled op rules to each Stmt's input masks to obtain its output masks.
    Opaque / unmodeled ops reset their outputs to all-False (unknown).

    Read-only: the program is never mutated.
    """
    input_masks: dict[str, Mask] = {}
    cells_id_to_mask: dict[int, Mask] = {}
    bulk_name_to_mask: dict[str, Mask] = {}

    for name, sa in program.input_arrays.items():
        m = _array_mask(sa)
        input_masks[name] = m
        if sa._bulk is not None:
            bulk_name_to_mask[sa._bulk.name] = m
        else:
            cells_id_to_mask[id(sa._cells)] = m

    stmt_out_masks: dict[tuple[int, int], Mask] = {}
    for si, stmt in enumerate(program.statements):
        in_masks = [
            _resolve_ref_mask(
                r, input_masks, stmt_out_masks,
                cells_id_to_mask, bulk_name_to_mask,
            )
            for r in stmt.in_
        ]
        out_masks = _apply_op(stmt, in_masks)
        for oi, osa in enumerate(stmt.out):
            m = out_masks[oi] if oi < len(out_masks) else _false_mask(osa)
            stmt_out_masks[(si, oi)] = m
            if osa._bulk is not None:
                bulk_name_to_mask[osa._bulk.name] = m
            else:
                cells_id_to_mask[id(osa._cells)] = m

    output_masks: dict[str, Mask] = {}
    for name, sa in program.outputs.items():
        if sa._bulk is not None:
            output_masks[name] = bulk_name_to_mask.get(
                sa._bulk.name, _false_mask(sa)
            )
            continue
        m = cells_id_to_mask.get(id(sa._cells))
        output_masks[name] = (
            m if m is not None else cells_sparsity(np.asarray(sa._cells))
        )

    return SparsityReport(
        input_masks=input_masks,
        output_masks=output_masks,
        stmt_out_masks=stmt_out_masks,
        _cells_id_to_mask=cells_id_to_mask,
        _bulk_name_to_mask=bulk_name_to_mask,
    )
