"""Batched execution of a :class:`~polyarray.ir.Program` over a leading batch axis.

``batched_run(program, values)`` evaluates ``program`` for a whole BATCH of inputs at once — each entry of
``values`` carries a leading batch axis of length ``B`` — reproducing ``[program.run(values[b]) for b in
range(B)]`` in ONE vectorized pass, but without the Python-per-call / per-Stmt interpreter overhead that
dominates when a program is tiny and run many times (a per-sample-point loop).

Two lanes, mirroring :meth:`Program.run`:

* **Cell lane** — a program's symbolic ``RationalFunction`` cells are polynomials over per-element input
  atoms.  The batch axis stays IMPLICIT: each atom is bound to its ``(B,)`` slice and ``eval_numeric_fast``
  broadcasts, so a cell of declared shape ``s`` evaluates to ``(B, *s)``.  (Shapes are NOT rewritten — that
  would rename atoms and break the cells.)
* **Stmt lane** — each op is applied with the batch as leading axis 0.  Dispatch is by ``isinstance``
  over polyarray's own op vocabulary: einsum ellipsis-batches, ``numpy`` linalg batches over leading
  axes, axis ops shift by one.  A front-end op is not one of these and falls to the generic branch,
  which runs it unbatched or declines — never mistaking it for the builtin of the same name.

Agreement with the per-element loop is EXACT for the structural and elementwise ops (reshape, scale,
add, slice, axis length — verified ``assert_array_equal`` in ``tests/test_batch.py``), but NOT exact in
general.  ``EinsumStmtOp`` batches by rewriting the spec with an ellipsis and passes ``optimize=``, so
numpy may choose a different contraction ORDER than the unbatched call; ``np.linalg`` likewise takes a
stacked path rather than a per-matrix one.  Both re-associate floating-point arithmetic.

Agreement is therefore to rounding, not to the bit — a claim of bit-identity holds only for
the exact ops above.

Raises :class:`NotImplementedError` for any op without a batch rule (or a sub-Program / dynamic
construct it does not handle) so the caller can fall back to the per-element loop.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from .ir import (AbsOp, AddOp, AssertOp, AxisLenOp, ColStackOp, ConcatOp, Const, DetOp,
                 EinsumStmtOp, EmbedOp, FirstColsOp, IdentityOp, InputRef, IntAtomRef, InvOp,
                 LastColsOp, MoveaxisOp, OutputRef, PinvOp, Program, ProjectOp, RationalRef, Ref,
                 ReshapeOp, ScaleByOp, ScaleOp, SignOp, SolveOp, SqrtOp, StmtOp, SymArray,
                 SymArrayRef, TransposeOp, _cached_einsum, is_dynamic)
from .rational import RationalFunction

__all__ = ["batched_run", "BatchUnsupported"]


class BatchUnsupported(NotImplementedError):
    """An op / construct in the program has no batch rule — fall back to the per-element loop."""


def _batched_einsum_spec(spec: str) -> str:
    """Prepend a fresh batch index to every operand and the output of ``spec``."""
    lhs, _, rhs = spec.partition("->")
    used = set(spec) - set(",->")
    batch = next((ch for ch in "BNZYXWVUTSRQPOMLKJ" if ch not in used), None)
    if batch is None:
        raise BatchUnsupported(f"no free einsum index to batch {spec!r}")
    ins = ",".join(batch + t.strip() for t in lhs.split(","))
    return f"{ins}->{batch}{rhs.strip()}"


def _apply(fn: StmtOp, ins: list[tuple[np.ndarray, bool]]) -> tuple[np.ndarray, bool]:
    """Apply op ``fn`` to operands ``[(value, is_batched), …]``, returning ``(value, is_batched)``.

    Parameters
    ----------
    fn
        The statement's op. Dispatch is by class name so that front-end ops need no
        import here.
    ins
        One ``(value, is_batched)`` pair per operand, in operand order.

    Returns
    -------
    tuple of (numpy.ndarray, bool)
        The op's value and whether it carries the leading batch axis.

    Raises
    ------
    BatchUnsupported
        If ``fn`` has no batch rule, or its operands mix batched and unbatched in a way
        the rule cannot express.
    """
    name = type(fn).__name__
    vals = [v for v, _ in ins]
    bflags = [b for _, b in ins]
    anyb = any(bflags)

    def A(i: int) -> np.ndarray:
        return np.asarray(vals[i], dtype=float)

    def _shift[T: (int, tuple[int, ...], list[int])](ax: T) -> T:
        """Re-express an axis index as seen from a batched operand: one extra leading axis.

        Negative indices already count from the end and so need no shift; shifting one
        would silently transpose the wrong pair of axes rather than fail.
        """
        if isinstance(ax, (tuple, list)):
            return type(ax)(_shift(a) for a in ax)
        return ax if ax < 0 else ax + 1

    if isinstance(fn, EinsumStmtOp):
        # Through the shared path cache rather than `np.einsum(..., optimize=…)` directly: the
        # contraction ORDER is unchanged (the cache replays the path numpy would have planned),
        # so results are bit-identical — it only stops re-planning on every batched call, which
        # this lane makes once per Stmt per batch.
        ops = tuple(A(i) for i in range(len(vals)))
        if not anyb:
            return _cached_einsum(fn.spec, ops, fn.optimize), False
        lhs, rhs = fn.spec.split("->")
        bspec = ",".join("..." + t for t in lhs.split(",")) + "->..." + rhs   # ellipsis-broadcast batch
        return _cached_einsum(bspec, ops, fn.optimize), True
    if isinstance(fn, AssertOp | IdentityOp):                # runtime check / capture-freeze → passthrough
        return vals[0], bflags[0]
    if isinstance(fn, ScaleOp):
        return fn.factor * A(0), bflags[0]
    if isinstance(fn, ScaleByOp):                            # s·x, s a RUNTIME scalar (the ScaleOp sibling)
        x, s = A(0), A(1)
        if not bflags[1]:                                    # scalar unbatched → plain broadcast
            return s * x, bflags[0]
        if not bflags[0]:                                    # (B,)·(*s) → (B, *s)
            return s.reshape((-1,) + (1,) * x.ndim) * x, True
        return s.reshape((s.shape[0],) + (1,) * (x.ndim - 1)) * x, True
    if isinstance(fn, AddOp):                                # left-fold over ALL n operands, not just two
        out = A(0)
        for i in range(1, len(vals)):
            out = out + A(i)                                 # (B,*s)+(*s) broadcasts on the trailing axes
        return out, anyb
    if isinstance(fn, PinvOp | InvOp | DetOp):               # numpy linalg batches over leading axes
        linalg = {PinvOp: np.linalg.pinv, InvOp: np.linalg.inv, DetOp: np.linalg.det}
        return linalg[type(fn)](A(0)), bflags[0]
    if isinstance(fn, SolveOp):
        return np.linalg.solve(A(0), A(1)), anyb
    if isinstance(fn, SqrtOp | AbsOp | SignOp):
        elementwise = {SqrtOp: np.sqrt, AbsOp: np.abs, SignOp: np.sign}
        return elementwise[type(fn)](A(0)), bflags[0]
    if isinstance(fn, AxisLenOp):                            # axis length is batch-invariant (a 0-d int)
        return np.asarray(int(A(0).shape[fn.axis + (1 if bflags[0] else 0)])), False
    if isinstance(fn, ReshapeOp):
        if bflags[0]:
            return A(0).reshape((A(0).shape[0],) + tuple(fn.shape)), True
        return A(0).reshape(fn.shape), False
    if isinstance(fn, FirstColsOp | LastColsOp):             # A[:, :rank] / A[:, rank:]; rank is a runtime int
        if bflags[1]:
            raise BatchUnsupported(f"{name}: a batched column RANK would give ragged results")
        rank = int(vals[1])
        sl = slice(None, rank) if isinstance(fn, FirstColsOp) else slice(rank, None)
        return (A(0)[:, :, sl], True) if bflags[0] else (A(0)[:, sl], False)
    if isinstance(fn, ProjectOp):                            # Pᵀ @ v(raveled)
        P, v = A(0), A(1)
        if bflags[0]:                                        # batched projector: one P per element
            vb = v.reshape(v.shape[0], -1) if bflags[1] else v.reshape(1, -1)
            return _cached_einsum("bij,bi->bj", (P, np.broadcast_to(vb, (P.shape[0],) + vb.shape[1:])),
                                  True), True
        if bflags[1]:
            return _cached_einsum("ij,bi->bj", (P, v.reshape(v.shape[0], -1)), True), True
        return P.T @ v.reshape(-1), False
    if isinstance(fn, EmbedOp):                              # P @ vsub, reshaped to fn.shape
        P, v = A(0), A(1)
        if bflags[0]:                                        # batched projector: one P per element
            vb = v if bflags[1] else np.broadcast_to(v, (P.shape[0],) + v.shape)
            out = _cached_einsum("bij,bj->bi", (P, vb), True)
            return (out.reshape((out.shape[0],) + tuple(fn.shape)) if fn.shape else out), True
        if bflags[1]:
            out = _cached_einsum("ij,bj->bi", (P, v), True)
            return (out.reshape((out.shape[0],) + tuple(fn.shape)) if fn.shape else out), True
        out = P @ v
        return (out.reshape(fn.shape) if fn.shape else out), False
    # --- pure SHAPE ops: the unbatched op with its axes shifted past the batch axis ---------
    # A batched operand carries one extra LEADING axis, so each of these is the unbatched op
    # re-expressed one axis over.
    #
    # Each rule mirrors its op's actual body, which is not always what the name suggests:
    # `TransposeOp` is a FULL-REVERSE transpose (`A.T`), not a last-two-axes swap, and
    # `ConcatOp` / `ColStackOp` FLATTEN each operand first. The obvious-looking rule would
    # give wrong numbers on ndim > 2 rather than fail — and a wrong batch rule is worse than
    # no rule, because the per-element fallback it replaces is correct.
    if isinstance(fn, TransposeOp):
        if not bflags[0]:
            return A(0).T, False
        x = A(0)                                             # (B, *s) — reverse the NON-batch axes
        return x.transpose((0, *range(x.ndim - 1, 0, -1))), True
    if isinstance(fn, MoveaxisOp):
        if not bflags[0]:
            return np.moveaxis(A(0), fn.source, fn.destination), False
        return np.moveaxis(A(0), _shift(fn.source), _shift(fn.destination)), True
    if isinstance(fn, ConcatOp):                             # flatten each, then concatenate
        if anyb and not all(bflags):
            raise BatchUnsupported("ConcatOp: mixed batched/unbatched operands")
        if not anyb:
            return np.concatenate([A(i).reshape(-1) for i in range(len(vals))]), False
        b = A(0).shape[0]
        return np.concatenate([A(i).reshape(b, -1) for i in range(len(vals))], axis=1), True
    if isinstance(fn, ColStackOp):                           # flatten each, stack AS COLUMNS
        if anyb and not all(bflags):
            raise BatchUnsupported("ColStackOp: mixed batched/unbatched operands")
        if not anyb:
            return np.stack([A(i).reshape(-1) for i in range(len(vals))], axis=1), False
        b = A(0).shape[0]
        return np.stack([A(i).reshape(b, -1) for i in range(len(vals))], axis=2), True
    if not anyb:                                             # unbatched operands → run the op as-is
        return np.asarray(fn(*vals)), False
    raise BatchUnsupported(f"no batch rule for op {name!r}")


def _cells_batched(cells: np.ndarray, batched_atoms: set[str]) -> bool:
    """Return whether any cell of ``cells`` mentions an atom bound to a batched slice."""
    arr = np.asarray(cells)
    for c in (arr.ravel() if arr.shape else [arr[()] if arr.shape == () else cells]):
        if isinstance(c, RationalFunction) and (set(c.gens) & batched_atoms):
            return True
    return False


def _eval_cells(cells: np.ndarray, atoms: Mapping[str, np.ndarray], B: int | None) -> np.ndarray:
    """Evaluate an ndarray of rational/float cells against ``atoms``.

    Parameters
    ----------
    cells
        Array of :class:`~polyarray.rational.RationalFunction` or float cells.
    atoms
        Atom name to its ``(B,)`` slice of input values.
    B
        Batch length, or ``None`` to evaluate the cells with no atom bindings at all.

    Returns
    -------
    numpy.ndarray
        ``cells.shape`` when ``B`` is ``None``, else ``(B, *cells.shape)``.
    """
    arr = np.asarray(cells)
    shape = arr.shape
    if B is None:
        out = np.empty(shape, dtype=float)
        for idx in (np.ndindex(*shape) if shape else [()]):
            c = arr[idx] if shape else arr[()]
            out[idx] = c.eval_numeric_fast({}) if isinstance(c, RationalFunction) else float(c)
        return out if shape else out[()]
    out = np.empty((B,) + shape, dtype=float)
    for idx in (np.ndindex(*shape) if shape else [()]):
        c = arr[idx] if shape else arr[()]
        out[(slice(None),) + idx] = c.eval_numeric_fast(atoms) if isinstance(c, RationalFunction) else float(c)
    return out


def batched_run(program: Program, values: Mapping[str, npt.ArrayLike]) -> np.ndarray:
    """Evaluate ``program`` over a whole batch in one pass.

    Parameters
    ----------
    program
        The program to evaluate.
    values
        Input name to a ``(B, *decl_shape)`` array carrying the batch on axis 0.

    Returns
    -------
    numpy.ndarray
        The ``result`` output, shaped ``(B, *out_shape)``.

    Raises
    ------
    BatchUnsupported
        If the program holds an op or construct with no batch rule; the caller should
        fall back to one :meth:`Program.run` per batch element.
    """
    B: int | None = None
    for inp in program.inputs:
        B = int(np.asarray(values[inp.name]).shape[0]); break
    atoms: dict[str, np.ndarray] = {}         # atom name → (B,) array
    bulk: dict[str, np.ndarray] = {}          # input/output name → (B, *) bulk tensor
    batched_atoms: set[str] = set()
    batched_bulk: set[str] = set()

    for inp in program.inputs:
        arr = np.asarray(values[inp.name], dtype=float)
        if is_dynamic(inp.shape):             # dynamic (DimAtom) input rides whole-tensor: (B, *)
            bulk[inp.name] = arr
            batched_bulk.add(inp.name)
            continue
        cells = program.input_arrays[inp.name].cells
        shp = tuple(int(d) for d in inp.shape)               # non-dynamic here → all concrete ints
        for idx in (np.ndindex(*shp) if shp else [()]):
            cell = cells[idx] if shp else cells[()]
            if isinstance(cell, RationalFunction):
                nm = cell.gens[0]
                atoms[nm] = arr[(slice(None),) + idx]
                batched_atoms.add(nm)

    def eval_sa(sa: SymArray) -> tuple[np.ndarray, bool]:
        if sa._bulk is not None:
            return bulk[sa._bulk.name], sa._bulk.name in batched_bulk
        if _cells_batched(sa.cells, batched_atoms):
            return _eval_cells(sa.cells, atoms, B), True
        return _eval_cells(sa.cells, atoms, None), False

    def resolve(ref: Ref) -> tuple[np.ndarray, bool]:
        if isinstance(ref, InputRef):
            if ref.name in bulk:
                return bulk[ref.name], ref.name in batched_bulk
            cells = program.input_arrays[ref.name].cells
            c = np.array(cells[ref.indices]) if ref.indices else cells
            if _cells_batched(c, batched_atoms):
                return _eval_cells(c, atoms, B), True
            return _eval_cells(c, atoms, None), False
        if isinstance(ref, OutputRef):
            v, b = eval_sa(program.statements[ref.stmt_idx].out[ref.out_idx])
            if ref.indices:
                v = v[(slice(None),) + ref.indices] if b else v[ref.indices]
            return v, b
        if isinstance(ref, Const):
            return np.asarray(ref.value), False
        if isinstance(ref, RationalRef):
            if set(ref.rf.gens) & batched_atoms:
                return _eval_cells(np.array(ref.rf), atoms, B), True
            return np.asarray(ref.rf.eval_numeric_fast({})), False
        if isinstance(ref, SymArrayRef):
            if ref._bulk is not None:
                return bulk[ref._bulk.name], ref._bulk.name in batched_bulk
            if _cells_batched(ref.cells, batched_atoms):
                return _eval_cells(ref.cells, atoms, B), True
            return _eval_cells(ref.cells, atoms, None), False
        if isinstance(ref, IntAtomRef):
            return np.asarray(int(values[ref.name])), False
        raise BatchUnsupported(f"unknown ref kind {type(ref).__name__}")

    for stmt in program.statements:
        if stmt.fn is None:
            continue
        if isinstance(stmt.fn, Program):
            raise BatchUnsupported("sub-Program statement")
        ins = [resolve(r) for r in stmt.in_]
        val, b = _apply(stmt.fn, ins)
        outs = list(val) if isinstance(val, tuple) else [val]
        for k, bound in enumerate(stmt.out):
            v = np.asarray(outs[k], dtype=float)
            if bound._bulk is not None:
                bulk[bound._bulk.name] = v
                if b:
                    batched_bulk.add(bound._bulk.name)
            else:
                for idx in (np.ndindex(*bound.cells.shape) if bound.cells.shape else [()]):
                    cell = bound.cells[idx] if bound.cells.shape else bound.cells[()]
                    if isinstance(cell, RationalFunction):
                        nm = cell.gens[0]
                        atoms[nm] = v[(slice(None),) + idx] if b else v[idx]
                        if b:
                            batched_atoms.add(nm)

    result, _ = eval_sa(program.outputs["result"])
    return result
