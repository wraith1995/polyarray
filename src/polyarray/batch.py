"""Batched execution of a :class:`~polyarray.ir.Program` over a leading batch axis.

``batched_run(program, values)`` evaluates ``program`` for a whole BATCH of inputs at once — each entry of
``values`` carries a leading batch axis of length ``B`` — reproducing ``[program.run(values[b]) for b in
range(B)]`` in ONE vectorized pass, but without the Python-per-call / per-Stmt interpreter overhead that
dominates when a program is tiny and run many times (a per-quadrature-point loop).

Two lanes, mirroring :meth:`Program.run`:

* **Cell lane** — a program's symbolic ``RationalFunction`` cells are polynomials over per-element input
  atoms.  The batch axis stays IMPLICIT: each atom is bound to its ``(B,)`` slice and ``eval_numeric_fast``
  broadcasts, so a cell of declared shape ``s`` evaluates to ``(B, *s)``.  (Shapes are NOT rewritten — that
  would rename atoms and break the cells.)
* **Stmt lane** — each op is applied with the batch as leading axis 0.  Ops are dispatched by CLASS NAME so
  this module has no dependency on the front end (grassmann) that defines some of them; einsum ellipsis-
  batches, ``numpy`` linalg batches over leading axes, axis ops shift by one.

Agreement with the per-element loop is EXACT for the structural and elementwise ops (reshape, scale,
add, slice, axis length — verified ``assert_array_equal`` in ``tests/test_batch.py``), but NOT exact in
general.  ``EinsumStmtOp`` batches by rewriting the spec with an ellipsis and passes ``optimize=``, so
numpy may choose a different contraction ORDER than the unbatched call; ``np.linalg`` likewise takes a
stacked path rather than a per-matrix one.  Both re-associate floating-point arithmetic.

Measured on a high-degree interpolation residual: in 2D max|Δ| = 0 exactly; in 3D max|Δ| =
1.0e-13 on values of magnitude ~1.4e2, i.e. ~1 ulp relative — within tolerance. An earlier
"byte-identical, max|Δ|=0" claim here was measured when most programs silently fell back to the
per-element loop, so the batched path was barely exercised; do not restore it.

Raises :class:`NotImplementedError` for any op without a batch rule (or a sub-Program / dynamic
construct it does not handle) so the caller can fall back to the per-element loop.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .ir import (Const, InputRef, IntAtomRef, OutputRef, Program, RationalRef, SymArrayRef,
                 _cached_einsum, is_dynamic)
from .rational import RationalFunction

__all__ = ["batched_run", "BatchUnsupported"]


class BatchUnsupported(NotImplementedError):
    """An op / construct in the program has no batch rule — fall back to the per-element loop."""


def _batched_einsum_spec(spec: str) -> str:
    """Prepend a fresh batch index to every operand and the output (mirrors ``pyab._batched_einsum_spec``)."""
    lhs, _, rhs = spec.partition("->")
    used = set(spec) - set(",->")
    batch = next((ch for ch in "BNZYXWVUTSRQPOMLKJ" if ch not in used), None)
    if batch is None:
        raise BatchUnsupported(f"no free einsum index to batch {spec!r}")
    ins = ",".join(batch + t.strip() for t in lhs.split(","))
    return f"{ins}->{batch}{rhs.strip()}"


def _apply(fn: Any, ins: list[tuple[np.ndarray, bool]]) -> tuple[np.ndarray, bool]:
    """Apply op ``fn`` to operands ``ins`` = ``[(value, is_batched), …]``; return ``(value, is_batched)``."""
    name = type(fn).__name__
    vals = [v for v, _ in ins]
    bflags = [b for _, b in ins]
    anyb = any(bflags)

    def A(i: int) -> np.ndarray:
        return np.asarray(vals[i], dtype=float)

    def _shift(ax):
        """An axis index as seen from a BATCHED operand: one extra leading axis.

        Negative indices already count from the end and so need no shift — getting that wrong
        would silently transpose the wrong pair rather than fail."""
        if isinstance(ax, (tuple, list)):
            return type(ax)(_shift(a) for a in ax)
        return ax if ax < 0 else ax + 1

    if name == "EinsumStmtOp":
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
    if name in ("AssertOp", "IdentityOp"):                   # runtime check / capture-freeze → passthrough
        return vals[0], bflags[0]
    if name == "ScaleOp":
        return fn.factor * A(0), bflags[0]
    if name == "ScaleByOp":                                  # s·x, s a RUNTIME scalar (the ScaleOp sibling)
        x, s = A(0), A(1)
        if not bflags[1]:                                    # scalar unbatched → plain broadcast
            return s * x, bflags[0]
        if not bflags[0]:                                    # (B,)·(*s) → (B, *s)
            return s.reshape((-1,) + (1,) * x.ndim) * x, True
        return s.reshape((s.shape[0],) + (1,) * (x.ndim - 1)) * x, True
    if name == "AddOp":                                      # left-fold over ALL n operands, not just two
        out = A(0)
        for i in range(1, len(vals)):
            out = out + A(i)                                 # (B,*s)+(*s) broadcasts on the trailing axes
        return out, anyb
    if name in ("PinvOp", "InvOp", "DetOp"):                 # numpy linalg batches over leading axes
        return getattr(np.linalg, {"PinvOp": "pinv", "InvOp": "inv", "DetOp": "det"}[name])(A(0)), bflags[0]
    if name == "SolveOp":
        return np.linalg.solve(A(0), A(1)), anyb
    if name in ("SqrtOp", "AbsOp", "SignOp"):
        return {"SqrtOp": np.sqrt, "AbsOp": np.abs, "SignOp": np.sign}[name](A(0)), bflags[0]
    if name == "AxisLenOp":                                  # axis length is batch-invariant (a 0-d int)
        return np.asarray(int(A(0).shape[fn.axis + (1 if bflags[0] else 0)])), False
    if name == "ReshapeOp":
        if bflags[0]:
            return A(0).reshape((A(0).shape[0],) + tuple(fn.shape)), True
        return A(0).reshape(fn.shape), False
    if name in ("FirstColsOp", "LastColsOp"):                # A[:, :rank] / A[:, rank:]; rank is a runtime int
        if bflags[1]:
            raise BatchUnsupported(f"{name}: a batched column RANK would give ragged results")
        rank = int(vals[1])
        sl = slice(None, rank) if name == "FirstColsOp" else slice(rank, None)
        return (A(0)[:, :, sl], True) if bflags[0] else (A(0)[:, sl], False)
    if name == "ProjectOp":                                  # Pᵀ @ v(raveled)
        P, v = A(0), A(1)
        if bflags[0]:                                        # batched projector: one P per element
            vb = v.reshape(v.shape[0], -1) if bflags[1] else v.reshape(1, -1)
            return _cached_einsum("bij,bi->bj", (P, np.broadcast_to(vb, (P.shape[0],) + vb.shape[1:])),
                                  True), True
        if bflags[1]:
            return _cached_einsum("ij,bi->bj", (P, v.reshape(v.shape[0], -1)), True), True
        return P.T @ v.reshape(-1), False
    if name == "EmbedOp":                                    # P @ vsub, reshaped to fn.shape
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
    # re-expressed one axis over. They had no rule at all, so any program containing one fell
    # back to the per-element loop entirely — correct, and slow in a way nothing reports.
    #
    # Each rule below mirrors its op's ACTUAL body, which is not always what the name suggests:
    # `TransposeOp` is a FULL-REVERSE transpose (`A.T`), not a last-two-axes swap, and
    # `ConcatOp` / `ColStackOp` FLATTEN each operand first. Writing the obvious-looking rule
    # instead would produce wrong numbers on ndim > 2 rather than a loud failure — and a wrong
    # batch rule is worse than none, because the fallback it replaces is correct.
    if name == "TransposeOp":
        if not bflags[0]:
            return A(0).T, False
        x = A(0)                                             # (B, *s) — reverse the NON-batch axes
        return x.transpose((0, *range(x.ndim - 1, 0, -1))), True
    if name == "MoveaxisOp":
        if not bflags[0]:
            return np.moveaxis(A(0), fn.source, fn.destination), False
        return np.moveaxis(A(0), _shift(fn.source), _shift(fn.destination)), True
    if name == "ConcatOp":                                   # flatten each, then concatenate
        if anyb and not all(bflags):
            raise BatchUnsupported("ConcatOp: mixed batched/unbatched operands")
        if not anyb:
            return np.concatenate([A(i).reshape(-1) for i in range(len(vals))]), False
        b = A(0).shape[0]
        return np.concatenate([A(i).reshape(b, -1) for i in range(len(vals))], axis=1), True
    if name == "ColStackOp":                                 # flatten each, stack AS COLUMNS
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
    arr = np.asarray(cells)
    for c in (arr.ravel() if arr.shape else [arr[()] if arr.shape == () else cells]):
        if isinstance(c, RationalFunction) and (set(c.gens) & batched_atoms):
            return True
    return False


def _eval_cells(cells: np.ndarray, atoms: Mapping[str, Any], B: int | None) -> np.ndarray:
    """Evaluate an ndarray of RF/float cells. With ``B`` (batched atom bindings) → ``(B, *cells.shape)``."""
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


def batched_run(program: Program, values: Mapping[str, Any]) -> np.ndarray:
    """Evaluate ``program`` for a batch: each ``values[name]`` has a leading batch axis ``(B, *decl_shape)``.
    Returns the ``result`` output as ``(B, *out_shape)``. Byte-identical to looping ``program.run`` per
    batch element. Raises :class:`BatchUnsupported` on any op/construct without a batch rule."""
    B: int | None = None
    for inp in program.inputs:
        B = int(np.asarray(values[inp.name]).shape[0]); break
    atoms: dict[str, Any] = {}                # atom name → (B,) array
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

    def eval_sa(sa: Any) -> tuple[np.ndarray, bool]:
        if sa._bulk is not None:
            return bulk[sa._bulk.name], sa._bulk.name in batched_bulk
        if _cells_batched(sa.cells, batched_atoms):
            return _eval_cells(sa.cells, atoms, B), True
        return _eval_cells(sa.cells, atoms, None), False

    def resolve(ref: Any) -> tuple[np.ndarray, bool]:
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
