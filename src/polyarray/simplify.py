"""Post-build partial evaluation of a :class:`Program` — the ``simplify`` pass.

The pass re-interprets a program against a *partial* numeric environment and returns a fresh
program computing the same outputs with every build-time-known value already substituted in.
``fold_numeric`` and ``bind_inputs`` are its numeric-propagation floor; ``specialize`` layers
optional symbolic substitution and post-build budget moderation on top.

The floor runs against a growing ``known: dict[atom_name -> float]`` of every generator whose
value is determined at build time, seeded from ``bind`` (empty for a bare ``fold_numeric``)::

    for each Stmt, in order:
        inputs all resolve numeric?  ── yes ─→ execute it, its outputs enter
                                               ``known``, drop the Stmt
                                    ── no  ─→ keep it, fold ``known`` into its refs
    then: fold ``known`` into every program output cell
          drop every input a concrete ``bind`` replaced

Folding a statement cascades: its outputs unlock downstream statements whose inputs then
resolve numeric. A fully-bound output cell becomes a float; a partially-bound cell becomes a
smaller rational function over the leftover generators, leaving residual symbols in place.

The pass never mutates shared state. ``Program.copy`` shares cell arrays and ref tuples, so it
builds *fresh* folded cells and refs rather than rewriting in place. Exactness holds
throughout: ``fold_numeric(p, bind=b).run(rest)`` equals ``p.run({**b, **rest})``.

It is conservative by construction. Anything not confidently foldable — a bulk or dynamic
output, a control-flow op — stays symbolic, so the worst case degrades to ``copy()``. A
partially-numeric sub-program or ``CallOp`` statement is instead *descended* into: its body is
recursively specialized with the numeric operands bound, shrinking the statement to its
still-symbolic operands.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .ir import (
    BulkOut,
    CallOp,
    Cell,
    Const,
    DimSource,
    DimAtom,
    InputRef,
    IntAtomRef,
    OutputRef,
    Program,
    RationalRef,
    Ref,
    Stmt,
    StmtOp,
    SymArray,
    SymArrayRef,
    VmapClosure,
    WhileOp,
    is_dynamic,
)
from .budget import SimplifyBudget, _apply_budget
from .rational import RationalFunction

# Ops we do not execute at build time.  ``WhileOp`` could loop; ``CallOp`` and
# raw sub-Programs ARE foldable (executed when every operand is numeric — see
# ``_exec_fn``).  ``SwitchOp`` is fine too: it only resolves once its IntAtom
# selector is bound, otherwise its inputs stay symbolic and it survives.
_SKIP_OPS = (WhileOp,)

# Recursion ceiling for P6 partial descent into nested sub-Program / CallOp
# bodies.  Bodies are acyclic in practice; the ``seen`` set already breaks any
# cycle, so this is a belt-and-braces cap against pathological nesting.
_MAX_DESCENT_DEPTH = 32


# ---------------------------------------------------------------------------
# Predicates / small helpers
# ---------------------------------------------------------------------------

def _simple_stmt(stmt: Stmt) -> bool:
    """Report whether a statement may be executed at build time.

    A callable ``fn``, a typed op or a sub-program qualifies; a loop does not.

    A dynamic (runtime ``DimAtom``-sized) bulk output is foldable too: with all-numeric
    inputs the statement is a value-invariant map — a constant SVD, GSVD, QR or pinv whose
    numerical rank is statically knowable — so it is executed, its concrete output shape
    read, and the dimension atoms it created resolved. With any non-numeric input the fold loop
    skips it and the dynamic dimension survives unchanged.
    """
    if stmt.fn is None:
        return False
    if isinstance(stmt.fn, _SKIP_OPS):
        return False
    return True


def _exec_fn(fn: StmtOp, resolved: list[np.ndarray]) -> list[np.ndarray]:
    """Execute a statement's ``fn`` on concrete numeric operands.

    Mirrors ``Program._run_stmt``'s dispatch: a raw sub-program runs via ``.run``, and
    everything else is called.
    """
    if isinstance(fn, Program):
        value_map = {inp.name: np.asarray(v) for inp, v in zip(fn.inputs, resolved)}
        return list(fn.run(value_map).values())
    results = fn(*resolved)
    return list(results) if isinstance(results, tuple) else [results]


def _try_eval_ref(
    prog: Program, ref: Ref, stmt_idx: int, known: Mapping[str, float],
) -> np.ndarray | None:
    """Resolve ``ref`` to a concrete float array using ``known``.

    Returns ``None`` if any generator it needs is still symbolic.

    A ``dict`` ``known`` is passed straight through with no defensive copy: every branch of
    :meth:`Program._resolve_ref` only reads its bindings. Copying instead costs one pass
    over ``known`` per ref of every surviving statement, and ``known`` reaches hundreds of
    thousands of atoms on a high-degree build. A non-``dict`` mapping is still materialised,
    so the signature's contract is unchanged.
    """
    b: dict[str, float] = known if isinstance(known, dict) else dict(known)
    try:
        return np.asarray(prog._resolve_ref(ref, stmt_idx, b), dtype=float)
    except Exception:
        return None


def _cells_touch_known(cells: np.ndarray, known: Mapping[str, float]) -> bool:
    """Report whether folding ``known`` into ``cells`` would substitute anything.

    True when some :class:`RationalFunction` cell references a generator in ``known``; when
    False the fold is a structural no-op on these cells.
    """
    if not known:
        return False
    for c in cells.reshape(-1):
        if isinstance(c, RationalFunction) and any(g in known for g in c.gens):
            return True
    return False


def _fold_cells(cells: np.ndarray, known: Mapping[str, float]) -> np.ndarray:
    """Fold ``known`` into an ndarray of cells by partially evaluating each cell.

    A :class:`RationalFunction` cell becomes a float once all its generators are bound and a
    smaller rational function over the leftover generators otherwise.

    Returns
    -------
    numpy.ndarray
        A float array when every cell becomes numeric, else an object array of floats and
        smaller rational functions holding the residual symbols.

    Notes
    -----
    The fold is transparent to structure. When no cell references a generator in ``known`` it
    substitutes nothing, so the original ``cells`` object is returned as-is rather than a fresh
    copy, keeping ``id(cells)`` stable. A downstream identity-based structural read depends on
    that stability: the degree walker (``polyarray.program_degree``) links a statement's
    producer to its consumer by ``id(ref._cells) == id(out._cells)`` and, on a miss, falls back
    to scoring the cells' generators by name — a fallback that can silently drop a generator's
    degree and mis-score the result. A fold that substitutes nothing must therefore leave the
    cell array's identity intact.
    """
    if cells.dtype.kind == "f":
        return cells.copy()
    if not _cells_touch_known(cells, known):
        return cells
    out = np.empty(cells.shape, dtype=object)
    flat_in = cells.reshape(-1)
    flat_out = out.reshape(-1)
    all_numeric = True
    for i, c in enumerate(flat_in):
        if isinstance(c, RationalFunction):
            v = c.eval(known)  # float if all gens in known, else leftover-ring RF
            flat_out[i] = v
            if isinstance(v, RationalFunction):
                all_numeric = False
        elif isinstance(c, (int, float)):
            flat_out[i] = float(c)
        else:
            flat_out[i] = c
            all_numeric = False
    return out.astype(float) if all_numeric else out


def _numify_constant_cells(cells: np.ndarray) -> np.ndarray:
    """Reduce constant :class:`RationalFunction` cells, those with no live generators, to plain floats.

    A structural fold that collapsed a cell to a constant should present it as numeric. A
    fully-constant array then carries float dtype, so :meth:`SymArray.evaluate` reads it
    directly (``dtype.kind == 'f'``) without the program's now-unused input bindings, matching
    the ``partial_eval_numeric`` intent that unused inputs simply go unread. A cell that still
    carries a live generator stays a :class:`RationalFunction`, so a cell-dependent array keeps
    object dtype and ``evaluate({})`` still raises — which a downstream constancy check reads as
    "does not fold to a constant".
    """
    if cells.dtype.kind == "f":
        return cells
    out = np.empty(cells.shape, dtype=object)
    flat_in, flat_out = cells.reshape(-1), out.reshape(-1)
    all_numeric = True
    for i, c in enumerate(flat_in):
        if isinstance(c, RationalFunction):
            v = c.eval({})                       # float iff the RF has no live generators
            flat_out[i] = v
            if isinstance(v, RationalFunction):
                all_numeric = False
        elif isinstance(c, (int, float)):
            flat_out[i] = float(c)
        else:
            flat_out[i] = c
            all_numeric = False
    return out.astype(float) if all_numeric else out


def _fold_symarray(
    sa: SymArray, known: Mapping[str, float | np.ndarray], program: Program, name: str | None,
) -> SymArray:
    if sa._bulk is not None:
        # A folded bulk producer recorded its whole tensor under the bulk name;
        # materialise the output as a numeric SymArray.  Otherwise the handle is
        # still symbolic — keep it (do NOT touch ``.cells``, which would unpack).
        val = known.get(sa._bulk.name)
        if val is not None:
            return SymArray(np.asarray(val, dtype=float), program=program, name=name)
        return sa
    return SymArray(_fold_cells(np.asarray(sa.cells), known), program=program, name=name)


def _fold_ref(
    prog: Program, ref: Ref, stmt_idx: int,
    known: Mapping[str, float], idx_map: Mapping[int, int],
) -> Ref:
    """Rewrite a surviving statement's input ref against ``known``.

    The ref becomes numeric where its value is determined, and a symbolically folded version
    otherwise, with any ``OutputRef`` statement index remapped onto the surviving statement
    list through ``idx_map``.
    """
    if isinstance(ref, IntAtomRef):
        return ref
    num = _try_eval_ref(prog, ref, stmt_idx, known)
    if num is not None:
        return SymArrayRef(np.asarray(num, dtype=float))
    if isinstance(ref, SymArrayRef):
        if ref._bulk is not None:
            return ref
        return SymArrayRef(_fold_cells(np.asarray(ref.cells), known))
    if isinstance(ref, OutputRef):
        return OutputRef(idx_map[ref.stmt_idx], ref.out_idx, ref.indices)
    if isinstance(ref, RationalRef):
        v = ref.rf.eval(known)
        return RationalRef(v) if isinstance(v, RationalFunction) \
            else SymArrayRef(np.asarray(float(v)))
    return ref  # InputRef over an unbound input, Const — unchanged


def _seed_bind(
    prog: Program, bind: Mapping[str, npt.ArrayLike],
) -> tuple[dict[str, float], set[str]]:
    """Seed ``known`` from the concrete arrays in ``bind``.

    Each bound input's per-cell generator atom takes the matching array entry as its value.

    Returns
    -------
    tuple
        The seeded ``known`` map and the set of bound input names to drop.

    Raises
    ------
    NotImplementedError
        For a bulk or dynamic input, which cannot be seeded cell by cell.
    ValueError
        When a bound array's shape does not match the input's cell shape.
    """
    known: dict[str, float] = {}
    dropped: set[str] = set()
    for name, val in bind.items():
        sa = prog.input_arrays[name]
        if sa._bulk is not None:
            raise NotImplementedError(f"bind of bulk/dynamic input {name!r} unsupported")
        cells = np.asarray(sa.cells)
        arr = np.asarray(val, dtype=float)
        if tuple(arr.shape) != tuple(cells.shape):
            raise ValueError(
                f"bind {name!r}: expected shape {cells.shape}, got {arr.shape}"
            )
        shape = cells.shape
        for idx in (np.ndindex(*shape) if shape else [()]):
            cell = cells[idx] if shape else cells[()]
            if isinstance(cell, RationalFunction):
                known[cell.gens[0]] = float(arr[idx] if shape else arr)
        dropped.add(name)
    return known, dropped


def _resolve_stmt_dims(
    stmt: Stmt, stmt_idx: int, outs: list[np.ndarray], dim_subst: dict[DimSource, int],
) -> None:
    """Resolve the dimension atoms a folded statement's concrete outputs size.

    The build-time mirror of :meth:`Program._bind_output`: a not-yet-bound ``DimAtom`` in a
    bulk output's declared shape binds from the realised array's actual axis size, keyed by
    the atom's hashable ``source``, so :func:`_substitute_dims` can substitute it into every
    remaining shape.

    The ``source`` is a logical forward link — it may name a producing statement whose own
    output does not carry the axis, as with an SVD rank output whose dimension physically
    sizes a downstream column take — so binding follows where the axis actually appears,
    exactly as at run time. A dimension already resolved by an earlier fold is left
    untouched: first realised output wins.
    """
    for k, bound in enumerate(stmt.out):
        if bound._bulk is None or not is_dynamic(bound._bulk.shape):
            continue
        arr = np.asarray(outs[k])
        for axis, d in enumerate(bound._bulk.shape):
            if isinstance(d, DimAtom) and d.source not in dim_subst:
                if axis >= arr.ndim:
                    raise ValueError("folded dynamic output: dim axis out of range")
                dim_subst[d.source] = int(arr.shape[axis])


def _record_known(
    stmt: Stmt, outs: list[np.ndarray], known: dict[str, float | np.ndarray],
    dim_subst: Mapping[DimSource, int] | None = None,
) -> None:
    """Record a folded statement's numeric outputs into ``known``.

    A bulk output records its whole tensor under the bulk handle name; a per-cell output
    records each cell's atom. A dynamic bulk output's declared shape is resolved against
    ``dim_subst`` before its produced shape is validated.

    Raises
    ------
    ValueError
        On a shape mismatch, so the caller can discard a bad fold.
    KeyError
        When a ``DimAtom`` is still unresolved; the caller treats this as a failed fold and
        keeps the statement symbolic.
    """
    for k, bound in enumerate(stmt.out):
        arr = np.asarray(outs[k], dtype=float)
        if bound._bulk is not None:
            if is_dynamic(bound._bulk.shape):
                if dim_subst is None:
                    raise ValueError("dynamic bulk output without resolved dims")
                expected = tuple(
                    int(dim_subst[d.source]) if isinstance(d, DimAtom) else int(d)
                    for d in bound._bulk.shape
                )
            else:
                expected = tuple(bound._bulk.shape)
            if tuple(arr.shape) != expected:
                raise ValueError("bulk fold output shape mismatch")
            known[bound._bulk.name] = arr
            continue
        cells = np.asarray(bound.cells)
        if tuple(arr.shape) != tuple(cells.shape):
            raise ValueError("fold output shape mismatch")
        shape = cells.shape
        for idx in (np.ndindex(*shape) if shape else [()]):
            cell = cells[idx] if shape else cells[()]
            if isinstance(cell, RationalFunction):
                known[cell.gens[0]] = float(arr[idx] if shape else arr)


# ---------------------------------------------------------------------------
# DimAtom → concrete-int substitution (the one new uniform helper)
# ---------------------------------------------------------------------------

def _subst_shape(
    shape: tuple[Any, ...], dim_subst: Mapping[DimSource, int],
) -> tuple[tuple[Any, ...], bool]:
    """Replace every resolved ``DimAtom`` in ``shape`` with its concrete int.

    Returns ``(new_shape, changed)``.  A ``DimAtom`` not in ``dim_subst`` (from
    a Stmt that did *not* fold — genuinely data-dependent) passes through
    untouched; concrete entries are unchanged.  ``changed`` lets callers avoid
    rebuilding shared frozen objects when nothing was substituted.
    """
    if not is_dynamic(shape):
        return shape, False
    out: list[Any] = []
    changed = False
    for d in shape:
        if isinstance(d, DimAtom) and d.source in dim_subst:
            out.append(int(dim_subst[d.source]))
            changed = True
        else:
            out.append(d)
    return tuple(out), changed


def _subst_bulk_symarray(
    sa: SymArray, dim_subst: Mapping[DimSource, int],
) -> SymArray:
    """Return a fresh SymArray with its bulk shape's resolved dimensions substituted.

    Returns ``sa`` unchanged when there is nothing to substitute. Never mutates in place,
    since the existing bulk handle may be shared.
    """
    if sa._bulk is None:
        return sa
    new_shape, changed = _subst_shape(sa._bulk.shape, dim_subst)
    if not changed:
        return sa
    out = SymArray(sa._cells, program=sa.program, name=sa.name)
    out._bulk = BulkOut(name=sa._bulk.name, shape=new_shape)
    return out


def _subst_ref_dims(
    ref: Ref, dim_subst: Mapping[DimSource, int],
) -> Ref:
    """Substitute any resolved dimension in a surviving input ref's bulk handle."""
    if isinstance(ref, SymArrayRef) and ref._bulk is not None:
        new_shape, changed = _subst_shape(ref._bulk.shape, dim_subst)
        if changed:
            out = SymArrayRef(ref._cells)
            out._bulk = BulkOut(name=ref._bulk.name, shape=new_shape)
            return out
    return ref


def _substitute_dims(
    program: Program, dim_subst: Mapping[DimSource, int],
) -> None:
    """Substitute every resolved ``DimAtom`` across ``program``'s remaining shapes, in place.

    Covers statement input refs, statement output SymArrays and program outputs. Operating
    in place is safe because the caller passes a freshly copied program, so no shared
    upstream state is touched.

    This is what makes a folded constant SVD or GSVD eliminate its dimension uniformly: once
    the rank is resolved from the concrete output shape, every downstream shape that was
    sized by it becomes static and consistent.

    Program inputs are deliberately left untouched. A dynamic bulk input binds as a whole
    tensor only through ``build_runtime_bindings``' dynamic-input path, which is gated on
    the input's shape *being* dynamic; resolving its dimension to a static int would drop it
    from that path and its value would never bind. A statement-sourced dimension in an input
    shape pointing at a folded-away statement is harmless, because the fed array binds
    directly and the input is validated axis by axis.
    """
    if not dim_subst:
        return
    # Statements: input refs + output SymArrays.
    for idx, stmt in enumerate(program.statements):
        new_in = tuple(_subst_ref_dims(r, dim_subst) for r in stmt.in_)
        new_out = tuple(_subst_bulk_symarray(o, dim_subst) for o in stmt.out)
        if any(a is not b for a, b in zip(new_in, stmt.in_)) or any(
            a is not b for a, b in zip(new_out, stmt.out)
        ):
            program.statements[idx] = Stmt(
                fn=stmt.fn, in_=new_in, out=new_out, note=stmt.note,
                provenance=stmt.provenance, inline=stmt.inline,
            )
    # Program outputs.
    program.outputs = {
        k: _subst_bulk_symarray(sa, dim_subst) for k, sa in program.outputs.items()
    }


# ---------------------------------------------------------------------------
# P6: partial descent into nested sub-Program / CallOp bodies
# ---------------------------------------------------------------------------

def _descent_body(fn: StmtOp) -> tuple[Program, Callable[[Program], StmtOp]] | None:
    """Return a directly-positional body program plus a re-wrapper, or ``None``.

    A statement is descendable only where its operands map to the body's inputs by position
    with matching shapes and swapping ``fn`` for the specialized body is exactly equivalent.
    Two cases qualify: a raw sub-:class:`Program`, which :meth:`Program._run_stmt` runs via
    ``zip(inputs, ops)``, and a :class:`CallOp` wrapping a program, whose ``_invoke`` maps
    operands by position too. Each returns the body together with a rewrapper that restores the
    original wrapping.

    A genuine ``vmap`` closure is not descendable here: its operands carry a batch axis (and
    ``in_axes=None`` broadcasts), so they do not match the body inputs by shape, and replacing
    the closure with the bare body would drop the batching. Such statements — along with
    ``WhileOp`` and opaque callables — return ``None`` and stay symbolic, the conservative
    degrade.
    """
    if isinstance(fn, Program):
        return fn, (lambda body: body)
    if isinstance(fn, CallOp) and isinstance(fn.fn, Program):
        return fn.fn, (lambda body: CallOp(fn=body))
    return None


def _try_descend(
    prog: Program,
    s: Stmt,
    stmt_idx: int,
    known: Mapping[str, float],
    idx_map: Mapping[int, int],
    depth: int,
    seen: frozenset[int],
) -> tuple[Any, tuple[Any, ...]] | None:
    """Attempt partial descent on a surviving statement ``s``.

    Returns ``(new_fn, new_in)`` when ``s`` has a descendable body, some — not all, not none —
    of its operands resolve numeric, and the specialized body drops exactly those operands'
    inputs. Returns ``None`` in every other case, falling back to the plain symbolic ref-fold.
    """
    info = _descent_body(s.fn)
    if info is None:
        return None
    body, rewrap = info
    if id(body) in seen:
        return None  # cycle guard — a body reaching itself / an ancestor
    if len(s.in_) != len(body.inputs):
        return None  # arity must line up to bind operands -> body inputs
    numeric_bind: dict[str, Any] = {}
    symbolic_idx: list[int] = []
    for k, r in enumerate(s.in_):
        v = _try_eval_ref(prog, r, stmt_idx, known)
        if v is None:
            symbolic_idx.append(k)
        else:
            numeric_bind[body.inputs[k].name] = v
    if not numeric_bind or not symbolic_idx:
        # none numeric -> normal symbolic path; all numeric -> the fold loop
        # already handled it (or it raised, in which case keep it whole).
        return None
    try:
        spec_body = _specialize(body, numeric_bind, depth + 1, seen | {id(body)})
    except Exception:
        return None  # bulk/dynamic bind, shape mismatch, ... -> degrade
    remaining = [inp.name for inp in spec_body.inputs]
    kept = [body.inputs[k].name for k in symbolic_idx]
    if remaining != kept:
        return None  # specialized body didn't drop exactly the bound inputs
    new_fn = rewrap(spec_body)
    new_in = tuple(
        _fold_ref(prog, s.in_[k], stmt_idx, known, idx_map) for k in symbolic_idx
    )
    return new_fn, new_in


# ---------------------------------------------------------------------------
# P6b: constant-fold INSIDE a vmap body (keep the batching)
# ---------------------------------------------------------------------------

def _is_rebuildable_vmap(fn: StmtOp) -> bool:
    """Report whether ``fn`` is a vmap closure this pass can rebuild.

    Rebuilding needs all three attributes :func:`~polyarray.ir.vmap` sets: the body and both
    axis tuples. A front-end wrapper may expose only the body, for introspection; such a
    closure is skipped rather than treated as rebuildable.
    """
    return all(hasattr(fn, a) for a in ("_vmap_body", "_in_axes", "_out_axes"))


def _vmap_closure_of(fn: StmtOp) -> tuple[StmtOp, Callable[[StmtOp], StmtOp]] | None:
    """Return ``(closure, rewrap)`` when ``fn`` is a rebuildable vmap closure.

    Accepts a bare closure or a :class:`CallOp` wrapping one. ``rewrap`` rebuilds an
    equivalent ``fn`` from a fresh closure. A wrapper missing the axis tuples returns
    ``None`` and degrades to no fold.
    """
    if _is_rebuildable_vmap(fn):
        return fn, (lambda c: c)
    if isinstance(fn, CallOp) and _is_rebuildable_vmap(fn.fn):
        return fn.fn, (lambda c: CallOp(fn=c))
    return None


def _drop_unread_inputs(prog: Program) -> Program:
    """Return ``prog`` with every input that no statement and no output references removed.

    An unread input can be the only reason an enclosing statement does not fold: a closure that
    declares an input it never reads has a non-numeric operand, so the whole statement stays
    symbolic even though the input is unused. Dropping it is value-preserving by definition,
    since nothing reads it.

    A reference is decided over atoms — :func:`symarray_atoms` across every statement operand
    and every output — rather than by a syntactic scan, so an input reached through a folded
    cell still counts as read.
    """
    if not prog.inputs:
        return prog
    used: set[str] = set()
    for s in prog.statements:
        for r in s.in_:
            used |= {str(a) for a in symarray_atoms(r)}
    for sa in prog.outputs.values():
        used |= {str(a) for a in symarray_atoms(sa)}
    dead = []
    for inp in prog.inputs:
        arr = prog.input_arrays.get(inp.name)
        if arr is None:
            continue
        atoms = {str(a) for a in symarray_atoms(arr)}
        if atoms and not (atoms & used):
            dead.append(inp.name)
    if not dead:
        return prog
    out = prog.copy()
    out.inputs = tuple(i for i in out.inputs if i.name not in dead)
    for nm in dead:
        out.input_arrays.pop(nm, None)
    return out


def _fold_vmap_body(
    fn: StmtOp, depth: int, seen: frozenset[int],
    operand_values: list[np.ndarray | None] | None = None,
) -> tuple[StmtOp, list[bool] | None]:
    """Constant-fold the numeric subcomputations inside a vmap body, keeping the batching.

    :func:`_descent_body` refuses to descend a vmap closure, because swapping the closure for
    the bare body would drop the per-point batching. The body's internal statements whose
    inputs are all build-time-numeric — a QR or SVD on a fixed constant operand, independent of
    the per-point vmap arguments — can still fold to constants without touching the batching.
    This recurses the floor-fold into the body and, if it folded anything away, rewraps the
    folded body in an equivalent vmap closure. It falls back to ``fn`` unchanged whenever
    nothing folds or anything looks off, a sound no-op degrade that preserves the original
    identity and sharing when there is nothing to gain.

    Parameters
    ----------
    operand_values
        The caller's already-resolved statement operands, ``None`` where an operand is not
        build-time numeric. An operand whose ``in_axes`` entry is ``None`` is not batched: the
        same array is handed to every slice of the body, so substituting its value into the
        body is value-preserving by the definition of ``vmap``, while the batched (integer
        ``in_axes``) operands are untouched. That substitution is what lets the floor-fold see a
        closed-over operand the body does not actually read, turning it into a build-time
        constant hidden behind a closure.

    Returns
    -------
    tuple
        ``(fn', keep)``. ``keep`` is ``None`` when the operand list is unchanged, else a
        per-operand mask the caller applies: a bound operand is no longer an input of the body,
        so it must leave the statement too or the ``in_axes`` would misalign.
    """
    info = _vmap_closure_of(fn)
    if info is None or depth >= _MAX_DESCENT_DEPTH:
        return fn, None
    closure, rewrap = info
    assert isinstance(closure, VmapClosure)
    body = closure._vmap_body
    if id(body) in seen:
        return fn, None  # cycle guard
    in_axes = closure._in_axes
    inner_bind: dict[str, Any] = {}
    if (operand_values is not None and isinstance(in_axes, (tuple, list))
            and len(in_axes) == len(body.inputs) == len(operand_values)):
        for ax, inp, val in zip(in_axes, body.inputs, operand_values):
            if ax is None and val is not None:
                inner_bind[inp.name] = np.asarray(val, dtype=float)
    try:
        folded = _specialize(body, inner_bind, depth + 1, seen | {id(body)})
        folded = _drop_unread_inputs(folded)
    except Exception:
        return fn, None
    if (len(folded.statements) >= len(body.statements)
            and len(folded.inputs) >= len(body.inputs)):
        return fn, None  # nothing folded away -> keep the original (preserve id / sharing)
    kept = [inp.name for inp in folded.inputs]
    orig = [inp.name for inp in body.inputs]
    if kept != orig:
        # `_specialize` + `_drop_unread_inputs` only ever REMOVE inputs (bound, or unread), never
        # reorder or add — so a `kept` that is a SUBSEQUENCE of `orig` is exactly a set of drops
        # and the surviving `in_axes` are its parallel restriction. Anything else means the
        # signature moved under us: degrade.
        kept_set = set(kept)
        if not set(orig) >= kept_set or [n for n in orig if n in kept_set] != kept:
            return fn, None
        keep = [n in kept_set for n in orig]
        if not isinstance(in_axes, (tuple, list)) or len(in_axes) != len(orig):
            return fn, None
        new_axes = tuple(ax for ax, k in zip(in_axes, keep) if k)
    else:
        keep = None
        new_axes = tuple(in_axes) if isinstance(in_axes, (tuple, list)) else in_axes
    from .ir import vmap as _vmap
    return rewrap(_vmap(folded, in_axes=new_axes, out_axes=closure._out_axes)), keep


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def specialize(
    program: Program,
    *,
    bind: Mapping[str, npt.ArrayLike] | None = None,
    subs: Mapping[str, SymArray | np.ndarray] | None = None,
    sparsity: bool = False,
    budget: SimplifyBudget | None = None,
) -> Program:
    """Specialize ``program`` against optional ``subs`` and ``bind`` values, preserving its exact results.

    Substitution runs first, then the ``bind``-seeded numeric-fold floor, then the optional
    post-build ``budget`` moderation; every step preserves the program's exact results.

    Parameters
    ----------
    program
        The program to specialize; it is copied, never mutated.
    bind
        Replace an input with a concrete numeric array. Folds every build-time-numeric
        subcomputation, drops the producing statements, and descends into a
        partially-numeric sub-program or ``CallOp`` body.
    subs
        Replace an input with an expression over other inputs. Applied first: each
        substituted input's per-cell atoms are rewritten throughout the program via
        :meth:`RationalFunction.compose`, then the input is dropped.
    sparsity
        Accepted for API parity, but a no-op passthrough. Use
        :func:`polyarray.sparsity.propagate_sparsity` directly.
    budget
        Post-build moderation applied after the unconditional numeric-fold floor: it
        collapses, extracts or keeps the residual symbolic structure. ``None`` runs the
        floor only.

    Returns
    -------
    Program
        The specialized program.
    """
    del sparsity  # P4 sparsity is a separate pass; accepted here for API parity.
    # P3: apply symbolic substitution first, then the bind+fold+descent floor,
    # then (P5) the post-build budget moderation.
    prog = substitute(program, subs) if subs else program
    result = _specialize(prog, dict(bind or {}), 0, frozenset())
    if budget is not None:
        result = _apply_budget(result, budget)
    return result


def _specialize(
    program: Program,
    bind: Mapping[str, npt.ArrayLike],
    depth: int,
    seen: frozenset[int],
) -> Program:
    new = program.copy()
    known, dropped = _seed_bind(new, bind)
    # Guard self / ancestor reference for any descent spawned below.
    seen_here = seen | {id(program), id(new)}

    foldable: set[int] = set()
    # Forward-linked resolution of dynamic dims (δ) created by a folded Stmt:
    # ``source-tuple -> concrete int`` (see _resolve_stmt_dims / _substitute_dims).
    dim_subst: dict[DimSource, int] = {}
    for i, stmt in enumerate(new.statements):
        if not _simple_stmt(stmt):
            continue
        resolved: list[np.ndarray] = []
        ok = True
        for r in stmt.in_:
            v = _try_eval_ref(new, r, i, known)
            if v is None:
                ok = False
                break
            resolved.append(v)
        if not ok:
            continue
        try:
            outs = _exec_fn(stmt.fn, resolved)
            staged_dims: dict[tuple[Any, ...], int] = dict(dim_subst)
            _resolve_stmt_dims(stmt, i, outs, staged_dims)
            staged: dict[str, Any] = dict(known)
            _record_known(stmt, outs, staged, staged_dims)
        except Exception:
            continue  # any failure -> keep the Stmt symbolic
        known = staged
        dim_subst = staged_dims
        foldable.add(i)

    # Resolve every δ a folded Stmt created into its concrete rank across all
    # remaining shapes, so a folded constant SVD/GSVD/… leaves no lingering
    # dynamic dim downstream (a matrix that was square stays square, etc.).
    _substitute_dims(new, dim_subst)

    survivors = [i for i in range(len(new.statements)) if i not in foldable]
    idx_map = {old: new_i for new_i, old in enumerate(survivors)}
    new_statements: list[Stmt] = []
    for i in survivors:
        s = new.statements[i]
        descended = None
        if depth < _MAX_DESCENT_DEPTH and _simple_stmt(s):
            descended = _try_descend(new, s, i, known, idx_map, depth, seen_here)
        if descended is not None:
            new_fn, new_in = descended
        else:
            # P6b: descend the floor-fold into a surviving vmap body to collapse its
            # data-independent (constant) subcomputations — the QR/SVD frame prep on the
            # fixed reference basis — without dropping the per-point batching.  The already-
            # resolved operand VALUES ride along so an UNBATCHED (``in_axes=None``) numeric
            # operand can be substituted inside the body; ``keep`` then prunes it from the Stmt.
            vals = [_try_eval_ref(new, r, i, known) for r in s.in_]
            new_fn, keep = _fold_vmap_body(s.fn, depth, seen_here, vals)
            new_in = tuple(_fold_ref(new, r, i, known, idx_map) for r in s.in_)
            if keep is not None and len(keep) == len(new_in):
                new_in = tuple(r for r, k in zip(new_in, keep) if k)
        new_statements.append(
            Stmt(fn=new_fn, in_=new_in, out=s.out, note=s.note,
                 provenance=s.provenance, inline=s.inline)
        )
    new.statements = new_statements
    new.outputs = {k: _fold_symarray(sa, known, new, k) for k, sa in new.outputs.items()}

    if dropped:
        new.inputs = tuple(inp for inp in new.inputs if inp.name not in dropped)
        for nm in dropped:
            new.input_arrays.pop(nm, None)
    return new


def fold_numeric(program: Program) -> Program:
    """Constant-fold a program and drop the dead statements, with no substitution.

    This is :func:`specialize` with an empty ``bind``: it folds only the subcomputations
    already numeric in the program, so a fully-symbolic program returns as a no-op copy.
    """
    return specialize(program)


def bind_inputs(program: Program, bind: Mapping[str, npt.ArrayLike]) -> Program:
    """Replace inputs with concrete numeric arrays, then fold and drop them."""
    return specialize(program, bind=bind)


def _read_stmt_outs(stmt: Stmt, bindings: Mapping[str, Any]) -> list[np.ndarray] | None:
    """Read a statement's output arrays back from a completed run's bindings.

    A bulk output is read under its bulk name; a per-cell output is assembled from its cell
    atoms.

    Returns
    -------
    list of numpy.ndarray or None
        ``None`` when any piece is absent, such as a dynamic shape this pass does not
        handle.
    """
    outs: list[np.ndarray] = []
    for bound in stmt.out:
        if bound._bulk is not None:
            val = bindings.get(bound._bulk.name)
            if val is None:
                return None
            outs.append(np.asarray(val, dtype=float))
            continue
        cells = np.asarray(bound.cells)
        arr = np.empty(cells.shape, dtype=float)
        for idx in (np.ndindex(*cells.shape) if cells.shape else [()]):
            cell = cells[idx] if cells.shape else cells[()]
            if isinstance(cell, RationalFunction):
                key = cell.gens[0] if cell.gens else None
                if key is None or key not in bindings:
                    return None
                arr[idx] = float(bindings[key])
            else:
                try:
                    arr[idx] = float(cell)
                except (TypeError, ValueError):
                    return None
        outs.append(arr)
    return outs


class NonExactFoldWarning(UserWarning):
    """Warning that a build-time constant was certified non-exactly, by random-probe identity testing.

    It is raised whenever ``mode="hybrid"`` freezes a statement the exact lane could not
    normalize: the certificate for those statements is probabilistic, with measure-zero
    failure, rather than exact-by-construction. Silence it with ``mode="probe"`` to accept
    probing, or forbid the freeze with ``mode="exact"``.
    """


class NonDeterministicFoldWarning(NonExactFoldWarning):
    """Warning that the exact lane's wall-clock backstop fired, leaving the result machine-dependent.

    The exact lane is budgeted in deterministic work units precisely so that a certificate
    means the same thing on every machine. The clock survives only as a backstop against a
    mis-calibrated cost model on a pathological program, and if it fires the very property the
    work budget exists to provide is gone: a re-run on a faster machine may certify more.

    It subclasses :class:`NonExactFoldWarning` so that every existing consumer of fold
    provenance — notably a certificate cache that decides the ``exact`` bit by walking this
    hierarchy — treats it as non-exact without changes. Seeing it means the cost model needs
    fixing, not the budget raising.
    """


_PARTIAL_EVAL_MODES = ("exact", "hybrid", "probe")


def _resolve_legacy_time_budget(time_budget: float | None) -> float | None:
    """Translate a ``time_budget=`` into a wall-clock backstop, loudly.

    ``time_budget`` is accepted because the committed surface carries it and it has external
    consumers, but it no longer selects what the exact lane folds — that is ``work_budget``, in
    deterministic work units. It sizes the backstop instead, still guaranteeing the call
    terminates, which is the reason callers pass it. Passing it warns, because it no longer
    bounds what folds.
    """
    if time_budget is None:
        return None                            # `None` = exact_fold's default backstop
    warnings.warn(
        "partial_eval_numeric(time_budget=…) no longer decides what the exact lane folds — "
        "that is now `work_budget`, in deterministic work units, so a certificate no longer "
        "depends on how loaded the machine is. The value is being used as a wall-clock "
        "BACKSTOP only: it still bounds how long the call may run, and if it fires you get a "
        "NonDeterministicFoldWarning. Pass work_budget= to bound what folds.",
        DeprecationWarning, stacklevel=3)
    return float(time_budget)


def _resolve_partial_eval_mode(mode: str | None) -> str:
    """Validate ``mode``, falling back to the ``POLYARRAY_PARTIAL_EVAL_MODE`` env default.

    The explicit parameter is the API; the environment variable only moves the default used
    when ``mode`` is ``None``, so an explicit argument always wins.
    """
    if mode is None:
        mode = os.environ.get("POLYARRAY_PARTIAL_EVAL_MODE", "hybrid")
    if mode not in _PARTIAL_EVAL_MODES:
        raise ValueError(
            f"partial_eval_numeric: mode must be one of {_PARTIAL_EVAL_MODES}, got {mode!r}")
    return mode


def _warn_probe_freezes(
    program: Program, probe_frozen: set[int], reasons: Mapping[int, str],
    *, probes: int, rtol: float,
) -> None:
    """Emit one aggregated :class:`NonExactFoldWarning` naming the probe-frozen sites."""
    if not probe_frozen:
        return
    def _site(i: int) -> str:
        stmt = program.statements[i]
        names = [o._bulk.name if o._bulk is not None else
                 next((c.gens[0] for c in np.asarray(o.cells).reshape(-1)
                       if isinstance(c, RationalFunction) and c.gens), "?")
                 for o in stmt.out]
        note = f" note={stmt.note!r}" if stmt.note else ""
        why = reasons.get(i, "?")
        return f"stmt {i}{note} -> {', '.join(names)} (exact lane: {why})"
    shown = [_site(i) for i in sorted(probe_frozen)[:5]]
    more = len(probe_frozen) - len(shown)
    # Surface the DEEPEST distinct blockers across the whole unresolved set — a frozen
    # statement's own reason is often just the downstream cascade ("operand depends on
    # an unresolved statement"), while the root cause (the QR / front-end sign-fix /
    # vmap closure inside a sub-program) sits on an earlier unresolved statement.
    roots = sorted({r for r in reasons.values()
                    if not r.startswith("operand depends")})
    root_note = (" Exact-lane root blockers: " + "; ".join(roots[:4])
                 + (f"; … {len(roots) - 4} more" if len(roots) > 4 else "") + "."
                 if roots else "")
    warnings.warn(NonExactFoldWarning(
        f"partial_eval_numeric(mode='hybrid') froze {len(probe_frozen)} statement(s) of "
        f"program {program.name!r} by PROBE (non-exact) polynomial identity testing "
        f"(probes={probes}, rtol={rtol}): "
        + "; ".join(shown) + (f"; … {more} more" if more > 0 else "")
        + ". These certificates are probabilistic, not exact-by-construction: the "
          "entries could not be brought to rational normal form." + root_note
        + " Pass mode='probe' to accept probing silently, or mode='exact' to refuse "
          "non-exact folds."),
        stacklevel=3)


def _partial_eval_numeric(
    program: Program, *, probes: int, seed: int, rtol: float, atol: float,
    mode: str = "probe", work_budget: int | None = None, max_sym_mass: int | None = None,
    wall_backstop: float | None = None,
) -> tuple[Program, dict, Any]:
    """Fold every statement whose outputs are invariant under the program's symbolic inputs.

    Invariance is certified by exact normalization, by probing, or by both, according to
    ``mode``. Where ``fold_numeric`` folds only numeric-closed subcomputations — those with no
    symbolic ancestor in the dataflow — this pass folds the strictly larger class whose outputs
    merely do not *depend* on the symbolic inputs. The canonical case is a chain
    ``inv(A) → A·inv(A)``, identically ``I`` for every ``A``: dataflow calls it symbolic,
    identity testing calls it constant.

    ``mode`` selects how invariance is certified (see :func:`partial_eval_numeric` for the
    public contract):

    * ``"exact"`` uses :mod:`polyarray.exact_fold` alone, certifying constancy from the exact
      rational normal form of each output entry (flint ``fmpq`` arithmetic,
      exact-by-construction). An entry that cannot be normalized, such as an opaque op on the
      symbolic path, is simply left unfolded.
    * ``"hybrid"`` runs exact first, then falls back to the probe pass for the statements the
      exact lane left unresolved, raising one aggregated :class:`NonExactFoldWarning` for every
      such non-exact freeze. Statements the exact lane refuted as provably non-constant are
      never probed, which closes the colluding-probe false-freeze hole.
    * ``"probe"`` freezes silently: it draws ``probes`` random input bindings over
      ``[0.6, 1.6]`` and freezes every statement output that is finite and equal
      (``rtol``/``atol``) across runs. This is probabilistic, with measure-zero false freezes
      rather than exact-by-construction, for diagnostic or performance call sites that do not
      need exactness.

    Folding is at statement granularity: an intermediate that genuinely varies stays symbolic
    even when a downstream output is invariant, and that downstream statement still folds on its
    own. The exact lane's entry-level fold in :func:`partial_eval_numeric_symarray` additionally
    certifies cell-level cancellations.

    Static inputs only; a ``DimAtom``-shaped input raises :class:`NotImplementedError`. Inputs
    are never dropped, and unused ones simply go unread at ``run`` time.
    """
    if probes < 2:
        raise ValueError(f"partial_eval_numeric needs ≥ 2 probes, got {probes}")
    for inp in program.inputs:
        for dim in inp.shape:
            if not isinstance(dim, int):
                raise NotImplementedError(
                    f"partial_eval_numeric: dynamic input {inp.name!r} (DimAtom axis) unsupported")

    known: dict[str, Any] = {}
    foldable: set[int] = set()
    exact_state = None
    if mode in ("exact", "hybrid"):
        from .exact_fold import _MAX_SYM_MASS, exact_partial_eval
        exact_state = exact_partial_eval(
            program, work_budget=work_budget, wall_backstop=wall_backstop,
            max_sym_mass=_MAX_SYM_MASS if max_sym_mass is None else max_sym_mass)
        known.update(exact_state.known)
        foldable |= exact_state.folded

    probe_frozen: set[int] = set()
    if mode in ("probe", "hybrid"):
        rng = np.random.default_rng(seed)
        runs: list[Mapping[str, Any]] = []
        for _ in range(probes):
            vals = {inp.name: rng.uniform(0.6, 1.6, size=tuple(inp.shape))
                    for inp in program.inputs}
            runs.append(program.build_runtime_bindings(vals))

        per_probe_outs: list[list[list[np.ndarray] | None]] = [
            [_read_stmt_outs(stmt, b) for stmt in program.statements] for b in runs
        ]

        # Exactly-refuted statements are PROVABLY non-constant: never probe-freeze
        # them (a colluding probe set is exactly the unsound case this closes).
        skip: set[int] = set(foldable)
        if exact_state is not None:
            skip |= exact_state.refuted
        for i, stmt in enumerate(program.statements):
            if i in skip:
                continue
            outs0 = per_probe_outs[0][i]
            if outs0 is None or any(not np.all(np.isfinite(o)) for o in outs0):
                continue
            invariant = all(
                per_probe_outs[p][i] is not None
                and all(np.allclose(a, b, rtol=rtol, atol=atol)
                        for a, b in zip(outs0, per_probe_outs[p][i], strict=True))
                for p in range(1, probes)
            )
            if not invariant:
                continue
            # ATOMICITY WITHOUT AN O(N²) COPY. `_record_known` must be all-or-nothing (it
            # raises `ValueError` on a shape mismatch, and a half-recorded fold would poison
            # `known`), which used to be spelled `staged = dict(known)` — a FULL COPY of the
            # accumulated bindings for EVERY frozen statement, i.e. quadratic in the number of
            # freezes. `_record_known` only ever WRITES into the dict it is handed (it never
            # reads it), so recording into an EMPTY dict and merging on success writes exactly
            # the same entries, and a raise still leaves `known` untouched. Measured on the
            # A high-degree affine gate freezes tens of thousands of times over a `known` that
            # grows past 10⁵ atoms.
            staged: dict[str, Any] = {}
            try:
                _record_known(stmt, outs0, staged)
            except ValueError:
                continue
            known.update(staged)
            probe_frozen.add(i)
        foldable |= probe_frozen

    # The backstop firing is reported REGARDLESS of mode and regardless of whether anything was
    # probe-frozen: it says the result is machine-dependent, which is a different (and worse)
    # claim than "some certificates are probabilistic", and `mode="exact"` — the mode chosen
    # precisely to refuse non-exact folds — must not swallow it.
    if exact_state is not None and exact_state.hit_wall_clock:
        warnings.warn(NonDeterministicFoldWarning(
            f"exact_fold's WALL-CLOCK BACKSTOP fired on program {program.name!r} after "
            f"{exact_state.spent} work units (budget {exact_state.limit}). The remaining "
            f"statements were left unresolved on a TIMER, so this result is NOT reproducible: "
            f"a faster machine would certify more. The work cost model under-counted this "
            f"program — that is the bug to fix, not the budget to raise."),
            stacklevel=3)

    if mode == "hybrid":
        _warn_probe_freezes(
            program, probe_frozen,
            exact_state.unresolved if exact_state is not None else {},
            probes=probes, rtol=rtol)

    new = program.copy()
    survivors = [i for i in range(len(new.statements)) if i not in foldable]
    idx_map = {old: new_i for new_i, old in enumerate(survivors)}
    new_statements: list[Stmt] = []
    for i in survivors:
        st = new.statements[i]
        new_in = tuple(_fold_ref(new, r, i, known, idx_map) for r in st.in_)
        new_statements.append(Stmt(fn=st.fn, in_=new_in, out=st.out, note=st.note,
                                   provenance=st.provenance, inline=st.inline))
    new.statements = new_statements
    new.outputs = {k: _fold_symarray(sa, known, new, k) for k, sa in new.outputs.items()}
    return new, known, exact_state


def partial_eval_numeric(
    program: Program, *, probes: int = 3, seed: int = 0,
    rtol: float = 1e-9, atol: float = 1e-12,
    mode: str | None = None, work_budget: int | None = None,
    max_sym_mass: int | None = None, time_budget: float | None = None,
) -> Program:
    """Fold every statement whose outputs are invariant under the program's symbolic inputs.

    See :func:`_partial_eval_numeric` for the mechanics. ``mode`` selects how invariance is
    certified:

    * ``"exact"`` certifies from the exact rational normal form alone
      (:mod:`polyarray.exact_fold`, exact-by-construction), leaving non-normalizable statements
      symbolic.
    * ``"hybrid"``, the default, certifies exactly where it can and falls back to the probe
      pass for opaque or unresolved statements only, raising an aggregated
      :class:`NonExactFoldWarning` that names the probe-frozen sites. Statements the exact lane
      proved non-constant are never probe-frozen.
    * ``"probe"`` freezes silently and probabilistically, for diagnostic or performance sites
      that do not need exactness.

    ``mode=None`` reads the ``POLYARRAY_PARTIAL_EVAL_MODE`` env default, else ``"hybrid"``, and
    an explicit parameter always wins. ``probes`` sets the probe count of the probe and
    hybrid-fallback lanes. ``work_budget`` (deterministic work units, charged between
    operations) and ``max_sym_mass`` (the monomial mass one symbolic op's operands may carry,
    checked before it runs, since an object-dtype einsum or Gauss pass is uninterruptible once
    started) jointly bound the exact lane: an oversized or out-of-budget statement degrades to
    the warned probe fallback rather than hang. ``max_sym_mass=None`` uses
    ``exact_fold._MAX_SYM_MASS``, and ``work_budget=None`` uses the
    ``POLYARRAY_EXACT_WORK_BUDGET`` env knob, else ``exact_fold._DEFAULT_WORK_BUDGET``.

    The budget is work, not seconds: what certifies is a function of the program alone, so the
    same input yields the same certificate on any machine under any load. See
    :class:`NonDeterministicFoldWarning` for the one case where that guarantee lapses.
    """
    new, _known, _state = _partial_eval_numeric(
        program, probes=probes, seed=seed, rtol=rtol, atol=atol,
        mode=_resolve_partial_eval_mode(mode), work_budget=work_budget,
        max_sym_mass=max_sym_mass,
        wall_backstop=_resolve_legacy_time_budget(time_budget))
    return new


def partial_eval_numeric_symarray(
    sa: SymArray, *, probes: int = 3, seed: int = 0,
    rtol: float = 1e-9, atol: float = 1e-12,
    mode: str | None = None, work_budget: int | None = None,
    max_sym_mass: int | None = None, time_budget: float | None = None,
) -> SymArray:
    """Apply :func:`partial_eval_numeric` to a SymArray whose cells reference program atoms.

    Folds the threaded program and the cells together, so an invariant atom becomes a
    numeric cell — the structural form of the array.

    Under the exact and hybrid modes the cells additionally get the entry-level exact fold:
    a cell is certified constant when its rational normal form over the feed atoms has total
    degree zero, so a cancellation that completes only at the entry, with no single
    statement invariant, still folds exactly. ``mode``, ``probes``, ``work_budget`` and
    ``max_sym_mass`` behave as in :func:`partial_eval_numeric`.
    """
    from .ir import SymArray
    if sa.program is None:
        return sa
    mode_r = _resolve_partial_eval_mode(mode)
    backstop = _resolve_legacy_time_budget(time_budget)
    new, known, state = _partial_eval_numeric(
        sa.program, probes=probes, seed=seed, rtol=rtol, atol=atol,
        mode=mode_r, work_budget=work_budget, max_sym_mass=max_sym_mass,
        wall_backstop=backstop)
    cells = _fold_cells(np.asarray(sa.cells), known)
    if state is not None:
        from .exact_fold import _MAX_SYM_MASS, exact_fold_cells
        cells = exact_fold_cells(
            cells, state, sa.program, work_budget=work_budget, wall_backstop=backstop,
            max_sym_mass=_MAX_SYM_MASS if max_sym_mass is None else max_sym_mass)
    folded = _numify_constant_cells(cells)
    return SymArray(folded, program=new)


# ---------------------------------------------------------------------------
# P3: symbolic argument substitution (`subs`)
# ---------------------------------------------------------------------------

def _as_rf(value: Cell) -> RationalFunction:
    """Coerce a cell value, a :class:`RationalFunction` or a number, to a :class:`RationalFunction`."""
    if isinstance(value, RationalFunction):
        return value
    if isinstance(value, (int, float)):
        return RationalFunction.constant(float(value))
    raise TypeError(f"substitute expression cell must be RF/numeric; got {value!r}")


def _build_subs_map(
    in_sa: SymArray, expr: SymArray | np.ndarray,
) -> tuple[dict[str, RationalFunction], np.ndarray]:
    """Map each cell atom of the substituted input to its replacement rational function.

    Returns ``(gen_name -> repl_rf, composed_input_cells)``, where the second element is the
    input's cells with each atom replaced by its expression cell, used to rewrite any direct
    ``InputRef`` to the dropped input.
    """
    if in_sa._bulk is not None:
        raise NotImplementedError("substitute of a bulk/dynamic input is unsupported")
    cells = np.asarray(in_sa.cells)
    is_array_expr = isinstance(expr, SymArray)
    if is_array_expr:
        ecells = np.asarray(expr.cells)
        if tuple(ecells.shape) != tuple(cells.shape):
            raise ValueError(
                f"substitute expression shape {ecells.shape} != input shape {cells.shape}"
            )
    subs_map: dict[str, RationalFunction] = {}
    composed = np.empty(cells.shape, dtype=object)
    shape = cells.shape
    for idx in (np.ndindex(*shape) if shape else [()]):
        cell = cells[idx] if shape else cells[()]
        repl = _as_rf(ecells[idx] if shape else ecells[()]) if is_array_expr else _as_rf(expr)
        if isinstance(cell, RationalFunction):
            subs_map[cell.gens[0]] = repl
        composed[idx] = repl
    return subs_map, composed


def _compose_cells(cells: np.ndarray, subs_map: Mapping[str, RationalFunction]) -> np.ndarray:
    """Apply :meth:`RationalFunction.compose_multi` to every :class:`RationalFunction` cell."""
    cells = np.asarray(cells)
    if cells.dtype.kind == "f":
        return cells.copy()
    out = np.empty(cells.shape, dtype=object)
    shape = cells.shape
    for idx in (np.ndindex(*shape) if shape else [()]):
        c = cells[idx] if shape else cells[()]
        if isinstance(c, RationalFunction):
            v = c.compose_multi(subs_map)
        else:
            v = c
        if shape:
            out[idx] = v
        else:
            out[()] = v
    return out


def _compose_symarray(
    sa: SymArray, subs_map: Mapping[str, RationalFunction], program: Program, name: str | None,
) -> SymArray:
    """Rewrite a SymArray's cells under ``subs_map`` (bulk arrays pass through)."""
    if sa._bulk is not None:
        return sa  # a bulk handle holds no per-cell substituted-input atoms
    return SymArray(_compose_cells(np.asarray(sa.cells), subs_map), program=program, name=name)


def _compose_ref(
    ref: Ref,
    subs_map: Mapping[str, RationalFunction],
    composed_inputs: Mapping[str, np.ndarray],
) -> Ref:
    """Rewrite a statement input ref under ``subs_map``.

    Cells that carry the substituted input's atoms — ``SymArrayRef`` /
    ``RationalRef`` — are composed.  A direct ``InputRef`` to a dropped input is
    rewritten to a ``SymArrayRef`` over the input's composed expression cells (so
    the dropped input never needs to be bound at run time).
    """
    if isinstance(ref, SymArrayRef):
        if ref._bulk is not None:
            return ref
        return SymArrayRef(_compose_cells(np.asarray(ref.cells), subs_map))
    if isinstance(ref, RationalRef):
        return RationalRef(ref.rf.compose_multi(subs_map))
    if isinstance(ref, InputRef) and ref.name in composed_inputs:
        cells = composed_inputs[ref.name]
        sub = cells[ref.indices] if ref.indices else cells
        return SymArrayRef(np.asarray(sub, dtype=object))
    return ref  # OutputRef / Const / IntAtomRef / InputRef over a live input


def substitute(program: Program, subs: Mapping[str, SymArray | np.ndarray]) -> Program:
    """Replace inputs with expressions over the program's *other* inputs.

    ``subs`` maps an input name to a :class:`SymArray` (per-cell, shape-matched)
    or a :class:`RationalFunction` (broadcast to every cell) whose generators are
    the program's other existing input atoms.  Each substituted input's per-cell
    atom ``g`` is replaced everywhere it appears — program outputs and Stmt-input
    ``SymArrayRef`` / ``RationalRef`` / ``InputRef`` cells — via
    :meth:`RationalFunction.compose`, after which the input is dropped from
    :attr:`Program.inputs` / :attr:`Program.input_arrays`.

    Exactness: ``substitute(p, {b: expr}).run(rest) ==
    p.run({**rest, b: expr_evaluated_at(rest)})``.  Read-only on ``program``.
    """
    if not subs:
        return program.copy()
    new = program.copy()
    subs_map: dict[str, RationalFunction] = {}
    composed_inputs: dict[str, np.ndarray] = {}
    dropped: set[str] = set()
    for name, expr in subs.items():
        if name not in new.input_arrays:
            raise KeyError(f"unknown input {name!r}; have {list(new.input_arrays)}")
        gmap, composed = _build_subs_map(new.input_arrays[name], expr)
        subs_map.update(gmap)
        composed_inputs[name] = composed
        dropped.add(name)

    new.statements = [
        Stmt(
            fn=s.fn,
            in_=tuple(_compose_ref(r, subs_map, composed_inputs) for r in s.in_),
            out=s.out,
            note=s.note,
            provenance=s.provenance,
            inline=s.inline,
        )
        for s in new.statements
    ]
    new.outputs = {
        k: _compose_symarray(sa, subs_map, new, k) for k, sa in new.outputs.items()
    }
    new.inputs = tuple(inp for inp in new.inputs if inp.name not in dropped)
    for nm in dropped:
        new.input_arrays.pop(nm, None)
    return new


# ---------------------------------------------------------------------------
# Dependency-cone evaluation — evaluate ONE SymArray without running the whole
# program (so an unrelated singular / failing op elsewhere never runs).
# ---------------------------------------------------------------------------


def symarray_atoms(sa: SymArray) -> set[str]:
    """Return the run-time binding-key atoms a ``SymArray`` reads.

    Exported so a consumer can ask whether a value depends on a given atom without
    reimplementing the traversal — the alternative being the cells-unwrapping the stack
    rules forbid.
    """
    return _symarray_atoms(sa)


def _symarray_atoms(sa: SymArray) -> set[str]:
    """Return the run-time binding-key atoms a SymArray's value depends on.

    Its bulk name, or its cells' RationalFunction generators (the keys ``SymArray.evaluate``
    resolves).

    Parameters
    ----------
    sa
        The array to read.

    Returns
    -------
    set of str
        The binding keys the value varies with.  Empty when nothing outside the array is read.

    Notes
    -----
    ⚠ ``RationalFunction.gens`` is the generator list of the cell's **ring**, not the set of
    generators the cell's value actually varies with.  A cell of total degree zero in every
    generator — ``is_constant()``, or the zero polynomial, for which ``is_constant()`` is
    ``False`` only because ``_total_degree`` spells the zero polynomial ``-1`` — has the same
    value under every binding, so it reads none of its ring's generators and contributes no
    atom.  Both tests are exact and structural (a total-degree test on the numerator and
    denominator), never a sample, so excluding such a cell removes atoms the value provably
    does not depend on: strictly more precise, and never a claim of independence that cannot be
    proved.  Without it, a constant that merely rides on a ring built around a ``vertex`` /
    ``point`` feed reads as feed-dependent, and every dependency question downstream —
    :func:`is_structurally_constant` above all — answers conservatively wrong.
    """
    bulk = getattr(sa, "_bulk", None)
    if bulk is not None:
        return {bulk.name}
    cells = np.asarray(sa._cells, dtype=object)
    atoms: set = set()
    for c in (np.ravel(cells) if cells.shape else [cells[()]]):
        if isinstance(c, RationalFunction) and not (c.is_constant() or c.is_zero()):
            atoms.update(c.gens)
    return atoms


def _dimatom_stmt(d: DimAtom) -> int | None:
    """Return the producing statement index of a ``DimAtom`` sourced from a prior output."""
    src = getattr(d, "source", None)
    if isinstance(src, tuple) and len(src) == 3 and src[0] == "stmt":
        return int(src[1])
    if isinstance(src, tuple) and len(src) == 2 and isinstance(src[0], int):
        return int(src[0])                                   # legacy (stmt_idx, out_idx)
    return None


def _ref_deps(ref: Ref) -> tuple[set[str], set[int]]:
    """Return the atom names and direct statement indices one input ref pulls in."""
    if isinstance(ref, OutputRef):
        return set(), {ref.stmt_idx}
    if isinstance(ref, SymArrayRef):
        bulk = ref._bulk
        if bulk is not None:
            return {bulk.name}, set()
        cells = np.asarray(ref._cells, dtype=object)
        atoms: set = set()
        for c in (np.ravel(cells) if cells.shape else [cells[()]]):
            if isinstance(c, RationalFunction):
                atoms.update(c.gens)
        return atoms, set()
    if isinstance(ref, RationalRef):
        return set(ref.rf.gens), set()
    if isinstance(ref, IntAtomRef):
        return {ref.name}, set()
    return set(), set()                                      # InputRef / Const → no stmt dep


def _shape_dim_stmts(shape: Sequence[int | DimAtom]) -> set[int]:
    """Return the statement indices any ``DimAtom`` in ``shape`` is sourced from."""
    out: set = set()
    for d in shape or ():
        if not isinstance(d, int):
            s = _dimatom_stmt(d)
            if s is not None:
                out.add(s)
    return out


def dependency_cone(program: Program, target: SymArray) -> set[int]:
    """Return the statement indices ``target`` transitively depends on — its cone."""
    producers: dict = {}
    for i, stmt in enumerate(program.statements):
        for out in stmt.out:
            for a in _symarray_atoms(out):
                producers.setdefault(a, i)

    cone: set = set()
    stack: list = list(producers[a] for a in _symarray_atoms(target) if a in producers)
    tbulk = getattr(target, "_bulk", None)
    if tbulk is not None:
        stack.extend(_shape_dim_stmts(tbulk.shape))
    while stack:
        i = stack.pop()
        if i in cone:
            continue
        cone.add(i)
        stmt = program.statements[i]
        for ref in stmt.in_:
            atoms, stmts = _ref_deps(ref)
            stack.extend(stmts)
            stack.extend(producers[a] for a in atoms if a in producers)
        for out in stmt.out:
            b = getattr(out, "_bulk", None)
            if b is not None:
                stack.extend(_shape_dim_stmts(b.shape))
    return cone


def is_structurally_constant(program: Program, target: SymArray) -> bool:
    """Decide, by dependency rather than sampling, whether ``target`` is a build-time constant.

    This walks ``target``'s dependency cone — the set of statements feeding it, from
    :func:`dependency_cone` — and returns ``True`` exactly when that cone is a *closed constant
    subprogram*, meaning both of:

    1. No statement input is an :class:`~polyarray.ir.InputRef` (a reference to a declared
       program input) or an :class:`~polyarray.ir.IntAtomRef` (a runtime-bound integer), both
       of which are values supplied at run time.
    2. No cell generator anywhere in the cone, nor in ``target`` itself, has a provenance
       ``kind`` other than ``"stmt_out"`` — that is, no ``vertex``, ``point``, ``coeff``,
       ``param_dof`` or ``per_point`` feed atom is read. It must be *read*, not merely named:
       the generators come from :func:`_symarray_atoms`, which excludes a cell of total degree
       zero (see its note), since such a cell holds the same value under every binding and a
       feed generator sitting in its ring is then not a dependency.

    The test is sound. A cone with no live-input reference and only ``stmt_out`` generators
    depends on nothing outside itself: its leaves are nullary constant ops (``ConstOp`` /
    ``EyeOp`` …), every intermediate atom is produced by a prior statement *in the cone*, and no
    generator traces to a program feed. Its value is therefore identical under every binding, a
    build-time constant that folds without any sampling. This is exact rather than heuristic,
    and never mistakes a feed-varying operator for a constant.

    The test is also conservative. Any ref type that cannot be positively classified as
    constant-safe (an :class:`~polyarray.ir.OutputRef` — a prior in-cone statement output — or a
    :class:`~polyarray.ir.Const` literal), or any generator whose provenance cannot be looked
    up, forces ``False``. It never claims constancy it cannot prove, so a value with any live
    input dependency is reported non-constant.
    """
    def _gens_ok(atoms: set) -> bool:
        for name in atoms:
            try:
                prov = program.env.of(name)
            except KeyError:
                return False                         # unknown generator ⇒ can't prove ⇒ conservative
            if prov.kind != "stmt_out":
                return False                         # a feed atom (vertex/point/coeff/…) ⇒ NOT constant
        return True

    # ``target`` itself must carry no feed generator (a value whose cells directly reference
    # a vertex/point atom depends on the feed even if no Stmt produces it).
    if not _gens_ok(_symarray_atoms(target)):
        return False

    try:
        cone = dependency_cone(program, target)
    except Exception:  # noqa: BLE001 — can't enumerate the cone ⇒ don't claim constancy
        return False

    for i in cone:
        stmt = program.statements[i]
        for ref in stmt.in_:
            if isinstance(ref, OutputRef):
                continue                             # a prior IN-CONE Stmt output ⇒ constant-safe
            if isinstance(ref, Const):
                continue                             # a frozen numeric literal ⇒ constant-safe
            if isinstance(ref, InputRef):
                return False                         # references a declared program INPUT ⇒ live
            if isinstance(ref, IntAtomRef):
                return False                         # a runtime-bound integer ⇒ live
            if isinstance(ref, (SymArrayRef, RationalRef)):
                if not _gens_ok(_ref_deps(ref)[0]):  # its generators must all be stmt_out
                    return False
                continue
            return False                             # unknown ref type ⇒ conservative
    return True


def _read_symarray(program: Program, sa: SymArray, bindings: dict[str, float]) -> np.ndarray:
    bulk = getattr(sa, "_bulk", None)
    if bulk is not None:
        return np.asarray(bindings[bulk.name], dtype=float)
    return np.asarray(program._evaluate_cells(np.asarray(sa._cells), bindings), dtype=float)


def evaluate_cone(program: Program, target: SymArray, values: Mapping[str, Any]) -> np.ndarray:
    """Evaluate ``target`` by running only its dependency cone at ``values``.

    Statements outside :func:`dependency_cone` are never executed, so a singular or failing op
    elsewhere in ``program`` — a common hazard when probing a partially-built shared program at
    a generic feed — can neither affect nor crash the evaluation of ``target``.

    Returns
    -------
    numpy.ndarray
        ``target``'s float array. This equals ``target.evaluate(values)`` whenever the full run
        would succeed, and it additionally succeeds when the full run would raise on an
        unrelated statement. ``values`` must bind every program input; the unused ones simply go
        unread.
    """
    cone = dependency_cone(program, target)
    bindings = program.build_runtime_bindings(values, only=cone)
    return _read_symarray(program, target, bindings)
