"""Post-build partial evaluation of a Program (the `simplify` pass).

First cut: ``fold_numeric`` / ``bind_inputs`` — the numeric-propagation floor
of "fold_numeric, precisely".

The pass re-interprets a Program against a *partial* numeric environment,
seeded by ``bind`` (empty for a bare ``fold_numeric``).  It maintains a growing
``known: dict[atom_name -> float]`` of every generator whose value is determined
at build time, then:

* executes every Stmt whose inputs all resolve numeric (cascading: a folded
  Stmt's outputs enter ``known`` and unlock downstream Stmts), dropping it;
* folds ``known`` into every surviving Stmt's input refs and every program
  output cell (a fully-bound cell becomes a float; a partially-bound cell
  becomes a smaller RF over the leftover gens — the "leave residual symbols"
  case);
* drops inputs replaced by a concrete ``bind`` value.

It never mutates shared state: ``Program.copy`` shares cell arrays and ref
tuples, so the pass builds *fresh* folded cells/refs rather than rewriting in
place.  Exactness: ``fold_numeric(p, bind=b).run(rest) == p.run({**b, **rest})``.

Conservative by construction — anything not confidently foldable (bulk / dynamic
outputs, control-flow ops) is kept symbolic, so the worst case degrades to
``copy()``.  A partially-numeric sub-Program / ``CallOp`` Stmt is *descended*
into (P6): its body is recursively specialized with the numeric operands bound,
shrinking the Stmt to the still-symbolic operands.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Mapping

import numpy as np

from .ir import (
    BulkOut,
    CallOp,
    Const,
    DimAtom,
    InputRef,
    IntAtomRef,
    OutputRef,
    Program,
    RationalRef,
    Stmt,
    SymArray,
    SymArrayRef,
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
    """A Stmt we may try to execute at build time: a callable fn / typed Op /
    sub-Program (not a loop).

    A *dynamic* (runtime-``DimAtom``-sized) bulk output is now foldable too: when
    the Stmt's inputs are all numeric it is a value-invariant map (e.g. a
    constant SVD/GSVD/QR/pinv whose numerical rank is statically knowable), so we
    execute it, read the concrete output shape, and resolve the ``DimAtom``s it
    created (see :func:`_resolve_stmt_dims` / :func:`_substitute_dims`).  When its
    inputs are *not* all numeric the fold loop simply skips it and the dynamic δ
    survives unchanged — byte-identical to the pre-fold behaviour.
    """
    if stmt.fn is None:
        return False
    if isinstance(stmt.fn, _SKIP_OPS):
        return False
    return True


def _exec_fn(fn: Any, resolved: list[np.ndarray]) -> list[Any]:
    """Execute a Stmt fn on concrete numeric operands, mirroring ``_run_stmt``'s
    dispatch (a raw sub-Program runs via ``.run``; everything else is called).
    """
    if isinstance(fn, Program):
        value_map = {inp.name: np.asarray(v) for inp, v in zip(fn.inputs, resolved)}
        return list(fn.run(value_map).values())
    results = fn(*resolved)
    return list(results) if isinstance(results, tuple) else [results]


def _try_eval_ref(
    prog: Program, ref: Any, stmt_idx: int, known: Mapping[str, float],
) -> np.ndarray | None:
    """Resolve ``ref`` to a concrete float array using ``known``; None if any
    needed generator is still symbolic.

    NO DEFENSIVE COPY when ``known`` is already a ``dict``: every branch of
    :meth:`Program._resolve_ref` only READS its ``bindings`` (``_eval_symarray`` →
    ``_evaluate_cells`` → ``_eval_cell`` → ``_eval_rf``, ``_eval_rf`` on a
    ``RationalRef``, a plain lookup for an ``IntAtomRef``/bulk name) — nothing writes
    back. The ``dict(known)`` this used to do was a type coercion, and it ran ONCE PER
    REF of every surviving statement over a ``known`` that reaches 10⁵+ atoms on a
    high-degree affine gate. A non-``dict`` ``Mapping`` is still materialised, so the
    signature's contract is unchanged.
    """
    b: dict[str, float] = known if isinstance(known, dict) else dict(known)
    try:
        return np.asarray(prog._resolve_ref(ref, stmt_idx, b), dtype=float)
    except Exception:
        return None


def _cells_touch_known(cells: np.ndarray, known: Mapping[str, float]) -> bool:
    """True iff some ``RationalFunction`` cell references a generator in ``known`` — i.e.
    folding ``known`` into ``cells`` would actually substitute something.  When False the
    fold is a structural no-op on these cells (see :func:`_fold_cells`).
    """
    if not known:
        return False
    for c in cells.reshape(-1):
        if isinstance(c, RationalFunction) and any(g in known for g in c.gens):
            return True
    return False


def _fold_cells(cells: np.ndarray, known: Mapping[str, float]) -> np.ndarray:
    """Fold ``known`` into an ndarray of cells via partial ``RF.eval``.

    Returns a float array when every cell becomes numeric, else an object array
    of floats / smaller RationalFunctions (residual symbols).

    STRUCTURE-TRANSPARENT no-op: when no cell references a folded (``known``) generator
    the fold changes nothing, so the ORIGINAL ``cells`` object is returned unchanged —
    never a fresh copy.  This keeps ``id(cells)`` stable across the fold, which a
    downstream identity-based structural read depends on: the pointwise/grassmann
    quadrature-degree walker (``polyarray.program_degree``) links a Stmt's producer to its
    consumer by ``id(ref._cells) == id(out._cells)`` and, on a miss, falls back to scoring
    the cells' generators by NAME — a fallback that knows FIELD degrees but not the
    geometry/position generators, so a broken link silently drops the position's degree
    (the Koszul ``κ = x·`` factor) and under-integrates the drop-Vandermonde.  A fold
    that substitutes nothing must therefore leave the cell array's identity intact.
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
    """Reduce CONSTANT ``RationalFunction`` cells (no live generators) to plain floats.

    A structural fold that collapsed a cell to a constant should present it AS numeric: a
    fully-constant array then has float dtype, so :meth:`SymArray.evaluate` reads it directly
    (``dtype.kind == 'f'``) WITHOUT requiring the program's now-unused input bindings — the
    ``partial_eval_numeric`` intent that "unused inputs simply go unread". A cell that still
    carries a live generator (genuinely vertex-dependent) is left as an ``RF``, so a
    cell-dependent array stays object-dtype and ``evaluate({})`` still raises — which the
    ``affine_invariance`` / ``P(T)`` gates read as "does not fold to a constant".
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
    sa: SymArray, known: Mapping[str, Any], program: Program, name: str | None,
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
    prog: Program, ref: Any, stmt_idx: int,
    known: Mapping[str, float], idx_map: Mapping[int, int],
) -> Any:
    """Rewrite a surviving Stmt's input ref: numeric where determined, else a
    symbolically-folded version (and OutputRef stmt-indices remapped).
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
    prog: Program, bind: Mapping[str, Any],
) -> tuple[dict[str, float], set[str]]:
    """Seed ``known`` from concrete ``bind`` arrays; return (known, dropped)."""
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
    stmt: Stmt, stmt_idx: int, outs: list[Any], dim_subst: dict[tuple[Any, ...], int],
) -> None:
    """From a *folded* Stmt's concrete output arrays, resolve each ``DimAtom``
    that (first) sizes one of those outputs to its concrete axis size, recording
    ``source -> int`` in ``dim_subst``.

    This is the build-time mirror of :meth:`Program._bind_output`'s run-time
    behaviour: a not-yet-bound ``DimAtom`` in a bulk output's declared shape is
    bound from the realised array's actual axis size, keyed by the atom's
    hashable ``source`` (first realised output wins) — so :func:`_substitute_dims`
    can substitute it into every remaining shape.  Note the ``source`` is a
    logical forward-link (it may *name* a producing Stmt whose own output does not
    carry the axis — e.g. an SVD ``rank`` output whose δ physically sizes a
    downstream ``take_cols`` output); binding follows where the axis actually
    appears, exactly as at run time.  A δ already resolved by an earlier fold is
    left untouched (first-wins), matching ``d.source not in dim_bindings``.
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
    stmt: Stmt, outs: list[Any], known: dict[str, Any],
    dim_subst: Mapping[tuple[Any, ...], int] | None = None,
) -> None:
    """Record a folded Stmt's numeric outputs into ``known`` (raises on shape
    mismatch so the caller can discard a bad fold).

    A bulk output records its whole tensor under the bulk handle name (resolved
    directly by ``_resolve_ref``); a per-cell output records each cell's atom.
    A *dynamic* bulk output's declared shape (carrying ``DimAtom`` entries) is
    resolved against ``dim_subst`` before validating the produced tensor's shape
    — a KeyError there means an unresolved δ, which the caller treats as a failed
    fold (keep the Stmt symbolic).
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
    shape: tuple[Any, ...], dim_subst: Mapping[tuple[Any, ...], int],
) -> tuple[tuple[Any, ...], bool]:
    """Replace every resolved ``DimAtom`` in ``shape`` with its concrete int.

    Returns ``(new_shape, changed)``.  A ``DimAtom`` not in ``dim_subst`` (a δ
    from a Stmt that did *not* fold — genuinely data-dependent) passes through
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
    sa: SymArray, dim_subst: Mapping[tuple[Any, ...], int],
) -> SymArray:
    """A fresh SymArray with its bulk shape's resolved δ's substituted, or ``sa``
    unchanged when there is nothing to substitute (never mutates in place — the
    old bulk handle may be shared).
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
    ref: Any, dim_subst: Mapping[tuple[Any, ...], int],
) -> Any:
    """A surviving input ref with any resolved δ in a bulk handle substituted."""
    if isinstance(ref, SymArrayRef) and ref._bulk is not None:
        new_shape, changed = _subst_shape(ref._bulk.shape, dim_subst)
        if changed:
            out = SymArrayRef(ref._cells)
            out._bulk = BulkOut(name=ref._bulk.name, shape=new_shape)
            return out
    return ref


def _substitute_dims(
    program: Program, dim_subst: Mapping[tuple[Any, ...], int],
) -> None:
    """Substitute every resolved ``DimAtom`` (→ concrete int) across ALL remaining
    shapes of ``program`` **in place** (operates on a freshly-``copy``'d program,
    so this touches no shared upstream state): dynamic inputs, statement input
    refs, statement output SymArrays, and program outputs.

    This is what makes a folded constant SVD/GSVD/… uniformly eliminate its δ:
    once its rank is resolved from the concrete output shape, downstream shapes
    that were sized by that δ (a later Stmt's output, a Vandermonde's axis, an
    AssertOp's operand) become static and consistent — no lingering δ.

    Program **inputs are deliberately left untouched**: a dynamic bulk input is
    bound (as a whole tensor) only via ``build_runtime_bindings``' dynamic-input
    path, which is gated on the input's shape being dynamic (``is_dynamic``).
    Resolving its δ to a static int would drop it from that path so its bulk value
    would never bind (KeyError at run time).  A stmt-sourced δ in an input shape
    that now points at a folded-away Stmt is harmless: at run time the fed array is
    bound directly and that δ is never resolved (the producing Stmt would have, but
    the input itself is validated axis-by-axis, skipping stmt-sourced dims).
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

def _descent_body(fn: Any) -> tuple[Program, Any] | None:
    """If ``fn`` has a directly-positional body Program, return it plus a
    re-wrapper that rebuilds an ``fn`` from a specialized body; else ``None``.

    Only the cases where operands map to the body's inputs *by position with
    matching shapes* and an ``fn``-swap is exactly equivalent are descendable:

    * a raw sub-:class:`Program` (``_run_stmt`` runs it via ``zip(inputs, ops)``);
    * a :class:`CallOp` wrapping a Program (``_invoke`` maps by position too).

    A genuine ``vmap`` closure is **not** descendable here: its operands carry a
    batch axis (and ``in_axes=None`` broadcasts), so they do NOT match the body
    inputs by shape, and replacing the closure with the bare body would drop the
    batching.  Such Stmts (and ``WhileOp``, opaque callables) return ``None`` and
    stay symbolic — the conservative degrade.
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
    """Attempt P6 partial descent on a surviving Stmt ``s``.

    Returns ``(new_fn, new_in)`` when ``s`` has a descendable body, SOME (not
    all, not none) of its operands resolve numeric, and the specialized body
    drops exactly those operands' inputs.  Returns ``None`` to fall back to the
    plain symbolic ref-fold (today's behaviour) in every other case.
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

def _is_rebuildable_vmap(fn: Any) -> bool:
    """A vmap closure this pass can REBUILD needs all three attrs :func:`~polyarray.ir.vmap`
    sets — ``_vmap_body`` / ``_in_axes`` / ``_out_axes``. Some front-end wrappers expose only
    ``_vmap_body`` (for body introspection) without the axis tuples; those are NOT rebuildable
    here (``pa.ir.vmap`` needs the axes), so we must skip them — not crash on a missing attr.
    """
    return all(hasattr(fn, a) for a in ("_vmap_body", "_in_axes", "_out_axes"))


def _vmap_closure_of(fn: Any) -> tuple[Any, Any] | None:
    """If ``fn`` is a REBUILDABLE :func:`~polyarray.ir.vmap` closure (or a :class:`CallOp` wrapping
    one — carrying ``_vmap_body`` / ``_in_axes`` / ``_out_axes``), return ``(closure, rewrap)`` where
    ``rewrap`` rebuilds an equivalent ``fn`` from a fresh closure; else ``None`` (a wrapper missing the
    axis tuples degrades to no-fold).
    """
    if _is_rebuildable_vmap(fn):
        return fn, (lambda c: c)
    if isinstance(fn, CallOp) and _is_rebuildable_vmap(fn.fn):
        return fn.fn, (lambda c: CallOp(fn=c))
    return None


def _drop_unread_inputs(prog: Program) -> Program:
    """``prog`` with every input NO statement and NO output references removed (`frame-probe`).

    A DEAD input is not a small thing here: it is the only reason a higher-form DOF body is not
    recognised as a build-time constant. grassmann declares a sub-input for every Term-var that
    the binder's BASIS mentions — for a wedge slot that is ``J_face``, which NAMES the
    transported frame but is read through the constant inclusion ``ι`` alone — so the closure
    carries an input it never touches, and the enclosing Stmt therefore has a non-numeric operand
    and cannot be folded. Dropping it is value-preserving by definition: nothing reads it.

    Referencing is decided on ATOMS (:func:`symarray_atoms` over every Stmt operand and every
    output), not on a syntactic scan, so an input reached through a folded cell still counts.
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
    fn: Any, depth: int, seen: frozenset[int],
    operand_values: list[Any] | None = None,
) -> tuple[Any, list[bool] | None]:
    """Constant-fold the numeric subcomputations INSIDE a vmap body, keeping the batching.

    :func:`_descent_body` deliberately refuses to descend a vmap closure, because swapping the
    closure for the bare body would drop the per-point batching. But the body's INTERNAL Stmts
    whose inputs are all build-time-numeric — e.g. a QR/SVD frame orthonormalization on a FIXED
    reference basis, data-INDEPENDENT of the per-point vmap args — can still be folded to
    constants without touching the batching. Recurse the floor-fold into the body; if it folded
    anything away, rewrap the folded body in an equivalent vmap closure. Falls back to ``fn``
    unchanged whenever nothing folds or anything looks off — always a sound no-op degrade
    (identity/sharing preserved when there is nothing to gain).

    ``operand_values`` (`frame-probe`) — the caller's already-resolved Stmt operands, ``None``
    where an operand is not build-time numeric. An operand whose ``in_axes`` entry is ``None`` is
    **not batched**: the very same array is handed to every slice of the body, so substituting its
    value INTO the body is value-preserving by the definition of ``vmap``, and the batched
    (``in_axes`` integer) operands are untouched. Doing so is what lets the floor-fold see a
    closed-over operand that the body does not actually read — the higher-form case, where the
    binder's basis NAMES a frame map (``J_face``) that only the constant inclusion ``ι`` is read
    through, so the whole DOF is a build-time constant hidden behind a closure.

    Returns ``(fn', keep)``: ``keep`` is ``None`` when the operand list is unchanged, else a
    per-operand mask the caller applies (a bound operand is no longer an input of the body, so it
    must leave the Stmt too, or the ``in_axes`` would misalign).
    """
    info = _vmap_closure_of(fn)
    if info is None or depth >= _MAX_DESCENT_DEPTH:
        return fn, None
    closure, rewrap = info
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
    bind: Mapping[str, Any] | None = None,
    subs: Mapping[str, Any] | None = None,
    sparsity: bool = False,
    budget: SimplifyBudget | None = None,
) -> Program:
    """Partially evaluate ``program`` against optional ``subs`` / ``bind`` values.

    * ``subs`` — replace an input with an **expression over other inputs**
      (symbolic argument substitution, P3).  Applied first: each substituted
      input's per-cell atoms are rewritten throughout the program via
      :meth:`RationalFunction.compose`, then the input is dropped.
    * ``bind`` — replace an input with a **concrete numeric** array; folds every
      build-time-numeric subcomputation, drops the producing Stmts, and (P6)
      descends into a partially-numeric sub-Program / ``CallOp`` body.

    * ``budget`` (a :class:`~polyarray.budget.SimplifyBudget`) — the post-build
      moderation control surface (P5): after the unconditional numeric-fold
      floor runs, it collapses / extracts / keeps the residual symbolic
      structure.
      ``budget=None`` is the floor only.

    ``sparsity`` (P4) is accepted for API parity but is a no-op passthrough
    here — use :func:`polyarray.sparsity.propagate_sparsity` directly.
    Exactness-preserving throughout.
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
    bind: Mapping[str, Any],
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
    dim_subst: dict[tuple[Any, ...], int] = {}
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
    # dynamic dim downstream (the Vandermonde stays square, etc.).
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
    """Constant-fold + dead-stmt elimination with no substitution.

    ``specialize`` with an empty ``bind`` — folds only subcomputations that are
    already numeric in the program (a fully-symbolic program is a no-op copy).
    """
    return specialize(program)


def bind_inputs(program: Program, bind: Mapping[str, Any]) -> Program:
    """Replace inputs with concrete numeric arrays, then fold and drop them."""
    return specialize(program, bind=bind)


def _read_stmt_outs(stmt: Stmt, bindings: Mapping[str, Any]) -> list[np.ndarray] | None:
    """Read a Stmt's OUTPUT arrays back from a completed run's ``bindings`` (the dict
    ``Program.build_runtime_bindings`` returns): a bulk output under its bulk name, a
    per-cell output assembled from its cell atoms. ``None`` when any piece is absent
    (e.g. a dynamic shape this pass does not handle).
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
    """A build-time constant was certified NON-exactly (by random-probe polynomial
    identity testing) on a default/hybrid ``partial_eval_numeric`` path.

    Raised (as a warning) whenever ``mode="hybrid"`` freezes a statement the exact
    lane could not normalize — the certificate for those statements is probabilistic
    (measure-zero failure), not exact-by-construction.  Silence with
    ``mode="probe"`` (accept probing), or forbid with ``mode="exact"``.
    """


class NonDeterministicFoldWarning(NonExactFoldWarning):
    """The exact lane's WALL-CLOCK BACKSTOP fired, so this result is machine-dependent.

    The exact lane is budgeted in deterministic work units precisely so that a certificate
    means the same thing on every machine.  The clock survives only as a backstop against a
    mis-calibrated cost model on a pathological program — and if it fires, the very property
    the work budget exists to provide is gone: re-run on a faster box and more may certify.

    It is a :class:`NonExactFoldWarning` subclass so that every existing consumer of
    fold provenance (notably pointwise's certificate cache, which decides the ``exact`` bit by
    walking this hierarchy) treats it as non-exact without changes.  Seeing it means the cost
    model needs fixing, not the budget raising.
    """


_PARTIAL_EVAL_MODES = ("exact", "hybrid", "probe")


def _resolve_legacy_time_budget(time_budget: float | None) -> float | None:
    """Translate a legacy ``time_budget=`` into a wall-clock BACKSTOP, loudly.

    ``time_budget`` used to decide what the exact lane folded, in seconds — which is exactly
    the machine dependence the work budget removed.  It stays ACCEPTED because polyarray's
    committed surface carries it and pointwise has external consumers, but it no longer
    selects certificates: it now sizes the backstop, i.e. it still guarantees the call
    terminates, which is the reason callers passed it.  Because the meaning genuinely changed,
    passing it warns rather than silently doing something else than it used to.
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
    """``mode`` (or the ``POLYARRAY_PARTIAL_EVAL_MODE`` env default) validated.

    The explicit parameter is the API; the env var only moves the DEFAULT
    (``None``) — an explicit argument always wins.
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
    """One aggregated :class:`NonExactFoldWarning` naming the probe-frozen sites."""
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
    """Partial evaluation folding every Stmt whose outputs are INVARIANT under the
    program's symbolic inputs — by exact normalization, probing, or both (``mode``).

    ``fold_numeric`` folds only numeric-CLOSED subcomputations (dataflow: no symbolic
    ancestor). This pass folds the strictly larger class of subcomputations whose
    outputs merely do not DEPEND on the symbolic inputs. The canonical case: a
    chain ``inv(A) → A·inv(A)`` is identically ``I`` for every ``A`` — dataflow says
    symbolic, identity testing says constant. (The motivating case: a metric-free
    DOF's nested grass program is fed one vertex-symbolic Jacobian buffer whose
    contribution provably cancels; this pass collapses the whole ``grass_dof`` Stmt
    to its reference value, making the symbolic Vandermonde STRUCTURALLY constant.)

    ``mode`` (see :func:`partial_eval_numeric` for the public contract):

    * ``"exact"``  — :mod:`polyarray.exact_fold` only: constancy certified by the
      exact rational normal form of each output entry (flint ``fmpq`` arithmetic;
      exact-by-construction). Entries that cannot be normalized (opaque ops on the
      symbolic path) are simply NOT folded.
    * ``"hybrid"`` — exact first; statements the exact lane left *unresolved* fall
      back to the legacy probe pass, and every such non-exact freeze raises one
      aggregated :class:`NonExactFoldWarning`. Statements the exact lane REFUTED
      (provably non-constant) are never probed — this is what closes the
      colluding-probe false-freeze hole.
    * ``"probe"``  — the legacy behavior, unchanged and silent: ``probes`` random
      input bindings over ``[0.6, 1.6]``, freeze every Stmt output bit-finite and
      equal (``rtol``/``atol``) across runs. Probabilistic (measure-zero false
      freezes), NOT exact-by-construction — kept for diagnostic / performance
      call sites that don't need exactness.

    Statement granularity: an intermediate that genuinely varies (a per-probe ``Q``
    factor, say) stays symbolic even when a DOWNSTREAM output is invariant — the
    downstream Stmt still folds on its own (and the exact lane's ENTRY-level fold in
    :func:`partial_eval_numeric_symarray` also certifies cell-level cancellations).

    Static inputs only (a ``DimAtom``-shaped input raises ``NotImplementedError``).
    Inputs are never dropped — unused ones simply go unread at ``run`` time.
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
    """Fold every Stmt whose outputs are invariant under the program's symbolic
    inputs — see :func:`_partial_eval_numeric` for the mechanics.

    ``mode`` selects HOW invariance is certified:

    * ``"exact"``  — exact rational normal form only (:mod:`polyarray.exact_fold`;
      exact-by-construction). Non-normalizable statements are left symbolic.
    * ``"hybrid"`` (the default) — exact where possible; the legacy probe pass is
      the fallback for opaque/unresolved statements ONLY, and every probe-based
      freeze raises an aggregated :class:`NonExactFoldWarning` naming the sites.
      Statements the exact lane proved NON-constant are never probe-frozen.
    * ``"probe"``  — the legacy probe-and-freeze, unchanged and silent
      (probabilistic; for diagnostic/performance sites that don't need exactness).

    ``mode=None`` reads the ``POLYARRAY_PARTIAL_EVAL_MODE`` env default (else
    ``"hybrid"``); the explicit parameter always wins.  ``probes`` configures the
    probe count of the probe/hybrid-fallback lanes.  ``work_budget`` (DETERMINISTIC
    work units, charged BETWEEN operations) and ``max_sym_mass`` (the monomial mass one
    symbolic op's operands may carry, checked BEFORE it runs — an object-dtype einsum / Gauss
    pass is uninterruptible once started) JOINTLY bound the exact lane: oversized or
    out-of-budget statements degrade to the (warned) probe fallback rather than hang.
    ``max_sym_mass=None`` uses ``exact_fold._MAX_SYM_MASS``; ``work_budget=None`` uses the
    ``POLYARRAY_EXACT_WORK_BUDGET`` env knob, else ``exact_fold._DEFAULT_WORK_BUDGET``.

    The budget is work, NOT seconds: what certifies is a function of the program alone, so the
    same input yields the same certificate on any machine under any load.  See
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
    """:func:`partial_eval_numeric` for a ``SymArray`` whose CELLS reference the
    program's atoms (e.g. a symbolic Vandermonde whose cells are ``grass_dof.result``
    refs): folds the threaded program AND the cells together, so an invariant atom
    becomes a numeric cell — the STRUCTURAL form of the array.

    In the exact/hybrid modes the cells additionally get the ENTRY-LEVEL exact fold
    (:func:`polyarray.exact_fold.exact_fold_cells`): a cell is certified constant iff
    its rational normal form over the feed atoms has total degree zero — so a
    cancellation that completes only at the entry (no single statement invariant)
    still folds, exact-by-construction.  ``mode``/``probes``/``work_budget``/
    ``max_sym_mass`` as in
    :func:`partial_eval_numeric`.
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

def _as_rf(value: Any) -> RationalFunction:
    """Coerce a cell value (RF / numeric) to a :class:`RationalFunction`."""
    if isinstance(value, RationalFunction):
        return value
    if isinstance(value, (int, float)):
        return RationalFunction.constant(float(value))
    raise TypeError(f"substitute expression cell must be RF/numeric; got {value!r}")


def _build_subs_map(
    in_sa: SymArray, expr: Any,
) -> tuple[dict[str, RationalFunction], np.ndarray]:
    """Map each cell-atom of the substituted input to its replacement RF.

    Returns ``(gen_name -> repl_rf, composed_input_cells)`` where the second is
    the input's cells with each atom replaced by its expression cell (used to
    rewrite any direct ``InputRef`` to the dropped input).
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
    """Apply :meth:`RationalFunction.compose_multi` to every RF cell."""
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
    ref: Any,
    subs_map: Mapping[str, RationalFunction],
    composed_inputs: Mapping[str, np.ndarray],
) -> Any:
    """Rewrite a Stmt input ref under ``subs_map``.

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


def substitute(program: Program, subs: Mapping[str, Any]) -> Program:
    """Replace inputs with expressions over the program's *other* inputs.

    ``subs`` maps an input name to a :class:`SymArray` (per-cell, shape-matched)
    or a :class:`RationalFunction` (broadcast to every cell) whose generators are
    the program's OTHER existing input atoms.  Each substituted input's per-cell
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


def symarray_atoms(sa: SymArray) -> set:
    """The run-time binding-key atoms a ``SymArray`` reads — the PUBLIC name for what
    :func:`_symarray_atoms` has always computed.

    Exported so a consumer can ask "does this value DEPEND on that atom?" without
    reimplementing the traversal. First caller: savo's σ-channel invariant INV-3, which must show
    that the geometry reaching ``AssemblyInput.sample`` carries no orientation atom. Reading the
    cells' generators is exactly the question, and the alternative — a consumer hand-rolling it —
    is the object-dtype/cells-unwrapping the stack rules forbid.
    """
    return _symarray_atoms(sa)


def _symarray_atoms(sa: SymArray) -> set:
    """The run-time binding-key atoms a SymArray reads: its bulk name, or its
    cells' RationalFunction generators (the keys ``SymArray.evaluate`` resolves).
    """
    bulk = getattr(sa, "_bulk", None)
    if bulk is not None:
        return {bulk.name}
    cells = np.asarray(sa._cells, dtype=object)
    atoms: set = set()
    for c in (np.ravel(cells) if cells.shape else [cells[()]]):
        if isinstance(c, RationalFunction):
            atoms.update(c.gens)
    return atoms


def _dimatom_stmt(d) -> int | None:
    """The producing stmt index of a ``DimAtom`` sourced from a prior Stmt output."""
    src = getattr(d, "source", None)
    if isinstance(src, tuple) and len(src) == 3 and src[0] == "stmt":
        return int(src[1])
    if isinstance(src, tuple) and len(src) == 2 and isinstance(src[0], int):
        return int(src[0])                                   # legacy (stmt_idx, out_idx)
    return None


def _ref_deps(ref) -> tuple[set, set]:
    """The (atom names, direct stmt indices) one input Ref pulls in."""
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


def _shape_dim_stmts(shape) -> set:
    out: set = set()
    for d in shape or ():
        if not isinstance(d, int):
            s = _dimatom_stmt(d)
            if s is not None:
                out.add(s)
    return out


def dependency_cone(program: Program, target: SymArray) -> set:
    """The set of statement indices ``target`` transitively depends on (its cone)."""
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
    """Is ``target``'s value a PROVABLE build-time constant — soundly, by DEPENDENCY
    (not by sampling)?

    Walks ``target``'s dependency cone (:func:`dependency_cone` — the set of Stmts
    feeding ``target``) and returns ``True`` iff the cone is a *closed constant
    subprogram*:

    1. **No Stmt input is an** :class:`~polyarray.ir.InputRef` (a reference to a declared
       program input) — nor an :class:`~polyarray.ir.IntAtomRef` (a runtime-bound integer),
       both of which are values supplied at run time; and
    2. **no cell generator anywhere in the cone** (nor in ``target`` itself) has a
       provenance ``kind`` other than ``"stmt_out"`` — i.e. no ``vertex`` / ``point`` /
       ``coeff`` / ``param_dof`` / ``per_point`` (feed) atom is read.

    **Soundness.**  A cone with no live-input reference and only ``stmt_out`` generators
    depends on nothing outside itself: its leaves are nullary constant ops (``ConstOp`` /
    ``EyeOp`` …), every intermediate atom is produced by a prior Stmt *in the cone*, and no
    generator traces to a program feed.  Its value is therefore identical under every
    binding — a build-time constant that folds without any sampling.  This is EXACT, not
    heuristic: it never mistakes a feed-varying operator for a constant.

    **Conservative.**  Any ref type that cannot be positively classified as constant-safe
    (an :class:`~polyarray.ir.OutputRef` — a prior in-cone Stmt output — or a
    :class:`~polyarray.ir.Const` literal), or any generator whose provenance cannot be
    looked up, forces ``False``.  We never claim constancy we cannot prove, so a value with
    any live input dependency is reported non-constant.
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
    """Evaluate ``target`` by running ONLY its dependency cone at ``values``.

    Statements outside :func:`dependency_cone` are NEVER executed — so a singular
    / failing op elsewhere in ``program`` (a common hazard when probing a
    partially-built shared program at a generic feed) cannot affect, or crash,
    the evaluation of ``target``.  Returns ``target``'s float ndarray.

    Semantics: equal to ``target.evaluate(values)`` WHENEVER the full run would
    succeed; but it also succeeds when the full run would raise on an unrelated
    statement.  ``values`` must bind every program input (the unused ones simply
    go unread).
    """
    cone = dependency_cone(program, target)
    bindings = program.build_runtime_bindings(values, only=cone)
    return _read_symarray(program, target, bindings)
