"""Post-build partial evaluation of a Program (the `simplify` pass).

First cut: ``fold_numeric`` / ``bind_inputs`` — the numeric-propagation floor
described in ``plans/00-simplify-design.md`` §"fold_numeric, precisely".

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

from typing import Any, Mapping

import numpy as np

from .ir import (
    CallOp,
    Const,
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
    sub-Program (not a loop), with statically-shaped outputs (per-cell or bulk;
    a runtime-``DimAtom``-sized output cannot be materialised at build time)."""
    if stmt.fn is None:
        return False
    if isinstance(stmt.fn, _SKIP_OPS):
        return False
    for o in stmt.out:
        if o._bulk is not None and is_dynamic(o._bulk.shape):
            return False
    return True


def _exec_fn(fn: Any, resolved: list[np.ndarray]) -> list[Any]:
    """Execute a Stmt fn on concrete numeric operands, mirroring ``_run_stmt``'s
    dispatch (a raw sub-Program runs via ``.run``; everything else is called)."""
    if isinstance(fn, Program):
        value_map = {inp.name: np.asarray(v) for inp, v in zip(fn.inputs, resolved)}
        return list(fn.run(value_map).values())
    results = fn(*resolved)
    return list(results) if isinstance(results, tuple) else [results]


def _try_eval_ref(
    prog: Program, ref: Any, stmt_idx: int, known: Mapping[str, float],
) -> np.ndarray | None:
    """Resolve ``ref`` to a concrete float array using ``known``; None if any
    needed generator is still symbolic."""
    try:
        return np.asarray(prog._resolve_ref(ref, stmt_idx, dict(known)), dtype=float)
    except Exception:
        return None


def _fold_cells(cells: np.ndarray, known: Mapping[str, float]) -> np.ndarray:
    """Fold ``known`` into an ndarray of cells via partial ``RF.eval``.

    Returns a float array when every cell becomes numeric, else an object array
    of floats / smaller RationalFunctions (residual symbols)."""
    if cells.dtype.kind == "f":
        return cells.copy()
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
    symbolically-folded version (and OutputRef stmt-indices remapped)."""
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


def _record_known(stmt: Stmt, outs: list[Any], known: dict[str, Any]) -> None:
    """Record a folded Stmt's numeric outputs into ``known`` (raises on shape
    mismatch so the caller can discard a bad fold).

    A bulk output records its whole tensor under the bulk handle name (resolved
    directly by ``_resolve_ref``); a per-cell output records each cell's atom."""
    for k, bound in enumerate(stmt.out):
        arr = np.asarray(outs[k], dtype=float)
        if bound._bulk is not None:
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
      structure per ``plans/01-budget-moderated-simplification.md``.
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
            staged: dict[str, Any] = dict(known)
            _record_known(stmt, outs, staged)
        except Exception:
            continue  # any failure -> keep the Stmt symbolic
        known = staged
        foldable.add(i)

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
            new_fn = s.fn
            new_in = tuple(_fold_ref(new, r, i, known, idx_map) for r in s.in_)
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
    (e.g. a dynamic shape this pass does not handle)."""
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


def _partial_eval_numeric(
    program: Program, *, probes: int, seed: int, rtol: float, atol: float,
) -> tuple[Program, dict]:
    """Probe-and-freeze partial evaluation: fold every Stmt whose outputs are
    NUMERICALLY INVARIANT under the program's symbolic inputs.

    ``fold_numeric`` folds only numeric-CLOSED subcomputations (dataflow: no symbolic
    ancestor). This pass folds the strictly larger class of subcomputations whose
    outputs merely do not DEPEND on the symbolic inputs — discovered by running the
    program under ``probes`` random input bindings and freezing every Stmt output that
    is bit-finite and equal (``rtol``/``atol``) across all runs. The canonical case: a
    chain ``inv(A) → A·inv(A)`` is identically ``I`` for every ``A`` — dataflow says
    symbolic, probing says constant. (The FEEC motivation: a metric-free DOF's nested
    grass program is fed one vertex-symbolic Jacobian buffer whose contribution
    provably cancels; this pass collapses the whole ``grass_dof`` Stmt to its
    reference value, making the symbolic Vandermonde STRUCTURALLY constant.)

    SOUNDNESS: probabilistic (polynomial identity testing), NOT exact-by-construction
    like ``specialize`` — a cell equal at every probe by coincidence would be frozen
    wrongly. Probes are i.i.d. over a continuous box, so for the rational-function
    cells this IR produces, false freezes have measure zero; raise ``probes`` for
    belt-and-braces. Statement granularity: an intermediate that genuinely varies
    (a per-probe ``Q`` factor, say) stays symbolic even when a DOWNSTREAM output is
    invariant — the downstream Stmt still folds on its own.

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

    rng = np.random.default_rng(seed)
    runs: list[Mapping[str, Any]] = []
    for _ in range(probes):
        vals = {inp.name: rng.uniform(0.6, 1.6, size=tuple(inp.shape))
                for inp in program.inputs}
        runs.append(program.build_runtime_bindings(vals))

    per_probe_outs: list[list[list[np.ndarray] | None]] = [
        [_read_stmt_outs(stmt, b) for stmt in program.statements] for b in runs
    ]

    new = program.copy()
    known: dict[str, Any] = {}
    foldable: set[int] = set()
    for i, stmt in enumerate(new.statements):
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
        try:
            staged = dict(known)
            _record_known(stmt, outs0, staged)
        except ValueError:
            continue
        known = staged
        foldable.add(i)

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
    return new, known


def partial_eval_numeric(
    program: Program, *, probes: int = 3, seed: int = 0,
    rtol: float = 1e-9, atol: float = 1e-12,
) -> Program:
    new, _known = _partial_eval_numeric(program, probes=probes, seed=seed, rtol=rtol, atol=atol)
    return new


def partial_eval_numeric_symarray(
    sa: SymArray, *, probes: int = 3, seed: int = 0,
    rtol: float = 1e-9, atol: float = 1e-12,
) -> SymArray:
    """:func:`partial_eval_numeric` for a ``SymArray`` whose CELLS reference the
    program's atoms (e.g. a symbolic Vandermonde whose cells are ``grass_dof.result``
    refs): folds the threaded program AND the cells together, so an invariant atom
    becomes a numeric cell — the STRUCTURAL form of the array."""
    from .ir import SymArray
    if sa.program is None:
        return sa
    new, known = _partial_eval_numeric(sa.program, probes=probes, seed=seed, rtol=rtol, atol=atol)
    return SymArray(_fold_cells(np.asarray(sa.cells), known), program=new)


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
