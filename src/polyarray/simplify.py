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
outputs, sub-Program fns, control-flow ops) is kept symbolic, so the worst case
degrades to ``copy()``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .ir import (
    Const,
    InputRef,
    IntAtomRef,
    OutputRef,
    Program,
    RationalRef,
    Stmt,
    SymArray,
    SymArrayRef,
    is_dynamic,
)
from .rational import RationalFunction

# Ops we do not execute at build time.  ``WhileOp`` could loop; ``CallOp`` and
# raw sub-Programs ARE foldable (executed when every operand is numeric — see
# ``_exec_fn``).  ``SwitchOp`` is fine too: it only resolves once its IntAtom
# selector is bound, otherwise its inputs stay symbolic and it survives.
_SKIP_OP_NAMES = frozenset({"WhileOp"})


# ---------------------------------------------------------------------------
# Predicates / small helpers
# ---------------------------------------------------------------------------

def _simple_stmt(stmt: Stmt) -> bool:
    """A Stmt we may try to execute at build time: a callable fn / typed Op /
    sub-Program (not a loop), with statically-shaped outputs (per-cell or bulk;
    a runtime-``DimAtom``-sized output cannot be materialised at build time)."""
    if stmt.fn is None:
        return False
    if type(stmt.fn).__name__ in _SKIP_OP_NAMES:
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
# Entry points
# ---------------------------------------------------------------------------

def specialize(program: Program, *, bind: Mapping[str, Any] | None = None) -> Program:
    """Partially evaluate ``program`` against optional numeric ``bind`` values.

    Folds every build-time-numeric subcomputation, drops the Stmts that produced
    it, and drops inputs replaced by a ``bind`` value.  Exactness-preserving.
    """
    new = program.copy()
    known, dropped = _seed_bind(new, bind or {})

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
        new_in = tuple(_fold_ref(new, r, i, known, idx_map) for r in s.in_)
        new_statements.append(
            Stmt(fn=s.fn, in_=new_in, out=s.out, note=s.note,
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
