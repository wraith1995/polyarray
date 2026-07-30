"""EXACT (non-sampling) statement/entry folding for ``partial_eval_numeric``.

``simplify._partial_eval_numeric`` historically certified "this output does not depend
on the symbolic inputs" by PROBE-AND-FREEZE polynomial identity testing — random probe
bindings + ``np.allclose`` — documented as probabilistic, NOT exact-by-construction.
Since the ``P(T)=I`` affine-invariance certificate started riding on that fold
(pointwise ``pti.affine_invariance``), a sound lane is required: this module is the
EXACT lane behind ``mode="exact"`` / ``mode="hybrid"``.

**Semantics.** The program's dependency graph is re-executed over EXACT values:

* a statement whose resolved operands are all NUMERIC runs its real ``fn`` — the same
  deterministic evaluation ``Program.run`` would perform (the ``fold_numeric``
  exactness contract), so QR/SVD frame preps on constant reference data stay foldable;
* a statement with symbolic operands is executed over exact
  :class:`~polyarray.rational.RationalFunction` cells (flint ``fmpq`` coefficients —
  exact rational arithmetic) through a closed set of RATIONAL op twins: einsum /
  tensordot / transpose / reshape / add / scale / concat / stack, and exact
  field-arithmetic Gauss elimination for ``inv`` / ``inv-transpose`` / ``solve`` /
  ``det``; sub-``Program`` / ``CallOp(Program)`` bodies are recursed;
* everything else on a symbolic path — ``QrOp`` / ``SvdOp`` / ``SqrtOp`` /
  control-flow / vmap closures / front-end ops polyarray does not own (they coerce
  ``dtype=float`` and carry semantics of the layer above) — is OPAQUE: its outputs and
  everything downstream are *unresolved*, never guessed.

A statement all of whose output entries resolve to exact CONSTANTS is *folded* (the
certificate is exact-by-construction: a rational normal form of total degree zero).
A statement with a provably NON-constant output entry is *refuted* — soundly excluded
from any probe fallback (this is precisely the case where colluding probes could have
frozen a vertex-dependent value).  Only *unresolved* statements are eligible for the
(warned) probe fallback in ``mode="hybrid"``.

Entry-level folding (:func:`exact_fold_cells`) additionally normalizes each OUTPUT
CELL as a rational function of the feed atoms — so an entry whose statement-level
pieces vary but whose composition cancels (the FEEC motivation) still certifies,
without any statement being frozen.

The pass is TIME-BOXED (``time_budget`` seconds): rational normal forms can be
expensive, and a pathological entry must degrade to the (loud) probe fallback rather
than hang the gate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .ir import (
    AddOp,
    CallOp,
    ColStackOp,
    ConcatOp,
    Const,
    DetOp,
    EinsumOp,
    EinsumStmtOp,
    HStackOp,
    IdentityOp,
    InputRef,
    IntAtomRef,
    InvOp,
    InvTransposeOp,
    MoveaxisOp,
    OutputRef,
    Program,
    RationalRef,
    ReshapeOp,
    ScaleByOp,
    ScaleOp,
    SolveOp,
    Stmt,
    SymArrayRef,
    TensordotOp,
    TransposeOp,
    WhileOp,
    is_dynamic,
)
from .rational import RationalFunction

# Sentinel: a value the exact lane cannot resolve (opaque op / unresolved atom /
# dynamic shape / timeout).  Never a valid array value.
_OPAQUE = object()

# Recursion ceiling for sub-Program descent (mirrors simplify._MAX_DESCENT_DEPTH).
_MAX_DEPTH = 32


class _Timeout(Exception):
    """Internal: the exact pass ran out of its time budget."""


class _Reason:
    """Deepest-first record of WHY the current statement went opaque (for the
    hybrid-mode warning): the first op that could not be executed symbolically
    wins — e.g. the QR / front-end sign-fix inside a ``grass_dof`` sub-program,
    not the outer ``Program`` wrapper.  One instance is created PER
    :func:`exact_partial_eval` call and threaded explicitly (no module global —
    thread-safe, and an escaping exception cannot leak a stale note)."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: str | None = None

    def note(self, reason: str) -> None:
        if self.value is None:
            self.value = reason

    def take(self, fallback: str) -> str:
        v = self.value if self.value is not None else fallback
        self.value = None
        return v


@dataclass
class ExactState:
    """Result of one exact pass over a program.

    ``known``     — atom / bulk name → exact-constant value (float or float ndarray),
                    exactly the shape ``simplify._record_known`` produces;
    ``sym``       — atom name → exact but NON-constant :class:`RationalFunction` over
                    the feed atoms (statement-level exact values, used by the
                    entry-level cell fold);
    ``sym_bulk``  — bulk name → object ndarray of exact cells (float / RF);
    ``folded``    — statement indices folded exactly (every output entry constant);
    ``refuted``   — statement indices whose outputs are PROVABLY non-constant
                    (excluded from any probe fallback);
    ``unresolved``— statement index → short reason (opaque op name, "time budget", …).
    """

    known: dict[str, Any] = field(default_factory=dict)
    sym: dict[str, RationalFunction] = field(default_factory=dict)
    sym_bulk: dict[str, np.ndarray] = field(default_factory=dict)
    folded: set[int] = field(default_factory=set)
    refuted: set[int] = field(default_factory=set)
    unresolved: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _as_numeric(val: Any) -> np.ndarray | None:
    """``val`` as a float ndarray when it carries no symbolic cell, else ``None``."""
    arr = np.asarray(val)
    if arr.dtype.kind in "fiub":
        return arr.astype(float)
    flat = arr.reshape(-1) if arr.shape else [arr[()]]
    for c in flat:
        if isinstance(c, RationalFunction):
            return None
        if not isinstance(c, (int, float, np.integer, np.floating)):
            return None
    return arr.astype(float)


def _exact_eval_at(rf: RationalFunction, k: int) -> RationalFunction:
    """``rf`` EXACTLY evaluated at deterministic generic point #``k`` (distinct per
    generator; no RNG): every generator is composed with an exact DYADIC rational
    constant (``0.5 + 0.25·m`` — float→fmpq conversion is exact), so the result is
    an exact constant :class:`RationalFunction`, with no float rounding anywhere.
    Raises when the point is singular (denominator vanishes there)."""
    sub = {g: RationalFunction.constant(0.5 + 0.25 * ((7 * i + 3 * k + 1) % 11))
           for i, g in enumerate(sorted(rf.gens))}
    return rf.compose_multi(sub)


def _check_deadline(deadline: float | None) -> None:
    """Raise :class:`_Timeout` when ``deadline`` (monotonic seconds) has passed.

    The time-box granularity is PER OPERATION: the check runs before each exact
    evaluation / gcd, so one already-started flint operation can still overrun —
    accepted; the budget bounds the number of such operations started."""
    if deadline is not None and time.monotonic() > deadline:
        raise _Timeout()


def _constant_value(rf: RationalFunction, deadline: float | None = None) -> float | None:
    """The exact float value of ``rf`` when it is IDENTICALLY constant, else ``None``.

    Constancy is decided by the exact rational normal form: gcd-cancel (flint exact
    division) then total-degree-zero of numerator and denominator.  A sound EXACT
    filter avoids paying the gcd on varying cells: ``rf`` is evaluated at two
    deterministic points in EXACT fmpq arithmetic (:func:`_exact_eval_at` — never
    float term-summation, whose rounding would falsely refute a constant like
    ``(1/3)·p/p``); two exactly-different values prove non-constancy, while equal
    values are only a hint — the ``clean`` normal form still decides.

    DEADLINE-AWARE: every expensive step (each exact evaluation, the gcd) is
    preceded by a ``deadline`` check that raises :class:`_Timeout` — a pathological
    cell (an enormous uncancelled quintic; the Argyris regime) must degrade to
    *unresolved* (⇒ the warned probe fallback), never hang the gate."""
    if rf.is_zero():
        return 0.0                           # exact structural zero (num ≡ 0)
    if not rf.gens:
        return float(rf.eval({}))
    if rf.is_constant():
        return float(rf.eval({g: 0.0 for g in rf.gens}))
    try:
        _check_deadline(deadline)
        e0 = _exact_eval_at(rf, 0)
        _check_deadline(deadline)
        e1 = _exact_eval_at(rf, 1)
        if e0 != e1:                         # exact inequality ⇒ provably non-constant
            return None
    except _Timeout:
        raise
    except Exception:  # noqa: BLE001 — singular point etc.: the normal form decides
        pass
    _check_deadline(deadline)
    cleaned = rf.clean()
    if cleaned.is_zero():
        return 0.0
    if cleaned.is_constant():
        return float(cleaned.eval({g: 0.0 for g in cleaned.gens}))
    return None


def _cell_constant(cell: Any, deadline: float | None = None) -> float | None:
    """Exact constant value of one cell (float / RF), else ``None``.

    Raises :class:`_Timeout` (via :func:`_constant_value`) on deadline expiry."""
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return float(cell)
    if isinstance(cell, RationalFunction):
        return _constant_value(cell, deadline)
    return None


# ---------------------------------------------------------------------------
# Exact executor
# ---------------------------------------------------------------------------

@dataclass
class _Env:
    """Per-program symbolic environment: exact values for atoms / bulks / stmt outs."""

    atom: dict[str, Any] = field(default_factory=dict)      # name -> float | RF
    bulk: dict[str, Any] = field(default_factory=dict)      # bulk name -> ndarray
    outs: dict[tuple[int, int], Any] = field(default_factory=dict)


def _resolve_rf(rf: RationalFunction, env: _Env, program: Program) -> Any:
    """``rf`` with every resolvable stmt-out atom substituted by its exact value.

    Feed atoms (vertex / point / coeff provenance) stay symbolic generators; a
    stmt-out atom without an exact value makes the whole cell unresolvable."""
    sub: dict[str, RationalFunction] = {}
    for g in rf.gens:
        if g in env.atom:
            v = env.atom[g]
            sub[g] = v if isinstance(v, RationalFunction) \
                else RationalFunction.constant(float(v))
            continue
        try:
            kind = program.env.of(g).kind
        except KeyError:
            return _OPAQUE                    # unknown generator — cannot prove anything
        if kind == "stmt_out":
            return _OPAQUE                    # produced by an unresolved statement
        # feed atom — remains a live generator of the entry's normal form
    return rf.compose_multi(sub) if sub else rf


def _resolve_cells(cells: np.ndarray, env: _Env, program: Program) -> Any:
    """Resolve an ndarray of cells to exact values (object ndarray), or ``_OPAQUE``."""
    cells = np.asarray(cells)
    if cells.dtype.kind == "f":
        return cells
    out = np.empty(cells.shape, dtype=object)
    flat_in = cells.reshape(-1)
    flat_out = out.reshape(-1)
    for i, c in enumerate(flat_in):
        if isinstance(c, RationalFunction):
            v = _resolve_rf(c, env, program)
            if v is _OPAQUE:
                return _OPAQUE
            flat_out[i] = v
        elif isinstance(c, (int, float, np.integer, np.floating)):
            flat_out[i] = float(c)
        else:
            return _OPAQUE
    return out


def _resolve_ref(ref: Any, env: _Env, program: Program) -> Any:
    """Resolve one Stmt input ref to an exact value (ndarray), or ``_OPAQUE``."""
    if isinstance(ref, Const):
        return np.asarray(ref.value, dtype=float)
    if isinstance(ref, InputRef):
        sa = program.input_arrays.get(ref.name)
        if sa is None or getattr(sa, "_bulk", None) is not None:
            return _OPAQUE                    # dynamic / bulk input — not exact-resolvable
        val = _resolve_cells(np.asarray(sa.cells), env, program)
        if val is _OPAQUE:
            return _OPAQUE
        return val[ref.indices] if ref.indices else val
    if isinstance(ref, OutputRef):
        val = env.outs.get((ref.stmt_idx, ref.out_idx), _OPAQUE)
        if val is _OPAQUE:
            return _OPAQUE
        return val[ref.indices] if ref.indices else val
    if isinstance(ref, SymArrayRef):
        if ref._bulk is not None:
            return env.bulk.get(ref._bulk.name, _OPAQUE)
        return _resolve_cells(np.asarray(ref._cells), env, program)
    if isinstance(ref, RationalRef):
        v = _resolve_rf(ref.rf, env, program)
        return _OPAQUE if v is _OPAQUE else np.asarray(v, dtype=object)
    if isinstance(ref, IntAtomRef):
        return _OPAQUE                        # runtime-bound integer — not build-time exact
    return _OPAQUE


# --- exact field arithmetic (Gauss) ----------------------------------------

def _fe_is_zero(x: Any) -> bool:
    return x.is_zero() if isinstance(x, RationalFunction) else float(x) == 0.0


def _exact_inv(a: np.ndarray) -> np.ndarray:
    """Exact Gauss–Jordan inverse over the rational-function field.

    Pivots are chosen exactly (structurally non-zero — a non-zero rational function
    is a unit of the field); the resulting entries are the inverse AS RATIONAL
    FUNCTIONS, i.e. the identity ``A·A⁻¹ = I`` holds identically wherever
    ``det A ≠ 0`` — the correct generic-inverse semantics for identity
    certification.  Raises on an exactly singular matrix."""
    a = np.asarray(a, dtype=object)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"exact inverse needs a square matrix, got {a.shape}")
    n = a.shape[0]
    m = np.empty((n, 2 * n), dtype=object)
    m[:, :n] = a
    m[:, n:] = 0.0
    for i in range(n):
        m[i, n + i] = 1.0
    for col in range(n):
        piv = None
        for row in range(col, n):
            if not _fe_is_zero(m[row, col]):
                # prefer a plain-numeric pivot (cheaper downstream arithmetic)
                if piv is None:
                    piv = row
                if not isinstance(m[row, col], RationalFunction):
                    piv = row
                    break
        if piv is None:
            raise ZeroDivisionError("exact inverse: structurally singular matrix")
        if piv != col:
            m[[col, piv]] = m[[piv, col]]
        p = m[col, col]
        for j in range(2 * n):
            m[col, j] = m[col, j] / p
        for row in range(n):
            if row == col or _fe_is_zero(m[row, col]):
                continue
            f = m[row, col]
            for j in range(2 * n):
                m[row, j] = m[row, j] - f * m[col, j]
    return m[:, n:]


def _exact_det(a: np.ndarray) -> Any:
    """Exact determinant by fraction-free-ish Gauss elimination over the field."""
    a = np.asarray(a, dtype=object).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"exact det needs a square matrix, got {a.shape}")
    n = a.shape[0]
    det: Any = 1.0
    for col in range(n):
        piv = next((r for r in range(col, n) if not _fe_is_zero(a[r, col])), None)
        if piv is None:
            return 0.0
        if piv != col:
            a[[col, piv]] = a[[piv, col]]
            det = det * -1.0
        p = a[col, col]
        det = det * p
        for row in range(col + 1, n):
            if _fe_is_zero(a[row, col]):
                continue
            f = a[row, col] / p
            for j in range(col, n):
                a[row, j] = a[row, j] - f * a[col, j]
    return det


def _exact_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=object)
    inv = _exact_inv(a)
    if b.ndim == 1:
        return np.einsum("ij,j->i", inv, b, optimize=False)
    return np.einsum("ij,j...->i...", inv, b, optimize=False)


def _obj(a: Any) -> np.ndarray:
    """An operand as an object ndarray (einsum over mixed float/RF cells)."""
    arr = np.asarray(a)
    return arr if arr.dtype == object else arr.astype(object)


# --- symbolic op twins ------------------------------------------------------

def _sym_apply(
    fn: Any, args: list[Any], deadline: float, depth: int, reason: _Reason,
) -> list[Any] | None:
    """Execute one op over exact symbolic operands; ``None`` = opaque (no guess)."""
    if isinstance(fn, EinsumStmtOp):
        return [np.einsum(fn.spec, *[_obj(a) for a in args], optimize=False)]
    if isinstance(fn, EinsumOp):
        rhs = np.frombuffer(fn.rhs_bytes, dtype=fn.rhs_dtype).reshape(fn.rhs_shape)
        return [np.einsum(fn.spec, _obj(args[0]), _obj(rhs), optimize=False)]
    if isinstance(fn, TensordotOp):
        return [np.tensordot(_obj(args[0]), _obj(args[1]), axes=fn.axes)]
    if isinstance(fn, TransposeOp):
        return [np.asarray(args[0]).T]
    if isinstance(fn, MoveaxisOp):
        return [np.moveaxis(np.asarray(args[0]), fn.source, fn.destination)]
    if isinstance(fn, ReshapeOp):
        return [np.asarray(args[0]).reshape(fn.shape)]
    if isinstance(fn, IdentityOp):
        return [np.asarray(args[0])]
    if isinstance(fn, AddOp):
        out = _obj(args[0]).copy()
        for x in args[1:]:
            out = out + _obj(x)
        return [out]
    if isinstance(fn, ScaleOp):
        return [fn.factor * _obj(args[0])]
    if isinstance(fn, ScaleByOp):
        return [_obj(args[1]) * _obj(args[0])]
    if isinstance(fn, ConcatOp):
        return [np.concatenate([_obj(x).reshape(-1) for x in args])]
    if isinstance(fn, HStackOp):
        return [np.hstack([_obj(m) for m in args])]
    if isinstance(fn, ColStackOp):
        return [np.stack([_obj(c).reshape(-1) for c in args], axis=1)]
    if isinstance(fn, InvOp):
        return [_exact_inv(args[0])]
    if isinstance(fn, InvTransposeOp):
        return [_exact_inv(args[0]).T]
    if isinstance(fn, SolveOp):
        return [_exact_solve(args[0], args[1])]
    if isinstance(fn, DetOp):
        return [np.asarray(_exact_det(args[0]), dtype=object)]
    if isinstance(fn, Program):
        return _run_program(fn, args, deadline, depth + 1, reason)
    if isinstance(fn, CallOp) and isinstance(fn.fn, Program):
        return _run_program(fn.fn, args, deadline, depth + 1, reason)
    name = type(fn).__name__
    if isinstance(fn, CallOp):
        inner = fn.fn
        name = f"CallOp({'vmap closure' if hasattr(inner, '_vmap_body') else type(inner).__name__})"
    elif hasattr(fn, "_vmap_body"):
        name = f"vmap closure ({name})"
    reason.note(f"opaque op {name} on the symbolic path")
    return None                               # opaque: QR / SVD / sqrt / vmap / front-end op


def _bind_body_inputs(body: Program, args: list[Any], env: _Env) -> bool:
    """Bind operand values onto a sub-program's input atoms; False on mismatch."""
    if len(args) != len(body.inputs):
        return False
    for inp, val in zip(body.inputs, args):
        sa = body.input_arrays.get(inp.name)
        if sa is None or getattr(sa, "_bulk", None) is not None:
            return False                      # bulk / dynamic body input — opaque
        cells = np.asarray(sa.cells)
        arr = np.asarray(val)
        if tuple(arr.shape) != tuple(cells.shape):
            return False
        shape = cells.shape
        for idx in (np.ndindex(*shape) if shape else [()]):
            cell = cells[idx] if shape else cells[()]
            if isinstance(cell, RationalFunction) and cell.gens:
                env.atom[cell.gens[0]] = arr[idx] if shape else arr[()]
    return True


def _run_program(
    body: Program, args: list[Any], deadline: float, depth: int, reason: _Reason,
) -> list[Any] | None:
    """Recursively execute a sub-program over exact values; ``None`` = opaque."""
    if depth > _MAX_DEPTH:
        reason.note("sub-program descent depth cap")
        return None
    env = _Env()
    if not _bind_body_inputs(body, args, env):
        reason.note("sub-program input binding mismatch (bulk/dynamic input or shape)")
        return None
    for i, stmt in enumerate(body.statements):
        if not _exec_stmt(body, i, stmt, env, deadline, depth, reason):
            return None                       # one opaque stmt poisons the body — no guess
    outs: list[Any] = []
    for sa in body.outputs.values():
        bulk = getattr(sa, "_bulk", None)
        if bulk is not None:
            val = env.bulk.get(bulk.name, _OPAQUE)
        else:
            val = _resolve_cells(np.asarray(sa.cells), env, body)
        if val is _OPAQUE:
            return None
        outs.append(val)
    return outs


def _exec_stmt(
    program: Program, i: int, stmt: Stmt, env: _Env, deadline: float, depth: int,
    reason: _Reason,
) -> bool:
    """Execute one statement into ``env``; False when opaque/unresolvable."""
    if time.monotonic() > deadline:
        raise _Timeout()
    if stmt.fn is None or isinstance(stmt.fn, WhileOp):
        reason.note("WhileOp / fn-less statement (never executed at build time)")
        return False                          # loops are never executed at build time
    args: list[Any] = []
    for r in stmt.in_:
        v = _resolve_ref(r, env, program)
        if v is _OPAQUE:
            reason.note("operand depends on an unresolved statement / runtime-only ref")
            return False
        args.append(v)
    numeric = [_as_numeric(a) for a in args]
    outs: list[Any] | None
    if all(n is not None for n in numeric):
        # Numeric-closed: run the REAL op — deterministic, the fold_numeric contract.
        if any(o._bulk is not None and is_dynamic(o._bulk.shape) for o in stmt.out):
            reason.note("dynamic output shape (runtime-ranked)")
            return False                      # dynamic output shape — leave to the runtime
        try:
            from .simplify import _exec_fn
            outs = _exec_fn(stmt.fn, [n for n in numeric if n is not None])
        except Exception:  # noqa: BLE001 — a failing fold is a non-fold, never a crash
            reason.note(f"{type(stmt.fn).__name__} raised on numeric operands")
            return False
    else:
        try:
            outs = _sym_apply(stmt.fn, args, deadline, depth, reason)
        except _Timeout:
            raise
        except Exception as exc:  # noqa: BLE001 — symbolic twin failed ⇒ opaque, never a guess
            reason.note(f"symbolic {type(stmt.fn).__name__} failed ({type(exc).__name__})")
            return False
    if outs is None or len(outs) != len(stmt.out):
        return False
    # Record outputs onto the env.
    for k, (bound, val) in enumerate(zip(stmt.out, outs)):
        arr = np.asarray(val)
        if bound._bulk is not None:
            if is_dynamic(bound._bulk.shape) or tuple(arr.shape) != tuple(bound._bulk.shape):
                reason.note("bulk output shape mismatch / dynamic")
                return False
            env.bulk[bound._bulk.name] = arr
            env.outs[(i, k)] = arr
            continue
        cells = np.asarray(bound.cells)
        if tuple(arr.shape) != tuple(cells.shape):
            reason.note("output cell shape mismatch")
            return False
        shape = cells.shape
        for idx in (np.ndindex(*shape) if shape else [()]):
            cell = cells[idx] if shape else cells[()]
            if isinstance(cell, RationalFunction) and cell.gens:
                env.atom[cell.gens[0]] = arr[idx] if shape else arr[()]
        env.outs[(i, k)] = arr
    return True


# ---------------------------------------------------------------------------
# Entry points (consumed by simplify)
# ---------------------------------------------------------------------------

def exact_partial_eval(program: Program, *, time_budget: float) -> ExactState:
    """One exact pass over ``program``: fold / refute / leave-unresolved each Stmt.

    Never raises on op content — every failure mode degrades to ``unresolved``
    (which ``mode="hybrid"`` hands to the loud probe fallback)."""
    from .simplify import _record_known

    deadline = time.monotonic() + float(time_budget)
    state = ExactState()
    env = _Env()
    reason = _Reason()                         # per-call holder (threaded explicitly)
    for i, stmt in enumerate(program.statements):
        reason.value = None                    # fresh slate per top-level statement
        try:
            ok = _exec_stmt(program, i, stmt, env, deadline, 0, reason)
        except _Timeout:
            for j in range(i, len(program.statements)):
                state.unresolved[j] = "time budget exhausted"
            break
        if not ok:
            fallback = type(stmt.fn).__name__ if stmt.fn is not None else "no fn"
            state.unresolved[i] = reason.take(fallback)
            continue
        # Classify: all-constant ⇒ fold; any provably non-constant cell ⇒ refute.
        # DEADLINE-AWARE (the classification is where a pathological cell's exact
        # evaluation/gcd bill lands — the Argyris regime): expiry degrades this and
        # every remaining statement to *unresolved* (⇒ the warned probe fallback),
        # exactly like a timeout in the execution loop above.
        const_outs: list[np.ndarray] = []
        verdict = "fold"
        try:
            for k in range(len(stmt.out)):
                arr = env.outs[(i, k)]
                num = _as_numeric(arr)
                if num is not None:
                    const_outs.append(num)
                    continue
                vals = np.empty(arr.shape, dtype=float)
                for idx in (np.ndindex(*arr.shape) if arr.shape else [()]):
                    c = arr[idx] if arr.shape else arr[()]
                    cv = _cell_constant(c, deadline)
                    if cv is None:
                        verdict = "refute"
                        break
                    vals[idx] = cv
                if verdict != "fold":
                    break
                const_outs.append(vals)
        except _Timeout:
            for j in range(i, len(program.statements)):
                state.unresolved[j] = "time budget exhausted"
            break
        if verdict == "fold":
            try:
                staged = dict(state.known)
                _record_known(stmt, const_outs, staged)
            except ValueError:
                state.unresolved[i] = "fold shape mismatch"
                continue
            state.known = staged
            state.folded.add(i)
            # Downstream exact execution should see the folded constants.
            for k, num in enumerate(const_outs):
                env.outs[(i, k)] = num.astype(object)
                bound = stmt.out[k]
                if bound._bulk is None:
                    cells = np.asarray(bound.cells)
                    shape = cells.shape
                    for idx in (np.ndindex(*shape) if shape else [()]):
                        cell = cells[idx] if shape else cells[()]
                        if isinstance(cell, RationalFunction) and cell.gens:
                            env.atom[cell.gens[0]] = float(num[idx] if shape else num[()])
        else:
            state.refuted.add(i)
            for k, bound in enumerate(stmt.out):
                if bound._bulk is not None:
                    state.sym_bulk[bound._bulk.name] = env.outs[(i, k)]
                    continue
                cells = np.asarray(bound.cells)
                shape = cells.shape
                arr = env.outs[(i, k)]
                for idx in (np.ndindex(*shape) if shape else [()]):
                    cell = cells[idx] if shape else cells[()]
                    if isinstance(cell, RationalFunction) and cell.gens:
                        v = arr[idx] if shape else arr[()]
                        if isinstance(v, RationalFunction):
                            state.sym[cell.gens[0]] = v
                        else:
                            state.known.setdefault(cell.gens[0], float(v))
    return state


def exact_fold_cells(
    cells: np.ndarray, state: ExactState, program: Program, *,
    time_budget: float,
) -> np.ndarray:
    """ENTRY-LEVEL exact fold of an output cell array.

    Each cell is normalized as a rational function of the FEED atoms by substituting
    the exact (possibly non-constant) statement values from ``state``; a cell whose
    normal form has total degree zero is replaced by its exact constant.  Cells that
    cannot be normalized (opaque statements in their cone) or that are genuinely
    non-constant are left UNTOUCHED — including object identity when nothing folds
    (the ``_fold_cells`` structure-transparency contract)."""
    cells = np.asarray(cells)
    if cells.dtype.kind == "f":
        return cells
    deadline = time.monotonic() + float(time_budget)
    scalar_known = {k: v for k, v in state.known.items()
                    if not isinstance(v, np.ndarray) or v.shape == ()}
    env = _Env(atom={**scalar_known, **state.sym}, bulk=dict(state.sym_bulk))
    out = np.empty(cells.shape, dtype=object)
    flat_in = cells.reshape(-1)
    flat_out = out.reshape(-1)
    flat_out[:] = flat_in                     # pre-fill: a deadline break must leave no hole
    changed = False
    for i, c in enumerate(flat_in):
        if not isinstance(c, RationalFunction):
            continue
        if time.monotonic() > deadline:
            break
        v = _resolve_rf(c, env, program)
        if v is _OPAQUE or not isinstance(v, RationalFunction):
            continue
        try:
            cv = _constant_value(v, deadline)  # deadline INSIDE the cell too — a single
        except _Timeout:                       # pathological normal form must not overrun
            break                              # (remaining cells stay untouched)
        if cv is not None:
            flat_out[i] = cv
            changed = True
    return out if changed else cells
