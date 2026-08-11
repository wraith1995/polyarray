"""Exact, non-sampling statement and entry folding for ``partial_eval_numeric``.

This is the sound lane behind ``mode="exact"`` / ``mode="hybrid"``. The alternative,
probe-and-freeze polynomial identity testing — random probe bindings plus
``np.allclose`` — is probabilistic, not exact by construction, and so cannot carry a
certificate that a downstream invariance claim rides on.

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
pieces vary but whose composition cancels (the motivating case) still certifies,
without any statement being frozen.

**Bounded cost — TWO knobs, both required.** ``work_budget`` (:class:`_Meter` work units,
charged BETWEEN operations) bounds the pass as a whole, and ``max_sym_mass``
(:data:`_MAX_SYM_MASS`) caps the monomial mass of ONE symbolic op's operands BEFORE it runs.
The global budget alone is not enough: a single ``np.einsum`` / Gauss pass over object-dtype
cells runs to completion once started, so a degree-5 element could spend an hour inside one
uninterrupted op while nominally under budget.  Rejected-by-size and out-of-budget statements
both degrade to *unresolved* ⇒ the (loud) probe fallback — never a hang, never a guess.

**The budget is DETERMINISTIC — that is the point.**  It used to be wall-clock seconds, which
made the certificate a property of the machine rather than of the mathematics: the same program
certified exactly on a quiet box and probe-backed on a loaded one.  Work units are a function
of the program alone, so a certificate means the same thing everywhere.  A generous wall-clock
*backstop* survives only to catch a mis-calibrated cost model, and it is loud when it fires
(:attr:`ExactState.hit_wall_clock`) precisely because it reintroduces machine dependence.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, assert_never

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
    DimAtom,
    Ref,
    Stmt,
    StmtFn,
    StmtOp,
    SumDimOp,
    SumShapeOp,
    SvdOp,
    SwitchOp,
    SymArrayRef,
    TensordotOp,
    TransposeOp,
    WhileOp,
    is_builtin_op,
    is_dynamic,
)
from .rational import RationalFunction

#: One exact cell: a Python float, or a rational function over the surviving feed atoms.
type ExactCell = float | RationalFunction

#: One exact operand: an array of :data:`ExactCell`, a bare cell, or the
#: :data:`_OPAQUE` sentinel when the value could not be resolved exactly.
type ExactValue = np.ndarray | ExactCell | _Opaque


# Sentinel: a value the exact lane cannot resolve — an opaque op, an unresolved atom, a
# dynamic shape, a timeout.  Never a valid array value.
class _Opaque:
    """Sentinel type for a value this lane could not resolve exactly."""

    __slots__ = ()


#: The single :class:`_Opaque` instance. A value equal to it, and everything downstream
#: of it, is *unresolved* — never guessed.
_OPAQUE = _Opaque()

# Recursion ceiling for sub-Program descent (mirrors simplify._MAX_DESCENT_DEPTH).
_MAX_DEPTH = 32

# WORK CAP for ONE exact symbolic op: the total monomial mass (numerator + denominator
# terms, summed over every symbolic cell) its operands may carry.  The time budget alone
# cannot bound the exact lane, because a single ``np.einsum`` / Gauss elimination over
# object-dtype ``RationalFunction`` cells runs to completion once STARTED, inside a single
# budget check — a degree-5, 18-DOF symbolic Vandermonde stalled the lowering gate for
# >55 min in one such einsum.  An op whose operands exceed the cap is
# declared *unresolved* up front (⇒ the warned probe fallback), so the exact lane's cost
# per statement is bounded BEFORE the expensive work starts, not merely interrupted
# after it.  Rationale for the size: the entries this lane certifies are small by
# nature (a vertex-rational Vandermonde entry measures in the tens of monomials); a five-figure operand mass means the exact route has already lost
# to the probe route, whatever the wall clock says.
_MAX_SYM_MASS = 4096

# WORK CAP for ONE vmap-closure descent: the number of BODY EXECUTIONS it may start.
# Descending a vmap multiplies the exact work by the batch size — the batched operand's
# monomial mass is still bounded by ``_MAX_SYM_MASS`` (that is the SUM over the slices, so the
# total symbolic work stays inside the same box), but each slice also pays the body's fixed
# per-statement overhead, and one slice's flint arithmetic cannot be interrupted.  So the batch
# itself is bounded BEFORE the loop starts (the stall lesson: an uninterruptible op
# ignores any budget).  Over the cap ⇒ *unresolved* ⇒ the warned probe fallback.
_MAX_VMAP_BATCH = 512

# GLOBAL WORK BUDGET for one exact pass, in *work units* — a deterministic function of the
# program, never of the machine.  One unit is roughly one monomial touched: a statement costs
# `_STMT_COST` plus the monomial mass of the operands it actually processes.  See `_Meter`.
#
# WHY NOT SECONDS.  This budget used to be `time_budget`, wall-clock seconds off
# `time.monotonic()`.  That made the CERTIFICATE a property of the machine: the same program on
# the same input certified exactly on a quiet box and probe-backed on a loaded one — measured
# 6874 / 6895 / 7726 frozen statements across three runs of ONE leg.  Since a certificate is
# persisted and selects a whole compilation lane downstream, "this element certifies exactly"
# was recording the load average as much as the mathematics.  Work units are reproducible: the
# same program spends the same number, on any box, under any load, in any order.
#
# SIZED FROM MEASUREMENT, not guessed.  Calibration over the symbolic P(T) legs (TRI/TET,
# r = 1..4, k = 1,2) gives a strikingly stable ~128 000 work units per second (123 k–158 k
# across every leg), so the old nominal 10 s ceiling was worth ~1.3 M units on a quiet box.
# This is ~3x that, which is the "no leg certifies less than it used to" margin the budget was
# chosen for — and it is ~800x the heaviest single pass any current leg actually spends
# (P⁻₃Λ¹(TET), 5 046 units), so today nothing comes close to the ceiling.
#
# It still BOUNDS a pathological program: ~31 s of exact-lane work, after which the statement
# degrades to the warned probe fallback exactly as an over-mass one does.  Re-derive it by
# running the symbolic P(T) legs with the budget unbounded (``POLYARRAY_EXACT_WORK_BUDGET=0``)
# and reading `ExactState.spent` — if the cost model or the lowering changes, that measurement
# is the thing to repeat, not this constant to adjust by feel.
_DEFAULT_WORK_BUDGET = 4_000_000

_WORK_BUDGET_ENV = "POLYARRAY_EXACT_WORK_BUDGET"

# The fixed price of ADMITTING one statement (or one vmap slice) to the exact lane, before any
# of its mass is counted: operand resolution, shape checks, the env writes.  A program of many
# tiny statements must still exhaust a budget, or "deterministic" would only bound the heavy
# ones.  Small relative to a real symbolic op, which measures in the hundreds-to-thousands.
_STMT_COST = 16

# The gcd (`RationalFunction.clean`) is superlinear in the operand mass and is the single most
# expensive step in `_constant_value`, so it is charged a multiple of the mass rather than the
# mass itself.  The factor need not model flint exactly — it only has to keep the budget from
# being dominated by cheap work while a handful of enormous gcds run uncounted.
_GCD_COST_FACTOR = 4

# Early-exit cap for the per-Gauss-column mass measurement.  `_sym_mass` stops accumulating once
# it passes its cap, so measuring a blown-up row costs O(cap) rather than O(row); the cap only
# has to be large enough that an ordinary row is measured exactly.
_GAUSS_ROW_MASS_CAP = 1 << 16

# LOUD WALL-CLOCK BACKSTOP.  The work budget decides the RESULT; the clock only guards against a
# mis-calibrated cost model on a pathological program, and it is deliberately generous.  If it
# ever fires the result IS machine-dependent again, so it is never silent: the statement's reason
# says so, `ExactState.hit_wall_clock` is set, and `simplify` raises `NonDeterministicFoldWarning`.
# A test asserts it does not fire on any calibration leg.
#
# Seconds.  At the API, `wall_backstop=None` means "this default" and `math.inf` means "no
# backstop" — `None` deliberately does NOT mean "disabled", because a caller who passes nothing
# should get the guard, and only someone who writes `inf` should be able to remove it.
_WALL_BACKSTOP_SECONDS = 900.0


def _wall_deadline(wall_backstop: float | None) -> float | None:
    """Return the monotonic instant the wall-clock backstop fires.

    Parameters
    ----------
    wall_backstop
        Seconds to allow. ``None`` takes the module default; ``math.inf`` disables the
        backstop entirely.

    Returns
    -------
    float or None
        The deadline, or ``None`` when there is no backstop.
    """
    seconds = _WALL_BACKSTOP_SECONDS if wall_backstop is None else float(wall_backstop)
    return None if math.isinf(seconds) else time.monotonic() + seconds


class _Exhausted(Exception):
    """Internal: the exact pass ran out of its budget.

    ``wall`` distinguishes the two exits — ``False`` is the ordinary deterministic work budget,
    ``True`` is the backstop firing, which is a MEASUREMENT BUG (the cost model under-counted
    this program) and is reported loudly rather than folded into the normal degrade path.
    """

    def __init__(self, wall: bool = False) -> None:
        super().__init__("wall-clock backstop" if wall else "work budget")
        self.wall = wall


@dataclass
class _Meter:
    """Deterministic work accounting for ONE exact pass.

    The unit is a monomial touched.  `charge` is called where the old code read the clock — at
    each statement, each exact evaluation/gcd, each Gauss column, each vmap slice, each folded
    cell — so the granularity of the bound is unchanged; only its currency is.  As before, one
    already-started flint operation can still overrun: the budget bounds the work *started*,
    which is why `_MAX_SYM_MASS` (per-op, checked BEFORE the op runs) remains the other half of
    the cost story and is not replaced by this.

    `limit <= 0` means unbounded (used by the calibration harness to measure a full pass).
    """

    limit: int
    wall_deadline: float | None = None
    spent: int = 0
    hit_wall_clock: bool = False

    def charge(self, units: int) -> None:
        """Spend ``units`` of work; raise :class:`_Exhausted` when the budget is gone."""
        self.spent += int(units)
        self.check()

    def check(self) -> None:
        """Raise :class:`_Exhausted` if the budget is spent, or the backstop has fired."""
        if 0 < self.limit < self.spent:
            raise _Exhausted(wall=False)
        if self.wall_deadline is not None and time.monotonic() > self.wall_deadline:
            self.hit_wall_clock = True
            raise _Exhausted(wall=True)

    @property
    def remaining(self) -> int:
        """Work units left — also the natural early-exit cap for a `_sym_mass` measurement."""
        return _MAX_SYM_MASS if self.limit <= 0 else max(0, self.limit - self.spent)


def _exhausted_note(exc: _Exhausted, meter: _Meter) -> str:
    """Phrase the ``unresolved`` reason for a budget exit.

    The two exits read differently on purpose. The ordinary one names the budget and what
    was spent, so that raising it is the legible fix. The wall-clock one is emphatic,
    because it means the result has become machine-dependent and the cost model, not the
    budget, is what needs fixing.
    """
    if exc.wall:
        return (f"WALL-CLOCK BACKSTOP fired after {meter.spent} work units — this result is "
                f"NOT reproducible; the work cost model under-counted this program")
    return f"work budget exhausted ({meter.spent}/{meter.limit} units)"


def _resolve_work_budget(work_budget: int | None) -> int:
    """Resolve the work budget: explicit argument, else the environment knob, else the default.

    Exposing the budget as an environment knob is load-bearing for consumers that cache
    verdicts: it makes the budget part of the cache key, so a certificate produced under a
    different budget is never served for this one.
    """
    if work_budget is not None:
        return int(work_budget)
    raw = os.environ.get(_WORK_BUDGET_ENV)
    if raw is None:
        return _DEFAULT_WORK_BUDGET
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_WORK_BUDGET


class _Reason:
    """Deepest-first record of why the current statement went opaque.

    The first op that could not be executed symbolically wins, so the note names the
    innermost cause — an op inside a nested sub-program rather than the outer program
    wrapper. One instance per :func:`exact_partial_eval` call, threaded explicitly rather
    than held in a module global, so it is thread-safe and an escaping exception cannot
    leak a stale note.
    """

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

    * ``known`` — atom / bulk name → exact-constant value (float or float ndarray),
      exactly the shape ``simplify._record_known`` produces.
    * ``sym`` — atom name → exact but NON-constant :class:`RationalFunction` over
      the feed atoms (statement-level exact values, used by the entry-level cell fold).
    * ``sym_bulk`` — bulk name → object ndarray of exact cells (float / RF).
    * ``folded`` — statement indices folded exactly (every output entry constant).
    * ``refuted`` — statement indices whose outputs are PROVABLY non-constant
      (excluded from any probe fallback).
    * ``unresolved`` — statement index → short reason (opaque op name, "work budget", …).
    * ``spent`` / ``limit`` — work units consumed and allowed.  ``spent`` is a DETERMINISTIC
      function of the program: two runs of the same pass spend the same amount,
      which is what makes the certificate reproducible and is worth asserting
      directly in a test.
    * ``hit_wall_clock`` — the loud backstop fired, so this result is NOT reproducible and the
      cost model under-counted this program.  Never expected; ``simplify`` warns.
    """

    known: dict[str, Any] = field(default_factory=dict)
    sym: dict[str, RationalFunction] = field(default_factory=dict)
    sym_bulk: dict[str, np.ndarray] = field(default_factory=dict)
    folded: set[int] = field(default_factory=set)
    refuted: set[int] = field(default_factory=set)
    unresolved: dict[int, str] = field(default_factory=dict)
    spent: int = 0
    limit: int = 0
    hit_wall_clock: bool = False


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _as_numeric(val: ExactValue) -> np.ndarray | None:
    """Return ``val`` as a float ndarray when it carries no symbolic cell, else ``None``."""
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
    """Evaluate ``rf`` exactly at deterministic generic point ``k``.

    Every generator is composed with a distinct dyadic rational constant, whose float to
    ``fmpq`` conversion is exact, so the result is an exact constant
    :class:`RationalFunction` with no float rounding anywhere. No RNG is involved.

    Raises
    ------
    ZeroDivisionError
        If the point is singular, i.e. the denominator vanishes there.
    """
    sub = {g: RationalFunction.constant(0.5 + 0.25 * ((7 * i + 3 * k + 1) % 11))
           for i, g in enumerate(sorted(rf.gens))}
    return rf.compose_multi(sub)


def _rf_mass(rf: RationalFunction) -> int:
    """Monomial mass of one cell: numerator + denominator term counts."""
    return int(rf.num.n_terms()) + int(rf.den.n_terms())


def _sym_mass(values: ExactValue | list[ExactValue], cap: int) -> int:
    """Measure the total monomial mass of one operand or a list of them.

    Accumulation stops as soon as ``cap`` is exceeded, so an enormous operand costs
    ``O(cap)`` to reject rather than ``O(size)`` to measure exactly.
    """
    total = 0
    stack = list(values) if isinstance(values, list) else [values]
    while stack:
        v = stack.pop()
        if isinstance(v, RationalFunction):
            total += _rf_mass(v)
        else:
            arr = np.asarray(v)
            if arr.dtype != object:
                continue                      # numeric operand — no symbolic mass
            for c in (arr.reshape(-1) if arr.shape else [arr[()]]):
                if isinstance(c, RationalFunction):
                    total += _rf_mass(c)
                    if total > cap:
                        return total
        if total > cap:
            return total
    return total


def _charge(meter: _Meter | None, units: int) -> None:
    """Spend ``units`` on ``meter`` when there is one (``None`` = an unbudgeted call).

    The granularity is PER OPERATION: the charge lands before each exact evaluation / gcd /
    Gauss column / vmap slice, so one already-started flint operation can still overrun —
    accepted; the budget bounds the work *started*, exactly as the wall-clock check it
    replaces did.  What changed is the currency, not the granularity.
    """
    if meter is not None:
        meter.charge(units)


def _constant_value(rf: RationalFunction, meter: _Meter | None = None) -> float | None:
    """Return the exact float value of ``rf`` when it is identically constant, else ``None``.

    Constancy is decided by the exact rational normal form: gcd-cancel by exact division,
    then total degree zero of numerator and denominator. A sound exact filter avoids paying
    the gcd on varying cells: ``rf`` is evaluated at two deterministic points in exact
    rational arithmetic, never by float term summation, whose rounding would falsely refute
    a constant such as ``(1/3)·p/p``. Two exactly different values prove non-constancy;
    equal values are only a hint, and the normal form still decides.

    BUDGET-AWARE: every expensive step (each exact evaluation, the gcd) is CHARGED to the
    meter, which raises :class:`_Exhausted` when the budget is gone — a pathological cell (an
    enormous uncancelled quintic — the degree-5 regime) must degrade to *unresolved* (⇒ the
    warned probe fallback), never hang the gate.  The charge is this cell's own monomial mass,
    because that is what both the evaluation and the gcd scale with.
    """
    if rf.is_zero():
        return 0.0                           # exact structural zero (num ≡ 0)
    if not rf.gens:
        return float(rf.eval({}))
    if rf.is_constant():
        return float(rf.eval({g: 0.0 for g in rf.gens}))
    mass = _rf_mass(rf)
    try:
        _charge(meter, mass)
        e0 = _exact_eval_at(rf, 0)
        _charge(meter, mass)
        e1 = _exact_eval_at(rf, 1)
        if e0 != e1:                         # exact inequality ⇒ provably non-constant
            return None
    except _Exhausted:
        raise
    except Exception:  # noqa: BLE001 — singular point etc.: the normal form decides
        pass
    # The gcd is the expensive one and superlinear in the mass, so it is charged the heaviest.
    _charge(meter, _GCD_COST_FACTOR * mass)
    cleaned = rf.clean()
    if cleaned.is_zero():
        return 0.0
    if cleaned.is_constant():
        return float(cleaned.eval({g: 0.0 for g in cleaned.gens}))
    return None


def _cell_constant(cell: ExactCell, meter: _Meter | None = None) -> float | None:
    """Return the exact constant value of one cell, or ``None`` if it is not constant.

    Raises
    ------
    _Exhausted
        Via :func:`_constant_value`, when the work budget is gone.
    """
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return float(cell)
    if isinstance(cell, RationalFunction):
        return _constant_value(cell, meter)
    return None


# ---------------------------------------------------------------------------
# Exact executor
# ---------------------------------------------------------------------------

@dataclass
class _Env:
    """Per-program symbolic environment: exact values for atoms / bulks / stmt outs."""

    atom: dict[str, ExactCell] = field(default_factory=dict)
    bulk: dict[str, np.ndarray] = field(default_factory=dict)
    outs: dict[tuple[int, int], ExactValue] = field(default_factory=dict)
    #: Atoms KNOWN to be unresolvable — a sub-program input bound to an OPAQUE operand
    #: (:func:`_bind_body_inputs`).  Explicit rather than "absent from ``atom``", because an
    #: absent generator is otherwise read as a FEED atom and left live, which would hand back a
    #: value symbolically depending on an operand we never resolved.
    opaque: set[str] = field(default_factory=set)


def _resolve_rf(
    rf: RationalFunction, env: _Env, program: Program, max_sym_mass: int | None = None,
) -> ExactCell | _Opaque:
    """Substitute every resolvable statement-out atom in ``rf`` by its exact value.

    Feed atoms — vertex, point and coefficient provenance — stay symbolic generators. A
    statement-out atom with no exact value makes the whole cell unresolvable.

    ``max_sym_mass``, when given, caps the substitution's monomial mass: one
    ``compose_multi`` runs uninterrupted, so an oversized replacement set is declared
    unresolvable up front rather than exhausting the time budget from inside.
    """
    sub: dict[str, RationalFunction] = {}
    mass = _rf_mass(rf)
    for g in rf.gens:
        if g in env.opaque:
            return _OPAQUE                    # a body input bound to an unresolved operand
        if g in env.atom:
            v = env.atom[g]
            if isinstance(v, RationalFunction):
                if max_sym_mass is not None:
                    mass += _rf_mass(v)
                    if mass > max_sym_mass:
                        return _OPAQUE        # substitution too large for exact work
                sub[g] = v
            else:
                sub[g] = RationalFunction.constant(float(v))
            continue
        try:
            kind = program.env.of(g).kind
        except KeyError:
            return _OPAQUE                    # unknown generator — cannot prove anything
        if kind == "stmt_out":
            return _OPAQUE                    # produced by an unresolved statement
        # feed atom — remains a live generator of the entry's normal form
    return rf.compose_multi(sub) if sub else rf


def _resolve_cells(cells: np.ndarray, env: _Env, program: Program) -> np.ndarray | _Opaque:
    """Resolve an ndarray of cells to exact values, or to :data:`_OPAQUE`."""
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


def _resolve_ref(ref: Ref, env: _Env, program: Program) -> np.ndarray | _Opaque:
    """Resolve one statement input ref to an exact value, or to :data:`_OPAQUE`."""
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

def _fe_is_zero(x: ExactCell) -> bool:
    """Report whether an exact field element is zero."""
    return x.is_zero() if isinstance(x, RationalFunction) else float(x) == 0.0


def _exact_inv(a: np.ndarray, meter: _Meter | None = None) -> np.ndarray:
    """Exact Gauss–Jordan inverse over the rational-function field.

    Pivots are chosen exactly (structurally non-zero — a non-zero rational function
    is a unit of the field); the resulting entries are the inverse AS RATIONAL
    FUNCTIONS, i.e. the identity ``A·A⁻¹ = I`` holds identically wherever
    ``det A ≠ 0`` — the correct generic-inverse semantics for identity
    certification.  Raises on an exactly singular matrix.
    """
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
        # Per-column, charged by the PIVOT ROW's actual mass: Gauss over RF cells is slow in
        # proportion to the fill-in it has already produced, which a fixed per-column price
        # would miss entirely (the operand cap bounds what ENTERS the op, not what it grows to).
        _charge(meter, _sym_mass(list(m[col]), _GAUSS_ROW_MASS_CAP))
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


def _exact_det(a: np.ndarray, meter: _Meter | None = None) -> ExactCell:
    """Exact determinant by fraction-free-ish Gauss elimination over the field."""
    a = np.asarray(a, dtype=object).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"exact det needs a square matrix, got {a.shape}")
    n = a.shape[0]
    det: Any = 1.0
    for col in range(n):
        _charge(meter, _sym_mass(list(a[col]), _GAUSS_ROW_MASS_CAP))   # see `_exact_inv`
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


def _exact_solve(a: np.ndarray, b: np.ndarray, meter: _Meter | None = None) -> np.ndarray:
    b = np.asarray(b, dtype=object)
    inv = _exact_inv(a, meter)
    if b.ndim == 1:
        return np.einsum("ij,j->i", inv, b, optimize=False)
    return np.einsum("ij,j...->i...", inv, b, optimize=False)


def _exact_pinv(a: np.ndarray, meter: _Meter | None = None) -> np.ndarray:
    """Compute the generic, full-rank pseudo-inverse over the rational-function field.

    ``A⁺ = (AᵀA)⁻¹Aᵀ`` for a tall or square ``A``, ``Aᵀ(AAᵀ)⁻¹`` for a wide one: the
    normal-equation form, which equals the Moore-Penrose pseudo-inverse wherever ``A`` has
    full rank. These are the same generic-inverse semantics :func:`_exact_inv` and
    :func:`_exact_solve` carry — the identity holds identically wherever the Gram
    determinant is non-zero — and :func:`_exact_inv` raises on a structurally singular
    Gram, so a rank-deficient ``A`` goes opaque rather than being guessed.

    This does **not** agree with ``np.linalg.pinv`` on a rank-deficient matrix. The
    Moore-Penrose inverse is discontinuous at a rank drop and is not a rational function of
    the entries, so no exact twin of it can exist; this lane certifies the generic branch
    only, and the numeric lane is unaffected.
    """
    a = np.asarray(a, dtype=object)
    if a.ndim != 2:
        raise ValueError(f"exact pinv needs a matrix, got shape {a.shape}")
    n, k = a.shape
    if n >= k:                                        # tall/square: (AᵀA)⁻¹Aᵀ
        gram = np.einsum("ik,il->kl", a, a, optimize=False)
        return _exact_solve(gram, a.T, meter)
    gram = np.einsum("ik,jk->ij", a, a, optimize=False)   # wide: Aᵀ(AAᵀ)⁻¹
    return np.einsum("ik,ij->kj", a, _exact_inv(gram, meter), optimize=False)


def _exact_assert(fn: AssertOp, args: list[Any], meter: _Meter | None) -> list[Any]:
    """Reproduce :class:`~polyarray.AssertOp` over symbolic operands.

    ``AssertOp`` returns its first input unchanged; the predicate is a guard. This twin
    reproduces the value and re-checks every predicate that is decidable over the
    rational-function field, raising — hence going opaque, never silently passing — when a
    decidable check fails:

    * ``shape_eq``          — structural, always decidable;
    * ``rank_eq``           — decidable when both operands resolve to exact constants; else
                              opaque (a rank is an integer, so a non-constant one is a bug,
                              not a case to guess through);
    * ``square_full_rank``  — squareness structurally, full rank as the exact ``det ≠ 0``
                              (GENERIC rank — the ``_exact_inv`` semantics);
    * ``spd``               — exact SYMMETRY (decidable: ``A = Aᵀ`` as rational functions) plus
                              generic non-singularity.  **Positive-definiteness itself is a
                              real INEQUALITY, not decidable over the rational-function field**,
                              so the twin does not certify it.  This is the one place the exact
                              lane is weaker than the numeric op it stands in for; it is a
                              build-time certificate over a symbolic cell, and the runtime
                              ``AssertOp`` still runs the true test on real data whenever the
                              statement survives (a symbolic, non-constant operand is *refuted*,
                              never folded away).  Flagged for review rather than hidden.
    """
    x = np.asarray(args[0])
    if fn.kind == "shape_eq":
        other = np.asarray(args[1])
        if x.shape != other.shape:
            raise AssertionError(f"{fn.msg} [shape_eq] {x.shape} != {other.shape}")
    elif fn.kind == "rank_eq":
        a, b = _cell_constant(np.asarray(args[1])[()], meter), \
            _cell_constant(np.asarray(args[2])[()], meter)
        if a is None or b is None:
            raise ValueError("exact AssertOp(rank_eq): non-constant rank operand")
        if int(a) != int(b):
            raise AssertionError(f"{fn.msg} [rank_eq] {int(a)} != {int(b)}")
    elif fn.kind in ("spd", "square_full_rank"):
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise AssertionError(f"{fn.msg} [{fn.kind}] shape={x.shape}")
        if fn.kind == "spd":
            xo = _obj(x)
            for idx in np.ndindex(*x.shape):
                # Exact equality (no tolerance) — a float cell carrying fp noise therefore
                # raises ⇒ OPAQUE, the safe direction (less folding, never a false pass).
                if not _fe_is_zero(xo[idx] - xo[idx[::-1]]):
                    raise AssertionError(f"{fn.msg} [spd] matrix not symmetric at {idx}")
        det = _exact_det(x, meter)
        if _fe_is_zero(det):
            raise AssertionError(f"{fn.msg} [{fn.kind}] structurally singular")
    elif fn.kind == "in_span":
        # `‖v − P·pinv(P)·v‖ ≤ tol·scale` — a NORM INEQUALITY, so like `spd`'s positive-definiteness
        # it is NOT decidable over the rational-function field. Pass the value through WITHOUT
        # certifying, rather than raising: the raise made this statement OPAQUE, and everything
        # downstream of it with it — a silent capability loss, not a check.
        #
        # Safe by the same argument `spd` records above: this twin only ever sees SYMBOLIC operands
        # (an all-numeric statement takes the `_exec_fn` path, which runs the REAL `AssertOp` and
        # performs the true test), and a symbolic non-constant operand is REFUTED, never folded away —
        # so the run-time assert survives and still checks it on real data.
        pass
    else:
        raise ValueError(f"AssertOp: unknown kind {fn.kind!r}")
    return [x]


def _obj(a: ExactValue) -> np.ndarray:
    """Return an operand as an object ndarray, so einsum can mix float and rational cells."""
    arr = np.asarray(a)
    return arr if arr.dtype == object else arr.astype(object)


# --- symbolic op twins ------------------------------------------------------

def _opaque_name(fn: StmtOp) -> str:
    """Name an op the way this lane reports it when refusing to execute it symbolically."""
    name = type(fn).__name__
    if isinstance(fn, CallOp):
        inner = fn.fn
        return f"CallOp({'vmap closure' if hasattr(inner, '_vmap_body') else type(inner).__name__})"
    if hasattr(fn, "_vmap_body"):
        return f"vmap closure ({name})"
    return name


def _const_int(val: ExactValue) -> int | None:
    """Return ``val`` as a build-time constant int, else ``None``.

    Several ops carry an integer operand — a rank, a block count — that sizes the result
    rather than contributing to it. Such an operand is not something to approximate: if it
    has not resolved to a constant the statement is genuinely unresolved, so the twins bail
    rather than pick a value.
    """
    num = _as_numeric(val)
    if num is None or num.size != 1:
        return None
    return int(num.reshape(-1)[0])


def _exact_block_repeat(a: np.ndarray, count: int, reason: _Reason) -> list[ExactValue] | None:
    """Lay out ``count`` block-diagonal copies of ``a`` over exact cells.

    The shared body of the static :class:`BlockRepeatOp` twin and the runtime-counted
    :class:`DynBlockRepeatOp` one.
    """
    if a.ndim != 2:
        reason.note("BlockRepeat: non-matrix operand")
        return None
    if count < 0:
        reason.note(f"BlockRepeat: negative block count {count}")
        return None
    rows, cols = a.shape
    out = np.full((count * rows, count * cols), 0.0, dtype=object)
    for i in range(count):
        out[i * rows:(i + 1) * rows, i * cols:(i + 1) * cols] = a
    return [out]


def _opaque(fn: StmtOp, reason: _Reason) -> None:
    """Record ``fn`` as opaque on the symbolic path, and return ``None`` — never a guess."""
    reason.note(f"opaque op {_opaque_name(fn)} on the symbolic path")
    return None


def _is_exact_zero(val: ExactValue) -> bool:
    """Report whether ``val`` is the exact zero array, every entry a syntactic zero of its own kind.

    Exact, never a tolerance: a float cell must be ``0.0`` and a
    :class:`~polyarray.rational.RationalFunction` cell must have a zero numerator
    (:func:`_fe_is_zero`).

    Parameters
    ----------
    val
        A resolved operand value.  ``Any`` because an operand is whatever the lane resolved it to —
        an ndarray of floats, an ndarray of cells, or the ``_OPAQUE`` sentinel.

    Returns
    -------
    bool
        ``True`` only when every entry is an exact zero.  An empty operand is not a zero (there is
        nothing to be zero), and ``_OPAQUE`` is never a zero.
    """
    if val is _OPAQUE:
        return False
    arr = np.asarray(val)
    if arr.size == 0:
        return False
    if arr.dtype.kind in ("f", "i", "u"):
        return bool((arr == 0).all())
    if arr.dtype != object:
        return False
    return all(_fe_is_zero(c) if isinstance(c, (RationalFunction, int, float, np.integer,
                                                np.floating)) else False
               for c in arr.reshape(-1))


def _zero_absorbing(fn: StmtFn) -> bool:
    """Report whether ``fn`` is multilinear in each operand.

    Multilinearity is what makes one exactly-zero operand force
    an exactly-zero result, whatever the other operands hold (even unresolved ones)?

    This is the only shape of "reason about an op without all of its inputs" this lane admits, and
    it is an identity rather than an inference: each output entry of a multilinear op is a sum of
    products carrying exactly one factor from each operand, so a zero factor annihilates every
    term.  It is emphatically NOT a blanket "zeros pass through opaque ops" — ``QrOp(0)`` is a
    degenerate factorization, not zero, and every algebraic/order op below says so.

    Parameters
    ----------
    fn
        A builtin op.  The caller establishes ``is_builtin_op(fn)`` first.

    Returns
    -------
    bool
        ``True`` for the multilinear ops only.

    Notes
    -----
    EXHAUSTIVE over :data:`~polyarray.ir.StmtFn` (``assert_never``): a new op must state which side
    it is on, because answering ``True`` wrongly would fabricate a zero.
    """
    match fn:
      # --- multilinear: one zero operand ⇒ zero result ----------------------------
      case EinsumStmtOp() | EinsumOp() | TensordotOp() | KronOp() | KronFreeOp() \
              | ScaleOp() | ScaleByOp() | ProjectOp() | EmbedOp():
        # Contractions and (Kronecker/elementwise/scalar) products: every output entry is a sum
        # of products with one factor per operand.  `EinsumOp`'s second factor is BAKED on the op
        # and `ScaleOp`'s is a literal, which does not change the argument.
        return True

      # --- NOT multilinear: a zero operand says nothing about the result ----------
      case AddOp() | ConcatOp() | HStackOp() | ColStackOp() | BlockDiagOp() \
              | BlockRepeatOp() | DynBlockRepeatOp():
        # Sums and assemblies: a zero SUMMAND / block leaves the other entries untouched.
        return False
      case TransposeOp() | MoveaxisOp() | ReshapeOp() | IdentityOp() | FirstColsOp() \
              | LastColsOp():
        # Pure rearrangements. A zero operand does give a zero result, but they are UNARY (a
        # slice's second operand is a rank, not a factor), so there is never another operand to
        # be unresolved — admitting them would buy nothing and widen the rule.
        return False
      case InvOp() | InvTransposeOp() | SolveOp() | PinvOp() | DetOp() | ComposeViaStdOp():
        # Division-like: an inverse of zero does not exist, `solve`'s zero RHS gives zero only
        # when the matrix is non-singular (which is what the solve itself has to establish), and
        # the exact twins already refuse a singular operand.
        return False
      case QrOp() | SvdOp() | GSvdOp() | GSvdFullOp() | SqrtOp() | SqrtSpdOp() \
              | MetricOrthonormalOp() | SinvFullOp() | AbsOp() | SignOp() | RankOp() \
              | CompRankOp():
        # Factorizations, roots and order predicates: ⚠ a factorization OF a zero matrix is
        # DEGENERATE, not zero (`Q` is an arbitrary orthonormal frame), and these do not take a
        # second operand to be zero-annihilated by in any case.
        return False
      case AxisLenOp() | ConstOp() | EyeOp() | DynEyeOp() | DynZerosOp() | DynEyeTensorOp() \
              | ProdShapeOp() | SumShapeOp() | SumDimOp() | ProdDimOp() | ScaleAxisDimOp() \
              | MulAxisDimOp():
        # STRUCTURAL / nullary: the result is a function of shapes, not of cell values.
        return False
      case AssertOp() | SwitchOp() | WhileOp() | CallOp():
        # Guards and control flow: not arithmetic at all (`SwitchOp` has its own twin, which
        # already reasons from the branches without the scrutinee).
        return False

      case _ as unreachable:
        assert_never(unreachable)


def _tolerates_opaque_operand(fn: StmtOp) -> bool:
    """Report whether this statement may still be executed with an unresolved operand.

    Two arguments, and the open half of ``Stmt.fn`` is handled first (`polyarray/CLAUDE.md`):

    * LIVENESS — a sub-:class:`~polyarray.ir.Program`, or a REBUILDABLE ``vmap`` closure over
      one (:func:`_vmap_closure`): the operand may simply be DEAD in the body, and that is
      decided by EXECUTING it (:func:`_bind_body_inputs` marks the unresolved input's atoms
      unresolvable, so anything reading them goes opaque and only a body whose declared outputs
      resolve anyway returns a value).  This is the same liveness argument :func:`_run_program`
      already makes for an opaque STATEMENT inside a body, applied to its inputs.  The vmap case
      says nothing extra about the body — it is still per-slice :func:`_run_program`, with the
      unresolved operand handed to EVERY slice unresolved (:func:`_run_vmap`);
    * MULTILINEARITY — a builtin :func:`_zero_absorbing` op: an exactly-zero factor annihilates
      the result, so the unresolved factor cannot matter.  The caller checks the zero actually
      exists before executing.

    Parameters
    ----------
    fn
        A ``Stmt.fn``.  ``Any`` because that field is half-open: a sub-``Program``, a vmap closure,
        a front-end op or a plain callable are all valid, and only the builtin half has a type.

    Returns
    -------
    bool
        ``True`` when one of the two arguments applies.

    Notes
    -----
    ⚠ The two are NOT interchangeable, and the caller must not treat them as one permission: the
    liveness cases never permit inventing a value for the unresolved operand (see the guard on the
    zero short-circuit in :func:`_exec_stmt`).
    """
    if isinstance(fn, Program):
        return True
    if isinstance(fn, CallOp) and isinstance(fn.fn, Program):
        return True
    if _vmap_closure(fn) is not None:
        return True
    return is_builtin_op(fn) and _zero_absorbing(fn)


def _ref_shape(ref: Ref, program: Program) -> tuple[int, ...] | None:
    """Return the static shape one statement input ref will carry, without resolving its value.

    Only needed for an operand the zero-absorbing rule annihilates: the twin still has to be
    handed an array of the right shape to produce a correctly-shaped zero.

    Parameters
    ----------
    ref
        A Stmt input ref.  ``Any`` to match :func:`_resolve_ref`, which this mirrors: the ref
        vocabulary is open at the edges (an unrecognised ref simply has no static shape).
    program
        The program the ref reads from.

    Returns
    -------
    tuple[int, ...] or None
        The shape, or ``None`` when it cannot be read off — a dynamic axis, an indexed ref, a
        runtime-bound integer.
    """
    def _static(shape: Sequence[int | DimAtom]) -> tuple[int, ...] | None:
        shp = tuple(shape)
        return None if is_dynamic(shp) else tuple(int(d) for d in shp)

    if isinstance(ref, Const):
        return tuple(np.asarray(ref.value).shape)
    if isinstance(ref, RationalRef):
        return ()
    if isinstance(ref, (InputRef, OutputRef)) and ref.indices:
        return None                           # an indexed slice: shape depends on the indices
    if isinstance(ref, InputRef):
        sa = program.input_arrays.get(ref.name)
        return None if sa is None else _static(sa.shape)
    if isinstance(ref, OutputRef):
        if not (0 <= ref.stmt_idx < len(program.statements)):
            return None
        outs = program.statements[ref.stmt_idx].out
        if not (0 <= ref.out_idx < len(outs)):
            return None
        return _static(outs[ref.out_idx].shape)
    if isinstance(ref, SymArrayRef):
        if ref._bulk is not None:
            return _static(ref._bulk.shape)
        return tuple(np.asarray(ref._cells).shape)
    return None                               # IntAtomRef / unknown


def _sym_apply(
    fn: StmtOp, args: list[ExactValue], meter: _Meter, depth: int, reason: _Reason,
    max_sym_mass: int,
) -> list[ExactValue] | None:
    """Execute one op over exact symbolic operands, returning ``None`` when it is opaque.

    Two layers, because ``Stmt.fn`` is only half closed. The open half — a nested
    :class:`~polyarray.ir.Program`, a ``vmap`` closure, a front-end op class, a plain
    callable — is dispatched here; polyarray's own :data:`~polyarray.ir.StmtFn` vocabulary
    is dispatched exhaustively by :func:`_sym_apply_builtin`.
    """
    if isinstance(fn, Program):
        return _run_program(fn, args, meter, depth + 1, reason, max_sym_mass)
    if isinstance(fn, CallOp) and isinstance(fn.fn, Program):
        return _run_program(fn.fn, args, meter, depth + 1, reason, max_sym_mass)
    closure = _vmap_closure(fn)
    if closure is not None:
        return _run_vmap(closure, args, meter, depth, reason, max_sym_mass)
    if is_builtin_op(fn):
        return _sym_apply_builtin(fn, args, meter, reason)
    return _opaque(fn, reason)                 # front-end op / plain callable / hookless vmap


def _sym_apply_builtin(
    fn: StmtFn, args: list[ExactValue], meter: _Meter, reason: _Reason,
) -> list[ExactValue] | None:
    """Apply the exact twin of one polyarray-owned op, or ``None`` when it is opaque.

    Exhaustive over :data:`~polyarray.ir.StmtFn`: every op either has a rational twin or an
    explicit arm stating why it has none. The closing ``assert_never`` makes a newly added
    op a mypy error here, which is load-bearing — an op silently missing from the ladder is
    opaque, and opacity stalls a certificate rather than failing it.
    """
    match fn:
      case EinsumStmtOp():
        return [np.einsum(fn.spec, *[_obj(a) for a in args], optimize=False)]
      case EinsumOp():
        rhs = np.frombuffer(fn.rhs_bytes, dtype=fn.rhs_dtype).reshape(fn.rhs_shape)
        return [np.einsum(fn.spec, _obj(args[0]), _obj(rhs), optimize=False)]
      case TensordotOp():
        return [np.tensordot(_obj(args[0]), _obj(args[1]), axes=fn.axes)]
      case TransposeOp():
        return [np.asarray(args[0]).T]
      case MoveaxisOp():
        return [np.moveaxis(np.asarray(args[0]), fn.source, fn.destination)]
      case ReshapeOp():
        return [np.asarray(args[0]).reshape(fn.shape)]
      case IdentityOp():
        return [np.asarray(args[0])]
      case AddOp():
        out = _obj(args[0]).copy()
        for x in args[1:]:
            out = out + _obj(x)
        return [out]
      case ScaleOp():
        return [fn.factor * _obj(args[0])]
      case ScaleByOp():
        return [_obj(args[1]) * _obj(args[0])]
      case ConcatOp():
        return [np.concatenate([_obj(x).reshape(-1) for x in args])]
      case HStackOp():
        return [np.hstack([_obj(m) for m in args])]
      case ColStackOp():
        return [np.stack([_obj(c).reshape(-1) for c in args], axis=1)]
      case InvOp():
        return [_exact_inv(args[0], meter)]
      case InvTransposeOp():
        return [_exact_inv(args[0], meter).T]
      case SolveOp():
        return [_exact_solve(args[0], args[1], meter)]
      case DetOp():
        return [np.asarray(_exact_det(args[0], meter), dtype=object)]
      case PinvOp():
        return [_exact_pinv(args[0], meter)]
      case ProjectOp():
        # ``Pᵀ @ v.reshape(-1)`` — the grassmann-origin drop-complement matvec.  Pure field
        # arithmetic (no float coercion, unlike ``ProjectOp.__call__``'s numeric contract), so
        # it threads RF cells exactly.
        return [np.einsum("nr,n->r", _obj(args[0]), _obj(args[1]).reshape(-1), optimize=False)]
      case EmbedOp():
        # ``(P @ vsub).reshape(op.shape)`` — the ambient reconstruction.
        out = np.einsum("ij,j...->i...", _obj(args[0]), _obj(args[1]), optimize=False)
        return [out.reshape(fn.shape) if fn.shape else out]
      case SwitchOp():
        # `select_x` over an `IntAtom`: inputs are `(scrutinee, branch_0, …, branch_{n-1})`.
        #
        # MINIMAL fold, deliberately — two sound cases and no guessing:
        #   (a) EQUAL BRANCHES ⇒ the scrutinee cannot matter, so the switch is that value. This is the
        #       one that pays: it is the same question `savo _blocks_equal` asks at the folded block
        #       ("does reorienting this entity change this value?"), asked per-operand.
        #   (b) a scrutinee that IS a build-time constant selects its branch.
        # Anything else is opaque, as before. DISTRIBUTION — pushing an op through a switch,
        # `f(select_x(a,[x,y]), c)` → `select_x(a,[f(x,c), f(y,c)])` — is a program REWRITE, not
        # something a per-op twin can do, and is deliberately left out of this pass.
        #
        # Why it matters: design item B moves the σ switch from the folded OUTPUT to the geometry
        # INPUTS, i.e. UPSTREAM of the whole body. Without (a), every σ-carrying expression would meet
        # an unfoldable node first and degrade to probes — trading the `QrOp` just removed from the
        # dependency path for a `SwitchOp`.
        branches = args[1:]
        if len(branches) != fn.n_branches or not branches:
            reason.note(
                f"SwitchOp: expected {fn.n_branches} branches, got {len(branches)}")
            return None
        if any(b is _OPAQUE for b in branches):
            reason.note("SwitchOp: a branch is unresolved")
            return None
        vals = [_obj(b) for b in branches]
        first = vals[0]
        if all(v.shape == first.shape and np.array_equal(v, first) for v in vals[1:]):
            return [first]                     # (a) the scrutinee is irrelevant
        scr = args[0]
        if scr is not _OPAQUE:                 # (b) a constant scrutinee picks its branch
            s = np.asarray(_as_numeric(scr) if _as_numeric(scr) is not None else scr)
            if s.size == 1:
                try:
                    k = int(s.reshape(-1)[0])
                except (TypeError, ValueError):
                    k = None
                if k is not None and 0 <= k < len(vals):
                    return [vals[k]]
        reason.note("SwitchOp: branches differ and the scrutinee is not a build-time constant")
        return None
      case KronOp():
        # Chained Kronecker product — pure field MULTIPLICATION of entries, so it threads
        # RationalFunction cells exactly (unlike `KronOp.__call__`, which coerces to float).
        # STRICTLY 2-D only: the twin below is the matrix Kronecker product, and a non-matrix
        # operand would silently produce a wrong SHAPE (which does not fail here — it fails far
        # downstream in an unrelated einsum's arity check).  Anything else falls through to the
        # opaque tail, i.e. exactly the pre-existing behaviour.
        mats = [_obj(a) for a in args]
        if any(m.ndim != 2 for m in mats):
            reason.note(f"KronOp over non-matrix operands (ndim={[m.ndim for m in mats]})")
            return None
        out = mats[0]
        for m in mats[1:]:
            out = (out[:, None, :, None] * m[None, :, None, :]).reshape(
                out.shape[0] * m.shape[0], out.shape[1] * m.shape[1])
        return [out]
      case KronFreeOp():
        # The Kron block with trailing free axes — the same elementwise product as
        # `KronFreeOp.__call__`, done in object dtype so exact cells survive.  Leaving this
        # un-normalizable made the exact lane refuse every `Λᵏ` certificate: the wedge's
        # two traced slots meet in exactly this op, so `KronFreeOp` sat directly on the result
        # path with NOTHING irrational about it (a Kronecker product is field arithmetic).
        F, G = _obj(args[0]), _obj(args[1])
        # Mirror `KronFreeOp.__call__`'s shape contract exactly: each operand is
        # ``(dom, cod, *free)`` with the declared free-axis counts.  A mismatch would build a
        # wrong-shaped result that only fails much later, so verify before committing.
        if (F.ndim != 2 + fn.nf_free) or (G.ndim != 2 + fn.ng_free):
            reason.note(f"KronFreeOp operand rank mismatch (F.ndim={F.ndim}, G.ndim={G.ndim}, "
                        f"nf_free={fn.nf_free}, ng_free={fn.ng_free})")
            return None
        df, cf, ff = F.shape[0], F.shape[1], F.shape[2:]
        dg, cg, gg = G.shape[0], G.shape[1], G.shape[2:]
        fr = F.reshape(df, 1, cf, 1, *ff, *([1] * fn.ng_free))
        gr = G.reshape(1, dg, 1, cg, *([1] * fn.nf_free), *gg)
        return [(fr * gr).reshape(df * dg, cf * cg, *ff, *gg)]
      case AxisLenOp():
        # A STRUCTURAL read: the operand's static axis length, independent of its cells — so it
        # is exact (and constant) even when the operand is fully symbolic.
        return [np.asarray(float(np.asarray(args[0]).shape[fn.axis]))]
      case AssertOp():
        return _exact_assert(fn, args, meter)

      # --- deliberately opaque: NOT a rational function of the operands ------------
      #
      # The exact lane's field is ℚ(atoms) — rational functions with exact `fmpq`
      # coefficients.  Each op below leaves that field, so there is no twin to write; a
      # symbolic operand here is genuinely unresolved and the statement degrades to the
      # (warned) probe fallback.  This is a DECISION, not an omission.
      case QrOp() | SvdOp() | GSvdOp() | GSvdFullOp() | SqrtOp() | SqrtSpdOp() \
              | MetricOrthonormalOp() | SinvFullOp():
        # Orthogonal / singular-value factorizations, square roots and Cholesky: the
        # entries are ALGEBRAIC (roots of the operand's entries), not rational, and the
        # factors are not even unique without a sign/ordering convention.  `SinvFullOp`
        # additionally divides by singular values under a runtime rank cut.
        return _opaque(fn, reason)
      case AbsOp() | SignOp() | RankOp() | CompRankOp():
        # ORDER predicates on the reals (|·|, sign, a tolerance-gated numeric rank).  A
        # rational function has no decidable sign over ℚ(atoms) — deciding one would mean
        # choosing a branch, i.e. guessing.  (`CompRankOp` = ambient − rank inherits it.)
        return _opaque(fn, reason)
      case WhileOp():
        # Control flow: never executed at build time.  `_exec_stmt` bails on a `WhileOp`
        # before it reaches here; the arm exists so the decision is stated, not implied.
        return _opaque(fn, reason)
      case CallOp():
        # A `CallOp` over a `Program` or a REBUILDABLE vmap closure is descended by
        # `_sym_apply` before this function is called.  What is left wraps something this
        # lane cannot enter — a front-end callable, or a vmap closure that does not expose
        # its axis tuples (slicing it would be a guess).
        return _opaque(fn, reason)

      # --- deliberately opaque: unreachable on the symbolic path -------------------
      #
      # `_exec_stmt` only calls this function when at least one operand failed to resolve
      # numerically.  These ops take NO operands at all, so they are always numeric-closed
      # and run their real `__call__` on the numeric lane (the `fold_numeric` contract).
      case ConstOp() | EyeOp():
        return _opaque(fn, reason)

      # --- exact twins for the ops that ARE field arithmetic / slices / shape reads -----
      #
      # Each op below is expressible over ℚ(atoms) — pure field arithmetic, a pure slice, or
      # a read of static SHAPE data — so each twin EXTENDS the certificate: statements that
      # previously degraded to the (warned) probe lane now certify exactly.  That changes what
      # folds and hence downstream numerics at roundoff, which is why each landed with its own
      # value A/B rather than as one sweep.
      case ComposeViaStdOp():
        # `solve(R_to, R_from)` — the SAME exact Gauss elimination the `SolveOp` arm uses, on
        # the same `_exact_solve`.  This was the sharpest of the listed gaps precisely because
        # the twin it needed was already sitting in this module.
        return [_exact_solve(_obj(args[0]), _obj(args[1]), meter)]
      case BlockDiagOp():
        # Block-diagonal assembly: pure STRUCTURE — every entry is either an operand cell or
        # an exact zero.  Mirrors `BlockDiagOp.__call__` shape-for-shape, in object dtype so
        # the exact cells survive instead of being coerced to float.
        mats = [_obj(a) for a in args]
        if any(m.ndim != 2 for m in mats):
            reason.note("BlockDiagOp: non-matrix operand")
            return None
        out = np.full((sum(m.shape[0] for m in mats), sum(m.shape[1] for m in mats)),
                      0.0, dtype=object)
        r = c = 0
        for m in mats:
            out[r:r + m.shape[0], c:c + m.shape[1]] = m
            r += m.shape[0]
            c += m.shape[1]
        return [out]
      case BlockRepeatOp():
        # `kron(eye(n), A)` with a STATIC count — the same argument that earned `KronOp` its
        # twin: a Kronecker product against an identity is structure plus field arithmetic.
        return _exact_block_repeat(_obj(args[0]), fn.n, reason)
      case DynBlockRepeatOp():
        # As `BlockRepeatOp`, but the count is an operand: it must resolve to a build-time
        # constant.  A block count is an integer — guessing one is not a rounding decision.
        count = _const_int(args[1])
        if count is None:
            reason.note("DynBlockRepeatOp: block count is not a build-time constant")
            return None
        return _exact_block_repeat(_obj(args[0]), count, reason)
      case FirstColsOp():
        # `A[:, :int(rank)]` — a pure SLICE, exact over RF cells once `rank` is constant.
        rank = _const_int(args[1])
        if rank is None:
            reason.note("FirstColsOp: rank operand is not a build-time constant")
            return None
        return [np.asarray(args[0])[:, :rank]]
      case LastColsOp():
        rank = _const_int(args[1])
        if rank is None:
            reason.note("LastColsOp: rank operand is not a build-time constant")
            return None
        return [np.asarray(args[0])[:, rank:]]
      case DynEyeOp() | DynZerosOp() | DynEyeTensorOp() | ProdShapeOp() | SumShapeOp() \
              | SumDimOp() | ProdDimOp() | ScaleAxisDimOp():
        # STRUCTURAL reads: the result is a function of the operands' static SHAPES, not of
        # their cells, so it is an exact constant even under a FULLY SYMBOLIC operand — the
        # same argument that makes the `AxisLenOp` arm above sound.  Each of these ops touches
        # `.shape` and nothing else, so running the real op is exact by construction; it
        # cannot read a cell, symbolic or otherwise.
        return [np.asarray(fn(*[np.asarray(a) for a in args]))]
      case MulAxisDimOp():
        # The same structural argument, except the runtime COUNT operand is a value, not a
        # shape — so it must resolve to a build-time constant before the read is exact.
        count = _const_int(args[0])
        if count is None:
            reason.note("MulAxisDimOp: runtime count operand is not a build-time constant")
            return None
        return [np.asarray(count * int(np.asarray(args[1]).shape[fn.axis]))]

      case _ as unreachable:
        assert_never(unreachable)


def _vmap_closure(fn: StmtOp) -> tuple[Program, tuple[int | None, ...], tuple[int | None, ...]] | None:
    """Return ``(body, in_axes, out_axes)`` when ``fn`` is a rebuildable vmap closure.

    All three attributes must be present: a front-end wrapper exposing only the body,
    without the axis tuples, is not descendable, because the slicing would be a guess.
    ``CallOp``-wrapped closures are unwrapped first.
    """
    inner = fn.fn if isinstance(fn, CallOp) else fn
    body = getattr(inner, "_vmap_body", None)
    in_axes = getattr(inner, "_in_axes", None)
    out_axes = getattr(inner, "_out_axes", None)
    if not isinstance(body, Program) or in_axes is None or out_axes is None:
        return None
    return body, tuple(in_axes), tuple(out_axes)


def _run_vmap(
    closure: tuple[Program, tuple, tuple], args: list[Any], meter: _Meter, depth: int,
    reason: _Reason, max_sym_mass: int,
) -> list[Any] | None:
    """Descend exactly into a vmap closure, the batched twin of :func:`_run_program`.

    Reproduces ``ir.vmap``'s own semantics exactly (slice each batched operand along its
    ``in_axes`` entry, run the body per slice, ``np.stack`` on ``out_axes``), but over exact
    values, so the certificate no longer stops at the closure boundary.

    COST IS BOUNDED BEFORE THE LOOP STARTS (the stall rule — an uninterruptible flint op
    ignores any budget): the batched operands' total monomial mass must fit ``max_sym_mass``
    (it is the SUM over the slices, so the whole descent stays inside the one box the caller
    budgeted for this statement), AND the batch size must fit :data:`_MAX_VMAP_BATCH` (each
    slice pays the body's fixed per-statement overhead).  The budget is additionally charged
    INSIDE the loop, once per slice.  Every rejection degrades to *unresolved* ⇒ the warned
    probe fallback — never a hang, never a guess.

    An UNRESOLVED operand (``_OPAQUE``) is admitted, on LIVENESS only
    (:func:`_tolerates_opaque_operand`): it cannot be sliced — neither its value nor its batch
    length is known — so it is handed to EVERY slice unresolved, exactly as it arrived, and
    :func:`_bind_body_inputs` marks that body input's atoms unresolvable.  Only a body whose
    declared outputs resolve without them returns a value, which is the proof that the operand was
    dead.  Nothing about the body is assumed — in particular NOT multilinearity, so a zero in
    another operand buys nothing here; a body that reads the unresolved input simply stays
    unresolved.  It also contributes no batch size: at least one RESOLVED batched operand must
    fix ``batch`` (else the arity check below refuses).
    """
    body, in_axes, out_axes = closure
    if len(args) != len(in_axes) or len(in_axes) != len(body.inputs):
        reason.note(f"vmap closure ({body.name}): operand/in_axes/body-input arity mismatch")
        return None
    arrs = [a if a is _OPAQUE else np.asarray(a) for a in args]
    norm: list[int | None] = []
    sizes: list[int] = []
    for arr, ax in zip(arrs, in_axes):
        if ax is None or arr is _OPAQUE:
            norm.append(None)                 # pass through whole (an `_OPAQUE` rides unchanged)
            continue
        a = int(ax) if int(ax) >= 0 else arr.ndim + int(ax)
        if not (0 <= a < arr.ndim):
            reason.note(f"vmap closure ({body.name}): in_axis {ax} out of range for {arr.shape}")
            return None
        norm.append(a)
        sizes.append(int(arr.shape[a]))
    if len(set(sizes)) != 1:
        reason.note(f"vmap closure ({body.name}): inconsistent batch sizes {sizes}")
        return None
    batch = sizes[0]
    if batch > _MAX_VMAP_BATCH:
        reason.note(f"vmap closure ({body.name}): batch {batch} > {_MAX_VMAP_BATCH}")
        return None
    mass = _sym_mass(args, max_sym_mass)
    if mass > max_sym_mass:
        reason.note(f"vmap closure ({body.name}): batched operands too large for exact "
                    f"execution (monomial mass > {max_sym_mass})")
        return None
    per_slice: list[list[Any]] = []
    for i in range(batch):
        _charge(meter, _STMT_COST)            # each slice pays the body's fixed admission price
        sliced: list[Any] = []
        for arr, ax in zip(arrs, norm):
            if ax is None:
                sliced.append(arr)
                continue
            idx: list[Any] = [slice(None)] * arr.ndim
            idx[ax] = i
            sliced.append(arr[tuple(idx)])
        outs = _run_program(body, sliced, meter, depth + 1, reason, max_sym_mass)
        if outs is None:
            return None                       # one opaque slice poisons the batch — no guess
        per_slice.append(outs)
    n_out = len(per_slice[0])
    if len(out_axes) != n_out or any(len(o) != n_out for o in per_slice):
        reason.note(f"vmap closure ({body.name}): out_axes/body-output arity mismatch")
        return None
    stacked: list[Any] = []
    for k in range(n_out):
        col = [np.asarray(s[k]) for s in per_slice]
        if any(c.dtype == object for c in col):
            col = [_obj(c) for c in col]
        stacked.append(np.stack(col, axis=int(out_axes[k])))
    return stacked


def _bind_body_inputs(body: Program, args: list[Any], env: _Env) -> bool:
    """Bind operand values onto a sub-program's input atoms; False on mismatch."""
    if len(args) != len(body.inputs):
        return False
    for inp, val in zip(body.inputs, args):
        sa = body.input_arrays.get(inp.name)
        if sa is None or getattr(sa, "_bulk", None) is not None:
            return False                      # bulk / dynamic body input — opaque
        cells = np.asarray(sa.cells)
        if val is _OPAQUE:
            # An UNRESOLVED operand: mark this input's atoms unresolvable instead of refusing the
            # whole body (`_tolerates_opaque_operand`). Everything that reads them goes opaque, so
            # a body whose declared outputs still resolve has PROVED the operand dead — the same
            # liveness argument `_run_program` makes for an opaque statement inside the body.
            for idx in (np.ndindex(*cells.shape) if cells.shape else [()]):
                cell = cells[idx] if cells.shape else cells[()]
                if not (isinstance(cell, RationalFunction) and cell.gens):
                    return False              # a non-atom placeholder cell would be read as a
                    # VALUE (nothing marks it unresolvable) — refuse rather than use it
                env.opaque.add(cell.gens[0])
            continue
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
    body: Program, args: list[Any], meter: _Meter, depth: int, reason: _Reason,
    max_sym_mass: int,
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
        if not _exec_stmt(body, i, stmt, env, meter, depth, reason, max_sym_mass):
            # KEEP GOING.  This used to `return None` ("one opaque stmt poisons the body"), which
            # made the lane refuse on the PRESENCE of an un-normalizable op rather than on the
            # RESULT depending on it.  A statement we cannot execute simply leaves its outputs
            # unbound, and `_resolve_ref` already yields `_OPAQUE` for those — so opacity cascades
            # to exactly the consumers that genuinely need it, and the declared-output check below
            # still returns `None` if the result is among them.  If the body's outputs resolve
            # anyway, the opaque statement was DEAD and refusing was pure loss.
            #
            # This is what unblocks a certificate whose `grass_dof` body still CONTAINS a `QrOp`
            # that nothing downstream reads: liveness becomes implicit — "the result is a number"
            # is itself the proof that the op did not matter, and the statements can be discarded
            # afterwards.  (Measured: `Λᵏ` face DOFs reach 0 LIVE frame ops but keep a dead
            # `QrOp` in the body; presence-based refusal alone kept them on the probe lane.)
            continue
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
    program: Program, i: int, stmt: Stmt, env: _Env, meter: _Meter, depth: int,
    reason: _Reason, max_sym_mass: int,
) -> bool:
    """Execute one statement into ``env``; False when opaque/unresolvable."""
    _charge(meter, _STMT_COST)
    if stmt.fn is None or isinstance(stmt.fn, WhileOp):
        reason.note("WhileOp / fn-less statement (never executed at build time)")
        return False                          # loops are never executed at build time
    args: list[Any] = []
    # A `SwitchOp`'s SCRUTINEE is allowed to stay unresolved. Its inputs are
    # `(scrutinee, branch_0, …)`, and when every branch carries the SAME value the switch is the
    # identity no matter which is picked — so the branch that a run-time-only `IntAtom` would select
    # is irrelevant. Aborting on the unresolved scrutinee (as every other opaque operand does, and as
    # this did) throws that away and freezes the whole downstream expression. Any opaque BRANCH is
    # still fatal, exactly as before. See the `SwitchOp` twin in `_sym_apply`.
    _is_switch = isinstance(stmt.fn, SwitchOp)
    # An unresolved operand is NOT automatically fatal (`_tolerates_opaque_operand`): a
    # sub-Program may not read it, and a MULTILINEAR op with an exactly-zero factor is zero
    # whatever it holds. Both are decided below, from the operands — never assumed.
    _tolerant = _tolerates_opaque_operand(stmt.fn)
    opaque_at: list[int] = []
    for _j, r in enumerate(stmt.in_):
        v = _resolve_ref(r, env, program)
        if v is _OPAQUE:
            if _is_switch and _j == 0:
                args.append(_OPAQUE)          # decided by the twin, from the branches alone
                continue
            if _tolerant:
                opaque_at.append(_j)
                args.append(_OPAQUE)
                continue
            reason.note("operand depends on an unresolved statement / runtime-only ref")
            return False
        args.append(v)
    if opaque_at and is_builtin_op(stmt.fn) and _zero_absorbing(stmt.fn):
        # THE ZERO SHORT-CIRCUIT, and it belongs to MULTILINEARITY ALONE. Substituting a zero for an
        # operand we could not resolve is only ever justified by `_zero_absorbing`: each output entry
        # is then a sum of products with one factor per operand, so the zero factor annihilates every
        # term and what the unresolved one holds cannot matter.
        #
        # ⚠ The guard must NOT be `is_builtin_op` alone. `CallOp` IS a builtin op, and
        # `_tolerates_opaque_operand` admits `CallOp(Program)` for a completely different reason —
        # the LIVENESS argument, decided by executing the body. Fabricating a zero there claims a
        # value for an ADDITIVE body: a `CallOp` over `u + v` with `u = 0` and `v` unresolved would
        # certify `0` where the true value is `v`. Tolerating an unresolved operand and inventing a
        # value for it are two different permissions.
        #
        # Substituting a zero array of the unresolved operand's DECLARED shape lets the existing
        # rational twin compute both the value and its shape — no separate zero-shaped result to
        # get wrong.
        zeros_ok = any(_is_exact_zero(a) for j, a in enumerate(args) if j not in set(opaque_at))
        shapes = [_ref_shape(stmt.in_[j], program) for j in opaque_at] if zeros_ok else []
        if not zeros_ok or any(s is None for s in shapes):
            reason.note("operand depends on an unresolved statement / runtime-only ref")
            return False
        for j, shp in zip(opaque_at, shapes):
            args[j] = np.zeros(shp)
    _has_opaque = any(a is _OPAQUE for a in args)
    numeric = [None] * len(args) if _has_opaque else [_as_numeric(a) for a in args]
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
        # WORK CAP before the expensive symbolic execution: one einsum / Gauss pass over
        # object-dtype RF cells cannot be interrupted mid-flight, so oversized operands
        # are rejected UP FRONT (⇒ unresolved ⇒ the warned probe fallback).  Sub-program
        # bodies are exempt here — their own statements are capped individually as they
        # execute, which is the finer (and cheaper) granularity.
        if not isinstance(stmt.fn, (Program, CallOp)) and not _has_opaque:
            mass = _sym_mass(args, max_sym_mass)
            if mass > max_sym_mass:
                reason.note(
                    f"operands too large for exact execution ({type(stmt.fn).__name__}: "
                    f"monomial mass > {max_sym_mass})")
                return False
            # The operand mass is the work this op is about to do, and it has ALREADY been
            # measured for the cap — so charging it is free.  This is the dominant term of the
            # whole budget: it is what the wall clock was really measuring.
            _charge(meter, mass)
        try:
            outs = _sym_apply(stmt.fn, args, meter, depth, reason, max_sym_mass)
        except _Exhausted:
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

def exact_partial_eval(
    program: Program, *, work_budget: int | None = None, max_sym_mass: int = _MAX_SYM_MASS,
    wall_backstop: float | None = None,
) -> ExactState:
    """One exact pass over ``program``: fold / refute / leave-unresolved each Stmt.

    Cost is bounded by BOTH knobs: ``work_budget`` (deterministic work units, charged between
    operations — see :class:`_Meter`) and ``max_sym_mass`` (:data:`_MAX_SYM_MASS` — the monomial
    mass one symbolic op's operands may carry, checked BEFORE the op runs, because a single
    object-dtype einsum / Gauss pass cannot be interrupted mid-flight).

    The budget is a function of the PROGRAM, not of the machine: the same program spends the
    same units under any load, so the resulting certificate is reproducible.  ``wall_backstop``
    is a generous seconds cap that only guards a mis-calibrated cost model; when it fires the
    result IS machine-dependent, so it is recorded on :attr:`ExactState.hit_wall_clock` and
    `simplify` turns it into a loud ``NonDeterministicFoldWarning``.  ``None`` disables it.

    Never raises on op content — every failure mode degrades to ``unresolved``
    (which ``mode="hybrid"`` hands to the loud probe fallback).
    """
    from .simplify import _record_known

    limit = _resolve_work_budget(work_budget)
    meter = _Meter(limit=limit,
                   wall_deadline=_wall_deadline(wall_backstop))
    state = ExactState(limit=limit)
    env = _Env()
    reason = _Reason()                         # per-call holder (threaded explicitly)
    for i, stmt in enumerate(program.statements):
        reason.value = None                    # fresh slate per top-level statement
        try:
            ok = _exec_stmt(program, i, stmt, env, meter, 0, reason, max_sym_mass)
        except _Exhausted as exc:
            note = _exhausted_note(exc, meter)
            for j in range(i, len(program.statements)):
                state.unresolved[j] = note
            break
        if not ok:
            fallback = type(stmt.fn).__name__ if stmt.fn is not None else "no fn"
            state.unresolved[i] = reason.take(fallback)
            continue
        # Classify: all-constant ⇒ fold; any provably non-constant cell ⇒ refute.
        # BUDGET-AWARE (the classification is where a pathological cell's exact
        # evaluation/gcd bill lands — the degree-5 regime): exhaustion degrades this and
        # every remaining statement to *unresolved* (⇒ the warned probe fallback),
        # exactly like exhaustion in the execution loop above.
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
                    cv = _cell_constant(c, meter)
                    if cv is None:
                        verdict = "refute"
                        break
                    vals[idx] = cv
                if verdict != "fold":
                    break
                const_outs.append(vals)
        except _Exhausted as exc:
            note = _exhausted_note(exc, meter)
            for j in range(i, len(program.statements)):
                state.unresolved[j] = note
            break
        if verdict == "fold":
            # All-or-nothing WITHOUT copying the accumulated bindings per statement (the same
            # quadratic spelling `simplify`'s probe lane carried): `_record_known` only WRITES
            # into the dict it is given, so recording into an empty one and merging on success
            # is entry-for-entry identical, while a raise still leaves `state.known` untouched.
            staged: dict[str, Any] = {}
            try:
                _record_known(stmt, const_outs, staged)
            except ValueError:
                state.unresolved[i] = "fold shape mismatch"
                continue
            state.known.update(staged)
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
    state.spent = meter.spent
    state.hit_wall_clock = meter.hit_wall_clock
    return state


def exact_fold_cells(
    cells: np.ndarray, state: ExactState, program: Program, *,
    work_budget: int | None = None, max_sym_mass: int = _MAX_SYM_MASS,
    wall_backstop: float | None = None,
) -> np.ndarray:
    """ENTRY-LEVEL exact fold of an output cell array.

    Each cell is normalized as a rational function of the FEED atoms by substituting
    the exact (possibly non-constant) statement values from ``state``; a cell whose
    normal form has total degree zero is replaced by its exact constant.  Cells that
    cannot be normalized (opaque statements in their cone) or that are genuinely
    non-constant are left UNTOUCHED — including object identity when nothing folds
    (the ``_fold_cells`` structure-transparency contract).
    """
    cells = np.asarray(cells)
    if cells.dtype.kind == "f":
        return cells
    meter = _Meter(limit=_resolve_work_budget(work_budget),
                   wall_deadline=_wall_deadline(wall_backstop))
    scalar_known = {k: v for k, v in state.known.items()
                    if not isinstance(v, np.ndarray) or v.shape == ()}
    env = _Env(atom={**scalar_known, **state.sym}, bulk=dict(state.sym_bulk))
    out = np.empty(cells.shape, dtype=object)
    flat_in = cells.reshape(-1)
    flat_out = out.reshape(-1)
    flat_out[:] = flat_in                     # pre-fill: a budget break must leave no hole
    changed = False
    for i, c in enumerate(flat_in):
        if not isinstance(c, RationalFunction):
            continue
        # WORK CAP on the substitution too: `_resolve_rf` composes the (possibly huge)
        # statement values into this cell, and one `compose_multi` runs uninterrupted.
        mass = _rf_mass(c)
        if mass > max_sym_mass:
            continue
        try:
            _charge(meter, mass)               # the substitution this cell is about to pay for
        except _Exhausted:
            break                              # (remaining cells stay untouched — pre-filled)
        v = _resolve_rf(c, env, program, max_sym_mass)
        if v is _OPAQUE or not isinstance(v, RationalFunction):
            continue
        try:
            cv = _constant_value(v, meter)     # budget INSIDE the cell too — a single
        except _Exhausted:                     # pathological normal form must not overrun
            break                              # (remaining cells stay untouched)
        if cv is not None:
            flat_out[i] = cv
            changed = True
    # ACCUMULATE onto the state: `spent` is the work this certificate cost in total, across
    # both passes, not just the statement pass.  On the FEEC legs the statement pass refutes
    # the vertex-dependent pieces and folds nothing — the cancellation completes HERE, per
    # cell — so a `spent` that omitted this pass would under-report the real cost of the
    # certificate by most of it.
    state.spent += meter.spent
    state.hit_wall_clock = state.hit_wall_clock or meter.hit_wall_clock
    return out if changed else cells
