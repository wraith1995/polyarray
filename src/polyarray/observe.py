"""Staged compile observability: measure a stage's IR, log it, warn on it, dump it.

The consumer packages drive a multi-stage compile — enumerate, sample,
represent, integrate, lower, torch-compile — and every hard bug in this stack has been "some
stage blew the IR up and we could not see which".  This module is the instrument: a consumer
calls :meth:`CompileTrace.stage` at a boundary with whatever that boundary produced (a
:class:`~polyarray.ir.Program`, a :class:`~polyarray.ir.SymArray`, an array), and gets a
:class:`Snapshot` recorded — symbolic mass, operand mass, cell counts, provenance histogram,
polynomial degree — plus the **delta from the previous stage**, which is the thing that
actually names a culprit.

Everything is measured with the analysis polyarray already owns
(:func:`polyarray.forward.analyze`, :func:`polyarray.degree.program_degree`,
:func:`polyarray.ir._cell_size`); this module adds no parallel cost model.

**Layering.**  This lives in polyarray because polyarray owns those primitives *and* the
pyab→torch dump hand-off, and because polyarray is a dependency of every layer that wants to
be observed — so instrumentation never inverts the stack.  The lower algebra and geometry
layers are deliberately NOT instrumented and must not import this.

**Levels** (``FEM_OBSERVE``, default ``warn``)::

    off    nothing at all; `stage()` returns on its first line
    warn   proactive warnings only (oversized IR, degree jump)
    info   + one line per stage with mass / operand_mass / degree and the delta
    debug  + the per-sub-program breakdown and the provenance histogram
    dump   + write `<FEM_OBSERVE_DIR>/NN-<stage>/` — the expensive level

At ``dump`` a stage directory doubles as a pyab ``dump_dir``
(:meth:`CompileTrace.dump_dir_for`), so a lowering stage's ``torch.py`` / ``ir.txt`` /
``inductor/`` / ``dynamo_explain.txt`` land beside the symbolic snapshot that produced them:
one numbered listing is the whole compile, symbolic IR through to the generated kernel.

Two guarantees the call sites depend on:

* **Measurement never breaks a compile.**  Every probe is failure-tolerant; a Snapshot with
  missing numbers is the worst case.
* **Nothing is forced.**  Bulk (deferred) nodes are counted as deferred and never
  materialised — :func:`polyarray.forward.analyze` already honours this, and the dump writer
  reads placeholder cells the same way.
"""
from __future__ import annotations

import logging
import os
import time
import warnings
import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .forward import IRReport
    from .ir import Program, SymArray

#: Anything a call site can hand the tracer: a :class:`~polyarray.ir.Program`, a value that
#: carries one (a ``SymArray``, a front-end compile result), a plain array or scalar with no
#: program at all, or ``None`` for a stage with no product.
type Observable = object

logger = logging.getLogger("fem.observe")

_ENV_LEVEL = "FEM_OBSERVE"
_ENV_DIR = "FEM_OBSERVE_DIR"
_ENV_MASS = "FEM_OBSERVE_MASS_CEILING"
_ENV_OPERAND = "FEM_OBSERVE_OPERAND_CEILING"
_ENV_CELLS = "FEM_OBSERVE_CELLS_CEILING"
_ENV_DEGREE = "FEM_OBSERVE_DEGREE_CEILING"

# Warning thresholds.  Generous on purpose: a warning must be signal, not noise.  These are the
# "this stage is already pathological" marks, not "this stage is big" — a high-degree value kernel
# legitimately carries tens of thousands of monomials.  Tunable per run via the env vars above.
_MASS_CEILING = 250_000        # Σ monomials over symbolic OUTPUT cells
_OPERAND_CEILING = 1_000_000   # Σ monomials over non-bulk RF OPERAND cells (the real lowering cost)
_CELLS_CEILING = 16_384        # symbolic output cells (the eager per-cell scatter fingerprint)
_DEGREE_CEILING = 24.0         # polynomial degree in the program's inputs

# A stage whose operand mass multiplies by more than this over its parent is a *jump* — the
# blow-up signature the epic is built to surface (pullback ξ going rational, Faà di Bruno
# transport).  Growth is normal; an order of magnitude in one stage is not.
_JUMP_FACTOR = 10.0
_JUMP_FLOOR = 10_000           # below this absolute mass a jump is uninteresting

_INF = float("inf")


class Level(IntEnum):
    """Observability verbosity.  Ordered — a feature is active at its level and above."""

    OFF = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    DUMP = 4


_LEVEL_NAMES = {lv.name.lower(): lv for lv in Level}


def level_from_env() -> Level:
    """Read the level named by ``FEM_OBSERVE``, defaulting to :attr:`Level.WARN`.

    An unrecognised value falls back to ``WARN`` and warns: a typo in the environment variable
    must not silently disable the instrument the caller just asked for.
    """
    raw = os.environ.get(_ENV_LEVEL)
    if raw is None:
        return Level.WARN
    key = raw.strip().lower()
    lv = _LEVEL_NAMES.get(key)
    if lv is None:
        warnings.warn(
            f"{_ENV_LEVEL}={raw!r} is not one of {sorted(_LEVEL_NAMES)}; using 'warn'",
            stacklevel=2)
        return Level.WARN
    return lv


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not an integer; using {default}", stacklevel=2)
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not a number; using {default}", stacklevel=2)
        return default


def _ensure_logging(lv: Level) -> None:
    """Give ``fem.observe`` a stderr handler when the app has not configured logging.

    Observability nobody sees is not observability: the point of ``FEM_OBSERVE=info`` is that a
    plain script prints the staged trace, and a test that passes ``level=Level.DUMP`` in code
    must get the same treatment (hence this runs per-trace, not once at import).  Deliberately a
    no-op when this logger already has a handler or the root logger is configured, so an
    application's own logging setup always wins.
    """
    if lv < Level.INFO:
        return
    want = logging.DEBUG if lv >= Level.DEBUG else logging.INFO
    if logger.level == logging.NOTSET or logger.level > want:
        logger.setLevel(want)
    if logger.handlers or logging.getLogger().handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """The size/degree/provenance numbers for one observed object.

    ``kind`` names what was measured (``program`` / ``symarray`` / ``array`` / ``scalar`` /
    ``none`` / ``unmeasurable``).  Every count is ``0`` and ``degree`` ``None`` when the object
    carries no symbolic IR — a numeric stage is legitimately weightless, and that is itself the
    signal "this stage stayed numeric".
    """

    kind: str = "none"
    n_programs: int = 0
    n_stmts: int = 0
    n_defer: int = 0
    out_mass: int = 0
    operand_mass: int = 0
    n_output_cells: int = 0
    symbolic_cells: int = 0
    n_operand_cells: int = 0
    max_cell: int = 0
    max_operand: int = 0
    degree: float | None = None
    prov_kinds: Mapping[str, int] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    detail: str = ""          # the per-sub-program IRReport text, for `debug`/`dump`
    error: str = ""           # set when a probe failed; the compile is unaffected
    # Identity of the Program these numbers were read off, when there was one.
    program_id: int | None = None

    @property
    def mass(self) -> int:
        """The headline cost — operand mass when there is any, else output mass.

        Operand mass is the number that predicts lowering cost (einsum/linalg statements defer
        their *outputs* to atoms, so output mass reads tiny while the operands expand into the
        codegen AST); output mass is the meaningful one for a stage that has not yet emitted
        statements.
        """
        return self.operand_mass or self.out_mass


_EMPTY = Measurement()


# The attribute chain the consumers' wrappers use to hold the program they were built from:
# a SymArray and a pointwise `Compiled` both expose `.program`, but a `Compiled`'s is a
# `GrassmannProgram` wrapper whose polyarray `Program` is `.prog`.  Walking a couple of links
# lets a call site hand this module whatever its stage produced without unwrapping first.
_PROGRAM_ATTRS = ("program", "prog")
_MAX_UNWRAP = 4


def _program_of(obj: Observable) -> Program | None:
    """Find the :class:`Program` behind ``obj``.

    Accepts a program directly, or any wrapper reachable from it through
    :data:`_PROGRAM_ATTRS` within :data:`_MAX_UNWRAP` links. This is measurement plumbing,
    not dispatch: nothing here branches on what the wrapper *is*. It looks for a program and
    gives up quietly if there is none, in which case the stage is recorded as weightless.
    """
    from .ir import Program

    seen: set[int] = set()
    cur = obj
    for _ in range(_MAX_UNWRAP):
        if isinstance(cur, Program):
            return cur
        if cur is None or id(cur) in seen:
            return None
        seen.add(id(cur))
        for attr in _PROGRAM_ATTRS:
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                cur = nxt
                break
        else:
            return None
    return cur if isinstance(cur, Program) else None


#: Name of the output :func:`_program_with_result` declares on its copy for the staged value.
_RESULT_OUTPUT = "staged_result"


def _program_with_result(program: Program, obj: Observable) -> tuple[Program, str]:
    """``program`` extended with the staged value as a declared OUTPUT, plus a note for the header.

    Parameters
    ----------
    program
        The :class:`~polyarray.ir.Program` the staged value was computed on.
    obj
        Whatever the call site staged.  Only a :class:`~polyarray.ir.SymArray` riding ``program``
        gains an output; anything else (a bare ``Program``, a wrapper) is returned untouched.

    Returns
    -------
    tuple
        ``(program_to_render, note)`` — a COPY carrying the extra output when one was added, else
        ``program`` itself; ``note`` is a header comment for the rendered file, empty when nothing
        changed.

    Notes
    -----
    A staged ``SymArray`` is usually not a declared output of the program its cells are expressions
    over — the statements are the sampling/lowering chain, and the value the stage is *about* lives
    only in the cells the caller holds.  Rendered as-is, the emitted function therefore computes
    every statement and then ends ``return None``: nothing connects the code to the matrix.
    Declaring the array as an output makes the rendered function return it.

    ⚠ OBSERVATION-ONLY.  The output is declared on :meth:`~polyarray.ir.Program.copy`, never on the
    live program: a dump must not change what the compile does, and ``add_output`` mutates the
    program it is called on.  ``copy`` shares only the frozen input descriptors and ring data, so
    the caller's ``outputs`` dict is untouched.

    Nothing is FORCED.  ``add_output`` accepts the ``SymArray`` itself, which copies the cell
    placeholder and the bulk handle without reading ``cells`` — so a bulk (deferred) array is
    declared as bulk rather than materialised.
    """
    from .ir import SymArray

    if not isinstance(obj, SymArray):
        return program, ""
    own = obj.program
    if own is None or own is not program:
        return program, ""
    if any(out is obj or out._cells is obj._cells for out in own.outputs.values()):
        return program, ""                       # already the program's own result — render as-is
    try:
        copy = own.copy()
        name, n = _RESULT_OUTPUT, 1
        while name in copy.outputs:
            n += 1
            name = f"{_RESULT_OUTPUT}_{n}"
        copy.add_output(name, obj)
    except Exception as exc:                      # noqa: BLE001 — a dump never breaks a compile
        return program, (f"# the staged value could not be declared as an output "
                         f"({type(exc).__name__}: {exc}); the program alone is below\n")
    return copy, (f"# the value this stage was measured on is returned as {name!r}"
                  f" (declared on a COPY of the program — the compile is unchanged)\n")


def _one(_name: str) -> int:
    """Score every generator as degree 1, the seed this module reports degrees against."""
    return 1


def _degree_of(program: Program, seed: Mapping[str, float] | None) -> float | None:
    """Polynomial degree of ``program``'s outputs, **in its symbolic atoms**.

    The default scores every declared input AND every stray generator as degree 1, which makes
    the number a consistent monotone signal across stages without this module needing any
    finite-element domain knowledge (that seeding lives in the front end's degree estimation, where
    it belongs).  A caller with a sharper seed — vertices only, say — passes one.

    Scoring the *generators* (``gen_deg``), not only the input names, is load-bearing: a stage
    that has emitted no statements yet carries its whole computation in the output cells'
    monomials, and those cells are object arrays the degree walk did not produce, so they are
    scored as weighted leaves.  Seeding input names alone reports 0 for them.

    Degrees are read after deferral, so a captured atom (``stmt_out``) counts as 1 even though it
    stands for something larger — the honest reading is "degree in the atoms live at this stage".
    """
    from .degree import program_degree

    if not program.outputs:
        # No declared outputs yet — mid-build, the stage's value lives in the SymArray the caller
        # is holding, not in the program.  Reporting 0.0 here would read as "constant", which is a
        # different and wrong claim; `None` says "not known at this stage".
        return None
    if seed is not None:                                # a caller-supplied seed bypasses the cache
        return float(program_degree(program, seed, gen_deg=_one))

    def compute(p: Program) -> float:
        return float(program_degree(p, {name: 1.0 for name in p.input_arrays}, gen_deg=_one))

    return _cached(program, _SLOT_DEGREE, compute)


def _symarray_cells(obj: SymArray) -> tuple[int, int, int, int]:
    """Count a SymArray's own cells as ``(n_cells, symbolic_cells, mass, max_cell)``.

    A bulk (deferred) SymArray reports its cell count from the placeholder shape and zero
    mass: its value has not been materialised, and reading it is exactly the forcing this
    module must never do.
    """
    import numpy as np

    from .ir import _cell_size
    from .rational import RationalFunction

    cells = obj._cells
    if getattr(obj, "_bulk", None) is not None:
        return int(np.asarray(cells).size), 0, 0, 0
    arr = np.asarray(cells)
    n = sym = mass = mx = 0
    for c in arr.reshape(-1):
        n += 1
        if isinstance(c, RationalFunction):
            sym += 1
            sz = _cell_size(c)
            mass += sz
            mx = max(mx, sz)
    return n, sym, mass, mx


def measure(obj: Observable, *, degree_seed: Mapping[str, float] | None = None) -> Measurement:
    """Measure a program, SymArray, ndarray, scalar, or ``None``.

    Failure-tolerant by construction: a probe that raises yields a :class:`Measurement`
    carrying the exception text in :attr:`Measurement.error` rather than propagating.
    Instrumentation that can break a compile is worse than no instrumentation.
    """
    if obj is None:
        return _EMPTY
    try:
        return _measure(obj, degree_seed)
    except Exception as exc:                                  # never break a compile
        return Measurement(kind="unmeasurable", error=f"{type(exc).__name__}: {exc}")


def _measure(obj: Observable, degree_seed: Mapping[str, float] | None) -> Measurement:
    """Measure ``obj``, letting any probe failure propagate to :func:`measure`."""
    import numpy as np

    from .ir import Program, SymArray

    program = _program_of(obj)
    if isinstance(obj, SymArray):
        n, sym, mass, mx = _symarray_cells(obj)
        rep = _analyze(program) if program is not None else None
        return Measurement(
            kind="symarray",
            n_programs=0 if rep is None else rep.n_programs,
            n_stmts=0 if rep is None else sum(r.n_stmts for r in rep.rows),
            n_defer=0 if rep is None else rep.n_defer,
            out_mass=mass,
            operand_mass=0 if rep is None else rep.total_operand_mass,
            n_output_cells=n,
            symbolic_cells=sym,
            n_operand_cells=0 if rep is None else sum(r.n_operand_cells for r in rep.rows),
            max_cell=mx,
            max_operand=0 if rep is None else max((r.max_operand for r in rep.rows), default=0),
            degree=None if program is None else _degree_of(program, degree_seed),
            prov_kinds={} if rep is None else rep.prov_kinds(),
            shape=tuple(np.asarray(obj._cells).shape),
            detail="" if rep is None else str(rep),
            program_id=None if program is None else id(program),
        )
    if isinstance(obj, Program):
        rep = _analyze(obj)
        return Measurement(
            kind="program",
            n_programs=rep.n_programs,
            n_stmts=sum(r.n_stmts for r in rep.rows),
            n_defer=rep.n_defer,
            out_mass=rep.total_mass,
            operand_mass=rep.total_operand_mass,
            n_output_cells=sum(r.n_output_cells for r in rep.rows),
            symbolic_cells=sum(r.symbolic_cells for r in rep.rows),
            n_operand_cells=sum(r.n_operand_cells for r in rep.rows),
            max_cell=max((r.max_cell for r in rep.rows), default=0),
            max_operand=max((r.max_operand for r in rep.rows), default=0),
            degree=_degree_of(obj, degree_seed),
            prov_kinds=rep.prov_kinds(),
            detail=str(rep),
            program_id=id(obj),
        )
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # An object array of RationalFunctions — the pre-SymArray shape a sampler hands
            # back.  Weigh the cells directly; this is the same accounting `_symarray_cells`
            # does, without a program to attribute it to.
            from .ir import _cell_size
            from .rational import RationalFunction

            n = sym = mass = mx = 0
            for c in obj.reshape(-1):
                n += 1
                if isinstance(c, RationalFunction):
                    sym += 1
                    sz = _cell_size(c)
                    mass += sz
                    mx = max(mx, sz)
            return Measurement(kind="array", out_mass=mass, n_output_cells=n,
                               symbolic_cells=sym, max_cell=mx, shape=tuple(obj.shape))
        return Measurement(kind="array", n_output_cells=int(obj.size), shape=tuple(obj.shape))
    if isinstance(obj, (list, tuple)):
        # A stage that produced several artefacts (savo's `entries`, a jet's per-order arrays).
        # Sum the parts so the stage still reports one honest total.
        parts = [_measure(o, degree_seed) for o in obj]
        # Propagate the program identity when every part came off the SAME program (a jet's
        # per-order arrays do), so the roll-up can still tell that repeated occurrences are
        # views of one growing program rather than independent work.
        pids = {p.program_id for p in parts if p.program_id is not None}
        return Measurement(
            kind="array",
            program_id=pids.pop() if len(pids) == 1 else None,
            n_programs=sum(p.n_programs for p in parts),
            n_stmts=sum(p.n_stmts for p in parts),
            n_defer=sum(p.n_defer for p in parts),
            out_mass=sum(p.out_mass for p in parts),
            operand_mass=sum(p.operand_mass for p in parts),
            n_output_cells=sum(p.n_output_cells for p in parts),
            symbolic_cells=sum(p.symbolic_cells for p in parts),
            n_operand_cells=sum(p.n_operand_cells for p in parts),
            max_cell=max((p.max_cell for p in parts), default=0),
            max_operand=max((p.max_operand for p in parts), default=0),
            degree=max((p.degree for p in parts if p.degree is not None), default=None),
            shape=(len(obj),),
        )
    if program is not None:
        return _measure(program, degree_seed)
    return Measurement(kind="scalar")


# Analysis cache.  `forward.analyze` and `program_degree` are both O(program size), and the
# instrumented boundaries re-measure the SAME program over and over — a high-degree element calls
# `bind-field` ~1000 times against one growing multi-million-monomial program, which made the
# default level 3.3× slower
# than no instrumentation at all.  Keyed weakly by program so nothing is kept alive, and
# invalidated by a cheap structural version (statement / output / input counts), which is what
# actually changes as a program is built up.
#
# The trade: a program mutated IN PLACE without changing any of those counts serves a stale
# reading until it next grows.  That is a measurement being briefly out of date, not a wrong
# compile — and it is the price of the default level being usable at all.
_ANALYSIS_CACHE: Any = None


def _program_version(program: Program) -> tuple[int, int, int]:
    """Summarise a program's structure, so a cached measurement can detect that it grew."""
    return (len(program.statements), len(program.outputs), len(program.input_arrays))


def _cached[T](program: Program, slot: int, compute: Callable[[Program], T]) -> T:
    """Return ``compute(program)``, memoized per program and structural version in ``slot``."""
    global _ANALYSIS_CACHE
    import weakref

    if _ANALYSIS_CACHE is None:
        _ANALYSIS_CACHE = weakref.WeakKeyDictionary()
    ver = _program_version(program)
    try:
        entry = _ANALYSIS_CACHE.get(program)
    except TypeError:                                   # not weak-referenceable / not hashable
        return compute(program)
    if entry is None or entry[0] != ver:
        entry = (ver, {})
        try:
            _ANALYSIS_CACHE[program] = entry
        except TypeError:
            return compute(program)
    if slot not in entry[1]:
        entry[1][slot] = compute(program)
    return entry[1][slot]


_SLOT_REPORT = 0
_SLOT_DEGREE = 1


def _analyze(program: Program) -> IRReport:
    """Return the cached :func:`polyarray.forward.analyze` report for ``program``."""
    from . import forward

    return _cached(program, _SLOT_REPORT, forward.analyze)


# ---------------------------------------------------------------------------
# Snapshots + the trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """One observed stage: what it was, what it produced, and how long it took.

    A stage inside a loop — ``single_compile`` runs once per enumerated match, so its
    ``sample`` / ``represent`` boundaries fire hundreds of times in one assembly — is **rolled
    up**: every recording of the same stage name at the same depth accumulates into this one
    Snapshot rather than appending a new row.  Without that a large compile produces an
    800-row table and 800 dump directories, which is not observability, it is a log flood.

    Under roll-up, :attr:`m` is the measurement of the largest MEASURED occurrence (the one that
    matters for a blow-up), :attr:`elapsed_s` is cumulative over every occurrence, :attr:`count`
    says how many times the stage ran, and :attr:`n_measured` how many of those were measured
    (see :meth:`CompileTrace._should_measure` — below ``debug`` the occurrences are sampled).
    """

    seq: int
    stage: str
    depth: int
    elapsed_s: float
    m: Measurement
    ctx: Mapping[str, Any] = field(default_factory=dict)
    parent: int | None = None       # seq of the stage this one is compared against
    count: int = 1                  # occurrences rolled up into this row
    n_measured: int = 1             # how many of them were actually MEASURED (see `_should_measure`)

    @property
    def sampled(self) -> bool:
        """Whether some occurrences were counted and timed but not measured."""
        return self.n_measured < self.count

    @property
    def slug(self) -> str:
        """Filesystem-safe ``NN-stage`` name — the dump directory and the report's row key."""
        return f"{self.seq:02d}-{_safe_name(self.stage)}"

    def merged_with(self, other: Snapshot, *, measured: bool) -> Snapshot:
        """Roll a fresh occurrence of the same stage into this row.

        ``measured`` says whether ``other`` carries a real measurement or is a counted-and-timed
        occurrence that was sampled out; an unmeasured one updates only the counters.
        """
        bigger = other.m if (measured and other.m.mass > self.m.mass) else self.m
        return dataclasses.replace(
            self,
            m=bigger,
            ctx=other.ctx if bigger is other.m else self.ctx,
            elapsed_s=self.elapsed_s + other.elapsed_s,
            count=self.count + 1,
            n_measured=self.n_measured + (1 if measured else 0),
        )


def _fmt_delta(cur: int, prev: int | None) -> str:
    """Compute the stage-over-stage mass delta, with a ratio only where a ratio means something.

    A ratio is dropped when either end is zero: ``×0.0`` on a stage that went numeric reads as a
    catastrophic shrink when it just means "no symbolic IR here any more", and a ratio against
    zero is undefined.
    """
    if prev is None or prev == cur:
        return ""
    if prev == 0 or cur == 0:
        return f"  ({cur - prev:+d})"
    return f"  ({cur - prev:+d}, ×{cur / prev:.1f})"


class CompileTrace:
    """A recording of one compile, stage by stage.

    Created by :func:`observe_compile` and reached from instrumented code via :func:`trace`.
    A trace is single-threaded-per-context (it rides a :class:`~contextvars.ContextVar`), so
    concurrent compiles do not interleave.
    """

    def __init__(self, name: str, *, level: Level | None = None,
                 dump_root: Path | None = None) -> None:
        self.name = name
        self.level = level_from_env() if level is None else level
        self.snapshots: list[Snapshot] = []
        self._index: dict[tuple[int, str], int] = {}   # (depth, stage) -> row, for roll-up
        self._depth = 0
        self._seq = 0
        self._dump_root = dump_root
        # (depth, stage) -> its dump dir. Keyed the SAME way rows are: the same stage name at two
        # DEPTHS is two distinct rows, and keying the directory by name alone silently collapsed
        # them into one (savo's `dof-table` runs at two depths and the deeper, 30-occurrence row
        # lost its directory entirely).
        self._dirs: dict[tuple[int, str], Path] = {}
        self._off_paths: dict[str, str] = {}   # uninstrumented routes taken (see `off_path`)
        self._detail_text: dict[tuple[int, str], str] = {}    # rendered caller descriptions
        self._program_text: dict[tuple[int, str], bool] = {}  # stages whose IR has been rendered
        self._t0 = time.perf_counter()
        _ensure_logging(self.level)           # a level set in code, not env, must still be visible

    # -- the recording surface ---------------------------------------------

    def stage(self, name: str, obj: Observable = None, *, elapsed_s: float = 0.0,
              degree_seed: Mapping[str, float] | None = None,
              detail: Callable[[], str] | None = None, **ctx: object) -> Snapshot | None:
        """Record ``obj`` as the result of stage ``name``.

        ``detail`` is a zero-argument callable returning extra text for the stage's dump — what
        this stage actually produced in the caller's own vocabulary (which basis, which terms,
        which bindings), which this module cannot know.  It is called **only** at
        :attr:`Level.DUMP`, and only once per stage, so building an expensive description costs
        nothing at the other levels.  Because it is a thunk it may close over locals that are not
        yet bound when ``phase()`` is entered — they are read when the phase exits.

        Returns the :class:`Snapshot`, or ``None`` at :attr:`Level.OFF` — where this method does
        no work beyond the level test, so instrumented call sites cost a comparison.

        ``**ctx`` is free-form, but the parameter names above are RESERVED — a ``ctx`` key called
        ``name``, ``obj``, ``elapsed_s``, ``degree_seed`` or ``detail`` collides with the signature
        and raises ``TypeError`` at the call site. Name such context something else
        (``element=``, ``stage_name=``).
        """
        if self.level <= Level.OFF:
            return None
        key = (self._depth, name)
        at = self._index.get(key)
        if at is not None:                      # a repeat of a stage inside a loop — roll it up
            prev = self.snapshots[at]
            do_measure = self._should_measure(prev.count + 1)
            m = measure(obj, degree_seed=degree_seed) if do_measure else _EMPTY
            fresh = Snapshot(seq=0, stage=name, depth=self._depth, elapsed_s=elapsed_s,
                             m=m, ctx=dict(ctx))
            snap = prev.merged_with(fresh, measured=do_measure)
            self.snapshots[at] = snap
            # Emit only when this occurrence RAISED the stage's high-water mark; an ordinary
            # repeat re-logging the same line a few hundred times is the flood roll-up prevents.
            self._emit(snap, obj, detail, quiet=snap.m is prev.m)
            return snap
        m = measure(obj, degree_seed=degree_seed)
        self._seq += 1
        snap = Snapshot(seq=self._seq, stage=name, depth=self._depth, elapsed_s=elapsed_s,
                        m=m, ctx=dict(ctx), parent=self._parent_seq())
        self._index[key] = len(self.snapshots)
        self.snapshots.append(snap)
        self._emit(snap, obj, detail)
        return snap

    @contextmanager
    def phase(self, name: str, *, degree_seed: Mapping[str, float] | None = None,
              detail: Callable[[], str] | None = None, **ctx: object) -> Iterator[list[Observable]]:
        """Time a block and record its result as a stage.

        Yields a one-element list; assign the stage's product to ``box[0]`` (or call
        ``box.append``) and it gets measured on exit.  Nested phases are indented in the report.

        The stage is recorded even when the block raises — a stage that blew up is *precisely*
        the one you want in the trace — and the exception then propagates untouched.
        """
        if self.level <= Level.OFF:
            yield []
            return
        box: list = []
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield box
        finally:
            self._depth -= 1
            self.stage(name, box[0] if box else None, elapsed_s=time.perf_counter() - t0,
                       degree_seed=degree_seed, detail=detail, **ctx)

    def note_off_path(self, name: str, why: str = "") -> None:
        """Record an uninstrumented / non-canonical path taken during this trace.

        See :func:`off_path` for why this exists. Recorded as a zero-cost stage (so it appears in
        the report *in sequence*, where the reader can see what it happened between) and warned
        about once, because the whole value is "you did not see this part."
        """
        if name in self._off_paths:
            return
        self._off_paths[name] = why
        self.stage(f"off-path/{name}", None, why=why or "uninstrumented")
        warnings.warn(
            f"[{self.name}] OFF-PATH: {name}"
            + (f" — {why}" if why else "")
            + ". This code path is not instrumented, so the staged trace does NOT account for it"
              " — treat any attribution across it as incomplete.",
            stacklevel=3)

    @property
    def off_paths(self) -> Mapping[str, str]:
        """The uninstrumented paths this trace hit — ``{name: why}``, in first-seen order."""
        return dict(self._off_paths)

    def _should_measure(self, occurrence: int) -> bool:
        """Whether the ``occurrence``-th run of a stage gets measured, or just counted and timed.

        Measuring is O(program size) — a full `forward.analyze` plus a degree walk — and the
        instrumented boundaries sit inside hot loops: a high-degree element calls `bind-field` ~1000
        times, each
        against a program that has GROWN since the last call, so the analysis cache cannot help
        and measuring every occurrence made the default level **3.3× slower than no
        instrumentation at all** (37s → 122s). That is not a tool anyone would leave on.

        So below :attr:`Level.DEBUG` the occurrences are sampled geometrically — 1, 2, 4, 8, … —
        giving O(log n) measurements instead of O(n) while keeping FULL fidelity for the small
        loops where n is small anyway. A blow-up that shows on occurrence 900 but on none of
        1, 2, 4, …, 512 is not a thing that happens. `debug` and `dump` measure everything,
        because there the user has explicitly asked for the detail and accepted the cost.
        """
        if self.level >= Level.DEBUG:
            return True
        return occurrence & (occurrence - 1) == 0        # powers of two

    # -- dump directories (also the pyab/torch hand-off) --------------------

    @property
    def dump_root(self) -> Path | None:
        """The dump root, or ``None`` below :attr:`Level.DUMP`."""
        if self.level < Level.DUMP:
            return None
        if self._dump_root is None:
            self._dump_root = Path(os.environ.get(_ENV_DIR, "fem-observe")) / self.name
        return self._dump_root

    def dump_dir_for(self, stage: str) -> Path | None:
        """Return the directory for ``stage``, created on demand, or ``None`` below :attr:`Level.DUMP`.

        This is the pyab hand-off: pass the result as ``pyab.compile_torch(dump_dir=...)`` and
        the generated ``torch.py`` / ``ir.txt`` / ``inductor/`` / ``dynamo_explain.txt`` land in
        the same numbered directory as the symbolic snapshot of the program that produced them.
        Passing ``None`` through to pyab is harmless — it then falls back to ``PYAB_DUMP_DIR``,
        which is unset in the normal case.
        """
        root = self.dump_root
        if root is None:
            return None
        key = (self._depth, stage)
        d = self._dirs.get(key)
        if d is None:
            # Number by the stage that has just been (or is about to be) recorded, so the
            # lowering artefacts sort next to their symbolic snapshot.
            d = root / f"{self._seq + 1:02d}-{_safe_name(stage)}"
            self._dirs[key] = d
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- emission ----------------------------------------------------------

    def _parent_seq(self) -> int | None:
        """Find the most recent snapshot at or above this depth — what a delta is measured against.

        Called from :meth:`stage` *before* the new snapshot is appended, so this walks the whole
        list: the last element IS the previous stage, not the current one.
        """
        for s in reversed(self.snapshots):
            if s.depth <= self._depth:
                return s.seq
        return None

    def _by_seq(self, seq: int | None) -> Snapshot | None:
        """Look a snapshot up by its sequence number."""
        if seq is None:
            return None
        for s in self.snapshots:
            if s.seq == seq:
                return s
        return None

    def _emit(self, snap: Snapshot, obj: Observable = None,
              detail: Callable[[], str] | None = None, *, quiet: bool = False) -> None:
        """Log / warn / dump a recorded stage.

        ``quiet`` is set for a rolled-up repeat that did NOT raise the stage's high-water mark:
        re-logging a line and re-raising the same warning on every one of a few hundred loop
        iterations is the flood roll-up exists to prevent.  A repeat that sets a new maximum is
        the interesting one, so it emits and refreshes the stage's dump.
        """
        if quiet:
            return
        if self.level >= Level.WARN:
            self._warn(snap)
        if self.level >= Level.INFO:
            logger.info("%s", self._line(snap))
        if self.level >= Level.DEBUG and snap.m.detail:
            logger.debug("%s", _indent(snap.m.detail, "      "))
        if self.level >= Level.DUMP:
            self._dump(snap, obj, detail)

    def _line(self, snap: Snapshot) -> str:
        m, prev = snap.m, self._by_seq(snap.parent)
        pad = "  " * snap.depth
        deg = "—" if m.degree is None else (f"{m.degree:.0f}" if m.degree != float("inf") else "inf")
        bits = [
            f"[{self.name}] {pad}{snap.seq:02d} {snap.stage:<22}",
            f"mass={m.mass:>9d}{_fmt_delta(m.mass, prev.m.mass if prev else None)}",
            f"cells={m.symbolic_cells}/{m.n_output_cells}",
            f"deg={deg}",
        ]
        if snap.count > 1:
            bits.append(f"n={snap.count}")
        if m.n_defer:
            bits.append(f"defer={m.n_defer}")
        if snap.elapsed_s:
            bits.append(f"{snap.elapsed_s:.2f}s")
        if m.error:
            bits.append(f"!{m.error}")
        for k, v in snap.ctx.items():
            bits.append(f"{k}={v}")
        return "  ".join(bits)

    def _warn(self, snap: Snapshot) -> None:
        """Proactive warnings: the absolute ceilings, and the stage-over-stage jump."""
        m = snap.m
        reasons: list[str] = []
        if m.out_mass > _int_env(_ENV_MASS, _MASS_CEILING):
            reasons.append(f"output mass {m.out_mass} > {_int_env(_ENV_MASS, _MASS_CEILING)}")
        if m.operand_mass > _int_env(_ENV_OPERAND, _OPERAND_CEILING):
            reasons.append(
                f"operand mass {m.operand_mass} > {_int_env(_ENV_OPERAND, _OPERAND_CEILING)}")
        if m.symbolic_cells > _int_env(_ENV_CELLS, _CELLS_CEILING):
            reasons.append(
                f"{m.symbolic_cells} symbolic cells > {_int_env(_ENV_CELLS, _CELLS_CEILING)} "
                "(eager per-cell scatter?)")
        # A FINITE degree above the ceiling is a real signal.  An INFINITE one is not: the degree
        # walk returns inf for any rational/algebraic op on a seed, and a non-affine value kernel
        # legitimately contains both (the measure's sqrt, the 1/det of an inverse pullback).  So
        # `inf` here means "not a polynomial in the inputs", which is the normal state of affairs
        # downstream of geometry — warning on it would fire on every curved compile and train the
        # reader to ignore the warnings.  It is reported in the table instead
        # (:meth:`rational_stage` names where it first happened, which IS the interesting part).
        if (m.degree is not None and m.degree != _INF
                and m.degree > _float_env(_ENV_DEGREE, _DEGREE_CEILING)):
            reasons.append(f"degree {m.degree} > {_float_env(_ENV_DEGREE, _DEGREE_CEILING)}")
        prev = self._by_seq(snap.parent)
        if (prev is not None and m.mass >= _JUMP_FLOOR and prev.m.mass > 0
                and m.mass / prev.m.mass >= _JUMP_FACTOR):
            reasons.append(
                f"mass jumped ×{m.mass / prev.m.mass:.1f} over {prev.stage!r} "
                f"({prev.m.mass} → {m.mass})")
        if not reasons:
            return
        warnings.warn(
            f"[{self.name}] stage {snap.stage!r}: " + "; ".join(reasons)
            + " — a stage this size is the fingerprint of an eager materialization or a"
              " symbolic pullback that should have stayed numeric/deferred."
            + f" Re-run with {_ENV_LEVEL}=dump for the staged breakdown.",
            stacklevel=2)

    def _dump(self, snap: Snapshot, obj: Observable = None,
              detail: Callable[[], str] | None = None) -> None:
        """Write a stage's numbers, caller description and rendered IR into its dump directory."""
        root = self.dump_root
        if root is None:
            return
        try:
            key = (snap.depth, snap.stage)
            d = self._dirs.get(key)
            if d is None:
                d = root / snap.slug
                self._dirs[key] = d
            d.mkdir(parents=True, exist_ok=True)
            key = (snap.depth, snap.stage)
            # The caller's own description of what this stage produced.  `_dump` runs only on an
            # occurrence that RAISED the stage's high-water mark, so re-rendering here keeps the
            # description in step with the numbers in `stage.txt` — which describe that same
            # largest occurrence.  Caching the FIRST render instead would pair a description of
            # occurrence 1 with the measurements of occurrence 900, which is worse than useless.
            # Cost is bounded by the number of new maxima (O(log n)), not by the occurrence count;
            # `write_report`'s closing re-dump passes no thunk and reuses what is stored.
            if detail is not None:
                self._detail_text[key] = _safe_call(detail, "detail")
            text = self._detail_text.get(key)
            if text:
                (d / "detail.txt").write_text(text, encoding="utf-8")
            (d / "stage.txt").write_text(self._stage_text(snap), encoding="utf-8")
            self._dump_program(d, obj, key)
        except Exception as exc:                              # a dump must not break a compile
            logger.warning("observe: dump of stage %r failed: %r", snap.stage, exc)

    def _dump_program(self, d: Path, obj: Observable, key: tuple[int, str]) -> None:
        """Write the stage's polyarray IR as readable source — "what is the output in polyarray".

        Uses :func:`polyarray.numpy_source.to_numpy_source`, the renderer polyarray already owns,
        so a stage directory shows the ACTUAL program, not just its size.  Front-end ops
        render through their own hooks; anything the renderer cannot express is written as
        a note rather than raising, because a dump must never break a compile.

        A staged :class:`~polyarray.ir.SymArray` is rendered as the program's RESULT
        (:func:`_program_with_result`), so the emitted function ends in a ``return`` of the value
        the stage is about rather than the ``return None`` an undeclared array gives.  Should that
        enriched render fail, the plain program is rendered instead — a dump degrades, it does not
        vanish.

        Rendered once per stage — the first occurrence — for the same reason as ``detail``.
        """
        if obj is None or key in self._program_text:
            return
        # A stage may hand over a SEQUENCE of artefacts (savo's per-match integrands, a jet's
        # per-order arrays). Render the first one that carries a program and say how many there
        # were, rather than silently writing nothing.
        note = ""
        target = obj
        if isinstance(obj, (list, tuple)):
            with_prog = [o for o in obj if _program_of(o) is not None]
            if not with_prog:
                return
            target = with_prog[0]
            if len(with_prog) > 1:
                note = (f"# this stage produced {len(with_prog)} programs; "
                        f"the FIRST is rendered below\n")
        program = _program_of(target)
        if program is None:
            return
        self._program_text[key] = True
        from . import numpy_source

        with_result, result_note = _program_with_result(program, target)
        src = None
        if with_result is not program:
            try:
                src = numpy_source.to_numpy_source(with_result)
                note += result_note
            except Exception as exc:
                note += (f"# rendering the staged value as the result failed "
                         f"({type(exc).__name__}: {exc}); the program alone is below\n")
        else:
            note += result_note
        if src is None:
            try:
                src = numpy_source.to_numpy_source(program)
            except Exception as exc:
                src = (f"# to_numpy_source failed: {type(exc).__name__}: {exc}\n"
                       f"# (the program is still summarised in stage.txt)\n")
        (d / "program.py").write_text(note + src, encoding="utf-8")

    def _stage_text(self, snap: Snapshot) -> str:
        m = snap.m
        ran = (f"{snap.count}× (the numbers below are the LARGEST occurrence)"
               if snap.count > 1 else "once")
        lines = [
            f"stage   {snap.seq:02d} {snap.stage}",
            f"trace   {self.name}",
            f"kind    {m.kind}",
            f"ran     {ran}",
            f"elapsed {snap.elapsed_s:.3f}s cumulative",
            f"shape   {m.shape}",
            "",
        ]
        if snap.sampled:
            lines.append(f"measured       {snap.n_measured} of {snap.count} occurrences "
                         "(geometric sampling; raise to FEM_OBSERVE=debug to measure all)")
        lines += [
            f"out_mass       {m.out_mass}",
            f"operand_mass   {m.operand_mass}",
            f"cells          {m.symbolic_cells} symbolic / {m.n_output_cells} total",
            f"operand_cells  {m.n_operand_cells}",
            f"max_cell       {m.max_cell}",
            f"max_operand    {m.max_operand}",
            f"degree         {m.degree}",
            f"programs       {m.n_programs}  stmts={m.n_stmts}  defer={m.n_defer}",
            f"prov_kinds     {dict(m.prov_kinds)}",
        ]
        if snap.ctx:
            lines += ["", "context:"] + [f"  {k} = {v!r}" for k, v in snap.ctx.items()]
        if m.error:
            lines += ["", f"probe error: {m.error}"]
        if m.detail:
            lines += ["", "IRReport:", m.detail]
        return "\n".join(lines) + "\n"

    # -- the report --------------------------------------------------------

    def report(self) -> str:
        """Render the staged delta waterfall, the table read to find the culprit stage."""
        head = (f"{'':2} {'stage':<26} {'n':>5} {'meas':>5} {'mass':>10} {'Δ':>12} "
                f"{'cells':>12} {'deg':>5} {'time':>7}")
        lines = [f"compile trace {self.name!r} — {len(self.snapshots)} stage(s), "
                 f"{time.perf_counter() - self._t0:.2f}s total",
                 "n = times the stage ran; meas = how many were measured (sampled 1,2,4,8,… "
                 "below `debug`)",
                 "mass = the largest MEASURED occurrence; time = cumulative over all n",
                 head, "-" * len(head)]
        for s in self.snapshots:
            m, prev = s.m, self._by_seq(s.parent)
            deg = "—" if m.degree is None else ("inf" if m.degree == _INF else f"{m.degree:.0f}")
            lines.append(
                f"{s.seq:02d} {'  ' * s.depth}{s.stage:<{max(0, 26 - 2 * s.depth)}} "
                f"{s.count:>5d} {s.n_measured:>5d} "
                f"{m.mass:>10d} {_fmt_delta(m.mass, prev.m.mass if prev else None):>12} "
                f"{f'{m.symbolic_cells}/{m.n_output_cells}':>12} {deg:>5} "
                f"{s.elapsed_s:>6.2f}s")
        peak = self.peak_stage()
        if peak is not None:
            lines.append("")
            lines.append(f"peak: stage {peak.seq:02d} {peak.stage!r} at mass {peak.m.mass}")
        jump = self.jump_stage()
        if jump is not None:
            prev = self._by_seq(jump.parent)
            lines.append(
                f"largest jump: stage {jump.seq:02d} {jump.stage!r} "
                f"×{jump.m.mass / prev.m.mass:.1f} over {prev.stage!r}")
        if self._off_paths:
            lines.append("")
            lines.append("⚠ OFF-PATH — this trace does NOT account for:")
            for nm, why in self._off_paths.items():
                lines.append(f"    {nm}" + (f" — {why}" if why else ""))
        rat = self.rational_stage()
        if rat is not None:
            lines.append(
                f"went rational: stage {rat.seq:02d} {rat.stage!r} "
                "(degree inf — no longer polynomial in its inputs)")
        return "\n".join(lines)

    def peak_stage(self) -> Snapshot | None:
        """Return the stage with the largest mass — where the IR is biggest."""
        return max(self.snapshots, key=lambda s: s.m.mass, default=None)

    def rational_stage(self) -> Snapshot | None:
        """Return the first stage whose degree went infinite — where the IR stopped being polynomial.

        This is the epic's stage-4/stage-6 signature (the pullback ξ going rational, the Faà di
        Bruno transport).  It is *reported*, never warned about: downstream of a curved geometry
        it is the expected state, and only its POSITION is informative — a value kernel going
        rational is routine, a reference basis jet going rational is a bug.
        """
        prev_finite = True
        for s in self.snapshots:
            if s.m.degree is None:
                continue
            if s.m.degree == _INF and prev_finite:
                return s
            prev_finite = s.m.degree != _INF
        return None

    def jump_stage(self) -> Snapshot | None:
        """Return the stage with the largest mass multiplier over its parent — where it *got* big.

        This, not the peak, is usually the culprit: the peak stage often just carries forward
        what an earlier stage already inflated.
        """
        best: Snapshot | None = None
        best_ratio = 1.0
        for s in self.snapshots:
            prev = self._by_seq(s.parent)
            if prev is None or prev.m.mass <= 0 or s.m.mass < _JUMP_FLOOR:
                continue
            ratio = s.m.mass / prev.m.mass
            if ratio > best_ratio:
                best, best_ratio = s, ratio
        return best

    def write_report(self) -> Path | None:
        """Write :meth:`report` to ``<dump_root>/report.txt``; ``None`` below ``DUMP``.

        Also REWRITES every stage's ``stage.txt``.  A stage that ran in a loop was dumped on its
        first occurrence, when its roll-up counters still read ``1``; only here are the final
        counts known, so the on-disk snapshot must be refreshed or it lies about how many times
        the stage ran and how long it took in total.
        """
        root = self.dump_root
        if root is None:
            return None
        root.mkdir(parents=True, exist_ok=True)
        for snap in self.snapshots:
            self._dump(snap)
        p = root / "report.txt"
        p.write_text(self.report() + "\n", encoding="utf-8")
        return p


def describe_value(x: object, *, _depth: int = 0) -> str:
    """Render a one-line identity for a value, for ``detail`` text.

    A bare ``repr`` is useless in a dump: a field wrapper prints its whole numeric matrix, so
    two of them bury the rest of the description under forty lines of numbers. This names the
    wrapper class, the value's ``name``, the *shapes* of any arrays but never their contents,
    a few identifying scalars, and recurses into a wrapped ``base`` — usually the question
    being asked, namely which basis an argument really is.

    This is generic dump plumbing, so it lives here rather than in the consumers, which are
    forbidden to name a class in text at all.
    """
    import numpy as np

    if x is None:
        return "None"
    if _depth > 4:
        return "…"
    kind = type(x).__name__
    bits: list[str] = []
    name = getattr(x, "name", None)
    if isinstance(name, str):
        bits.append(f"name={name!r}")
    for attr in ("K", "coeffs", "V"):
        v = getattr(x, attr, None)
        if isinstance(v, np.ndarray):
            bits.append(f"{attr}={v.shape}")               # the SHAPE, never the contents
    for attr in ("element", "degree", "order"):
        v = getattr(x, attr, None)
        if v is not None and not isinstance(v, np.ndarray) and not callable(v):
            # BOUNDED. A finite element's repr carries its whole Σ (every DOF functional's PTEM),
            # which runs to thousands of characters and buries the rest of the line — the same
            # failure as dumping a numeric matrix. Identify it, don't transcribe it.
            bits.append(f"{attr}={_short(v)}")
    base = getattr(x, "base", None)
    if base is not None:
        bits.append(f"base={describe_value(base, _depth=_depth + 1)}")
    return f"{kind}({', '.join(bits)})"


def _short(v: object, limit: int = 72) -> str:
    """Render an identifying attribute on one bounded line.

    Falls back to ``ClassName(name=…)`` rather than a truncated repr when the repr is long: a
    clipped repr of a finite element is neither readable nor informative, whereas its class and
    name are exactly what the reader needs.
    """
    text = str(v)
    if len(text) <= limit:
        return text
    name = getattr(v, "name", None)
    if name is not None:
        return f"{type(v).__name__}(name={name!r})"
    return f"{type(v).__name__}(…)"


def _safe_call(thunk: Callable[[], str], what: str) -> str:
    """Call a caller-supplied description thunk; never let it break the compile."""
    try:
        return thunk()
    except Exception as exc:
        return f"# {what} callable raised: {type(exc).__name__}: {exc}\n"


def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in s)


def _indent(text: str, pad: str) -> str:
    return "\n".join(pad + ln for ln in text.splitlines())


# ---------------------------------------------------------------------------
# The ambient trace
# ---------------------------------------------------------------------------


class _NullTrace(CompileTrace):
    """The trace instrumented code sees when nobody is observing.

    Not a separate protocol — a :class:`CompileTrace` pinned to :attr:`Level.OFF`, so every
    ``stage`` / ``phase`` / ``dump_dir_for`` call takes its early-out and the call sites need no
    ``if trace is not None`` guard anywhere.
    """

    def __init__(self) -> None:
        super().__init__("null", level=Level.OFF)


_NULL = _NullTrace()
_CURRENT: ContextVar[CompileTrace] = ContextVar("fem_observe_trace", default=_NULL)


def trace() -> CompileTrace:
    """Return the active trace, or the null trace when nothing is being observed.

    Never returns ``None``, so instrumentation reads
    ``observe.trace().stage("represent", program)`` with no guard.
    """
    return _CURRENT.get()


def active() -> bool:
    """Whether anything is being recorded — for skipping *expensive* context gathering.

    The ``stage`` call itself is already free when off; use this only to guard work you would
    otherwise do purely to hand to the tracer.
    """
    return _CURRENT.get().level > Level.OFF


@contextmanager
def observe_compile(name: str, *, level: Level | None = None,
                    dump_root: Path | str | None = None,
                    report: bool = True) -> Iterator[CompileTrace]:
    """Observe everything instrumented inside the block as one trace.

    ``level`` overrides ``FEM_OBSERVE`` for this block — a test or a script can force
    ``Level.DUMP`` without touching the environment.  ``report=True`` logs the delta waterfall on
    exit (and writes ``report.txt`` at ``DUMP``), including when the block raises: a compile that
    died is the one whose staged trace you most want.
    """
    root = Path(dump_root) if dump_root is not None else None
    tr = CompileTrace(name, level=level, dump_root=root)
    token = _CURRENT.set(tr)
    try:
        yield tr
    finally:
        _CURRENT.reset(token)
        if report and tr.level >= Level.INFO and tr.snapshots:
            logger.info("%s", tr.report())
        if tr.level >= Level.DUMP:
            tr.write_report()


def off_path(name: str, why: str = "") -> None:
    """Mark that execution entered a path the trace does NOT cover — a blind spot, loudly.

    The instrumented boundaries describe *one* route through the stack, and the failure this
    guards against is a quantity taking a different route in a different consumer. A trace that
    silently omits the route actually taken is worse than no trace, because it reads like a
    complete account of the compile.

    So an uninstrumented or non-canonical path calls this. When a trace is ACTIVE it records an
    ``off-path/<name>`` stage and warns ONCE per path; when nobody is observing it is silent, so
    ordinary consumers of these APIs see nothing. That asymmetry is the point — the warning means
    *"your trace has a hole here"*, which is only meaningful to someone holding a trace.

    Fires once per distinct ``name`` per trace: it flags a route taken, not a count.
    """
    tr = _CURRENT.get()
    if tr.level <= Level.OFF:
        return
    tr.note_off_path(name, why)


@contextmanager
def scope(name: str, **kw: object) -> Iterator[CompileTrace]:
    """Open a trace for this block **unless one is already open**, in which case reuse it.

    This is what a library entry point uses (a top-level assembly or compile routine):
    calling it makes the entry point observable on its own, while a caller
    who wrapped a larger region in :func:`observe_compile` still gets ONE trace spanning
    everything rather than an inner trace that shadows theirs and reports half the compile.
    """
    if active():
        yield _CURRENT.get()
        return
    with observe_compile(name, **kw) as tr:
        yield tr


def stage(name: str, obj: Observable = None, **kw: object) -> Snapshot | None:
    """Record a stage on the ambient trace — the one-liner form of ``trace().stage(...)``."""
    return _CURRENT.get().stage(name, obj, **kw)


def phase(name: str, **kw: object) -> AbstractContextManager[list[Observable]]:
    """Time a block on the ambient trace — the one-liner form of ``trace().phase(...)``."""
    return _CURRENT.get().phase(name, **kw)


def dump_dir_for(name: str) -> Path | None:
    """Return the ambient trace's dump directory for ``name``, ready for ``compile_torch``."""
    return _CURRENT.get().dump_dir_for(name)


def dump_root() -> Path | None:
    """Return the ambient trace's dump directory, the parent of every stage directory.

    Returns
    -------
    Path or None
        The root, or ``None`` below :attr:`Level.DUMP`.

    Notes
    -----
    For artefacts that belong to the RUN rather than to one stage. A process-wide compiler cache is
    the case this exists for: one cache serves every stage, so writing it inside a stage directory
    would attribute all of it to whichever stage happened to compile first, and the stages that
    followed would look as though they generated nothing.
    """
    return _CURRENT.get().dump_root


__all__ = [
    "CompileTrace",
    "Level",
    "Measurement",
    "Snapshot",
    "active",
    "describe_value",
    "dump_dir_for",
    "dump_root",
    "level_from_env",
    "measure",
    "off_path",
    "observe_compile",
    "phase",
    "scope",
    "stage",
    "trace",
]


_ensure_logging(level_from_env())
