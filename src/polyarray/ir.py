"""Imperative IR: descriptors, registry, ``SymArray``, ``Program``, ``Stmt``, ``Ref``.

This module is the single home for everything program-scoped in the
symbolic interpreter:

Metadata / registry
    :class:`Provenance`, :class:`SymInput`, :class:`SymbolEnv`,
    :func:`allocate_input`, :func:`cells_sparsity`.  Value-free
    descriptors plus the global generator-name table.

Live values
    :class:`SymArray` — wraps an ``ndarray`` of
    ``RationalFunction | float`` cells plus a back-reference to its
    owning :class:`Program`.  Quacks like a numpy ndarray for read
    access; knows how to extend its program by emitting a :class:`Stmt`
    (slice B).  Replaces the former ``SymArray`` / ``Output`` /
    ``BoundOutput`` triple — there is exactly one type now.

IR structure
    :class:`InputRef`, :class:`OutputRef`, :class:`Const`,
    :class:`RationalRef`, :class:`Ref`, :class:`Stmt`,
    :class:`Program`.  The :meth:`Program.run` plain-Python evaluator
    plus :meth:`Program.copy` for sub-interpreter spawning.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import (Any, Literal, Protocol, TypeAlias, TypeGuard, Union, get_args,
                    runtime_checkable)

import numpy as np
import numpy.typing as npt

from .int_atom import IntAtom
from .poly_backend import Poly
from .rational import RationalFunction, simple_zero

Cell = Union[RationalFunction, float, int]

@runtime_checkable
class VmapClosure(Protocol):
    """A ``vmap`` closure, as the metadata :func:`vmap` stamps on it describes.

    The attributes are additive and purely descriptive — evaluation never reads them — so a
    tool can recover the batched body and the normalised axis tuples without spelunking the
    closure cells. A front end may stamp the same attributes on a wrapper of its own; naming
    the shape here is what lets a consumer narrow to it instead of probing with ``getattr``.
    """

    #: The per-element body the closure batches.
    _vmap_body: Program
    #: One entry per operand: the axis to map over, or ``None`` to broadcast it.
    _in_axes: tuple[int | None, ...]
    #: One entry per output: the axis the results stack on.
    _out_axes: tuple[int | None, ...]


@runtime_checkable
class NestedVmapClosure(Protocol):
    """A multi-variable nested ``vmap`` closure, batched once per bound variable.

    The first :attr:`_nested_n_vars` operands each get their own vmap level; the remaining
    :attr:`_nested_n_free` broadcast.
    """

    #: How many leading operands are batched, one nesting level each.
    _nested_n_vars: int
    #: How many trailing operands broadcast unbatched.
    _nested_n_free: int


#: A :class:`DimAtom`'s hashable identity: the tuple that names where the runtime
#: dimension comes from.  Opaque by design — only ever compared and used as a dict key.
type DimSource = tuple[object, ...]

#: Anything numpy accepts as an index: an int, a slice, an index array, or a tuple
#: mixing them.
type Index = int | slice | np.ndarray | Sequence[int] | tuple[int | slice | np.ndarray | Sequence[int] | None, ...]


# ---------------------------------------------------------------------------
# Ambient budget override
# ---------------------------------------------------------------------------
# A `Program` built with no explicit `budget=` consults this context. It lets a
# caller (e.g. oracle's sparsity mask) request the non-deferred, inline-covariant
# symbolic lane for the SHARED sampling program — so the covariant/φ-jet chain is
# built inline over the vertices (not deferred to `out_cov`/`phi_jet` Stmts) — without
# threading a budget through every builder (pointwise never references it).
_BUDGET_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar(
    "_pa_budget_override", default=None
)


@contextlib.contextmanager
def budget_override(budget: SymbolicBudget | None) -> Iterator[None]:
    """Within this context, budget-less `Program`s use `budget` (None restores default)."""
    tok = _BUDGET_OVERRIDE.set(budget)
    try:
        yield
    finally:
        _BUDGET_OVERRIDE.reset(tok)


def current_budget_override() -> SymbolicBudget | None:
    """Return the ambient :func:`budget_override` budget, or ``None``.

    For builders that construct a :class:`Program` with an explicit budget and want to honour
    the override themselves.
    """
    return _BUDGET_OVERRIDE.get()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

ProvenanceKind = Literal[
    "vertex", "point", "coeff", "stmt_out", "param_dof", "per_point"
]


@dataclass(frozen=True)
class Provenance:
    """Where a generator came from.

    ``kind`` is one of ``"vertex"``, ``"point"``, ``"coeff"``,
    ``"stmt_out"``.  ``origin`` identifies the source object
    (a :class:`Geometry`, a basis vector, a :class:`Stmt`, …).
    ``index`` is the position inside that origin (vertex j, query
    point row, coefficient n, output cell index, …).  ``label`` is
    the human-readable name attached to the generator.
    """

    kind: ProvenanceKind
    origin: Any
    index: tuple[int, ...]
    label: str


# ---------------------------------------------------------------------------
# SymbolEnv
# ---------------------------------------------------------------------------

class SymbolEnv:
    """Global identity table for symbolic generators.

    Allocates fresh generator names with attached :class:`Provenance`
    metadata.  Two rational functions with the same generator name
    refer to the same algebraic object regardless of which ring they
    were originally constructed in — :class:`SymbolEnv` is what
    enforces that invariant.

    The class is *not* responsible for owning a single shared
    :class:`PolyRing`; rings are per-:class:`RationalFunction` and
    rebuilt on demand.
    """

    def __init__(self) -> None:
        self._provenance: dict[str, Provenance] = {}
        self._counter: dict[str, int] = {}

    # -- allocation -----------------------------------------------------

    def fresh(self, label: str, provenance: Provenance) -> str:
        """Allocate a fresh generator with the given label and provenance."""
        name = label
        if name in self._provenance:
            i = self._counter.get(label, 1) + 1
            while f"{label}__{i}" in self._provenance:
                i += 1
            name = f"{label}__{i}"
            self._counter[label] = i
        self._provenance[name] = provenance
        return name

    def declare(self, name: str, provenance: Provenance) -> None:
        """Record an externally-chosen name with its provenance.

        Raises if ``name`` is already declared (we do not silently
        rebind — a clash means the caller has a bug in its naming).
        """
        if name in self._provenance:
            raise ValueError(f"generator {name!r} already declared")
        self._provenance[name] = provenance

    # -- inspection -----------------------------------------------------

    def of(self, name: str) -> Provenance:
        """Return the :class:`Provenance` for the named generator."""
        try:
            return self._provenance[name]
        except KeyError as exc:
            raise KeyError(f"unknown generator {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        """Return all generator names ever issued, in insertion order."""
        return tuple(self._provenance.keys())

    def __contains__(self, name: str) -> bool:
        return isinstance(name, str) and name in self._provenance

    def __len__(self) -> int:
        return len(self._provenance)

    # -- copy -----------------------------------------------------------

    def copy(self) -> SymbolEnv:
        """Return an independent copy of the env (for sub-program forks)."""
        new = SymbolEnv()
        new._provenance = dict(self._provenance)
        new._counter = dict(self._counter)
        return new


# ---------------------------------------------------------------------------
# SymInput + allocator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymInput:
    """A flat, atom-per-cell descriptor for a named symbolic input.

    See plan §1.4: a *static* :class:`SymInput` allocates *one atom per
    cell* of its declared shape — there is no granularity knob.  The
    ``provenance`` field is either a fixed :class:`Provenance` (used
    for every cell, with the cell's own index appended) or a callable
    mapping a cell index to its :class:`Provenance` for fully custom
    labelling.

    **Dynamic inputs (Stage C).** A ``shape`` entry may be a
    :class:`DimAtom` (an axis whose size is known only at run time — e.g. an
    FFS-typed Grassmann input ``a : Λᵏ`` whose dimension ``δ`` is a runtime
    rank).  Such an input is *dynamic* (:func:`is_dynamic`); it does **not**
    enumerate per-cell atoms but is allocated as a single **bulk**
    :class:`SymArray` whose axis sizes are read from the provided array at
    :meth:`Program.run` time and bound into ``dim_bindings``.  Static inputs
    are unchanged (byte-identical).
    """

    name: str
    shape: tuple[int | DimAtom, ...]
    provenance: Provenance | Callable[[tuple[int, ...]], Provenance]


def allocate_input(env: SymbolEnv, spec: SymInput) -> np.ndarray:
    """Allocate per-cell :class:`RationalFunction` atoms for a :class:`SymInput`.

    Returns a numpy ``object`` ndarray of shape ``spec.shape``.  Each
    atom lives in its own single-generator :class:`PolyRing` whose
    generator name is freshly allocated against ``env``.  Cross-cell
    arithmetic rebuilds joint rings on demand (see
    :func:`chartlib._symbolic.rational._join_rings`).
    """
    prov_fn: Callable[[tuple[int, ...]], Provenance]
    if isinstance(spec.provenance, Provenance):
        base_prov = spec.provenance

        def prov_fn(idx: tuple[int, ...]) -> Provenance:
            return Provenance(
                kind=base_prov.kind,
                origin=base_prov.origin,
                index=base_prov.index + idx,
                label=f"{base_prov.label}{_format_idx(idx)}",
            )
    else:
        prov_fn = spec.provenance

    flat_idx = list(np.ndindex(*spec.shape)) if spec.shape else [()]
    cells = np.empty(spec.shape, dtype=object)
    for idx in flat_idx:
        prov = prov_fn(idx)
        name = env.fresh(prov.label, prov)
        rf = RationalFunction.atom(name)
        if spec.shape:
            cells[idx] = rf
        else:
            cells[()] = rf
    return cells


def _format_idx(idx: tuple[int, ...]) -> str:
    if not idx:
        return ""
    return "_" + "_".join(str(i) for i in idx)


# ---------------------------------------------------------------------------
# Sparsity helpers (rational lane only)
# ---------------------------------------------------------------------------

def cells_sparsity(arr: np.ndarray) -> np.ndarray:
    """Return a structural-zero mask for an ndarray of cells."""
    if arr.dtype.kind == "f":
        return arr == 0.0
    out = np.zeros(arr.shape, dtype=bool)
    for idx in np.ndindex(*arr.shape):
        out[idx] = simple_zero(arr[idx])
    return out


# ---------------------------------------------------------------------------
# SymbolicBudget — knobs for the rational lane / closed-form-vs-Stmt dispatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolicBudget:
    """Tunable knobs that drive lane choice for the symbolic interpreter.

    ``naive_inverse_max_size`` and ``inverse_max_degree`` are static
    structural bounds (matrix size, total-degree estimate) above which
    ``det`` / ``inverse`` route from the closed-form rational lane (the
    naive cofactor / Schur expansion) into a :class:`Stmt`-emitting
    fallback (slice B).

    Plan B step 4 adds the **deferral gates** below.  Their *defaults
    reproduce the legacy symbolic path exactly* — i.e. ``SymbolicBudget()``
    (== :meth:`legacy`) is byte-identical to the pre-Plan-B behaviour, and
    that path is a *preserved, first-class budget choice*, not an accident of
    defaults.  :meth:`build_big_symbols` is the opposite end (budget zero):
    every modeled op stays symbolic so a sampler produces big-symbol IR over
    its vertex / DoF inputs instead of collapsing to numeric Stmts.

    Deferral gates (all default to "defer", == legacy):

    * ``einsum_bag_threshold`` — per-output-cell monomial bound above which a
      multi-operand symbolic einsum offloads to a numeric Stmt.  ``None`` means
      "use the module / ``CHARTLIB_EINSUM_BAG_THRESHOLD`` default" (64) — the
      legacy, env-aware behaviour.  Budget-zero sets it huge (never offload).
    * ``defer_phi_jet`` — when True (legacy) the single-operand φ-jet einsum
      (``runtime_einsum``) offloads its object-dtype LHS to a numeric Stmt.
      False keeps the contraction symbolic, so the **parameterization built
      from the vertex / DoF atoms comes through** (the step-4 retention target).
    * ``defer_covariant`` — when True (legacy) the order-≥1 covariant chain
      (``covariant.defer_covariant``) becomes one numeric Stmt.  False keeps it
      symbolic — heavier (the 3-D high-order blow-up Plan A measured), so it is
      opt-in even at budget zero (``build_big_symbols(retain_covariant=...)``).
    * ``freeze`` — when True (legacy) :func:`freeze_array` caps cell complexity
      by capturing big cells as fresh atoms.  False disables it (retain).
    """

    naive_inverse_max_size: int = 6
    inverse_max_degree: int | None = None
    cancel_num_dom: bool = False
    iszero_tol: float = 0.0
    # Retention knob: when True, the covariant frame conversion (orthonormal /
    # euclidean) is applied OUTSIDE the deferred numeric Stmt, so Q / R / R⁻¹
    # come through as visible QrSignFixOp atoms (the frame conversion is the
    # last step).  Default False keeps the WS3 full-deferral (cheap; the whole
    # covariant chain incl. frame is one numeric Stmt).  Surfacing the frame
    # reintroduces order-≥2 symbolic cost on 3-D cells, so it is opt-in — the
    # knob Plan B's budget-zero "build big symbols" mode flips on.
    surface_frame: bool = False
    # Deferral gates (Plan B step 4) — defaults == legacy symbolic path.
    einsum_bag_threshold: int | None = None
    defer_phi_jet: bool = True
    defer_covariant: bool = True
    freeze: bool = True
    # Plan-D #4 / task #11 — the "middle ground": when surfacing the frame
    # (``surface_frame``), defer the per-slot R⁻¹/Q contraction to numeric
    # TensordotOp/MoveaxisOp Stmts instead of running it inline as symbolic RF
    # arithmetic.  Keeps Q/R/R⁻¹ visible while making the surfaced frame usable
    # on 3-D high-order cells (where the inline contraction blows up).  Default
    # False = today's inline rational lane (byte-identical).
    defer_frame_contraction: bool = False
    # Downstream block-inverse deferral (oracle's Schur `symbolic_inverse`): the matrix size at/above
    # which a base-case inverse / a Schur-combine matrix product offloads to a NUMERIC Stmt instead of
    # inline RationalFunction arithmetic (the degree blow-up on a single connected
    # component). ``None`` ⇒ the consumer's own module defaults; ``build_big_symbols`` raises them (stay
    # symbolic / never defer), ``force_stmts`` lowers them (always defer). Read via
    # ``current_budget_override``; other budget fields left at legacy keep sampling byte-identical, so a
    # caller can tune ONLY the inverse without perturbing the sampling lane.
    schur_inverse_stmt_size: int | None = None
    schur_matmul_stmt_size: int | None = None

    @classmethod
    def legacy(cls) -> SymbolicBudget:
        """Return the fully-deferring budget, identical to a bare ``SymbolicBudget()``.

        Named so a caller can select it explicitly and on the record rather than relying on
        the default. Every deferral gate defers.
        """
        return cls()

    @classmethod
    def build_big_symbols(
        cls,
        *,
        retain_covariant: bool = False,
        surface_frame: bool | None = None,
        **overrides: int | bool | None,
    ) -> SymbolicBudget:
        """Budget-zero: retain the parameterization (chart φ) built from vertices.

        Turns off the φ-jet single-operand offload, raises the multi-operand
        einsum threshold so contractions stay symbolic, and disables
        :func:`freeze_array` — so the parameterization comes through as big
        symbols over the vertex / DoF inputs.

        ``retain_covariant`` (default False) additionally keeps the order-≥1
        covariant chain symbolic.  It is **not** required merely to see the
        parameterization, and it is prone to the 3-D high-order blow-up Plan A
        measured, so it is opt-in.  ``surface_frame`` defaults to track
        ``retain_covariant`` (surface the frame only when the covariant chain is
        retained).  Extra ``overrides`` are forwarded to the constructor.
        """
        if surface_frame is None:
            surface_frame = retain_covariant
        return cls(
            einsum_bag_threshold=1 << 62,
            defer_phi_jet=False,
            defer_covariant=not retain_covariant,
            freeze=False,
            surface_frame=surface_frame,
            schur_inverse_stmt_size=1 << 62,
            schur_matmul_stmt_size=1 << 62,
            **overrides,
        )

    @classmethod
    def force_stmts(cls, **overrides: int | bool | None) -> SymbolicBudget:
        """Return the no-symbolic-budget preset, driving every modeled op to a statement.

        The opposite of :meth:`build_big_symbols`: rather than retaining closed-form rational
        structure, this forces every modeled op to
        emit an imperative :class:`Stmt` (no closed-form rational lane), so
        Grassman's symbolic inputs flow through as deferred Stmts that are
        simplified afterward (the build-then-simplify novelty).

        * ``naive_inverse_max_size=0`` / ``inverse_max_degree=0`` — every
          ``det`` / ``inverse`` is over-budget ⇒ emits a Stmt.
        * ``einsum_bag_threshold=1`` — every multi-operand symbolic einsum
          offloads to a numeric Stmt.
        * ``freeze=True`` — cap any residual cell to an atom.

        Extra ``overrides`` are forwarded to the constructor.
        """
        return cls(
            naive_inverse_max_size=0,
            inverse_max_degree=0,
            einsum_bag_threshold=1,
            freeze=True,
            schur_inverse_stmt_size=0,
            schur_matmul_stmt_size=0,
            **overrides,
        )


# ---------------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputRef:
    """Reference into a named program input.

    ``indices`` is a tuple of ``int`` for full-cell indexing; an empty
    tuple means "the whole input array."
    """

    name: str
    indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class OutputRef:
    """Reference into a prior :class:`Stmt`'s output.

    ``stmt_idx`` is the index of the producing Stmt within the parent
    program; ``out_idx`` is the index of the :class:`SymArray` within
    that Stmt; ``indices`` is the cell index inside the output array.
    """

    stmt_idx: int
    out_idx: int
    indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class Const:
    """A frozen numeric literal — scalar or ndarray."""

    value: float | np.ndarray


@dataclass(frozen=True)
class IntAtomRef:
    """Reference to a named :class:`IntAtom` registered on the program.

    Resolves at run time to the bound integer value from the
    ``values`` dict passed to :meth:`Program.run` (or the bindings
    dict on a standalone :meth:`SymArray.evaluate`).
    """

    name: str


@dataclass(frozen=True)
class RationalRef:
    """A :class:`RationalFunction` whose generators are program atoms.

    Used to splice rational expressions over inputs / prior stmt-out
    atoms into a Stmt's input list — slice B uses these heavily.
    """

    rf: RationalFunction


@dataclass(frozen=True)
class BulkOut:
    """Handle for a :class:`Stmt` output kept as one whole numeric tensor.

    A *bulk* output is NOT scattered into per-cell atom RationalFunctions:
    at :meth:`Program.run` time the producing Stmt's numeric result is
    bound under a single name, and the owning :class:`SymArray` evaluates
    to that whole tensor directly.  This avoids minting one RF + one ring
    per cell for tensors that are only consumed in bulk (terminal outputs,
    or anything not dissected per-cell).

    A consumer that needs per-cell symbolic structure (an einsum / rational
    arithmetic) calls :func:`unpack`, which emits an unpack Stmt converting
    the bulk tensor into the usual per-cell atom array.

    ``name`` is the run-time binding key; ``shape`` is the tensor shape.
    For a *dynamic* bulk output (Stage B) ``shape`` may carry
    :class:`DimAtom` entries (the symbolic shape); the concrete axis sizes
    are resolved at :meth:`Program.run` time.
    """

    name: str
    shape: tuple[int | DimAtom, ...]


class SymArrayRef:
    """Reference to an array of cells; resolved to a float ndarray at run time.

    Wraps the ``cells`` ndarray directly (program-independent), so a
    :meth:`Program.copy` that rebinds SymArrays leaves SymArrayRefs
    untouched — the cells themselves only refer to generator names,
    which the env copy preserves.

    Use this when an imperative Stmt's input is "the current value of
    this whole SymArray" (e.g. ``inverse(M)`` where ``M`` is a matrix
    of rational expressions).  At ``Program.run`` time each cell is
    evaluated against the accumulated bindings and the resulting
    float ndarray is passed to the Stmt's ``fn``.
    """

    __slots__ = ("_cells", "_bulk")

    def __init__(self, source: SymArray | np.ndarray) -> None:
        self._bulk: BulkOut | None = None
        if isinstance(source, SymArray):
            # Use the placeholder ``_cells`` (NOT the auto-unpacking ``cells``
            # property): a bulk SymArrayRef carries ``_bulk`` and resolves to
            # the whole tensor at run time, so it must not materialise here.
            self._cells = source._cells
            self._bulk = source._bulk
        else:
            self._cells = np.asarray(source)

    @property
    def cells(self) -> np.ndarray:
        """Return the underlying cell array."""
        return self._cells

    def __repr__(self) -> str:
        if self._bulk is not None:
            return f"SymArrayRef(bulk{self._bulk.shape})"
        kind = "num" if self.cells.dtype.kind == "f" else "rat"
        return f"SymArrayRef({kind}{tuple(self.cells.shape)})"


Ref = Union[InputRef, OutputRef, Const, RationalRef, SymArrayRef, IntAtomRef]


# ---------------------------------------------------------------------------
# Stmt-output declaration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DimAtom:
    """A runtime array dimension produced by a :class:`Stmt` output or input.

    Unlike :class:`IntAtom` (a pre-declared integer-valued *input*), a
    ``DimAtom`` is an integer resolved at :meth:`Program.run` time.  It may be
    used as an entry in an :class:`OutSpec.shape` / :class:`SymInput.shape`,
    making that shape **dynamic**: the axis size is resolved at run time from
    the bound value of its source.

    * ``name`` — provenance, e.g. ``"rank:Alt"``.
    * ``source`` — a *tagged* hashable tuple identifying the run-time origin
      of the dimension and used as the ``dim_bindings`` key:

      - ``("stmt", stmt_idx, out_idx)`` — a prior Stmt output (Stage B), e.g.
        the ``rank`` (δ_f) output of an :class:`SvdOp`.
      - ``("in", input_name, axis)`` — an axis of a dynamic :class:`SymInput`
        (Stage C); resolved from the provided array's actual shape.

    Backward-compat shim: a bare 2-tuple ``(stmt_idx, out_idx)`` (Stage B's
    original form) is normalised to ``("stmt", stmt_idx, out_idx)`` so old
    callers keep working and the key still matches the binder.

    Frozen so it hashes by value (usable as a dict key and in a shape
    tuple alongside concrete ints).
    """

    name: str
    source: tuple[Any, ...]

    def __post_init__(self) -> None:
        src = self.source
        # Compat shim: normalise the Stage-B 2-tuple (stmt_idx, out_idx) to the
        # tagged ("stmt", stmt_idx, out_idx) form so it matches the binder key.
        if (
            len(src) == 2
            and isinstance(src[0], (int, np.integer))
            and isinstance(src[1], (int, np.integer))
        ):
            object.__setattr__(
                self, "source", ("stmt", int(src[0]), int(src[1]))
            )
        else:
            object.__setattr__(self, "source", tuple(src))

    @staticmethod
    def from_input(name: str, input_name: str, axis: int) -> DimAtom:
        """Construct an input-axis ``DimAtom`` (Stage C)."""
        return DimAtom(name=name, source=("in", input_name, int(axis)))


def is_dynamic(shape: tuple[int | DimAtom, ...]) -> bool:
    """Report whether any entry of ``shape`` is a :class:`DimAtom`, i.e. a runtime dimension.

    The single gate that routes a shape away from the static per-cell
    build-time path (cell-atom allocation, ``np.ndindex``, cell-size math)
    and into the dynamic bulk lane.  A fully-concrete shape is *not*
    dynamic, so static-shape programs hit none of the dynamic branches and
    remain byte-identical.
    """
    return any(isinstance(d, DimAtom) for d in shape)


def _resolve_shape(
    shape: tuple[int | DimAtom, ...],
    dim_bindings: Mapping[tuple[Any, ...], int],
) -> tuple[int, ...]:
    """Substitute each :class:`DimAtom` in ``shape`` with its bound int.

    Concrete entries pass through unchanged.  A ``DimAtom`` whose source
    output has not yet been bound raises ``KeyError`` — the producing Stmt
    must run before any consumer that uses its dimension (guaranteed by
    Stmt ordering).
    """
    out: list[int] = []
    for d in shape:
        if isinstance(d, DimAtom):
            if d.source not in dim_bindings:
                raise KeyError(
                    f"DimAtom {d.name!r}: dimension from output {d.source} "
                    f"not yet bound at run time"
                )
            out.append(int(dim_bindings[d.source]))
        else:
            out.append(int(d))
    return tuple(out)


@dataclass(frozen=True)
class OutSpec:
    """Declaration of one Stmt output: a name and a shape.

    Passed to :meth:`SymArray.emit_stmt` so that fresh atom RFs can be
    allocated for each cell of the declared shape.  The ``name`` is
    used as the provenance label prefix for the allocated atoms — it
    surfaces in repr / debug output.

    ``shape`` entries are ordinarily ``int``; an entry may also be a
    :class:`DimAtom` (a runtime dimension produced by a prior Stmt
    output), making the shape **dynamic** — see :func:`is_dynamic`.  A
    dynamic output is always emitted bulk (whole-tensor), its axis sizes
    resolved at :meth:`Program.run` time.
    """

    name: str
    shape: tuple[int | DimAtom, ...] = ()


# ---------------------------------------------------------------------------
# SymArray
# ---------------------------------------------------------------------------

class SymArray:
    """Cell ndarray attached to a :class:`Program`.

    Unifies the previous ``SymArray`` / ``Output`` / ``BoundOutput``
    triple into one type.  Cells are ``RationalFunction | float``;
    ``program`` is the owning Program (``None`` for standalone literals).
    ``name`` is set when the array is registered as a program output or
    a Stmt-bound output.

    Read protocol mirrors a numpy ndarray:

    * ``np.asarray(self)`` returns the underlying cells (via
      ``__array__``);
    * :attr:`shape`, :attr:`dtype`, :attr:`ndim`, :attr:`flat` proxy
      to the cells array;
    * ``self[idx]`` returns a SymArray (for ndarray slices) or a single
      cell (for full indexing);
    * iteration yields per-row SymArrays / cells.

    The back-reference to ``program`` is what lets operations on this
    array extend the program (slice B): array methods like
    :meth:`inverse` / :meth:`matmul` route closed-form work through
    rational arithmetic on cells and over-budget / imperative work
    through ``self.program.emit_stmt(...)``.
    """

    __slots__ = ("_cells", "program", "name", "_bulk", "_unpacked")

    # ``_cells`` is a numpy ndarray (object-dtype for symbolic backends,
    # float-dtype for numeric); typed ``Any`` so 0-d indexing yields the
    # underlying cell rather than ndarray.  For a *bulk* array it is a
    # shape-only placeholder (None-filled); read it via the ``cells``
    # property, which auto-unpacks a bulk array to per-cell atoms.
    _cells: Any
    program: Program | None
    name: str | None
    # When set, this SymArray is a *bulk* Stmt output: ``cells`` is a
    # placeholder of the right shape and eval reads the whole tensor from
    # the run-time binding ``_bulk.name`` instead of per-cell atoms.
    _bulk: BulkOut | None
    # Memoised materialisation of a bulk array (see :func:`unpack`).
    _unpacked: SymArray | None

    def __init__(
        self,
        cells: np.ndarray | Sequence[Cell] | Cell,
        program: Program | None = None,
        name: str | None = None,
    ) -> None:
        self._bulk = None
        self._unpacked = None
        if isinstance(cells, SymArray):
            self._cells = cells._cells
            self._bulk = cells._bulk
        elif isinstance(cells, np.ndarray):
            if cells.dtype.kind in ("i", "u"):
                self._cells = cells.astype(float)
            else:
                self._cells = cells
        elif isinstance(cells, RationalFunction):
            arr = np.empty((), dtype=object)
            arr[()] = cells
            self._cells = arr
        elif isinstance(cells, (int, float)):
            self._cells = np.array(float(cells), dtype=float)
        else:
            arr = np.asarray(cells)
            if arr.dtype.kind in ("i", "u"):
                arr = arr.astype(float)
            self._cells = arr
        self.program = program
        self.name = name

    # ------------------------------------------------------------------
    # numpy-like read protocol
    # ------------------------------------------------------------------

    @property
    def cells(self) -> np.ndarray:
        """Per-cell ndarray; auto-unpacks a bulk array to atom RFs.

        For a bulk array this materialises (one ``unpack`` Stmt, memoised)
        — the safety net that lets any per-cell consumer work unchanged.
        Shape/dtype/ndim below read the placeholder directly, so querying
        them does NOT force materialisation.
        """
        if self._bulk is not None:
            return unpack(self)._cells
        return self._cells

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        if dtype is None:
            return self.cells
        return self.cells.astype(dtype)

    @property
    def shape(self) -> tuple[int | DimAtom, ...]:
        # A dynamic bulk array (an axis sized by a runtime ``DimAtom``) has a
        # 0-d placeholder ``_cells``; its build-time shape is the symbolic
        # ``_bulk.shape`` (may carry ``DimAtom`` entries — see B6).
        """Return the array's shape, with a ``DimAtom`` for any runtime-sized axis."""
        if self._bulk is not None and is_dynamic(self._bulk.shape):
            return self._bulk.shape
        return self._cells.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the cell array's dtype."""
        return self._cells.dtype

    @property
    def ndim(self) -> int:
        """Return the number of axes."""
        return self._cells.ndim

    @property
    def flat(self) -> np.flatiter:
        """Iterate the cells in row-major order."""
        return self.cells.flat

    @property
    def is_numeric(self) -> bool:
        """Report whether this array is purely numeric — float cells, no deferred bulk."""
        return self._bulk is None and self._cells.dtype.kind == "f"

    @property
    def is_scalar(self) -> bool:
        """Report whether this array is a scalar, i.e. has no axes."""
        return self._cells.shape == ()

    def __len__(self) -> int:
        if self._cells.ndim == 0:
            raise TypeError("len() of 0-d SymArray")
        return self._cells.shape[0]

    def __getitem__(self, idx: Index) -> SymArray:
        out = self.cells[idx]
        if isinstance(out, np.ndarray):
            return SymArray(out, program=self.program)
        return out

    def __iter__(self) -> Iterator[SymArray]:
        for c in self.cells:
            if isinstance(c, np.ndarray):
                yield SymArray(c, program=self.program)
            else:
                yield c

    # ------------------------------------------------------------------
    # Arithmetic operators (numpy-like; closed-form cell arithmetic).
    #
    # SymArray exposes matmul/transpose/etc. as named methods; these
    # dunders let operator-based numeric code (``A @ B``, ``A + B``,
    # ``D - C``, ``-M``) run on a SymArray directly, delegating to the
    # cell helpers (numeric fast path, object-RationalFunction otherwise).
    # Program back-reference is propagated so downstream passes still see
    # one owning Program.
    # ------------------------------------------------------------------

    def _elementwise(self, other: SymArray | npt.ArrayLike,
                     op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> SymArray:
        a = self.cells
        if isinstance(other, SymArray):
            b = other.cells
            prog = self.program or other.program
        else:
            b = _to_cells(other)
            prog = self.program
        if a.dtype.kind == "f" and getattr(b, "dtype", None) is not None and b.dtype.kind == "f":
            return SymArray(op(a, b), program=prog)
        return SymArray(op(_ensure_object(a), _ensure_object(b)), program=prog)

    def __matmul__(self, other: SymArray | np.ndarray) -> SymArray:
        return self.matmul(other)

    def __rmatmul__(self, other: SymArray | np.ndarray) -> SymArray:
        return SymArray(_to_cells(other), program=self.program).matmul(self)

    def __neg__(self) -> SymArray:
        a = self.cells
        return SymArray(-a if a.dtype.kind == "f" else -_ensure_object(a), program=self.program)

    def __add__(self, other: SymArray | npt.ArrayLike) -> SymArray:
        return self._elementwise(other, lambda x, y: x + y)

    def __radd__(self, other: SymArray | npt.ArrayLike) -> SymArray:
        return self._elementwise(other, lambda x, y: y + x)

    def __sub__(self, other: SymArray | npt.ArrayLike) -> SymArray:
        return self._elementwise(other, lambda x, y: x - y)

    def __rsub__(self, other: SymArray | npt.ArrayLike) -> SymArray:
        return self._elementwise(other, lambda x, y: y - x)

    def __repr__(self) -> str:
        if self._bulk is not None:
            return f"SymArray(bulk{self._bulk.shape})"
        kind = "num" if self.is_numeric else "rat"
        if self.name is None:
            return f"SymArray({kind}{tuple(self._cells.shape)})"
        return f"SymArray({self.name!r}: {kind}{tuple(self._cells.shape)})"

    # ------------------------------------------------------------------
    # Numeric evaluation
    # ------------------------------------------------------------------

    def evaluate(self, bindings: Mapping[str, np.ndarray | float], *,
                 compiled: bool = True) -> np.ndarray:
        """Substitute numeric ``bindings`` into the cells; return float ndarray.

        ``compiled`` selects the per-cell RF evaluator: the default codegen-and-cache path
        (``eval_numeric_fast`` — amortizes over MANY evaluations of the same cells), or, with
        ``compiled=False``, a direct term-sum (``eval_numeric_direct``) that skips the per-RF
        ``compile`` — the right choice for a FEW evaluations, e.g. the 3-point structural-mask
        probe (:func:`polyarray.schur._structural_mask`). Byte-identical values either way.

        ``bindings`` is keyed by symbolic-input name (e.g. ``"V_0"``) and
        each value is the numeric vector / scalar to substitute for that
        input.  Vector-shape values are auto-flattened to per-generator
        names ``f"{input_name}_{k}"``.

        When :attr:`program` has statements (Stmt-emitting primitives have
        been run), this method runs those statements first to bind the
        Stmt-output atoms — same machinery as :meth:`Program.run`, but
        targeted at this single SymArray rather than the program's
        registered outputs.
        """
        if self._bulk is not None:
            # Bulk: run the producing Stmt and read the whole tensor —
            # never materialise per-cell.
            if self.program is None:
                raise ValueError("bulk SymArray.evaluate requires a program")
            flat = self.program.build_runtime_bindings(bindings)
            return np.asarray(flat[self._bulk.name], dtype=float)
        if self._cells.dtype.kind == "f":
            return self._cells.astype(float)
        if self.program is not None:
            flat = self.program.build_runtime_bindings(bindings)
        else:
            flat = _flatten_values_loose(bindings)
        out = np.empty(self._cells.shape, dtype=float)
        for idx in np.ndindex(*self._cells.shape):
            cell = self._cells[idx]
            if isinstance(cell, RationalFunction):
                try:
                    out[idx] = (cell.eval_numeric_fast(flat) if compiled
                                else cell.eval_numeric_direct(flat))
                except KeyError as exc:
                    missing = [n for n in cell.gens if n not in flat]
                    raise KeyError(
                        f"SymArray.evaluate: missing bindings for "
                        f"{sorted(missing)!r} (cell at index {idx})"
                    ) from exc
            else:
                out[idx] = float(cell)
        return out

    # ------------------------------------------------------------------
    # Operation methods (closed-form-or-emit)
    # ------------------------------------------------------------------

    def reshape(self, shape: tuple[int, ...] | list[int]) -> SymArray:
        """``self.reshape(shape)``, keeping a bulk array BULK.

        A reshape wants no cell VALUES — only a shape. But ``.cells`` auto-unpacks a bulk array
        (one ``unpack`` Stmt materialising every per-cell atom), so ``SymArray(sa.cells.reshape(...))``
        forces the whole tensor to answer a question about its layout. Six sites across pointwise and
        savo were doing exactly that, for no other reason than that this method did not exist —
        `ReshapeOp` has been in the IR (and rendered by pyab, and degree-transparent) all along.

        Bulk in, bulk out: emits one ``ReshapeOp`` Stmt so the chain stays deferred. Non-bulk goes
        through ``_cells`` directly, which is already materialised, so nothing is forced there either.

        ``-1`` is resolved here rather than left to numpy, because the emitted ``OutSpec`` needs a
        concrete shape.
        """
        cur = self.shape
        static_cur = not is_dynamic(cur)
        shape = tuple(int(s) for s in shape)
        if -1 in shape:
            if not static_cur:
                raise ValueError(
                    "SymArray.reshape: -1 needs a static source shape; this array's shape is "
                    f"dynamic ({cur}). Pass the resolved shape.")
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            total = 1
            for s in cur:
                total *= int(s)
            if known == 0 or total % known:
                raise ValueError(f"SymArray.reshape: cannot resolve -1 in {shape} from {cur}")
            shape = tuple(total // known if s == -1 else s for s in shape)
        if static_cur:
            total, want = 1, 1
            for s in cur:
                total *= int(s)
            for s in shape:
                want *= s
            if total != want:
                raise ValueError(
                    f"SymArray.reshape: cannot reshape size {total} {tuple(cur)} into {shape}")
        if self._bulk is not None and self.program is not None:
            [out] = self.program.emit_stmt(
                ReshapeOp(shape), [self], [OutSpec("reshape", shape)],
                note="SymArray.reshape", bulk=True,
            )
            return out                                   # still bulk — nothing materialised
        return SymArray(self._cells.reshape(shape), program=self.program)

    def expand_dims(self, axis: int) -> SymArray:
        """Insert a length-1 axis at ``axis`` — the deferral-safe ``arr[:, np.newaxis]``.

        Spelled on top of :meth:`reshape`, so it keeps a bulk array bulk for the same reason.
        """
        cur = self.shape
        if is_dynamic(cur):
            raise ValueError(
                "SymArray.expand_dims: needs a static shape to place the new axis; this array's "
                f"shape is dynamic ({cur}). Use `reshape` with the resolved shape.")
        n = len(cur)
        if not -n - 1 <= axis <= n:
            raise ValueError(f"SymArray.expand_dims: axis {axis} out of range for ndim {n}")
        at = axis if axis >= 0 else axis + n + 1
        return self.reshape((*tuple(int(s) for s in cur[:at]), 1,
                             *tuple(int(s) for s in cur[at:])))

    def transpose(self) -> SymArray:
        """Return ``self.T`` as a new SymArray bound to the same program."""
        return SymArray(self.cells.T, program=self.program)

    @property
    def T(self) -> SymArray:
        """Alias for :meth:`transpose`."""
        return self.transpose()

    def matmul(self, other: SymArray | np.ndarray) -> SymArray:
        """Matrix product ``self @ other``.

        Numeric short-circuit when both operands are float arrays;
        cell-arithmetic dispatch otherwise.
        """
        Aa = self.cells
        Bb = _to_cells(other)
        if Aa.dtype.kind == "f" and Bb.dtype.kind == "f":
            return SymArray(Aa @ Bb, program=self.program)
        result = _matmul_cells(_ensure_object(Aa), _ensure_object(Bb))
        return SymArray(result, program=self.program)

    def matvec(self, v: SymArray | np.ndarray) -> SymArray:
        """Matrix–vector product ``self @ v``."""
        Aa = self.cells
        vv = _to_cells(v)
        if Aa.dtype.kind == "f" and vv.dtype.kind == "f":
            return SymArray(Aa @ vv, program=self.program)
        result = _matvec_cells(_ensure_object(Aa), _ensure_object(vv))
        return SymArray(result, program=self.program)

    def einsum(self, subscripts: str, *others: SymArray | np.ndarray) -> SymArray:
        """Contract cells with ``np.einsum(subscripts, self, *others)``, threading the program.

        A general axis-specified contraction, as in ``"mi...,ip->mp..."``. Numeric operands
        short-circuit to a float einsum; otherwise the cells contract through
        :class:`RationalFunction` arithmetic. One program, two lanes, as in :meth:`matmul`.
        The bound program is ``self``'s; any ``SymArray`` operand must share it (or be
        program-less / numeric).

        **Bulk in, bulk out** (as :meth:`reshape`): a contraction wants no per-cell VALUES, only the
        contraction itself — but ``.cells`` auto-unpacks a bulk array (one ``unpack`` Stmt materialising
        every per-cell atom, then an object-dtype ``np.einsum`` doing ring arithmetic over them). When any
        operand is a bulk (deferred whole-tensor) :class:`SymArray` and a program is present, the
        contraction is handed to :func:`runtime_einsum_multi`, whose bulk branch emits ONE
        :class:`EinsumStmtOp` :class:`Stmt` and keeps the chain deferred. Mathematically the same
        contraction; it is simply not densified into the symbolic ring on the way. A bulk array is by
        construction a Stmt output (opaque atoms), so this never intercepts the numeric lane, and a
        non-bulk operand set takes exactly the code below.
        """
        prog = self.program
        operands: list[SymArray | np.ndarray] = [self]
        for o in others:
            if isinstance(o, SymArray):
                if o.program is not None and prog is not None and o.program is not prog:
                    raise ValueError("SymArray.einsum: operands bind different programs")
                prog = prog or o.program
            operands.append(o)
        if prog is not None and any(
            isinstance(o, SymArray) and o._bulk is not None for o in operands
        ):
            out = runtime_einsum_multi(
                subscripts, *operands, program=prog,
                out_shape=_einsum_output_shape(subscripts, operands), name="einsum")
            return out if isinstance(out, SymArray) else SymArray(np.asarray(out), program=prog)
        cells = [_to_cells(o) for o in operands]
        if all(c.dtype.kind == "f" for c in cells):                # numeric lane
            return SymArray(np.einsum(subscripts, *cells), program=prog)
        return SymArray(np.einsum(subscripts, *[_ensure_object(c) for c in cells]), program=prog)  # object lane

    def det(self, budget: SymbolicBudget | None = None) -> SymArray:
        """Return ``det(self)`` as a 0-d SymArray.

        Numeric short-circuit; closed-form Bareiss for object cells in
        budget; ``Stmt(fn=numpy.linalg.det)`` otherwise.
        """
        if self.cells.dtype.kind == "f":
            return SymArray(np.asarray(np.linalg.det(self.cells)), program=self.program)
        n, m = self.cells.shape
        if n != m:
            raise ValueError(f"det requires square matrix; got {self.cells.shape}")
        eff = budget or _program_budget(self.program)
        if n > eff.naive_inverse_max_size:
            if self.program is None:
                raise RuntimeError(
                    "det: over-budget Stmt requires a program (this SymArray is standalone)"
                )
            A_arr = SymArray(_ensure_object(self.cells), program=self.program)
            [out] = self.program.emit_stmt(
                DetOp(),
                [A_arr],
                [OutSpec("det", ())],
                note=f"det({n}x{n})",
            )
            return out
        # In-budget closed form via Bareiss.
        from .rational import bareiss_det
        result = bareiss_det(_ensure_object(self.cells))
        cells = np.empty((), dtype=object)
        cells[()] = result
        return SymArray(cells, program=self.program)

    def inverse(self, budget: SymbolicBudget | None = None) -> SymArray:
        """Return ``inv(self)`` as an ``(n, n)`` SymArray.

        Numeric short-circuit; closed-form cofactor for object cells in
        budget; ``Stmt(fn=numpy.linalg.inv)`` otherwise.
        """
        if self.cells.dtype.kind == "f":
            return SymArray(np.linalg.inv(self.cells), program=self.program)
        n, m = self.cells.shape
        if n != m:
            raise ValueError(f"inverse requires square matrix; got {self.cells.shape}")
        eff = budget or _program_budget(self.program)
        if n > eff.naive_inverse_max_size:
            if self.program is None:
                raise RuntimeError(
                    "inverse: over-budget Stmt requires a program (this SymArray is standalone)"
                )
            A_arr = SymArray(_ensure_object(self.cells), program=self.program)
            [out] = self.program.emit_stmt(
                InvOp(),
                [A_arr],
                [OutSpec("inv", (n, n))],
                note=f"inv({n}x{n})",
            )
            return out
        from .rational import cofactor_inverse
        return SymArray(cofactor_inverse(_ensure_object(self.cells)), program=self.program)

    def pinv(self) -> SymArray:
        """Moore–Penrose pseudoinverse.  Numeric eager; symbolic always emits."""
        if self.cells.dtype.kind == "f":
            return SymArray(np.linalg.pinv(self.cells), program=self.program)
        if self.program is None:
            raise RuntimeError(
                "pinv on a symbolic SymArray requires a program (this is standalone)"
            )
        if self.cells.ndim != 2:
            raise ValueError(f"pinv requires a 2-D matrix; got {self.cells.shape}")
        m, n = self.cells.shape
        A_arr = SymArray(_ensure_object(self.cells), program=self.program)
        [out] = self.program.emit_stmt(
            PinvOp(),
            [A_arr],
            [OutSpec("pinv", (n, m))],
            note=f"pinv({m}x{n})",
        )
        return out

    def solve(self, b: SymArray | np.ndarray) -> SymArray:
        """Solve ``self @ x = b``.  Numeric eager; symbolic emits ``np.linalg.solve``."""
        Aa = self.cells
        Bb = _to_cells(b)
        if Aa.dtype.kind == "f" and Bb.dtype.kind == "f":
            return SymArray(np.linalg.solve(Aa, Bb), program=self.program)
        if self.program is None:
            raise RuntimeError(
                "solve on a symbolic SymArray requires a program (this is standalone)"
            )
        Ao = _ensure_object(Aa)
        Bo = _ensure_object(Bb)
        A_arr = SymArray(Ao, program=self.program)
        B_arr = SymArray(Bo, program=self.program)
        [out] = self.program.emit_stmt(
            SolveOp(),
            [A_arr, B_arr],
            [OutSpec("solve", Bb.shape)],
            note=f"solve({Aa.shape[0]}x{Aa.shape[1]})",
        )
        return out

    # ------------------------------------------------------------------
    # Scalar Stmt-emitters (sqrt, abs, sign)
    # ------------------------------------------------------------------

    def sqrt(self) -> SymArray:
        """Element-wise sqrt; numeric eager, symbolic emits a 0-d Stmt per cell.

        Restricted to 0-d (scalar) SymArrays in slice B — broader
        broadcasting comes later.
        """
        return self._scalar_op(SqrtOp(), "sqrt", float_fn=np.sqrt)

    def abs(self) -> SymArray:
        """Element-wise abs; 0-d only in slice B."""
        return self._scalar_op(AbsOp(), "abs", float_fn=np.abs)

    def sign(self) -> SymArray:
        """Element-wise sign; 0-d only in slice B."""
        return self._scalar_op(SignOp(), "sign", float_fn=np.sign)

    def _scalar_op(self, stmt_fn: StmtFn, name: str, *,
                   float_fn: Callable[[np.ndarray], np.ndarray]) -> SymArray:
        if self.cells.dtype.kind == "f":
            return SymArray(np.asarray(float_fn(self.cells)), program=self.program)
        if self.program is None:
            raise RuntimeError(
                f"{name} on a symbolic SymArray requires a program "
                "(this is standalone)"
            )
        # Scalar element-ops produce 0-d outputs — bulk gives no memory
        # benefit, and downstream reads them per-cell, so keep them
        # materialised (per-cell atoms).
        if self.cells.shape == ():
            [out] = self.program.emit_stmt(
                stmt_fn, [self], [OutSpec(name, ())], note=name, bulk=False,
            )
            return out
        # Arbitrary-shape: emit one 0-d Stmt per cell and pack the results
        # back into an object-dtype ndarray of the same shape.
        flat = self.cells.reshape(-1)
        out_cells = np.empty(flat.shape, dtype=object)
        for i, c in enumerate(flat):
            cell_sa = SymArray(np.asarray(c), program=self.program)
            [scalar_out] = self.program.emit_stmt(
                stmt_fn, [cell_sa], [OutSpec(f"{name}[{i}]", ())],
                note=f"{name}[{i}]", bulk=False,
            )
            out_cells[i] = scalar_out.cells.item()
        return SymArray(out_cells.reshape(self.cells.shape), program=self.program)

    # ------------------------------------------------------------------
    # Internal: rebinding for Program.copy()
    # ------------------------------------------------------------------

    def _rebind(self, program: Program) -> SymArray:
        new = SymArray(self._cells, program=program, name=self.name)
        new._bulk = self._bulk
        return new


def _program_budget(program: Program | None) -> SymbolicBudget:
    """Return the budget on ``program``, or a default ``SymbolicBudget()``."""
    if program is None:
        return SymbolicBudget()
    return program.budget


# ---------------------------------------------------------------------------
# Typed numpy ops (Plan B step 2)
# ---------------------------------------------------------------------------
#
# Each op is a frozen, hashable dataclass with a ``__call__`` that runs the
# modeled numpy operation on the bound *numeric* operands at ``Program.run``
# time — the same shape as the pre-existing :class:`EinsumOp` / :class:`SwitchOp`
# / :class:`covariant.QrSignFixOp`.  Promoting the linalg / scalar deferrals from
# bare ``np.linalg.*`` callables (which the SymArray methods previously stored as
# ``Stmt.fn``) to these typed ops is behaviour-preserving — the numeric result is
# identical — but gives the IR typed, inspectable nodes with a stable hashable
# identity (the hook the future ``fingerprint`` / partial-eval transform needs to
# recognise a linalg / structural node without ``is np.linalg.det`` sniffing).
#
# ``TensordotOp`` / ``MoveaxisOp`` are the contraction / structural surface the
# covariant frame conversion applies to symbolic data today (``covariant.py``
# ``_apply_per_slot`` runs them *inline on object cells* — the rational lane).
# They are added here as round-trip-tested vocabulary but are NOT yet emitted by
# any sampler: turning those inline contractions into deferred numeric Stmts is a
# symbolic→numeric deferral decision, deferred to Plan B step 4 (budget-zero) and
# gated on flagging it first.


@dataclass(frozen=True)
class DetOp:
    """Typed ``np.linalg.det`` over a bound numeric matrix."""

    def __call__(self, A: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(np.linalg.det(np.asarray(A)))


@dataclass(frozen=True)
class InvOp:
    """Typed ``np.linalg.inv`` over a bound numeric matrix."""

    def __call__(self, A: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.linalg.inv(np.asarray(A))


@dataclass(frozen=True)
class PinvOp:
    """Typed ``np.linalg.pinv`` over a bound numeric matrix."""

    def __call__(self, A: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.linalg.pinv(np.asarray(A))


@dataclass(frozen=True)
class SolveOp:
    """Typed ``np.linalg.solve`` over bound numeric ``A``, ``b``."""

    def __call__(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.linalg.solve(np.asarray(A), np.asarray(b))


@dataclass(frozen=True)
class SqrtOp:
    """Typed element-wise ``np.sqrt`` over a bound numeric array."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.sqrt(np.asarray(x, dtype=float))


@dataclass(frozen=True)
class AbsOp:
    """Typed element-wise ``np.abs`` over a bound numeric array."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.abs(np.asarray(x, dtype=float))


@dataclass(frozen=True)
class SignOp:
    """Typed element-wise ``np.sign`` over a bound numeric array."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.sign(np.asarray(x, dtype=float))


@dataclass(frozen=True)
class SvdOp:
    """Stmt-compatible ``np.linalg.svd`` plus a thresholded numerical rank.

    Multi-output: returns ``(U, S, Vh, rank)`` where ``rank`` is a 0-d
    integer ndarray giving the numerical rank of ``A`` at tolerance
    ``rcond`` (defaulting to numpy's ``matrix_rank`` rule).  The ``rank``
    output is the runtime dimension ``δ_f`` consumed as a :class:`DimAtom`
    (Stage B) — sizing image / projected / FFS arrays at run time.

    ``full_matrices=False`` (the default) yields the reduced SVD, matching
    the orthonormal-basis use; ``rcond`` mirrors ``np.linalg.matrix_rank``.
    """

    rcond: float | None = None
    full_matrices: bool = False

    def __call__(self, A: np.ndarray) -> tuple[np.ndarray, ...]:
        A = np.asarray(A, dtype=float)
        U, S, Vh = np.linalg.svd(A, full_matrices=self.full_matrices)
        if S.size == 0:
            rank = 0
        elif self.rcond is None:
            tol = S.max() * max(A.shape) * np.finfo(float).eps
            rank = int((S > tol).sum())
        else:
            tol = self.rcond * S.max()
            rank = int((S > tol).sum())
        return U, S, Vh, np.asarray(rank)


@dataclass(frozen=True)
class GSvdOp:
    """Metric-aware generalised SVD of a map ``A : V → W``.

    Multi-output: returns ``(U, UI, V, VI, S, rank)``.  ``A`` is the
    ``|W|×|V|`` matrix of the map; ``M_V`` (``|V|×|V|`` SPD) is the domain
    inner-product / metric and ``M_W`` (``|W|×|W|`` SPD) the codomain metric.

    The factors are orthonormal **in the respective metrics** and split into
    the four-fundamental-subspace (FFS) blocks at the numerical ``rank``
    (``δ_f``), exactly as in `50 §A1`/`§A5`:

    .. list-table::
       :header-rows: 1

       * - output
         - columns
         - FFS block / property
       * - ``U``
         - first ``rank`` cols of the codomain factor
         - **image** basis of ``A`` in ``W``; ``Uᵀ M_W U = I``
       * - ``UI``
         - remaining ``|W|−rank`` cols
         - **coker** basis (``M_W``-⊥ complement of the image)
       * - ``V``
         - first ``rank`` cols of the domain factor
         - **coimg** basis of ``A`` in ``V``; ``Vᵀ M_V V = I``
       * - ``VI``
         - remaining ``|V|−rank`` cols
         - **ker** basis (``A·VI ≈ 0``)
       * - ``S``
         - descending singular values
         - the ``rank`` nonzero ones lead
       * - ``rank``
         - 0-d int ndarray
         - numerical rank ``δ_f``

    Both ``U`` (``=[U|UI]``) and ``V`` (``=[V|VI]``) are returned as the
    leading (image / coimg) blocks only; ``UI`` / ``VI`` carry the
    complementary (coker / ker) blocks — so the full metric-orthonormal
    factors are the horizontal concatenations ``[U, UI]`` and ``[V, VI]``.

    **Construction.**  Whiten ``A`` by the Cholesky factors of the two
    metrics (``M_W = Lw Lwᵀ``, ``M_V = Rw Rwᵀ``)::

        Aw = Lwᵀ · A · Rw⁻ᵀ            # both spaces now orthonormal
        Ũ, S, Ṽᵀ = svd(Aw)            # ordinary SVD in white coords

    then de-whiten the factors back into the metrics::

        [U|UI] = Lw⁻ᵀ · Ũ             # ⇒ ([U|UI])ᵀ M_W ([U|UI]) = I
        [V|VI] = Rw⁻ᵀ · Ṽ             # ⇒ ([V|VI])ᵀ M_V ([V|VI]) = I

    **Reconstruction identity** (note the trailing ``M_V`` — the dual
    pairing makes the coordinates contravariant)::

        A = U · diag(S[:rank]) · Vᵀ · M_V
          = [U|UI] · Sfull · [V|VI]ᵀ · M_V        # full form

    where ``Sfull`` is the ``|W|×|V|`` rectangular-diagonal padding of ``S``.

    **Reduction to** :class:`SvdOp`.  With ``M_V = M_W = I`` the Cholesky
    factors are the identity, ``U = Ũ``, ``V = Ṽ``, and the identity above
    collapses to ``A = U diag(S) Vᵀ`` — i.e. ``Vᵀ`` is :class:`SvdOp`'s
    ``Vh`` and ``rank``/``S`` agree.  ``rcond`` mirrors ``SvdOp`` /
    ``np.linalg.matrix_rank`` (threshold on the *whitened* singular values).
    """

    rcond: float | None = None

    def __call__(
        self, A: np.ndarray, M_V: np.ndarray, M_W: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Evaluate the op on concrete numeric operands."""
        A = np.asarray(A, dtype=float)
        Lw = np.linalg.cholesky(np.asarray(M_W, dtype=float))  # M_W = Lw Lwᵀ
        Rw = np.linalg.cholesky(np.asarray(M_V, dtype=float))  # M_V = Rw Rwᵀ
        Aw = Lw.T @ A @ np.linalg.inv(Rw.T)  # whiten both spaces
        Ut, S, Vth = np.linalg.svd(Aw, full_matrices=True)
        if S.size == 0:
            rank = 0
        elif self.rcond is None:
            tol = S.max() * max(Aw.shape) * np.finfo(float).eps
            rank = int((S > tol).sum())
        else:
            tol = self.rcond * S.max()
            rank = int((S > tol).sum())
        # De-whiten the factors back into the two metrics:
        #   [U|UI] = Lw⁻ᵀ Ũ   so ([U|UI])ᵀ M_W ([U|UI]) = I
        #   [V|VI] = Rw⁻ᵀ Ṽ   so ([V|VI])ᵀ M_V ([V|VI]) = I
        U_full = np.linalg.solve(Lw.T, Ut)
        V_full = np.linalg.solve(Rw.T, Vth.T)
        # Split into image/coimg (first ``rank``) vs coker/ker (complement):
        U, UI = U_full[:, :rank], U_full[:, rank:]
        V, VI = V_full[:, :rank], V_full[:, rank:]
        return U, UI, V, VI, S, np.asarray(rank)


@dataclass(frozen=True)
class QrOp:
    """Stmt-compatible ``np.linalg.qr`` over a bound numeric matrix.

    Multi-output: returns ``(Q, R)``.  ``mode`` follows numpy's QR modes
    (``"reduced"`` by default — the orthonormal-basis / image use).
    """

    mode: str = "reduced"

    def __call__(self, A: np.ndarray) -> tuple[np.ndarray, ...]:
        Q, R = np.linalg.qr(np.asarray(A, dtype=float), mode=self.mode)
        return Q, R


@dataclass(frozen=True)
class TransposeOp:
    """Full-reverse transpose of a bound array (``A.T``).

    Value-preserving on the cells (no float coercion), so it threads symbolic
    ``RationalFunction`` cells as well as numeric ones.
    """

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(A).T


@dataclass(frozen=True)
class SinvFullOp:
    """The ``nrows×ncols`` rectangular-diagonal pseudo-inverse ``S⁻¹``.

    ``out[i, i] = 1/Sᵢ`` for ``i < rank`` (the numerical rank), else ``0`` — the
    metric pseudo-inverse assembled from a :class:`GSvdOp`'s singular values
    (grassmann `50 §A5`).  ``rank`` is a runtime 0-d int; the diagonal beyond it
    is masked to zero (so trailing near-zero singular values never divide).
    """

    nrows: int
    ncols: int

    def __call__(self, S: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        r = int(rank)
        out = np.zeros((self.nrows, self.ncols))
        s = np.asarray(S, dtype=float)
        for i in range(r):
            out[i, i] = 1.0 / s[i]
        return out


@dataclass(frozen=True)
class GSvdFullOp:
    """Run :class:`GSvdOp` and return the full-width metric-orthonormal factors.

    Multi-output ``(Ufull, Vfull, S, rank)`` with *static* widths: ``Ufull =
    [U|UI]`` (``cod×cod``) and ``Vfull = [V|VI]`` (``dom×dom``) are the full
    de-whitened factors (image/coimg = leading ``rank`` cols, coker/ker = the
    trailing complement).  The runtime-δ block slice happens in a downstream
    Stmt referencing this op's ``rank`` output (grassmann `50 §A1`).
    """

    rcond: float | None = None

    def __call__(
        self, A: np.ndarray, M_V: np.ndarray, M_W: np.ndarray
    ) -> tuple[np.ndarray, ...]:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        U, UI, V, VI, S, rank = GSvdOp(rcond=self.rcond)(A, M_V, M_W)
        Ufull = np.concatenate([np.asarray(U), np.asarray(UI)], axis=1)
        Vfull = np.concatenate([np.asarray(V), np.asarray(VI)], axis=1)
        return Ufull, Vfull, np.asarray(S), np.asarray(rank)


@dataclass(frozen=True)
class BlockDiagOp:
    """Block-diagonal assembly ``diag(A, B, …)`` of the bound operands."""

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        arrs = [np.asarray(m, dtype=float) for m in mats]
        rows = sum(a.shape[0] for a in arrs)
        cols = sum(a.shape[1] for a in arrs)
        out = np.zeros((rows, cols))
        r = c = 0
        for a in arrs:
            out[r:r + a.shape[0], c:c + a.shape[1]] = a
            r += a.shape[0]
            c += a.shape[1]
        return out


@dataclass(frozen=True)
class BlockRepeatOp:
    """``n`` block-diagonal copies of ``A`` — ``kron(eye(n), A)`` (static ``n``)."""

    n: int

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        a = np.asarray(A, dtype=float)
        out = np.zeros((self.n * a.shape[0], self.n * a.shape[1]))
        for i in range(self.n):
            out[i * a.shape[0]:(i + 1) * a.shape[0], i * a.shape[1]:(i + 1) * a.shape[1]] = a
        return out


@dataclass(frozen=True)
class DynBlockRepeatOp:
    """Lay out ``n`` block-diagonal copies of ``A`` for a runtime count, as ``kron(eye(n), A)``."""

    def __call__(self, A: np.ndarray, n: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        a = np.asarray(A, dtype=float)
        count = int(np.asarray(n))
        out = np.zeros((count * a.shape[0], count * a.shape[1]))
        for i in range(count):
            out[i * a.shape[0]:(i + 1) * a.shape[0], i * a.shape[1]:(i + 1) * a.shape[1]] = a
        return out


# ---------------------------------------------------------------------------
# Generic array builtins relocated from grassmann's lowering layer (Batch 2).
# Each is a frozen, hashable Stmt.fn with a numpy host ``__call__``; both render
# lanes live in ``pyab._ARRAY_OP_LOWERINGS`` and ``numpy_source._builtin_renderers``.
# Several produce a 0-d int sizing a downstream dynamic axis (DimAtom source).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynEyeOp:
    """A runtime identity sized by a reference array's axis.

    ``axis`` selects which axis of the reference sizes the identity; it defaults to the rank
    column at axis 1.
    """

    axis: int = 1

    def __call__(self, ref: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.eye(int(np.asarray(ref).shape[self.axis]))


@dataclass(frozen=True)
class DynZerosOp:
    """``np.zeros(d₀, d₁, …)`` sized off ``refs[i].shape[axes[i]]`` — a symbolic ℝⁿ zero."""

    axes: tuple[int, ...]

    def __call__(self, *refs: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.zeros(tuple(int(np.asarray(refs[i]).shape[a]) for i, a in enumerate(self.axes)))


@dataclass(frozen=True)
class DynEyeTensorOp:
    """The vmap identity of a multi-axis dimension binder, a matrix or tensor seed.

    Emits ``np.eye(∏dᵢ).reshape(∏dᵢ, d₀, d₁, …)``, with each ``dᵢ`` read from
    ``refs[i].shape[axes[i]]``.
    """

    axes: tuple[int, ...]

    def __call__(self, *refs: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        dims = [int(np.asarray(refs[i]).shape[a]) for i, a in enumerate(self.axes)]
        prod = int(np.prod(dims)) if dims else 1
        return np.eye(prod).reshape(prod, *dims)


@dataclass(frozen=True)
class ProdShapeOp:
    """``static · ∏ refs[i].shape[axes[i]]`` as a 0-d int — a flattened tensor-product dimension."""

    axes: tuple[int, ...]
    static: int = 1

    def __call__(self, *refs: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(int(self.static * int(np.prod(
            [int(np.asarray(refs[i]).shape[a]) for i, a in enumerate(self.axes)] or [1]))))


@dataclass(frozen=True)
class SumShapeOp:
    """``static + Σ refs[i].shape[axes[i]]`` as a 0-d int — a flattened direct-sum dimension."""

    axes: tuple[int, ...]
    static: int = 0

    def __call__(self, *refs: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(int(self.static + int(sum(
            int(np.asarray(refs[i]).shape[a]) for i, a in enumerate(self.axes)))))


@dataclass(frozen=True)
class SumDimOp:
    """Σ of the operands' ``axis`` lengths as a 0-d int — sizes a dynamic block-diag axis."""

    axis: int = 0

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(sum(int(np.asarray(m).shape[self.axis]) for m in mats))


@dataclass(frozen=True)
class ProdDimOp:
    """∏ of the operands' ``axis`` lengths as a 0-d int — sizes a dynamic Kron axis."""

    axis: int = 0

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        out = 1
        for m in mats:
            out *= int(np.asarray(m).shape[self.axis])
        return np.asarray(out)


@dataclass(frozen=True)
class ScaleAxisDimOp:
    """Return ``n`` times the length of ``mat``'s axis, as a 0-d int."""

    n: int
    axis: int = 0

    def __call__(self, mat: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(self.n * int(np.asarray(mat).shape[self.axis]))


@dataclass(frozen=True)
class MulAxisDimOp:
    """Return the runtime ``count`` times the length of ``mat``'s axis, as a 0-d int."""

    axis: int = 0

    def __call__(self, n: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(int(np.asarray(n)) * int(np.asarray(mat).shape[self.axis]))


@dataclass(frozen=True)
class CompRankOp:
    """The complement rank ``ambient − δ`` as a 0-d int — sizes the FFS complement (Ker/CoKer) axis."""

    ambient: int

    def __call__(self, rank: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(self.ambient - int(rank))


@dataclass(frozen=True)
class HStackOp:
    """Horizontally stack matrices (columns side by side) — ``[A | B | …]``."""

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.hstack([np.asarray(m, dtype=float) for m in mats])


@dataclass(frozen=True)
class ColStackOp:
    """Stack 1-D coordinate vectors as the columns of a matrix (the ``ListBasis`` matrix)."""

    def __call__(self, *cols: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.stack([np.asarray(c, dtype=float).reshape(-1) for c in cols], axis=1)


@dataclass(frozen=True)
class ScaleOp:
    """``factor · x`` — static scalar multiply."""

    factor: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return self.factor * np.asarray(x, dtype=float)


@dataclass(frozen=True)
class ScaleByOp:
    """``s · x`` — runtime-scalar multiply (the ``ScaleOp`` sibling)."""

    def __call__(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(s, dtype=float) * np.asarray(x, dtype=float)


@dataclass(frozen=True)
class AddOp:
    """Left-fold sum ``x0 + x1 + …`` over ``n`` operands."""

    n: int

    def __call__(self, *xs: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        out = np.asarray(xs[0], dtype=float).copy()
        for x in xs[1:]:
            out = out + np.asarray(x, dtype=float)
        return out


@dataclass(frozen=True)
class ConcatOp:
    """Flatten each operand then concatenate on axis 0."""

    def __call__(self, *xs: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in xs])


@dataclass(frozen=True)
class AxisLenOp:
    """The (runtime) length of ``x``'s axis ``axis`` as a 0-d int."""

    axis: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(int(np.asarray(x).shape[self.axis]))


@dataclass(frozen=True)
class ReshapeOp:
    """``A.reshape(shape)`` (static shape)."""

    shape: tuple[int, ...]

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(A, dtype=float).reshape(self.shape)


@dataclass(frozen=True)
class ConstOp:
    """A frozen numeric constant rematerialized from raw bytes (no args)."""

    key: str
    data_bytes: bytes
    shape: tuple[int, ...]
    dtype: str

    def __call__(self) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.frombuffer(self.data_bytes, dtype=self.dtype).reshape(self.shape).copy()


@dataclass(frozen=True)
class EyeOp:
    """The static ``n×n`` identity ``np.eye(n)`` (no args)."""

    n: int

    def __call__(self) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.eye(self.n)


@dataclass(frozen=True)
class FirstColsOp:
    """``A[:, :int(rank)]`` — the leading (runtime-``rank``) columns."""

    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(A)[:, : int(rank)]


@dataclass(frozen=True)
class LastColsOp:
    """``A[:, int(rank):]`` — the complementary (trailing) columns."""

    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(A)[:, int(rank):]


# --- Batch-3 relocated generic array / linalg builtins ----------------------
# Subspace project/embed, Kronecker assembly, inverse-transpose / basis-compose,
# the SPD operator square root, numeric rank, and metric orthonormalization —
# bodies moved verbatim from grassmann's ``lower/represent.py`` (pure numpy, no
# grassmann types).  Both lanes render (pyab ``_ARRAY_OP_LOWERINGS`` + numpy
# ``_builtin_renderers``).


@dataclass(frozen=True)
class ProjectOp:
    """``Pᵀ @ v`` — project ambient coords onto the orthonormal sub-basis (drop comp).

    ``v`` may be a multi-axis ambient tensor (e.g. ``V⊗ᵏ`` stored ``(n,…,n)``); it is
    raveled to the ambient (column) coordinate before the projection.
    """

    def __call__(self, P: np.ndarray, v: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(P, dtype=float).T @ np.asarray(v, dtype=float).reshape(-1)


@dataclass(frozen=True)
class EmbedOp:
    """``P @ vsub`` — pad sub-coords into the ambient (zeros in the complement).

    The ambient result is reshaped to the ambient space's multi-axis layout
    (``shape``), e.g. ``V⊗ᵏ`` ⇒ ``(n,…,n)``.
    """

    shape: tuple[int, ...] = ()

    def __call__(self, P: np.ndarray, vsub: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        out = np.asarray(P, dtype=float) @ np.asarray(vsub, dtype=float)
        return out.reshape(self.shape) if self.shape else out


@dataclass(frozen=True)
class KronOp:
    """Chained Kronecker product ``kron(mats[0], mats[1], …)`` of ``n`` matrices."""

    n: int

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        out = np.asarray(mats[0], dtype=float)
        for m in mats[1:]:
            out = np.kron(out, np.asarray(m, dtype=float))
        return out


@dataclass(frozen=True)
class KronFreeOp:
    """``Kron`` of two maps that may carry trailing free (basis-index / seed) axes.

    Each operand is ``(dom, cod, *free)``; the result is ``(dom_f·dom_g, cod_f·cod_g,
    *f_free, *g_free)`` — the Kronecker block structure on the two space axes, an outer
    product on the (disjoint) free axes. With no free axes this is exactly ``np.kron``.
    """

    nf_free: int
    ng_free: int

    def __call__(self, F: np.ndarray, G: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        df, cf, ff = F.shape[0], F.shape[1], F.shape[2:]
        dg, cg, gg = G.shape[0], G.shape[1], G.shape[2:]
        fr = F.reshape(df, 1, cf, 1, *ff, *([1] * self.ng_free))
        gr = G.reshape(1, dg, 1, cg, *([1] * self.nf_free), *gg)
        return (fr * gr).reshape(df * dg, cf * cg, *ff, *gg)


@dataclass(frozen=True)
class InvTransposeOp:
    """``inv(A).T`` — the inverse-transpose (dual-basis change of coordinates)."""

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.linalg.inv(np.asarray(A, dtype=float)).T


@dataclass(frozen=True)
class ComposeViaStdOp:
    """``solve(R_to, R_from)`` — compose two changes-of-basis through the std rep."""

    def __call__(self, R_to: np.ndarray, R_from: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.linalg.solve(np.asarray(R_to, dtype=float), np.asarray(R_from, dtype=float))


@dataclass(frozen=True)
class SqrtSpdOp:
    """The symmetric-positive-definite operator square root ``S``, with ``S·S = G``.

    Computed by eigendecomposition of the symmetrized matrix. Raises if ``G`` is not SPD.
    """

    def __call__(self, G: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        A = 0.5 * (np.asarray(G, dtype=float) + np.asarray(G, dtype=float).T)
        w, V = np.linalg.eigh(A)
        if w.min() <= 0:
            raise ValueError(f"Sqrt: operator is not SPD (min eigenvalue {w.min():.3e})")
        return (V * np.sqrt(w)) @ V.T


@dataclass(frozen=True)
class RankOp:
    """The numeric rank of a column matrix as a 0-d int — sizes a RREF-selected basis."""

    tol: float = 1e-9

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        return np.asarray(int(np.linalg.matrix_rank(np.asarray(A, dtype=float), tol=self.tol)))


@dataclass(frozen=True)
class MetricOrthonormalOp:
    """Orthonormalize columns w.r.t. a metric ``G``: ``A·L⁻ᵀ`` with ``LLᵀ = AᵀGA`` (Cholesky)."""

    def __call__(self, A: np.ndarray, G: np.ndarray) -> np.ndarray:  # noqa: N803
        """Evaluate the op on concrete numeric operands."""
        a = np.asarray(A, dtype=float)
        gram = a.T @ np.asarray(G, dtype=float) @ a
        L = np.linalg.cholesky(gram)
        return a @ np.linalg.inv(L).T


def _check(ok: bool, msg: str, detail: str = "") -> None:
    """Raise ``AssertionError(msg + detail)`` unless ``ok``."""
    if not ok:
        raise AssertionError(msg + detail)


def _is_spd(A: np.ndarray) -> bool:
    """Report whether ``A`` is numerically symmetric positive-definite."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return False
    return True


def _full_rank(A: np.ndarray) -> bool:
    """Report whether ``A`` has full numerical rank, i.e. the smaller of its two dimensions."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        return False
    return int(np.linalg.matrix_rank(A)) == min(A.shape)


@dataclass(frozen=True)
class AssertOp:
    """Passthrough predicate check over bound inputs (the D-scope asserts).

    Validates ``kind`` against the bound inputs and **returns the first
    input unchanged**, so a downstream consumer data-depends on the assert
    (preserving Stmt ordering).  Kinds:

    * ``"shape_eq"``       — ``x.shape == rest[0].shape``
    * ``"rank_eq"``        — ``int(rest[0]) == int(rest[1])`` (δ vs asserted)
    * ``"spd"``            — ``x`` is symmetric positive-definite
    * ``"square_full_rank"`` — ``x`` is square and full rank
    * ``"in_span"``        — ``x`` lies in the column span of ``rest[0]`` (exact projection)

    On failure raises ``AssertionError(msg + detail)``; ``msg`` is the
    caller-supplied prefix.
    """

    kind: str
    msg: str = ""

    def __call__(self, x: np.ndarray, *rest: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self.kind == "shape_eq":
            other = np.asarray(rest[0])
            _check(
                x.shape == other.shape,
                self.msg,
                f"[shape_eq] {x.shape} != {other.shape}",
            )
        elif self.kind == "rank_eq":
            a, b = int(rest[0]), int(rest[1])
            _check(a == b, self.msg, f"[rank_eq] {a} != {b}")
        elif self.kind == "spd":
            _check(_is_spd(x), self.msg, "[spd] matrix not symmetric positive-definite")
        elif self.kind == "square_full_rank":
            _check(
                x.ndim == 2 and x.shape[0] == x.shape[1] and _full_rank(x),
                self.msg,
                f"[square_full_rank] shape={x.shape}",
            )
        elif self.kind == "in_span":
            # `x` (a vector) lies in the column span of `rest[0]` (a frame `P`), i.e. the
            # projection onto that subspace is EXACT.  A consumer that realizes `P` in a
            # non-canonical ambient basis needs this: `pinv(P)` projects along the AMBIENT
            # orthogonal complement, so the answer is ambient-basis-dependent UNLESS the residual
            # is zero, in which case `pinv(P)·x` is the unique exact coordinate vector and every
            # ambient basis agrees.  Guarding at run time is what makes that substitution honest.
            P = np.asarray(rest[0], dtype=float)
            v = np.asarray(x, dtype=float).reshape(-1)
            resid = v - P @ (np.linalg.pinv(P) @ v)
            scale = max(float(np.linalg.norm(v)), 1.0)
            _check(
                float(np.linalg.norm(resid)) <= 1e-9 * scale,
                self.msg,
                f"[in_span] residual {float(np.linalg.norm(resid)):.3e} "
                f"(relative {float(np.linalg.norm(resid)) / scale:.3e}) — the operand does NOT lie "
                f"in the subspace, so projecting it in a non-canonical ambient basis would change "
                f"the answer (the complement direction is ambient-basis-dependent)",
            )
        else:
            raise ValueError(f"AssertOp: unknown kind {self.kind!r}")
        return x


@dataclass(frozen=True)
class TensordotOp:
    """Stmt-compatible ``np.tensordot`` over two bound numeric operands.

    ``axes`` follows the :func:`np.tensordot` convention — an ``int`` or a
    ``(a_axes, b_axes)`` pair.  Build via :meth:`from_axes`, which normalises
    list-form axes to nested tuples so the dataclass stays frozen + hashable
    (consistent with the other typed ops).
    """

    axes: Any = 2

    @classmethod
    def from_axes(cls, axes: int | Sequence[Sequence[int]]) -> TensordotOp:
        """Build the op from ``np.tensordot``'s ``axes`` argument in any accepted form."""
        if isinstance(axes, (int, np.integer)):
            return cls(int(axes))
        a, b = axes
        norm = (
            tuple(int(i) for i in np.atleast_1d(a)),
            tuple(int(i) for i in np.atleast_1d(b)),
        )
        return cls(norm)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.tensordot(np.asarray(a), np.asarray(b), axes=self.axes)


@dataclass(frozen=True)
class MoveaxisOp:
    """Stmt-compatible ``np.moveaxis`` over a bound numeric array.

    ``source`` / ``destination`` follow :func:`np.moveaxis`.  Build via
    :meth:`from_spec` to normalise list-form axis specs to ``int`` / tuple so
    the dataclass stays frozen + hashable.
    """

    source: Any
    destination: Any

    @classmethod
    def from_spec(cls, source: int | Sequence[int], destination: int | Sequence[int]) -> MoveaxisOp:
        """Build the op from ``np.moveaxis``'s source and destination in any accepted form."""
        def _norm(x: int | Sequence[int]) -> tuple[int, ...]:
            if isinstance(x, (int, np.integer)):
                return int(x)
            return tuple(int(i) for i in x)

        return cls(_norm(source), _norm(destination))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        return np.moveaxis(np.asarray(x), self.source, self.destination)


# ---------------------------------------------------------------------------
# Control-flow ops (Plan B step 3): Call + While
# ---------------------------------------------------------------------------
#
# ``SwitchOp`` (the conditional) is the only pre-existing control-flow op.  These
# two make the IR a real mini-language rather than a flat tape: a ``CallOp``
# invokes a sub-:class:`Program` or opaque callable as one IR node, and a
# ``WhileOp`` loops a body over loop-carried state until a predicate fails.  Both
# run on the bound *numeric* operands at :meth:`Program.run` time (the same
# contract as every other Stmt fn), so the wrapped functions see floats, not
# symbolic cells.  Like ``TensordotOp`` / ``MoveaxisOp`` they are round-trip-
# tested vocabulary — not yet emitted by any sampler; Plan C's Monte-Carlo demo
# is the intended first exerciser.
#
# Note: a sub-:class:`Program` can already be a ``Stmt.fn`` directly (``_run_stmt``
# special-cases ``isinstance(fn, Program)``).  ``CallOp`` is the *typed, uniform*
# call node — it also wraps opaque callables and gives the call a stable hashable
# identity for the future ``fingerprint`` / partial-eval pass.


def _invoke(fn: StmtOp, args: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Run a callable / sub-:class:`Program` on ``args``; return a result tuple.

    A :class:`Program` is run with ``args`` mapped to its inputs by position;
    its outputs come back as a tuple in insertion order.  An opaque callable is
    called directly and its return normalised to a tuple.
    """
    if isinstance(fn, Program):
        values = {inp.name: np.asarray(a) for inp, a in zip(fn.inputs, args)}
        return tuple(fn.run(values).values())
    res = fn(*args)
    return res if isinstance(res, tuple) else (res,)


@dataclass(frozen=True)
class CallOp:
    """Invoke a wrapped callable / sub-:class:`Program` as one IR node.

    ``fn`` is either a :class:`Program` (operands mapped to its inputs by
    position; outputs returned in insertion order) or an opaque callable (run on
    the bound numeric operands).  Returns a bare array for a single output, a
    tuple otherwise — matching the multi-output unpacking in ``_run_stmt``.
    Frozen + hashable (``fn`` is hashed by identity), like the other typed ops.
    """

    fn: Callable[..., Any] | Program

    def __call__(self, *operands: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Evaluate the op on concrete numeric operands."""
        outs = _invoke(self.fn, operands)
        return outs[0] if len(outs) == 1 else outs


@dataclass(frozen=True)
class WhileOp:
    """Loop ``body`` over loop-carried state while ``cond`` holds.

    Semantics at run time, over the resolved numeric operands as the initial
    state tuple::

        state = operands
        while cond(*state):          # cond returns a truthy scalar
            state = body(*state)     # body returns the new state (same arity)
        return state

    ``cond`` / ``body`` are each a :class:`Program` or an opaque callable (see
    :func:`_invoke`).  ``body`` must return the carried state in the same arity
    and order as the operands.  ``max_iters`` is a non-termination guard — the
    op raises :class:`RuntimeError` if exceeded.  Returns a bare array for a
    single carried value, a tuple otherwise.
    """

    cond: Callable[..., Any] | Program
    body: Callable[..., Any] | Program
    max_iters: int = 100_000

    def __call__(self, *operands: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Evaluate the op on concrete numeric operands."""
        state: tuple = tuple(np.asarray(o) for o in operands)
        n = len(state)
        for _ in range(self.max_iters):
            if not bool(np.asarray(_invoke(self.cond, state)[0])):
                break
            state = _invoke(self.body, state)
            if len(state) != n:
                raise ValueError(
                    f"WhileOp body returned {len(state)} values; "
                    f"expected {n} (loop-carried state arity)"
                )
        else:
            raise RuntimeError(
                f"WhileOp exceeded max_iters={self.max_iters} without cond "
                "going false"
            )
        return state[0] if n == 1 else state


# ---------------------------------------------------------------------------
# Intermediate-binding "freeze" — keeps polynomial sizes bounded
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentityOp:
    """Stmt-compatible passthrough: at run-time, return the bound input.

    Used by :func:`freeze_array` to capture a heavy
    :class:`RationalFunction` cell as a fresh atom.  Downstream code
    sees the small atom; the original (potentially large) symbolic
    expression is computed once at :meth:`Program.run` time and bound
    to the atom.  Hashable / equal across construction so multiple
    freeze sites share a stable identity.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)


@dataclass(frozen=True)
class EinsumOp:
    """Stmt-compatible runtime einsum.

    At Stmt-build time we replace a heavy ``np.einsum(spec, lhs_RF,
    rhs)`` over object-dtype :class:`RationalFunction` cells with a
    single :class:`Stmt` whose output SymArray is fresh atom RFs and
    whose run-time callable evaluates the einsum on the bound numeric
    LHS array (and the captured numeric RHS).  Cuts the build-time
    symbolic blow-up from ``Σ_k RF * float`` over object dtype to a
    pure-numeric ``np.einsum`` at :meth:`Program.run` time.

    ``spec`` is the einsum subscript string (e.g. ``"mk...,kn->mn..."``).
    ``rhs`` is captured by the Op (numeric ndarray); the Stmt's only
    symbolic input is the LHS tensor.  Not used for contractions
    where both operands are symbolic — those still need RF arithmetic.

    Hashable: ``rhs`` is wrapped in ``rhs_bytes`` so the dataclass is
    `frozen` + ``eq=True`` consistent with :class:`Program.fingerprint`
    (slice C).
    """

    spec: str
    rhs_shape: tuple
    rhs_dtype: str
    rhs_bytes: bytes

    @classmethod
    def from_rhs(cls, spec: str, rhs: np.ndarray) -> EinsumOp:
        rhs = np.ascontiguousarray(rhs)
        return cls(
            spec=spec,
            rhs_shape=tuple(rhs.shape),
            rhs_dtype=str(rhs.dtype),
            rhs_bytes=rhs.tobytes(),
        )

    @property
    def _rhs(self) -> np.ndarray:
        return np.frombuffer(
            self.rhs_bytes, dtype=np.dtype(self.rhs_dtype),
        ).reshape(self.rhs_shape)

    def __call__(self, lhs: np.ndarray) -> np.ndarray:
        # ``lhs`` arrives bound to numeric values (float).  Run the
        # einsum on bound floats; the result is the same shape as
        # the symbolic Stmt's declared output.
        return np.einsum(self.spec, np.asarray(lhs), self._rhs)


def runtime_einsum(
    spec: str,
    lhs: np.ndarray | SymArray,
    rhs: np.ndarray,
    *,
    program: Program | None,
    out_shape: tuple,
    name: str = "einsum",
) -> np.ndarray | SymArray:
    """Defer a symbolic ``einsum(spec, lhs, rhs)`` to a runtime :class:`EinsumOp` statement.

    ``rhs`` must already be numeric.  Behaviour by ``lhs``:

    * ``program is None`` or numeric ``lhs`` → eager ``np.einsum`` (the
      float path is untouched).
    * **bulk** ``lhs`` (a deferred whole-tensor SymArray) → offload-keeps-
      bulk: emit one :class:`EinsumOp` Stmt that takes the whole tensor as
      input and a **bulk** output, returned as a bulk :class:`SymArray`.
      The ``×M`` chain never materialises.
    * materialised symbolic ``lhs`` (RF cells, e.g. geometry atoms) →
      offload to a per-cell-atom Stmt and return the ndarray, as before
      (consumed-per-cell intermediates like the φ-jet stay materialised).
    """
    rhs = np.asarray(rhs)
    if isinstance(lhs, SymArray) and lhs._bulk is not None:
        if program is None:
            return np.einsum(spec, np.asarray(lhs), rhs)
        op = EinsumOp.from_rhs(spec, rhs)
        [out] = program.emit_stmt(
            op, [lhs], [OutSpec(name, out_shape)],
            note=f"runtime_einsum_{name}", bulk=True,
        )
        return out  # bulk SymArray — keep the chain deferred
    lhs = np.asarray(lhs)
    if program is None or lhs.dtype != object:
        return np.einsum(spec, lhs, rhs)
    # Budget-zero retention: keep the φ-jet contraction symbolic over its object
    # cells (vertex / DoF atoms) so the parameterization comes through built from
    # vertices, instead of offloading the LHS to a numeric Stmt.
    if not program.budget.defer_phi_jet:
        return np.einsum(spec, lhs, rhs)
    sa = SymArray(lhs, program=program)
    op = EinsumOp.from_rhs(spec, rhs)
    [intermediate] = program.emit_stmt(
        op,
        [sa],
        [OutSpec(name, out_shape)],
        note=f"runtime_einsum_{name}",
        bulk=False,
    )
    return np.asarray(intermediate.cells)


def _cell_size(cell: Cell) -> int:
    """Approximate symbolic-cost measure for a cell — total monomial count."""
    if not isinstance(cell, RationalFunction):
        return 0
    try:
        return len(list(cell._num.terms())) + len(list(cell._den.terms()))
    except Exception:
        return 0


def cells_use_only_stmt_atoms(arr: np.ndarray, env: SymbolEnv) -> bool:
    """Report whether every symbolic cell of ``arr`` is built only from runtime atoms.

    Such cells are opaque at build time — their value exists only once
    the producing :class:`Stmt` runs — so deferring a computation over
    them to another numeric Stmt loses NO polynomial visibility.  Cells
    carrying any model generator (``vertex`` / ``point`` / ``coeff``
    provenance) fail the gate: downstream consumers may rely on exact
    rational structure in those, so callers must keep the symbolic path.

    Conservative on shared rings: the check walks the cell's ring
    generators, so a cell living in a ring that *declares* a model
    symbol fails even if its numerator never uses it.  Numeric cells
    pass vacuously; non-RationalFunction symbolic cells (e.g. sympy
    exprs) fail.
    """
    arr = np.asarray(arr)
    if arr.dtype != object:
        return True
    for cell in arr.reshape(-1) if arr.shape else [arr[()]]:
        if cell is None:
            return False
        if isinstance(cell, RationalFunction):
            for g in cell.gens:
                try:
                    if env.of(g).kind != "stmt_out":
                        return False
                except KeyError:
                    return False
            continue
        if isinstance(cell, (int, float, complex)) or (
            isinstance(cell, np.generic) and np.isscalar(cell)
        ):
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# Multi-operand symbolic einsum offload
# ---------------------------------------------------------------------------

_EINSUM_BAG_THRESHOLD_DEFAULT = 64


def _einsum_bag_threshold() -> int:
    raw = os.environ.get("CHARTLIB_EINSUM_BAG_THRESHOLD")
    if raw is None:
        return _EINSUM_BAG_THRESHOLD_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return _EINSUM_BAG_THRESHOLD_DEFAULT


# Contraction paths depend only on (spec, optimize, operand shapes), never on values, but
# ``np.einsum(..., optimize=True)`` recomputes the path on EVERY call — ~28 µs of pure
# planning that dominated symbolic-Vandermonde builds (hundreds of thousands of identical
# small einsums at high degree). Cache the path by that key and pass it back: numpy
# then skips the planning and runs the SAME contraction order ⇒ bit-identical results.
_EINSUM_PATH_CACHE: dict[tuple[str, object, tuple[tuple[int, ...], ...]], Any] = {}


def _cached_einsum(spec: str, operands: tuple[np.ndarray, ...], optimize: object) -> np.ndarray:
    if optimize is False:
        return np.einsum(spec, *operands, optimize=False)
    key = (spec, optimize, tuple(o.shape for o in operands))
    path = _EINSUM_PATH_CACHE.get(key)
    if path is None:
        if len(_EINSUM_PATH_CACHE) > 8192:               # bounded; distinct keys are few per element
            _EINSUM_PATH_CACHE.clear()
        path, _ = np.einsum_path(spec, *operands, optimize=optimize)
        _EINSUM_PATH_CACHE[key] = path
    return np.einsum(spec, *operands, optimize=path)


@dataclass(frozen=True)
class EinsumStmtOp:
    """Stmt-compatible einsum over an arbitrary number of symbolic operands.

    Generalisation of :class:`EinsumOp`: where ``EinsumOp`` captures
    exactly one numeric RHS into ``rhs_bytes``, this op captures
    nothing — every operand is an input :class:`SymArrayRef` on the
    :class:`Stmt`, so ``spec`` alone identifies the contraction.

    Used when ≥2 operands are symbolic (i.e., object-dtype cells
    flowing through an einsum after the §7.10.6 freeze rule).  For
    the one-symbolic-one-numeric case prefer :class:`EinsumOp` so the
    numeric RHS is embedded in the Stmt's hash.

    Hashable: ``spec`` + ``optimize`` form a small frozen dataclass
    consistent with :class:`Program.fingerprint` (slice C).
    """

    spec: str
    optimize: bool = True

    def __call__(self, *operands: np.ndarray) -> np.ndarray:
        return _cached_einsum(self.spec, tuple(np.asarray(op) for op in operands), self.optimize)


@dataclass(frozen=True)
class SwitchOp:
    """Stmt-compatible runtime switch: pick a branch by integer scrutinee.

    The Stmt's inputs are ``(scrutinee_int, branch_0, branch_1, …)``;
    at run time the op returns ``branch[int(scrutinee_int)]``.

    Used to lower :meth:`SymbolicInterpreter.select_x` over an
    :class:`IntAtom` scrutinee: when ``select_x`` is called with an
    IntAtom, it eagerly evaluates each branch (so any RF arithmetic
    happens up-front), then emits a single :class:`Stmt` whose ``fn``
    is this op.  The output is a fresh :class:`SymArray` of atoms
    with the shape of one branch.

    Hashable / frozen so :class:`Program.fingerprint` (slice C) can
    cache compiled forms.  Stateless — the branch ordering matches
    the canonical orientation order of the entity type at the call
    site.
    """

    n_branches: int

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        """Evaluate the op on concrete numeric operands."""
        if len(args) != 1 + self.n_branches:
            raise ValueError(
                f"SwitchOp: expected 1 scrutinee + {self.n_branches} "
                f"branches; got {len(args)} args"
            )
        scrutinee = int(args[0])
        if not 0 <= scrutinee < self.n_branches:
            raise IndexError(
                f"SwitchOp: scrutinee {scrutinee} out of range "
                f"[0, {self.n_branches})"
            )
        return np.asarray(args[1 + scrutinee])


# ---------------------------------------------------------------------------
# The op union — polyarray's CLOSED ``Stmt.fn`` vocabulary
# ---------------------------------------------------------------------------
#
# ``Stmt.fn`` itself is deliberately OPEN (see :data:`StmtOp`): a front end above
# polyarray may put its own op class, a ``vmap`` closure, or a plain callable there, and
# a sub-:class:`Program` is not an op at all.  What IS closed is the set of ops
# **polyarray owns** — the frozen dataclasses defined above.  :data:`StmtFn` names that
# set so a pass over it can be written as an exhaustive ``match`` finished by
# ``typing.assert_never``.
#
# This is load-bearing: an op left out of a pass is opaque to it, which degrades
# silently — a folded constant stays symbolic, a mask stays dense, a certificate stalls
# at zero — rather than raising.  Naming the union makes the omission a mypy error.
#
# Rule for a new op: add it to :data:`StmtFn`, then run mypy — it will name every pass
# that must now state a decision.  Deliberately not handling an op is fine, but it must
# be an explicit arm that says why.
StmtFn: TypeAlias = Union[
    # linalg / scalar deferrals
    DetOp, InvOp, PinvOp, SolveOp, SqrtOp, AbsOp, SignOp, SvdOp, GSvdOp, QrOp,
    InvTransposeOp, ComposeViaStdOp, SqrtSpdOp, RankOp, MetricOrthonormalOp,
    SinvFullOp, GSvdFullOp,
    # contraction / structural
    TensordotOp, MoveaxisOp, EinsumOp, EinsumStmtOp, TransposeOp, ReshapeOp,
    HStackOp, ColStackOp, ConcatOp, AddOp, ScaleOp, ScaleByOp, AxisLenOp,
    BlockDiagOp, BlockRepeatOp, DynBlockRepeatOp, KronOp, KronFreeOp,
    FirstColsOp, LastColsOp, ProjectOp, EmbedOp,
    # constants / shape-derived sizes (DimAtom sources)
    ConstOp, EyeOp, DynEyeOp, DynZerosOp, DynEyeTensorOp, ProdShapeOp, SumShapeOp,
    SumDimOp, ProdDimOp, ScaleAxisDimOp, MulAxisDimOp, CompRankOp,
    # capture / guard / control flow
    IdentityOp, AssertOp, SwitchOp, CallOp, WhileOp,
]

#: The same vocabulary as a runtime ``isinstance`` tuple — derived from :data:`StmtFn`,
#: so the two can never drift.
STMT_FN_OPS: tuple[type, ...] = get_args(StmtFn)

#: Everything a :attr:`Stmt.fn` may hold: one of polyarray's own ops, a nested
#: :class:`Program`, a front-end op or plain callable, or ``None`` for a statement that
#: exists only to splice rational expressions.  Narrow it with :func:`is_builtin_op`.
StmtOp: TypeAlias = Union[StmtFn, "Program", Callable[..., Any], None]


def is_builtin_op(fn: StmtOp) -> TypeGuard[StmtFn]:
    """Return whether ``fn`` is one of polyarray's own ops (:data:`StmtFn`).

    The narrowing gate in front of an exhaustive ``match``: everything else a
    ``Stmt.fn`` may hold — a sub-:class:`Program`, a ``vmap`` closure, a front-end op
    class, a plain callable, ``None`` — is genuinely open and must be handled *before*
    this check, not inside the match.

    Parameters
    ----------
    fn
        The value held by a statement's ``fn`` slot.

    Returns
    -------
    bool
        True when ``fn`` is a member of the closed :data:`StmtFn` union.
    """
    return isinstance(fn, STMT_FN_OPS)


def _einsum_int_to_str(args: Sequence[np.ndarray | Sequence[int]]) -> tuple[str, list[np.ndarray]]:
    """Convert ``np.einsum`` integer-label form to ``(spec_string, operands)``.

    Mirror of ``np.einsum``'s signature: ``args`` is the alternating
    sequence ``[operand_0, labels_0, operand_1, labels_1, …]`` with an
    optional trailing ``output_labels`` list.  Each ``labels`` is a
    list of ints; an ellipsis is the literal ``Ellipsis`` object in
    that list per the numpy convention.

    Returns a string-form spec and the operand list — the same call
    is then ``np.einsum(spec, *operands, …)``.
    """
    operands: list[np.ndarray] = []
    label_lists: list[list[Any]] = []  # Any: each label is int or Ellipsis
    output_labels: list[Any] | None = None  # Any: int or Ellipsis

    i = 0
    while i < len(args):
        if i + 1 < len(args) and isinstance(args[i + 1], (list, tuple)):
            operands.append(args[i])
            label_lists.append(list(args[i + 1]))
            i += 2
        else:
            output_labels = list(args[i])
            i += 1

    # Build a char map from int labels.  Ellipsis is special: it
    # renders as ``...`` in the string form and is *not* a normal
    # label.
    all_ints: set[int] = set()
    for ll in label_lists:
        for n in ll:
            if n is not Ellipsis:
                all_ints.add(int(n))
    if output_labels is not None:
        for n in output_labels:
            if n is not Ellipsis:
                all_ints.add(int(n))
    if len(all_ints) > 26:
        raise ValueError(
            f"too many distinct int labels ({len(all_ints)}) for string-form spec; "
            "increase the alphabet or fall back to int-form einsum"
        )
    int_to_char = {n: chr(ord("a") + i) for i, n in enumerate(sorted(all_ints))}

    def _render(labels: list[Any]) -> str:
        return "".join(
            "..." if c is Ellipsis else int_to_char[int(c)] for c in labels
        )

    input_str = ",".join(_render(ll) for ll in label_lists)
    if output_labels is None:
        return input_str, operands
    return f"{input_str}->{_render(output_labels)}", operands


def _einsum_parse_spec(spec: str) -> tuple[list[str], str | None]:
    """Split ``"ij,jk->ik"`` (or implicit-output form) into input + output sides.

    Returns ``(input_parts, output_part)`` where ``output_part`` is
    ``None`` for implicit-output specs.
    """
    if "->" in spec:
        lhs, rhs = spec.split("->", 1)
        return lhs.split(","), rhs
    return spec.split(","), None


def _op_shape(o: SymArray | np.ndarray) -> tuple[int, ...]:
    """Operand shape without materialising a bulk SymArray (reads placeholder)."""
    if isinstance(o, SymArray):
        return tuple(o.shape)
    return tuple(np.asarray(o).shape)


def _einsum_label_dims(
    input_parts: Sequence[str], operands: Sequence[SymArray | np.ndarray],
) -> dict[str, int]:
    """Map each non-ellipsis label in the spec to the operand axis size it carries.

    Operands may be ``SymArray`` (including BULK ones): shapes are read through
    :func:`_op_shape`, which answers from the placeholder and never materialises cells.
    """
    label_dim: dict[str, int] = {}
    for part, arr in zip(input_parts, operands):
        shp = _op_shape(arr)
        if "..." in part:
            pre, post = part.split("...")
        else:
            pre, post = part, ""
        ell_count = len(shp) - len(pre) - len(post)
        for i, c in enumerate(pre):
            label_dim[c] = shp[i]
        for i, c in enumerate(post):
            label_dim[c] = shp[len(pre) + ell_count + i]
    return label_dim


def _estimate_einsum_output_terms(
    spec: str, operands: Sequence[np.ndarray],
) -> int:
    """Upper-bound per-output-cell monomial count after the contraction.

    The bound is ``(prod of contracted-axis sizes) × (prod of input
    cell complexities)``.  For the post-§7.10.6 case where every
    input is a frozen atom, the operand complexity factor collapses
    to 1 and the bound is just the contracted-axis product.
    """
    input_parts, output_part = _einsum_parse_spec(spec)
    label_dim = _einsum_label_dims(input_parts, operands)
    all_input_labels = {c for part in input_parts for c in part if c != "."}
    if output_part is not None:
        output_labels = {c for c in output_part if c != "."}
    else:
        from collections import Counter as _Counter
        counts = _Counter(c for part in input_parts for c in part if c != ".")
        output_labels = {l for l, n in counts.items() if n == 1}
    contracted_labels = all_input_labels - output_labels
    contracted_size = 1
    for c in contracted_labels:
        contracted_size *= label_dim[c]

    operand_complexity = 1
    for arr in operands:
        a = np.asarray(arr)
        if a.dtype != object:
            continue
        max_t = 1
        for cell in a.reshape(-1):
            t = _cell_size(cell)
            if t > max_t:
                max_t = t
        op_t = 1 if max_t <= 2 else max_t
        operand_complexity *= op_t

    return contracted_size * operand_complexity


def _einsum_output_shape(
    spec: str, operands: Sequence[SymArray | np.ndarray],
) -> tuple[int, ...]:
    """Compute the output shape of ``np.einsum(spec, *operands)``.

    Handles explicit-output (``"ij,jk->ik"``) and implicit-output
    (``"ij,jk"``) specs, plus ellipsis-bearing parts.  Ellipsis dims
    are taken from whichever operand carries them; ellipses across
    operands must be consistent (numpy broadcasting rules — slice 1
    doesn't reproduce that, just trusts the operand that has the
    fullest ellipsis).
    """
    input_parts, output_part = _einsum_parse_spec(spec)
    label_dim = _einsum_label_dims(input_parts, operands)
    ell_shape: tuple[int, ...] = ()
    for part, arr in zip(input_parts, operands):
        if "..." in part:
            shp = _op_shape(arr)
            pre, post = part.split("...")
            ell_count = len(shp) - len(pre) - len(post)
            cand = tuple(shp[len(pre):len(pre) + ell_count])
            if len(cand) > len(ell_shape):
                ell_shape = cand
    if output_part is None:
        from collections import Counter as _Counter
        counts = _Counter(c for part in input_parts for c in part if c != ".")
        output_letters = sorted(l for l, n in counts.items() if n == 1)
        has_ellipsis = any("..." in p for p in input_parts)
        out_shape = tuple(label_dim[c] for c in output_letters)
        return ell_shape + out_shape if has_ellipsis else out_shape
    if "..." in output_part:
        pre, post = output_part.split("...")
        return (
            tuple(label_dim[c] for c in pre)
            + ell_shape
            + tuple(label_dim[c] for c in post)
        )
    return tuple(label_dim[c] for c in output_part)


def dispatch_einsum(
    *args: str | SymArray | np.ndarray,
    program: Program | None,
    name: str = "einsum",
    optimize: bool = True,
    force_defer: bool = False,
) -> np.ndarray | SymArray:
    """``np.einsum``-compatible signature that dispatches through the offload.

    Accepts the string-form ``(spec, *operands)`` or the integer-label
    form ``(op0, labels0, op1, labels1, …, [output_labels])`` —
    same shapes as :func:`np.einsum`.  Normalises to string-form via
    :func:`_einsum_int_to_str` then forwards to
    :func:`runtime_einsum_multi`.  When ``program`` is ``None`` or all
    operands are numeric, this is equivalent to bare ``np.einsum``.
    """
    if len(args) == 0:
        raise ValueError("dispatch_einsum requires at least one argument")
    if isinstance(args[0], str):
        spec = args[0]
        operands: list[Any] = list(args[1:])
    else:
        spec, operands = _einsum_int_to_str(list(args))
    out_shape = _einsum_output_shape(spec, operands)
    return runtime_einsum_multi(
        spec, *operands,
        program=program,
        out_shape=out_shape,
        name=name,
        force_defer=force_defer,
    )


def runtime_einsum_multi(
    spec: str,
    *operands: np.ndarray | SymArray,
    program: Program | None,
    out_shape: tuple,
    name: str = "einsum_multi",
    force_defer: bool = False,
) -> np.ndarray | SymArray:
    """Multi-operand symbolic einsum with size-aware materialisation.

    Estimates the per-output-cell monomial count from the contracted
    axes and input cell complexities.  If the bound is at most
    :data:`CHARTLIB_EINSUM_BAG_THRESHOLD` (default 64), runs
    ``np.einsum`` over the object-dtype operands directly (RF path —
    symbolic structure preserved).  Otherwise emits an
    :class:`EinsumStmtOp` :class:`Stmt` whose outputs are fresh atom
    RFs of shape ``out_shape``.

    ``force_defer=True`` skips the size estimate + eager short-circuit
    for the object-dtype, program-present, NON-bulk case, so the
    contraction ALWAYS emits a fresh-atom :class:`EinsumStmtOp`
    regardless of the estimated size (used when a caller must keep an
    operand symbolic and deferred — e.g. the symbolic-``K`` recombine
    lane — rather than densifying it into ``np.einsum``).  It does NOT
    affect the bulk branch (already force-defers) nor the all-numeric /
    no-program fallthrough (cannot defer without a program).

    All-numeric operands and no-program calls fall through to bare
    ``np.einsum`` (mirrors :func:`runtime_einsum`).

    Bulk-aware: if any operand is a bulk :class:`SymArray` (a deferred whole
    tensor), the contraction offloads-keeps-bulk — every operand flows as a
    Stmt input (a bulk one via its ``_bulk`` handle, a whole-tensor input)
    and the output is bulk.  The size estimate is NEVER run on a bulk
    operand (it would read placeholder cells / force materialisation); bulk
    presence forces the offload branch directly.
    """
    if program is not None and any(
        isinstance(o, SymArray) and o._bulk is not None for o in operands
    ):
        op = EinsumStmtOp(spec=spec)
        refs: list[Any] = []
        for o in operands:
            if isinstance(o, SymArray):
                refs.append(o)  # emit_stmt wraps -> SymArrayRef (carries _bulk)
            else:
                refs.append(SymArray(np.asarray(o), program=program))
        [out] = program.emit_stmt(
            op, refs, [OutSpec(name, out_shape)],
            note=f"runtime_einsum_multi_{name}", bulk=True,
        )
        return out  # bulk SymArray — chain stays deferred

    arrs = [np.asarray(o) for o in operands]
    any_object = any(a.dtype == object for a in arrs)
    if program is None or not any_object:
        # Through the path cache, not `np.einsum(..., optimize=True)` directly: the ORDER is
        # unchanged (the cache stores and replays the very path numpy would have planned), so
        # this is bit-identical — it only stops re-planning it on every call.
        return _cached_einsum(spec, tuple(arrs), True)

    if not force_defer:
        bound = _estimate_einsum_output_terms(spec, arrs)
        # Threshold from the budget when set; None == legacy env-aware default.
        bt = program.budget.einsum_bag_threshold
        threshold = _einsum_bag_threshold() if bt is None else bt
        if bound <= threshold:
            # The under-threshold "bag" lane — the hottest of the build-time einsums, since it
            # fires on every small symbolic contraction. Same path-cache reasoning as above.
            return _cached_einsum(spec, tuple(arrs), True)

    op = EinsumStmtOp(spec=spec)
    refs = [SymArray(a, program=program) for a in arrs]
    [intermediate] = program.emit_stmt(
        op,
        refs,
        [OutSpec(name, out_shape)],
        note=f"runtime_einsum_multi_{name}",
        bulk=False,
    )
    return np.asarray(intermediate.cells)


def freeze_array_bulk(
    arr: np.ndarray | SymArray,
    *,
    program: Program | None,
    name: str = "frozen",
) -> np.ndarray | SymArray:
    """Bulk variant: emit a single :class:`Stmt` for the whole tensor.

    Equivalent in run-time semantics to :func:`freeze_array` but
    amortises emit-stmt overhead — for tensors with many cells (e.g.
    a Lagrange ``source_ref[k]`` shape ``(M, N, d, d)`` with
    ``N=125``), per-cell freeze costs ~1 ms × N cells in Python
    overhead.  ``freeze_array_bulk`` allocates a single Stmt whose
    output is a SymArray of fresh atoms with the same shape; at
    :meth:`Program.run` time the whole tensor is evaluated in one
    pass.

    Parameters
    ----------
    arr
        The tensor to freeze, as a raw ndarray or as a :class:`SymArray`.
    program
        The program the freeze Stmt is emitted into.  ``None`` passes ``arr`` through unchanged,
        as does a budget-zero program (``budget.freeze`` False) — then the cells stay symbolic
        (retained) rather than being captured as fresh atoms.
    name
        Output name of the emitted Stmt.

    Returns
    -------
    np.ndarray or SymArray
        CARRIER-PRESERVING: a :class:`SymArray` in gives a ``SymArray`` back (riding
        ``program``); a raw ndarray in gives cells back.  So a caller that already threads a
        ``SymArray`` never has to unwrap ``.cells`` to freeze it — that unwrap drops the owning
        ``Program`` and is exactly what the ``SYM-CELLS-UNWRAP`` rule forbids in the consumer
        repos.  Numeric / no-program / float-dtype arrays pass through unchanged.

    Raises
    ------
    ValueError
        When ``arr`` is a ``SymArray`` riding a *different* program.  Such an array is REFUSED
        rather than silently re-homed: its cells may name that program's Stmt outputs, and
        relabelling them onto ``program`` strands the Stmts that produce them (the "generator has
        no binding" failure).  Use :meth:`Program.graft` first, which brings those Stmts along.
    """
    if program is None or not program.budget.freeze:
        return arr
    sym_in = isinstance(arr, SymArray)
    if isinstance(arr, SymArray) and arr.program is not None and arr.program is not program:
        raise ValueError(
            "freeze_array_bulk: `arr` rides a different Program — freezing would re-home its "
            "cells and strand the Stmts producing their generators. Bring it over with "
            "`program.graft(arr)` first.")
    cells = np.asarray(arr)
    if cells.dtype != object:
        return arr if sym_in else cells
    sa = SymArray(cells, program=program)
    [intermediate] = program.emit_stmt(
        IdentityOp(),
        [sa],
        [OutSpec(name, cells.shape)],
        note=f"freeze_bulk_{name}",
        bulk=False,  # freeze materialises by design — bulk would be churn
    )
    return intermediate if sym_in else np.asarray(intermediate.cells)


def freeze_array(
    arr: np.ndarray,
    *,
    program: Program | None,
    threshold: int = 20,
    name: str = "frozen",
) -> np.ndarray:
    """Cap symbolic cell complexity by introducing intermediate atoms.

    For each cell of ``arr`` (object dtype), if its total monomial count
    exceeds ``threshold``, emit an :class:`IdentityOp` Stmt that
    captures the cell's value as a fresh atom; replace the cell with
    that atom.  Numeric / float / scalar arrays pass through unchanged.

    The freeze keeps subsequent symbolic arithmetic small.  At
    evaluation time, the captured cell is computed once (against the
    bound vertex/DoF inputs) and substituted for the atom — same final
    answer, much smaller intermediate expressions.

    Returns the (possibly modified) ndarray; the original is not
    mutated.  Budget-zero (``budget.freeze`` False) disables freezing — cells
    stay symbolic (retained).
    """
    if program is None or not program.budget.freeze:
        return arr
    arr = np.asarray(arr)
    if arr.dtype != object:
        return arr
    flat = arr.reshape(-1)
    needs_freeze = [(i, c) for i, c in enumerate(flat) if _cell_size(c) > threshold]
    if not needs_freeze:
        return arr
    out = arr.copy()
    out_flat = out.reshape(-1)
    op = IdentityOp()
    for i, cell in needs_freeze:
        cell_arr = np.empty((), dtype=object)
        cell_arr[()] = cell
        sa = SymArray(cell_arr, program=program)
        [intermediate] = program.emit_stmt(
            op,
            [sa],
            [OutSpec(f"{name}_{i}", ())],
            note=f"freeze_{name}_{i}",
            bulk=False,  # freeze materialises by design — bulk would be churn
        )
        out_flat[i] = intermediate.cells.item()
    return out


def unpack(sa: SymArray | np.ndarray, *, program: Program | None = None) -> SymArray:
    """Materialise a bulk :class:`SymArray` into per-cell atom RFs.

    A *bulk* SymArray (``_bulk`` set, produced by ``emit_stmt(bulk=True)``)
    carries no per-cell symbolic structure — it is a deferred whole tensor.
    A consumer that needs per-cell rational arithmetic (the keep-structure
    einsum branch, RF matmul/det/inverse) calls ``unpack`` first: it emits a
    single :class:`IdentityOp` Stmt that, at run time, scatters the bulk
    tensor into freshly-allocated per-cell atoms (``bulk=False`` output).

    Idempotent: the materialised result is memoised on the bulk array, so
    repeated unpacks reuse the same atoms (one unpack Stmt per bulk array).
    Non-bulk inputs (already materialised, or a bare ndarray) pass through.
    """
    if not isinstance(sa, SymArray) or sa._bulk is None:
        return sa
    if sa._unpacked is not None:
        return sa._unpacked
    prog = program if program is not None else sa.program
    if prog is None:
        raise ValueError("unpack requires a program-bound bulk SymArray")
    [mat] = prog.emit_stmt(
        IdentityOp(),
        [sa],
        [OutSpec(sa.name or "unpacked", sa._bulk.shape)],
        note="unpack",
        bulk=False,
    )
    sa._unpacked = mat
    return mat


# ---------------------------------------------------------------------------
# Cell-array helpers
# ---------------------------------------------------------------------------

def _to_cells(x: SymArray | npt.ArrayLike) -> np.ndarray:
    """Coerce a SymArray / ndarray / RF / scalar to a cell ndarray."""
    if isinstance(x, SymArray):
        return x.cells
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, RationalFunction):
        arr = np.empty((), dtype=object)
        arr[()] = x
        return arr
    return np.asarray(x)


def _ensure_object(arr: np.ndarray) -> np.ndarray:
    """Cast a cell ndarray to ``dtype=object`` (no-op if already object)."""
    if arr.dtype == object:
        return arr
    return arr.astype(object)


def _matmul_cells(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Object-dtype matmul that short-circuits on structural-zero cells."""
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    n, k = A.shape
    k2, p = B.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {A.shape} @ {B.shape}")
    out = np.empty((n, p), dtype=object)
    for i in range(n):
        for j in range(p):
            acc: Cell = 0.0
            for r in range(k):
                a = A[i, r]
                b = B[r, j]
                if simple_zero(a) or simple_zero(b):
                    continue
                acc = acc + a * b
            out[i, j] = acc
    if squeeze:
        return out.reshape(-1)
    return out


def _matvec_cells(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Object-dtype matvec that short-circuits on structural-zero cells."""
    if A.ndim == 1:
        A = A.reshape(1, -1)
    n, k = A.shape
    if v.ndim != 1 or v.shape[0] != k:
        raise ValueError(f"matvec shape mismatch: {A.shape} . {v.shape}")
    out = np.empty(n, dtype=object)
    for i in range(n):
        acc: Cell = 0.0
        for r in range(k):
            a = A[i, r]
            b = v[r]
            if simple_zero(a) or simple_zero(b):
                continue
            acc = acc + a * b
        out[i] = acc
    return out


def _flatten_values_loose(
    bindings: Mapping[str, np.ndarray | float],
) -> dict[str, float]:
    """Loose flattening for standalone-SymArray ``evaluate`` calls.

    Each value is flattened to per-generator names ``f"{name}_{k}"``;
    scalar inputs are also bound under the bare name.  Used when a
    SymArray has no owning :class:`Program` and the caller passes
    bindings directly (no input-shape validation).
    """
    flat: dict[str, float] = {}
    for n, value in bindings.items():
        v = np.asarray(value, dtype=float).ravel()
        for k, vk in enumerate(v):
            flat[f"{n}_{k}"] = float(vk)
        if v.size == 1:
            flat.setdefault(n, float(v[0]))
    return flat


def _stmt_out_label(prefix: str, out_name: str, idx: tuple[int, ...]) -> str:
    """Build a generator label for a Stmt-output cell.

    Whitespace in the prefix or output name is collapsed to ``_`` so the
    resulting label is a single sympy ``Symbol`` (sympy's ``PolyRing``
    parser splits whitespace-separated names into multiple symbols).
    """
    safe_prefix = "_".join(prefix.split())
    safe_name = "_".join(out_name.split())
    if not idx:
        return f"{safe_prefix}.{safe_name}"
    return f"{safe_prefix}.{safe_name}_" + "_".join(str(i) for i in idx)


# ---------------------------------------------------------------------------
# Stmt
# ---------------------------------------------------------------------------

@dataclass
class Stmt:
    """An imperative IR statement.

    ``fn`` is the actual Python callable invoked at run time
    (``numpy.linalg.qr``, ``numpy.sqrt``, or another :class:`Program`).
    For slice A every Stmt-fn is either ``None`` (a no-op statement
    used purely to splice rational expressions — *this is rare*) or a
    sub-:class:`Program`.

    ``in_`` is a list of :class:`Ref` objects describing where this Stmt's
    inputs come from.

    ``out`` is a tuple of :class:`SymArray` objects — one per fn return, in
    fn-return order.  Each SymArray's cells are fresh atom RFs
    allocated on the program's :class:`SymbolEnv`.

    ``note`` is a human-readable provenance string.

    ``provenance`` is an optional *structured* :class:`Provenance` describing
    what produced this Stmt (e.g. a lowering front-end's algebra-centric node /
    basis-choice record).  It is **purely descriptive metadata**: it is never
    read by :meth:`run`/evaluation, so a program with ``provenance=None`` (the
    default) is byte-identical to one before this field existed.  Preserved
    across :meth:`Program.copy` (hence through ``partial_eval``).

    ``inline`` controls sub-Program composition (§1.3).  When ``True``
    the sub-program's rational outputs are spliced into the parent at
    construction time and this Stmt is dropped from the parent's
    statement list — see :func:`call_subprogram_inline`.
    """

    fn: Callable[..., Any] | Program | None
    in_: tuple[Ref, ...]
    out: tuple[SymArray, ...]
    note: str = ""
    provenance: Provenance | None = None
    inline: bool = False


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

class Program:
    """A symbolic-interpreter compilation unit.

    A :class:`Program` is built by passing it inputs (each a
    :class:`SymInput` describing a shape and provenance), then
    populating its outputs as :class:`SymArray`s.  Statements are
    appended via :meth:`add_stmt`.

    The :class:`SymbolEnv` is owned by the program and shared across
    every cell uttered against it.

    Plain-Python execution: :meth:`run` walks ``statements`` in order,
    builds a numeric binding for every Stmt-out cell, then evaluates
    each output's :class:`RationalFunction` cells against the
    accumulated bindings.

    :meth:`copy` returns an independent copy of the program (deep-clones
    state, rebinds back-references on internal SymArrays) so that a
    sub-:class:`SymbolicInterpreter` can extend without disturbing the
    original.
    """

    def __init__(
        self,
        name: str = "anon",
        inputs: Sequence[SymInput] = (),
        env: SymbolEnv | None = None,
        budget: SymbolicBudget | None = None,
    ) -> None:
        self.name = name
        self.env: SymbolEnv = env if env is not None else SymbolEnv()
        self.inputs: tuple[SymInput, ...] = tuple(inputs)
        self.input_arrays: dict[str, SymArray] = {}
        for inp in self.inputs:
            if is_dynamic(inp.shape):
                # Stage C: a dynamic (DimAtom-sized) input cannot enumerate
                # per-cell atoms — allocate one whole-tensor bulk handle whose
                # symbolic shape carries the DimAtom axes (resolved at run
                # time from the provided array; see build_runtime_bindings).
                cells = np.empty((), dtype=object)
                sa = SymArray(cells, program=self, name=inp.name)
                sa._bulk = BulkOut(name=inp.name, shape=tuple(inp.shape))
                self.input_arrays[inp.name] = sa
            else:
                cells = allocate_input(self.env, inp)
                self.input_arrays[inp.name] = SymArray(
                    cells, program=self, name=inp.name
                )
        self.statements: list[Stmt] = []
        self.outputs: dict[str, SymArray] = {}
        # An explicit `budget=` always wins; otherwise consult the ambient `budget_override`
        # context (oracle's sparsity mask requests the inline-covariant lane), else the default.
        self.budget: SymbolicBudget = (
            budget if budget is not None
            else (_BUDGET_OVERRIDE.get() or SymbolicBudget())
        )
        # IntAtom registry — keyed by name; resolved at run time by
        # reading the matching entry out of the values dict.  Distinct
        # from the rational ``inputs`` table because IntAtoms do not
        # allocate RF generators.
        self.int_atoms: dict[str, IntAtom] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def input(self, name: str) -> SymArray:
        """Return the :class:`SymArray` for the named input."""
        if name not in self.input_arrays:
            raise KeyError(f"unknown input {name!r}; have {list(self.input_arrays)}")
        return self.input_arrays[name]

    # Backward-compatible alias for callers that still want a bare ndarray.
    @property
    def input_cells(self) -> dict[str, np.ndarray]:
        """Return each input's cell array, by input name."""
        return {k: sa.cells for k, sa in self.input_arrays.items()}

    def declare_int_atom(self, name: str, domain: range) -> IntAtom:
        """Register an :class:`IntAtom` on the program.

        Re-declaring the same name with the same domain is a no-op
        (returns the existing atom).  A clash with a different domain
        raises ``ValueError`` — orientation atoms allocated by
        :meth:`SymbolicInterpreter.int_atom` get a deterministic
        domain per cell-type, so a clash indicates a real bug.
        """
        existing = self.int_atoms.get(name)
        if existing is not None:
            if tuple(existing.domain) != tuple(domain):
                raise ValueError(
                    f"IntAtom {name!r} already declared with domain "
                    f"{existing.domain!r}; new domain {domain!r}"
                )
            return existing
        atom = IntAtom(name=name, domain=domain)
        self.int_atoms[name] = atom
        return atom

    def add_output(self, name: str, cells: SymArray | np.ndarray) -> SymArray:
        """Register a named output and return the :class:`SymArray`."""
        if name in self.outputs:
            raise ValueError(f"output {name!r} already declared on {self.name!r}")
        sa = SymArray(cells, program=self, name=name)
        self.outputs[name] = sa
        return sa

    def add_stmt(self, stmt: Stmt) -> int:
        """Append a pre-built :class:`Stmt` and return its index.

        Most callers should use :meth:`emit_stmt` (which allocates atom
        RFs for each output and builds the Stmt for you).  This raw
        appender is for hand-built Stmts (e.g. ``fn=None`` splices).
        """
        idx = len(self.statements)
        self.statements.append(stmt)
        return idx

    def graft(self, foreign: SymArray, *, note: str = "graft") -> SymArray:
        """Re-home a value COMPUTED ON ANOTHER program onto ``self`` — WITH its producing Stmts.

        ``foreign`` is a :class:`SymArray` whose cells are :class:`RationalFunction`s over
        ``foreign.program``'s generators — some of which may be **Stmt outputs** (deferred numeric ops:
        matrix inverse / matmul, a grassmann-lowered sub-Program, …), not just shared input atoms. A bare
        ``SymArray(foreign.cells, program=self)`` relabel carries the CELLS but strands those producing
        Stmts on the by-product program, so a later lowering of ``self`` references generators ``self``
        never produces ("no binding"). :meth:`graft` instead emits ``foreign.program`` as ONE sub-Program
        :class:`Stmt` of ``self`` (the same mechanism a grassmann-lowered DOF body uses to compose onto the
        shared sampling program), whose fresh atom outputs carry ``foreign``'s value on ``self`` — so the
        whole foreign Stmt DAG runs when ``self`` runs/lowers, and the fresh outputs are dedup'd by
        ``self``'s env (several grafts of like-named by-product programs do not collide).

        ``foreign.program`` and ``self`` must be built over the SAME shared symbolic inputs (e.g. the
        cell-vertex atoms of one ``make_symbolic_geometry`` reference): the sub-Program is fed ``foreign``'s
        own input atoms relabeled onto ``self``, so ``self`` resolves their leaf generators (``V_0_0``…) by
        name when it lowers — a generator ``self`` cannot produce surfaces downstream as an unbound generator.
        A program-less ``foreign`` (numeric, or already inline on ``self``) needs no Stmts and is a plain
        relabel. Idempotent on ``self``'s own arrays (``foreign.program is self`` ⇒ returned unchanged).
        """
        src = foreign.program
        if src is self:
            return foreign
        if src is None:
            return SymArray(foreign.cells, program=self)
        # A one-output VIEW of `src` (share env / inputs / Stmts / budget — never mutate the possibly-cached
        # source), whose single output is `foreign`'s cells; `_invoke` runs it and returns that output.
        view = Program.__new__(Program)
        view.name = src.name
        view.env = src.env
        view.inputs = src.inputs
        view.input_arrays = src.input_arrays
        view.statements = src.statements
        view.budget = src.budget
        view.int_atoms = src.int_atoms
        view.outputs = {"grafted": SymArray(foreign.cells, program=view, name="grafted")}
        # Operands feeding the sub-Program's inputs (by position): each is `src`'s own input atoms RELABELED
        # onto `self`. The two programs are BUILT OVER THE SAME shared symbolic inputs (e.g. the cell-vertex
        # atoms of one ``make_symbolic_geometry`` reference), so `src`'s input generators (``V_0_0``…) are the
        # SAME names `self` produces — `self` resolves them by name when it lowers, without `self` needing to
        # DECLARE those inputs itself (its own value kernel binds them at compile). A `src` input whose
        # generators `self` cannot produce is a genuine mismatch, caught downstream as an unbound generator.
        operands = [SymArray(src.input(inp.name).cells, program=self) for inp in src.inputs]
        (out,) = self.emit_stmt(
            view, operands, [OutSpec(note, tuple(foreign.cells.shape))], note=note, bulk=False)
        return out

    def emit_stmt(
        self,
        fn: StmtOp,
        in_refs: Sequence[Ref | SymArray],
        out_specs: Sequence[OutSpec],
        note: str = "",
        bulk: bool = True,
        provenance: Provenance | None = None,
    ) -> tuple[SymArray, ...]:
        """Append an imperative :class:`Stmt` and return its outputs.

        Allocates one fresh atom RF per cell of every declared output
        shape on ``self.env`` (provenance ``"stmt_out"``), builds the
        Stmt, appends it to :attr:`statements`, and returns a tuple of
        :class:`SymArray` objects back-referenced to ``self``.  Downstream
        rational arithmetic uses the returned cells as ordinary
        RationalFunction generators — that is how rational-valued code
        *after* an imperative statement keeps being rational.

        Inputs that are bare :class:`SymArray` objects are auto-wrapped in
        :class:`SymArrayRef`.  Inputs that are already :class:`Ref` objects
        flow through verbatim.

        Raises :class:`ValueError` if ``fn`` is ``None`` (use
        :meth:`add_stmt` for splice-only ``fn=None`` Stmts).
        """
        if fn is None:
            raise ValueError(
                "emit_stmt requires fn; use add_stmt for fn=None splices"
            )
        stmt_idx = len(self.statements)
        label_prefix = note or f"stmt{stmt_idx}"

        wrapped_refs: list[Ref] = []
        for ref in in_refs:
            if isinstance(ref, SymArray):
                wrapped_refs.append(SymArrayRef(ref))
            else:
                wrapped_refs.append(ref)

        out_arrays: list[SymArray] = []
        for spec in out_specs:
            dynamic = is_dynamic(spec.shape)
            # A dynamic shape (an axis sized by a runtime ``DimAtom``) cannot
            # enumerate per-cell atoms — force the whole-tensor bulk lane.
            if bulk or dynamic:
                # One whole-tensor handle, no per-cell atoms.  Placeholder
                # cells (None) exist only for ``.shape``; they are never read
                # per-cell — eval routes through ``_bulk`` (a per-cell read
                # would raise on None, catching any missed bulk-aware path).
                name = self.env.fresh(
                    label_prefix + ":" + spec.name,
                    Provenance(
                        kind="stmt_out", origin=stmt_idx, index=(),
                        label=label_prefix + ":" + spec.name,
                    ),
                )
                # A dynamic shape has no concrete placeholder array (a
                # ``DimAtom`` axis is unknown until run time); use a 0-d
                # object placeholder — ``.shape`` on a dynamic bulk array is
                # the symbolic ``_bulk.shape`` (read via the override below).
                if dynamic:
                    cells = np.empty((), dtype=object)
                else:
                    cells = np.empty(spec.shape, dtype=object)
                sa = SymArray(cells, program=self, name=spec.name)
                sa._bulk = BulkOut(name=name, shape=tuple(spec.shape))
                out_arrays.append(sa)
                continue
            cells = np.empty(spec.shape, dtype=object)
            indices = np.ndindex(*spec.shape) if spec.shape else [()]
            for cell_idx in indices:
                label = _stmt_out_label(label_prefix, spec.name, cell_idx)
                atom_name = self.env.fresh(
                    label,
                    Provenance(
                        kind="stmt_out",
                        origin=stmt_idx,
                        index=cell_idx,
                        label=label,
                    ),
                )
                rf = RationalFunction.atom(atom_name)
                if spec.shape:
                    cells[cell_idx] = rf
                else:
                    cells[()] = rf
            out_arrays.append(SymArray(cells, program=self, name=spec.name))

        stmt = Stmt(
            fn=fn,
            in_=tuple(wrapped_refs),
            out=tuple(out_arrays),
            note=note,
            provenance=provenance,
        )
        self.statements.append(stmt)
        return tuple(out_arrays)

    # ------------------------------------------------------------------
    # Copy (for sub-interpreters)
    # ------------------------------------------------------------------

    def copy(self) -> Program:
        """Return an independent copy with all SymArrays rebound to the new program.

        Inputs (frozen :class:`SymInput` descriptors) and ring data are
        shared; everything else is copied so the new program can be
        extended without touching ``self``.
        """
        new = Program.__new__(Program)
        new.name = self.name
        new.env = self.env.copy()
        new.inputs = self.inputs
        new.input_arrays = {
            k: sa._rebind(new) for k, sa in self.input_arrays.items()
        }
        new.statements = [
            Stmt(
                fn=s.fn,
                in_=s.in_,
                out=tuple(o._rebind(new) for o in s.out),
                note=s.note,
                provenance=s.provenance,
                inline=s.inline,
            )
            for s in self.statements
        ]
        new.outputs = {k: sa._rebind(new) for k, sa in self.outputs.items()}
        new.budget = self.budget
        new.int_atoms = dict(self.int_atoms)
        return new

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, values: Mapping[str, npt.ArrayLike]) -> dict[str, np.ndarray]:
        """Evaluate the program against numeric ``values``.

        ``values`` is keyed by input *name*; each entry is a numeric
        array (or scalar) matching the input's declared shape.  The
        return is a dict from output name to ``np.ndarray`` (numeric)
        in insertion order.
        """
        bindings = self.build_runtime_bindings(values)
        out: dict[str, np.ndarray] = {}
        for name, sa in self.outputs.items():
            out[name] = self._eval_symarray(sa, bindings)
        return out

    def build_runtime_bindings(
        self, values: Mapping[str, npt.ArrayLike], *, only: Iterable[int] | None = None,
    ) -> dict[str, float]:
        """Run statements and return the full per-atom bindings dict.

        Equivalent to the bindings dict :meth:`run` constructs before
        evaluating outputs — exposed so a :class:`SymArray` can resolve
        its Stmt-output atoms via :meth:`SymArray.evaluate` without
        going through full :meth:`run`.

        ``only`` (a set of statement indices) runs ONLY those statements,
        in program order — the dependency-cone lane
        (:func:`~polyarray.simplify.evaluate_cone`): a caller that wants one
        SymArray's value without executing unrelated statements (e.g. a
        singular op elsewhere in the program) passes the target's cone.
        ``None`` (the default) runs every statement, byte-identical to before.
        """
        bindings = self._bindings_from_values(values)
        # Runtime dimension table: maps a producing output ``(stmt_idx,
        # out_idx)`` to its concrete int value, so a later dynamic
        # ``OutSpec.shape`` carrying a :class:`DimAtom` (whose ``source`` is
        # that output) resolves to a concrete shape.  Built ONLY for programs
        # that actually carry a dynamic shape; static programs pass ``None``
        # so ``_run_stmt`` skips all per-output dim bookkeeping (B5) — the
        # static path stays byte-identical AND free of per-output overhead.
        # The `only=` (cone) lane always uses the table: it may run mid-build (from
        # `evaluate_cone` inside a partially-built program), so it must NOT read/write the
        # whole-program `_has_dynamic_shape` memo (a partial program would cache a stale
        # answer that the later full run then trusts) — and a cone can carry a dynamic
        # (DimAtom-sized) output regardless.
        dim_bindings: dict[DimSource, int] | None = (
            {} if (only is not None or self._has_dynamic_shape()) else None
        )
        # Stage C: before walking statements, bind each dynamic INPUT.  Read
        # the provided array's actual shape, record each DimAtom axis into
        # dim_bindings (keyed by the DimAtom's ("in", name, axis) source so a
        # later dynamic OutSpec.shape / sibling resolves against it), and bind
        # the whole tensor as a bulk value.  Concrete axes are validated.
        if dim_bindings is not None:
            for inp in self.inputs:
                if not is_dynamic(inp.shape):
                    continue
                arr = np.asarray(values[inp.name], dtype=float)
                if arr.ndim != len(inp.shape):
                    raise ValueError(
                        f"dynamic input {inp.name!r}: expected {len(inp.shape)} "
                        f"axes; got array of ndim {arr.ndim} (shape {arr.shape})"
                    )
                for axis, dim in enumerate(inp.shape):
                    if isinstance(dim, DimAtom):
                        # An input axis sourced from a prior Stmt's output (e.g. an FFS-typed
                        # input Var sized by the factorization's canonical rank atom,
                        # ``source[0] == "stmt"``) is NOT bound from the fed array's shape:
                        # the producing Stmt records it later in ``_run_stmt``.  Binding it
                        # here would shadow that canonical value with the fed shape and defeat
                        # the rank-consistency check.  The fed axis length is still validated
                        # against the resolved atom by the ``rank_eq`` AssertOp downstream.
                        # An ``("in", …)`` atom is genuinely input-sourced — bind it here.
                        if dim.source and dim.source[0] == "stmt":
                            continue
                        dim_bindings[dim.source] = int(arr.shape[axis])
                    elif int(dim) != int(arr.shape[axis]):
                        raise ValueError(
                            f"dynamic input {inp.name!r}: axis {axis} expected "
                            f"{int(dim)}; got {int(arr.shape[axis])}"
                        )
                bindings[inp.name] = arr  # type: ignore[assignment]
        order = range(len(self.statements)) if only is None else sorted(only)
        for stmt_idx in order:
            self._run_stmt(stmt_idx, self.statements[stmt_idx], bindings, dim_bindings)
        return bindings

    def _has_dynamic_shape(self) -> bool:
        """Report whether any statement output is a dynamic, ``DimAtom``-sized bulk array.

        Such a program needs the run-time dimension-binding table; a purely static program
        does not, and skipping the table keeps its execution path free of that cost.

        Memoized: ``run`` (hence this) is called once per slice / sub-Program
        recursion, so the O(#stmts) scan must not repeat per call. Statements
        are fixed by run time, so the answer is cached on the instance.
        """
        cached = getattr(self, "_dynamic_shape_memo", None)
        if cached is None:
            cached = any(
                is_dynamic(inp.shape) for inp in self.inputs
            ) or any(
                o._bulk is not None and is_dynamic(o._bulk.shape)
                for stmt in self.statements
                for o in stmt.out
            )
            self._dynamic_shape_memo = cached
        return cached

    # -- helpers --------------------------------------------------------

    def _bindings_from_values(self, values: Mapping[str, npt.ArrayLike]) -> dict[str, float]:
        """Flatten input values into a generator-name → float dict.

        Each *declared* input is validated against its shape and bound
        via the cells' actual generator names (so atom names assigned
        by ``allocate_input``'s fresh suffixing are respected).  Any
        *extra* entry in ``values`` that doesn't match a declared input
        is treated as a "geometry-side" input (typically a
        :attr:`Geometry.sym_inputs` entry that flowed in through a
        :class:`SymArrayRef`); those entries are flattened by
        ``f"{name}_{k}"`` per cell of the value, matching the
        convention used by :meth:`SymArray.evaluate`.
        """
        bindings: dict[str, float] = {}
        declared = set()
        for inp in self.inputs:
            declared.add(inp.name)
            if inp.name not in values:
                raise KeyError(f"missing value for input {inp.name!r}")
            # Dynamic (DimAtom-sized) inputs are bulk; their whole-tensor value
            # and DimAtom axis sizes are bound in build_runtime_bindings (C3),
            # not per-cell here.  Mark declared so the geometry-side fallthrough
            # below leaves them alone, and skip the static per-cell binding.
            if is_dynamic(inp.shape):
                continue
            arr = np.asarray(values[inp.name], dtype=float)
            if tuple(arr.shape) != tuple(inp.shape):
                raise ValueError(
                    f"input {inp.name!r}: expected shape {inp.shape}; "
                    f"got {arr.shape}"
                )
            cells = self.input_arrays[inp.name].cells
            for idx in np.ndindex(*inp.shape) if inp.shape else [()]:
                cell = cells[idx] if inp.shape else cells[()]
                if isinstance(cell, RationalFunction):
                    name = cell.gens[0]
                    bindings[name] = float(arr[idx]) if inp.shape else float(arr)
        for atom_name, atom in self.int_atoms.items():
            declared.add(atom_name)
            if atom_name not in values:
                raise KeyError(
                    f"missing value for IntAtom {atom_name!r}"
                )
            raw = values[atom_name]
            try:
                ival = int(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"IntAtom {atom_name!r}: cannot coerce {raw!r} to int"
                ) from exc
            if ival not in atom.domain:
                raise ValueError(
                    f"IntAtom {atom_name!r}: value {ival} outside domain "
                    f"{atom.domain!r}"
                )
            bindings[atom_name] = float(ival)
        for name, value in values.items():
            if name in declared:
                continue
            v = np.asarray(value, dtype=float).ravel()
            for k, vk in enumerate(v):
                bindings[f"{name}_{k}"] = float(vk)
            if v.size == 1:
                bindings.setdefault(name, float(v[0]))
        return bindings

    def _run_stmt(
        self,
        stmt_idx: int,
        stmt: Stmt,
        bindings: dict[str, float],
        dim_bindings: dict[DimSource, int] | None = None,
    ) -> None:
        """Resolve refs, call fn, scatter returns into ``bindings``.

        ``dim_bindings`` (when supplied) records each scalar output's int
        value keyed by ``(stmt_idx, out_idx)`` so a later dynamic shape can
        resolve a :class:`DimAtom` (B4).  ``None`` (the legacy default)
        skips all dim bookkeeping.
        """
        if stmt.fn is None:
            return
        resolved = [self._resolve_ref(r, stmt_idx, bindings) for r in stmt.in_]
        if isinstance(stmt.fn, Program):
            sub_results = stmt.fn.run(
                self._sub_value_map(stmt.fn, resolved)
            )
            outs = list(sub_results.values())
        else:
            results = stmt.fn(*resolved)
            outs = list(results) if isinstance(results, tuple) else [results]
        for k, bound in enumerate(stmt.out):
            self._bind_output(bound, outs[k], bindings, dim_bindings)
            # Record any scalar (0-d) output as a candidate runtime
            # dimension, keyed by its producing output position.  Only
            # 0-d ints can size an axis (the SVD ``rank`` / δ_f output).
            if dim_bindings is not None:
                arr = np.asarray(outs[k])
                if arr.ndim == 0:
                    try:
                        dim_bindings[("stmt", stmt_idx, k)] = int(arr)
                    except (TypeError, ValueError):
                        pass

    def _resolve_ref(
        self,
        ref: Ref,
        stmt_idx: int,
        bindings: dict[str, float],
    ) -> np.ndarray:
        if isinstance(ref, InputRef):
            cells = self.input_arrays[ref.name].cells
            if ref.indices:
                return self._evaluate_cells(np.array(cells[ref.indices]), bindings)
            return self._evaluate_cells(cells, bindings)
        if isinstance(ref, OutputRef):
            if ref.stmt_idx >= stmt_idx:
                raise ValueError(
                    f"OutputRef stmt_idx={ref.stmt_idx} not yet executed "
                    f"(at stmt_idx={stmt_idx})"
                )
            target = self.statements[ref.stmt_idx].out[ref.out_idx]
            arr = self._eval_symarray(target, bindings)
            if ref.indices:
                return arr[ref.indices]
            return arr
        if isinstance(ref, Const):
            return ref.value
        if isinstance(ref, RationalRef):
            return _eval_rf(ref.rf, bindings)
        if isinstance(ref, SymArrayRef):
            if ref._bulk is not None:
                return np.asarray(bindings[ref._bulk.name])
            return self._evaluate_cells(ref.cells, bindings)
        if isinstance(ref, IntAtomRef):
            if ref.name not in bindings:
                raise KeyError(
                    f"IntAtom {ref.name!r} not bound at run time"
                )
            return int(bindings[ref.name])
        raise TypeError(f"unknown Ref kind: {type(ref).__name__}")

    def _sub_value_map(
        self,
        sub: Program,
        resolved_inputs: list[Any],
    ) -> dict[str, np.ndarray]:
        """Build a ``values`` dict for a sub-program's :meth:`run`."""
        if len(resolved_inputs) != len(sub.inputs):
            raise ValueError(
                f"sub-program {sub.name!r} expects {len(sub.inputs)} inputs; "
                f"got {len(resolved_inputs)}"
            )
        return {
            sub_inp.name: np.asarray(val)
            for sub_inp, val in zip(sub.inputs, resolved_inputs)
        }

    def _bind_output(
        self,
        bound: SymArray,
        value: np.ndarray,
        bindings: dict[str, float],
        dim_bindings: dict[DimSource, int] | None = None,
    ) -> None:
        """Bind a Stmt output: whole-tensor for bulk, per-cell atoms otherwise.

        For a *dynamic* bulk output (an axis sized by a :class:`DimAtom`),
        the declared ``_bulk.shape`` is resolved against ``dim_bindings``
        before validating the produced tensor's shape (B4).
        """
        if bound._bulk is not None:
            arr = np.asarray(value)
            if is_dynamic(bound._bulk.shape):
                if dim_bindings is None:
                    raise RuntimeError(
                        f"dynamic bulk output {bound.name!r} requires the run "
                        f"path's dim-binding table"
                    )
                # Admit runtime dims: a not-yet-bound DimAtom in this output's declared shape is
                # bound HERE from the realized array's actual axis size — the produced output IS
                # the runtime dimension (its canonical source), so a consumer can resolve it next.
                for axis, d in enumerate(bound._bulk.shape):
                    if (isinstance(d, DimAtom) and d.source not in dim_bindings
                            and axis < arr.ndim):
                        dim_bindings[d.source] = int(arr.shape[axis])
                expected = _resolve_shape(bound._bulk.shape, dim_bindings)
            else:
                expected = tuple(bound._bulk.shape)
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"bulk Stmt output {bound.name!r}: expected shape "
                    f"{expected}; got {arr.shape}"
                )
            bindings[bound._bulk.name] = arr  # type: ignore[assignment]
        else:
            self._bind_atoms(bound, value, bindings)

    def _eval_symarray(
        self, sa: SymArray, bindings: dict[str, float],
    ) -> np.ndarray:
        """Evaluate a SymArray to a float ndarray (bulk reads the whole tensor)."""
        if sa._bulk is not None:
            return np.asarray(bindings[sa._bulk.name], dtype=float)
        return self._evaluate_cells(sa.cells, bindings)

    def _bind_atoms(
        self,
        bound: SymArray,
        value: np.ndarray,
        bindings: dict[str, float],
    ) -> None:
        """Record numeric values for the atoms of a Stmt-output array."""
        arr = np.asarray(value)
        if tuple(arr.shape) != tuple(bound.cells.shape):
            raise ValueError(
                f"Stmt output {bound.name!r}: expected shape {bound.cells.shape}; "
                f"got {arr.shape}"
            )
        shape = bound.cells.shape
        for idx in np.ndindex(*shape) if shape else [()]:
            cell = bound.cells[idx] if shape else bound.cells[()]
            if isinstance(cell, RationalFunction):
                name = cell.gens[0]
                bindings[name] = float(arr[idx]) if shape else float(arr)

    @staticmethod
    def _evaluate_cells(
        cells: np.ndarray,
        bindings: dict[str, float],
    ) -> np.ndarray:
        """Evaluate an ndarray of cells against ``bindings``."""
        if cells.dtype.kind == "f":
            return cells.astype(float)
        # ``cells`` is object-dtype (or rarely float-dtype, handled above);
        # each scalar element is a Cell — mypy can't see this through ndarray
        # indexing, hence the cast.
        if cells.shape == ():
            return np.asarray(
                _eval_cell(cells[()], bindings),  # type: ignore[arg-type]
                dtype=float,
            )
        out = np.empty(cells.shape, dtype=float)
        for idx in np.ndindex(*cells.shape):
            out[idx] = _eval_cell(cells[idx], bindings)  # type: ignore[arg-type]
        return out


# ---------------------------------------------------------------------------
# Cell-level evaluation
# ---------------------------------------------------------------------------

def _eval_cell(cell: Cell, bindings: dict[str, float]) -> float:
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, RationalFunction):
        return _eval_rf(cell, bindings)
    raise TypeError(f"cannot evaluate cell of type {type(cell).__name__}")


# When set (via :func:`probe_direct_eval`), the program runner evaluates each
# ``RationalFunction`` by DIRECT term-summation (``eval_numeric_direct``) instead of the
# codegen-and-cache ``eval_numeric_fast``. The codegen amortizes over MANY runs of the same
# program, but the structural-mask probe (:func:`polyarray.schur._structural_mask`) runs a
# program only 3× — where compiling every RF (``builtins.compile`` + ``_poly_term_strings``)
# is most of a degree-5 P(T) build for nothing. Values are byte-identical, so
# the probed mask is unchanged. Module-scoped: symbolic builds are single-threaded.
_PROBE_DIRECT_EVAL = False


@contextlib.contextmanager
def probe_direct_eval() -> Iterator[None]:
    """Make the program runner evaluate cells directly, without per-cell codegen.

    For probes run only a handful of times, where compiling an evaluator costs more than it
    saves. See :data:`_PROBE_DIRECT_EVAL`.
    """
    global _PROBE_DIRECT_EVAL
    prev = _PROBE_DIRECT_EVAL
    _PROBE_DIRECT_EVAL = True
    try:
        yield
    finally:
        _PROBE_DIRECT_EVAL = prev


def _eval_rf(rf: RationalFunction, bindings: dict[str, float]) -> float:
    """Numeric eval of a :class:`RationalFunction`.

    Uses :meth:`RationalFunction.eval_numeric_fast` (codegen-and-cache) by default, or
    :meth:`~RationalFunction.eval_numeric_direct` (no per-RF ``compile``) inside a
    :func:`probe_direct_eval` scope — see :data:`_PROBE_DIRECT_EVAL`.  On the rare
    missing-binding path (a programming error) we re-raise with a friendlier message.
    """
    try:
        return (rf.eval_numeric_direct(bindings) if _PROBE_DIRECT_EVAL
                else rf.eval_numeric_fast(bindings))
    except KeyError as exc:
        # Compiled-fast path will raise KeyError for the first missing
        # name; surface the full set of missing gens for diagnostics.
        missing = [n for n in rf.gens if n not in bindings]
        raise KeyError(
            f"RationalFunction.eval: missing bindings for {sorted(missing)!r}"
        ) from exc


# ---------------------------------------------------------------------------
# vmap: batched sub-Program execution
# ---------------------------------------------------------------------------

def vmap(
    body: Program,
    in_axes: int | None | tuple[int | None, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
) -> Callable[..., Any]:
    """Return a callable that batches ``body.run`` along a leading axis.

    The returned callable accepts one positional ndarray per body input
    (in declaration order).  Each input array is sliced along its
    corresponding ``in_axes`` position; the body program is invoked
    once per slice and its outputs are stacked along the corresponding
    ``out_axes`` position.

    Use this as the ``fn`` of a :class:`Stmt` so that an M-element
    per-point computation extends the parent program with one Stmt
    rather than M sub-program calls.  Today this is a Python loop;
    a torch / jax backend can swap in their native ``vmap``.

    ``in_axes`` and ``out_axes`` may be a single int (broadcast to
    every input / output) or a per-input / per-output tuple.  An
    in-axis entry of ``None`` means "broadcast as-is" — the same
    array is fed to every iteration without slicing (used to share a
    static input like vertex coordinates across all M points while
    only ξ varies).  If ``body`` has exactly one output, the callable
    returns a bare ndarray; otherwise it returns a tuple in
    body-output insertion order.
    """
    n_in = len(body.inputs)
    n_out = len(body.outputs)
    if isinstance(in_axes, int) or in_axes is None:
        in_axes_tup: tuple[int | None, ...] = (in_axes,) * n_in
    else:
        in_axes_tup = tuple(in_axes)
    if isinstance(out_axes, int):
        out_axes_tup = (out_axes,) * n_out
    else:
        out_axes_tup = tuple(out_axes)
    if len(in_axes_tup) != n_in:
        raise ValueError(
            f"vmap: in_axes length {len(in_axes_tup)} does not match "
            f"body input count {n_in}"
        )
    if len(out_axes_tup) != n_out:
        raise ValueError(
            f"vmap: out_axes length {len(out_axes_tup)} does not match "
            f"body output count {n_out}"
        )
    if all(ax is None for ax in in_axes_tup):
        raise ValueError("vmap: at least one input must be batched (axis ≠ None)")
    out_names = tuple(body.outputs.keys())

    def fn(*args: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        if len(args) != n_in:
            raise ValueError(
                f"vmap({body.name}): expected {n_in} args; got {len(args)}"
            )
        arrs = [np.asarray(a) for a in args]
        batch_sizes: list[int] = []
        norm_axes: list[int | None] = []
        for arr, ax in zip(arrs, in_axes_tup):
            if ax is None:
                norm_axes.append(None)
                continue
            ax_norm = ax if ax >= 0 else arr.ndim + ax
            batch_sizes.append(arr.shape[ax_norm])
            norm_axes.append(ax_norm)
        if len(set(batch_sizes)) != 1:
            raise ValueError(
                f"vmap({body.name}): inconsistent batch sizes {batch_sizes}"
            )
        M = batch_sizes[0]
        per_slice: list[dict[str, np.ndarray]] = []
        for i in range(M):
            slice_values: dict[str, np.ndarray] = {}
            for inp, arr, ax in zip(body.inputs, arrs, norm_axes):
                if ax is None:
                    slice_values[inp.name] = arr
                    continue
                idx: list[Any] = [slice(None)] * arr.ndim
                idx[ax] = i
                slice_values[inp.name] = arr[tuple(idx)]
            per_slice.append(body.run(slice_values))
        stacked: list[np.ndarray] = []
        for k, name in enumerate(out_names):
            slices = [s[name] for s in per_slice]
            stacked.append(np.stack(slices, axis=out_axes_tup[k]))
        if n_out == 1:
            return stacked[0]
        return tuple(stacked)

    fn.__qualname__ = f"vmap({body.name})"
    fn.__name__ = f"vmap_{body.name}"
    # Additive, descriptive metadata so introspecting tools (e.g.
    # to_numpy_source) can recover the batched body and the normalised
    # in/out axis tuples without spelunking the closure cells.  These are
    # never read by ``run``/evaluation — purely informational.
    fn._vmap_body = body
    fn._in_axes = in_axes_tup
    fn._out_axes = out_axes_tup
    return fn


# ---------------------------------------------------------------------------
# Sub-program inline-mode helper
# ---------------------------------------------------------------------------

def call_subprogram_inline(
    parent: Program,
    sub: Program,
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Splice a sub-program's rational outputs into ``parent``'s symbol space.

    For every output of ``sub``, this returns a fresh ``ndarray`` of
    cells whose :class:`RationalFunction`s have been *substituted*: any
    sub-input atoms are replaced by the corresponding parent cell from
    ``inputs``.  No new generators are allocated; no Stmt is emitted.
    """
    sub_to_parent: dict[str, Cell] = {}
    for sub_inp in sub.inputs:
        if sub_inp.name not in inputs:
            raise KeyError(f"sub-program {sub.name!r} missing input {sub_inp.name!r}")
        parent_arr = np.asarray(inputs[sub_inp.name], dtype=object)
        if tuple(parent_arr.shape) != tuple(sub_inp.shape):
            raise ValueError(
                f"sub input {sub_inp.name!r}: expected shape {sub_inp.shape}; "
                f"got {parent_arr.shape}"
            )
        sub_cells = sub.input_arrays[sub_inp.name].cells
        for idx in np.ndindex(*sub_inp.shape) if sub_inp.shape else [()]:
            sub_cell = sub_cells[idx] if sub_inp.shape else sub_cells[()]
            if not isinstance(sub_cell, RationalFunction):
                continue
            atom_name = sub_cell.gens[0]
            parent_cell = parent_arr[idx] if sub_inp.shape else parent_arr[()]
            sub_to_parent[atom_name] = parent_cell  # type: ignore[assignment]

    if sub.statements:
        raise NotImplementedError(
            "inline-mode composition does not yet handle sub-programs with "
            "imperative statements; call as opaque mode instead"
        )
    result: dict[str, np.ndarray] = {}
    for name, sa in sub.outputs.items():
        shape = sa.cells.shape
        out_cells = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            cell = sa.cells[idx] if shape else sa.cells[()]
            new_cell = _substitute_cell(cell, sub_to_parent)
            if shape:
                out_cells[idx] = new_cell
            else:
                out_cells[()] = new_cell
        result[name] = out_cells
    return result


def _substitute_cell(cell: Cell, mapping: Mapping[str, Cell]) -> Cell:
    """Substitute a sub-cell against a sub-atom-name → parent-cell mapping."""
    if isinstance(cell, (int, float)):
        return cell
    if not isinstance(cell, RationalFunction):
        raise TypeError(f"cannot substitute cell of type {type(cell).__name__}")
    used = set(cell.gens)
    needed = used & set(mapping.keys())
    if not needed:
        return cell
    return _rebuild_rf(cell, mapping)


def _rebuild_rf(rf: RationalFunction, mapping: Mapping[str, Cell]) -> Cell:
    """Rebuild ``rf`` by substituting sub-atom names with parent cells."""
    num_val = _rebuild_poly(rf.num, mapping)
    den_val = _rebuild_poly(rf.den, mapping)
    return _safe_div(num_val, den_val)


def _rebuild_poly(poly: Poly, mapping: Mapping[str, str]) -> Poly:
    """Reconstruct a sympy poly under a generator-name substitution."""
    from .rational import _ring_names

    names = _ring_names(poly.ring)
    gens = poly.ring.gens
    acc: Any = 0.0
    for monom, coeff in poly.terms():
        term: Any = float(coeff.numerator) / float(coeff.denominator) if hasattr(coeff, "numerator") else float(coeff)
        for k, exp in enumerate(monom):
            if exp == 0:
                continue
            name = names[k]
            if name in mapping:
                base: Any = mapping[name]
            else:
                base = RationalFunction(gens[k], poly.ring.one)
            for _ in range(exp):
                term = term * base
        acc = acc + term
    return acc


def _is_numeric(v: object) -> bool:
    return isinstance(v, (int, float))


def _safe_div(a: Cell, b: Cell) -> Cell:
    if _is_numeric(a) and _is_numeric(b):
        return float(a) / float(b)  # type: ignore[arg-type]
    if isinstance(b, (int, float)) and float(b) == 1.0:
        return a
    return _to_rf_or_num(a) / _to_rf_or_num(b)


def _to_rf_or_num(v: Cell) -> RationalFunction:
    if isinstance(v, RationalFunction):
        return v
    if isinstance(v, (int, float)):
        return RationalFunction.constant(v)
    raise TypeError(f"cannot coerce {type(v).__name__} for division")


__all__ = [
    "AbsOp",
    "AddOp",
    "AxisLenOp",
    "BlockDiagOp",
    "BlockRepeatOp",
    "BulkOut",
    "CallOp",
    "Cell",
    "ColStackOp",
    "CompRankOp",
    "ComposeViaStdOp",
    "ConcatOp",
    "Const",
    "ConstOp",
    "DetOp",
    "DynBlockRepeatOp",
    "DynEyeOp",
    "DynEyeTensorOp",
    "DynZerosOp",
    "EmbedOp",
    "EyeOp",
    "FirstColsOp",
    "GSvdFullOp",
    "GSvdOp",
    "HStackOp",
    "InputRef",
    "IntAtom",
    "IntAtomRef",
    "InvOp",
    "InvTransposeOp",
    "KronFreeOp",
    "KronOp",
    "LastColsOp",
    "MetricOrthonormalOp",
    "MoveaxisOp",
    "MulAxisDimOp",
    "OutSpec",
    "OutputRef",
    "PinvOp",
    "ProdDimOp",
    "ProdShapeOp",
    "Program",
    "ProjectOp",
    "Provenance",
    "ProvenanceKind",
    "RankOp",
    "RationalRef",
    "Ref",
    "ReshapeOp",
    "ScaleAxisDimOp",
    "ScaleByOp",
    "ScaleOp",
    "SignOp",
    "SinvFullOp",
    "SolveOp",
    "SqrtOp",
    "SqrtSpdOp",
    "Stmt",
    "SumDimOp",
    "SumShapeOp",
    "SwitchOp",
    "SymArray",
    "SymArrayRef",
    "SymInput",
    "SymbolEnv",
    "TensordotOp",
    "TransposeOp",
    "WhileOp",
    "allocate_input",
    "call_subprogram_inline",
    "cells_sparsity",
    "unpack",
    "vmap",
]
