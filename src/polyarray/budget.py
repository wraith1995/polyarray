"""`SimplifyBudget` — the post-build control surface for the `simplify` pass.


Where ``simplify.specialize`` runs the *unconditional floor* (numeric folding +
dead-stmt elimination), this module adds
the **discretionary band**: a frozen :class:`SimplifyBudget` policy and a
post-pass :func:`_apply_budget` that moderates how far the residual *symbolic*
structure is collapsed (toward numeric atoms) or kept/exposed (legible symbols).

Every branch is **exactness-preserving** — collapse / intermediate-extraction
capture a cell for run-time evaluation via an :class:`IdentityOp` Stmt (the same
primitive ``forward.partial_eval`` and ``freeze_array`` use), so for all budgets
``B`` and inputs ``x``::

    specialize(p, budget=B).run(x) == p.run(x)         # form changes, value never does

Structural decisions only (monomial mass / denominator degree / provenance kind)
— never the numeric *value* of a cell.  The input program is never mutated; we
operate on the fresh program ``specialize`` already built and only ever append
fresh capture Stmts / rebuild output cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .ir import IdentityOp, OutSpec, Program, SymArray, _cell_size
from .rational import RationalFunction, _total_degree


# ---------------------------------------------------------------------------
# The policy object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimplifyBudget:
    """Post-build policy moderating the ``simplify`` collapse <-> expose band.

    Knobs:

    * ``max_cell_mass`` — per-cell monomial ceiling.  A symbolic output cell
      whose ``_cell_size`` mass exceeds this is *collapsed* to a fresh atom via
      an :class:`IdentityOp` Stmt.  ``None`` disables per-cell collapse;
      ``0`` collapses every symbolic cell.  This subsumes
      ``forward.partial_eval``'s ``max_cell_size``.
    * ``total_mass`` — whole-program ceiling.  If the program's total symbolic
      output mass exceeds this, the heaviest cells are greedily collapsed until
      it is under budget (a global complement to the per-cell ceiling).
    * ``expose`` — re-expansion policy (``"never"`` / ``"if_under_budget"`` /
      ``"always"``).  ``"never"`` (default) does no re-expansion.  The genuine
      inverse of ``partial_eval`` (re-expanding a deferred *numeric* Stmt back
      to symbolic) is not implemented in this slice; the other two modes are a
      **documented best-effort no-op** — they never do anything unsound, they
      simply do not re-expand.  Denominator extraction below is the implemented
      "expose more symbols" lever.
    * ``den_degree_max`` — a cell whose *denominator* total degree exceeds this
      is extracted as a fresh **named intermediate** symbol (clean / capture),
      exposing a symbol rather than collapsing to a number.
    * ``keep_provenance`` — never collapse / extract a cell while it still
      carries a generator of one of these provenance kinds (e.g.
      ``{"vertex","coeff"}``).  The post-build analogue of "keep the
      parameterization".
    * ``inherit_freeze`` — read the source ``Program.budget`` freeze threshold
      to cap post-processing growth consistently with how the program was built.
      Reserved for first cut (see :func:`_apply_budget`); it never relaxes
      exactness.
    """

    # --- collapse side (mass ceilings) ---
    max_cell_mass: int | None = None
    total_mass: int | None = None
    # --- expose side (legibility) ---
    expose: Literal["never", "if_under_budget", "always"] = "never"
    den_degree_max: int = 1
    keep_provenance: frozenset[str] = field(default_factory=frozenset)
    # --- consistency with how it was built ---
    inherit_freeze: bool = True

    # -- presets (end-state correspondence in) ------------------

    @classmethod
    def none(cls) -> SimplifyBudget:
        """All knobs off: numeric fold + DCE only, symbolic structure untouched."""
        return cls()

    # ``legacy`` is a spelling alias for ``none`` (the always-safe core).
    legacy = none

    @classmethod
    def numeric_only(cls) -> SimplifyBudget:
        """Collapse every symbolic cell to a numeric/atom (``max_cell_mass=0``).

        On the symbolic-cell set this is equivalent to
        ``partial_eval(p, max_cell_size=0)``.
        """
        return cls(max_cell_mass=0)

    @classmethod
    def balanced(cls, m: int) -> SimplifyBudget:
        """Keep cheap symbolic structure, collapse the heavy tail above ``m``."""
        return cls(max_cell_mass=m, expose="if_under_budget")

    @classmethod
    def expose_symbols(cls, keep: frozenset[str] | set[str] = frozenset()) -> SimplifyBudget:
        """Maximize visible symbols; protect ``keep`` provenance kinds.

        ``max_cell_mass=None`` (no per-cell collapse), ``expose="always"``,
        bounded only by ``total_mass`` if the caller also sets it.
        """
        return cls(max_cell_mass=None, expose="always", keep_provenance=frozenset(keep))


# ---------------------------------------------------------------------------
# Cell predicates (structural only)
# ---------------------------------------------------------------------------

def _den_degree(cell: RationalFunction) -> int:
    """Total degree of the cell's denominator (``-1`` for the zero/constant)."""
    return _total_degree(cell.den)


def _captured_atom_gen(cell: RationalFunction) -> str | None:
    """If ``cell`` is a *bare single-generator atom* (numerator a single
    degree-1 monomial, denominator a constant), return that generator's name;
    else ``None``.

    A bare atom is already as collapsed as a capture could make it, so we never
    re-collapse one — this is what makes collapse **idempotent**.
    """
    if _total_degree(cell.den) != 0:
        return None
    try:
        monoms = list(cell.num.monoms())
    except Exception:
        return None
    if len(monoms) != 1:
        return None
    m = tuple(int(e) for e in monoms[0])
    if sum(m) != 1:
        return None
    pos = m.index(1)
    gens = cell.gens
    if pos >= len(gens):
        return None
    return gens[pos]


def _is_captured_atom(cell: RationalFunction, program: Program) -> bool:
    """True iff ``cell`` is a bare atom over a ``stmt_out`` generator — i.e. an
    already-captured intermediate.  Re-collapsing it is a no-op, so we skip it
    (idempotence).  Bare atoms over *model* inputs (vertex/coeff/...) are NOT
    skipped: those are genuine symbolic cells a collapse should still capture
    (matching ``partial_eval``)."""
    gen = _captured_atom_gen(cell)
    if gen is None:
        return False
    try:
        return program.env.of(gen).kind == "stmt_out"
    except Exception:
        return False


def _protected(cell: RationalFunction, program: Program, keep: frozenset[str]) -> bool:
    """True iff ``cell`` carries a generator whose provenance kind is in
    ``keep`` — such a cell is never collapsed nor extracted."""
    if not keep:
        return False
    for g in cell.gens:
        try:
            kind = program.env.of(g).kind
        except Exception:
            kind = "extern"
        if kind in keep:
            return True
    return False


# ---------------------------------------------------------------------------
# The capture primitive (faithful replica of forward.partial_eval's move)
# ---------------------------------------------------------------------------

def _capture(
    out: Program,
    op: IdentityOp,
    cell: RationalFunction,
    name: str,
    idx: int,
    *,
    prefix: str,
    note: str,
) -> RationalFunction:
    """Capture ``cell`` as a fresh atom via an ``IdentityOp`` Stmt and return
    the captured atom RF.  This is exactly the mechanism in
    ``forward.partial_eval`` / ``freeze_array``: the (heavy) cell is evaluated
    once at ``run`` time and bound to the small atom downstream."""
    cell_arr = np.empty((), dtype=object)
    cell_arr[()] = cell
    [captured] = out.emit_stmt(
        op,
        [SymArray(cell_arr, program=out)],
        [OutSpec(f"{prefix}_{name}_{idx}", ())],
        note=f"{note}_{name}_{idx}",
        bulk=False,
    )
    return captured.cells.item()


# ---------------------------------------------------------------------------
# The moderation procedure
# ---------------------------------------------------------------------------

def _apply_budget(program: Program, budget: SimplifyBudget) -> Program:
    """Apply ``budget`` as a post-pass to an already numeric-folded program.

    Operates on the program's **output** cells (a numeric-folded floor is
    assumed to have already run, so any remaining ``RationalFunction`` cell is
    genuinely symbolic). Per §"The moderation procedure":

    1. per-cell pass: ``keep_provenance`` protection, ``den_degree_max``
       intermediate extraction, ``max_cell_mass`` collapse;
    2. global pass: greedy heaviest-first collapse until ``total_mass``.

    ``inherit_freeze`` is read for first-cut consistency but never relaxes
    exactness; ``expose != "never"`` is a documented best-effort no-op (see
    :class:`SimplifyBudget`).  ``program`` is the fresh program ``specialize``
    built; we append capture Stmts / rebuild output cells on it directly.
    """
    out = program
    op = IdentityOp()

    # --- per-cell pass over every (non-bulk, object-dtype) output array ---
    for name, sa in list(out.outputs.items()):
        if sa._bulk is not None:
            continue
        cells = np.asarray(sa.cells)
        if cells.dtype != object:
            continue
        new_cells = cells.copy()
        flat = new_cells.reshape(-1)
        for i, cell in enumerate(flat):
            if not isinstance(cell, RationalFunction):
                continue
            if _is_captured_atom(cell, out):
                continue  # already a bare captured atom -> idempotent no-op
            if _protected(cell, out, budget.keep_provenance):
                continue  # carries a kept provenance kind -> never touch
            # den_degree_max: extract a heavy-denominator cell as a NAMED
            # intermediate symbol (expose), preferring an exact cancellation.
            if _den_degree(cell) > budget.den_degree_max:
                cleaned = _try_clean(cell)
                if _den_degree(cleaned) <= budget.den_degree_max:
                    flat[i] = cleaned  # cancellation simplified it -> keep symbolic
                else:
                    flat[i] = _capture(
                        out, op, cleaned, name, i,
                        prefix="iv", note="den_extract",
                    )
                continue
            # expose: "never"/best-effort no-op (no re-expansion in this slice).
            # max_cell_mass collapse (subsumes partial_eval's max_cell_size).
            if budget.max_cell_mass is not None and _cell_size(cell) > budget.max_cell_mass:
                flat[i] = _capture(
                    out, op, cell, name, i,
                    prefix="pe", note="partial_eval",
                )
        out.outputs[name] = SymArray(new_cells, program=out, name=name)

    # --- global pass: greedy heaviest-first collapse until under total_mass ---
    if budget.total_mass is not None:
        _enforce_total_mass(out, op, budget)

    return out


def _try_clean(cell: RationalFunction) -> RationalFunction:
    """Best-effort exact gcd cancellation; returns ``cell`` unchanged on any
    backend failure (exactness-preserving either way)."""
    try:
        return cell.clean()
    except Exception:
        return cell


def _collapsible(cell: object, program: Program, budget: SimplifyBudget) -> bool:
    """Eligible for global total_mass collapse: a symbolic, non-protected,
    not-already-captured cell."""
    if not isinstance(cell, RationalFunction):
        return False
    if _is_captured_atom(cell, program):
        return False
    if _protected(cell, program, budget.keep_provenance):
        return False
    return True


def _enforce_total_mass(out: Program, op: IdentityOp, budget: SimplifyBudget) -> None:
    """Greedily collapse the heaviest eligible output cell until the program's
    total symbolic output mass is at or under ``budget.total_mass``."""
    assert budget.total_mass is not None
    while True:
        total = 0
        heaviest: tuple[int, str, int] | None = None  # (mass, name, flat_idx)
        for name, sa in out.outputs.items():
            if sa._bulk is not None:
                continue
            cells = np.asarray(sa.cells)
            if cells.dtype != object:
                continue
            for i, cell in enumerate(cells.reshape(-1)):
                if not isinstance(cell, RationalFunction):
                    continue
                total += _cell_size(cell)
                if _collapsible(cell, out, budget):
                    mass = _cell_size(cell)
                    if heaviest is None or mass > heaviest[0]:
                        heaviest = (mass, name, i)
        if total <= budget.total_mass or heaviest is None:
            return
        _, name, idx = heaviest
        sa = out.outputs[name]
        new_cells = np.asarray(sa.cells).copy()
        flat = new_cells.reshape(-1)
        flat[idx] = _capture(
            out, op, flat[idx], name, idx,
            prefix="tm", note="total_mass",
        )
        out.outputs[name] = SymArray(new_cells, program=out, name=name)
