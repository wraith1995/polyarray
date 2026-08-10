"""Post-build analysis and cost-driven partial evaluation of a :class:`Program`.

Building a program under a :class:`~polyarray.ir.SymbolicBudget` decides how much
symbolic structure it carries. This module is the half that runs afterwards: look at
what was generated, and reshape it by choice.

* :func:`analyze` walks a Program (and every nested sub-Program / vmap body) and
  reports structure + cost — statement counts, deferral/offload nodes, per-output
  symbolic mass, and the generator-provenance histogram (vertex / point / coeff /
  stmt_out / per_point).  This is the inspection tool the budget machinery exists
  to feed: build big, then decide what matters.

* :func:`partial_eval` is the cost-driven transform: collapse every output cell
  whose symbolic cost exceeds a chosen bound (capturing it as a fresh atom via an
  ``IdentityOp`` Stmt), leaving cheaper cells symbolic.  It is the post-build
  counterpart of the build-time defer-when-over-budget gate, and is
  **exactness-preserving** — the captured cell is evaluated at ``run`` time, so
  ``partial_eval(p).run(x) == p.run(x)`` for all ``x`` — it only changes *form*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import assert_never

import numpy as np

from .ir import (
    Const,
    IdentityOp,
    InputRef,
    IntAtomRef,
    OutSpec,
    OutputRef,
    Program,
    RationalRef,
    Ref,
    SymArray,
    StmtOp,
    SymArrayRef,
    _cell_size,
    is_builtin_op,
)
from .rational import RationalFunction


# ---------------------------------------------------------------------------
# Descent into nested programs (sub-Programs + vmap closure bodies)
# ---------------------------------------------------------------------------

def _body_of(fn: StmtOp) -> Program | None:
    """Pull a per-point body :class:`Program` out of a ``vmap`` closure."""
    for c in getattr(fn, "__closure__", None) or ():
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if isinstance(v, Program):
            return v
    return None


def iter_programs(top: Program) -> list[Program]:
    """Return ``top`` plus every nested program (sub-Program fns and vmap bodies).

    Parameters
    ----------
    top
        The outermost program.

    Returns
    -------
    list of Program
        ``top`` first, then nested programs in statement / discovery order,
        deduplicated by identity.
    """
    seen: dict[int, Program] = {}
    order: list[Program] = []
    stack = [top]
    while stack:
        p = stack.pop(0)
        if id(p) in seen:
            continue
        seen[id(p)] = p
        order.append(p)
        for st in getattr(p, "statements", []):
            sub = st.fn if isinstance(st.fn, Program) else _body_of(st.fn)
            if sub is not None and id(sub) not in seen:
                stack.append(sub)
    return order


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgramRow:
    """Per-program slice of an :class:`IRReport`."""

    name: str
    n_stmts: int
    n_defer: int          # statements whose fn is a typed Op (offload/freeze/...)
    n_output_cells: int
    symbolic_cells: int   # output cells that are RationalFunctions
    total_mass: int       # Σ monomial count over symbolic output cells
    max_cell: int         # largest single output-cell monomial count
    prov_kinds: dict[str, int]  # generator-provenance histogram over output cells
    # OPERAND mass — the real lowering cost.  Every einsum/linalg Stmt defers its
    # OUTPUT to fresh atoms (so ``total_mass`` above reads tiny), but pyab lowers each
    # non-bulk RationalFunction OPERAND cell (``stmt.in_``) by expanding its monomials
    # into the codegen AST.  This is where a degree-d symbolic build actually costs —
    # ``total_mass`` alone hides it (a 9K-output / 8.35M-operand mismatch is invisible in the sum).
    operand_mass: int = 0     # Σ monomial count over non-bulk RF operand cells
    max_operand: int = 0      # largest single operand-cell monomial count
    n_operand_cells: int = 0  # number of symbolic (RF) operand cells


@dataclass(frozen=True)
class IRReport:
    """Structural + cost report over a Program and its nested bodies."""

    rows: tuple[ProgramRow, ...] = field(default_factory=tuple)

    @property
    def n_programs(self) -> int:
        """Number of programs covered, counting ``top`` and every nested body."""
        return len(self.rows)

    @property
    def n_defer(self) -> int:
        """Number of statements whose ``fn`` is a typed op, summed over all programs."""
        return sum(r.n_defer for r in self.rows)

    @property
    def total_mass(self) -> int:
        """Monomial count over symbolic output cells, summed over all programs."""
        return sum(r.total_mass for r in self.rows)

    @property
    def total_operand_mass(self) -> int:
        """Monomial count over non-bulk rational OPERAND cells, over all programs.

        This is the real codegen cost. ``total_mass`` counts only outputs, which
        einsums defer to atoms, so it hides the operand expansion entirely.
        """
        return sum(r.operand_mass for r in self.rows)

    def prov_kinds(self) -> dict[str, int]:
        """Return the generator-provenance histogram aggregated over all programs."""
        out: dict[str, int] = {}
        for r in self.rows:
            for k, v in r.prov_kinds.items():
                out[k] = out.get(k, 0) + v
        return out

    def __str__(self) -> str:
        lines = [
            f"IRReport: {self.n_programs} program(s), "
            f"defer={self.n_defer}, out_mass={self.total_mass}, "
            f"operand_mass={self.total_operand_mass}, "
            f"prov={self.prov_kinds()}",
        ]
        for r in self.rows:
            lines.append(
                f"  {r.name:14} stmts={r.n_stmts:3d} defer={r.n_defer:3d} "
                f"sym_cells={r.symbolic_cells:3d}/{r.n_output_cells:<3d} "
                f"out_mass={r.total_mass:6d} max={r.max_cell:5d}  "
                f"operand_mass={r.operand_mass:8d} max_op={r.max_operand:7d} "
                f"n_op={r.n_operand_cells:5d} prov={r.prov_kinds}"
            )
        return "\n".join(lines)


def _is_op(fn: StmtOp) -> bool:
    """Return whether ``fn`` is a typed IR op rather than a plain callable or program.

    polyarray's own vocabulary answers by type (:data:`~polyarray.ir.StmtFn`). The
    class-name suffix covers ops belonging to a front end above polyarray, which this
    layer must not import and therefore cannot name: the counted quantity — how many
    statements defer to an op — is deliberately open-world.
    """
    return is_builtin_op(fn) or type(fn).__name__.endswith("Op")


def _prov_kind(program: Program, gen: str) -> str:
    """Return the provenance kind of a generator as seen by this program's env.

    A generator not declared in ``program.env`` reports ``"extern"``: a free symbol the
    program receives a binding for at run time, typically a geometry vertex or DoF atom
    living in the geometry's own env. ``"extern"`` therefore reads as "still symbolic
    over an outside input", the opposite of ``"stmt_out"``.
    """
    p = program.env._provenance.get(gen)
    return getattr(p, "kind", "extern") if p is not None else "extern"


def _ref_rf_cells(ref: Ref) -> list[RationalFunction]:
    """Return the rational operand cells a statement input ref carries, bulk excluded.

    Exhaustive over :data:`~polyarray.ir.Ref`, polyarray's other closed sum type: a ref
    kind falling off the end would contribute nothing to ``operand_mass``, which reads
    as "this program is cheap" rather than as a missing case.
    """
    match ref:
        case SymArrayRef():
            if ref._bulk is not None:
                return []
            arr = np.asarray(ref._cells)
            return [c for c in arr.reshape(-1) if isinstance(c, RationalFunction)]
        case RationalRef():
            rf = ref.rf
            return [rf] if isinstance(rf, RationalFunction) else []
        case Const():
            arr = np.asarray(ref.value, dtype=object)
            return [c for c in arr.reshape(-1) if isinstance(c, RationalFunction)]
        case InputRef() | OutputRef() | IntAtomRef():
            # A name, a prior-Stmt handle, a runtime integer selector: no monomials of
            # their own — the mass lives on the array they point at, counted there.
            return []
        case _ as unreachable:
            assert_never(unreachable)


def _row(program: Program) -> ProgramRow:
    """Measure one program's statement, output-cell and operand-cell costs."""
    n_defer = sum(_is_op(st.fn) for st in program.statements)
    n_cells = sym_cells = total_mass = max_cell = 0
    op_mass = max_op = n_op = 0
    prov: dict[str, int] = {}
    for st in program.statements:
        for ref in st.in_:
            for cell in _ref_rf_cells(ref):
                n_op += 1
                sz = _cell_size(cell)
                op_mass += sz
                max_op = max(max_op, sz)
    for sa in program.outputs.values():
        # Read placeholder cells for a bulk output (no per-cell symbols);
        # never force a materialisation here.
        cells = sa._cells if sa._bulk is not None else np.asarray(sa.cells)
        if sa._bulk is not None:
            continue
        for cell in cells.reshape(-1):
            n_cells += 1
            if isinstance(cell, RationalFunction):
                sym_cells += 1
                sz = _cell_size(cell)
                total_mass += sz
                max_cell = max(max_cell, sz)
                for g in cell.gens:
                    k = _prov_kind(program, g)
                    prov[k] = prov.get(k, 0) + 1
    return ProgramRow(
        name=program.name,
        n_stmts=len(program.statements),
        n_defer=n_defer,
        n_output_cells=n_cells,
        symbolic_cells=sym_cells,
        total_mass=total_mass,
        max_cell=max_cell,
        prov_kinds=prov,
        operand_mass=op_mass,
        max_operand=max_op,
        n_operand_cells=n_op,
    )


def analyze(program: Program) -> IRReport:
    """Walk ``program`` and every nested body, returning a structural and cost report.

    Parameters
    ----------
    program
        The outermost program.

    Returns
    -------
    IRReport
        One :class:`ProgramRow` per program, ``program`` first.
    """
    return IRReport(rows=tuple(_row(p) for p in iter_programs(program)))


# ---------------------------------------------------------------------------
# Cost-driven partial evaluation (selective collapse)
# ---------------------------------------------------------------------------

def partial_eval(program: Program, *, max_cell_size: int) -> Program:
    """Return an equivalent Program with over-budget output cells collapsed.

    Every output cell whose monomial count exceeds ``max_cell_size`` is captured
    as a fresh atom via an ``IdentityOp`` :class:`Stmt` (the same primitive as
    ``freeze_array``, but applied here as an explicit post-build transform,
    independent of the program's budget).  Cells at or below the bound stay
    symbolic.

    Exactness-preserving: the captured cell is evaluated against the run-time
    bindings, so for every input ``x`` the returned program's outputs equal the
    original's.

    Parameters
    ----------
    program
        The program to transform; it is copied, never mutated.
    max_cell_size
        Monomial-count ceiling per output cell. ``0`` collapses every symbolic output
        cell to an atom; a bound above the heaviest cell is a no-op.

    Returns
    -------
    Program
        A copy with over-budget cells captured. Only the top program's outputs are
        touched; nested vmap bodies are left intact.
    """
    out = program.copy()
    op = IdentityOp()
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
            if _cell_size(cell) <= max_cell_size:
                continue
            cell_arr = np.empty((), dtype=object)
            cell_arr[()] = cell
            [captured] = out.emit_stmt(
                op,
                [SymArray(cell_arr, program=out)],
                [OutSpec(f"pe_{name}_{i}", ())],
                note=f"partial_eval_{name}_{i}",
                bulk=False,
            )
            flat[i] = captured.cells.item()
        out.outputs[name] = SymArray(new_cells, program=out, name=name)
    return out
