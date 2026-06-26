"""Emit a standalone, runnable numpy ``.py`` source string from a :class:`Program`.

A :class:`polyarray.Program` (built e.g. by grassmann's ``G.compile(term)``)
can be *run* (:meth:`Program.run`) but not serialised to a readable numpy
module.  :func:`to_numpy_source` walks the program and emits a self-contained
function — one parameter per program input (named by the input name), each
statement lowered to the numpy call its typed op models, and the output(s)
assembled from their :class:`RationalFunction` cells — shaped like
matrixcalculus.org's output::

    import numpy as np

    def f(A, x):
        ...
        result = ...
        return result

The emitter mirrors :meth:`Program.run` semantics exactly (the same
ref-resolution + per-cell ``RationalFunction`` evaluation), but emits source
instead of executing.  It reuses :func:`polyarray.rational._poly_to_pyexpr`
(the helper that already renders a polynomial to a Python expression) for the
rational-cell lane, so numerator/denominator rendering is shared with the
runtime ``RationalFunction`` evaluator.

The generated module depends on ``numpy`` only.

Extending to non-polyarray ops
------------------------------
polyarray's own typed ops (``DetOp``/``EinsumStmtOp``/…) are covered by the
built-in registry.  Front-ends that lower onto polyarray with their *own* Stmt
ops (e.g. grassmann's ``_AxisLenOp``) pass an ``op_renderers`` mapping keyed by
the op class name to :func:`to_numpy_source` — keeping polyarray free of any
dependency on the front-end.  An op with no renderer raises a clear
``NotImplementedError`` naming the op.
"""
from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from .ir import (
    AbsOp,
    AssertOp,
    Const,
    DetOp,
    EinsumOp,
    EinsumStmtOp,
    IdentityOp,
    InputRef,
    IntAtomRef,
    InvOp,
    MoveaxisOp,
    OutputRef,
    PinvOp,
    Program,
    QrOp,
    RationalRef,
    SignOp,
    SolveOp,
    SqrtOp,
    SwitchOp,
    SymArray,
    SymArrayRef,
    TensordotOp,
    is_dynamic,
)
from .rational import RationalFunction, _poly_to_pyexpr, _ring_names

__all__ = ["to_numpy_source", "OpRenderer"]

# A renderer maps ``(op, arg_exprs)`` to a single Python expression string that,
# when evaluated, yields the op's result (a tuple for multi-output ops).
OpRenderer = Callable[[Any, "list[str]"], str]


# ---------------------------------------------------------------------------
# Small expression helpers
# ---------------------------------------------------------------------------

def _safe_param_name(name: str) -> str:
    """Return a valid, non-keyword Python identifier for an input ``name``."""
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    sane = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not sane or not sane[0].isalpha() and sane[0] != "_":
        sane = "_" + sane
    return "p_" + sane if keyword.iskeyword(sane) else sane


def _index_suffix(idx: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(i) for i in idx) + "]"


def _const_expr(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"np.array({value.tolist()!r}, dtype=float)"
    return repr(float(value)) if isinstance(value, (int, float)) else repr(value)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class _Emitter:
    def __init__(
        self,
        program: Program,
        func_name: str,
        op_renderers: Mapping[str, OpRenderer] | None,
    ) -> None:
        self.prog = program
        self.func_name = func_name
        self.extra_renderers = dict(op_renderers or {})
        self.lines: list[str] = []
        # generator-name -> Python scalar expression producing its value.
        self.genmap: dict[str, str] = {}
        # bulk binding-name -> Python variable holding the whole tensor.
        self.bulkmap: dict[str, str] = {}
        # (stmt_idx, out_idx) -> Python variable (for OutputRef).
        self.outvar: dict[tuple[int, int], str] = {}
        # input/int-atom name -> parameter identifier.
        self.param: dict[str, str] = {}
        self.builtin = _builtin_renderers()

    # -- driver ---------------------------------------------------------

    def emit(self) -> str:
        params = self._collect_params()
        self._bind_inputs()
        body: list[str] = []
        for stmt_idx, stmt in enumerate(self.prog.statements):
            body.extend(self._emit_stmt(stmt_idx, stmt))
        body.extend(self._emit_outputs())

        sig = f"def {self.func_name}({', '.join(params)}):"
        src = ["import numpy as np", "", sig]
        src.append('    """Generated from polyarray Program '
                   f'{self.prog.name!r} by to_numpy_source."""')
        src.extend("    " + ln if ln else "" for ln in body)
        return "\n".join(src) + "\n"

    # -- inputs ---------------------------------------------------------

    def _collect_params(self) -> list[str]:
        params: list[str] = []
        for inp in self.prog.inputs:
            p = _safe_param_name(inp.name)
            self.param[inp.name] = p
            params.append(p)
        for atom_name in self.prog.int_atoms:
            p = _safe_param_name(atom_name)
            self.param[atom_name] = p
            params.append(p)
        return params

    def _bind_inputs(self) -> None:
        for inp in self.prog.inputs:
            pname = self.param[inp.name]
            if is_dynamic(inp.shape):
                # Dynamic input is bulk: bound whole-tensor under inp.name.
                self.bulkmap[inp.name] = pname
                continue
            cells = self.prog.input_arrays[inp.name].cells
            shape = tuple(inp.shape)
            for idx in (np.ndindex(*shape) if shape else [()]):
                cell = cells[idx] if shape else cells[()]
                if isinstance(cell, RationalFunction):
                    gen = cell.gens[0]
                    self.genmap[gen] = (
                        pname if not shape else pname + _index_suffix(idx)
                    )

    # -- statements -----------------------------------------------------

    def _emit_stmt(self, stmt_idx: int, stmt: "Stmt") -> list[str]:
        if stmt.fn is None:
            return []
        arg_exprs = [self._ref_expr(r) for r in stmt.in_]
        expr = self._render_op(stmt.fn, arg_exprs)
        out = stmt.out
        lines: list[str] = []
        if stmt.note:
            lines.append(f"# {stmt.note}")
        if len(out) == 1:
            var = f"_t{stmt_idx}_0"
            lines.append(f"{var} = {expr}")
            self._bind_output(stmt_idx, 0, out[0], var)
        else:
            tmp = [f"_t{stmt_idx}_{k}" for k in range(len(out))]
            lines.append(f"({', '.join(tmp)}) = {expr}")
            for k, bound in enumerate(out):
                self._bind_output(stmt_idx, k, bound, tmp[k])
        return lines

    def _bind_output(
        self, stmt_idx: int, out_idx: int, bound: SymArray, var: str
    ) -> None:
        self.outvar[(stmt_idx, out_idx)] = var
        if bound._bulk is not None:
            self.bulkmap[bound._bulk.name] = var
            return
        # Per-cell atoms: map each output cell's generator to indexing ``var``.
        cells = bound.cells
        shape = cells.shape
        for idx in (np.ndindex(*shape) if shape else [()]):
            cell = cells[idx] if shape else cells[()]
            if isinstance(cell, RationalFunction):
                self.genmap[cell.gens[0]] = (
                    var if not shape else var + _index_suffix(idx)
                )

    # -- ref resolution (mirrors Program._resolve_ref) ------------------

    def _ref_expr(self, ref: Any) -> str:
        if isinstance(ref, InputRef):
            e = self.param[ref.name]
            return e + _index_suffix(ref.indices) if ref.indices else e
        if isinstance(ref, OutputRef):
            e = self.outvar[(ref.stmt_idx, ref.out_idx)]
            return e + _index_suffix(ref.indices) if ref.indices else e
        if isinstance(ref, Const):
            return _const_expr(ref.value)
        if isinstance(ref, RationalRef):
            return self._rf_expr(ref.rf)
        if isinstance(ref, SymArrayRef):
            if ref._bulk is not None:
                return self.bulkmap[ref._bulk.name]
            return self._cells_expr(ref.cells)
        if isinstance(ref, IntAtomRef):
            return f"int({self.param[ref.name]})"
        raise TypeError(f"to_numpy_source: unknown Ref {type(ref).__name__}")

    # -- op dispatch ----------------------------------------------------

    def _render_op(self, fn: Any, args: list[str]) -> str:
        renderer = self.builtin.get(type(fn))
        if renderer is None:
            renderer = self.extra_renderers.get(type(fn).__name__)
        if renderer is None:
            if isinstance(fn, Program):
                raise NotImplementedError(
                    "to_numpy_source: sub-Program Stmt fns are not yet "
                    f"supported (stmt fn is Program {fn.name!r})"
                )
            raise NotImplementedError(
                f"to_numpy_source: no renderer for op {type(fn).__name__!r}. "
                "Pass op_renderers={'"
                f"{type(fn).__name__}': lambda op, args: ...}} to emit it."
            )
        return renderer(fn, args)

    # -- cells / rational functions -------------------------------------

    def _cells_expr(self, cells: np.ndarray) -> str:
        """Assemble an ndarray of cells into a numpy-array expression."""
        cells = np.asarray(cells)
        if cells.dtype.kind == "f":
            return f"np.array({cells.tolist()!r}, dtype=float)"
        if cells.shape == ():
            return f"np.asarray({self._cell_expr(cells[()])}, dtype=float)"
        nested = self._nested_cell_list(cells)
        return f"np.array({nested}, dtype=float)"

    def _nested_cell_list(self, cells: np.ndarray) -> str:
        if cells.ndim == 0:
            return self._cell_expr(cells[()])
        if cells.ndim == 1:
            return "[" + ", ".join(self._cell_expr(c) for c in cells) + "]"
        return "[" + ", ".join(
            self._nested_cell_list(cells[i]) for i in range(cells.shape[0])
        ) + "]"

    def _cell_expr(self, cell: Any) -> str:
        if isinstance(cell, RationalFunction):
            return self._rf_expr(cell)
        return repr(float(cell))

    def _rf_expr(self, rf: RationalFunction) -> str:
        names = _ring_names(rf._ring)
        var_exprs: list[str] = []
        for n in names:
            if n not in self.genmap:
                raise KeyError(
                    f"to_numpy_source: generator {n!r} has no binding "
                    "(is it produced before it is consumed?)"
                )
            var_exprs.append("(" + self.genmap[n] + ")")
        num = _poly_to_pyexpr(rf.num, names, var_exprs)
        if rf.den == rf._ring.one:
            return num
        den = _poly_to_pyexpr(rf.den, names, var_exprs)
        return f"(({num}) / ({den}))"

    # -- outputs --------------------------------------------------------

    def _emit_outputs(self) -> list[str]:
        lines: list[str] = []
        returned: list[str] = []
        for name, sa in self.prog.outputs.items():
            var = _safe_param_name(name)
            if sa._bulk is not None:
                expr = self.bulkmap[sa._bulk.name]
            else:
                expr = self._cells_expr(sa.cells)
            lines.append(f"{var} = {expr}")
            returned.append(var)
        if not returned:
            lines.append("return None")
        elif len(returned) == 1:
            lines.append(f"return {returned[0]}")
        else:
            lines.append(f"return ({', '.join(returned)})")
        return lines


# ---------------------------------------------------------------------------
# Built-in op renderers (the polyarray typed-op vocabulary)
# ---------------------------------------------------------------------------

def _builtin_renderers() -> dict[type, OpRenderer]:
    def det(op, a):
        return f"np.linalg.det({a[0]})"

    def inv(op, a):
        return f"np.linalg.inv({a[0]})"

    def pinv(op, a):
        return f"np.linalg.pinv({a[0]})"

    def solve(op, a):
        return f"np.linalg.solve({a[0]}, {a[1]})"

    def sqrt(op, a):
        return f"np.sqrt(np.asarray({a[0]}, dtype=float))"

    def absop(op, a):
        return f"np.abs(np.asarray({a[0]}, dtype=float))"

    def sign(op, a):
        return f"np.sign(np.asarray({a[0]}, dtype=float))"

    def tensordot(op, a):
        return f"np.tensordot({a[0]}, {a[1]}, axes={op.axes!r})"

    def moveaxis(op, a):
        return f"np.moveaxis({a[0]}, {op.source!r}, {op.destination!r})"

    def identity(op, a):
        return f"np.asarray({a[0]})"

    def assertop(op, a):
        # AssertOp returns its first input unchanged (the predicate guards the
        # data dependency); we emit the passthrough only.
        return f"{a[0]}"

    def einsum(op, a):
        rhs = op._rhs
        return (
            f"np.einsum({op.spec!r}, {a[0]}, "
            f"np.array({rhs.tolist()!r}, dtype=np.{op.rhs_dtype}))"
        )

    def einsum_stmt(op, a):
        return (
            f"np.einsum({op.spec!r}, {', '.join(a)}, optimize={op.optimize!r})"
        )

    def qr(op, a):
        return f"np.linalg.qr({a[0]}, mode={op.mode!r})"

    def switch(op, a):
        branches = ", ".join(a[1:])
        return f"np.asarray([{branches}][int({a[0]})])"

    return {
        DetOp: det,
        InvOp: inv,
        PinvOp: pinv,
        SolveOp: solve,
        SqrtOp: sqrt,
        AbsOp: absop,
        SignOp: sign,
        TensordotOp: tensordot,
        MoveaxisOp: moveaxis,
        IdentityOp: identity,
        AssertOp: assertop,
        EinsumOp: einsum,
        EinsumStmtOp: einsum_stmt,
        QrOp: qr,
        SwitchOp: switch,
    }


def to_numpy_source(
    program: Program,
    func_name: str = "f",
    op_renderers: Mapping[str, OpRenderer] | None = None,
) -> str:
    """Emit a standalone numpy module source string for ``program``.

    The returned string defines ``import numpy as np`` and a function
    ``func_name`` taking one positional parameter per program input (named by
    the input name) plus one per declared ``IntAtom``, computing every Stmt in
    order, and returning the program output(s) — a single array when there is
    one output, else a tuple in declaration order.  ``exec``-ing the string and
    calling the function reproduces ``program.run(values)`` for the same inputs.

    ``op_renderers`` maps an op *class name* to a renderer
    ``(op, arg_exprs) -> python_expression_str`` so that front-ends with their
    own Stmt ops (e.g. grassmann's ``_AxisLenOp``) can be emitted without
    polyarray depending on them.  An op with neither a built-in nor a supplied
    renderer raises :class:`NotImplementedError` naming the op.
    """
    return _Emitter(program, func_name, op_renderers).emit()
