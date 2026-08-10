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
ops (keyed by their own class name) pass an ``op_renderers`` mapping keyed by
the op class name to :func:`to_numpy_source` — keeping polyarray free of any
dependency on the front-end.  An op with no renderer raises a clear
``NotImplementedError`` naming the op.

Plain-callable Stmt fns: the ``__numpy_source__`` emit hook
-----------------------------------------------------------
A ``Stmt.fn`` may also be a *plain Python callable* that is neither an op class,
a sub-:class:`Program`, nor a ``vmap`` closure (e.g. a producer's
``lambda *arrs: np.einsum(spec, *arrs)``).  Such a fn emits **iff** it carries a
``__numpy_source__(args) -> str`` attribute: a callable taking the list of
already-rendered argument-expression strings (in operand order) and returning a
single numpy *expression string* (written in terms of ``np`` and those arg
exprs).  Producers opt a plain-callable Stmt into emission by stamping
``fn.__numpy_source__``; a plain fn with no such hook still raises the clear
``NotImplementedError``.  This keeps the protocol additive and polyarray free of
any front-end dependency.
"""
from __future__ import annotations

import keyword
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .ir import (
    AbsOp,
    Cell,
    Ref,
    StmtOp,
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
    Stmt,
    SumDimOp,
    SumShapeOp,
    SvdOp,
    SwitchOp,
    SymArray,
    SymArrayRef,
    TensordotOp,
    TransposeOp,
    is_dynamic,
)
from .rational import RationalFunction, _poly_to_pyexpr, _ring_names

__all__ = ["to_numpy_source", "OpRenderer"]


def _vmap_body_of(fn: StmtOp) -> Program | None:
    """Pull a per-point body :class:`Program` out of a ``vmap`` closure.

    Mirrors :func:`polyarray.forward._body_of` but kept local so this module
    has no import-time dependency on ``forward``.  ``vmap`` also stamps the
    body onto ``fn._vmap_body`` (additive); prefer that, fall back to the
    closure scan for robustness.
    """
    body = getattr(fn, "_vmap_body", None)
    if isinstance(body, Program):
        return body
    for c in getattr(fn, "__closure__", None) or ():
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if isinstance(v, Program):
            return v
    return None


class _HelperRegistry:
    """Shared, module-level helper-def accumulator across nested emitters.

    Sub-``Program`` bodies and ``vmap`` bodies are emitted once each as a
    module-level ``def`` (deduplicated by object identity) and referenced by
    name from the call sites that need them.
    """

    def __init__(self) -> None:
        self.helpers: dict[int, str] = {}   # id(obj) -> function name
        self.defs: list[list[str]] = []     # ordered list of def line-blocks
        self._counter = 0

    def fresh(self, stem: str) -> str:
        name = f"_{stem}{self._counter}"
        self._counter += 1
        return name

# A renderer maps ``(op, arg_exprs)`` to a single Python expression string that,
# when evaluated, yields the op's result (a tuple for multi-output ops).
#: Renders one statement op to a numpy expression string: ``(op, arg_exprs) -> str``.
#: ``arg_exprs`` holds the already-rendered operand expressions, in operand order.
type OpRenderer = Callable[[Any, list[str]], str]


class DeadOpKeyWarning(UserWarning):
    """A caller-supplied op-name key can never match any op, so its entry is unreachable.

    Both public string-keyed extension points (``to_numpy_source(op_renderers=)`` and
    ``pyab.LowerOpts.op_lowerings``) exist so a front end can render ITS ops without polyarray
    importing it. The price is that a key naming nothing is INVISIBLE: the lookup simply misses
    and the caller gets the fallback — which is usually correct, so nothing fails and nothing
    warns. Seven such keys survived a year in ``batch.py``.

    polyarray cannot validate a genuine front-end name (it deliberately cannot see that
    namespace). It CAN catch the specific spelling that caused every instance so far: a private
    ``_Name`` for an op that now lives in polyarray as ``Name``. Ops were relocated here from
    grassmann's lowering layer, so every such key was correct before the move and silently dead
    after it — and the fallback kept the results right, which is precisely why nobody noticed.
    """


def _warn_dead_op_keys(keys: Iterable[str], where: str) -> None:
    """Warn about supplied keys that name a polyarray op with a private spelling.

    Deliberately narrow. A key that resolves to nothing at all is NOT flagged: it may name a
    front-end op polyarray genuinely cannot see, and a false alarm on a legitimate key would
    train people to ignore this warning. A key whose underscore-stripped form IS one of our op
    classes is a different matter — there is no reading of it that can ever match.
    """
    from . import ir as _ir

    for key in keys or ():
        if not isinstance(key, str) or isinstance(getattr(_ir, key, None), type):
            continue
        public = key.lstrip("_")
        if public != key and isinstance(getattr(_ir, public, None), type):
            warnings.warn(
                f"{where}: the key {key!r} names polyarray's {public!r} with a PRIVATE spelling, "
                f"so it can never match — dispatch is on `type(fn).__name__`, which is "
                f"{public!r}. This entry is unreachable; the built-in renderer handles the op "
                f"instead, so results stay correct and the entry silently does nothing. Key it "
                f"{public!r}, or drop it if the built-in already covers the op.",
                DeadOpKeyWarning, stacklevel=3)


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


def _const_expr(value: npt.ArrayLike) -> str:
    """Render a constant operand as a numpy source expression."""
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
        registry: _HelperRegistry | None = None,
    ) -> None:
        self.prog = program
        self.func_name = func_name
        self.extra_renderers = dict(op_renderers or {})
        # Shared across nested emitters so sub-Program / vmap bodies become
        # module-level helper defs emitted exactly once.
        self.registry = registry if registry is not None else _HelperRegistry()
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
        src = ["import numpy as np", ""]
        # Helper defs (sub-Program / vmap bodies) accumulated while emitting
        # the body above; emitted module-level, before the entrypoint.
        for block in self.registry.defs:
            src.extend(block)
            src.append("")
        src.append(sig)
        src.append('    """Generated from polyarray Program '
                   f'{self.prog.name!r} by to_numpy_source."""')
        src.extend("    " + ln if ln else "" for ln in body)
        return "\n".join(src) + "\n"

    # -- nested-program helper emission ---------------------------------

    def _emit_program_helper(self, prog: Program) -> str:
        """Emit ``prog`` as a module-level helper def; return its name.

        Reproduces :meth:`Program._run_stmt`'s sub-Program dispatch: the
        helper takes one positional parameter per body input (in declaration
        order) and returns the body outputs in insertion order — a bare value
        for a single output, else a tuple.  Deduplicated by identity, so a
        shared body is emitted once and recursion terminates.
        """
        key = id(prog)
        existing = self.registry.helpers.get(key)
        if existing is not None:
            return existing
        name = self.registry.fresh("sub")
        self.registry.helpers[key] = name  # reserve before recursing
        sub = _Emitter(prog, name, self.extra_renderers, registry=self.registry)
        self.registry.defs.append(sub._emit_helper_lines())
        return name

    def _emit_helper_lines(self) -> list[str]:
        """Build this program's lines as a module-level helper def block.

        Like :meth:`emit` but: no ``import`` line, parameters are the body
        inputs only (positional, matching ``_run_stmt``'s operand→input map),
        and the helper returns the outputs rather than being an entrypoint.
        """
        params = self._collect_helper_params()
        self._bind_inputs()
        body: list[str] = []
        for stmt_idx, stmt in enumerate(self.prog.statements):
            body.extend(self._emit_stmt(stmt_idx, stmt))
        body.extend(self._emit_outputs())
        lines = [f"def {self.func_name}({', '.join(params)}):"]
        lines.append('    """Sub-Program/vmap body '
                     f'{self.prog.name!r} from to_numpy_source."""')
        lines.extend("    " + ln if ln else "" for ln in body)
        return lines

    def _collect_helper_params(self) -> list[str]:
        """Positional params for a helper: the body inputs, in order.

        IntAtoms are intentionally excluded — ``_run_stmt`` passes only the
        resolved operands to a sub-Program's :meth:`run`, so a body that
        references an IntAtom would already fail at run time.
        """
        params: list[str] = []
        for inp in self.prog.inputs:
            # A DimAtom-shaped (runtime-rank FFS) input is fine as a positional array param — its
            # actual shape is fixed at call time; the body's ops read it from the passed array.
            p = _safe_param_name(inp.name)
            self.param[inp.name] = p
            params.append(p)
        return params

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

    def _emit_stmt(self, stmt_idx: int, stmt: Stmt) -> list[str]:
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

    def _ref_expr(self, ref: Ref) -> str:
        """Render the expression a statement input ref reads from."""
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

    def _render_op(self, fn: StmtOp, args: list[str]) -> str:
        """Render one statement op, via the builtin table, a supplied renderer, or its own hook."""
        renderer = self.builtin.get(type(fn))
        if renderer is None:
            renderer = self.extra_renderers.get(type(fn).__name__)
        if renderer is not None:
            return renderer(fn, args)
        # CallOp wraps a Program / vmap-closure / opaque callable — unwrap and
        # dispatch on the inner fn (matches CallOp.__call__ -> _invoke(self.fn, ...)).
        if isinstance(fn, CallOp):
            return self._render_op(fn.fn, args)
        # A sub-Program Stmt fn: dispatch like ``_run_stmt`` (operands map to
        # body inputs by position; outputs in insertion order).
        if isinstance(fn, Program):
            helper = self._emit_program_helper(fn)
            return f"{helper}({', '.join(args)})"
        # A ``vmap(body)`` closure Stmt fn: batch the body over its axes.
        if _vmap_body_of(fn) is not None:
            helper = self._emit_vmap_helper(fn)
            return f"{helper}({', '.join(args)})"
        # A plain Python callable Stmt fn (not an op class, Program, or vmap
        # closure) emits IFF it carries an ``__numpy_source__(args)->str`` hook:
        # the producer stamps ``fn.__numpy_source__`` with a callable taking the
        # already-rendered arg-expression strings and returning a numpy
        # expression string (in terms of ``np`` and those arg exprs). This lets
        # any front-end opt a plain-callable Stmt into emission without polyarray
        # importing it. Hookless plain-fn Stmts fall through to the raise below.
        hook = getattr(fn, "__numpy_source__", None)
        if callable(hook):
            return hook(args)
        raise NotImplementedError(
            f"to_numpy_source: no renderer for op {type(fn).__name__!r}. "
            "Pass op_renderers={'"
            f"{type(fn).__name__}': lambda op, args: ...}} to emit it."
        )

    def _emit_vmap_helper(self, fn: StmtOp) -> str:
        """Emit a ``vmap(body)`` closure as a module-level helper; return name.

        Mirrors :func:`polyarray.ir.vmap`'s loop exactly: each batched input
        (``in_axes[i] != None``) is sliced along its normalised axis at index
        ``i``; ``None`` operands are broadcast unchanged; the body runs once
        per slice and outputs are ``np.stack``-ed along ``out_axes``.
        """
        key = id(fn)
        existing = self.registry.helpers.get(key)
        if existing is not None:
            return existing
        body = _vmap_body_of(fn)
        if getattr(fn, "_nested_n_vars", None) is not None:
            return self._emit_nested_vmap_helper(fn, body)
        in_axes = getattr(fn, "_in_axes", None)
        out_axes = getattr(fn, "_out_axes", None)
        if in_axes is None or out_axes is None:
            raise NotImplementedError(
                "to_numpy_source: vmap closure does not expose _in_axes/"
                "_out_axes (rebuild it via polyarray.ir.vmap)"
            )
        in_axes = tuple(in_axes)
        out_axes = tuple(out_axes)
        # Emit the per-slice body as its own helper (deduplicated by identity).
        body_helper = self._emit_program_helper(body)

        name = self.registry.fresh("vmap")
        self.registry.helpers[key] = name
        n_in = len(body.inputs)
        n_out = len(body.outputs)
        params = [f"a{i}" for i in range(n_in)]
        L = "    "
        block: list[str] = [f"def {name}({', '.join(params)}):"]
        block.append(L + '"""vmap('
                     f'{body.name!r}) batched body from to_numpy_source."""')
        block.append(L + f"_arrs = [np.asarray(a) for a in ({', '.join(params)}"
                     f"{',' if n_in == 1 else ''})]")
        block.append(L + f"_in_axes = {in_axes!r}")
        block.append(L + "_norm = []")
        block.append(L + "_M = None")
        block.append(L + "for _a, _ax in zip(_arrs, _in_axes):")
        block.append(L + "    if _ax is None:")
        block.append(L + "        _norm.append(None)")
        block.append(L + "    else:")
        block.append(L + "        _axn = _ax if _ax >= 0 else _a.ndim + _ax")
        block.append(L + "        _norm.append(_axn)")
        block.append(L + "        _M = _a.shape[_axn]")
        block.append(L + "_per = []")
        block.append(L + "for _i in range(_M):")
        block.append(L + "    _sv = []")
        block.append(L + "    for _a, _ax in zip(_arrs, _norm):")
        block.append(L + "        if _ax is None:")
        block.append(L + "            _sv.append(_a)")
        block.append(L + "        else:")
        block.append(L + "            _idx = [slice(None)] * _a.ndim")
        block.append(L + "            _idx[_ax] = _i")
        block.append(L + "            _sv.append(_a[tuple(_idx)])")
        block.append(L + f"    _per.append({body_helper}(*_sv))")
        if n_out == 1:
            block.append(L + f"return np.stack(_per, axis={out_axes[0]!r})")
        else:
            parts = ", ".join(
                f"np.stack([_p[{k}] for _p in _per], axis={out_axes[k]!r})"
                for k in range(n_out)
            )
            block.append(L + f"return ({parts})")
        self.registry.defs.append(block)
        return name

    def _emit_nested_vmap_helper(self, fn: StmtOp, body: Program) -> str:
        """Emit a multi-variable nested vmap as a helper, returning its name.

        Loops the cartesian product of the bound-variable axes, runs the per-slice body,
        stacks into ``(*var_sizes, *cod)``, then moves the variable axes after the codomain
        axes — mirroring the runtime nested-vmap closure exactly.
        """
        key = id(fn)
        existing = self.registry.helpers.get(key)
        if existing is not None:
            return existing
        n_vars = int(fn._nested_n_vars)
        n_free = int(fn._nested_n_free)
        body_helper = self._emit_program_helper(body)
        name = self.registry.fresh("vmapnest")
        self.registry.helpers[key] = name
        params = [f"a{i}" for i in range(n_vars + n_free)]
        L = "    "
        b = [f"def {name}({', '.join(params)}):"]
        b.append(L + f'"""nested vmap({body.name!r}) — {n_vars} bound vars — from to_numpy_source."""')
        b.append(L + "import itertools as _it")
        b.append(L + f"_eyes = [np.asarray(a) for a in ({', '.join(params[:n_vars])},)]")
        free_list = ", ".join(params[n_vars:])
        b.append(L + f"_free = [np.asarray(a) for a in ({free_list}{',' if n_free == 1 else ''})]"
                 if n_free else L + "_free = []")
        b.append(L + "_sizes = [e.shape[0] for e in _eyes]")
        b.append(L + "_grid = {}")
        b.append(L + "for _c in _it.product(*[range(s) for s in _sizes]):")
        b.append(L + "    _sv = [_eyes[_v][_c[_v]] for _v in range(len(_eyes))] + _free")
        b.append(L + f"    _grid[_c] = np.asarray({body_helper}(*_sv))")
        b.append(L + "_cod = next(iter(_grid.values())).shape")
        b.append(L + "_full = np.empty(tuple(_sizes) + _cod, dtype=float)")
        b.append(L + "for _c, _r in _grid.items(): _full[_c] = _r")
        b.append(L + f"return np.moveaxis(_full, range({n_vars}), range(-{n_vars}, 0))")
        self.registry.defs.append(b)
        return name

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

    def _cell_expr(self, cell: Cell) -> str:
        """Render one cell as a numpy source expression."""
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
    """Build the renderer table for polyarray's own typed op vocabulary."""
    def det(op: DetOp, a: list[str]) -> str:
        return f"np.linalg.det({a[0]})"

    def inv(op: InvOp, a: list[str]) -> str:
        return f"np.linalg.inv({a[0]})"

    def pinv(op: PinvOp, a: list[str]) -> str:
        return f"np.linalg.pinv({a[0]})"

    def solve(op: SolveOp, a: list[str]) -> str:
        return f"np.linalg.solve({a[0]}, {a[1]})"

    def sqrt(op: SqrtOp, a: list[str]) -> str:
        return f"np.sqrt(np.asarray({a[0]}, dtype=float))"

    def absop(op: AbsOp, a: list[str]) -> str:
        return f"np.abs(np.asarray({a[0]}, dtype=float))"

    def sign(op: SignOp, a: list[str]) -> str:
        return f"np.sign(np.asarray({a[0]}, dtype=float))"

    def tensordot(op: TensordotOp, a: list[str]) -> str:
        return f"np.tensordot({a[0]}, {a[1]}, axes={op.axes!r})"

    def moveaxis(op: MoveaxisOp, a: list[str]) -> str:
        return f"np.moveaxis({a[0]}, {op.source!r}, {op.destination!r})"

    def identity(op: IdentityOp, a: list[str]) -> str:
        return f"np.asarray({a[0]})"

    def assertop(op: AssertOp, a: list[str]) -> str:
        # AssertOp returns its first input unchanged (the predicate guards the
        # data dependency); we emit the passthrough only.
        return f"{a[0]}"

    def einsum(op: EinsumOp, a: list[str]) -> str:
        rhs = op._rhs
        return (
            f"np.einsum({op.spec!r}, {a[0]}, "
            f"np.array({rhs.tolist()!r}, dtype=np.{op.rhs_dtype}))"
        )

    def einsum_stmt(op: EinsumStmtOp, a: list[str]) -> str:
        return (
            f"np.einsum({op.spec!r}, {', '.join(a)}, optimize={op.optimize!r})"
        )

    def qr(op: QrOp, a: list[str]) -> str:
        return f"np.linalg.qr({a[0]}, mode={op.mode!r})"

    def svd(op: SvdOp, a: list[str]) -> str:
        # Multi-output (U, S, Vh, rank): np.linalg.svd + a thresholded numerical rank (mirrors
        # SvdOp.__call__). Emitted as one self-contained expression (no prelude helper needed).
        rc = "None" if op.rcond is None else repr(op.rcond)
        return (
            f"(lambda _A: (lambda _U, _S, _Vh: (_U, _S, _Vh, np.asarray("
            f"0 if _S.size == 0 else int((_S > (("
            f"_S.max() * max(_A.shape) * np.finfo(float).eps) if {rc} is None "
            f"else {rc} * _S.max())).sum()))))(*np.linalg.svd("
            f"_A, full_matrices={op.full_matrices!r})))({a[0]})"
        )

    def switch(op: SwitchOp, a: list[str]) -> str:
        branches = ", ".join(a[1:])
        return f"np.asarray([{branches}][int({a[0]})])"

    def transpose(op: TransposeOp, a: list[str]) -> str:
        return f"np.asarray({a[0]}).T"

    def sinv_full(op: SinvFullOp, a: list[str]) -> str:
        # out[i,j] = 1/S[i] on the diagonal for i < rank, else 0 (nrows×ncols).
        n, m = int(op.nrows), int(op.ncols)
        return (
            f"(lambda _S, _rk, _sp: np.where("
            f"(np.arange({n})[:, None] == np.arange({m})[None, :]), "
            f"np.where(np.arange({n}) < _rk, 1.0 / np.where(np.arange({n}) < _rk, _sp, 1.0), 0.0)"
            f"[:, None], 0.0))({a[0]}, int({a[1]}), "
            f"np.concatenate([np.asarray({a[0]}, dtype=float).reshape(-1), np.ones({n})])[:{n}])"
        )

    def gsvd_full(op: GSvdFullOp, a: list[str]) -> str:
        # GSvdOp then re-concatenate [U|UI] / [V|VI] == the full de-whitened factors
        # (mirrors GSvdOp.__call__ exactly). Self-contained: GSvdOp has no numpy builtin.
        rc = "None" if op.rcond is None else repr(float(op.rcond))
        return (
            "(lambda _A, _MV, _MW: (lambda _Lw, _Rw: (lambda _Aw: (lambda _Ut, _S, _Vt: "
            "(lambda _rank: (np.linalg.solve(_Lw.T, _Ut), np.linalg.solve(_Rw.T, _Vt.T), "
            "_S, np.asarray(_rank)))"
            f"(0 if _S.size == 0 else int((_S > (({rc} * _S.max()) if {rc} is not None "
            "else (_S.max() * max(_Aw.shape) * np.finfo(float).eps))).sum())))"
            "(*np.linalg.svd(_Aw, full_matrices=True)))"
            "(_Lw.T @ _A @ np.linalg.inv(_Rw.T)))"
            "(np.linalg.cholesky(np.asarray(_MW, dtype=float)), "
            "np.linalg.cholesky(np.asarray(_MV, dtype=float))))"
            f"(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float), "
            f"np.asarray({a[2]}, dtype=float))"
        )

    def block_diag(op: BlockDiagOp, a: list[str]) -> str:
        ms = "[" + ", ".join(f"np.asarray({e}, dtype=float)" for e in a) + "]"
        return (
            "(lambda _ms: (lambda _w: np.concatenate([np.concatenate("
            "[np.zeros((_m.shape[0], sum(_w[:_i]))), _m, np.zeros((_m.shape[0], sum(_w[_i+1:])))], "
            "axis=1) for _i, _m in enumerate(_ms)], axis=0))([_m.shape[1] for _m in _ms]))"
            f"({ms})"
        )

    def block_repeat(op: BlockRepeatOp, a: list[str]) -> str:
        return f"np.kron(np.eye({int(op.n)}), np.asarray({a[0]}, dtype=float))"

    def dyn_block_repeat(op: DynBlockRepeatOp, a: list[str]) -> str:
        return f"np.kron(np.eye(int({a[1]})), np.asarray({a[0]}, dtype=float))"

    # --- Batch-2 relocated generic array ops (bodies moved from grassmann) ---
    def dyn_eye(op: DynEyeOp, a: list[str]) -> str:
        return f"np.eye(int(np.asarray({a[0]}).shape[{op.axis}]))"

    def dyn_zeros(op: DynZerosOp, a: list[str]) -> str:
        dims = ", ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
        return f"np.zeros(({dims},))"

    def dyn_eye_tensor(op: DynEyeTensorOp, a: list[str]) -> str:
        dims = ", ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
        return f"(lambda _d: np.eye(int(np.prod(_d))).reshape(int(np.prod(_d)), *_d))([{dims}])"

    def prod_shape(op: ProdShapeOp, a: list[str]) -> str:
        inner = ", ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
        return f"np.asarray(int({op.static} * int(np.prod([{inner}] or [1]))))"

    def sum_shape(op: SumShapeOp, a: list[str]) -> str:
        inner = " + ".join(f"int(np.asarray({a[i]}).shape[{ax}])" for i, ax in enumerate(op.axes))
        return f"np.asarray(int({op.static} + ({inner or '0'})))"

    def sum_dim(op: SumDimOp, a: list[str]) -> str:
        return f"np.asarray(int(sum(np.asarray(_m).shape[{op.axis}] for _m in [{', '.join(a)}])))"

    def prod_dim(op: ProdDimOp, a: list[str]) -> str:
        return (f"np.asarray(int(np.prod([np.asarray(_m).shape[{op.axis}] "
                f"for _m in [{', '.join(a)}]])))")

    def scale_axis_dim(op: ScaleAxisDimOp, a: list[str]) -> str:
        return f"np.asarray(int({op.n} * np.asarray({a[0]}).shape[{op.axis}]))"

    def mul_axis_dim(op: MulAxisDimOp, a: list[str]) -> str:
        return f"np.asarray(int(int({a[0]}) * np.asarray({a[1]}).shape[{op.axis}]))"

    def comp_rank(op: CompRankOp, a: list[str]) -> str:
        return f"np.asarray(int({op.ambient} - int({a[0]})))"

    def hstack(op: HStackOp, a: list[str]) -> str:
        return f"np.hstack([np.asarray(_m, dtype=float) for _m in [{', '.join(a)}]])"

    def col_stack(op: ColStackOp, a: list[str]) -> str:
        return (f"np.stack([np.asarray(_c, dtype=float).reshape(-1) for _c in "
                f"[{', '.join(a)}]], axis=1)")

    def scale(op: ScaleOp, a: list[str]) -> str:
        return f"({op.factor!r} * np.asarray({a[0]}, dtype=float))"

    def scale_by(op: ScaleByOp, a: list[str]) -> str:
        return f"(np.asarray({a[1]}, dtype=float) * np.asarray({a[0]}, dtype=float))"

    def add(op: AddOp, a: list[str]) -> str:
        return "(" + " + ".join(f"np.asarray({e}, dtype=float)" for e in a) + ")"

    def concat(op: ConcatOp, a: list[str]) -> str:
        return ("np.concatenate(["
                + ", ".join(f"np.asarray({e}, dtype=float).reshape(-1)" for e in a) + "])")

    def axis_len(op: AxisLenOp, a: list[str]) -> str:
        return f"np.asarray(int(np.asarray({a[0]}).shape[{op.axis}]))"

    def reshape(op: ReshapeOp, a: list[str]) -> str:
        return f"np.asarray({a[0]}, dtype=float).reshape({tuple(op.shape)})"

    def const(op: ConstOp, a: list[str]) -> str:
        val = np.frombuffer(op.data_bytes, dtype=op.dtype).reshape(op.shape)
        return f"np.array({val.tolist()!r}, dtype=np.{op.dtype})"

    def eye(op: EyeOp, a: list[str]) -> str:
        return f"np.eye({op.n})"

    def first_cols(op: FirstColsOp, a: list[str]) -> str:
        return f"np.asarray({a[0]}, dtype=float)[:, :int({a[1]})]"

    def last_cols(op: LastColsOp, a: list[str]) -> str:
        return f"np.asarray({a[0]}, dtype=float)[:, int({a[1]}):]"

    # --- Batch-3 relocated generic array / linalg ops (bodies from grassmann) ---
    def project(op: ProjectOp, a: list[str]) -> str:  # Pᵀ @ v  (ambient coords -> sub-basis coords)
        return f"(np.asarray({a[0]}, dtype=float).T @ np.asarray({a[1]}, dtype=float).reshape(-1))"

    def embed(op: EmbedOp, a: list[str]) -> str:  # P @ vsub  (sub-coords -> ambient), reshaped to op.shape
        base = f"(np.asarray({a[0]}, dtype=float) @ np.asarray({a[1]}, dtype=float))"
        return base + (f".reshape({tuple(op.shape)})" if op.shape else "")

    def kron(op: KronOp, a: list[str]) -> str:  # chained Kronecker product
        ms = ", ".join(f"np.asarray({e}, dtype=float)" for e in a)
        return f"__import__('functools').reduce(np.kron, [{ms}])"

    def kron_free(op: KronFreeOp, a: list[str]) -> str:  # Kron of two maps with trailing free (basis/seed) axes (block ⊗)
        nf, ng = op.nf_free, op.ng_free
        return (
            f"(lambda _F, _G: (_F.reshape(_F.shape[0], 1, _F.shape[1], 1, *_F.shape[2:], *([1]*{ng})) "
            f"* _G.reshape(1, _G.shape[0], 1, _G.shape[1], *([1]*{nf}), *_G.shape[2:])).reshape("
            f"_F.shape[0]*_G.shape[0], _F.shape[1]*_G.shape[1], *_F.shape[2:], *_G.shape[2:]))"
            f"(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))")

    def inv_transpose(op: InvTransposeOp, a: list[str]) -> str:  # inv(A).T
        return f"np.linalg.inv(np.asarray({a[0]}, dtype=float)).T"

    def compose_via_std(op: ComposeViaStdOp, a: list[str]) -> str:  # solve(R_to, R_from)
        return f"np.linalg.solve(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))"

    def sqrt_spd(op: SqrtSpdOp, a: list[str]) -> str:  # the SPD square root via eigh of the symmetrized G
        return (f"(lambda _w, _V: (_V * np.sqrt(_w)) @ _V.T)(*np.linalg.eigh("
                f"0.5 * (np.asarray({a[0]}, dtype=float) + np.asarray({a[0]}, dtype=float).T)))")

    def rank(op: RankOp, a: list[str]) -> str:  # numerical rank of A as a 0-d int
        return f"np.asarray(int(np.linalg.matrix_rank(np.asarray({a[0]}, dtype=float), tol={op.tol!r})))"

    def metric_orthonormal(op: MetricOrthonormalOp, a: list[str]) -> str:  # A @ inv(chol(Aᵀ G A))ᵀ
        return (f"(lambda _a, _g: _a @ np.linalg.inv(np.linalg.cholesky(_a.T @ _g @ _a)).T)"
                f"(np.asarray({a[0]}, dtype=float), np.asarray({a[1]}, dtype=float))")

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
        SvdOp: svd,
        SwitchOp: switch,
        TransposeOp: transpose,
        SinvFullOp: sinv_full,
        GSvdFullOp: gsvd_full,
        BlockDiagOp: block_diag,
        BlockRepeatOp: block_repeat,
        DynBlockRepeatOp: dyn_block_repeat,
        DynEyeOp: dyn_eye,
        DynZerosOp: dyn_zeros,
        DynEyeTensorOp: dyn_eye_tensor,
        ProdShapeOp: prod_shape,
        SumShapeOp: sum_shape,
        SumDimOp: sum_dim,
        ProdDimOp: prod_dim,
        ScaleAxisDimOp: scale_axis_dim,
        MulAxisDimOp: mul_axis_dim,
        CompRankOp: comp_rank,
        HStackOp: hstack,
        ColStackOp: col_stack,
        ScaleOp: scale,
        ScaleByOp: scale_by,
        AddOp: add,
        ConcatOp: concat,
        AxisLenOp: axis_len,
        ReshapeOp: reshape,
        ConstOp: const,
        EyeOp: eye,
        FirstColsOp: first_cols,
        LastColsOp: last_cols,
        ProjectOp: project,
        EmbedOp: embed,
        KronOp: kron,
        KronFreeOp: kron_free,
        InvTransposeOp: inv_transpose,
        ComposeViaStdOp: compose_via_std,
        SqrtSpdOp: sqrt_spd,
        RankOp: rank,
        MetricOrthonormalOp: metric_orthonormal,
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
    own Stmt ops can be emitted without
    polyarray depending on them.  An op with neither a built-in nor a supplied
    renderer raises :class:`NotImplementedError` naming the op.
    """
    _warn_dead_op_keys(op_renderers, "to_numpy_source(op_renderers=…)")
    return _Emitter(program, func_name, op_renderers).emit()
