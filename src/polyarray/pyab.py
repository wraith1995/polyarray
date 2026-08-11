"""Optional: lower a :class:`polyarray.Program` to PyArrayBackend (PyAB) IR.

A :class:`~polyarray.ir.Program` can be *run* (:meth:`Program.run`) or emitted as
standalone numpy source (:func:`polyarray.numpy_source.to_numpy_source`).  This
module lowers it instead to **PyArrayBackend IR** — the ``pyarraybackend.ir``
node vocabulary — so it can be compiled with PyAB's torch backend (Inductor /
``@torch.compile``) or its numpy backend.

Two ways to use the result, mirroring the two things you do with a lowered
program:

* **Standalone** — :func:`as_function_def` emits the program as one PyAB
  ``FunctionDefStmt`` (a ``def f(...)``); hand it (and any helper defs) to
  ``pyarraybackend.backends.torch_backend.compile`` and call ``module.f(...)``.
  :func:`compile_torch` / :func:`compile_numpy` are one-call conveniences.

* **Embedded** — the program is a *per-element kernel* you want to invoke from
  inside a **larger PyAB program** (e.g. a lowered element kernel
  called by an assembly program).  :func:`call_lowered` emits the callable once
  and a *placed* call at the current build point.  The call site chooses:

    * ``place="plain"`` — a direct ``CallExpr``.
    * ``place="vmap"``  — wrap in a ``VMapHop`` (→ ``torch.vmap``); the kernel
      runs batched over the leading (or ``in_dims``-chosen) axis.
    * ``place="fuse"``  — emit the call inside a ``FusionRegionExpr``
      (→ ``@torch.compile``), fusing it with surrounding IR into one kernel.

  ``place="vmap"`` and ``place="fuse"`` compose — a fused region may contain a
  vmapped call.

Small QR interception
---------------------
``QrOp`` on a *statically small* matrix (``max(m, n) <= small_qr.max_dim``,
default 4) is lowered to **unrolled scalar Householder QR** — backend-agnostic
arithmetic + ``where`` (no ``linalg.qr`` call) — reproducing LAPACK/numpy's
R-diagonal sign convention.  This keeps tiny factorisations inside the fused
kernel (Inductor can lower the scalar ops to Triton; ``torch.vmap`` batches
them) instead of bouncing out to a batched-QR kernel.  Above the threshold, or
when disabled, a plain ``linalg.qr`` ``CallExpr`` is emitted.

Optionality
-----------
Importing this module requires ``pyarraybackend`` (imported lazily, inside the
functions).  ``import polyarray`` never imports it, so polyarray keeps no
hard dependency on pyarraybackend or torch.  ``torch`` itself is needed only
when you actually *compile* the emitted IR through the torch backend.
"""
from __future__ import annotations

import dataclasses
import enum
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

    from pyarraybackend.backends.python_backend import CompiledModule as NumpyModule
    from pyarraybackend.backends.python_backend import PythonBackendOptions
    from pyarraybackend.backends.torch_backend import CompiledModule as TorchModule
    from pyarraybackend.backends.torch_backend import TorchBackendOptions
    from pyarraybackend.ir.builder import StmtBuilder
    from pyarraybackend.ir.core import Device, FusionOpts, KernelKind, PyExprs, PyStmts

from .ir import (
    AbsOp,
    AddOp,
    AssertOp,
    AxisLenOp,
    BlockDiagOp,
    BlockRepeatOp,
    CallOp,
    Cell,
    ColStackOp,
    ComposeViaStdOp,
    CompRankOp,
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
    NestedVmapClosure,
    OutputRef,
    PinvOp,
    ProdDimOp,
    ProdShapeOp,
    Program,
    ProjectOp,
    QrOp,
    RankOp,
    RationalRef,
    Ref,
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
    StmtOp,
    SumDimOp,
    SumShapeOp,
    SvdOp,
    SwitchOp,
    SymArray,
    SymArrayRef,
    TensordotOp,
    TransposeOp,
    WhileOp,
    is_dynamic,
)
from .poly_backend import Poly
from .rational import RationalFunction, _coeff_to_float, _ring_names

__all__ = [
    "SmallQrOpts",
    "LowerOpts",
    "OpLowering",
    "prepare",
    "collapse_vmap",
    "lower_program_into",
    "as_function_def",
    "call_lowered",
    "emit_module_stmts",
    "compile_torch",
    "compile_numpy",
]

Target = Literal["torch", "numpy"]
Placement = Literal["plain", "vmap", "fuse"]

#: An op-lowering hook: ``(op, builder, arg_exprs, lowerer)`` to one expression per op
#: output.  Keyed by op *class name* so a front end can lower its own statement ops
#: without polyarray importing them, mirroring ``to_numpy_source``'s ``op_renderers``.
#: Spelled loosely at run time because pyab is an optional dependency; the precise
#: shape is the one written above.
type OpLowering = Callable[..., list[Any]]

#: What :func:`prepare` reports about the simplifications it applied, keyed by pass name.
type PrepReport = dict[str, int | str | bool]


@dataclass(frozen=True)
class SmallQrOpts:
    """Policy for intercepting small fixed-size ``QrOp`` as scalar Householder.

    ``max_dim`` — intercept iff ``max(m, n) <= max_dim`` and the operand shape
    is statically known.  ``enabled=False`` always emits a ``linalg.qr`` call.
    Only reduced/complete real QR with ``m >= n`` is intercepted; other cases
    fall through to the ``linalg.qr`` ``CallExpr``.
    """

    max_dim: int = 4
    enabled: bool = True


@dataclass(frozen=True)
class LowerOpts:
    """Knobs for :func:`lower_program_into` and friends.

    ``target`` selects the array namespace emitted for ops that have no
    backend-agnostic IR node (``linalg.det/inv/solve/qr``, ``sqrt``/``abs``/
    ``sign``/``moveaxis``): ``"torch"`` → ``torch.*``, ``"numpy"`` → ``np.*``.
    Must match the backend you compile with.  ``small_qr`` controls small-QR
    interception.  ``op_lowerings`` supplies renderers for front-end Stmt ops
    keyed by class name.
    """

    target: Target = "torch"
    small_qr: SmallQrOpts = field(default_factory=SmallQrOpts)
    op_lowerings: Mapping[str, OpLowering] | None = None
    dtype_f64: bool = True
    # Backend-prep simplifications (all semantics-preserving; opt out to disable):
    #   collapse_vmap — replace a verifiably-batchable ``vmap`` with the batched op
    #   simplify_gsvd — fold identity metrics out of a ``GSvdOp`` (toward a plain SVD)
    collapse_vmap: bool = True
    simplify_gsvd: bool = True


# ---------------------------------------------------------------------------
# Lazy PyAB import — keeps ``import polyarray`` free of a pyarraybackend dep.
# ---------------------------------------------------------------------------

def _core() -> ModuleType:
    """Import and return ``pyarraybackend.ir.core``, the pyab node vocabulary."""
    from pyarraybackend.ir import core

    return core


def _builder_mod() -> ModuleType:
    """Import and return ``pyarraybackend.ir.builder``, the statement builder module."""
    from pyarraybackend.ir import builder

    return builder


# ---------------------------------------------------------------------------
# pyab renderers for polyarray's relocated generic array ops.  Each
# ``(op, builder, args, low) -> list[expr]`` renderer builds PyAB IR from
# ``low.core`` primitives only (the same signature/behaviour as an op-carried
# ``__pyab_lower__`` hook), and is registered in the type-keyed
# ``_ARRAY_OP_LOWERINGS`` table consulted inside ``_Lowerer._render_op``.
# ---------------------------------------------------------------------------


def _axis_len(low: _Lowerer, x: PyExprs, axis: int) -> PyExprs:
    """Emit ``x.shape[axis]`` as a pyab integer expression."""
    c = low.core
    return c.AccessExpr(base=c.ShapeExpr(x=x), index=(c.IntLit(value=int(axis)),))


def _f64_scalar(low: _Lowerer, v: float) -> PyExprs:
    """Emit ``v`` as a 0-d float64 array expression."""
    c = low.core
    return c.ArrayExpr(obj=c.FloatLit(value=float(v)), dtype=low._dtype_f64())


def _static_eye(low: _Lowerer, n_expr: PyExprs) -> PyExprs:
    """Emit an exact ``n x n`` float64 identity.

    Built as ``where(arange(n)[:, None] == arange(n)[None, :], 1, 0)`` rather than a
    namespace ``eye``, whose torch default is float32. ``n_expr`` may be a static ``IntLit``
    or a runtime integer expression, so one builder serves both static and dynamic ``n``.
    """
    c = low.core
    r = c.ArangeExpr(start=c.IntLit(value=0), stop=n_expr)
    ri = c.AccessExpr(base=r, index=(c.Slice(), c.NewAxisTok()))
    rj = c.AccessExpr(base=r, index=(c.NewAxisTok(), c.Slice()))
    eq = c.BinaryExpr(op=c.BinaryOp.EQ, lhs=ri, rhs=rj)
    return c.WhereExpr(cond=eq, x=_f64_scalar(low, 1.0), y=_f64_scalar(low, 0.0))


def _render_transpose(op: TransposeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``A.T`` — full-reverse transpose (``axes=None``)."""
    c = low.core
    return [c.TransposeExpr(a=args[0], axes=None)]


def _render_sinv_full(op: SinvFullOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``out[i,j] = 1/Sᵢ`` on the diagonal for ``i < rank`` else 0 (an ``nrows×ncols`` array).

    Built as ``where(eye_nm, recip_row[:,None], 0)`` where ``recip_row`` is ``1/Sᵢ`` masked to
    zero beyond ``rank`` (and guarded against 1/0 by substituting 1 in the masked band),
    padded/truncated to length ``nrows`` so it broadcasts onto the rectangular diagonal.
    """
    c = low.core
    S, rank = args
    n, m = int(op.nrows), int(op.ncols)
    f64 = low._dtype_f64()
    n_lit, m_lit = c.IntLit(value=n), c.IntLit(value=m)
    ar_n = c.ArangeExpr(start=c.IntLit(value=0), stop=n_lit)
    ar_m = c.ArangeExpr(start=c.IntLit(value=0), stop=m_lit)
    ri = c.AccessExpr(base=ar_n, index=(c.Slice(), c.NewAxisTok()))
    rj = c.AccessExpr(base=ar_m, index=(c.NewAxisTok(), c.Slice()))
    eye = c.BinaryExpr(op=c.BinaryOp.EQ, lhs=ri, rhs=rj)
    rk = c.CallExpr(fn=c.Var(name="int"), args=(rank,))
    within = c.BinaryExpr(op=c.BinaryOp.LT, lhs=ar_n, rhs=rk)   # (n,) bool
    # Pad/truncate S to length n: concat with ones then slice [:n] (len S = min(n,m) <= n).
    ones_n = c.OnesExpr(shape=(n_lit,), dtype=f64)
    s_pad = c.ConcatExpr(arrays=(S, ones_n), axis=0)
    s_padn = c.AccessExpr(base=s_pad, index=(c.Slice(stop=n_lit),))
    one, zero = _f64_scalar(low, 1.0), _f64_scalar(low, 0.0)
    s_safe = c.WhereExpr(cond=within, x=s_padn, y=one)          # no 1/0 outside the used band
    recip = c.WhereExpr(
        cond=within,
        x=c.BinaryExpr(op=c.BinaryOp.TRUEDIV, lhs=one, rhs=s_safe),
        y=zero,
    )                                                           # (n,), zero beyond rank
    recip_col = c.AccessExpr(base=recip, index=(c.Slice(), c.NewAxisTok()))
    return [c.WhereExpr(cond=eye, x=recip_col, y=zero)]


def _render_gsvd_full(op: GSvdFullOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Run the GSVD and re-concatenate ``[U|UI]`` and ``[V|VI]``.

    Reuses the pyab ``_gsvd`` composite — whiten, svd, rank-threshold, de-whiten — so this
    matches the native GSVD lowering. ``_gsvd`` reads the operand refs off
    ``low._cur_in_refs``.
    """
    c = low.core
    u, ui, v, vi, s, rank = low._gsvd(GSvdOp(rcond=op.rcond), args, None, low._cur_in_refs)
    ufull = c.ConcatExpr(arrays=(u, ui), axis=1)
    vfull = c.ConcatExpr(arrays=(v, vi), axis=1)
    return [ufull, vfull, s, rank]


def _render_block_diag(op: BlockDiagOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Build ``diag(A, B, ...)`` from zero-padded row blocks concatenated on axis 0.

    Block sizes are read off runtime ``ShapeExpr``, so this is vmap-safe and correct under
    dynamic shapes.
    """
    c = low.core
    f64 = low._dtype_f64()
    col_widths = [_axis_len(low, x, 1) for x in args]

    def add(a: PyExprs, b: PyExprs) -> PyExprs:
        return c.BinaryExpr(op=c.BinaryOp.ADD, lhs=a, rhs=b)

    row_blocks = []
    for i, x in enumerate(args):
        ri = _axis_len(low, x, 0)
        pieces = []
        left = col_widths[:i]
        right = col_widths[i + 1:]
        if left:
            w = left[0]
            for extra in left[1:]:
                w = add(w, extra)
            pieces.append(c.ZerosExpr(shape=(ri, w), dtype=f64))
        pieces.append(x)
        if right:
            w = right[0]
            for extra in right[1:]:
                w = add(w, extra)
            pieces.append(c.ZerosExpr(shape=(ri, w), dtype=f64))
        row_blocks.append(
            pieces[0] if len(pieces) == 1 else c.ConcatExpr(arrays=tuple(pieces), axis=1))
    return [row_blocks[0] if len(row_blocks) == 1
            else c.ConcatExpr(arrays=tuple(row_blocks), axis=0)]


def _render_block_repeat(op: BlockRepeatOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``n`` block-diagonal copies of ``A`` = ``kron(eye(n), A)`` (static ``n``)."""
    eye = _static_eye(low, low.core.IntLit(value=int(op.n)))
    return [low._ns_call("kron", (eye, args[0]))]


def _render_dyn_block_repeat(op: DynBlockRepeatOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``int(n)`` block-diagonal copies of ``A`` = ``kron(eye(int(n)), A)`` (runtime ``n``)."""
    c = low.core
    nn = c.CallExpr(fn=c.Var(name="int"), args=(args[1],))
    eye = _static_eye(low, nn)
    return [low._ns_call("kron", (eye, args[0]))]


# --- Generic array-op lowerings ---------------------------------------------
# ``__pyab_lower__`` hooks for generic array ops; ``_shape_axis_expr``/``_dyn_eye_expr``
# map to the polyarray helpers ``_axis_len``/``_static_eye``.


def _render_dyn_eye(op: DynEyeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Runtime ``eye(ref.shape[axis])`` sized by ``ref``'s (un-batched) axis."""
    return [_static_eye(low, _axis_len(low, args[0], int(op.axis)))]


def _render_dyn_zeros(op: DynZerosOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``zeros((d₀, d₁, …))`` with each ``dᵢ`` a runtime axis of ``refs[i]``."""
    c = low.core
    shape = tuple(_axis_len(low, args[i], int(ax)) for i, ax in enumerate(op.axes))
    return [c.ZerosExpr(shape=shape, dtype=low._dtype_f64())]


def _render_dyn_eye_tensor(op: DynEyeTensorOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``eye(∏dᵢ).reshape(∏dᵢ, d₀, …)`` with each ``dᵢ`` a runtime axis of ``refs[i]``."""
    c = low.core
    dims = [_axis_len(low, args[i], int(ax)) for i, ax in enumerate(op.axes)]
    prod: Any = c.IntLit(value=1)
    for d in dims:
        prod = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=prod, rhs=d)
    prod = builder.assign_new(prod, base="eyeT_prod")   # reuse in eye size AND reshape
    eye = _static_eye(low, prod)
    return [c.ReshapeExpr(a=eye, shape=(prod, *dims))]


def _render_prod_shape(op: ProdShapeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Emit ``static · ∏ refs[i].shape[axes[i]]`` as a runtime 0-d int."""
    c = low.core
    out: Any = c.IntLit(value=int(op.static))
    for i, ax in enumerate(op.axes):
        out = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=out, rhs=_axis_len(low, args[i], int(ax)))
    return [out]


def _render_sum_shape(op: SumShapeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Emit ``static + Σ refs[i].shape[axes[i]]`` as a runtime 0-d int."""
    c = low.core
    out: Any = c.IntLit(value=int(op.static))
    for i, ax in enumerate(op.axes):
        out = c.BinaryExpr(op=c.BinaryOp.ADD, lhs=out, rhs=_axis_len(low, args[i], int(ax)))
    return [out]


def _render_sum_dim(op: SumDimOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``Σ mats[i].shape[axis]`` as a runtime 0-d int."""
    c = low.core
    out: Any = c.IntLit(value=0)
    for x in args:
        out = c.BinaryExpr(op=c.BinaryOp.ADD, lhs=out, rhs=_axis_len(low, x, int(op.axis)))
    return [out]


def _render_prod_dim(op: ProdDimOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``∏ mats[i].shape[axis]`` as a runtime 0-d int."""
    c = low.core
    out: Any = c.IntLit(value=1)
    for x in args:
        out = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=out, rhs=_axis_len(low, x, int(op.axis)))
    return [out]


def _render_scale_axis_dim(op: ScaleAxisDimOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``n · mat.shape[axis]`` as a runtime 0-d int."""
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=c.IntLit(value=int(op.n)),
                         rhs=_axis_len(low, args[0], int(op.axis)))]


def _render_mul_axis_dim(op: MulAxisDimOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``int(n) · mat.shape[axis]`` as a runtime 0-d int."""
    c = low.core
    n = c.CallExpr(fn=c.Var(name="int"), args=(args[0],))
    return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=n, rhs=_axis_len(low, args[1], int(op.axis)))]


def _render_comp_rank(op: CompRankOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``ambient − int(rank)`` as a runtime 0-d int."""
    c = low.core
    rank = c.CallExpr(fn=c.Var(name="int"), args=(args[0],))
    return [c.BinaryExpr(op=c.BinaryOp.SUB, lhs=c.IntLit(value=int(op.ambient)), rhs=rank)]


def _render_hstack(op: HStackOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Concatenate 2-D matrices along axis 1, giving ``[A | B | ...]``."""
    c = low.core
    return [c.ConcatExpr(arrays=tuple(args), axis=1)]


def _render_col_stack(op: ColStackOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Flatten each operand then stack as columns on axis 1."""
    c = low.core
    flat = [c.ReshapeExpr(a=x, shape=(c.IntLit(value=-1),)) for x in args]
    return [c.StackExpr(arrays=tuple(flat), axis=1)]


def _render_scale(op: ScaleOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``factor · x``."""
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=c.FloatLit(value=float(op.factor)), rhs=args[0])]


def _render_scale_by(op: ScaleByOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``s · x`` (runtime scalar)."""
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=args[1], rhs=args[0])]


def _render_add(op: AddOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Left-fold ``x0 + x1 + …`` over the operands."""
    c = low.core
    out = args[0]
    for x in args[1:]:
        out = c.BinaryExpr(op=c.BinaryOp.ADD, lhs=out, rhs=x)
    return [out]


def _render_concat(op: ConcatOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Flatten each operand then concatenate on axis 0."""
    c = low.core
    flat = [c.ReshapeExpr(a=x, shape=(c.IntLit(value=-1),)) for x in args]
    return [c.ConcatExpr(arrays=tuple(flat), axis=0)]


def _render_axis_len(op: AxisLenOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``x.shape[axis]`` as a 0-d int."""
    return [_axis_len(low, args[0], int(op.axis))]


def _render_reshape(op: ReshapeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``A.reshape(shape)`` (static shape)."""
    c = low.core
    return [c.ReshapeExpr(a=args[0], shape=tuple(c.IntLit(value=int(d)) for d in op.shape))]


def _render_const(op: ConstOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Rematerialize the frozen constant as a pyab f64 array literal."""
    arr = np.frombuffer(op.data_bytes, dtype=op.dtype).reshape(op.shape)
    return [low._const_expr(np.asarray(arr, dtype=float))]


def _render_eye(op: EyeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Emit the static ``n x n`` identity, exact in float64."""
    return [_static_eye(low, low.core.IntLit(value=int(op.n)))]


def _render_first_cols(op: FirstColsOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Take the leading ``rank`` columns, ``A[:, :int(rank)]``."""
    c = low.core
    rank = c.CallExpr(fn=c.Var(name="int"), args=(args[1],))
    return [c.AccessExpr(base=args[0], index=(c.Slice(), c.Slice(stop=rank)))]


def _render_last_cols(op: LastColsOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Take the trailing columns, ``A[:, int(rank):]`` — the complement of the leading ones."""
    c = low.core
    rank = c.CallExpr(fn=c.Var(name="int"), args=(args[1],))
    return [c.AccessExpr(base=args[0], index=(c.Slice(), c.Slice(start=rank)))]


# --- Generic array / linalg lowerings ---------------------------------------
# ``__pyab_lower__`` hooks; ``_ns_call``
# routes ``linalg.*`` / ``kron`` / ``sqrt`` through the array namespace (numpy
# AND torch), and ``torch.vmap`` supplies the batch axis around the plain 2-D
# linalg. ``_shape_axis_expr`` maps to the polyarray helper ``_axis_len``.


def _render_project(op: ProjectOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``P.T @ v.reshape(-1)`` — project ambient coords onto the sub-basis."""
    c = low.core
    p_t = c.TransposeExpr(a=args[0], axes=None)
    v_flat = c.ReshapeExpr(a=args[1], shape=(c.IntLit(value=-1),))
    return [c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=p_t, rhs=v_flat)]


def _render_embed(op: EmbedOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``(P @ vsub).reshape(shape)`` — pad sub-coords into the ambient layout."""
    c = low.core
    mm = c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=args[0], rhs=args[1])
    if op.shape:
        return [c.ReshapeExpr(a=mm, shape=tuple(c.IntLit(value=int(d)) for d in op.shape))]
    return [mm]


def _render_kron(op: KronOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Chained Kronecker product ``kron(kron(a0, a1), …)``."""
    acc = args[0]
    for m in args[1:]:
        acc = low._ns_call("kron", (acc, m))
    return [acc]


def _render_kron_free(op: KronFreeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Take a block Kronecker product on the two space axes, outer on the trailing free axes.

    The free axes are disjoint, so they combine as an outer product. Reshapes read the
    runtime block and free dimensions off ``ShapeExpr``, keeping the lowering vmap-safe.
    """
    c = low.core
    f, g = args
    one = c.IntLit(value=1)

    def ax(x: PyExprs, i: int) -> PyExprs:
        return _axis_len(low, x, i)
    df, cf = ax(f, 0), ax(f, 1)
    dg, cg = ax(g, 0), ax(g, 1)
    ff = [ax(f, 2 + i) for i in range(op.nf_free)]
    gg = [ax(g, 2 + i) for i in range(op.ng_free)]
    fr = c.ReshapeExpr(a=f, shape=(df, one, cf, one, *ff, *([one] * op.ng_free)))
    gr = c.ReshapeExpr(a=g, shape=(one, dg, one, cg, *([one] * op.nf_free), *gg))
    prod = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=fr, rhs=gr)
    mul = c.BinaryOp.MUL
    out_shape = (c.BinaryExpr(op=mul, lhs=df, rhs=dg),
                 c.BinaryExpr(op=mul, lhs=cf, rhs=cg), *ff, *gg)
    return [c.ReshapeExpr(a=prod, shape=out_shape)]


def _render_inv_transpose(op: InvTransposeOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``inv(A).T`` — the dual-basis inverse-transpose."""
    c = low.core
    return [c.TransposeExpr(a=low._ns_call("linalg.inv", (args[0],)), axes=None)]


def _render_compose_via_std(op: ComposeViaStdOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """``solve(R_to, R_from)`` — compose two changes-of-basis through the std rep."""
    return [low._ns_call("linalg.solve", (args[0], args[1]))]


def _render_sqrt_spd(op: SqrtSpdOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Emit the SPD square root ``(V·√w)·Vᵀ`` from ``eigh`` of the symmetrized ``G``.

    The SPD guard is a runtime assertion that does not change the returned data, so, as with
    ``AssertOp``, it is omitted from the emitted data path. ``torch.vmap`` batches the plain
    2-D eigh.
    """
    c = low.core
    g = args[0]
    sym = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=c.FloatLit(value=0.5),
                       rhs=c.BinaryExpr(op=c.BinaryOp.ADD, lhs=g,
                                        rhs=c.TransposeExpr(a=g, axes=None)))
    sym = builder.assign_new(sym, base="symG")
    wv, vv = (builder.fresh_var(base=nm) for nm in ("w_spd", "V_spd"))
    builder.assign(c.TupleExpr(elts=(wv, vv)), low._ns_call("linalg.eigh", (sym,)))
    vscaled = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=vv, rhs=low._ns_call("sqrt", (wv,)))
    return [c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=vscaled, rhs=c.TransposeExpr(a=vv, axes=None))]


def _render_rank(op: RankOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Emit the numeric column rank of ``A`` as a 0-d int.

    ``matrix_rank`` has the same name in numpy and torch, but the absolute-tolerance keyword
    differs (``tol=`` versus ``atol=``), so it is passed per namespace.
    """
    c = low.core
    kw = (c.KwArg(name="tol" if low.opts.target != "torch" else "atol",
                  value=c.FloatLit(value=float(op.tol))),)
    return [low._ns_call("linalg.matrix_rank", (args[0],), kw)]


def _render_metric_orthonormal(op: MetricOrthonormalOp, builder: StmtBuilder, args: list[PyExprs], low: _Lowerer) -> list[PyExprs]:
    """Orthonormalize ``A`` in the ``G`` metric: ``A · L⁻ᵀ`` where ``L Lᵀ = AᵀGA``."""
    c = low.core

    def mm(x: PyExprs, y: PyExprs) -> PyExprs:
        return c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=x, rhs=y)
    a, g = args
    at = c.TransposeExpr(a=a, axes=None)
    gram = builder.assign_new(mm(mm(at, g), a), base="gram_mo")
    lchol = low._ns_call("linalg.cholesky", (gram,))
    linvt = c.TransposeExpr(a=low._ns_call("linalg.inv", (lchol,)), axes=None)
    return [mm(a, linvt)]


_ARRAY_OP_LOWERINGS: dict[type, OpLowering] = {
    TransposeOp: _render_transpose,
    SinvFullOp: _render_sinv_full,
    GSvdFullOp: _render_gsvd_full,
    BlockDiagOp: _render_block_diag,
    BlockRepeatOp: _render_block_repeat,
    DynBlockRepeatOp: _render_dyn_block_repeat,
    DynEyeOp: _render_dyn_eye,
    DynZerosOp: _render_dyn_zeros,
    DynEyeTensorOp: _render_dyn_eye_tensor,
    ProdShapeOp: _render_prod_shape,
    SumShapeOp: _render_sum_shape,
    SumDimOp: _render_sum_dim,
    ProdDimOp: _render_prod_dim,
    ScaleAxisDimOp: _render_scale_axis_dim,
    MulAxisDimOp: _render_mul_axis_dim,
    CompRankOp: _render_comp_rank,
    HStackOp: _render_hstack,
    ColStackOp: _render_col_stack,
    ScaleOp: _render_scale,
    ScaleByOp: _render_scale_by,
    AddOp: _render_add,
    ConcatOp: _render_concat,
    AxisLenOp: _render_axis_len,
    ReshapeOp: _render_reshape,
    ConstOp: _render_const,
    EyeOp: _render_eye,
    FirstColsOp: _render_first_cols,
    LastColsOp: _render_last_cols,
    ProjectOp: _render_project,
    EmbedOp: _render_embed,
    KronOp: _render_kron,
    KronFreeOp: _render_kron_free,
    InvTransposeOp: _render_inv_transpose,
    ComposeViaStdOp: _render_compose_via_std,
    SqrtSpdOp: _render_sqrt_spd,
    RankOp: _render_rank,
    MetricOrthonormalOp: _render_metric_orthonormal,
}


# ---------------------------------------------------------------------------
# Helper-def interning: structural fingerprint of a would-be ``_sub_N`` def.
# ---------------------------------------------------------------------------


def _fp_bound_names(core: ModuleType, params: tuple[PyExprs, ...],
                    body: tuple[PyStmts, ...]) -> set[str]:
    """Collect every name bound inside a def: params, assignment targets, binders, nested defs.

    Everything else is free — module globals such as ``torch`` or ``np``, and already-emitted
    helper callees — and must keep its spelling in the fingerprint.

    ⚠ This set is UNSCOPED — a binder anywhere in the body marks its spelling bound for the
    WHOLE body, so an outer FREE global sharing a nested binder's spelling would be
    alpha-normalized and two different globals could fingerprint alike. Today's emitters
    build no nested defs/lambdas (the only ``FunctionDefStmt``s are the module-level helper
    and entry), so the situation is unreachable; :func:`_helper_fingerprint` REFUSES on a
    nested def or lambda rather than rely on that, which makes the claim unconditional.
    """
    bound = {p.name for p in params}

    def collect_target(t: PyExprs) -> None:
        if isinstance(t, core.Var):
            bound.add(t.name)
        elif isinstance(t, core.TupleExpr):
            for el in t.elts:
                collect_target(el)

    def walk(node: object) -> None:
        if isinstance(node, core.AssignStmt):
            collect_target(node.target)
        elif isinstance(node, core.FunctionDefStmt):
            bound.add(node.name)
            for p in node.params:
                bound.add(p.name)
        elif hasattr(core, "LetStmt") and isinstance(node, core.LetStmt):
            bound.add(node.name)
        elif isinstance(node, core.ForStmt):
            collect_target(node.target)
        if isinstance(node, (tuple, list)):
            for c in node:
                walk(c)
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                walk(getattr(node, f.name))

    walk(body)
    return bound


def _helper_fingerprint(core: ModuleType, params: tuple[PyExprs, ...],
                        body: tuple[PyStmts, ...]) -> str | None:
    """Fingerprint a helper def's ``(params, body)`` completely, ignoring bound-name spelling.

    Params carry upstream-minted uid suffixes that differ across content-identical rebuilds,
    and locals ride along in the same normalization. Bound names are replaced by their
    first-occurrence index in a deterministic traversal; free names keep their spelling,
    so equal fingerprints ⇒ alpha-equivalent pure functions over the same module
    globals ⇒ one def can serve every call site. Returns ``None`` (⇒ no interning) on
    any leaf outside the known vocabulary — an unknown object could hide identity that
    the stream would erase. Also refuses on a NESTED def/lambda, whose binders would make
    :func:`_fp_bound_names`' unscoped set alpha-normalize an outer free global of the same
    spelling (unreachable with today's emitters; refusing keeps the invariant
    unconditional rather than contingent on them).
    """
    import hashlib

    def has_nested_scope(node: PyStmts) -> bool:
        if isinstance(node, core.FunctionDefStmt) or (
                hasattr(core, "LambdaExpr") and isinstance(node, core.LambdaExpr)):
            return True
        if isinstance(node, (tuple, list)):
            return any(has_nested_scope(c) for c in node)
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            return any(has_nested_scope(getattr(node, f.name)) for f in dataclasses.fields(node))
        return False

    if has_nested_scope(body):
        return None

    bound = _fp_bound_names(core, params, body)
    index: dict[str, int] = {}
    out: list[str] = []

    def name_tok(n: str) -> str:
        if n not in bound:
            return f"free:{n}"
        if n not in index:
            index[n] = len(index)
        return f"@{index[n]}"

    budget = [_FP_MAX_NODES]

    def ser(node: object) -> None:
        budget[0] -= 1
        if budget[0] < 0:
            raise _FpTooLarge(_FP_MAX_NODES)
        if node is None or isinstance(node, (bool, int, float, str, bytes)):
            out.append(repr(node))
            return
        if isinstance(node, enum.Enum):
            out.append(f"{type(node).__module__}.{type(node).__qualname__}.{node.name}")
            return
        if isinstance(node, (tuple, list)):
            out.append("(" if isinstance(node, tuple) else "[")
            for c in node:
                ser(c)
                out.append(",")
            out.append(")")
            return
        if isinstance(node, np.ndarray):
            a = np.ascontiguousarray(node)
            out.append(f"nd{a.shape}:{a.dtype}:{hashlib.sha256(a.tobytes()).hexdigest()}")
            return
        if isinstance(node, (np.integer, np.floating, np.bool_)):
            out.append(repr(node))
            return
        if isinstance(node, core.Var):
            out.append(f"Var({name_tok(node.name)},{node.locality!r})")
            return
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            t = type(node)
            out.append(f"{t.__module__}.{t.__qualname__}(")
            for f in dataclasses.fields(node):
                v = getattr(node, f.name)
                out.append(f"{f.name}=")
                if f.name == "name" and isinstance(v, str) and isinstance(
                        node, (core.Param, core.FunctionDefStmt)):
                    out.append(name_tok(v))
                elif hasattr(core, "LetStmt") and isinstance(node, core.LetStmt) \
                        and f.name == "name" and isinstance(v, str):
                    out.append(name_tok(v))
                else:
                    ser(v)
                out.append(",")
            out.append(")")
            return
        raise _FpRefuse(f"unserializable leaf {type(node).__qualname__}")

    try:
        ser(params)
        ser(body)
    except (_FpRefuse, RecursionError) as exc:
        # RecursionError: `ser` walks the same tree codegen already recurses on (measured max
        # depth 16 vs a 1000 limit), but a fingerprint must never be the thing that kills a
        # compile that would otherwise succeed — refuse and emit the def uninterned.
        if isinstance(exc, _FpTooLarge):
            # SIZE is the other axis of that same rule. A helper too big to walk is ANNOUNCED,
            # because an uninterned def is a real (if small) cost and a body this size is usually a
            # symptom upstream, not a fact of life.
            warnings.warn(
                f"pyab: helper body exceeds the {_FP_MAX_NODES} node fingerprint budget — emitting "
                "it UNINTERNED (correct, but duplicate helpers are not deduplicated). A body "
                "this large usually means a symbolic operand was threaded into a value kernel "
                "without being grounded first.", stacklevel=2)
        return None
    return hashlib.sha256("|".join(out).encode()).hexdigest()


#: Node budget for :func:`_helper_fingerprint`'s structural walk.
#:
#: A fingerprint exists ONLY to intern a helper, so — exactly as with the ``RecursionError`` guard
#: above — it must never be the thing that makes a compile fail or hang. That guard bounds DEPTH;
#: this bounds SIZE. A helper whose body fits the budget fingerprints in milliseconds; one an order
#: of magnitude past it takes minutes and gigabytes, which is a hang, not a slow path. Refusing
#: costs a duplicate helper; not refusing costs the build.
_FP_MAX_NODES = 2_000_000


class _FpRefuse(Exception):
    """A helper body with no stable structural serialization — skip interning."""


class _FpTooLarge(_FpRefuse):
    """A helper body too large to fingerprint within :data:`_FP_MAX_NODES` — skip interning."""


# ---------------------------------------------------------------------------
# Scalarization — represent a small per-cell intermediate as a GRID of scalar
# exprs (SoA) instead of one stacked tensor Var, so ``torch.vmap`` lifts every
# scalar to a ``(n_cells,)`` array and Inductor fuses the whole per-cell body
# into ONE loop over cells (differently-shaped inner tensors are the only fusion
# boundary once ``bmm``/reduction are gone). Off by default; a Stmt's output is
# scalarized only when ``POLYARRAY_SCALARIZE_MAX`` is set and the output is that
# small (guards the code-size blow-up at high polynomial degree).
# ---------------------------------------------------------------------------


class _ScalarGrid:
    """A statement output emitted as an ``np.ndarray`` of scalar exprs (one per cell), NOT a stacked tensor.

    Returned in a ``_render_op`` payload (``"single"``/``"exprs"``) and recognised by
    :meth:`_Lowerer._bind_payload`, which binds each scalar to its own ``Var`` and points that cell's
    generator at it — so no ``(n_cells, *S)`` buffer is realised and downstream reads are scalar.
    """

    __slots__ = ("grid",)

    def __init__(self, grid: np.ndarray) -> None:
        self.grid = np.asarray(grid, dtype=object)


def _scalarize_max() -> int:
    """Give the size cap under which a statement output is scalarized.

    Max product of a Stmt output's static shape for which it is emitted as a scalar grid rather than a
    stacked tensor (env ``POLYARRAY_SCALARIZE_MAX``; default ``0`` = never scalarize, keep tensor form).
    """
    v = os.environ.get("POLYARRAY_SCALARIZE_MAX")
    if v is not None:
        try:
            return int(v)
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# The lowerer — a structural mirror of numpy_source._Emitter that produces PyAB
# IR nodes (via a StmtBuilder) instead of Python source strings.
# ---------------------------------------------------------------------------

class _Lowerer:
    def __init__(
        self,
        program: Program,
        input_exprs: Mapping[str, PyExprs],
        intatom_exprs: Mapping[str, PyExprs],
        opts: LowerOpts,
        *,
        var_gen: Callable[[str], str] | None = None,
        defs: list[PyStmts] | None = None,
        helpers: dict[int, str] | None = None,
        fp_helpers: dict[str, str] | None = None,
    ) -> None:
        self.core = _core()
        self.prog = program
        self.opts = opts
        self.input_exprs = dict(input_exprs)
        self.intatom_exprs = dict(intatom_exprs)
        # The output SymArray tuple of the op currently being rendered — set at the top of
        # `_render_op` so `try_small_qr` (and op-carried hooks that call it) can read the QR result shape
        # without threading `out` through the hook signature.
        self._current_out: tuple[SymArray, ...] = ()
        # Body statements accumulate here; sub-program / vmap helper defs
        # accumulate in ``self.defs`` (shared across nested lowerers).
        self.b = _builder_mod().StmtBuilder(
            var_gen=var_gen if var_gen is not None else self.core.UniqueVarGen()
        )
        self.defs: list[Any] = defs if defs is not None else []
        self.helpers: dict[int, str] = helpers if helpers is not None else {}
        # Structural-fingerprint intern table (shared like ``helpers``): fingerprint of an
        # emitted helper's (params, body) -> its def name. ``helpers`` memoizes by
        # ``id(prog)`` only, so a content-identical sub-Program REBUILT as a fresh object
        # (closures rebuilt per branch / per functional) re-emitted its whole
        # def — measured 388 emitted defs collapsing to 168 on one assembly's
        # kernels. This interns at emission: one def per distinct body, every call site
        # shares it. See ``_helper_fingerprint`` for the soundness argument.
        self.fp_helpers: dict[str, str] = fp_helpers if fp_helpers is not None else {}
        # generator-name -> scalar PyAB expr producing its value.
        self.genmap: dict[str, Any] = {}
        # bulk binding-name -> PyAB expr (whole tensor).
        self.bulkmap: dict[str, Any] = {}
        # (stmt_idx, out_idx) -> PyAB Var for the produced output.
        self.outvar: dict[tuple[int, int], Any] = {}
        # (stmt_idx, out_idx) -> np.ndarray of scalar exprs for a SCALARIZED output (see ``_ScalarGrid``).
        # Consulted by ``_operand_grid`` (so a scalarized producer feeds a scalarized consumer scalar-wise)
        # and by ``_ref_expr`` (which lazily stacks only if a NON-scalar consumer indexes the whole tensor).
        self.grid_out: dict[tuple[int, int], np.ndarray] = {}
        self._scalarize_max = _scalarize_max()
        # CSE for non-bulk object-dtype operands: id(cells) -> the Var its nested
        # per-cell ``stack`` was assembled into once, so a SymArray reused as an
        # operand across many Stmts is materialised ONCE (a single Var) instead of
        # re-scattered element-wise at every consumption site.  Keyed on the shared
        # ``SymArrayRef._cells`` identity (stable across refs to the same SymArray).
        self._cells_cse: dict[int, Any] = {}
        self.extra = dict(opts.op_lowerings or {})
        # Same dead-key check the numpy emitter runs on its own string-keyed extension point:
        # a private `_Name` for an op that now lives in polyarray as `Name` can never match.
        from .numpy_source import _warn_dead_op_keys
        _warn_dead_op_keys(self.extra, "pyab LowerOpts(op_lowerings=…)")
        self.array_ns = self.core.Var(name="torch" if opts.target == "torch" else "np")
        # In-refs of the Stmt currently being rendered — set by ``_render_op`` just
        # before the ``_ARRAY_OP_LOWERINGS`` table dispatch, so a table renderer that
        # needs operand refs (e.g. ``GSvdFullOp`` reusing ``_gsvd``) can read them.
        self._cur_in_refs: Any = ()

    # -- namespace helpers ------------------------------------------------

    def _ns(self, path: str) -> PyExprs:
        """Build ``<ns>.a.b.c`` as a MemberExpr chain (e.g. ``torch.linalg.qr``)."""
        node = self.array_ns
        for part in path.split("."):
            node = self.core.MemberExpr(base=node, name=part)
        return node

    def _ns_call(self, path: str, args: Sequence[PyExprs],
                 kwargs: Sequence[tuple[str, PyExprs]] = ()) -> PyExprs:
        return self.core.CallExpr(fn=self._ns(path), args=tuple(args), kwargs=tuple(kwargs))

    def _dtype_f64(self) -> PyExprs:
        """Return the backend's float64 dtype expression."""
        if not self.opts.dtype_f64:
            return None
        from pyarraybackend.ir.types import f64

        return f64

    # -- driver: emit the program body, bind inputs, return output exprs --

    def run(self) -> list[PyExprs]:
        """Lower every statement and return one expression per program output."""
        self._bind_inputs()
        for stmt_idx, stmt in enumerate(self.prog.statements):
            self._emit_stmt(stmt_idx, stmt)
        return self._output_exprs()

    def body_stmts(self) -> tuple[PyStmts, ...]:
        """Return the statements emitted so far, without resetting the builder."""
        return self.b.finish(reset=False)

    # -- inputs -----------------------------------------------------------

    def _bind_inputs(self) -> None:
        for inp in self.prog.inputs:
            base = self.input_exprs[inp.name]
            if is_dynamic(inp.shape):
                self.bulkmap[inp.name] = base
                continue
            cells = self.prog.input_arrays[inp.name].cells
            # Past the is_dynamic guard every axis is a concrete int.
            shape = cast("tuple[int, ...]", tuple(inp.shape))
            for idx in (np.ndindex(*shape) if shape else [()]):
                cell = cells[idx] if shape else cells[()]
                if isinstance(cell, RationalFunction):
                    self.genmap[cell.gens[0]] = self._index(base, idx)

    # -- statements -------------------------------------------------------

    def _emit_stmt(self, stmt_idx: int, stmt: Stmt) -> None:
        if stmt.fn is None:
            return
        arg_exprs = [self._ref_expr(r) for r in stmt.in_]
        kind, payload = self._render_op(stmt.fn, arg_exprs, stmt.out, stmt.in_)
        out = stmt.out
        base = f"t{stmt_idx}"
        if kind == "single":
            self._bind_payload(stmt_idx, 0, out[0], payload, base)
        elif kind == "tuple":
            tvars = [self.b.fresh_var(base=f"{base}_{k}") for k in range(len(out))]
            self.b.assign(self.core.TupleExpr(elts=tuple(tvars)), payload)
            for k, bound in enumerate(out):
                self._bind_output(stmt_idx, k, bound, tvars[k])
        elif kind == "exprs":
            for k, (bound, expr) in enumerate(zip(out, payload)):
                self._bind_payload(stmt_idx, k, bound, expr, f"{base}_{k}")
        else:  # pragma: no cover - defensive
            raise AssertionError(kind)

    def _bind_payload(self, stmt_idx: int, out_idx: int, bound: SymArray,
                      payload: PyExprs, base: str) -> None:
        """Bind one statement output.

        A stacked tensor Var, or (``_ScalarGrid``) a grid of per-cell scalar Vars with no tensor buffer.
        """
        if isinstance(payload, _ScalarGrid):
            self._bind_grid(stmt_idx, out_idx, bound, payload.grid, base)
        else:
            var = self.b.assign_new(payload, base=base)
            self._bind_output(stmt_idx, out_idx, bound, var)

    def _bind_output(self, stmt_idx: int, out_idx: int, bound: SymArray, var: PyExprs) -> None:
        """Record the expression a statement output binds to, per cell or as a whole tensor."""
        self.outvar[(stmt_idx, out_idx)] = var
        if bound._bulk is not None:
            self.bulkmap[bound._bulk.name] = var
            return
        cells = bound.cells
        shape = cells.shape
        for idx in (np.ndindex(*shape) if shape else [()]):
            cell = cells[idx] if shape else cells[()]
            if isinstance(cell, RationalFunction):
                self.genmap[cell.gens[0]] = self._index(var, idx)

    def _bind_grid(self, stmt_idx: int, out_idx: int, bound: SymArray,
                   grid: np.ndarray, base: str) -> None:
        """Bind a SCALARIZED output.

        Assign each grid element to its own scalar Var, record the grid (so a scalarized consumer reads it
        component-wise), and point each output cell's generator at its Var — so NO ``(n_cells, *S)`` buffer
        is realised. A tensor consumer (rare) triggers a lazy stack in :meth:`_ref_expr`.
        """
        cells = bound.cells
        if grid.shape != cells.shape:
            raise AssertionError(
                f"pyab: scalar-grid shape {grid.shape} != output cells shape {cells.shape}"
            )
        gvars = np.empty(grid.shape, dtype=object)
        for idx in (np.ndindex(*grid.shape) if grid.shape else [()]):
            e = grid[idx] if grid.shape else grid[()]
            gvars[idx] = self.b.assign_new(e, base=f"{base}_s")
        self.grid_out[(stmt_idx, out_idx)] = gvars
        if bound._bulk is not None:
            # A whole-tensor (bulk) output is read via ``bulkmap`` (not per-cell atoms), so assemble the grid
            # into one stacked tensor there. This is the single unavoidable materialisation of a scalarized
            # output that flows on as a bulk tensor (e.g. the value kernel's final block).
            self.bulkmap[bound._bulk.name] = self.b.assign_new(self._stack_grid(gvars), base=f"{base}_bulk")
            return
        for idx in (np.ndindex(*cells.shape) if cells.shape else [()]):
            cell = cells[idx] if cells.shape else cells[()]
            if isinstance(cell, RationalFunction):
                self.genmap[cell.gens[0]] = gvars[idx]

    def _stack_grid(self, g: np.ndarray) -> PyExprs:
        """Assemble a scalar grid into a stacked backend tensor (only when a tensor consumer needs it)."""
        c = self.core
        if g.ndim == 0:
            return _as_tensor(self, g[()])
        if g.ndim == 1:
            return c.StackExpr(arrays=tuple(_as_tensor(self, g[i]) for i in range(g.shape[0])), axis=0)
        return c.StackExpr(arrays=tuple(self._stack_grid(g[i]) for i in range(g.shape[0])), axis=0)

    # -- ref resolution (mirrors Program._resolve_ref) --------------------

    def _ref_expr(self, ref: Ref) -> PyExprs:
        if isinstance(ref, InputRef):
            return self._index(self.input_exprs[ref.name], ref.indices)
        if isinstance(ref, OutputRef):
            key = (ref.stmt_idx, ref.out_idx)
            g = self.grid_out.get(key)
            if g is not None:
                if len(ref.indices) == g.ndim:
                    return g[ref.indices]                     # fully indexed → the scalar Var directly
                if key not in self.outvar:                    # a tensor consumer: stack the grid once, lazily
                    self.outvar[key] = self.b.assign_new(self._stack_grid(g), base=f"t{ref.stmt_idx}_stk")
            return self._index(self.outvar[key], ref.indices)
        if isinstance(ref, Const):
            return self._const_expr(ref.value)
        if isinstance(ref, RationalRef):
            return self._rf_expr(ref.rf)
        if isinstance(ref, SymArrayRef):
            if ref._bulk is not None:
                return self.bulkmap[ref._bulk.name]
            return self._cells_expr_shared(ref._cells)
        if isinstance(ref, IntAtomRef):
            return self.core.CallExpr(
                fn=self.core.Var(name="int"), args=(self.intatom_exprs[ref.name],)
            )
        raise TypeError(f"pyab: unknown Ref {type(ref).__name__}")

    def _index(self, base: PyExprs, idx: tuple[int, ...]) -> PyExprs:
        if not idx:
            return base
        return self.core.AccessExpr(
            base=base, index=tuple(self.core.IntLit(value=int(i)) for i in idx)
        )

    # -- op dispatch ------------------------------------------------------

    def _render_op(
        self, fn: StmtOp, args: list[PyExprs], out: tuple[SymArray, ...],
        in_refs: Sequence[Ref] = ()
    ) -> tuple[str, PyExprs | list[PyExprs]]:
        c = self.core
        self._current_out = out          # so try_small_qr / hooks can read this op's output shape
        self._cur_in_refs = in_refs      # so hooks (QrSignFix) / the scalar-einsum can read operand refs
        # Front-end ops first (keyed by class name), then the builtin vocabulary.
        renderer = self.extra.get(type(fn).__name__)
        if renderer is not None:
            return ("exprs", renderer(fn, self.b, args, self))
        # polyarray-owned array-op lowerings (type-keyed, no string dispatch): the
        # relocated generic array builtins (transpose / block-diag / GSVD-full / …).
        # Consulted after the front-end override and before the isinstance vocabulary.
        array_renderer = _ARRAY_OP_LOWERINGS.get(type(fn))
        if array_renderer is not None:
            self._cur_in_refs = in_refs
            return ("exprs", array_renderer(fn, self.b, args, self))

        if isinstance(fn, DetOp):
            return ("single", self._ns_call("linalg.det", args))
        if isinstance(fn, InvOp):
            return ("single", self._ns_call("linalg.inv", args))
        if isinstance(fn, PinvOp):
            return ("single", self._ns_call("linalg.pinv", args))
        if isinstance(fn, SolveOp):
            return ("single", self._ns_call("linalg.solve", args))
        if isinstance(fn, SqrtOp):
            return ("single", self._ns_call("sqrt", args))
        if isinstance(fn, AbsOp):
            return ("single", self._ns_call("abs", args))
        if isinstance(fn, SignOp):
            return ("single", self._ns_call("sign", args))
        if isinstance(fn, MoveaxisOp):
            return (
                "single",
                self._ns_call(
                    "moveaxis",
                    (args[0], _pylit(c, fn.source), _pylit(c, fn.destination)),
                ),
            )
        if isinstance(fn, TensordotOp):
            return ("single", c.TensorDotExpr(a=args[0], b=args[1], axes=fn.axes))
        if isinstance(fn, EinsumStmtOp):
            shapes = [_ref_shape(r) for r in in_refs] if len(in_refs) == len(args) else [None] * len(args)
            if self._scalarize_max > 0 and len(in_refs) == len(args):
                # Assembly fold ``table·weight``: read the weight scalar-wise (no stack) so the per-cell body
                # stays ONE loop. Tried before scalarize/unroll; keeps the (bulk) output a tensor.
                fw = _fold_weighted_einsum(self, fn.spec, in_refs, self._out_shape())
                if fw is not None:
                    return ("single", fw)
            if self._want_scalarize(self._out_shape()) and len(in_refs) == len(args):
                g = _scalar_einsum(self, fn.spec, [self._operand_grid(r) for r in in_refs], self._out_shape())
                if g is not None:
                    return ("single", g)
            u = _unroll_einsum(self, fn.spec, tuple(args), shapes, self._out_shape())
            if u is not None:
                return ("single", u)
            return ("single", c.EinsumExpr(subscripts=fn.spec, operands=tuple(args)))
        if isinstance(fn, EinsumOp):
            rhs_arr = np.asarray(fn._rhs, dtype=float)
            lhs_shape = _ref_shape(in_refs[0]) if in_refs else None
            if self._want_scalarize(self._out_shape()) and in_refs:
                g = _scalar_einsum(self, fn.spec,
                                   [self._operand_grid(in_refs[0]), _const_grid(c, rhs_arr)], self._out_shape())
                if g is not None:
                    return ("single", g)
            rhs = self._const_expr(rhs_arr)
            u = _unroll_einsum(self, fn.spec, (args[0], rhs),
                               [lhs_shape, tuple(int(s) for s in fn.rhs_shape)], self._out_shape())
            if u is not None:
                return ("single", u)
            return ("single", c.EinsumExpr(subscripts=fn.spec, operands=(args[0], rhs)))
        if isinstance(fn, IdentityOp):
            return ("single", args[0])
        if isinstance(fn, AssertOp):
            # Predicate guard; passthrough the first operand (matches numpy_source).
            return ("single", args[0])
        if isinstance(fn, SwitchOp):
            scrutinee_ref = in_refs[0] if in_refs else None
            return ("single", self._switch_expr(fn, args, scrutinee_ref))
        if isinstance(fn, QrOp):
            return self._qr(fn, args, out)
        if isinstance(fn, SvdOp):
            return ("exprs", self._svd(fn, args, out))
        if isinstance(fn, GSvdOp):
            return ("exprs", self._gsvd(fn, args, out, in_refs))
        if isinstance(fn, WhileOp):
            return self._while(fn, args, out)
        if isinstance(fn, CallOp):
            return self._render_op(fn.fn, args, out, in_refs)
        if isinstance(fn, Program):
            return self._call_subprogram(fn, args, out)
        vbody = _vmap_body_of(fn)
        if vbody is not None:
            return self._call_vmap(fn, vbody, args, out)
        # Op-carried lowering hook (the sanctioned twin of ``numpy_source``'s
        # ``__numpy_source__``): a front-end op above polyarray defines
        # ``__pyab_lower__(self, builder, args, low) -> list[expr]`` on its class
        # and pyab discovers it here — ONE canonical path, no ``op_lowerings``
        # dict threading. Consulted LAST (after the explicit ``op_lowerings``
        # override at line ~319, the builtin isinstance vocabulary, and the
        # CallOp/Program/vmap-closure unwraps), so a native op's canonical
        # lowering can never be shadowed by a stray hook. Renderers use only the
        # passed-in ``low`` (``low.core``, ``low._const_expr``, ``low._dtype_f64``)
        # — never importing pyab from the op's module — keeping the layering clean.
        hook = getattr(fn, "__pyab_lower__", None)
        if hook is not None:
            return ("exprs", hook(self.b, args, self))
        raise NotImplementedError(
            f"pyab: no lowering for op {type(fn).__name__!r}. "
            f"Define ``{type(fn).__name__}.__pyab_lower__(self, builder, args, low)`` "
            "on the op's class (the canonical op-carried hook), or supply "
            "LowerOpts(op_lowerings={'"
            f"{type(fn).__name__}': ...}}), or extend pyab._Lowerer._render_op."
        )

    def _unwrap_int_call(self, expr: PyExprs) -> PyExprs:
        """Strip the ``int(...)`` wrapper pyab puts on an :class:`IntAtomRef`.

        :meth:`_ref_expr` renders an :class:`~polyarray.ir.IntAtomRef` scrutinee as
        ``CallExpr(fn=Var("int"), args=(o_var,))`` — the ``int(tensor)`` is exactly
        what breaks ``torch.vmap`` (it calls ``.item()`` on a batched tensor).
        Recover the raw ``o_var`` so the one-hot switch lowers as pure arithmetic.
        Passthrough if not so wrapped.
        """
        c = self.core
        if isinstance(expr, c.CallExpr) and isinstance(expr.fn, c.Var) and expr.fn.name == "int":
            return expr.args[0]
        return expr

    def _switch_expr(self, fn: SwitchOp, args: list[PyExprs],
                     scrutinee_ref: Ref | None = None) -> PyExprs:
        """Canonical :class:`~polyarray.ir.SwitchOp` lowering — ``branch[scrutinee]``.

        Two lanes, chosen by whether the scrutinee is STATICALLY a concrete int:

        * **Concrete-int fast path** — the scrutinee ref is a :class:`~polyarray.ir.Const`
          holding an integer scalar (a bound compile-time value, NOT a batched/dynamic
          ref). Emit the direct branch pick ``branches[k]`` — preserving that branch's
          exact shape AND dtype (heterogeneous branches are fine; no f64 cast).

        * **One-hot vmap-safe default** — for a dynamic/batched scrutinee (an
          :class:`~polyarray.ir.IntAtomRef` — the per-cell orientation id, a BATCHED
          integer tensor under ``torch.vmap``) the direct pick's ``int(scrutinee)`` /
          advanced index is illegal (it ``.item()``s a batched tensor). Instead stack
          the ``N`` branches → ``(N, *S)``, build a one-hot ``arange(N) == scrutinee``
          (broadcasts a batched 0-d scrutinee to a batched ``(N,)`` selector), cast to
          f64, and contract the branch axis by ``einsum('n,n...->...')`` — pure
          arithmetic, so it lowers under ``torch.vmap`` (and works unbatched too).
          This is the safe default whenever the scrutinee is NOT a static ``Const``:
          pyab cannot know if the caller will vmap, and the one-hot is universally
          correct (it inherently needs the uniform branch shape that the vmap case has).
          Branch order matches the ``select_x`` scrutinee domain.
        """
        c = self.core
        if isinstance(scrutinee_ref, Const):
            v = scrutinee_ref.value
            arr = np.asarray(v)
            if arr.ndim == 0 and np.issubdtype(arr.dtype, np.integer):
                k = int(arr)
                return args[1 + k]                      # direct branch pick (exact dtype/shape)
        # Dynamic / batched scrutinee ⇒ one-hot · stacked branches (vmap-safe).
        scrutinee = self._unwrap_int_call(args[0])
        branch_exprs = tuple(args[1:])
        n = fn.n_branches
        if self._scalarize_max > 0:
            # UNROLLED one-hot select ``Σ_k (k == scrutinee)·branch_k`` — pure arithmetic, no ``stack`` and no
            # ``einsum('n,n...->...')``. The einsum path stacks the branches into an ``(N, *S)`` tensor and
            # contracts, which Inductor turns into an extern ``bmm`` (a fusion boundary that splits the per-cell
            # body). Each ``(k == scrutinee)`` is a batched 0-d bool → f64 scalar that broadcasts over its
            # branch; summed, it is the same value the one-hot contraction produces (vmap-safe, unbatched too).
            terms = [
                c.BinaryExpr(
                    op=c.BinaryOp.MUL,
                    lhs=branch_exprs[k],
                    rhs=c.ArrayExpr(
                        obj=c.BinaryExpr(op=c.BinaryOp.EQ, lhs=c.IntLit(value=k), rhs=scrutinee),
                        dtype=self._dtype_f64(),
                    ),
                )
                for k in range(n)
            ]
            return _balanced(c, terms, c.BinaryOp.ADD)
        stacked = c.StackExpr(arrays=branch_exprs, axis=0)                         # (N, *S)
        idx = c.ArangeExpr(start=c.IntLit(value=0), stop=c.IntLit(value=int(n)),
                           step=c.IntLit(value=1), dtype=None)                     # (N,) int
        eq = c.BinaryExpr(op=c.BinaryOp.EQ, lhs=idx, rhs=scrutinee)                # (N,) bool selector
        onehot = c.ArrayExpr(obj=eq, dtype=self._dtype_f64())                      # f64, matches branches
        return c.EinsumExpr(subscripts="n,n...->...", operands=(onehot, stacked))

    # -- SVD / GSVD (data-dependent rank -> eager fusion boundary) ---------
    #
    # ``rank = (S > tol).sum()`` is genuinely runtime, so the rank-dependent
    # split (``U[:, :rank]`` etc.) uses ``int(rank)`` — correct in eager
    # torch / numpy, but a graph break under ``@torch.compile``. Emit these in a
    # plain (undecorated) callable and they run eager; only ``place="fuse"``
    # across the rank hits the break.

    def _reduce(self, a: PyExprs, op: PyExprs) -> PyExprs:
        """Reduce ``a`` over every axis with ``op``."""
        return self.core.ReduceExpr(a=a, op=op, axis=None, keepdims=False)

    def _svd_rank(self, s_var: PyExprs, max_mn: int, rcond: float | None) -> PyExprs:
        """Rank = number of singular values above numpy's matrix_rank tolerance."""
        c = self.core
        smax = self.b.assign_new(self._reduce(s_var, c.ReduceOp.MAX), base="smax")
        if rcond is None:
            eps = float(np.finfo(float).eps)
            tol: Any = c.BinaryExpr(
                op=c.BinaryOp.MUL, lhs=smax, rhs=c.FloatLit(value=max_mn * eps)
            )
        else:
            tol = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=c.FloatLit(value=float(rcond)), rhs=smax)
        gt = c.BinaryExpr(op=c.BinaryOp.GT, lhs=s_var, rhs=tol)
        return self._reduce(gt, c.ReduceOp.SUM)

    def _svd(self, fn: SvdOp, args: list[PyExprs], out: tuple[SymArray, ...]) -> list[PyExprs]:
        c = self.core
        # U, S, Vh are static-shape; recover m, n from the U / Vh output specs.
        m = int(out[0].cells.shape[0])
        n = int(out[2].cells.shape[-1])
        uv, sv, vhv = (self.b.fresh_var(base=b) for b in ("U", "S", "Vh"))
        svd_call = self._ns_call(
            "linalg.svd",
            (args[0],),
            (c.KwArg(name="full_matrices", value=c.BoolLit(value=fn.full_matrices)),),
        )
        self.b.assign(c.TupleExpr(elts=(uv, sv, vhv)), svd_call)
        rankv = self.b.assign_new(self._svd_rank(sv, max(m, n), fn.rcond), base="rank")
        return [uv, sv, vhv, rankv]

    def _gsvd(self, fn: GSvdOp, args: list[PyExprs], out: tuple[SymArray, ...],
              in_refs: Sequence[Ref]) -> list[PyExprs]:
        """Metric-aware GSVD as a composite of cholesky / svd / solve + rank split.

        Mirrors :meth:`polyarray.ir.GSvdOp.__call__` exactly: whiten ``A`` by the
        Cholesky factors of the two metrics, SVD, threshold the rank, de-whiten,
        then split ``[U|UI]`` / ``[V|VI]`` at the rank.
        """
        c = self.core
        A, M_V, M_W = args
        n_w = int(self._ref_shape(in_refs[2])[0])
        n_v = int(self._ref_shape(in_refs[1])[0])
        # Identity-metric fold: cholesky(I)=I and solve(I,·)=·, so an identity
        # metric drops its whitening entirely (M_V=M_W=I collapses GSVD to SVD).
        id_w = self.opts.simplify_gsvd and self._ref_is_identity(in_refs[2], n_w)
        id_v = self.opts.simplify_gsvd and self._ref_is_identity(in_refs[1], n_v)

        def new(e: PyExprs, base: str) -> PyExprs:
            return self.b.assign_new(e, base=base)

        def mm(x: PyExprs, y: PyExprs) -> PyExprs:
            return c.BinaryExpr(op=c.BinaryOp.MATMUL, lhs=x, rhs=y)

        lwt = None if id_w else new(c.TransposeExpr(a=new(self._ns_call("linalg.cholesky", (M_W,)), "Lw"), axes=None), "LwT")
        rwt = None if id_v else new(c.TransposeExpr(a=new(self._ns_call("linalg.cholesky", (M_V,)), "Rw"), axes=None), "RwT")
        # Aw = Lw^T @ A @ inv(Rw^T), with identity factors dropped.
        aw_expr: Any = A
        if lwt is not None:
            aw_expr = mm(lwt, aw_expr)
        if rwt is not None:
            aw_expr = mm(aw_expr, self._ns_call("linalg.inv", (rwt,)))
        aw = new(aw_expr, "Aw")
        utv, sv, vthv = (self.b.fresh_var(base=b) for b in ("Ut", "S", "Vth"))
        self.b.assign(
            c.TupleExpr(elts=(utv, sv, vthv)),
            self._ns_call(
                "linalg.svd", (aw,), (c.KwArg(name="full_matrices", value=c.BoolLit(value=True)),)
            ),
        )
        rankv = new(self._svd_rank(sv, max(n_w, n_v), fn.rcond), "rank")
        # De-whiten: [U|UI] = solve(Lw^T, Ut); [V|VI] = solve(Rw^T, Vth^T).
        u_full = utv if lwt is None else new(self._ns_call("linalg.solve", (lwt, utv)), "Ufull")
        vth_t = c.TransposeExpr(a=vthv, axes=None)
        v_full = new(vth_t, "Vfull") if rwt is None else new(self._ns_call("linalg.solve", (rwt, vth_t)), "Vfull")
        ir = new(c.CallExpr(fn=c.Var(name="int"), args=(rankv,)), "r")
        colon = c.Slice()
        head = c.Slice(stop=ir)
        tail = c.Slice(start=ir)
        u = c.AccessExpr(base=u_full, index=(colon, head))
        ui = c.AccessExpr(base=u_full, index=(colon, tail))
        v = c.AccessExpr(base=v_full, index=(colon, head))
        vi = c.AccessExpr(base=v_full, index=(colon, tail))
        return [u, ui, v, vi, sv, rankv]

    def _ref_shape(self, ref: Ref) -> tuple[int, ...]:
        """Return an operand ref's static shape, as the rank tolerance and dimensions need it."""
        if isinstance(ref, SymArrayRef):
            base = ref._bulk.shape if ref._bulk is not None else ref.cells.shape
            return tuple(base)
        if isinstance(ref, InputRef):
            sa = self.prog.input_arrays[ref.name]
            base = sa._bulk.shape if sa._bulk is not None else sa.cells.shape
            return tuple(base)[len(ref.indices):]
        if isinstance(ref, OutputRef):
            st = self.prog.statements[ref.stmt_idx].out[ref.out_idx]
            base = st._bulk.shape if st._bulk is not None else st.cells.shape
            return tuple(base)[len(ref.indices):]
        if isinstance(ref, Const):
            return tuple(np.asarray(ref.value).shape)
        raise TypeError(f"pyab: cannot take a static shape of {type(ref).__name__}")

    def _ref_is_identity(self, ref: Ref, n: int) -> bool:
        """Report whether ``ref`` is a numerically-known ``n x n`` identity matrix."""
        if isinstance(ref, Const):
            val = np.asarray(ref.value)
            return val.shape == (n, n) and np.allclose(val, np.eye(n))
        if isinstance(ref, SymArrayRef) and ref._bulk is None:
            cells = np.asarray(ref.cells)
            if cells.shape != (n, n) or cells.dtype.kind != "f":
                return False
            return np.allclose(cells, np.eye(n))
        return False

    # -- WhileOp -> WhileHop (torch.while_loop / functional.while_loop) -----

    def _while(self, fn: WhileOp, args: list[PyExprs], out: tuple[SymArray, ...]) -> tuple[str, PyExprs]:
        c = self.core
        if not isinstance(fn.cond, Program) or not isinstance(fn.body, Program):
            raise NotImplementedError(
                "pyab: WhileOp cond/body must be sub-Programs; opaque Python "
                "callables cannot be lowered."
            )
        cond_name = self._helper_for_program(fn.cond)
        body_name = self._helper_for_program(fn.body)
        hop = c.WhileHop(
            cond_fn=c.Var(name=cond_name),
            body_fn=c.Var(name=body_name),
            carry=tuple(args),
        )
        wv = self.b.assign_new(hop, base="while")
        # torch.while_loop / functional.while_loop return the final carry tuple.
        return ("exprs", [c.AccessExpr(base=wv, index=(c.IntLit(value=i),)) for i in range(len(out))])

    # -- QR (small-Householder interception) ------------------------------

    def _out_shape(self) -> tuple[int, ...] | None:
        """Give the static shape of the single-output op currently being rendered.

        Read from ``self._current_out``; used by the einsum unroll to infer index sizes the operands alone
        may not fix. ``None`` if multi-output or not statically shaped.
        """
        co = self._current_out
        if not co or len(co) != 1:
            return None
        try:
            return tuple(int(s) for s in co[0].cells.shape)
        except Exception:
            return None

    def try_small_qr(self, a_expr: PyExprs, mode: str = "reduced") -> tuple[PyExprs, PyExprs] | None:
        """Inline an unrolled Householder QR of the CURRENTLY-rendered op's operand.

        Arith/``where``/``stack`` only — NO ``linalg.qr`` call, so it neither graph-breaks
        ``torch.compile`` nor calls LAPACK. Returns ``None`` to fall back to a ``linalg.qr`` call.

        Shared by :meth:`_qr` (``QrOp``) and op-carried ``__pyab_lower__`` hooks (e.g.
        ``QrSignFixOp``) so BOTH avoid the QR graph break. The operand shape is read off the current
        output tuple (``self._current_out``); intercept iff it is statically small (``max(m, n) <=
        small_qr.max_dim``, ``m >= n``) and small-QR is enabled.
        """
        sq = self.opts.small_qr
        m, n = _static_matrix_shape(self._current_out)
        if (
            sq.enabled
            and m is not None
            and n is not None
            and m >= n
            and max(m, n) <= sq.max_dim
            and mode in ("reduced", "complete")
        ):
            return _emit_householder_qr(self, a_expr, m, n, mode)
        return None

    def _qr(self, fn: QrOp, args: list[PyExprs], out: tuple[SymArray, ...]) -> tuple[str, PyExprs]:
        inline = self.try_small_qr(args[0], fn.mode)
        if inline is not None:
            return ("exprs", [inline[0], inline[1]])
        kwargs = (self.core.KwArg(name="mode", value=self.core.StringLit(value=fn.mode)),)
        return ("tuple", self._ns_call("linalg.qr", args, kwargs))

    # -- scalarization support --------------------------------------------

    def _want_scalarize(self, shape: tuple[int, ...] | None) -> bool:
        """Report whether the current output should be scalarized.

        True iff scalarization is enabled, ``shape`` is statically known and small enough
        (``prod(shape) <= POLYARRAY_SCALARIZE_MAX``), and the current output is NOT bulk.

        A bulk output is read as a WHOLE tensor by its consumers (via ``bulkmap``), so scalarizing it would
        only force an immediate re-stack — no fewer buffers, and it splits what Inductor would otherwise keep
        as one clean output loop (measured: scalarizing the value kernel's final bulk fold went 2 → 13 loops).
        Scalarize only outputs whose consumers read cells (→ scalar Vars) so the tensor never forms.
        """
        if self._scalarize_max <= 0 or shape is None:
            return False
        co = self._current_out
        if co and any(sa._bulk is not None for sa in co):
            return False
        size = 1
        for s in shape:
            size *= int(s)
        return 0 < size <= self._scalarize_max

    def scalar_grid(self, grid: np.ndarray) -> _ScalarGrid:
        """Wrap an ``np.ndarray`` of scalar exprs as a scalarized output.

        For op-carried ``__pyab_lower__`` hooks — e.g. ``QrSignFixOp`` — to return without
        importing pyab internals.
        """
        return _ScalarGrid(grid)

    def _operand_grid(self, ref: Ref) -> np.ndarray:
        """Resolve an operand ``ref`` to an ``np.ndarray`` of scalar exprs — one per element.

        So a scalarized op reads its inputs component-wise (no intermediate tensor). Feeds a scalarized
        producer (``grid_out``) straight through; otherwise reads a tensor/const/cells element-by-element.
        """
        c = self.core
        if isinstance(ref, Const):
            return _const_grid(c, np.asarray(ref.value, dtype=float))
        if isinstance(ref, SymArrayRef):
            if ref._bulk is not None:                    # a whole-tensor operand: int-index it component-wise
                shp = _ref_shape(ref)                    # None if any axis is dynamic (a DimAtom)
                if shp is None:
                    return np.array(self._ref_expr(ref), dtype=object)
                return self._tensor_grid(self.bulkmap[ref._bulk.name], shp)
            cells = np.asarray(ref._cells, dtype=object)
            g = np.empty(cells.shape, dtype=object)
            for idx in np.ndindex(*cells.shape) if cells.shape else [()]:
                cell = cells[idx] if cells.shape else cells[()]
                g[idx] = (self._rf_expr(cell) if isinstance(cell, RationalFunction)
                          else c.FloatLit(value=float(cell)))
            return g
        if isinstance(ref, OutputRef):
            key = (ref.stmt_idx, ref.out_idx)
            sub = self.grid_out.get(key)
            if sub is not None:
                return sub[ref.indices] if ref.indices else sub
            base = self._index(self.outvar[key], ref.indices)
            return self._tensor_grid(base, self._ref_shape(ref))
        if isinstance(ref, InputRef):
            base = self._index(self.input_exprs[ref.name], ref.indices)
            return self._tensor_grid(base, self._ref_shape(ref))
        # RationalRef / IntAtomRef and any other ref: fall back to a single scalar expr.
        return np.array(self._ref_expr(ref), dtype=object)

    def _tensor_grid(self, base: PyExprs, shape: tuple[int, ...]) -> np.ndarray:
        """Build a grid of all-integer subscripts of a tensor ``base`` — each element a scalar load."""
        c = self.core
        g = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            g[idx] = (base if not idx
                      else c.AccessExpr(base=base, index=tuple(c.IntLit(value=int(i)) for i in idx)))
        return g

    def try_small_qr_grid(self, mode: str = "reduced") -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(Q_grid, R_grid)`` — the scalar-grid twin of :meth:`try_small_qr`.

        ``np.ndarray``s of scalar exprs (no ``stack``), reading the current op's first operand
        component-wise from ``_cur_in_refs``. ``None`` when small-QR is disabled, the matrix is too
        large for ``sq.max_dim``, ``mode`` is not reduced/complete, the operand ref is unavailable or
        its grid is not ``(m, n)``, or an explicit ``POLYARRAY_SCALARIZE_MAX`` cap excludes it.
        """
        sq = self.opts.small_qr
        m, n = _static_matrix_shape(self._current_out)
        if not (sq.enabled and m is not None and n is not None and m >= n
                and max(m, n) <= sq.max_dim and mode in ("reduced", "complete")):
            return None
        # `_scalarize_max` is an OPTIONAL extra cap here, not an on/off switch. Its default is 0,
        # documented as "never scalarize", so reading 0 as "disabled" made this whole path dead
        # under stock environment however `small_qr` was configured — and `small_qr` is enabled by
        # default, so a reader configuring it would reasonably expect the grid path to fire. The
        # size guard it was meant to provide is already implied: `max(m, n) <= sq.max_dim` bounds
        # the emitted grid at `sq.max_dim ** 2` scalars (16 at the default 4). The tensor twin
        # `try_small_qr` carries no such gate. So honour an explicitly-set cap and nothing more.
        if self._scalarize_max > 0 and m * m > self._scalarize_max:
            return None
        refs = self._cur_in_refs
        if not refs:
            return None
        a_grid = self._operand_grid(refs[0])
        if a_grid.shape != (m, n):
            return None
        return _emit_householder_qr_grid(self, lambda i, j: a_grid[i, j], m, n, mode)

    # -- sub-program / vmap helper defs -----------------------------------

    def _helper_for_program(self, prog: Program) -> str:
        """Emit ``prog`` as a module-level ``def`` helper; return its name."""
        key = id(prog)
        existing = self.helpers.get(key)
        if existing is not None:
            return existing
        c = self.core
        name = f"_sub_{len(self.helpers)}"
        self.helpers[key] = name  # reserve before recursing
        # Params = the sub-program's own inputs, THEN its `int_atoms` (switch scrutinees) as trailing
        # params — mirroring the top-level lowering (`as_function_def`: inputs then int_atoms). Without
        # this a grafted sub-Program carrying an `o_<flat_id>` σ scrutinee lowers with an empty
        # `intatom_exprs` and its `IntAtomRef` KeyErrors; the caller threads the scrutinee in
        # (`_call_subprogram`), so a merged value kernel can carry per-DOF orientation switches.
        params = tuple(c.Param(name=_safe(inp.name)) for inp in prog.inputs)
        params += tuple(c.Param(name=_safe(a)) for a in prog.int_atoms)
        input_exprs = {
            inp.name: c.Var(name=_safe(inp.name)) for inp in prog.inputs
        }
        sub = _Lowerer(
            prog,
            input_exprs,
            {a: c.Var(name=_safe(a)) for a in prog.int_atoms},
            self.opts,
            var_gen=c.UniqueVarGen(),
            defs=self.defs,
            helpers=self.helpers,
            fp_helpers=self.fp_helpers,
        )
        out_exprs = sub.run()
        ret = _ret_value(c, out_exprs)
        body = sub.body_stmts() + (c.ReturnStmt(value=ret),)
        # Emission-time interning: if a structurally identical helper was already emitted
        # (a content-equal sub-Program under a DIFFERENT object id — ``helpers`` cannot
        # see it), point this prog at the existing def instead of appending a duplicate.
        # Sound bottom-up: nested helpers were interned during ``sub.run()`` above, so
        # two identical bodies reference identical callee spellings. The reserved
        # ``name`` is simply left unused (numbering gaps are harmless).
        # A (mutually) RECURSIVE body — one that calls back into a def still being
        # emitted, via the name reserved above — can never intern by accident: that
        # call site is a FREE ``Var`` in the fingerprint, keeping its unique ``_sub_N``
        # spelling, so it only matches a twin carrying the SAME reserved name, which no
        # other def has. Recursive helpers therefore always get their own def.
        fp = _helper_fingerprint(c, params, body)
        if fp is not None:
            twin = self.fp_helpers.get(fp)
            if twin is not None:
                self.helpers[key] = twin
                return twin
            self.fp_helpers[fp] = name
        self.defs.append(c.FunctionDefStmt(name=name, params=params, body=body))
        return name

    def _call_subprogram(self, prog: Program, args: list[PyExprs],
                         out: tuple[SymArray, ...]) -> tuple[str, PyExprs]:
        name = self._helper_for_program(prog)
        # Append the caller's expression for each of `prog`'s int_atoms as trailing call args (matching the
        # trailing params `_helper_for_program` adds). The caller must carry the scrutinee (a merged program
        # unions its sub-programs' int_atoms), else this KeyErrors loudly — a real mismatch, not silent.
        call_args = list(args) + [self.intatom_exprs[a] for a in prog.int_atoms]
        call = self.core.CallExpr(fn=self.core.Var(name=name), args=tuple(call_args))
        return ("tuple", call) if len(out) > 1 else ("single", call)

    def _call_vmap(self, fn: StmtOp, body: Program, args: list[PyExprs],
                   out: tuple[SymArray, ...]) -> tuple[str, PyExprs]:
        c = self.core
        if getattr(fn, "_nested_n_vars", None) is not None:
            return self._call_nested_vmap(fn, body, args)
        in_axes = getattr(fn, "_in_axes", None)
        out_axes = getattr(fn, "_out_axes", None)
        if in_axes is None or out_axes is None:
            raise NotImplementedError(
                "pyab: vmap closure does not expose _in_axes/_out_axes; rebuild "
                "it via polyarray.ir.vmap."
            )
        out_axes = tuple(out_axes)
        if len(set(out_axes)) > 1:
            raise NotImplementedError(
                "pyab: per-output differing vmap out_axes are not supported; "
                f"got {out_axes}."
            )
        name = self._helper_for_program(body)
        hop = c.VMapHop(
            fn=c.Var(name=name),
            in_dims=tuple(in_axes),
            out_dim=int(out_axes[0]),
        )
        call = c.CallExpr(fn=hop, args=tuple(args))
        return ("tuple", call) if len(out) > 1 else ("single", call)

    def _call_nested_vmap(self, fn: StmtOp, body: Program, args: list[PyExprs]) -> tuple[str, PyExprs]:
        """Lower a multi-variable nested vmap to nested ``VMapHop``s.

        The body's first ``n_vars`` args are batched, one vmap level each over axis 0; the
        remaining ``n_free`` args broadcast. torch stacks the batched axes in front, so,
        matching numpy_source's
        ``_emit_nested_vmap_helper`` — we then ``moveaxis`` the leading ``n_vars``
        var axes to the end (``(*cod, *var_sizes)``).
        """
        c = self.core
        assert isinstance(fn, NestedVmapClosure)
        n_vars = int(fn._nested_n_vars)
        n_free = int(fn._nested_n_free)
        n_args = n_vars + n_free
        name = self._helper_for_program(body)
        node: Any = c.Var(name=name)
        # Innermost level batches the last var; outermost (built last) batches
        # var 0, so var 0 becomes the leading output axis.
        for v in range(n_vars - 1, -1, -1):
            in_dims = tuple(0 if i == v else None for i in range(n_args))
            node = c.VMapHop(fn=node, in_dims=in_dims, out_dim=0)
        call = c.CallExpr(fn=node, args=tuple(args))
        moved = self._ns_call(
            "moveaxis",
            (
                call,
                _pylit(c, tuple(range(n_vars))),
                _pylit(c, tuple(range(-n_vars, 0))),
            ),
        )
        return ("single", moved)

    # -- cells / rational functions ---------------------------------------

    def _const_expr(self, value: npt.ArrayLike) -> PyExprs:
        c = self.core
        if isinstance(value, np.ndarray):
            return _const_array_expr(c, value)
        return c.FloatLit(value=float(value))

    def _cells_expr_shared(self, cells: np.ndarray) -> PyExprs:
        """Assemble an operand ref's cell array once and reuse it, as common-subexpression elimination.

        A non-bulk object-dtype cell array is materialised into a ``Var`` keyed on cell
        identity, so a SymArray consumed as an operand by many statements is built a single
        time rather than re-scattered element-wise (nested ``stack`` of per-cell ``_poly_to_ir``
        trees) at every consumption site — the dominant non-bulk codegen blow-up.

        Float arrays go through the same Var now.  They used to fall through ("not worth
        a Var") because each was a unique object and a cheap single expression; neither
        holds any more.  ``ConstArrayExpr`` makes the operand one node carrying the whole
        table, and pyab's CSE only dedupes an expression that is an assignment RHS — an
        inline operand is re-emitted in full at every use site.  Binding the Var turns
        those uses into references; pyab's content-addressed CSE then collapses the
        bindings themselves, so an id-keyed memo here still lands on distinct-by-content.
        Empty arrays keep falling through (nothing to share).
        """
        arr = np.asarray(cells)
        if arr.size == 0:
            return self._cells_expr(cells)
        key = id(cells)
        hit = self._cells_cse.get(key)
        if hit is not None:
            return hit
        var = self.b.assign_new(self._cells_expr(cells), base="op")
        self._cells_cse[key] = var
        return var

    def _cells_expr(self, cells: np.ndarray) -> PyExprs:
        """Assemble an ndarray of cells (floats and/or RationalFunctions)."""
        c = self.core
        cells = np.asarray(cells)
        if cells.size == 0:
            # A zero-sized axis (e.g. the empty ker/coker block of a full-rank
            # GSVD) has no cells to stack — emit a typed empty tensor.
            return c.ZerosExpr(
                shape=tuple(c.IntLit(value=int(s)) for s in cells.shape),
                dtype=self._dtype_f64(),
            )
        if cells.dtype.kind == "f":
            return _const_array_expr(c, cells)
        # Object dtype: RationalFunction and/or python floats. Assemble scalar
        # tensor leaves with ``stack`` so the result is a real backend tensor.
        return self._stack_cells(cells)

    def _leaf_access(self, cell: Cell) -> tuple[PyExprs, str, tuple[int, ...]] | None:
        """Return the tensor base and integer indices of a cell that is a pure subscript.

        If ``cell`` renders to a pure subscript of a tensor ``Var`` — ``X[i0, i1, …]`` with integer
        indices, or a bare ``X`` — return ``(base_var_expr, base_name, index_ints)``; else ``None``.

        Used by :meth:`_try_view_stack` to recognise a scalar ``stack`` that is really a VIEW of an
        existing tensor. A bare-atom :class:`RationalFunction` renders (via :meth:`_rf_expr`) to exactly
        its generator's expression, which for an input/Stmt-output atom is ``genmap[atom] = X[idx]``; a
        polynomial renders to arithmetic and returns ``None``.
        """
        c = self.core
        if not isinstance(cell, RationalFunction):
            return None
        e = self._rf_expr(cell)
        if isinstance(e, c.Var):
            return (e, e.name, ())
        if (isinstance(e, c.AccessExpr) and isinstance(e.base, c.Var)
                and all(isinstance(i, c.IntLit) for i in e.index)):
            return (e.base, e.base.name, tuple(int(i.value) for i in e.index))
        return None

    def _try_view_stack(self, cells: np.ndarray) -> PyExprs | None:
        """Emit a VIEW in place of a nested ``stack`` of SCALAR leaves.

        Peephole: emit a VIEW (a slice, or a ``stack`` of whole tensors) instead of a nested ``stack`` of
        SCALAR leaves when ``cells`` is exactly a natural-order arrangement of existing tensor ``Var``(s).

        The scalar-stack form otherwise realises one buffer PER ELEMENT (under ``vmap`` → one copy loop per
        cell), which dominates the generated code (e.g. the geometry Jacobian is rebuilt element-by-element).
        A view is a single op Inductor folds away. Provably semantics-preserving — every element emitted IS
        the same tensor element. Returns the view expr, or ``None`` to fall back to the scalar stack.
        Two shapes are recognised (else bail):
          (1) every cell is the IDENTITY access of ONE base ``X`` (``cell[idx] == X[idx]``) → ``X[:s0, :s1, …]``.
          (2) 2-D where each COLUMN ``j`` is a whole 1-D base ``X_j`` at indices ``0..m-1`` (symmetrically each
              ROW) → ``stack([X_0[:m], …], axis=1)`` — the vertex/Jacobian transpose that begins every kernel.
        """
        c = self.core
        cells = np.asarray(cells, dtype=object)
        if cells.ndim == 0 or cells.size == 0:
            return None
        idxs = list(np.ndindex(*cells.shape))
        acc: dict[tuple[int, ...], tuple[PyExprs, str, tuple[int, ...]]] = {}
        for i in idxs:
            a = self._leaf_access(cells[i])
            if a is None:
                return None                    # a non-atom leaf ⇒ not a pure view; bail
            acc[i] = a
        # (1) identity view of a single base.
        if len({acc[i][1] for i in idxs}) == 1 and all(acc[i][2] == i for i in idxs):
            base = acc[idxs[0]][0]
            return c.AccessExpr(base=base,
                                index=tuple(c.Slice(stop=c.IntLit(value=int(s))) for s in cells.shape))
        # (2) 2-D column-/row-homogeneous stack of whole 1-D bases.
        if cells.ndim == 2:
            m, n = int(cells.shape[0]), int(cells.shape[1])
            if all(acc[(i, j)][1] == acc[(0, j)][1] and acc[(i, j)][2] == (i,)
                   for j in range(n) for i in range(m)):
                cols = tuple(c.AccessExpr(base=acc[(0, j)][0], index=(c.Slice(stop=c.IntLit(value=m)),))
                             for j in range(n))
                return c.StackExpr(arrays=cols, axis=1)
            if all(acc[(i, j)][1] == acc[(i, 0)][1] and acc[(i, j)][2] == (j,)
                   for i in range(m) for j in range(n)):
                rows = tuple(c.AccessExpr(base=acc[(i, 0)][0], index=(c.Slice(stop=c.IntLit(value=n)),))
                             for i in range(m))
                return c.StackExpr(arrays=rows, axis=0)
        return None

    def _stack_cells(self, cells: np.ndarray) -> PyExprs:
        c = self.core
        cells = np.asarray(cells, dtype=object)
        if cells.ndim == 0:
            return self._leaf_tensor(cells[()])
        view = self._try_view_stack(cells)          # peephole: a scalar-stack that is really a tensor view
        if view is not None:
            return view
        if cells.ndim == 1:
            leaves = [self._leaf_tensor(cells[i]) for i in range(cells.shape[0])]
            return c.StackExpr(arrays=tuple(leaves), axis=0)
        rows = [self._stack_cells(cells[i]) for i in range(cells.shape[0])]
        return c.StackExpr(arrays=tuple(rows), axis=0)

    def _leaf_tensor(self, cell: Cell) -> PyExprs:
        """Render a single scalar cell as a 0-d backend tensor expression."""
        c = self.core
        if isinstance(cell, RationalFunction):
            return c.ArrayExpr(obj=self._rf_expr(cell), dtype=self._dtype_f64())
        return c.ArrayExpr(obj=c.FloatLit(value=float(cell)), dtype=self._dtype_f64())

    def _rf_expr(self, rf: RationalFunction) -> PyExprs:
        c = self.core
        names = _ring_names(rf._ring)
        gens: list[Any] = []
        for nm in names:
            if nm not in self.genmap:
                raise KeyError(
                    f"pyab: generator {nm!r} has no binding (produced before "
                    "it is consumed?)"
                )
            gens.append(self.genmap[nm])
        num = _poly_to_ir(c, rf.num, gens)
        if rf.den == rf._ring.one:
            return num
        den = _poly_to_ir(c, rf.den, gens)
        return c.BinaryExpr(op=c.BinaryOp.TRUEDIV, lhs=num, rhs=den)

    # -- outputs ----------------------------------------------------------

    def _output_exprs(self) -> list[PyExprs]:
        """Render one expression per program output."""
        exprs: list[PyExprs] = []
        for _, sa in self.prog.outputs.items():
            if sa._bulk is not None:
                exprs.append(self.bulkmap[sa._bulk.name])
            else:
                exprs.append(self._cells_expr(sa.cells))
        return exprs


# ---------------------------------------------------------------------------
# Small module-level helpers
# ---------------------------------------------------------------------------

def _vmap_body_of(fn: StmtOp) -> Program | None:
    body = getattr(fn, "_vmap_body", None)
    if isinstance(body, Program):
        return body
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, Program):
            return v
    return None


def _safe(name: str) -> str:
    import keyword

    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    sane = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not sane or (not sane[0].isalpha() and sane[0] != "_"):
        sane = "_" + sane
    return "p_" + sane if keyword.iskeyword(sane) else sane


def _pylit(core: ModuleType, value: object) -> PyExprs:
    """Render a python int / tuple-of-ints axis spec as an IR literal."""
    if isinstance(value, (tuple, list)):
        return core.TupleExpr(elts=tuple(core.IntLit(value=int(v)) for v in value))
    return core.IntLit(value=int(value))


def _const_array_expr(core: ModuleType, value: npt.ArrayLike) -> PyExprs:
    """Render a dense float constant as one ``ConstArrayExpr``.

    The element-wise spelling (``ArrayExpr`` over a nested ``TupleExpr`` of
    ``FloatLit``) costs one IR node per entry, which large constant tables make the
    dominant term in everything downstream: at high degree in 3D it was 96% of the nodes
    handed to the pyab passes and 84% of the emitted source text. ``ConstArray``
    carries the raw buffer instead — exact, content-addressed, one node.
    """
    return core.ConstArrayExpr(
        value=core.ConstArray.from_numpy(np.asarray(value, dtype=float))
    )


def _nested_tuple(core: ModuleType, obj: object) -> PyExprs:
    """Render a possibly nested Python list of floats as a nested ``TupleExpr`` of ``FloatLit``."""
    if isinstance(obj, (list, tuple)):
        return core.TupleExpr(elts=tuple(_nested_tuple(core, o) for o in obj))
    return core.FloatLit(value=float(obj))


def _ret_value(core: ModuleType, out_exprs: list[PyExprs]) -> PyExprs:
    if len(out_exprs) == 1:
        return out_exprs[0]
    return core.TupleExpr(elts=tuple(out_exprs))


def _static_matrix_shape(out: tuple[SymArray, ...]) -> tuple[int | None, int | None]:
    """(m, n) of the QR operand, recovered from the produced Q output shape.

    ``QrOp`` reduced: Q is ``m x n``.  We read the output SymArray cell shapes:
    Q (out[0]) has shape ``(m, n)``; R (out[1]) has shape ``(n, n)``.
    """
    try:
        q_shape = out[0].cells.shape
        r_shape = out[1].cells.shape
    except Exception:
        return (None, None)
    if len(q_shape) != 2 or len(r_shape) != 2:
        return (None, None)
    return (int(q_shape[0]), int(q_shape[1]))


def _einsum_unroll_max() -> int:
    """Give the size cap under which a small einsum is unrolled.

    Max product of contracted-dim sizes for which a small einsum is unrolled into pointwise ops
    (env `POLYARRAY_EINSUM_UNROLL_MAX`, default 64; `0` disables the unroll, keeping `torch.einsum`).
    """
    v = os.environ.get("POLYARRAY_EINSUM_UNROLL_MAX")
    if v is not None:
        try:
            return int(v)
        except ValueError:
            pass
    return 64


def _ref_shape(ref: object) -> tuple[int, ...] | None:
    """Give the static shape of an einsum operand :class:`Ref`.

    A bulk or cell-array :class:`SymArrayRef`; ``None`` when not statically known (so the einsum unroll
    bails to a plain ``torch.einsum``).
    """
    b = getattr(ref, "_bulk", None)
    if b is not None:
        shp = getattr(b, "shape", None)
        return None if shp is None else tuple(int(s) for s in shp)
    cells = getattr(ref, "cells", None)
    if cells is None:
        return None
    try:
        return tuple(int(s) for s in np.asarray(cells).shape)
    except Exception:
        return None


def _resolve_einsum_ellipsis(
    in_subs: list[str], out_sub: str, shapes: Sequence[tuple[int, ...] | None]
) -> tuple[list[list[str]], list[str]] | None:
    """Replace ``...`` in an einsum spec with concrete fresh labels.

    Fresh labels (``~0``, ``~1``, …) via operand ranks, right-aligned (numpy broadcasting). Returns
    ``(in_subs_resolved, out_sub_resolved)`` or ``None`` if unsupported (a repeated index within one
    operand, or a rank/shape mismatch).
    """
    ell_ranks: list[int] = []
    for sub, shp in zip(in_subs, shapes):
        explicit = sub.replace("...", "")
        if len(set(explicit)) != len(explicit):
            return None                                  # diagonal (repeated index in one operand)
        if "..." in sub:
            if shp is None:
                return None
            r = len(shp) - len(explicit)
            if r < 0:
                return None
            ell_ranks.append(r)
        else:
            if shp is not None and len(explicit) != len(shp):
                return None
            ell_ranks.append(0)
    maxr = max(ell_ranks) if ell_ranks else 0
    labels = [f"~{i}" for i in range(maxr)]
    def resolve(sub: str, r: int) -> list[str]:
        if "..." not in sub:
            return list(sub)
        left, _, right = sub.partition("...")
        return list(left) + labels[maxr - r:] + list(right)
    in2 = [resolve(sub, r) for sub, r in zip(in_subs, ell_ranks)]
    if "..." in out_sub:
        left, _, right = out_sub.partition("...")
        out2 = list(left) + labels + list(right)
    else:
        out2 = list(out_sub)
    return in2, out2


def _const_grid(core: ModuleType, arr: np.ndarray) -> np.ndarray:
    """Build a grid of ``FloatLit``s for a numeric constant array (an einsum RHS / a ``Const`` operand)."""
    g = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(*arr.shape) if arr.shape else [()]:
        g[idx] = core.FloatLit(value=float(arr[idx] if arr.shape else arr[()]))
    return g


def _einsum_index_sizes(
    in_subs: list[str], out_sub: str, shapes: Sequence[tuple[int, ...] | None],
    out_shape: tuple[int, ...] | None,
) -> tuple[list[list[str]], list[str], dict[str, int], list[str]] | None:
    """Resolve ``...``, then infer every index's extent from the operand and output shapes.

    Reads the operand shapes AND the output shape. Returns ``(in2, out2, size, contracted)`` or ``None``
    when unsupported / not statically inferable.
    """
    res = _resolve_einsum_ellipsis(in_subs, out_sub, list(shapes))
    if res is None:
        return None
    in2, out2 = res
    size: dict[str, int] = {}
    def add(sub: list[str], shp: tuple[int, ...] | None) -> bool:
        if shp is None:
            return True
        if len(sub) != len(shp):
            return False
        for ch, s in zip(sub, shp):
            if ch in size and size[ch] != int(s):
                return False
            size[ch] = int(s)
        return True
    for sub, shp in zip(in2, shapes):
        if not add(sub, shp):
            return None
    if not add(out2, out_shape):
        return None
    if any(ch not in size for ch in out2):
        return None
    contracted = [ch for ch in size if ch not in out2]
    return in2, out2, size, contracted


def _scalar_einsum(
    low: _Lowerer, spec: str, grids: Sequence[np.ndarray], out_shape: tuple[int, ...] | None,
) -> _ScalarGrid | None:
    """Lower a small einsum to a GRID of scalar exprs.

    Each output element an explicit sum-of-products over the operands' scalar grids. Fully scalar (no free
    tensor axes), so under ``torch.vmap`` every element is a ``(n_cells,)`` value and the whole per-cell
    body stays in one fused loop. ``None`` to fall back.
    """
    import itertools
    c = low.core
    if "->" not in spec:
        return None
    inp, _, outp = spec.partition("->")
    in_subs = [s.strip() for s in inp.split(",")]
    if len(in_subs) != len(grids):
        return None
    shapes = [tuple(int(s) for s in g.shape) for g in grids]
    info = _einsum_index_sizes(in_subs, outp.strip(), shapes, out_shape)
    if info is None:
        return None
    in2, out2, size, contracted = info
    out_dims = [size[ch] for ch in out2]
    grid = np.empty(tuple(out_dims), dtype=object)
    for oidx in (np.ndindex(*out_dims) if out_dims else [()]):
        label = {ch: oidx[k] for k, ch in enumerate(out2)}
        terms = []
        for cidx in itertools.product(*[range(size[ch]) for ch in contracted]):
            cval = {ch: cidx[k] for k, ch in enumerate(contracted)}
            full = {**label, **cval}
            factors = [g[tuple(full[ch] for ch in sub)] if sub else g[()]
                       for g, sub in zip(grids, in2)]
            terms.append(_balanced(c, factors, c.BinaryOp.MUL) if factors else c.FloatLit(value=1.0))
        grid[oidx] = _balanced(c, terms, c.BinaryOp.ADD) if terms else c.FloatLit(value=0.0)
    return _ScalarGrid(grid)


def _fold_weighted_einsum(
    low: _Lowerer, spec: str, in_refs: Sequence[Ref], out_shape: tuple[int, ...] | None,
) -> PyExprs | None:
    """Lower a contraction of a per-cell WEIGHT against a table.

    The WEIGHT is an operand indexed only by contracted axes, the table an operand carrying the free output
    axes; the fold is ``Σ_contracted (Π weight scalars) · table[contracted]`` — the weight read
    component-wise as SCALARS (never stacked into an ``(n_cells, K)`` tensor) and the table sliced at each
    contracted index so the free axes stay ONE tensor.

    This is the fold ``'qab,q->ab'`` at the end of every element-matrix assembly. Stacking the weight into an
    ``(n_cells, K)`` buffer forces Inductor to realise it — a fusion boundary that splits the per-cell body
    into a second loop over cells (measured). Reading it as scalars keeps the whole body in ONE loop. Returns
    the tensor expr, or ``None`` (no all-contracted operand, or the single-table shape assumption fails) to
    fall back to :func:`_unroll_einsum`. Handles exactly ONE table operand (the common assembly shape).
    """
    import itertools
    c = low.core
    if _einsum_unroll_max() <= 0 or "->" not in spec:
        return None
    inp, _, outp = spec.partition("->")
    in_subs = [s.strip() for s in inp.split(",")]
    if len(in_subs) != len(in_refs):
        return None
    shapes = [_ref_shape(r) for r in in_refs]
    info = _einsum_index_sizes(in_subs, outp.strip(), shapes, out_shape)
    if info is None:
        return None
    in2, out2, size, contracted = info
    if not contracted:
        return None
    cset = set(contracted)
    scalar_ops = [(k, sub) for k, sub in enumerate(in2) if all(ch in cset for ch in sub)]
    tensor_ops = [(k, sub) for k, sub in enumerate(in2) if any(ch not in cset for ch in sub)]
    if not scalar_ops or len(tensor_ops) != 1:
        return None                                          # no weight, or not the single-table shape
    csz = 1
    for ch in contracted:
        csz *= size[ch]
    if csz == 0 or csz > _einsum_unroll_max():
        return None
    tk, tsub = tensor_ops[0]
    tfree = [ch for ch in tsub if ch not in cset]
    if sorted(tfree) != sorted(out2):
        return None                                          # the table must carry exactly the free axes
    texpr = low._ref_expr(in_refs[tk])
    sgrids = {k: low._operand_grid(in_refs[k]) for k, _ in scalar_ops}
    terms: list[PyExprs] = []
    for combo in itertools.product(*[range(size[ch]) for ch in contracted]):
        cval = dict(zip(contracted, combo))
        coeff_factors = []
        for k, sub in scalar_ops:
            idx = tuple(cval[ch] for ch in sub)
            g = sgrids[k]
            coeff_factors.append(g[idx] if idx else g[()])
        coeff = _balanced(c, coeff_factors, c.BinaryOp.MUL)
        index = tuple(c.IntLit(value=int(cval[ch])) if ch in cset else c.Slice() for ch in tsub)
        slab = c.AccessExpr(base=texpr, index=index)         # free-shaped tensor slab at this contracted index
        if tfree != list(out2):                              # reorder the slab's free axes to the output order
            axes = tuple(tfree.index(ch) for ch in out2)
            slab = c.TransposeExpr(a=slab, axes=axes)
        terms.append(c.BinaryExpr(op=c.BinaryOp.MUL, lhs=slab, rhs=coeff))
    return _balanced(c, terms, c.BinaryOp.ADD)


def _unroll_einsum(
    low: _Lowerer, spec: str, exprs: Sequence[PyExprs],
    shapes: Sequence[tuple[int, ...] | None], out_shape: tuple[int, ...] | None,
) -> PyExprs | None:
    """Lower a SMALL einsum to a broadcast-multiply + an UNROLLED contraction sum.

    Pure pointwise ops that Inductor fuses, instead of a ``torch.einsum`` (which it turns into an extern
    ``bmm`` and/or a ``Reduction``, both hard fusion boundaries). Returns the expr, or ``None`` to fall
    back to ``EinsumExpr``.

    Each operand is transposed + ``newaxis``-padded to a common ``[output dims, contracted dims]`` layout,
    the operands are multiplied elementwise, and the trailing contracted axes are summed by EXPLICIT adds
    over their (small, static) index ranges — no ``.sum``/reduction, no matmul. Gated on a statically-known
    small contraction (``_einsum_unroll_max``).
    """
    c = low.core
    umax = _einsum_unroll_max()
    if umax <= 0 or "->" not in spec:
        return None
    inp, _, outp = spec.partition("->")
    in_subs = [s.strip() for s in inp.split(",")]
    if len(in_subs) != len(exprs):
        return None
    res = _resolve_einsum_ellipsis(in_subs, outp.strip(), list(shapes))
    if res is None:
        return None
    in2, out2 = res
    # infer every index size from the operand shapes AND the output shape (either can supply a size).
    size: dict[str, int] = {}
    def add(sub: list[str], shp: tuple[int, ...] | None) -> bool:
        if shp is None:
            return True
        if len(sub) != len(shp):
            return False
        for ch, s in zip(sub, shp):
            if ch in size and size[ch] != int(s):
                return False
            size[ch] = int(s)
        return True
    for sub, shp in zip(in2, shapes):
        if not add(sub, shp):
            return None
    if not add(out2, out_shape):
        return None
    if any(ch not in size for ch in out2):
        return None
    contracted = [ch for ch in size if ch not in out2]
    if any(ch not in size for ch in contracted):
        return None
    csz = 1
    for ch in contracted:
        csz *= size[ch]
    if csz == 0 or csz > umax:
        return None
    union = list(out2) + contracted                      # output dims first, contracted trailing
    aligned: list[PyExprs] = []
    for expr, sub in zip(exprs, in2):
        present = [ch for ch in union if ch in sub]
        if present != list(sub):
            axes = tuple(sub.index(ch) for ch in present)
            expr = c.TransposeExpr(a=expr, axes=axes)
        index = tuple(c.Slice() if ch in sub else c.NewAxisTok() for ch in union)
        aligned.append(c.AccessExpr(base=expr, index=index))
    prod = aligned[0]
    for e in aligned[1:]:
        prod = c.BinaryExpr(op=c.BinaryOp.MUL, lhs=prod, rhs=e)
    if not contracted:
        return prod
    prod_var = low.b.assign_new(prod, base="einsum")     # bind so unrolled slices reference, not duplicate
    n_out = len(out2)
    import itertools
    terms = [
        c.AccessExpr(base=prod_var,
                     index=tuple(c.Slice() for _ in range(n_out))
                           + tuple(c.IntLit(value=int(i)) for i in combo))
        for combo in itertools.product(*[range(size[ch]) for ch in contracted])
    ]
    return _balanced(c, terms, c.BinaryOp.ADD)


def _poly_to_ir(core: ModuleType, poly: Poly, gens: Sequence[PyExprs]) -> PyExprs:
    """Render a backend ``Poly`` as an IR arithmetic expr over ``gens`` exprs.

    Mirrors :func:`polyarray.rational._poly_term_strings`: each term
    ``coeff * prod_j gen_j ** e_j`` becomes a product tree; terms are summed.
    Both the term-product and the term-sum are built as **balanced** binary
    trees so deep polynomials do not blow codegen recursion.
    """
    terms: list[Any] = []
    for monom, coeff in poly.terms():
        cval = _coeff_to_float(coeff)
        if cval == 0.0:
            continue
        factors: list[Any] = []
        for pos, exp in enumerate(monom):
            if exp == 0:
                continue
            g = gens[pos]
            if exp == 1:
                factors.append(g)
            else:
                factors.append(
                    core.BinaryExpr(op=core.BinaryOp.POW, lhs=g, rhs=core.IntLit(value=int(exp)))
                )
        if not factors:
            terms.append(core.FloatLit(value=cval))
            continue
        prod = _balanced(core, factors, core.BinaryOp.MUL)
        if cval == 1.0:
            terms.append(prod)
        elif cval == -1.0:
            terms.append(core.UnaryExpr(op=core.UnaryOp.NEG, x=prod))
        else:
            terms.append(
                core.BinaryExpr(op=core.BinaryOp.MUL, lhs=core.FloatLit(value=cval), rhs=prod)
            )
    if not terms:
        return core.FloatLit(value=0.0)
    return _balanced(core, terms, core.BinaryOp.ADD)


def _balanced(core: ModuleType, exprs: list[PyExprs], op: PyExprs) -> PyExprs:
    """Fold ``exprs`` with binary ``op`` into a balanced tree (log depth)."""
    if len(exprs) == 1:
        return exprs[0]
    while len(exprs) > 1:
        nxt: list[Any] = []
        for i in range(0, len(exprs) - 1, 2):
            nxt.append(core.BinaryExpr(op=op, lhs=exprs[i], rhs=exprs[i + 1]))
        if len(exprs) % 2 == 1:
            nxt.append(exprs[-1])
        exprs = nxt
    return exprs[0]


# ---------------------------------------------------------------------------
# Small-matrix Householder QR — unrolled scalar code, LAPACK sign convention.
# ---------------------------------------------------------------------------

def _householder_qr_core(
    low: _Lowerer, access: Callable[[int, int], PyExprs], m: int, n: int
) -> tuple[list[list[PyExprs]], list[list[PyExprs]]]:
    """Emit an unrolled Householder QR of an ``m x n`` (m>=n) matrix.

    Returns the FULL ``Q`` (``m x m``) and ``R`` (``m x n``) as nested lists of scalar exprs (NOT
    stacked). ``access(i, j)`` supplies each input entry (a tensor subscript or a scalar grid element).
    Reproduces LAPACK/numpy's convention ``R[k,k] = -sign(x0)*||x||`` (``sign(0):=+1``). Every
    intermediate is bound to a fresh scalar Var. Shared by the tensor emitter
    (:func:`_emit_householder_qr`) and the scalar-grid one (:func:`_emit_householder_qr_grid`), which
    apply the reduced/complete shaping.
    """
    c = low.core
    b = low.b
    one = c.FloatLit(value=1.0)
    zero = c.FloatLit(value=0.0)

    def sca(base: str, e: PyExprs) -> PyExprs:
        return b.assign_new(e, base=base)

    def sub(x: PyExprs, y: PyExprs) -> PyExprs:
        return c.BinaryExpr(op=c.BinaryOp.SUB, lhs=x, rhs=y)

    def mul(x: PyExprs, y: PyExprs) -> PyExprs:
        return c.BinaryExpr(op=c.BinaryOp.MUL, lhs=x, rhs=y)

    def div(x: PyExprs, y: PyExprs) -> PyExprs:
        return c.BinaryExpr(op=c.BinaryOp.TRUEDIV, lhs=x, rhs=y)

    # Materialise the input matrix entries as scalar tensor Vars.
    R: list[list[Any]] = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(sca("a", access(i, j)))
        R.append(row)
    # Q accumulates the reflections (start as the m x m identity).
    Q: list[list[Any]] = [[one if i == j else zero for j in range(m)] for i in range(m)]

    for k in range(n):
        # When the strictly-below-diagonal tail is empty (k == m-1: a square
        # matrix's last column), LAPACK applies no reflection (tau=0) and R[k][k]
        # keeps its sign.  Skip statically — this both matches LAPACK and avoids a
        # constant-fold ``where(False, ...)`` (torch requires a tensor condition).
        if k + 1 >= m:
            continue
        # Householder vector v for column k, rows k..m-1.
        # normsq = sum_{i>=k} R[i][k]^2 (includes the diagonal entry x0).
        sq_terms = [mul(R[i][k], R[i][k]) for i in range(k, m)]
        normsq = sca("nsq", _balanced(c, sq_terms, c.BinaryOp.ADD))
        norm = sca("nrm", c.BinaryExpr(op=c.BinaryOp.POW, lhs=normsq, rhs=c.FloatLit(value=0.5)))
        x0 = R[k][k]
        # LAPACK ``dlarfg``: reflect only when the strictly-below-diagonal tail is
        # nonzero.  Otherwise tau=0 (H=I) and R[k][k] keeps its sign — this is why
        # the trailing column of a square QR is NOT reflected.  ``tail_sq`` is the
        # sum over rows k+1..m-1 (statically 0.0 when the tail is empty).
        tail_terms = [mul(R[i][k], R[i][k]) for i in range(k + 1, m)]
        tail_sq = sca("tsq", _balanced(c, tail_terms, c.BinaryOp.ADD)) if tail_terms else zero
        active = c.BinaryExpr(op=c.BinaryOp.GT, lhs=tail_sq, rhs=zero)
        # s = sign(x0) with sign(0):=+1  ->  where(x0 >= 0, 1, -1)
        s = sca(
            "s",
            c.WhereExpr(
                cond=c.BinaryExpr(op=c.BinaryOp.GE, lhs=x0, rhs=zero),
                x=one,
                y=c.FloatLit(value=-1.0),
            ),
        )
        alpha = sca("al", mul(c.UnaryExpr(op=c.UnaryOp.NEG, x=s), norm))  # -sign(x0)*||x||
        # v = x - alpha*e_k  (v[k] = x0 - alpha; v[i>k] = R[i][k])
        v: dict[int, Any] = {}
        v[k] = sca("v", sub(x0, alpha))
        for i in range(k + 1, m):
            v[i] = R[i][k]
        # beta = 2 / (v·v), but 0 when the reflection is inactive (H = I).
        vtv_terms = [mul(v[i], v[i]) for i in range(k, m)]
        vtv = sca("vtv", _balanced(c, vtv_terms, c.BinaryOp.ADD))
        vtv_safe = c.WhereExpr(
            cond=c.BinaryExpr(op=c.BinaryOp.GT, lhs=vtv, rhs=zero), x=vtv, y=one
        )
        beta = sca(
            "bt",
            c.WhereExpr(
                cond=active, x=div(c.FloatLit(value=2.0), vtv_safe), y=zero
            ),
        )
        # Apply H = I - beta v v^T to R[:, k:] and to Q[:, :] (columns k..m-1
        # of Q are affected; we apply to all for simplicity).
        for j in range(k, n):
            dot_terms = [mul(v[i], R[i][j]) for i in range(k, m)]
            w = sca("w", mul(beta, _balanced(c, dot_terms, c.BinaryOp.ADD)))
            for i in range(k, m):
                R[i][j] = sca("R", sub(R[i][j], mul(w, v[i])))
        for j in range(m):
            dot_terms = [mul(v[i], Q[i][j]) for i in range(k, m)]
            w = sca("qw", mul(beta, _balanced(c, dot_terms, c.BinaryOp.ADD)))
            for i in range(k, m):
                Q[i][j] = sca("Q", sub(Q[i][j], mul(w, v[i])))

    # Q currently holds H_{n-1}...H_0 (an m x m orthogonal matrix, applied to I).
    # The QR factor Q_full = (H_{n-1}...H_0)^T is its transpose; R keeps the upper
    # triangle (below-diagonal entries are zeroed by the shaping wrappers).
    return Q, R


def _qr_scalar_grids(
    Q: list[list[PyExprs]], R: list[list[PyExprs]], m: int, n: int, mode: str, zero: PyExprs
) -> tuple[np.ndarray, np.ndarray]:
    """Shape the full ``Q``/``R`` scalar lists to the reduced/complete factor grids.

    ``Q^T`` sliced to ``ncols_q`` columns, ``R`` sliced to ``nrows_r`` rows with the strict lower triangle
    set to ``zero``.
    """
    ncols_q = m if mode == "complete" else n
    q_grid = np.empty((m, ncols_q), dtype=object)
    for i in range(m):
        for j in range(ncols_q):
            q_grid[i, j] = Q[j][i]                       # transpose of the accumulated product
    nrows_r = m if mode == "complete" else n
    r_grid = np.empty((nrows_r, n), dtype=object)
    for i in range(nrows_r):
        for j in range(n):
            r_grid[i, j] = R[i][j] if i <= j else zero
    return q_grid, r_grid


def _emit_householder_qr(
    low: _Lowerer, a_expr: PyExprs, m: int, n: int, mode: str
) -> tuple[PyExprs, PyExprs]:
    """Emit the TENSOR form of the unrolled Householder QR.

    Returns ``(Q_expr, R_expr)`` as stacked backend tensors (reduced ``m x n`` / ``n x n``, or full for
    ``mode="complete"``). Reads the input matrix by subscript.
    """
    c = low.core
    Q, R = _householder_qr_core(
        low, lambda i, j: c.AccessExpr(base=a_expr, index=(c.IntLit(value=i), c.IntLit(value=j))), m, n
    )
    q_grid, r_grid = _qr_scalar_grids(Q, R, m, n, mode, c.FloatLit(value=0.0))
    q_expr = c.StackExpr(
        arrays=tuple(c.StackExpr(arrays=tuple(_as_tensor(low, e) for e in q_grid[i]), axis=0)
                     for i in range(q_grid.shape[0])), axis=0)
    r_expr = c.StackExpr(
        arrays=tuple(c.StackExpr(arrays=tuple(_as_tensor(low, e) for e in r_grid[i]), axis=0)
                     for i in range(r_grid.shape[0])), axis=0)
    return q_expr, r_expr


def _emit_householder_qr_grid(
    low: _Lowerer, access: Callable[[int, int], PyExprs], m: int, n: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Emit the SCALAR-GRID form of the unrolled Householder QR.

    Returns ``(Q_grid, R_grid)`` as ``np.ndarray``s of scalar exprs (no ``stack``). ``access(i, j)``
    supplies each input entry component-wise.
    """
    Q, R = _householder_qr_core(low, access, m, n)
    return _qr_scalar_grids(Q, R, m, n, mode, low.core.FloatLit(value=0.0))


def _as_tensor(low: _Lowerer, e: PyExprs) -> PyExprs:
    c = low.core
    return c.ArrayExpr(obj=e, dtype=low._dtype_f64())


# ---------------------------------------------------------------------------
# Backend prep — eliminate avoidable vmap / GSVD before lowering
# ---------------------------------------------------------------------------
#
# These rewrites are semantics-preserving: the vmap collapse is numerically
# *verified* against the original vmap on a random probe before it is applied,
# and the GSVD identity-metric fold is an algebraic identity (cholesky(I)=I,
# solve(I,·)=·). Both benefit ``Program.run`` and ``to_numpy_source`` too, not
# just the backend.

# vmap bodies made of exactly one of these (leading-batch-broadcasting) ops
# collapse to the op applied directly to the batched operands. QrOp is excluded
# so small-QR Householder interception is preserved; SvdOp/GSvdOp are excluded
# because their rank output does not batch elementwise.
_VMAP_COLLAPSIBLE = (
    DetOp, InvOp, PinvOp, SolveOp, SqrtOp, AbsOp, SignOp, EinsumStmtOp,
)


def _sole_collapsible_op(body: Program) -> StmtOp | None:
    """Find the single collapsible op in a vmap body, or ``None``.

    ``IdentityOp`` statements (the bulk->per-cell unpack / freeze passthroughs)
    and ``fn=None`` splices are ignored — they never change values.  If the body
    holds any *other* real op, or more than one collapsible op, return None (the
    body is not a single batchable op).
    """
    found = None
    for s in body.statements:
        f = s.fn
        if f is None or isinstance(f, IdentityOp):
            continue
        if isinstance(f, _VMAP_COLLAPSIBLE):
            if found is not None:
                return None
            found = f
        else:
            return None
    return found


def _batched_einsum_spec(spec: str) -> str:
    """Prepend a fresh batch index to every operand and the output of ``spec``."""
    lhs, _, rhs = spec.partition("->")
    used = set(spec) - set(",->")
    batch = next((ch for ch in "BNZYXWVUTSRQPOMLKJ" if ch not in used), None)
    if batch is None:
        raise ValueError(f"no free einsum index to batch {spec!r}")
    ins = ",".join(batch + t.strip() for t in lhs.split(","))
    return f"{ins}->{batch}{rhs.strip()}"


def _batched_op(op: StmtOp) -> StmtOp | None:
    if isinstance(op, EinsumStmtOp):
        return EinsumStmtOp(spec=_batched_einsum_spec(op.spec), optimize=op.optimize)
    return op  # det / inv / pinv / solve / sqrt / abs / sign batch over leading dims


def _verify_collapse(fn: StmtOp, body: Program, batched_op: StmtOp, *, m: int, seed: int) -> bool:
    """Numerically check ``vmap(body)(probe) == batched_op(probe)`` on a probe."""
    rng = np.random.default_rng(seed)
    probes: list[np.ndarray] = []
    for inp in body.inputs:
        shp = cast("tuple[int, ...]", tuple(inp.shape))
        a = rng.standard_normal((m, *shp))
        if len(shp) >= 2 and shp[-1] == shp[-2]:
            a = a + shp[-1] * np.eye(shp[-1])  # well-condition inv/solve probes
        probes.append(a)
    try:
        got_v = np.asarray(fn(*probes), dtype=float)
        got_o = np.asarray(batched_op(*[np.asarray(p) for p in probes]), dtype=float)
    except Exception:
        return False
    return got_v.shape == got_o.shape and bool(
        np.allclose(got_v, got_o, rtol=1e-9, atol=1e-11)
    )


def collapse_vmap(program: Program, *, probe_batch: int = 4, seed: int = 12345) -> tuple[Program, list[str]]:
    """Return a copy of ``program`` with verifiably-batchable vmaps collapsed.

    A ``vmap(body)`` Stmt whose body is a single leading-batch op
    (det/inv/pinv/solve/sqrt/abs/sign/einsum; ``in_axes``/``out_axes`` all 0) is
    replaced by that op applied directly to the batched operands — but only after
    the rewrite is *numerically verified* equal to the original vmap on a random
    probe.  ``Program.run`` on the result is unchanged.  Returns
    ``(new_program, collapsed_op_names)``.
    """
    new = program.copy()
    collapsed: list[str] = []
    for stmt in new.statements:
        fn = stmt.fn
        body = _vmap_body_of(fn)
        if body is None or getattr(fn, "_nested_n_vars", None) is not None:
            continue
        in_axes = getattr(fn, "_in_axes", None)
        out_axes = getattr(fn, "_out_axes", None)
        if in_axes is None or out_axes is None:
            continue
        if any(a != 0 for a in in_axes) or any(a != 0 for a in out_axes):
            continue
        op = _sole_collapsible_op(body)
        if op is None:
            continue
        bop = _batched_op(op)
        if not _verify_collapse(fn, body, bop, m=probe_batch, seed=seed):
            continue
        stmt.fn = bop
        collapsed.append(type(op).__name__)
    return new, collapsed


def prepare(program: Program, *, opts: LowerOpts | None = None) -> tuple[Program, PrepReport]:
    """Apply backend-prep simplifications; return ``(program, report)``.

    Run automatically by :func:`as_function_def` / :func:`lower_program_into` /
    the ``compile_*`` conveniences (unless the corresponding ``LowerOpts`` flag
    is disabled).  Exposed standalone so callers can inspect what was eliminated
    (``report["collapsed_vmap"]`` is the list of collapsed op names).  GSVD
    identity-metric folding is applied later, at lowering time (it needs the
    operand exprs), and is reported per-call via ``report`` only when run here is
    not possible — see :meth:`_Lowerer._gsvd`.
    """
    opts = opts or LowerOpts()
    report: dict[str, Any] = {"collapsed_vmap": []}
    if opts.collapse_vmap:
        program, collapsed = collapse_vmap(program)
        report["collapsed_vmap"] = collapsed
    return program, report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lower_program_into(
    program: Program,
    builder: StmtBuilder,
    arg_exprs: Sequence[PyExprs],
    *,
    intatom_exprs: Mapping[str, PyExprs] | None = None,
    opts: LowerOpts | None = None,
) -> tuple[list[PyExprs], list[PyStmts]]:
    """Inline ``program`` into ``builder`` at the current point.

    ``arg_exprs`` are the PyAB exprs bound to the program inputs (in
    declaration order).  Statements are emitted into ``builder``; any
    sub-program / vmap helper ``def`` blocks are returned separately (they must be
    placed at an enclosing scope, before the statements that use them).

    Returns ``(helper_defs, output_exprs)`` — one expr per program output.
    """
    opts = opts or LowerOpts()
    program, _ = prepare(program, opts=opts)
    inputs = list(program.inputs)
    if len(arg_exprs) != len(inputs):
        raise ValueError(
            f"lower_program_into: program {program.name!r} has {len(inputs)} "
            f"inputs but {len(arg_exprs)} arg_exprs were given"
        )
    input_exprs = {inp.name: ex for inp, ex in zip(inputs, arg_exprs)}
    intatom_exprs = dict(intatom_exprs or {})
    low = _Lowerer(
        program, input_exprs, intatom_exprs, opts, var_gen=builder.var_gen
    )
    out_exprs = low.run()
    builder.extend(low.body_stmts())
    return low.defs, out_exprs


# Soft ceilings for the eager-materialization backstop. A HEALTHY (mostly-bulk) program reports ~0
# symbolic output mass (``analyze`` skips bulk cells — ``forward._row``), so these fire ONLY on genuine
# blow-ups (per-cell RF scatter / a live data-independent solve expanded per cell), before the expensive
# render/codegen that would otherwise OOM. Tunable; kept generous so it is signal, not noise. Override
# per run with ``PYAB_IR_MASS_CEILING`` / ``PYAB_IR_CELLS_CEILING``.
_IR_MASS_CEILING = 100_000        # Σ monomials over symbolic output cells (all programs + nested bodies)
_IR_CELLS_CEILING = 8192          # number of symbolic output cells (eager per-cell scatter)


class PyabIROversizedError(RuntimeError):
    """Raised by the oversized-IR backstop under ``PYAB_IR_CEILING=raise``.

    The program's symbolic IR has passed the cost ceiling — the fingerprint of an eager
    materialization, or an un-folded data-independent solve — so compilation aborts here,
    with a top-offenders breakdown, rather than proceeding to a render that would exhaust
    memory or hang. Catchable, so a caller can fall back or re-plan.
    """


def _ir_ceiling_mode() -> str:
    """Read how the oversized-IR backstop should react, from ``PYAB_IR_CEILING``.

    One of ``warn`` (the default), ``raise`` or ``off``. ``raise`` suits CI and development:
    it turns a blow-up into an instant, attributed failure at compile time rather than a slow
    exhaustion downstream.
    """
    return os.environ.get("PYAB_IR_CEILING", "warn").strip().lower()


def _check_ir_oversized(program: Program, name: str) -> None:
    """Guard against the eager-materialization blow-up before rendering.

    When ``program``'s symbolic IR is pathologically large — the fingerprint of a per-cell
    scatter that should be a deferred bulk node, or a data-independent solve or SVD that
    should have been constant-folded — react per :func:`_ir_ceiling_mode`:
    ``warn`` (one warning, the compile proceeds; the default), ``raise``
    (:class:`PyabIROversizedError`, abort now with a top-offenders breakdown), or ``off``. Rides
    :func:`polyarray.forward.analyze` (cheap on bulk programs, which it skips — the asymmetry is the
    signal). The ANALYSIS never breaks a compile; only ``raise`` mode raises, and only deliberately.
    """
    mode = _ir_ceiling_mode()
    if mode == "off":
        return
    try:                                                # the analysis itself must NEVER break a compile
        from . import forward
        rep = forward.analyze(program)
        n_cells = sum(r.n_output_cells for r in rep.rows)
        mass_ceiling = int(os.environ.get("PYAB_IR_MASS_CEILING", _IR_MASS_CEILING))
        cells_ceiling = int(os.environ.get("PYAB_IR_CELLS_CEILING", _IR_CELLS_CEILING))
        oversized = rep.total_mass > mass_ceiling or n_cells > cells_ceiling
    except Exception:
        return
    if not oversized:
        return
    top = sorted(rep.rows, key=lambda r: r.total_mass, reverse=True)[:5]
    breakdown = "\n".join(
        f"    {r.name:16} mass={r.total_mass:>8d} sym_cells={r.symbolic_cells}/{r.n_output_cells} "
        f"max_cell={r.max_cell} prov={r.prov_kinds}" for r in top)
    msg = (
        f"pyab.as_function_def({name!r}): oversized IR (mass={rep.total_mass}, output_cells={n_cells}; "
        f"ceilings mass={mass_ceiling}, cells={cells_ceiling}) — the fingerprint of an EAGER "
        f"materialization that should be a deferred `bulk` node (route contractions through "
        f"`runtime_einsum_multi`, emit large-shape outputs `bulk=True`, keep `.shape`/`.ndim` off "
        f"`.cells`), or a data-independent solve/SVD that should be constant-folded.\n"
        f"  Top sub-programs by symbolic mass:\n{breakdown}")
    if mode == "raise":
        raise PyabIROversizedError(msg)
    import warnings
    warnings.warn(msg, stacklevel=2)                    # default `warn`: surface it, compile proceeds


def as_function_def(
    program: Program,
    *,
    name: str = "f",
    opts: LowerOpts | None = None,
) -> tuple[PyStmts, ...]:
    """Emit ``program`` as PyAB statements: helper ``def``s + one ``def name``.

    The returned tuple is ready to hand to a PyAB backend ``compile``.  The
    entrypoint takes one array parameter per program input (named by the input
    name) plus one per ``IntAtom``, and returns the program output(s) — a bare
    value for one output, else a tuple in declaration order.
    """
    opts = opts or LowerOpts()
    program, _ = prepare(program, opts=opts)
    _check_ir_oversized(program, name)                  # backstop: catch eager-materialization blow-ups (§E2)
    core = _core()
    params = tuple(core.Param(name=_safe(inp.name)) for inp in program.inputs)
    params += tuple(core.Param(name=_safe(a)) for a in program.int_atoms)
    input_exprs = {inp.name: core.Var(name=_safe(inp.name)) for inp in program.inputs}
    intatom_exprs = {a: core.Var(name=_safe(a)) for a in program.int_atoms}
    low = _Lowerer(program, input_exprs, intatom_exprs, opts, var_gen=core.UniqueVarGen())
    out_exprs = low.run()
    ret = _ret_value(core, out_exprs)
    body = low.body_stmts() + (core.ReturnStmt(value=ret),)
    entry = core.FunctionDefStmt(name=name, params=params, body=body)
    return tuple(low.defs) + (entry,)


emit_module_stmts = as_function_def


def call_lowered(
    program: Program,
    builder: StmtBuilder,
    arg_exprs: Sequence[PyExprs],
    *,
    place: Placement = "plain",
    in_dims: Sequence[int | None] | None = None,
    out_dim: int = 0,
    device: Device | None = None,
    fusion_opts: FusionOpts | None = None,
    kernel_kind: KernelKind | None = None,
    name: str | None = None,
    opts: LowerOpts | None = None,
) -> tuple[tuple[PyStmts, ...], PyExprs]:
    """Emit ``program`` as a callable ``def`` + a *placed* call, into ``builder``.

    The ``def`` (plus helper defs) is returned as ``module_defs`` — the caller
    places it at module / enclosing-function scope.  The call itself is emitted
    into ``builder`` and its result expr returned.

    ``place``:

    * ``"plain"`` — ``f(*args)``.
    * ``"vmap"``  — ``vmap(f, in_dims=..., out_dim=...)(*args)`` (a ``VMapHop``);
      ``in_dims`` defaults to batching every arg on axis 0.
    * ``"fuse"``  — the call inside a ``FusionRegionExpr`` (→ ``@torch.compile``).

    Returns ``(module_defs, result_expr)``.
    """
    opts = opts or LowerOpts()
    core = _core()
    fname = name or f"_kernel_{id(program) & 0xFFFF:x}"
    module_defs = as_function_def(program, name=fname, opts=opts)
    fvar = core.Var(name=fname)
    args = tuple(arg_exprs)

    if place == "plain":
        call = core.CallExpr(fn=fvar, args=args)
        result = builder.assign_new(call, base="call")
        return module_defs, result
    if place == "vmap":
        dims = tuple(in_dims) if in_dims is not None else tuple(0 for _ in args)
        hop = core.VMapHop(fn=fvar, in_dims=dims, out_dim=out_dim)
        call = core.CallExpr(fn=hop, args=args)
        result = builder.assign_new(call, base="vcall")
        return module_defs, result
    if place == "fuse":
        from pyarraybackend.ir.types import Device

        dev = device if device is not None else Device.CPU
        kk = kernel_kind if kernel_kind is not None else core.KernelKind.TORCH_COMPILE

        def _body(child: PyStmts) -> PyStmts:
            return core.CallExpr(fn=fvar, args=args)

        result = builder.region(
            _body, device=dev, kernel_kind=kk, opts=fusion_opts, base="fused"
        )
        return module_defs, result
    raise ValueError(f"call_lowered: unknown place {place!r}")


def compile_torch(
    program: Program,
    *,
    name: str = "f",
    opts: LowerOpts | None = None,
    options: TorchBackendOptions | None = None,
    dump_dir: Path | str | None = None,
) -> TorchModule:
    """Lower ``program`` and compile it with PyAB's torch backend.

    Requires ``torch``. ``opts.target`` is forced to ``"torch"``.

    Returns
    -------
    TorchModule
        The compiled module; ``module.<name>`` is the callable.
    """
    from pyarraybackend.backends import torch_backend

    opts = _with_target(opts, "torch")
    stmts = as_function_def(program, name=name, opts=opts)
    return torch_backend.compile(stmts, options=options, dump_dir=dump_dir)


def compile_numpy(
    program: Program,
    *,
    name: str = "f",
    opts: LowerOpts | None = None,
    options: PythonBackendOptions | None = None,
    dump_dir: Path | str | None = None,
) -> NumpyModule:
    """Lower ``program`` and compile it with PyAB's numpy backend.

    Needs no torch. ``opts.target`` is forced to ``"numpy"``.

    Returns
    -------
    NumpyModule
        The compiled module; ``module.<name>`` is the callable.
    """
    from pyarraybackend.backends import python_backend

    opts = _with_target(opts, "numpy")
    stmts = as_function_def(program, name=name, opts=opts)
    return python_backend.compile(stmts, options=options, dump_dir=dump_dir)


def _with_target(opts: LowerOpts | None, target: Target) -> LowerOpts:
    if opts is None:
        return LowerOpts(target=target)
    if opts.target == target:
        return opts
    return LowerOpts(
        target=target,
        small_qr=opts.small_qr,
        op_lowerings=opts.op_lowerings,
        dtype_f64=opts.dtype_f64,
        collapse_vmap=opts.collapse_vmap,
        simplify_gsvd=opts.simplify_gsvd,
    )
