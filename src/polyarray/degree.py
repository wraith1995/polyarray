"""Whole-program polynomial-degree estimation.

:func:`program_degree` walks a :class:`~polyarray.ir.Program`'s statement graph and
propagates a polynomial degree per value from a per-input ``seed``, answering the
question "what polynomial degree does the program's output have in the seeded
variables?" A caller that must pick an approximation or integration order seeds its own
domain knowledge and gets back a single float. A result of ``inf`` means the output is
not a polynomial of the seeds — it is rational, algebraic, or unknown — and the caller
supplies its own order there.

Each op contributes a degree to its output from its operands' degrees. The estimate is
a sound over-estimate — products sum, sums take the max — because only under-estimation
is a correctness bug. The flat categories are::

    zero          output depends on shapes / structure, not values   ⇒ 0
    passthrough   reorder / select / additive / scale-by-constant     ⇒ max(operands)
    multilinear   products / contractions                             ⇒ sum(operands)
    rational      genuinely rational / algebraic / non-polynomial     ⇒ inf on a seed

Ahead of them sits the affine-constant short-circuit: when every operand is degree 0 the
output is a constant function of constants and so is degree 0 whatever the op, which is what
makes a constant-input chain — an SVD/QR/pinv fed by a constant matrix — degree 0 for free.
Two ops carry richer handling than a flat category. ``DetOp`` is a polynomial of its
entries, so an ``(n, n)`` operand of degree ``d`` gives degree ``≤ n·d`` (a sum of n-fold
entry products), falling back to ``inf`` on an unknown or dynamic operand shape. ``CallOp``
is not opaque: its vmap wrapper body Program is unwrapped
(:func:`polyarray.forward._body_of`) and recursed into, seeding its inputs by position.
Anything else on a seed-dependent operand is ``inf`` (Inv/Pinv/Solve/Sqrt/Svd/…, genuinely
rational or algebraic in the operand).

The categories are keyed by op class name and extendable per call (the
``to_numpy_source(op_renderers=)`` pattern): front ends pass their own op names —
their own ``_ProjectOp``/``_AddOp``/… — without polyarray importing them.

Leaf cells that are :class:`~polyarray.rational.RationalFunction`s (a ``RationalRef``
splice, or an object-dtype ref the walk did not produce) are scored with
``RationalFunction.compute_degree`` under ``gen_deg`` — the caller's generator-name →
degree weighting (default: every generator weighs 0).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from . import ir as _ir
from .forward import _body_of

_INF = float("inf")

# polyarray's native op degree categories, keyed by the class OBJECT rather than its
# name: a renamed or relocated builtin then breaks loudly at import instead of falling
# through to "unknown op" ⇒ `inf` degree ⇒ a wrong sample-point order.
# Every Stmt.fn op class in `ir` must appear here (guarded by tests/test_degree_coverage.py).
#   zero        — output depends on SHAPES/structure, not values      ⇒ 0
#   passthrough — reorder / select / additive / scale-by-constant     ⇒ MAX(operands)
#   multilinear — products / contractions                             ⇒ SUM(operands)
#   rational    — genuinely rational / algebraic / non-polynomial     ⇒ inf on a seed
#   special     — richer handling in `program_degree` (DetOp, CallOp), not a flat category
DEG_ZERO, DEG_PASS, DEG_MULT, DEG_RAT, DEG_SPECIAL = (
    "zero", "passthrough", "multilinear", "rational", "special")

DEFAULT_DEGREE_KINDS: dict[type, str] = {
    # zero — structural / constant (degree 0 whatever the operands)
    _ir.AxisLenOp: DEG_ZERO, _ir.CompRankOp: DEG_ZERO, _ir.ConstOp: DEG_ZERO,
    _ir.DynEyeOp: DEG_ZERO, _ir.DynEyeTensorOp: DEG_ZERO, _ir.DynZerosOp: DEG_ZERO,
    _ir.EyeOp: DEG_ZERO, _ir.MulAxisDimOp: DEG_ZERO, _ir.ProdDimOp: DEG_ZERO,
    _ir.ProdShapeOp: DEG_ZERO, _ir.RankOp: DEG_ZERO, _ir.ScaleAxisDimOp: DEG_ZERO,
    _ir.SumDimOp: DEG_ZERO, _ir.SumShapeOp: DEG_ZERO,
    # passthrough — reorder / select / additive / scale-by-constant
    _ir.AddOp: DEG_PASS, _ir.AssertOp: DEG_PASS, _ir.BlockDiagOp: DEG_PASS,
    _ir.BlockRepeatOp: DEG_PASS, _ir.ColStackOp: DEG_PASS, _ir.ConcatOp: DEG_PASS,
    _ir.DynBlockRepeatOp: DEG_PASS, _ir.FirstColsOp: DEG_PASS, _ir.HStackOp: DEG_PASS,
    _ir.IdentityOp: DEG_PASS, _ir.LastColsOp: DEG_PASS, _ir.MoveaxisOp: DEG_PASS,
    _ir.ReshapeOp: DEG_PASS, _ir.ScaleOp: DEG_PASS, _ir.SwitchOp: DEG_PASS,
    _ir.TransposeOp: DEG_PASS,
    # multilinear — products / contractions (degree = SUM of operands)
    _ir.ComposeViaStdOp: DEG_MULT, _ir.EinsumOp: DEG_MULT, _ir.EinsumStmtOp: DEG_MULT,
    _ir.EmbedOp: DEG_MULT, _ir.KronFreeOp: DEG_MULT, _ir.KronOp: DEG_MULT,
    _ir.ProjectOp: DEG_MULT, _ir.ScaleByOp: DEG_MULT, _ir.TensordotOp: DEG_MULT,
    # rational / algebraic / non-polynomial ⇒ inf on a seed-dependent operand
    _ir.AbsOp: DEG_RAT, _ir.GSvdFullOp: DEG_RAT, _ir.GSvdOp: DEG_RAT, _ir.InvOp: DEG_RAT,
    _ir.InvTransposeOp: DEG_RAT, _ir.MetricOrthonormalOp: DEG_RAT, _ir.PinvOp: DEG_RAT,
    _ir.QrOp: DEG_RAT, _ir.SignOp: DEG_RAT, _ir.SinvFullOp: DEG_RAT, _ir.SolveOp: DEG_RAT,
    _ir.SqrtOp: DEG_RAT, _ir.SqrtSpdOp: DEG_RAT, _ir.SvdOp: DEG_RAT, _ir.WhileOp: DEG_RAT,
    # special — DetOp (≤ n·d) and CallOp (recurse) are handled in `program_degree`
    _ir.DetOp: DEG_SPECIAL, _ir.CallOp: DEG_SPECIAL,
}

# Name-string category sets for FRONT-END ops, which polyarray must not import. A front
# end extends these through `program_degree`'s ``*_ops`` kwargs, or tags its op class with
# ``_DEGREE_KIND``. polyarray's own builtins are covered type-safely above.
DEFAULT_ZERO_OPS: frozenset[str] = frozenset()
DEFAULT_PASSTHROUGH_OPS: frozenset[str] = frozenset({"AssertOp", "IdentityOp", "MoveaxisOp"})
DEFAULT_MULTILINEAR_OPS: frozenset[str] = frozenset({"EinsumStmtOp", "EinsumOp", "TensordotOp"})


def _op_degree(fn: _ir.StmtOp, in_degs: list[float], zero: frozenset[str],
               passthrough: frozenset[str], multilinear: frozenset[str]) -> float:
    """Return the output degree of a non-``CallOp`` statement from its operand degrees.

    Categories resolve in order: the type-keyed native map, then an op-carried
    ``_DEGREE_KIND`` tag, then the per-call name sets. Unknown ops give ``inf``, the
    safe over-estimate.

    Parameters
    ----------
    fn
        The statement's op.
    in_degs
        Degree of each operand, in operand order.
    zero, passthrough, multilinear
        Front-end op names to add to the corresponding native category.

    Returns
    -------
    float
        The output degree, or ``inf`` when the output is not polynomial in the seeds.
    """
    kind = DEFAULT_DEGREE_KINDS.get(type(fn)) or getattr(fn, "_DEGREE_KIND", None)
    name = type(fn).__name__
    if kind == DEG_ZERO or name in zero:
        return 0.0
    if not in_degs:
        return 0.0
    if all(d == 0 for d in in_degs):                              # affine-constant short-circuit
        return 0.0
    if any(d == _INF for d in in_degs):
        return _INF
    if kind == DEG_PASS or name in passthrough:
        return float(max(in_degs))
    if kind == DEG_MULT or name in multilinear:
        return float(sum(in_degs))
    return _INF                                                   # rational/nonlinear/unknown on a seed


def program_degree(
    program: _ir.Program,
    seed: Mapping[str, float],
    *,
    gen_deg: Callable[[str], int] | None = None,
    zero_ops: frozenset[str] | set[str] = frozenset(),
    passthrough_ops: frozenset[str] | set[str] = frozenset(),
    multilinear_ops: frozenset[str] | set[str] = frozenset(),
    _depth: int = 0,
) -> float:
    """Return the polynomial degree of ``program``'s outputs given per-input degrees.

    Reads ``outputs["result"]`` when present, else takes the maximum over all outputs.

    Parameters
    ----------
    program
        The program to score.
    seed
        Degree of each input by name; missing names count as 0.
    gen_deg
        Degree weight of a bare generator name, used to score stray
        :class:`~polyarray.rational.RationalFunction` leaf cells. Defaults to 0.
    zero_ops, passthrough_ops, multilinear_ops
        Front-end op names extending the native degree categories described in the
        module docstring.
    _depth
        Recursion depth through nested ``CallOp`` bodies; a backstop against runaway
        nesting.

    Returns
    -------
    float
        The output degree, or ``inf`` when the output is not polynomial in the seeds.
    """
    zero = DEFAULT_ZERO_OPS | frozenset(zero_ops)
    passthrough = DEFAULT_PASSTHROUGH_OPS | frozenset(passthrough_ops)
    multilinear = DEFAULT_MULTILINEAR_OPS | frozenset(multilinear_ops)
    if gen_deg is None:
        def gen_deg(_name: str) -> int:
            return 0
    if _depth > 64:                                               # runaway CallOp nesting backstop
        return _INF

    deg_by_id: dict[int, float] = {}
    for name, sa in program.input_arrays.items():
        deg_by_id[id(sa._cells)] = float(seed.get(name, 0.0))

    def ref_deg(ref: _ir.Ref | _ir.SymArray) -> float:
        rf = getattr(ref, "rf", None)
        if rf is not None and hasattr(rf, "compute_degree"):      # RationalRef splice → weighted leaf
            return float(rf.compute_degree({n: gen_deg(n) for n in rf._ring.names}))
        cells = getattr(ref, "_cells", None)
        if cells is None:                                         # Const / IntAtomRef / Input|OutputRef
            iname = getattr(ref, "name", None)
            if iname is not None and iname in program.input_arrays:
                return float(seed.get(iname, 0.0))
            return 0.0
        d = deg_by_id.get(id(cells))
        if d is not None:
            return d
        if getattr(cells, "dtype", None) is not None and cells.dtype.kind == "f":
            return 0.0                                            # an inline numeric constant
        # an object-dtype ref this walk did not produce: score its RF cells as weighted leaves
        try:
            arr = np.asarray(cells)
            degs = [c.compute_degree({n: gen_deg(n) for n in c._ring.names})
                    for c in arr.reshape(-1) if hasattr(c, "compute_degree")]
            return float(max(degs, default=0))
        except Exception:
            return _INF

    for st in program.statements:
        in_degs = [ref_deg(r) for r in st.in_]
        fn = st.fn
        if isinstance(fn, _ir.DetOp):
            # det = Σ ± Π of n entries — polynomial of degree ≤ n·d in a degree-d operand.
            d0 = in_degs[0] if in_degs else 0.0
            if d0 == 0.0 or d0 == _INF:
                d = d0
            else:
                cells = getattr(st.in_[0], "_cells", None)
                shp = getattr(cells, "shape", None)
                n = shp[0] if (shp and len(shp) == 2 and isinstance(shp[0], int)
                               and shp[0] == shp[1]) else None
                d = float(n) * d0 if n is not None else _INF     # unknown/dynamic shape → safe inf
        elif isinstance(fn, _ir.CallOp):
            if in_degs and all(d == 0 for d in in_degs):          # constant call ⇒ constant
                d = 0.0
            else:
                body = fn.fn if isinstance(fn.fn, _ir.Program) else _body_of(fn.fn)
                if body is None:
                    d = _INF
                else:
                    sub_seed = {inp.name: in_degs[i] if i < len(in_degs) else 0.0
                                for i, inp in enumerate(body.inputs)}
                    d = program_degree(body, sub_seed, gen_deg=gen_deg, zero_ops=zero,
                                       passthrough_ops=passthrough, multilinear_ops=multilinear,
                                       _depth=_depth + 1)
        else:
            d = _op_degree(fn, in_degs, zero, passthrough, multilinear)
        for o in st.out:
            deg_by_id[id(o._cells)] = d

    out = program.outputs.get("result")
    outs = [out] if out is not None else list(program.outputs.values())
    # resolve via `ref_deg` (not a bare id-lookup) so a PASSTHROUGH output — `result` IS a
    # program input — is read by name/cells, not defaulted to inf.
    return max((ref_deg(sa) for sa in outs), default=0.0)
