"""Whole-program polynomial-degree estimation (fem task #9 — lifted from pointwise).

:func:`program_degree` walks a :class:`~polyarray.ir.Program`'s statement graph and
propagates a POLYNOMIAL DEGREE per value from a per-input ``seed``, answering "what
polynomial degree is the program's output in the seeded variables?" — the question a
quadrature-order chooser asks. Consumers seed their domain knowledge (pointwise seeds
field inputs with ``FE degree − derivative order`` and affine geometry with 0) and get
back a single float, ``inf`` meaning "not a polynomial of the seeds" (rational /
algebraic / unknown — the caller supplies its own order there).

Degree semantics per op category (sound over-estimation: products SUM, sums MAX;
only UNDER-estimation is a correctness bug):

* the AFFINE-CONSTANT short-circuit — if EVERY operand is degree 0 the output is a
  constant function of constants ⇒ degree 0, WHATEVER the op (this is what makes a
  constant geometry chain — SVD/QR/pinv fed by a constant Jacobian — degree 0 for free);
* ``zero_ops`` — output depends on SHAPES/structure, not values ⇒ 0;
* ``passthrough_ops`` — reorder / select / additive / scale-by-constant ⇒ MAX(operands);
* ``multilinear_ops`` — products / contractions ⇒ SUM(operands);
* ``CallOp`` — a vmap/llam wrapper is NOT opaque: unwrap the body Program
  (:func:`polyarray.forward._body_of`) and recurse, seeding its inputs by position;
* ``DetOp`` — det IS a polynomial of its entries: an ``(n, n)`` operand of degree ``d``
  gives degree ``≤ n·d`` (a sum of n-fold entry products); an unknown/dynamic operand
  shape falls back to ``inf``;
* anything else on a seed-dependent operand ⇒ ``inf`` (Inv/Pinv/Solve/Sqrt/Svd/… —
  genuinely rational or algebraic in the operand).

The categories are keyed by op CLASS NAME and extendable per call (the
``to_numpy_source(op_renderers=)`` pattern): front ends pass their own op names —
pointwise/grassmann's ``_ProjectOp``/``_AddOp``/… — without polyarray importing them.

Leaf cells that are :class:`~polyarray.rational.RationalFunction`s (a ``RationalRef``
splice, or an object-dtype ref the walk did not produce) are scored with
``RationalFunction.compute_degree`` under ``gen_deg`` — the caller's generator-name →
degree weighting (default: every generator weighs 0).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from . import ir as _ir
from .forward import _body_of

_INF = float("inf")

# polyarray's NATIVE op categories. Front-end ops (grassmann/chartlib/pointwise `_…Op`s)
# are supplied per call via the keyword sets; ops in no category fall to the
# nonlinear/unknown branch (`inf` on a seed-dependent operand — over-estimation is safe).
DEFAULT_ZERO_OPS: frozenset[str] = frozenset()
DEFAULT_PASSTHROUGH_OPS: frozenset[str] = frozenset({"AssertOp", "IdentityOp", "MoveaxisOp"})
DEFAULT_MULTILINEAR_OPS: frozenset[str] = frozenset({"EinsumStmtOp", "EinsumOp", "TensordotOp"})


def _op_degree(fn: Any, in_degs: list[float], zero: frozenset[str],
               passthrough: frozenset[str], multilinear: frozenset[str]) -> float:
    """Output degree of a non-``CallOp`` statement from its operand degrees."""
    name = type(fn).__name__
    if name in zero:
        return 0.0
    if not in_degs:
        return 0.0
    if all(d == 0 for d in in_degs):                              # affine-constant short-circuit
        return 0.0
    if any(d == _INF for d in in_degs):
        return _INF
    if name in passthrough:
        return float(max(in_degs))
    if name in multilinear:
        return float(sum(in_degs))
    return _INF                                                   # nonlinear/rational/unknown on a seed


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
    """The polynomial degree of ``program``'s output(s) given per-input degrees ``seed``.

    ``seed`` maps input names to degrees (missing names default to 0). ``gen_deg`` scores
    stray ``RationalFunction`` leaf cells by generator name (default 0). The ``*_ops``
    sets EXTEND the native categories (module docstring); pass a front end's op names
    there. Reads ``outputs["result"]`` when present, else the max over all outputs.
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

    def ref_deg(ref: Any) -> float:
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
