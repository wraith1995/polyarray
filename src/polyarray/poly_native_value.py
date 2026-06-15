"""Coefficient-type value handles for the ``native_py`` / ``native_cpp``
polynomial backends.

A :class:`ValueHandle` is a small frozen struct exposing every
coefficient-level op the poly engines call.  It is the genericity seam
that lets one engine module serve multiple coefficient types
(``double``, ``mpf``, ``quad``).

See ``plans/archive/poly_native_backend.md`` §1.2 for the full design.
Slice A1 shipped ``double``; slice A3 adds ``mpf`` (arbitrary precision
via :mod:`mpmath`) and ``quad`` (113-bit mpf, the closest portable
"quadruple precision" available without C-toolchain assumptions).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable

import mpmath


@dataclass(frozen=True, slots=True)
class ValueHandle:
    name: str
    zero: Any
    one: Any
    add: Callable[[Any, Any], Any]
    sub: Callable[[Any, Any], Any]
    mul: Callable[[Any, Any], Any]
    neg: Callable[[Any], Any]
    inv: Callable[[Any], Any]
    is_exact_zero: Callable[[Any], bool]
    is_one: Callable[[Any], bool]
    from_python: Callable[[Any], Any]
    to_float: Callable[[Any], float]


def _double_is_exact_zero(c: float) -> bool:
    return c == 0.0


def _double_is_one(c: float) -> bool:
    return c == 1.0


def _double_inv(c: float) -> float:
    return 1.0 / c


def _double_from_python(x: Any) -> float:
    # numpy float64 / float32 are subclasses of float but repr as
    # "np.float64(1.0)" — their coefficients would land in
    # ``_emit_poly_assign``'s ``repr(c)`` and produce code that
    # references an undefined ``np`` name.  Force every input to a
    # plain Python float so repr is always literal-parseable.
    if type(x) is float:
        return x
    return float(x)


DOUBLE = ValueHandle(
    name="double",
    zero=0.0,
    one=1.0,
    add=operator.add,
    sub=operator.sub,
    mul=operator.mul,
    neg=operator.neg,
    inv=_double_inv,
    is_exact_zero=_double_is_exact_zero,
    is_one=_double_is_one,
    from_python=_double_from_python,
    to_float=float,
)


# ---------------------------------------------------------------------------
# mpmath-mpf bindings — arbitrary precision (parametrised by dps/prec).
# ---------------------------------------------------------------------------

def _make_mpf_handle(name: str, prec_bits: int) -> ValueHandle:
    """Build a :class:`ValueHandle` bound to ``mpmath.mpf`` at a fixed
    binary precision.

    Coefficient ops route through ``mpmath`` with the precision set
    once at handle construction; per-op contexts are unnecessary
    because the global ``mpmath.mp`` precision is shared by every
    handle in the process — but we still pin ``mp.prec`` at handle
    construction so an `import` of this module establishes a known
    state for callers that haven't touched ``mp`` themselves.
    """
    # Set the global mpmath precision to at least what this handle
    # needs.  Don't shrink it below an existing higher value; another
    # caller may already be using a wider context.
    if mpmath.mp.prec < prec_bits:
        mpmath.mp.prec = prec_bits

    zero = mpmath.mpf(0)
    one = mpmath.mpf(1)

    def is_exact_zero(c: Any) -> bool:
        return c == 0

    def is_one(c: Any) -> bool:
        return c == 1

    def from_python(x: Any) -> Any:
        if isinstance(x, mpmath.mpf):
            return x
        return mpmath.mpf(x)

    def to_float_(c: Any) -> float:
        return float(c)

    def neg(c: Any) -> Any:
        return -c

    def inv(c: Any) -> Any:
        return one / c

    return ValueHandle(
        name=name,
        zero=zero,
        one=one,
        add=operator.add,
        sub=operator.sub,
        mul=operator.mul,
        neg=neg,
        inv=inv,
        is_exact_zero=is_exact_zero,
        is_one=is_one,
        from_python=from_python,
        to_float=to_float_,
    )


# ``mpf`` uses the mpmath default precision (53 bits, matching sympy's
# RR at dps=15).  Callers wanting more precision can mutate
# ``mpmath.mp.prec`` themselves *before* the first ring is built.
MPF = _make_mpf_handle("mpf", prec_bits=53)

# ``quad`` pins precision at 113 bits (binary128 / IEEE 754 quad).  We
# use mpmath rather than C ``__float128`` so the binding is portable
# across platforms (no GCC / libquadmath dependency).
QUAD = _make_mpf_handle("quad", prec_bits=113)


_HANDLES: dict[str, ValueHandle] = {
    "double": DOUBLE,
    "mpf": MPF,
    "quad": QUAD,
}


def get_value_handle(name: str) -> ValueHandle:
    """Look up a value handle by ``CHARTLIB_POLY_COEFF`` name."""
    handle = _HANDLES.get(name)
    if handle is not None:
        return handle
    raise ValueError(
        f"unknown CHARTLIB_POLY_COEFF {name!r}; "
        f"valid choices: {sorted(_HANDLES)!r}"
    )
