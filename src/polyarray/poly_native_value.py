"""Coefficient-type value handles for the ``native_py`` and ``native_cpp`` backends.

A :class:`ValueHandle` is a small frozen struct exposing every coefficient-level
operation a polynomial engine calls. It is the genericity seam that lets one engine
module serve several coefficient types by carrying the arithmetic as data rather than
branching on the type::

    polynomial engine  ──runs on──▶  ValueHandle
                                        ├─ double   Python float
                                        ├─ mpf      arbitrary precision (mpmath)
                                        └─ quad     113-bit mpf

The ``mpf`` handle gives arbitrary precision via :mod:`mpmath`, and ``quad`` pins that
to 113 bits — the closest portable quadruple precision available without a C toolchain.
The active handle is selected by the ``CHARTLIB_POLY_COEFF`` environment variable
through :func:`get_value_handle`.
"""
from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mpmath


@dataclass(frozen=True, slots=True)
class ValueHandle[C]:
    """The coefficient arithmetic one polynomial engine runs on.

    ``C`` is the coefficient type: :class:`float` for ``double``, ``mpmath.mpf`` for
    ``mpf`` and ``quad``. Every field is an operation the engine calls rather than a
    method, so the engine stays free of coefficient-type dispatch.

    Attributes
    ----------
    name
        The ``CHARTLIB_POLY_COEFF`` name this handle answers to.
    zero, one
        The additive and multiplicative identities.
    add, sub, mul, neg, inv
        Field arithmetic on coefficients.
    is_exact_zero, is_one
        Exact identity tests, used to drop structurally zero terms and to skip
        multiplications by one.
    from_python
        Coerce an arbitrary Python number to a coefficient.
    to_float
        Narrow a coefficient to a Python float.
    """

    name: str
    zero: C
    one: C
    add: Callable[[C, C], C]
    sub: Callable[[C, C], C]
    mul: Callable[[C, C], C]
    neg: Callable[[C], C]
    inv: Callable[[C], C]
    is_exact_zero: Callable[[C], bool]
    is_one: Callable[[C], bool]
    from_python: Callable[[object], C]
    to_float: Callable[[C], float]


def _double_is_exact_zero(c: float) -> bool:
    """Report whether a double coefficient is exactly zero."""
    return c == 0.0


def _double_is_one(c: float) -> bool:
    """Report whether a double coefficient is exactly one."""
    return c == 1.0


def _double_inv(c: float) -> float:
    """Return the reciprocal of a double coefficient."""
    return 1.0 / c


def _double_from_python(x: object) -> float:
    """Coerce ``x`` to a plain Python float.

    numpy's float64 / float32 are float subclasses but repr as ``np.float64(1.0)``.
    Emitted code renders coefficients with ``repr``, so a numpy scalar would produce a
    source line referencing an undefined ``np``. Narrowing here keeps every repr
    literal-parseable.
    """
    if type(x) is float:
        return x
    return float(x)


DOUBLE: ValueHandle[float] = ValueHandle(
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

def _make_mpf_handle(name: str, prec_bits: int) -> ValueHandle[mpmath.mpf]:
    """Build a :class:`ValueHandle` bound to ``mpmath.mpf`` at a fixed binary precision.

    Coefficient ops route through ``mpmath``, whose ``mp.prec`` is global to the
    process. Pinning it here means importing this module establishes a known precision
    for callers that never touch ``mp`` themselves.

    Parameters
    ----------
    name
        The ``CHARTLIB_POLY_COEFF`` name the handle answers to.
    prec_bits
        Binary precision to guarantee. An already-wider global precision is left alone,
        since another caller may depend on it.

    Returns
    -------
    ValueHandle
        A handle over ``mpmath.mpf`` coefficients.
    """
    # Set the global mpmath precision to at least what this handle
    # needs.  Don't shrink it below an existing higher value; another
    # caller may already be using a wider context.
    if mpmath.mp.prec < prec_bits:
        mpmath.mp.prec = prec_bits

    zero = mpmath.mpf(0)
    one = mpmath.mpf(1)

    def is_exact_zero(c: mpmath.mpf) -> bool:
        return c == 0

    def is_one(c: mpmath.mpf) -> bool:
        return c == 1

    def from_python(x: object) -> mpmath.mpf:
        if isinstance(x, mpmath.mpf):
            return x
        return mpmath.mpf(x)

    def to_float_(c: mpmath.mpf) -> float:
        return float(c)

    def neg(c: mpmath.mpf) -> mpmath.mpf:
        return -c

    def inv(c: mpmath.mpf) -> mpmath.mpf:
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


#: mpmath's default precision (53 bits, matching sympy's ``RR`` at dps=15). Callers
#: wanting more can raise ``mpmath.mp.prec`` themselves *before* the first ring is built.
MPF = _make_mpf_handle("mpf", prec_bits=53)

#: 113 bits, i.e. binary128 / IEEE 754 quadruple. Backed by mpmath rather than C
#: ``__float128`` so the binding is portable — no GCC or libquadmath dependency.
QUAD = _make_mpf_handle("quad", prec_bits=113)


_HANDLES: dict[str, ValueHandle[Any]] = {
    "double": DOUBLE,
    "mpf": MPF,
    "quad": QUAD,
}


def get_value_handle(name: str) -> ValueHandle[Any]:
    """Look up a value handle by its ``CHARTLIB_POLY_COEFF`` name.

    Parameters
    ----------
    name
        One of ``"double"``, ``"mpf"``, ``"quad"``.

    Returns
    -------
    ValueHandle
        The handle for that coefficient type.

    Raises
    ------
    ValueError
        If ``name`` is not a known coefficient type.
    """
    handle = _HANDLES.get(name)
    if handle is not None:
        return handle
    raise ValueError(
        f"unknown CHARTLIB_POLY_COEFF {name!r}; "
        f"valid choices: {sorted(_HANDLES)!r}"
    )
