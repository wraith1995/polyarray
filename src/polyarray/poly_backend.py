"""Polynomial-arithmetic backend abstraction.

The symbolic interpreter stores rational functions as
``num / den`` where ``num`` and ``den`` are sparse multivariate
polynomials over a coefficient field.  Two backends are wired here:

* ``"sympy"`` — :class:`sympy.polys.rings.PolyElement` over
  :data:`sympy.RR` (53-bit ``mpf`` coefficients).  Pure Python; the
  reference implementation.

* ``"flint"`` — :class:`flint.fmpq_mpoly` from ``python-flint``
  (FLINT C library, exact rational coefficients).  Roughly two orders of
  magnitude faster at multivariate multiplication, and an order of magnitude
  faster at ring construction, at realistic generator counts.

The backend is selected by the :data:`BACKEND` global (default
``"sympy"``).  Set the ``CHARTLIB_POLY_BACKEND`` environment variable
to ``"flint"`` to switch.

Both backends provide a uniform :class:`Poly` and :class:`Ring` duck-typed
protocol, which :class:`~polyarray.rational.RationalFunction` works against.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import mpmath
import sympy as sp
from sympy.polys.rings import PolyElement, PolyRing

if TYPE_CHECKING:
    from flint import fmpq_mpoly as _fmpq_mpoly
    from flint import fmpq_mpoly_ctx as _fmpq_mpoly_ctx

try:  # python-flint is optional; the alias below must resolve without it
    from flint import fmpq as _fmpq
except ImportError:  # with no flint there is no exact-rational coefficient type
    _fmpq = mpmath.mpf

#: A ground coefficient of whichever backend is active: a 53-bit ``mpf`` real under
#: sympy, an exact rational under flint.
type Coeff = mpmath.mpf | _fmpq

#: Anything a ring coerces to a :data:`Coeff` when building a polynomial.
type CoeffLike = Coeff | int | float | sp.Rational

#: An exponent vector, one entry per ring generator.
type Monom = tuple[int, ...]

# Backend default is resolved below once flint availability is known: when
# unset, prefer the sparse flint backend (multivariate monomials are stored
# sparsely, so large shared rings are cheap and ring-join churn is far lower
# than sympy's dense PolyRing) and fall back to sympy if python-flint is
# absent.  Set CHARTLIB_POLY_BACKEND explicitly to override.
BACKEND_NAME = os.environ.get("CHARTLIB_POLY_BACKEND")
COEFF_NAME = os.environ.get("CHARTLIB_POLY_COEFF", "double")


# ======================================================================
# Sympy backend
# ======================================================================

_RR = sp.RR


class _SympyRing:
    """A sympy :class:`PolyRing` behind the backend-neutral ring protocol.

    Holds a cached generator-name tuple and exposes the subset of the sympy ring API
    that :class:`~polyarray.rational.RationalFunction` and the eval codegen reach for.
    """

    __slots__ = ("_ring", "_names", "_one", "_zero", "_gens")

    def __init__(self, ring: PolyRing, names: tuple[str, ...]) -> None:
        self._ring = ring
        self._names = names
        self._one = _SympyPoly(ring.one, self)
        self._zero = _SympyPoly(ring.zero, self)
        self._gens = tuple(_SympyPoly(g, self) for g in ring.gens)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_gens(self) -> int:
        return len(self._names)

    @property
    def one(self) -> _SympyPoly:
        return self._one

    @property
    def zero(self) -> _SympyPoly:
        return self._zero

    @property
    def gens(self) -> tuple[_SympyPoly, ...]:
        return self._gens

    def gen(self, i: int) -> _SympyPoly:
        return self._gens[i]

    def ground_new(self, value: CoeffLike) -> _SympyPoly:
        """Return the constant polynomial ``value``."""
        if isinstance(value, sp.Rational):
            coerced = _RR.convert(value)
        elif isinstance(value, (int, float)):
            coerced = _RR(value)
        else:
            coerced = value
        return _SympyPoly(self._ring.ground_new(coerced), self)

    def from_dict(self, terms: dict[Monom, CoeffLike]) -> _SympyPoly:
        """Build a polynomial from an exponent-vector to coefficient mapping."""
        if not terms:
            return self._zero
        coerced = {m: _RR(c) if not isinstance(c, sp.Rational) else _RR.convert(c)
                   for m, c in terms.items()}
        return _SympyPoly(self._ring.from_dict(coerced), self)


class _SympyPoly:
    """A sympy :class:`PolyElement` with a back-reference to its :class:`_SympyRing`.

    Implements the ``+ - * == is_zero`` arithmetic the rational-function layer needs.
    Mixed operands coerce: ``Poly + scalar`` lifts the scalar to a constant polynomial
    in the same ring.
    """

    __slots__ = ("_p", "_ring")

    def __init__(self, p: PolyElement, ring: _SympyRing) -> None:
        self._p = p
        self._ring = ring

    @property
    def ring(self) -> _SympyRing:
        return self._ring

    @property
    def is_zero(self) -> bool:
        return bool(self._p.is_zero)

    @property
    def is_one(self) -> bool:
        return self._p == self._ring._ring.one

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _SympyPoly):
            return bool(self._p == other._p)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._p)

    def __neg__(self) -> _SympyPoly:
        return _SympyPoly(-self._p, self._ring)

    def __add__(self, other: _SympyPoly | int | float) -> _SympyPoly:
        if isinstance(other, _SympyPoly):
            return _SympyPoly(self._p + other._p, self._ring)
        if isinstance(other, (int, float)):
            return _SympyPoly(self._p + _RR(other), self._ring)
        return NotImplemented

    def __radd__(self, other: _SympyPoly | int | float) -> _SympyPoly:
        return self.__add__(other)

    def __sub__(self, other: _SympyPoly | int | float) -> _SympyPoly:
        if isinstance(other, _SympyPoly):
            return _SympyPoly(self._p - other._p, self._ring)
        if isinstance(other, (int, float)):
            return _SympyPoly(self._p - _RR(other), self._ring)
        return NotImplemented

    def __mul__(self, other: _SympyPoly | int | float) -> _SympyPoly:
        if isinstance(other, _SympyPoly):
            return _SympyPoly(self._p * other._p, self._ring)
        if isinstance(other, (int, float)):
            return _SympyPoly(self._p * _RR(other), self._ring)
        return NotImplemented

    def __rmul__(self, other: _SympyPoly | int | float) -> _SympyPoly:
        return self.__mul__(other)

    def __pow__(self, n: int) -> _SympyPoly:
        return _SympyPoly(self._p ** int(n), self._ring)

    def total_degree(self) -> int:
        return int(self._p.total_degree())

    def terms(self) -> list[tuple[Monom, Coeff]]:
        return list(self._p.terms())

    def monoms(self) -> list[Monom]:
        return list(self._p.monoms())

    def n_terms(self) -> int:
        return len(self._p.to_dict())

    @property
    def LC(self) -> Coeff:
        if self._p.is_zero:
            return _RR(0)
        return self._p.LC

    def cancel(self, other: _SympyPoly) -> tuple[_SympyPoly, _SympyPoly]:
        n2, d2 = self._p.cancel(other._p)
        return _SympyPoly(n2, self._ring), _SympyPoly(d2, self._ring)

    def as_expr(self) -> sp.Expr:
        """Render as a sympy expression, for display."""
        return self._p.as_expr()

    def __repr__(self) -> str:
        return f"_SympyPoly({self._p!r})"


_SYMPY_RING_CACHE: dict[tuple[str, ...], _SympyRing] = {}


def _sympy_make_ring(names: Sequence[str]) -> _SympyRing:
    key = tuple(names)
    cached = _SYMPY_RING_CACHE.get(key)
    if cached is not None:
        return cached
    raw = PolyRing(key, _RR) if key else PolyRing((), _RR)
    ring = _SympyRing(raw, key)
    _SYMPY_RING_CACHE[key] = ring
    return ring


def _sympy_lift(poly: _SympyPoly, target: _SympyRing) -> _SympyPoly:
    src = poly._ring
    if src is target:
        return poly
    src_names = src._names
    tgt_names = target._names
    if src_names == tgt_names:
        return _SympyPoly(target._ring.from_dict(dict(poly._p.terms())), target)
    missing = set(src_names) - set(tgt_names)
    if missing:
        raise ValueError(
            f"cannot lift polynomial: target ring is missing generators "
            f"{sorted(missing)!r}"
        )
    n_tgt = len(tgt_names)
    tgt_index = {name: i for i, name in enumerate(tgt_names)}
    pos_map = [tgt_index[name] for name in src_names]
    new_dict: dict[tuple[int, ...], Any] = {}
    for monom, coeff in poly._p.terms():
        new_monom = [0] * n_tgt
        for src_pos, exp in enumerate(monom):
            if exp:
                new_monom[pos_map[src_pos]] = exp
        new_dict[tuple(new_monom)] = coeff
    return _SympyPoly(target._ring.from_dict(new_dict), target)


# ======================================================================
# Flint backend (python-flint)
# ======================================================================

fmpq: type[_fmpq] | None
fmpq_mpoly: type[_fmpq_mpoly] | None
fmpq_mpoly_ctx: type[_fmpq_mpoly_ctx] | None
try:
    from flint import fmpq, fmpq_mpoly, fmpq_mpoly_ctx
    _FLINT_AVAILABLE = True
except ImportError:
    fmpq = fmpq_mpoly = fmpq_mpoly_ctx = None
    _FLINT_AVAILABLE = False


def _float_to_fmpq(x: CoeffLike) -> _fmpq:
    """Convert a Python float or int to an exact :class:`flint.fmpq`.

    :class:`fractions.Fraction` recovers the exact dyadic rational behind a float, so no
    precision is lost against the IEEE 754 double. Integers pass through directly, and an
    ``fmpq`` is returned unchanged.

    Raises
    ------
    TypeError
        If ``x`` is neither an ``fmpq``, an int, nor a float.
    """
    if isinstance(x, fmpq):
        return x
    if isinstance(x, int):
        return fmpq(x)
    if isinstance(x, float):
        n, d = Fraction(x).as_integer_ratio()
        return fmpq(n, d)
    raise TypeError(f"cannot convert {x!r} ({type(x).__name__}) to fmpq")


def _fmpq_to_float(q: _fmpq) -> float:
    """Narrow an exact flint rational to a Python float."""
    return float(q)


class _FlintRing:
    """A flint ``fmpq_mpoly_ctx`` behind the backend-neutral ring protocol."""

    __slots__ = ("_ctx", "_names", "_one", "_zero", "_gens")

    def __init__(self, ctx: _fmpq_mpoly_ctx, names: tuple[str, ...]) -> None:
        self._ctx = ctx
        self._names = names
        self._zero = _FlintPoly(ctx.from_dict({}), self)
        if names:
            one_monom = (0,) * len(names)
        else:
            one_monom = ()
        self._one = _FlintPoly(ctx.from_dict({one_monom: fmpq(1)}), self)
        if names:
            self._gens = tuple(
                _FlintPoly(ctx.gen(i), self) for i in range(len(names))
            )
        else:
            self._gens = ()

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_gens(self) -> int:
        return len(self._names)

    @property
    def one(self) -> _FlintPoly:
        return self._one

    @property
    def zero(self) -> _FlintPoly:
        return self._zero

    @property
    def gens(self) -> tuple[_FlintPoly, ...]:
        return self._gens

    def gen(self, i: int) -> _FlintPoly:
        return self._gens[i]

    def ground_new(self, value: CoeffLike) -> _FlintPoly:
        """Return the constant polynomial ``value``."""
        if not self._names:
            return _FlintPoly(self._ctx.from_dict({(): _float_to_fmpq(value)}), self)
        one_monom = (0,) * len(self._names)
        return _FlintPoly(
            self._ctx.from_dict({one_monom: _float_to_fmpq(value)}), self
        )

    def from_dict(self, terms: dict[Monom, CoeffLike]) -> _FlintPoly:
        """Build a polynomial from an exponent-vector to coefficient mapping."""
        if not terms:
            return self._zero
        coerced = {m: _float_to_fmpq(c) for m, c in terms.items()}
        return _FlintPoly(self._ctx.from_dict(coerced), self)


class _FlintPoly:
    __slots__ = ("_p", "_ring")

    def __init__(self, p: _fmpq_mpoly, ring: _FlintRing) -> None:
        self._p = p
        self._ring = ring

    @property
    def ring(self) -> _FlintRing:
        return self._ring

    @property
    def is_zero(self) -> bool:
        return bool(self._p.is_zero())

    @property
    def is_one(self) -> bool:
        return bool(self._p.is_one())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FlintPoly):
            return bool(self._p == other._p)
        return NotImplemented

    def __hash__(self) -> int:
        # fmpq_mpoly may not hash; fall back to repr identity.
        return hash(repr(self._p))

    def __neg__(self) -> _FlintPoly:
        return _FlintPoly(-self._p, self._ring)

    def __add__(self, other: _FlintPoly | int | float) -> _FlintPoly:
        if isinstance(other, _FlintPoly):
            return _FlintPoly(self._p + other._p, self._ring)
        if isinstance(other, (int, float)):
            return _FlintPoly(self._p + _float_to_fmpq(other), self._ring)
        return NotImplemented

    def __radd__(self, other: _FlintPoly | int | float) -> _FlintPoly:
        return self.__add__(other)

    def __sub__(self, other: _FlintPoly | int | float) -> _FlintPoly:
        if isinstance(other, _FlintPoly):
            return _FlintPoly(self._p - other._p, self._ring)
        if isinstance(other, (int, float)):
            return _FlintPoly(self._p - _float_to_fmpq(other), self._ring)
        return NotImplemented

    def __mul__(self, other: _FlintPoly | int | float) -> _FlintPoly:
        if isinstance(other, _FlintPoly):
            return _FlintPoly(self._p * other._p, self._ring)
        if isinstance(other, (int, float)):
            return _FlintPoly(self._p * _float_to_fmpq(other), self._ring)
        return NotImplemented

    def __rmul__(self, other: _FlintPoly | int | float) -> _FlintPoly:
        return self.__mul__(other)

    def __pow__(self, n: int) -> _FlintPoly:
        return _FlintPoly(self._p ** int(n), self._ring)

    def total_degree(self) -> int:
        if not self._ring._names:
            return 0
        if self._p.is_zero():
            return 0
        return int(max(sum(m) for m in self._p.monoms()))

    def terms(self) -> list[tuple[Monom, Coeff]]:
        return list(self._p.terms())

    def monoms(self) -> list[Monom]:
        return list(self._p.monoms())

    def n_terms(self) -> int:
        return len(self._p.monoms())

    @property
    def LC(self) -> Coeff:
        if self._p.is_zero():
            return fmpq(0)
        return self._p.leading_coefficient()

    def cancel(self, other: _FlintPoly) -> tuple[_FlintPoly, _FlintPoly]:
        # Flint exposes ``gcd`` on fmpq_mpoly; ``self / gcd`` is exact
        # division.  Skip the gcd call (which is non-trivial) when either
        # operand is one — ``a / 1 == a``.
        if self._p.is_one() or other._p.is_one():
            return self, other
        if self._p.is_zero():
            return self, other
        g = self._p.gcd(other._p)
        if g.is_one():
            return self, other
        return _FlintPoly(self._p / g, self._ring), _FlintPoly(other._p / g, self._ring)

    def as_expr(self) -> sp.Expr:
        """Render as a sympy expression, for display."""
        # Slow; not used on hot paths.
        names = self._ring._names
        syms = sp.symbols(names) if names else ()
        if isinstance(syms, sp.Symbol):
            syms = (syms,)
        expr = sp.S.Zero
        for monom, coeff in self._p.terms():
            c = sp.Rational(int(coeff.numer()), int(coeff.denom()))
            term: sp.Expr = c
            for i, exp in enumerate(monom):
                if exp:
                    term = term * syms[i] ** exp
            expr = expr + term
        return expr

    def __repr__(self) -> str:
        return f"_FlintPoly({self._p!r})"


_FLINT_RING_CACHE: dict[tuple[str, ...], _FlintRing] = {}


def _flint_make_ring(names: Sequence[str]) -> _FlintRing:
    if not _FLINT_AVAILABLE:
        raise RuntimeError("python-flint is not installed; cannot use flint backend")
    key = tuple(names)
    cached = _FLINT_RING_CACHE.get(key)
    if cached is not None:
        return cached
    ctx = fmpq_mpoly_ctx.get(key, "lex")
    ring = _FlintRing(ctx, key)
    _FLINT_RING_CACHE[key] = ring
    return ring


def _flint_lift(poly: _FlintPoly, target: _FlintRing) -> _FlintPoly:
    src = poly._ring
    if src is target:
        return poly
    src_names = src._names
    tgt_names = target._names
    if src_names == tgt_names:
        return _FlintPoly(target._ctx.from_dict(poly._p.to_dict()), target)
    missing = set(src_names) - set(tgt_names)
    if missing:
        raise ValueError(
            f"cannot lift polynomial: target ring is missing generators "
            f"{sorted(missing)!r}"
        )
    n_tgt = len(tgt_names)
    tgt_index = {name: i for i, name in enumerate(tgt_names)}
    pos_map = [tgt_index[name] for name in src_names]
    new_dict: dict[tuple[int, ...], Any] = {}
    for monom, coeff in poly._p.to_dict().items():
        new_monom = [0] * n_tgt
        for src_pos, exp in enumerate(monom):
            if exp:
                new_monom[pos_map[src_pos]] = exp
        new_dict[tuple(new_monom)] = coeff
    return _FlintPoly(target._ctx.from_dict(new_dict), target)


# ======================================================================
# Backend dispatch
# ======================================================================

if TYPE_CHECKING:
    Poly = _SympyPoly | _FlintPoly
    Ring = _SympyRing | _FlintRing

    #: Build the ring over ``names`` in the active backend, caching it by name tuple.
    make_ring: Callable[[Sequence[str]], Ring]
    #: Re-express a polynomial in a target ring that contains all of its generators.
    lift: Callable[[Poly, Ring], Poly]
else:
    if BACKEND_NAME is None:
        # Unset → prefer sparse flint when available, else sympy.
        BACKEND_NAME = "flint" if _FLINT_AVAILABLE else "sympy"
    if BACKEND_NAME == "flint":
        if not _FLINT_AVAILABLE:
            raise RuntimeError(
                "CHARTLIB_POLY_BACKEND=flint but python-flint is not installed"
            )
        make_ring = _flint_make_ring
        lift = _flint_lift
        Ring = _FlintRing
        Poly = _FlintPoly
    elif BACKEND_NAME == "sympy":
        make_ring = _sympy_make_ring
        lift = _sympy_lift
        Ring = _SympyRing
        Poly = _SympyPoly
    elif BACKEND_NAME == "native_py":
        from . import poly_native_py as _native_py
        from .poly_native_value import get_value_handle

        _native_py.configure(get_value_handle(COEFF_NAME))
        make_ring = _native_py.make_ring
        lift = _native_py.lift
        Ring = _native_py._NativePyRing
        Poly = _native_py._NativePyPoly
    elif BACKEND_NAME == "native_cpp":
        if COEFF_NAME == "double":
            try:
                from . import _poly_native_cpp_double as _native_cpp
            except ImportError as e:
                raise RuntimeError(
                    "CHARTLIB_POLY_BACKEND=native_cpp requires the Cython "
                    "extension to be built.  Run `make cython` (or "
                    "`python setup_cython.py build_ext --inplace`) first."
                ) from e
            Ring = _native_cpp.CppRing_double
            Poly = _native_cpp.CppPoly_double
        elif COEFF_NAME in ("mpf", "quad"):
            try:
                from . import _poly_native_cpp_mpf as _native_cpp
            except ImportError as e:
                raise RuntimeError(
                    "CHARTLIB_POLY_BACKEND=native_cpp requires the Cython "
                    "extension to be built.  Run `make cython` (or "
                    "`python setup_cython.py build_ext --inplace`) first."
                ) from e
            from .poly_native_value import get_value_handle
            _native_cpp.configure(get_value_handle(COEFF_NAME))
            Ring = _native_cpp.CppRing_mpf
            Poly = _native_cpp.CppPoly_mpf
        else:
            raise ValueError(
                f"unknown CHARTLIB_POLY_COEFF {COEFF_NAME!r} for native_cpp; "
                f"valid choices: 'double', 'mpf', 'quad'"
            )
        make_ring = _native_cpp.make_ring
        lift = _native_cpp.lift
    else:
        raise ValueError(
            f"CHARTLIB_POLY_BACKEND must be 'sympy', 'flint', 'native_py', "
            f"or 'native_cpp'; got {BACKEND_NAME!r}"
        )


# Tuple usable in ``isinstance(value, PolyTypes)`` calls where the
# backend is not statically known (e.g. shared boundary helpers that
# accept inputs from either backend).  Excludes the unselected backend
# only when python-flint is missing — otherwise both wrapper types are
# importable and a defensive isinstance check covers the cross-backend
# leakage cases (none in current code, but cheap insurance).
_extra_poly_types: list = []
_extra_ring_types: list = []
if BACKEND_NAME == "native_py":
    from . import poly_native_py as _native_py_module
    _extra_poly_types.append(_native_py_module._NativePyPoly)
    _extra_ring_types.append(_native_py_module._NativePyRing)
if BACKEND_NAME == "native_cpp":
    _extra_poly_types.append(Poly)
    _extra_ring_types.append(Ring)

if _FLINT_AVAILABLE:
    PolyTypes: tuple = (_SympyPoly, _FlintPoly, *_extra_poly_types)
    RingTypes: tuple = (_SympyRing, _FlintRing, *_extra_ring_types)
else:
    PolyTypes = (_SympyPoly, *_extra_poly_types)
    RingTypes = (_SympyRing, *_extra_ring_types)


def clear_ring_caches() -> dict[str, int]:
    """Drop all cached polynomial rings, returning the pre-clear sizes.

    The ring caches (:data:`_FLINT_RING_CACHE`, :data:`_SYMPY_RING_CACHE`)
    are keyed by generator-name tuple and provide heavy within-build
    reuse (a single symbolic sample re-requests the same union rings
    hundreds of times).  But the key space is unbounded across distinct
    builds: every new cell / degree / basis / derivative-order combination
    introduces fresh generator names, so a long-lived process that runs
    many distinct builds accumulates one cached ring (plus its pinned
    flint context and cached one/zero/gens wrappers) per distinct key.

    This function releases that accumulation at a build boundary.  Cached
    rings are referenced by identity only inside a single build, so
    clearing between builds is safe: any still-live polynomial keeps its
    own ring alive via its back-reference, and the next build simply
    re-populates the cache.  Returns ``{"flint": n, "sympy": m,
    "flint_ctx": k}`` giving the number of entries dropped (useful for
    instrumentation).

    ``flint_ctx`` covers python-flint's *own* global context-intern cache
    (:attr:`fmpq_mpoly_ctx._ctx_cache`).  ``fmpq_mpoly_ctx.get`` interns
    every context it builds by ``(names, ordering)`` and never evicts.
    Because :data:`_FLINT_RING_CACHE`'s key space is unbounded across
    builds (every fresh generator name spawns a fresh context), that
    intern table grows without bound, and each cached context pins its
    polynomial arena.  Dropping only :data:`_FLINT_RING_CACHE`
    frees nothing: flint's ``_ctx_cache`` keeps every context (hence every
    ring and its polys) alive on its own.  We therefore clear it here too.
    Live polynomials keep their context alive directly, so this stays safe
    at a build boundary.
    """
    sizes = {"flint": len(_FLINT_RING_CACHE), "sympy": len(_SYMPY_RING_CACHE)}
    _FLINT_RING_CACHE.clear()
    _SYMPY_RING_CACHE.clear()
    # Also drop python-flint's internal context-intern cache, the true
    # unbounded retainer (see docstring).  Guarded: ``_ctx_cache`` is a
    # python-flint implementation detail and may be absent / renamed in
    # other versions, in which case there is nothing extra to clear.
    flint_ctx_dropped = 0
    if _FLINT_AVAILABLE and fmpq_mpoly_ctx is not None:
        ctx_cache = getattr(fmpq_mpoly_ctx, "_ctx_cache", None)
        if isinstance(ctx_cache, dict):
            flint_ctx_dropped = len(ctx_cache)
            ctx_cache.clear()
    sizes["flint_ctx"] = flint_ctx_dropped
    return sizes


def coeff_zero(poly: _SympyPoly | _FlintPoly) -> bool:
    """Structural-zero test for a backend Poly."""
    return poly.is_zero


def coeff_to_float(coeff: CoeffLike) -> float:
    """Narrow a backend coefficient to a Python float."""
    if isinstance(coeff, float):
        return coeff
    if isinstance(coeff, int):
        return float(coeff)
    return float(coeff)
