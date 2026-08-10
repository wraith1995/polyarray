"""Pure-Python native polynomial backend.

A dict-of-monomial-tuples sparse multivariate polynomial engine, generic over the
coefficient type via a :class:`ValueHandle`. Selected by
``CHARTLIB_POLY_BACKEND=native_py``, with the coefficient type chosen by
``CHARTLIB_POLY_COEFF``.

It implements the same :class:`Poly` / :class:`Ring` protocol as the sympy and flint
backends, without the per-op Python cost of :class:`sympy.polys.rings.PolyElement`.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import mpmath
import sympy as sp

from .poly_native_value import ValueHandle

#: A coefficient of the active :class:`ValueHandle`: a Python float under ``double``,
#: an ``mpmath.mpf`` under ``mpf`` / ``quad``.
Coeff: TypeAlias = float | mpmath.mpf

#: Anything a value handle coerces to a :data:`Coeff`.
CoeffLike: TypeAlias = int | Coeff

#: An exponent vector, one entry per ring generator.
Monom: TypeAlias = tuple[int, ...]

_VALUE: ValueHandle[Any] | None = None
_RING_CACHE: dict[tuple[str, ...], _NativePyRing] = {}
_POS_MAP_CACHE: dict[tuple[int, int], tuple[int, ...]] = {}


def configure(value: ValueHandle[Any]) -> None:
    """Install the active coefficient :class:`ValueHandle`.

    Called once from :mod:`poly_backend` when it dispatches to this engine. Switching
    handles invalidates the ring and pos-map caches, which are keyed on object identity
    and hold rings built with the previous handle.

    Parameters
    ----------
    value
        The coefficient arithmetic every ring built from now on will use.
    """
    global _VALUE
    if _VALUE is value:
        return
    _RING_CACHE.clear()
    _POS_MAP_CACHE.clear()
    _VALUE = value


def _require_value() -> ValueHandle[Any]:
    """Return the installed value handle, or raise if the backend was never configured."""
    if _VALUE is None:
        raise RuntimeError(
            "native_py backend is not configured; "
            "set CHARTLIB_POLY_BACKEND=native_py and select a coeff via "
            "CHARTLIB_POLY_COEFF before importing chartlib"
        )
    return _VALUE


def _unit_monom(n: int, i: int) -> Monom:
    """Return the degree-1 exponent vector for generator ``i`` in an ``n``-generator ring."""
    m = [0] * n
    m[i] = 1
    return tuple(m)


class _NativePyRing:
    """A multivariate polynomial ring backed by python dicts."""

    __slots__ = ("_names", "_index", "_n", "_value",
                 "_one", "_zero", "_gens")

    def __init__(self, names: tuple[str, ...], value: ValueHandle) -> None:
        self._names = names
        self._index = {n: i for i, n in enumerate(names)}
        self._n = len(names)
        self._value = value
        # Bypass _NativePyPoly.__init__ — we know the terms are clean
        # and we want zero/one allocation to skip the is_exact_zero scan.
        self._zero = _NativePyPoly._make_raw({}, self)
        one_monom = (0,) * self._n
        self._one = _NativePyPoly._make_raw({one_monom: value.one}, self)
        self._gens = tuple(
            _NativePyPoly._make_raw(
                {_unit_monom(self._n, i): value.one}, self,
            )
            for i in range(self._n)
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_gens(self) -> int:
        return self._n

    @property
    def one(self) -> _NativePyPoly:
        return self._one

    @property
    def zero(self) -> _NativePyPoly:
        return self._zero

    @property
    def gens(self) -> tuple[_NativePyPoly, ...]:
        return self._gens

    def gen(self, i: int) -> _NativePyPoly:
        return self._gens[i]

    def ground_new(self, value: CoeffLike) -> _NativePyPoly:
        """Return the constant polynomial ``value``."""
        c = self._value.from_python(value)
        if self._value.is_exact_zero(c):
            return self._zero
        one_monom = (0,) * self._n
        return _NativePyPoly._make_raw({one_monom: c}, self)

    def from_dict(self, terms: dict[Monom, CoeffLike]) -> _NativePyPoly:
        """Build a polynomial from an exponent-vector to coefficient mapping."""
        if not terms:
            return self._zero
        from_python = self._value.from_python
        is_exact_zero = self._value.is_exact_zero
        clean: dict[tuple[int, ...], Any] = {}
        for m, c in terms.items():
            cc = from_python(c)
            if not is_exact_zero(cc):
                clean[m] = cc
        if not clean:
            return self._zero
        return _NativePyPoly._make_raw(clean, self)

    def __repr__(self) -> str:
        return f"_NativePyRing({self._names!r}, {self._value.name})"


class _NativePyPoly:
    """A sparse multivariate polynomial over a :class:`_NativePyRing`."""

    __slots__ = ("_terms", "_ring", "_hash")

    _hash: int | None

    def __init__(self, terms: dict[tuple[int, ...], Any],
                 ring: _NativePyRing) -> None:
        is_exact_zero = ring._value.is_exact_zero
        self._terms = {m: c for m, c in terms.items()
                       if not is_exact_zero(c)}
        self._ring = ring
        self._hash = None

    @classmethod
    def _make_raw(cls, terms: dict[Monom, Coeff],
                  ring: _NativePyRing) -> _NativePyPoly:
        """Build a polynomial from terms already known to carry no structural zeros.

        Skips the filter :meth:`from_dict` applies; callers must guarantee that no
        coefficient in ``terms`` satisfies the handle's ``is_exact_zero``.
        """
        p = cls.__new__(cls)
        p._terms = terms
        p._ring = ring
        p._hash = None
        return p

    @property
    def ring(self) -> _NativePyRing:
        return self._ring

    @property
    def is_zero(self) -> bool:
        return not self._terms

    @property
    def is_one(self) -> bool:
        if len(self._terms) != 1:
            return False
        n = self._ring._n
        one_monom = (0,) * n
        coeff = self._terms.get(one_monom)
        if coeff is None:
            return False
        return self._ring._value.is_one(coeff)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _NativePyPoly):
            if self._ring is not other._ring:
                return NotImplemented
            return self._terms == other._terms
        return NotImplemented

    def __hash__(self) -> int:
        h = self._hash
        if h is None:
            h = hash((id(self._ring), frozenset(self._terms.items())))
            self._hash = h
        return h

    def __neg__(self) -> _NativePyPoly:
        neg = self._ring._value.neg
        # All entries were structurally non-zero; negation preserves that.
        return _NativePyPoly._make_raw(
            {m: neg(c) for m, c in self._terms.items()}, self._ring,
        )

    def _coerce_scalar(self, other: CoeffLike) -> _NativePyPoly:
        return self._ring.ground_new(other)

    def __add__(self, other: _NativePyPoly | CoeffLike) -> _NativePyPoly:
        if isinstance(other, _NativePyPoly):
            if self._ring is not other._ring:
                return NotImplemented
            v = self._ring._value
            v_add = v.add
            is_exact_zero = v.is_exact_zero
            out = dict(self._terms)
            for m, c in other._terms.items():
                prev = out.get(m)
                if prev is None:
                    out[m] = c
                else:
                    new_c = v_add(prev, c)
                    if is_exact_zero(new_c):
                        del out[m]
                    else:
                        out[m] = new_c
            return _NativePyPoly._make_raw(out, self._ring)
        if isinstance(other, (int, float)):
            return self.__add__(self._coerce_scalar(other))
        return NotImplemented

    def __radd__(self, other: _NativePyPoly | CoeffLike) -> _NativePyPoly:
        return self.__add__(other)

    def __sub__(self, other: _NativePyPoly | CoeffLike) -> _NativePyPoly:
        if isinstance(other, _NativePyPoly):
            if self._ring is not other._ring:
                return NotImplemented
            v = self._ring._value
            v_sub = v.sub
            v_neg = v.neg
            is_exact_zero = v.is_exact_zero
            out = dict(self._terms)
            for m, c in other._terms.items():
                prev = out.get(m)
                if prev is None:
                    out[m] = v_neg(c)
                else:
                    new_c = v_sub(prev, c)
                    if is_exact_zero(new_c):
                        del out[m]
                    else:
                        out[m] = new_c
            return _NativePyPoly._make_raw(out, self._ring)
        if isinstance(other, (int, float)):
            return self.__sub__(self._coerce_scalar(other))
        return NotImplemented

    def __rsub__(self, other: _NativePyPoly | CoeffLike) -> _NativePyPoly:
        if isinstance(other, (int, float)):
            return self._coerce_scalar(other).__sub__(self)
        return NotImplemented

    def __mul__(self, other: _NativePyPoly | CoeffLike) -> _NativePyPoly:
        if isinstance(other, _NativePyPoly):
            if self._ring is not other._ring:
                return NotImplemented
            if not self._terms or not other._terms:
                return self._ring._zero
            v = self._ring._value
            v_mul, v_add, v_is_one = v.mul, v.add, v.is_one
            is_exact_zero = v.is_exact_zero
            n = self._ring._n
            one_monom = (0,) * n
            # Structural-one short-circuit: a ground 1 atom should not
            # pay for a full term-by-term mul.
            if len(self._terms) == 1:
                ((m_only, c_only),) = self._terms.items()
                if m_only == one_monom and v_is_one(c_only):
                    return other
            if len(other._terms) == 1:
                ((m_only, c_only),) = other._terms.items()
                if m_only == one_monom and v_is_one(c_only):
                    return self
            out: dict[Monom, Coeff] = {}
            for m_a, c_a in self._terms.items():
                for m_b, c_b in other._terms.items():
                    m = tuple(a + b for a, b in zip(m_a, m_b))
                    c = v_mul(c_a, c_b)
                    prev = out.get(m)
                    if prev is None:
                        out[m] = c
                    else:
                        out[m] = v_add(prev, c)
            # Strip any structural zeros introduced by cancellation
            # between cross-terms.
            out = {m: c for m, c in out.items() if not is_exact_zero(c)}
            return _NativePyPoly._make_raw(out, self._ring)
        if isinstance(other, (int, float)):
            return self.__mul__(self._coerce_scalar(other))
        return NotImplemented

    def __rmul__(self, other: _NativePyPoly | CoeffLike) -> _NativePyPoly:
        return self.__mul__(other)

    def __pow__(self, n: int) -> _NativePyPoly:
        n = int(n)
        if n < 0:
            raise ValueError("negative power not supported")
        if n == 0:
            return self._ring._one
        if n == 1:
            return self
        result = self._ring._one
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def total_degree(self) -> int:
        if not self._terms:
            return 0
        return max(sum(m) for m in self._terms)

    def n_terms(self) -> int:
        return len(self._terms)

    def terms(self) -> list[tuple[tuple[int, ...], Any]]:
        return list(self._terms.items())

    def monoms(self) -> list[tuple[int, ...]]:
        return list(self._terms)

    @property
    def LC(self) -> float:
        # Lex-leading coefficient — matches sympy's default 'lex' order.
        if not self._terms:
            return 0.0
        leading = max(self._terms)
        return self._ring._value.to_float(self._terms[leading])

    def cancel(self, other: _NativePyPoly,
               ) -> tuple[_NativePyPoly, _NativePyPoly]:
        if not isinstance(other, _NativePyPoly):
            raise TypeError("cancel expects another _NativePyPoly")
        if self._ring is not other._ring:
            raise ValueError("cancel across different rings")
        ring = self._ring
        v = ring._value
        if not other._terms:
            raise ZeroDivisionError("cancel by zero polynomial")

        # Case A: one-term denominator c · X^d.
        if len(other._terms) == 1:
            ((den_monom, den_coeff),) = other._terms.items()
            if not self._terms:
                return ring._zero, ring._one
            n = ring._n
            num_min = [min(m[i] for m in self._terms) for i in range(n)]
            strip = tuple(min(num_min[i], den_monom[i]) for i in range(n))
            inv_c = v.inv(den_coeff)
            new_num_terms = {
                tuple(m[i] - strip[i] for i in range(n)): v.mul(c, inv_c)
                for m, c in self._terms.items()
            }
            new_den_monom = tuple(den_monom[i] - strip[i] for i in range(n))
            if not any(new_den_monom):
                # Denominator collapsed to 1.
                return (
                    _NativePyPoly._make_raw(new_num_terms, ring),
                    ring._one,
                )
            return (
                _NativePyPoly._make_raw(new_num_terms, ring),
                _NativePyPoly._make_raw({new_den_monom: v.one}, ring),
            )

        # Case B: multi-term den, total degree ≤ 1.
        if other.total_degree() <= 1:
            q, r = _native_py_div_deg1(self, other)
            if not r._terms:
                return q, ring._one
            return self, other  # coprime up to non-cancellable scalar

        # Case C: heavier — safety net.
        return self, other

    def as_expr(self) -> sp.Expr:
        ring = self._ring
        names = ring._names
        v_to_float = ring._value.to_float
        if not self._terms:
            return sp.Integer(0)
        if not names:
            ((_, c),) = self._terms.items()
            return sp.Float(v_to_float(c))
        syms_raw = sp.symbols(names)
        if isinstance(syms_raw, sp.Symbol):
            syms: tuple[sp.Symbol, ...] = (syms_raw,)
        else:
            syms = tuple(syms_raw)
        acc: sp.Expr = sp.S.Zero
        for monom, coeff in self._terms.items():
            term: sp.Expr = sp.Float(v_to_float(coeff))
            for s, e in zip(syms, monom):
                if e:
                    term = term * s**e
            acc = acc + term
        return acc

    def __repr__(self) -> str:
        return f"_NativePyPoly({self.as_expr()})"


def _native_py_div_deg1(
    num: _NativePyPoly,
    den: _NativePyPoly,
) -> tuple[_NativePyPoly, _NativePyPoly]:
    """Divide by a divisor of total degree at most 1, schoolbook style.

    Any generator whose coefficient in ``den`` is non-zero serves as the pivot. Each
    step cancels the remainder term of highest pivot exponent by subtracting a scaled,
    shifted copy of ``den``; that exponent strictly decreases, so the loop terminates.

    Returns
    -------
    tuple of (_NativePyPoly, _NativePyPoly)
        Quotient and remainder. The remainder is structurally zero exactly when ``den``
        divides ``num``.
    """
    ring = num._ring
    n = ring._n
    v = ring._value
    v_mul, v_sub, v_neg, v_inv, v_add = v.mul, v.sub, v.neg, v.inv, v.add
    is_exact_zero = v.is_exact_zero

    # Locate a pivot generator: a monom of total-degree exactly 1
    # appearing in den.
    pivot: int | None = None
    pivot_coeff: Any = None
    for monom, coeff in den._terms.items():
        if sum(monom) != 1:
            continue
        for i in range(n):
            if monom[i] == 1:
                pivot = i
                pivot_coeff = coeff
                break
        if pivot is not None:
            break

    if pivot is None:
        # Den is a non-zero constant (no generator-bearing term).
        # Divide each numerator term by it.
        ((_, den_c),) = den._terms.items()
        inv_c = v_inv(den_c)
        q_terms: dict[tuple[int, ...], Any] = {}
        for m, c in num._terms.items():
            q_terms[m] = v_mul(c, inv_c)
        return _NativePyPoly._make_raw(q_terms, ring), ring._zero

    inv_pivot = v_inv(pivot_coeff)
    work: dict[tuple[int, ...], Any] = dict(num._terms)
    q_terms = {}

    while True:
        # Pick the term with the greatest pivot-exponent (>0), breaking
        # ties by the full tuple comparison so the loop is deterministic.
        best: tuple[int, ...] | None = None
        for m in work:
            if m[pivot] == 0:
                continue
            if (
                best is None
                or m[pivot] > best[pivot]
                or (m[pivot] == best[pivot] and m > best)
            ):
                best = m
        if best is None:
            break
        coeff = work.pop(best)
        q_coeff = v_mul(coeff, inv_pivot)
        q_monom = tuple(
            best[i] - (1 if i == pivot else 0) for i in range(n)
        )
        # Subtract q_coeff · x^q_monom · den from work.  The pivot-term
        # contribution lands exactly on `best`, which we already popped.
        for d_monom, d_coeff in den._terms.items():
            shifted = tuple(q_monom[i] + d_monom[i] for i in range(n))
            if shifted == best:
                continue
            sub_c = v_mul(q_coeff, d_coeff)
            prev = work.get(shifted)
            if prev is None:
                work[shifted] = v_neg(sub_c)
            else:
                new_c = v_sub(prev, sub_c)
                if is_exact_zero(new_c):
                    del work[shifted]
                else:
                    work[shifted] = new_c
        prev_q = q_terms.get(q_monom)
        if prev_q is None:
            q_terms[q_monom] = q_coeff
        else:
            combined = v_add(prev_q, q_coeff)
            if is_exact_zero(combined):
                del q_terms[q_monom]
            else:
                q_terms[q_monom] = combined

    return (
        _NativePyPoly._make_raw(q_terms, ring),
        _NativePyPoly._make_raw(work, ring),
    )


def make_ring(names: Sequence[str]) -> _NativePyRing:
    """Return the ring over ``names``, building and caching it on first use."""
    value = _require_value()
    key = tuple(names)
    cached = _RING_CACHE.get(key)
    if cached is not None:
        return cached
    ring = _NativePyRing(key, value)
    _RING_CACHE[key] = ring
    return ring


def _pos_map(src: _NativePyRing, target: _NativePyRing) -> tuple[int, ...]:
    """Map each generator position in ``src`` to its position in ``target``."""
    key = (id(src), id(target))
    pm = _POS_MAP_CACHE.get(key)
    if pm is None:
        if src._value is not target._value:
            raise ValueError("lift across value types is not supported")
        try:
            pm = tuple(target._index[n] for n in src._names)
        except KeyError as e:
            missing = sorted(set(src._names) - set(target._names))
            raise ValueError(
                f"cannot lift polynomial: target ring is missing generators "
                f"{missing!r}"
            ) from e
        _POS_MAP_CACHE[key] = pm
    return pm


def lift(poly: _NativePyPoly, target: _NativePyRing) -> _NativePyPoly:
    """Re-express ``poly`` in ``target``, which must contain every generator of its ring."""
    src = poly._ring
    if src is target:
        return poly
    if not poly._terms:
        return target._zero
    pos_map = _pos_map(src, target)
    n_tgt = target._n
    new_terms: dict[Monom, Coeff] = {}
    for monom, coeff in poly._terms.items():
        new_monom = [0] * n_tgt
        for src_pos, exp in enumerate(monom):
            if exp:
                new_monom[pos_map[src_pos]] = exp
        new_terms[tuple(new_monom)] = coeff
    return _NativePyPoly._make_raw(new_terms, target)
