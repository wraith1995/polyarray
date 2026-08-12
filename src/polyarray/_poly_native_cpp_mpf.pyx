# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython native polynomial backend — mpf instantiation.

Coefficients are stored as a Python list of :class:`mpmath.mpf`
objects.  C arithmetic can't apply to coefficients (mpmath ops dispatch
to Python callbacks), so the Cython win here is limited to:

* Exponent-vector layout (parallel ``int32[T, G]`` ndarray, the same
  storage as the double engine).
* Tight C loops over the exponent dimension for ``lift``, ``terms``,
  ``__eq__``.

The double engine is duplicated rather than parameterised because
Cython fused types don't compose well with ``cdef class`` methods
rationale.
"""
from __future__ import annotations

import numpy as np
cimport numpy as cnp
cimport cython

import mpmath
import sympy as sp


cnp.import_array()


_RING_CACHE: dict = {}
_POS_MAP_CACHE: dict = {}


def _coerce_mpf(x):
    """Coerce a Python scalar to :class:`mpmath.mpf`."""
    if isinstance(x, mpmath.mpf):
        return x
    return mpmath.mpf(x)


def _is_mpf_zero(c) -> bool:
    return c == 0


def _is_mpf_one(c) -> bool:
    return c == 1


cdef class CppRing_mpf:
    """A multivariate polynomial ring with mpmath.mpf coefficients."""

    cdef readonly tuple _names
    cdef readonly int _n_gens
    cdef readonly dict _index
    cdef CppPoly_mpf _zero
    cdef CppPoly_mpf _one
    cdef readonly tuple _gens

    def __cinit__(self, tuple names):
        self._names = names
        self._n_gens = len(names)
        self._index = {n: i for i, n in enumerate(names)}

        cdef cnp.ndarray empty_e = np.zeros((0, self._n_gens), dtype=np.int32)
        self._zero = _make_poly(self, empty_e, [])

        cdef cnp.ndarray one_e = np.zeros((1, self._n_gens), dtype=np.int32)
        self._one = _make_poly(self, one_e, [mpmath.mpf(1)])

        cdef list gens = []
        cdef cnp.ndarray gen_e
        cdef int i
        for i in range(self._n_gens):
            gen_e = np.zeros((1, self._n_gens), dtype=np.int32)
            gen_e[0, i] = 1
            gens.append(_make_poly(self, gen_e, [mpmath.mpf(1)]))
        self._gens = tuple(gens)

    @property
    def names(self):
        return self._names

    @property
    def n_gens(self):
        return self._n_gens

    @property
    def zero(self):
        return self._zero

    @property
    def one(self):
        return self._one

    @property
    def gens(self):
        return self._gens

    def gen(self, int i):
        return self._gens[i]

    def ground_new(self, value):
        c = _coerce_mpf(value)
        if _is_mpf_zero(c):
            return self._zero
        cdef cnp.ndarray e = np.zeros((1, self._n_gens), dtype=np.int32)
        return _make_poly(self, e, [c])

    @cython.wraparound(True)   # this method indexes Python lists with [-1]; the file-level
    #                            wraparound=False would read index -1 literally (OOB → crash).
    def from_dict(self, terms):
        cdef int T, G, i, k
        cdef cnp.ndarray e
        if not terms:
            return self._zero
        items = []
        for m, c in terms.items():
            cc = _coerce_mpf(c)
            if _is_mpf_zero(cc):
                continue
            items.append((tuple(int(x) for x in m), cc))
        if not items:
            return self._zero
        items.sort(key=lambda mc: mc[0])
        out_m = []
        out_c = []
        for m, c in items:
            if out_m and out_m[-1] == m:
                new_c = out_c[-1] + c
                if _is_mpf_zero(new_c):
                    out_m.pop()
                    out_c.pop()
                else:
                    out_c[-1] = new_c
            else:
                out_m.append(m)
                out_c.append(c)
        if not out_m:
            return self._zero
        T = len(out_m)
        G = self._n_gens
        e = np.empty((T, G), dtype=np.int32)
        for i in range(T):
            for k in range(G):
                e[i, k] = out_m[i][k]
        return _make_poly(self, e, list(out_c))

    def __repr__(self):
        return f"CppRing_mpf({self._names!r})"


cdef class CppPoly_mpf:
    """Sparse multivariate polynomial — mpmath.mpf coefficient
    instantiation.  Coefficients live in a Python list; exponents in a
    Cython-typed int32 ndarray.
    """

    cdef CppRing_mpf _ring
    cdef readonly int _n_gens
    cdef readonly int _n_terms
    cdef cnp.ndarray _exps
    cdef list _coeffs
    cdef long _hash
    cdef bint _hash_set

    @property
    def ring(self):
        return self._ring

    @property
    def is_zero(self):
        return self._n_terms == 0

    @property
    def is_one(self):
        if self._n_terms != 1:
            return False
        if not _is_mpf_one(self._coeffs[0]):
            return False
        cdef int k
        for k in range(self._n_gens):
            if self._exps[0, k] != 0:
                return False
        return True

    @property
    def LC(self):
        if self._n_terms == 0:
            return 0.0
        return float(self._coeffs[self._n_terms - 1])

    def n_terms(self):
        return self._n_terms

    def total_degree(self):
        if self._n_terms == 0:
            return 0
        cdef int max_deg = 0
        cdef int i, k, s
        cdef int T = self._n_terms
        cdef int G = self._n_gens
        for i in range(T):
            s = 0
            for k in range(G):
                s += self._exps[i, k]
            if s > max_deg:
                max_deg = s
        return max_deg

    def terms(self):
        if self._n_terms == 0:
            return []
        exps_list = np.asarray(self._exps).tolist()
        return [(tuple(exps_list[i]), self._coeffs[i])
                for i in range(self._n_terms)]

    def monoms(self):
        if self._n_terms == 0:
            return []
        exps_list = np.asarray(self._exps).tolist()
        return [tuple(exps_list[i]) for i in range(self._n_terms)]

    def __eq__(self, other):
        if not isinstance(other, CppPoly_mpf):
            return NotImplemented
        cdef CppPoly_mpf o = <CppPoly_mpf>other
        if self._ring is not o._ring:
            return NotImplemented
        if self._n_terms != o._n_terms:
            return False
        cdef int i, k
        cdef int T = self._n_terms
        cdef int G = self._n_gens
        for i in range(T):
            if self._coeffs[i] != o._coeffs[i]:
                return False
            for k in range(G):
                if self._exps[i, k] != o._exps[i, k]:
                    return False
        return True

    def __hash__(self):
        if self._hash_set:
            return self._hash
        # We cannot hash mpf reliably as part of a frozenset key (mpf
        # equality is value-based but hash isn't stable across precisions
        # in the same way Python float's is).  Hash on the structural
        # form: monomial tuples + per-term float-projected coefficient.
        cdef list items = []
        cdef int i, k
        exps_list = np.asarray(self._exps).tolist()
        for i in range(self._n_terms):
            items.append((tuple(exps_list[i]), float(self._coeffs[i])))
        h = hash((id(self._ring), tuple(items)))
        self._hash = h
        self._hash_set = True
        return h

    def __neg__(self):
        return _make_poly(
            self._ring,
            np.asarray(self._exps).copy(),
            [-c for c in self._coeffs],
        )

    def __add__(self, other):
        if isinstance(other, CppPoly_mpf):
            return _add_scaled(self, <CppPoly_mpf>other, mpmath.mpf(1))
        if isinstance(other, (int, float)):
            return _add_scaled(
                self, _ground_new_local(self._ring, other), mpmath.mpf(1),
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CppPoly_mpf):
            return _add_scaled(self, <CppPoly_mpf>other, mpmath.mpf(-1))
        if isinstance(other, (int, float)):
            return _add_scaled(
                self, _ground_new_local(self._ring, other), mpmath.mpf(-1),
            )
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _add_scaled(
                _ground_new_local(self._ring, other), self, mpmath.mpf(-1),
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, CppPoly_mpf):
            return _mul_two(self, <CppPoly_mpf>other)
        if isinstance(other, (int, float)):
            return _mul_scalar(self, _coerce_mpf(other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent, modulo):
        cdef int n = int(exponent)
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
            if n > 0:
                base = base * base
        return result

    def cancel(self, other):
        cdef CppPoly_mpf o
        cdef CppRing_mpf ring
        cdef int G, k
        cdef cnp.ndarray num_min, den_monom, strip
        cdef cnp.ndarray new_exps, new_den_monom
        cdef CppPoly_mpf n2, d2, q, r
        if not isinstance(other, CppPoly_mpf):
            raise TypeError("cancel expects another CppPoly_mpf")
        o = <CppPoly_mpf>other
        if self._ring is not o._ring:
            raise ValueError("cancel across different rings")
        if o._n_terms == 0:
            raise ZeroDivisionError("cancel by zero polynomial")

        ring = self._ring
        G = self._n_gens

        if o._n_terms == 1:
            if self._n_terms == 0:
                return ring._zero, ring._one
            num_min = np.min(np.asarray(self._exps), axis=0)
            den_monom = np.asarray(o._exps)[0]
            strip = np.minimum(num_min, den_monom)
            inv_c = mpmath.mpf(1) / o._coeffs[0]
            new_exps = (np.asarray(self._exps) - strip[None, :]).astype(np.int32)
            new_coeffs = [c * inv_c for c in self._coeffs]
            new_den_monom = (den_monom - strip).astype(np.int32)
            n2 = _make_poly(ring, new_exps, new_coeffs)
            if not np.any(new_den_monom):
                return n2, ring._one
            d2 = _make_poly(
                ring,
                new_den_monom.reshape(1, G),
                [mpmath.mpf(1)],
            )
            return n2, d2

        if o.total_degree() <= 1:
            q, r = _div_deg1(self, o)
            if r._n_terms == 0:
                return q, ring._one
            return self, o

        return self, o

    def as_expr(self):
        ring = self._ring
        names = ring._names
        cdef int i, k
        cdef int T = self._n_terms
        cdef int G = self._n_gens
        if T == 0:
            return sp.Integer(0)
        if not names:
            return sp.Float(float(self._coeffs[0]))
        syms_raw = sp.symbols(names)
        if isinstance(syms_raw, sp.Symbol):
            syms = (syms_raw,)
        else:
            syms = tuple(syms_raw)
        acc = sp.S.Zero
        for i in range(T):
            term = sp.Float(float(self._coeffs[i]))
            for k in range(G):
                e = int(self._exps[i, k])
                if e:
                    term = term * syms[k] ** e
            acc = acc + term
        return acc

    def __repr__(self):
        return f"CppPoly_mpf({self.as_expr()})"


cdef CppPoly_mpf _make_poly(CppRing_mpf ring,
                            cnp.ndarray exps,
                            list coeffs):
    cdef CppPoly_mpf p = CppPoly_mpf.__new__(CppPoly_mpf)
    p._ring = ring
    p._n_gens = ring._n_gens
    p._n_terms = len(coeffs)
    p._exps = exps
    p._coeffs = coeffs
    p._hash = 0
    p._hash_set = False
    return p


cdef CppPoly_mpf _ground_new_local(CppRing_mpf ring, value):
    c = _coerce_mpf(value)
    if _is_mpf_zero(c):
        return ring._zero
    cdef cnp.ndarray e = np.zeros((1, ring._n_gens), dtype=np.int32)
    return _make_poly(ring, e, [c])


cdef CppPoly_mpf _add_scaled(CppPoly_mpf a, CppPoly_mpf b, scale):
    if a._ring is not b._ring:
        raise ValueError("add across different rings")
    cdef int Ta = a._n_terms
    cdef int Tb = b._n_terms
    cdef int G = a._n_gens
    cdef int Tmax = Ta + Tb
    if Tmax == 0:
        return a._ring._zero
    cdef cnp.ndarray out_e = np.empty((Tmax, G), dtype=np.int32)
    out_c: list = [None] * Tmax
    cdef int[:, ::1] ae = a._exps
    cdef int[:, ::1] be = b._exps
    cdef int[:, ::1] oe = out_e
    cdef int i = 0, j = 0, o = 0, k
    cdef int cmp
    while i < Ta and j < Tb:
        cmp = 0
        for k in range(G):
            if ae[i, k] < be[j, k]:
                cmp = -1
                break
            if ae[i, k] > be[j, k]:
                cmp = 1
                break
        if cmp < 0:
            for k in range(G):
                oe[o, k] = ae[i, k]
            out_c[o] = a._coeffs[i]
            o += 1
            i += 1
        elif cmp > 0:
            for k in range(G):
                oe[o, k] = be[j, k]
            out_c[o] = scale * b._coeffs[j]
            o += 1
            j += 1
        else:
            s = a._coeffs[i] + scale * b._coeffs[j]
            if not _is_mpf_zero(s):
                for k in range(G):
                    oe[o, k] = ae[i, k]
                out_c[o] = s
                o += 1
            i += 1
            j += 1
    while i < Ta:
        for k in range(G):
            oe[o, k] = ae[i, k]
        out_c[o] = a._coeffs[i]
        o += 1
        i += 1
    while j < Tb:
        for k in range(G):
            oe[o, k] = be[j, k]
        out_c[o] = scale * b._coeffs[j]
        o += 1
        j += 1
    if o == 0:
        return a._ring._zero
    return _make_poly(a._ring, out_e[0:o].copy(), out_c[:o])


cdef CppPoly_mpf _mul_scalar(CppPoly_mpf a, s):
    if _is_mpf_zero(s):
        return a._ring._zero
    return _make_poly(
        a._ring,
        np.asarray(a._exps).copy(),
        [c * s for c in a._coeffs],
    )


cdef CppPoly_mpf _mul_two(CppPoly_mpf a, CppPoly_mpf b):
    if a._ring is not b._ring:
        raise ValueError("mul across different rings")
    if a._n_terms == 0 or b._n_terms == 0:
        return a._ring._zero
    cdef int G = a._n_gens
    cdef int Ta = a._n_terms
    cdef int Tb = b._n_terms

    cdef int k
    cdef bint is_one_a = (Ta == 1 and _is_mpf_one(a._coeffs[0]))
    if is_one_a:
        for k in range(G):
            if a._exps[0, k] != 0:
                is_one_a = False
                break
    if is_one_a:
        return b
    cdef bint is_one_b = (Tb == 1 and _is_mpf_one(b._coeffs[0]))
    if is_one_b:
        for k in range(G):
            if b._exps[0, k] != 0:
                is_one_b = False
                break
    if is_one_b:
        return a

    # Dict-based coalescing — for mpf the per-key Python cost dominates
    # anyway, so the dict path is cleanest.
    out_dict: dict = {}
    cdef int i, j
    for i in range(Ta):
        for j in range(Tb):
            monom = tuple(int(a._exps[i, k]) + int(b._exps[j, k])
                          for k in range(G))
            new_c = a._coeffs[i] * b._coeffs[j]
            prev = out_dict.get(monom)
            if prev is None:
                out_dict[monom] = new_c
            else:
                combined = prev + new_c
                if _is_mpf_zero(combined):
                    del out_dict[monom]
                else:
                    out_dict[monom] = combined
    if not out_dict:
        return a._ring._zero
    return _terms_dict_to_poly(a._ring, out_dict)


cdef tuple _div_deg1(CppPoly_mpf num, CppPoly_mpf den):
    """Schoolbook polynomial long division specialised to a total-
    degree-≤-1 divisor.  Same algorithm as the double engine.
    """
    cdef CppRing_mpf ring = num._ring
    cdef int G = num._n_gens
    cdef int pivot = -1
    cdef int i, k, s

    pivot_coeff = None
    for i in range(den._n_terms):
        s = 0
        for k in range(G):
            s += den._exps[i, k]
        if s == 1:
            for k in range(G):
                if den._exps[i, k] == 1:
                    pivot = k
                    pivot_coeff = den._coeffs[i]
                    break
            if pivot >= 0:
                break

    if pivot < 0:
        inv_c = mpmath.mpf(1) / den._coeffs[0]
        q_c = [c * inv_c for c in num._coeffs]
        return (
            _make_poly(ring, np.asarray(num._exps).copy(), q_c),
            ring._zero,
        )

    inv_pivot = mpmath.mpf(1) / pivot_coeff
    work: dict = {}
    for i in range(num._n_terms):
        m = tuple(int(num._exps[i, k]) for k in range(G))
        work[m] = num._coeffs[i]
    den_terms = []
    for i in range(den._n_terms):
        m = tuple(int(den._exps[i, k]) for k in range(G))
        den_terms.append((m, den._coeffs[i]))

    q_terms: dict = {}
    while True:
        best = None
        for m in work:
            if m[pivot] == 0:
                continue
            if best is None or m[pivot] > best[pivot] or (
                m[pivot] == best[pivot] and m > best
            ):
                best = m
        if best is None:
            break
        coeff = work.pop(best)
        q_coeff = coeff * inv_pivot
        q_monom = tuple(
            best[k] - (1 if k == pivot else 0) for k in range(G)
        )
        for d_monom, d_coeff in den_terms:
            shifted = tuple(
                q_monom[k] + d_monom[k] for k in range(G)
            )
            if shifted == best:
                continue
            sub_c = q_coeff * d_coeff
            prev = work.get(shifted)
            if prev is None:
                work[shifted] = -sub_c
            else:
                new_c = prev - sub_c
                if _is_mpf_zero(new_c):
                    del work[shifted]
                else:
                    work[shifted] = new_c
        prev_q = q_terms.get(q_monom)
        if prev_q is None:
            q_terms[q_monom] = q_coeff
        else:
            combined = prev_q + q_coeff
            if _is_mpf_zero(combined):
                del q_terms[q_monom]
            else:
                q_terms[q_monom] = combined

    q = _terms_dict_to_poly(ring, q_terms)
    r = _terms_dict_to_poly(ring, work)
    return q, r


cdef CppPoly_mpf _terms_dict_to_poly(CppRing_mpf ring, dict terms):
    if not terms:
        return ring._zero
    cdef int G = ring._n_gens
    items = sorted(terms.items(), key=lambda mc: mc[0])
    cdef int T = len(items)
    cdef cnp.ndarray e = np.empty((T, G), dtype=np.int32)
    cdef int i, k
    coeffs_out: list = []
    for i in range(T):
        for k in range(G):
            e[i, k] = items[i][0][k]
        coeffs_out.append(items[i][1])
    return _make_poly(ring, e, coeffs_out)


def make_ring(names):
    key = tuple(names)
    cached = _RING_CACHE.get(key)
    if cached is not None:
        return cached
    ring = CppRing_mpf(key)
    _RING_CACHE[key] = ring
    return ring


def _pos_map(CppRing_mpf src, CppRing_mpf target):
    key = (id(src), id(target))
    pm = _POS_MAP_CACHE.get(key)
    if pm is not None:
        return pm
    try:
        pm_list = [target._index[n] for n in src._names]
    except KeyError as e:
        missing = sorted(set(src._names) - set(target._names))
        raise ValueError(
            f"cannot lift polynomial: target ring is missing generators "
            f"{missing!r}"
        ) from e
    pm = np.asarray(pm_list, dtype=np.int32)
    _POS_MAP_CACHE[key] = pm
    return pm


def lift(CppPoly_mpf poly, CppRing_mpf target):
    src = poly._ring
    if src is target:
        return poly
    if poly._n_terms == 0:
        return target._zero
    cdef cnp.ndarray pm = _pos_map(src, target)
    cdef int G_tgt = target._n_gens
    cdef int T = poly._n_terms
    cdef cnp.ndarray new_e = np.zeros((T, G_tgt), dtype=np.int32)
    cdef int[:, ::1] sre = poly._exps
    cdef int[:, ::1] nre = new_e
    cdef int[::1] pmv = pm
    cdef int G_src = poly._n_gens
    cdef int i, k
    for i in range(T):
        for k in range(G_src):
            nre[i, pmv[k]] = sre[i, k]
    new_c = list(poly._coeffs)
    return _make_poly(target, new_e, new_c)


def configure(value):
    """Switch the mpmath precision context if needed.

    Called once from :mod:`poly_backend` so a ``quad`` request can pin
    ``mp.prec = 113`` before any ring is built.  ``value`` is the
    Python-side :class:`ValueHandle` chosen at the protocol layer; we
    don't store it directly because the Cython engine ignores
    coefficient ops (mpmath handles everything), but we do use it as a
    "what precision are we aiming for" signal.
    """
    name = getattr(value, "name", None)
    if name == "quad":
        if mpmath.mp.prec < 113:
            mpmath.mp.prec = 113
    elif name == "mpf":
        if mpmath.mp.prec < 53:
            mpmath.mp.prec = 53
