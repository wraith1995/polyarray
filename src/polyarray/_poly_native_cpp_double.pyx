# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""Cython native polynomial backend — double instantiation.

Storage: two parallel numpy arrays per polynomial — ``_exps`` of shape
``(T, G)`` int32 holding term exponents, ``_coeffs`` of shape ``(T,)``
float64.  Terms are kept sorted by exponent tuple (Python tuple
comparison), which gives stable hashing and lets ``__add__`` /
``__eq__`` run as a single linear pass.

The hot paths (multiplication, lift) drop to tight C-level loops; the
boundary methods (``cancel``, ``as_expr``, ``terms``/``monoms``) stay
in Python-readable Cython for clarity.

Slice B1 only ships ``double``; ``mpf`` and ``quad`` are separate
``.pyx`` modules.
"""
from __future__ import annotations

import numpy as np
cimport numpy as cnp
cimport cython

import sympy as sp


cnp.import_array()


# Module-level ring + pos_map caches.
_RING_CACHE: dict = {}
_POS_MAP_CACHE: dict = {}


cdef class CppRing_double:
    """A multivariate polynomial ring with float64 coefficients."""

    cdef readonly tuple _names
    cdef readonly int _n_gens
    cdef readonly dict _index
    cdef CppPoly_double _zero
    cdef CppPoly_double _one
    cdef readonly tuple _gens

    def __cinit__(self, tuple names):
        self._names = names
        self._n_gens = len(names)
        self._index = {n: i for i, n in enumerate(names)}

        # zero
        cdef cnp.ndarray empty_e = np.zeros((0, self._n_gens), dtype=np.int32)
        cdef cnp.ndarray empty_c = np.zeros(0, dtype=np.float64)
        self._zero = _make_poly(self, empty_e, empty_c)

        # one — single term, all-zero exponents, coeff 1.0
        cdef cnp.ndarray one_e = np.zeros((1, self._n_gens), dtype=np.int32)
        cdef cnp.ndarray one_c = np.ones(1, dtype=np.float64)
        self._one = _make_poly(self, one_e, one_c)

        # gens
        cdef list gens = []
        cdef cnp.ndarray gen_e
        cdef cnp.ndarray gen_c
        cdef int i
        for i in range(self._n_gens):
            gen_e = np.zeros((1, self._n_gens), dtype=np.int32)
            gen_e[0, i] = 1
            gen_c = np.ones(1, dtype=np.float64)
            gens.append(_make_poly(self, gen_e, gen_c))
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
        cdef double c = float(value)
        if c == 0.0:
            return self._zero
        cdef cnp.ndarray e = np.zeros((1, self._n_gens), dtype=np.int32)
        cdef cnp.ndarray cf = np.array([c], dtype=np.float64)
        return _make_poly(self, e, cf)

    @cython.wraparound(True)   # this method indexes Python lists with [-1]; the file-level
    #                            wraparound=False would read index -1 literally (OOB → crash).
    def from_dict(self, terms):
        cdef double cf
        cdef double new_c
        cdef int T, G, i, k
        cdef cnp.ndarray e, c_arr
        if not terms:
            return self._zero
        items = []
        for m, c in terms.items():
            cf = float(c)
            if cf == 0.0:
                continue
            items.append((tuple(int(x) for x in m), <object>cf))
        if not items:
            return self._zero
        items.sort(key=lambda mc: mc[0])
        # Coalesce duplicates after sorting.
        out_m = []
        out_c_list = []
        for m, c in items:
            if out_m and out_m[-1] == m:
                new_c = float(out_c_list[-1]) + float(c)
                if new_c == 0.0:
                    out_m.pop()
                    out_c_list.pop()
                else:
                    out_c_list[-1] = new_c
            else:
                out_m.append(m)
                out_c_list.append(float(c))
        if not out_m:
            return self._zero
        T = len(out_m)
        G = self._n_gens
        e = np.empty((T, G), dtype=np.int32)
        c_arr = np.empty(T, dtype=np.float64)
        for i in range(T):
            for k in range(G):
                e[i, k] = out_m[i][k]
            c_arr[i] = out_c_list[i]
        return _make_poly(self, e, c_arr)

    def __repr__(self):
        return f"CppRing_double({self._names!r})"


cdef class CppPoly_double:
    """Sparse multivariate polynomial — float64 coefficient instantiation."""

    cdef CppRing_double _ring
    cdef readonly int _n_gens
    cdef readonly int _n_terms
    cdef cnp.ndarray _exps
    cdef cnp.ndarray _coeffs
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
        if self._coeffs[0] != 1.0:
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
        # Lex-leading: terms are stored sorted lex-ascending by monomial
        # tuple, so the lex-max term is the last one.
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
        # Hot path — called per RF on every eval codegen.  Batch-convert
        # exponent and coefficient arrays to Python lists once
        # (``numpy.ndarray.tolist`` is a single C-level routine) rather
        # than doing a per-element ``int(...)`` / ``float(...)``.
        if self._n_terms == 0:
            return []
        exps_list = np.asarray(self._exps).tolist()
        coeffs_list = np.asarray(self._coeffs).tolist()
        return [(tuple(exps_list[i]), coeffs_list[i])
                for i in range(self._n_terms)]

    def monoms(self):
        if self._n_terms == 0:
            return []
        exps_list = np.asarray(self._exps).tolist()
        return [tuple(exps_list[i]) for i in range(self._n_terms)]

    def __eq__(self, other):
        if not isinstance(other, CppPoly_double):
            return NotImplemented
        cdef CppPoly_double o = <CppPoly_double>other
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
        # Canonical form: same one terms() yields.
        cdef list items = self.terms()
        h = hash((id(self._ring), tuple(items)))
        self._hash = h
        self._hash_set = True
        return h

    def __neg__(self):
        return _make_poly(
            self._ring,
            np.asarray(self._exps).copy(),
            -np.asarray(self._coeffs),
        )

    def __add__(self, other):
        if isinstance(other, CppPoly_double):
            return _add_two(self, <CppPoly_double>other)
        if isinstance(other, (int, float)):
            return _add_two(self, _ground_new_local(self._ring, other))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CppPoly_double):
            return _add_scaled(self, <CppPoly_double>other, -1.0)
        if isinstance(other, (int, float)):
            return _add_scaled(self, _ground_new_local(self._ring, other), -1.0)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _add_scaled(_ground_new_local(self._ring, other), self, -1.0)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, CppPoly_double):
            return _mul_two(self, <CppPoly_double>other)
        if isinstance(other, (int, float)):
            return _mul_scalar(self, float(other))
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
        cdef CppPoly_double o
        cdef CppRing_double ring
        cdef int G
        cdef cnp.ndarray num_min, den_monom, strip
        cdef cnp.ndarray new_exps, new_coeffs, new_den_monom
        cdef double inv_c
        cdef CppPoly_double n2, d2, q, r
        if not isinstance(other, CppPoly_double):
            raise TypeError("cancel expects another CppPoly_double")
        o = <CppPoly_double>other
        if self._ring is not o._ring:
            raise ValueError("cancel across different rings")
        if o._n_terms == 0:
            raise ZeroDivisionError("cancel by zero polynomial")

        ring = self._ring
        G = self._n_gens

        # Case A: one-term denominator c · X^d.
        if o._n_terms == 1:
            if self._n_terms == 0:
                return ring._zero, ring._one
            num_min = np.min(np.asarray(self._exps), axis=0)
            den_monom = np.asarray(o._exps)[0]
            strip = np.minimum(num_min, den_monom)
            inv_c = 1.0 / o._coeffs[0]
            new_exps = (np.asarray(self._exps) - strip[None, :]).astype(np.int32)
            new_coeffs = (np.asarray(self._coeffs) * inv_c).astype(np.float64)
            new_den_monom = (den_monom - strip).astype(np.int32)
            n2 = _make_poly(ring, new_exps, new_coeffs)
            if not np.any(new_den_monom):
                return n2, ring._one
            d2 = _make_poly(
                ring,
                new_den_monom.reshape(1, G),
                np.ones(1, dtype=np.float64),
            )
            return n2, d2

        # Case B: deg-≤-1 multi-term denominator — schoolbook division.
        if o.total_degree() <= 1:
            q, r = _div_deg1(self, o)
            if r._n_terms == 0:
                return q, ring._one
            return self, o

        # Case C: heavier — safety net.
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
        return f"CppPoly_double({self.as_expr()})"


# ---------------------------------------------------------------------------
# Construction helpers.
# ---------------------------------------------------------------------------

cdef CppPoly_double _make_poly(CppRing_double ring,
                               cnp.ndarray exps,
                               cnp.ndarray coeffs):
    cdef CppPoly_double p = CppPoly_double.__new__(CppPoly_double)
    p._ring = ring
    p._n_gens = ring._n_gens
    p._n_terms = coeffs.shape[0]
    p._exps = exps
    p._coeffs = coeffs
    p._hash = 0
    p._hash_set = False
    return p


cdef CppPoly_double _ground_new_local(CppRing_double ring, value):
    cdef double c = float(value)
    if c == 0.0:
        return ring._zero
    cdef cnp.ndarray e = np.zeros((1, ring._n_gens), dtype=np.int32)
    cdef cnp.ndarray cf = np.array([c], dtype=np.float64)
    return _make_poly(ring, e, cf)


# ---------------------------------------------------------------------------
# Arithmetic — merge / scale / multiply.
# ---------------------------------------------------------------------------

cdef CppPoly_double _add_scaled(CppPoly_double a, CppPoly_double b, double scale):
    """Return ``a + scale * b`` via sorted-merge.

    Both operand exponent arrays are stored lex-sorted ascending; the
    output preserves that invariant.
    """
    if a._ring is not b._ring:
        raise ValueError("add across different rings")
    cdef int Ta = a._n_terms
    cdef int Tb = b._n_terms
    cdef int G = a._n_gens
    cdef int Tmax = Ta + Tb
    if Tmax == 0:
        return a._ring._zero
    cdef cnp.ndarray out_e = np.empty((Tmax, G), dtype=np.int32)
    cdef cnp.ndarray out_c = np.empty(Tmax, dtype=np.float64)
    cdef int[:, ::1] ae = a._exps
    cdef int[:, ::1] be = b._exps
    cdef double[::1] ac = a._coeffs
    cdef double[::1] bc = b._coeffs
    cdef int[:, ::1] oe = out_e
    cdef double[::1] oc = out_c
    cdef int i = 0, j = 0, o = 0, k
    cdef int cmp
    cdef double s
    while i < Ta and j < Tb:
        # Lex compare exp rows.
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
            oc[o] = ac[i]
            o += 1
            i += 1
        elif cmp > 0:
            for k in range(G):
                oe[o, k] = be[j, k]
            oc[o] = scale * bc[j]
            o += 1
            j += 1
        else:
            s = ac[i] + scale * bc[j]
            if s != 0.0:
                for k in range(G):
                    oe[o, k] = ae[i, k]
                oc[o] = s
                o += 1
            i += 1
            j += 1
    while i < Ta:
        for k in range(G):
            oe[o, k] = ae[i, k]
        oc[o] = ac[i]
        o += 1
        i += 1
    while j < Tb:
        for k in range(G):
            oe[o, k] = be[j, k]
        oc[o] = scale * bc[j]
        o += 1
        j += 1
    if o == 0:
        return a._ring._zero
    return _make_poly(a._ring, out_e[0:o].copy(), out_c[0:o].copy())


cdef CppPoly_double _add_two(CppPoly_double a, CppPoly_double b):
    return _add_scaled(a, b, 1.0)


cdef CppPoly_double _mul_scalar(CppPoly_double a, double s):
    if s == 0.0:
        return a._ring._zero
    cdef cnp.ndarray new_c = np.asarray(a._coeffs) * s
    return _make_poly(a._ring, np.asarray(a._exps).copy(), new_c)


cdef CppPoly_double _mul_two(CppPoly_double a, CppPoly_double b):
    if a._ring is not b._ring:
        raise ValueError("mul across different rings")
    if a._n_terms == 0 or b._n_terms == 0:
        return a._ring._zero
    cdef int G = a._n_gens
    cdef int Ta = a._n_terms
    cdef int Tb = b._n_terms

    # Structural-one short-circuit: a single all-zero-monom term with
    # coeff 1.0 means ``a`` is ring.one.
    cdef int k
    cdef bint is_one_a = (Ta == 1 and a._coeffs[0] == 1.0)
    if is_one_a:
        for k in range(G):
            if a._exps[0, k] != 0:
                is_one_a = False
                break
    if is_one_a:
        return b
    cdef bint is_one_b = (Tb == 1 and b._coeffs[0] == 1.0)
    if is_one_b:
        for k in range(G):
            if b._exps[0, k] != 0:
                is_one_b = False
                break
    if is_one_b:
        return a

    cdef int Tmax = Ta * Tb
    cdef cnp.ndarray prod_e = np.empty((Tmax, G), dtype=np.int32)
    cdef cnp.ndarray prod_c = np.empty(Tmax, dtype=np.float64)
    cdef int[:, ::1] ae = a._exps
    cdef int[:, ::1] be = b._exps
    cdef double[::1] ac = a._coeffs
    cdef double[::1] bc = b._coeffs
    cdef int[:, ::1] pe = prod_e
    cdef double[::1] pc = prod_c
    cdef int i, j, o = 0
    for i in range(Ta):
        for j in range(Tb):
            for k in range(G):
                pe[o, k] = ae[i, k] + be[j, k]
            pc[o] = ac[i] * bc[j]
            o += 1

    # Coalesce duplicate monomials: sort by exponent row, then merge.
    # Python-side sort over a structured view — clean, slower than a
    # hand-rolled radix but fine for B1 correctness.
    idx = np.lexsort(prod_e.T[::-1])
    sorted_e = prod_e[idx]
    sorted_c = prod_c[idx]

    cdef cnp.ndarray out_e = np.empty((Tmax, G), dtype=np.int32)
    cdef cnp.ndarray out_c = np.empty(Tmax, dtype=np.float64)
    cdef int[:, ::1] se = sorted_e
    cdef double[::1] sc = sorted_c
    cdef int[:, ::1] oe = out_e
    cdef double[::1] oc = out_c
    cdef int w = -1
    cdef bint same
    for i in range(Tmax):
        same = False
        if w >= 0:
            same = True
            for k in range(G):
                if oe[w, k] != se[i, k]:
                    same = False
                    break
        if same:
            oc[w] = oc[w] + sc[i]
        else:
            w += 1
            for k in range(G):
                oe[w, k] = se[i, k]
            oc[w] = sc[i]
    # Drop trailing structural zeros.
    cdef int final = 0
    for i in range(w + 1):
        if oc[i] != 0.0:
            if final != i:
                for k in range(G):
                    oe[final, k] = oe[i, k]
                oc[final] = oc[i]
            final += 1
    if final == 0:
        return a._ring._zero
    return _make_poly(a._ring, out_e[0:final].copy(), out_c[0:final].copy())


# ---------------------------------------------------------------------------
# Deg-≤-1 schoolbook division (used by ``cancel`` case B).
# ---------------------------------------------------------------------------

cdef tuple _div_deg1(CppPoly_double num, CppPoly_double den):
    """Schoolbook polynomial long division specialised to a total-
    degree-≤-1 divisor.  Mirrors ``poly_native_py._native_py_div_deg1``;
    see there for the algorithm rationale.
    """
    cdef CppRing_double ring = num._ring
    cdef int G = num._n_gens
    cdef int pivot = -1
    cdef double pivot_coeff = 0.0
    cdef int i, k, s
    cdef double inv_c, inv_pivot
    cdef cnp.ndarray q_e, q_c
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
        # Constant den: scale each numerator term.
        # (This shouldn't be reached via the public cancel surface,
        # which routes one-term constants through case A; defensive.)
        inv_c = 1.0 / den._coeffs[0]
        q_e = np.asarray(num._exps).copy()
        q_c = np.asarray(num._coeffs) * inv_c
        return _make_poly(ring, q_e, q_c), ring._zero

    inv_pivot = 1.0 / pivot_coeff
    # Mutable "work" as a dict from monomial tuple -> coefficient.
    work = {}
    for i in range(num._n_terms):
        m = tuple(int(num._exps[i, k]) for k in range(G))
        work[m] = float(num._coeffs[i])
    den_terms = []
    for i in range(den._n_terms):
        m = tuple(int(den._exps[i, k]) for k in range(G))
        den_terms.append((m, float(den._coeffs[i])))

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
        # work -= q_coeff · x^q_monom · den
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
                if new_c == 0.0:
                    del work[shifted]
                else:
                    work[shifted] = new_c
        prev_q = q_terms.get(q_monom)
        if prev_q is None:
            q_terms[q_monom] = q_coeff
        else:
            combined = prev_q + q_coeff
            if combined == 0.0:
                del q_terms[q_monom]
            else:
                q_terms[q_monom] = combined

    q = _terms_dict_to_poly(ring, q_terms)
    r = _terms_dict_to_poly(ring, work)
    return q, r


cdef CppPoly_double _terms_dict_to_poly(CppRing_double ring, dict terms):
    if not terms:
        return ring._zero
    cdef int G = ring._n_gens
    items = sorted(terms.items(), key=lambda mc: mc[0])
    cdef int T = len(items)
    cdef cnp.ndarray e = np.empty((T, G), dtype=np.int32)
    cdef cnp.ndarray c = np.empty(T, dtype=np.float64)
    cdef int i, k
    for i in range(T):
        for k in range(G):
            e[i, k] = items[i][0][k]
        c[i] = items[i][1]
    return _make_poly(ring, e, c)


# ---------------------------------------------------------------------------
# Module-level entry points used by ``poly_backend.py``.
# ---------------------------------------------------------------------------

def make_ring(names):
    key = tuple(names)
    cached = _RING_CACHE.get(key)
    if cached is not None:
        return cached
    ring = CppRing_double(key)
    _RING_CACHE[key] = ring
    return ring


def _pos_map(CppRing_double src, CppRing_double target):
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


def lift(CppPoly_double poly, CppRing_double target):
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
    # Coefficients carry over unchanged; just copy to a fresh array.
    cdef cnp.ndarray new_c = np.asarray(poly._coeffs).copy()
    return _make_poly(target, new_e, new_c)


def configure(value):
    """No-op for the double-only Cython engine — the coefficient type
    is fixed at compile time.  Present for protocol compatibility with
    :mod:`poly_native_py`.
    """
    del value
