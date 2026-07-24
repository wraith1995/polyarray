"""Rational functions over a per-instance polynomial :class:`Ring`.

A :class:`RationalFunction` is a pair ``(num, den)`` of polynomials
sharing the same ring.  The ring abstraction is provided by
:mod:`chartlib._symbolic.poly_backend`, which has two implementations:

* ``sympy`` (default) — :class:`sympy.polys.rings.PolyElement` over
  :data:`sympy.RR` (53-bit mpmath floats).
* ``flint`` — :class:`flint.fmpq_mpoly` over exact rationals.  ~210×
  faster multivariate multiplication and ~30× faster ring construction
  on realistic generator counts.

The backend is selected at import time via
``CHARTLIB_POLY_BACKEND={sympy,flint}`` and is opaque to the rest of
chartlib — every call site here goes through ``Poly`` / ``Ring``
methods that both backends provide.

Cells of the symbolic interpreter's ``SymArray`` are
``RationalFunction | float``: numeric cells are plain Python floats,
generator-bearing cells are ``RationalFunction``s.

Per the plan (``plans/archive/symbolic_interpreter.md``):

* Each instance carries its own ring spanning the generators it
  actually uses.  Arithmetic combining two instances with different
  rings rebuilds a joint ring on demand.  Generator identity is by
  name, so a global :class:`SymbolEnv` (defined in ``ir.py``)
  remains the source of truth for "what does ``v_2_x`` mean."
* Arithmetic does **not** call :func:`sympy.simplify` or
  :func:`sympy.cancel`.  The class accumulates raw ``num/den`` pairs
  during a hot loop; the only simplification is in :meth:`clean`,
  which is reserved for boundary calls.
* On the sympy backend ``num/den`` are over ``sympy.RR`` (mpmath
  floats); on flint they are over ``QQ`` (exact rationals).  Chartlib
  samples numerically and the ring exists to carry generator
  structure, not to recover exact rationals from floats — converting
  floats to exact rationals only where needed.  Structural-zero
  detection (:func:`coeff_zero`, :func:`simple_zero`) on the sympy
  backend only fires on coefficients that are exactly ``0.0``;
  floating-point cancellation that produces ``1e-16`` will miss the
  sparsity short-circuit.  Flint's exact arithmetic doesn't have that
  problem.

Helpers exported alongside the class:

* :func:`coeff_zero` — structural-zero predicate on a polynomial.
* :func:`simple_zero` — structural-zero predicate on a
  :class:`RationalFunction` (numerator only; denominator is non-zero
  by construction).
* :func:`bareiss_det` — fraction-free determinant of a matrix of
  polynomials.
* :func:`schur_inverse` — block-Schur inverse for a ``2 x 2`` block
  matrix, used as the leaf of the structured rational inverse.
"""
from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np
import sympy as sp

from .poly_backend import Poly, PolyTypes, Ring
from .poly_backend import lift as _lift
from .poly_backend import make_ring as _make_ring

# --- eager cancellation (opt-in) -------------------------------------------
# `RationalFunction` accumulates raw num/den in the hot loop (cancel only at a boundary via
# `clean()`), so a long `*`/`+` chain can grow a large shared gcd (a euclidean-jet cell reached
# degree 29/28, 70k terms). Within `eager_cancel(k)`, `*`/`+`/`-` results whose denominator degree
# exceeds `k` are gcd-cancelled immediately, keeping intermediates reduced (the same chain stays ~10
# terms). Off by default → byte-identical; the low-degree hot loop never trips the threshold.
_EAGER_CANCEL: contextvars.ContextVar = contextvars.ContextVar("_pa_eager_cancel", default=0)


@contextlib.contextmanager
def eager_cancel(den_degree_threshold: int = 2):
    """Within this context, cancel RF arithmetic results whose denominator degree exceeds the
    threshold (0 restores the default lazy behavior)."""
    tok = _EAGER_CANCEL.set(int(den_degree_threshold))
    try:
        yield
    finally:
        _EAGER_CANCEL.reset(tok)


def _maybe_cancel(rf: RationalFunction) -> RationalFunction:
    thr = _EAGER_CANCEL.get()
    if thr and isinstance(rf, RationalFunction):
        try:
            if _total_degree(rf._den) > thr:
                return rf.clean()
        except Exception:  # noqa: BLE001 - cancellation is a best-effort optimization
            pass
    return rf

NumericLike = Union[int, float, sp.Rational, sp.Integer]


def _ring_names(ring: Ring) -> tuple[str, ...]:
    """Return the generator names of ``ring`` as a tuple of ``str``.

    Thin shim around :attr:`Ring.names` retained for the public-ish
    import in :mod:`chartlib._symbolic.ir` (``from .rational import
    _ring_names``).
    """
    return ring.names


def _coeff_to_float(coeff: Any) -> float:
    """Universal coefficient-to-float coercion.

    Used by the eval-codegen and partial-substitution paths to lower a
    polynomial's ground element to a plain Python ``float``.  Accepts
    mpf (sympy ``RR``), PythonMPQ / sympy ``Rational`` (sympy ``QQ``
    without python-flint), and ``flint.fmpq`` (sympy ``QQ`` *with*
    python-flint installed — sympy auto-promotes the ground domain in
    that case, so legacy ``QQ`` callers and the new ``RR`` callers both
    converge here).
    """
    if isinstance(coeff, float):
        return coeff
    if isinstance(coeff, int):
        return float(coeff)
    numer = getattr(coeff, "numerator", None)
    denom = getattr(coeff, "denominator", None)
    if numer is not None and denom is not None:
        return int(numer) / int(denom)
    return float(coeff)


def _to_qq(coeff: NumericLike | sp.Expr) -> float:
    """Coerce a Python / sympy numeric to a Python ``float`` for ring embedding.

    Both backends accept ``float`` in :meth:`Ring.ground_new` /
    :meth:`Ring.from_dict` and coerce it internally to their native
    ground type (``mpf`` for sympy ``RR``; ``fmpq`` for flint).  We
    keep the historical name ``_to_qq`` for grep continuity but
    otherwise the return type is just ``float`` now.
    """
    if isinstance(coeff, float):
        return coeff
    if isinstance(coeff, (int, sp.Integer)):
        return float(int(coeff))
    if isinstance(coeff, sp.Rational):
        return float(coeff.p) / float(coeff.q)
    if isinstance(coeff, sp.Expr) and coeff.is_number:
        return float(coeff)
    raise TypeError(f"cannot coerce {coeff!r} ({type(coeff).__name__}) to float")


def _coeff_abs_float(co: Any) -> float:
    """Best-effort ``|coeff|`` → ``float`` across coefficient backends (Python float / int,
    flint ``fmpq``, sympy Rational/mpf). Used by the numeric-magnitude sparsity test."""
    if isinstance(co, float):
        return abs(co)
    try:
        return abs(float(co))
    except (TypeError, ValueError):
        pass
    for num_attr, den_attr in (("p", "q"), ("numer", "denom"), ("numerator", "denominator")):
        n = getattr(co, num_attr, None)
        d = getattr(co, den_attr, None)
        if n is not None and d is not None:
            n = n() if callable(n) else n
            d = d() if callable(d) else d
            return abs(float(int(n)) / float(int(d)))
    return abs(float(str(co)))


def coeff_zero(poly: Poly) -> bool:
    """Return ``True`` iff ``poly`` is the zero polynomial of its ring.

    Structural test only: every monomial coefficient is exactly zero in
    the polynomial's coefficient field.  Used for branch decisions
    inside the rational lane (e.g. choosing pivots) where a numeric
    near-zero would be the wrong answer.
    """
    return bool(poly.is_zero)


# ---------------------------------------------------------------------------
# Ring rebuild on demand
# ---------------------------------------------------------------------------

def _join_rings(a: Ring, b: Ring) -> Ring:
    """Return a joint ring spanning the generators of ``a`` and ``b``."""
    names_a = a.names
    names_b = b.names
    if names_a == names_b:
        return a
    union = sorted(set(names_a) | set(names_b))
    return _make_ring(union)


# ---------------------------------------------------------------------------
# RationalFunction
# ---------------------------------------------------------------------------

class RationalFunction:
    """A rational function ``num / den`` over a polynomial :class:`Ring`.

    ``num`` and ``den`` are :class:`Poly` polynomials in the same
    :class:`Ring`.  ``den`` is *never* the zero polynomial: construction
    raises if the denominator is structurally zero.  The concrete
    backend (sympy or flint) is selected at module import time via
    ``CHARTLIB_POLY_BACKEND``; both implement the same ``Poly`` /
    ``Ring`` protocol so this class is backend-agnostic.

    The class is *not* a frozen dataclass — equality is structural via
    the underlying poly rings (rebuilding to a common ring on the fly)
    and hashing matches that.  Mutation methods are absent; the only
    public operations return new instances.

    Attributes
    ----------
    num : Poly
        Numerator polynomial.
    den : Poly
        Denominator polynomial.  Always non-zero.
    ring : Ring
        The shared ring of ``num`` and ``den``.  Convenience attribute,
        equal to ``num.ring``.
    """

    __slots__ = ("_num", "_den", "_ring", "_compiled")

    def __init__(self, num: Poly, den: Poly | None = None) -> None:
        if den is None:
            den = num.ring.one
        if num.ring is not den.ring:
            raise ValueError(
                "numerator and denominator must share a Ring; "
                f"got {num.ring} and {den.ring}"
            )
        if coeff_zero(den):
            raise ZeroDivisionError(
                "RationalFunction denominator is structurally zero"
            )
        # Canonicalise structural zero: ``0/d == 0/1`` for any nonzero
        # ``d``.  Without this the denominator accumulates dead weight
        # through chains of multiplications that produce a zero
        # numerator (e.g. Faà di Bruno on an affine chart, where
        # ``D²f == 0`` but the contraction's denominator inherits a
        # ``det(J)``-like polynomial).  Observed on
        # ``Lagrange(orthonormal)/tetrahedron/n_derivs=2``: 36 cells
        # carrying ``num=0`` and ``den`` with 10147 terms each — pure
        # eval-time waste since the cell is identically zero.
        if coeff_zero(num):
            den = num.ring.one
        self._num = num
        self._den = den
        self._ring = num.ring
        self._compiled = None  # populated lazily by :meth:`_compile_eval`

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def constant(cls, value: NumericLike, ring: Ring | None = None) -> RationalFunction:
        """Build a constant ``RationalFunction`` with the given value.

        When ``ring`` is omitted, a 0-generator ring is allocated.
        """
        r = ring if ring is not None else _make_ring(())
        c = _to_qq(value)
        return cls(r.ground_new(c), r.one)

    @classmethod
    def generator(cls, ring: Ring, name: str) -> RationalFunction:
        """Return the rational function ``g`` for the named generator of ``ring``."""
        names = ring.names
        try:
            i = names.index(name)
        except ValueError as exc:
            raise ValueError(
                f"ring has no generator named {name!r}; available: {names!r}"
            ) from exc
        return cls(ring.gens[i], ring.one)

    @classmethod
    def atom(cls, name: str) -> RationalFunction:
        """Build the rational function ``x`` for a fresh single-generator ring named ``name``.

        Convenience wrapper around :meth:`generator` that allocates the
        ring for you — matches the per-cell-ring convention used by
        :func:`chartlib._symbolic.ir.allocate_input`.
        """
        return cls.generator(_make_ring([name]), name)

    @classmethod
    def from_poly(
        cls,
        num: Poly,
        den: Poly | None = None,
    ) -> RationalFunction:
        """Build a :class:`RationalFunction` from raw polynomials."""
        return cls(num, den)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num(self) -> Poly:
        return self._num

    @property
    def den(self) -> Poly:
        return self._den

    @property
    def ring(self) -> Ring:
        return self._ring

    @property
    def gens(self) -> tuple[str, ...]:
        """Return the generator names of this rational function's ring."""
        return self._ring.names

    def is_zero(self) -> bool:
        """Structural-zero test on the numerator (denominator is always non-zero)."""
        return coeff_zero(self._num)

    def max_abs_coeff(self) -> float:
        """Largest ``|coefficient|`` of the numerator (``0.0`` if the numerator is zero).

        A *numeric* magnitude for a tolerance-based sparsity test: a cell whose numerator
        coefficients are all negligible (``≤ tol·scale``) is an identically-zero rational function
        up to float roundoff (e.g. chartLib QR / Orthopoly's irrational normalization) — a structural
        zero that exact :meth:`is_zero` misses. Distinct from ``is_zero`` (which is exact)."""
        m = 0.0
        for _, co in self._num.terms():
            c = _coeff_abs_float(co)
            if c > m:
                m = c
        return m

    def is_constant(self) -> bool:
        """True iff both numerator and denominator have total-degree zero."""
        return _total_degree(self._num) == 0 and _total_degree(self._den) == 0

    def to_constant(self) -> sp.Rational:
        """Return the rational value of a constant.  Raises if non-constant.

        Goes through float on the way out, so exact-rational backends
        lose precision here.  Display / debugging only.
        """
        if not self.is_constant():
            raise ValueError("RationalFunction is not constant")
        if coeff_zero(self._num):
            return sp.Rational(0)
        # For a constant poly the only term IS its leading coefficient,
        # so LC works on both backends (mpf for sympy, fmpq for flint).
        return sp.nsimplify(
            _to_python_float(self._num.LC) / _to_python_float(self._den.LC),
            rational=True,
        )

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------

    def _coerce(self, other: Any) -> RationalFunction:
        """Bring ``other`` into a ``RationalFunction`` over the same ring as ``self``."""
        if isinstance(other, RationalFunction):
            if other._ring is self._ring:
                return other
            joint = _join_rings(self._ring, other._ring)
            return RationalFunction(_lift(other._num, joint), _lift(other._den, joint))
        if isinstance(other, (int, float, sp.Integer, sp.Rational)):
            return RationalFunction.constant(other, self._ring)
        if isinstance(other, sp.Expr) and other.is_number:
            return RationalFunction.constant(other, self._ring)
        return NotImplemented

    def _aligned(self, other: RationalFunction) -> tuple[RationalFunction, RationalFunction]:
        """Return ``(self, other)`` rebuilt over a common ring."""
        if self._ring is other._ring:
            return self, other
        joint = _join_rings(self._ring, other._ring)
        a = RationalFunction(_lift(self._num, joint), _lift(self._den, joint))
        b = RationalFunction(_lift(other._num, joint), _lift(other._den, joint))
        return a, b

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __neg__(self) -> RationalFunction:
        return RationalFunction(-self._num, self._den)

    def __pos__(self) -> RationalFunction:
        return self

    def __add__(self, other: Any) -> RationalFunction:
        # Short-circuit on either operand being zero — saves the
        # ``_coerce`` / ``_aligned`` / ring-arithmetic cost in the
        # ``Σ ... + ...`` accumulators that drive ``compose_jets`` and
        # ``np.einsum`` over object dtype.  Same algebraic identity as
        # ``a + 0 = a`` and ``0 + b = b``; no backend dispatch.
        if coeff_zero(self._num):
            # ``0 + other = other``.  Return ``other`` in its *own* ring
            # rather than lifting it into ``join(self, other)``: the lift
            # (``_flint_lift`` rebuilding monomial-by-monomial) is the
            # dominant cost in order-3 covariant accumulators, where most
            # addends are structurally zero (sparse Christoffel / derivative
            # tensors).  Mirrors the ``0 * other`` short-circuit below.
            # Ring mixing is resolved lazily by the next non-zero operation.
            if isinstance(other, RationalFunction):
                return other
            coerced = self._coerce(other)
            if coerced is NotImplemented:
                return NotImplemented
            return coerced
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        if coeff_zero(coerced._num):
            return self
        a, b = self._aligned(coerced)
        if a._den == b._den:
            return _maybe_cancel(RationalFunction(a._num + b._num, a._den))
        return _maybe_cancel(RationalFunction(a._num * b._den + b._num * a._den, a._den * b._den))

    def __radd__(self, other: Any) -> RationalFunction:
        return self.__add__(other)

    def __sub__(self, other: Any) -> RationalFunction:
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        a, b = self._aligned(coerced)
        if a._den == b._den:
            return _maybe_cancel(RationalFunction(a._num - b._num, a._den))
        return _maybe_cancel(RationalFunction(a._num * b._den - b._num * a._den, a._den * b._den))

    def __rsub__(self, other: Any) -> RationalFunction:
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced.__sub__(self)

    def __mul__(self, other: Any) -> RationalFunction:
        # Short-circuit on either operand being zero — same identity
        # as ``a * 0 = 0`` and ``0 * b = 0``; saves the ring
        # multiplication on the carrier polynomials.
        if coeff_zero(self._num):
            return self
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        if coeff_zero(coerced._num):
            return coerced
        a, b = self._aligned(coerced)
        return _maybe_cancel(RationalFunction(a._num * b._num, a._den * b._den))

    def __rmul__(self, other: Any) -> RationalFunction:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> RationalFunction:
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        a, b = self._aligned(coerced)
        if coeff_zero(b._num):
            raise ZeroDivisionError("division by zero RationalFunction")
        return RationalFunction(a._num * b._den, a._den * b._num)

    def __rtruediv__(self, other: Any) -> RationalFunction:
        coerced = self._coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        return coerced.__truediv__(self)

    def __pow__(self, exponent: int) -> RationalFunction:
        if not isinstance(exponent, (int, sp.Integer)):
            raise TypeError(f"RationalFunction exponent must be int; got {type(exponent).__name__}")
        n = int(exponent)
        if n == 0:
            return RationalFunction.constant(1, self._ring)
        if n > 0:
            return RationalFunction(self._num ** n, self._den ** n)
        # Negative exponent: invert and raise to |n|.
        if coeff_zero(self._num):
            raise ZeroDivisionError("RationalFunction zero raised to negative exponent")
        return RationalFunction(self._den ** (-n), self._num ** (-n))

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (int, float, sp.Integer, sp.Rational)):
            other = RationalFunction.constant(other, self._ring)
        if not isinstance(other, RationalFunction):
            return NotImplemented
        a, b = self._aligned(other)
        # a/b == c/d iff a*d == b*c (both denominators non-zero by construction).
        return a._num * b._den == b._num * a._den

    def __hash__(self) -> int:
        # Hash on the canonical-form bytes of num/den after a cancel pass.
        # We do *not* keep canonical form on every instance (per §1.2.2 we run
        # cancellation off by default), so hashing pays the cost only when
        # something actually wants to hash.
        cleaned = self.clean()
        return hash((cleaned._num, cleaned._den))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, bindings: Mapping[str, Any]) -> Any:
        """Evaluate ``self`` against a generator-name → value mapping.

        Returns a Python ``float`` when every generator of the ring is
        bound to a numeric value, and a fresh :class:`RationalFunction`
        over the leftover-generator ring when the mapping is partial.

        ``bindings`` is keyed by generator *name*, not generator
        :class:`PolyElement`.  This is the boundary between "polynomial
        arithmetic" and "actual numeric values"; it does *not* go via
        :func:`sympy.subs`.

        Full-binding case: routes through a per-instance compiled
        Python evaluator (built on first call via :func:`exec`, then
        cached on the RF).  This avoids walking sympy term iterators
        on every invocation — critical for the vmap-style build-once
        path where the same RF is evaluated M times across query
        points.
        """
        names = _ring_names(self._ring)
        if not names:
            return self._eval_constant_to_float()
        # Fast path: every generator bound — go through the compiled
        # Python evaluator.
        if all(n in bindings for n in names):
            for v in bindings.values():
                if isinstance(v, RationalFunction):
                    raise TypeError(
                        "RationalFunction.eval expects numeric bindings; "
                        "use the program runner for symbolic substitution"
                    )
            fn = self._compiled
            if fn is None:
                fn = self._compile_eval(names)
                self._compiled = fn
            return fn(bindings)
        # Partial substitution — sympy's PolyElement.evaluate doesn't do
        # partial bindings (it requires every generator), so fold the
        # bound-gen contributions into each term's coefficient and emit
        # the residue into a fresh ring spanning the leftover names.
        leftover_names = [n for n in names if n not in bindings]
        target = _make_ring(leftover_names)
        new_num = _partial_substitute(self._num, names, bindings, target, leftover_names)
        new_den = _partial_substitute(self._den, names, bindings, target, leftover_names)
        return RationalFunction(new_num, new_den)

    def eval_numeric_fast(self, bindings: Mapping[str, float]) -> float:
        """Fast-path evaluator that trusts the caller.

        Assumes every generator of ``self.ring`` has a numeric entry in
        ``bindings``.  Skips both the per-call ``all(n in bindings)``
        membership scan and the ``for v in bindings.values()`` type
        check that :meth:`eval` performs defensively.  Intended for
        program-runner / SymArray-evaluator hot paths where the
        boundary already guarantees the dict is full and float-valued.

        Falls back to :meth:`eval` (which raises on missing keys) when
        the compiled-fast path does not apply.  Returns a Python
        ``float``.
        """
        names = _ring_names(self._ring)
        if not names:
            return self._eval_constant_to_float()
        fn = self._compiled
        if fn is None:
            fn = self._compile_eval(names)
            self._compiled = fn
        return fn(bindings)

    def eval_numeric_direct(self, bindings: Mapping[str, float]) -> float:
        """Numeric evaluation at a FULL binding by direct term-summation — NO per-RF Python
        codegen / ``compile``.

        Returns the same ``float`` as :meth:`eval_numeric_fast` (same terms in the same order,
        same ``_coeff_to_float`` coercion, so bit-identical), but with O(#terms) arithmetic
        instead of building and ``exec``-ing a reusable Python evaluator. That codegen
        amortizes over MANY evaluations (the vmap build-once path, an RF run at M query
        points) but is pure waste for a FEW: the 3-point structural-mask probe
        (:func:`polyarray.schur._structural_mask`) compiled every cell of a degree-5 ``C`` and
        used it 3× — ~31 s (``_compile_eval``: ``builtins.compile`` + ``_poly_term_strings``)
        of the ~36 s Argyris symbolic-P(T) build, for nothing. Byte-identical ⇒ identical mask."""
        names = _ring_names(self._ring)
        if not names:
            return self._eval_constant_to_float()
        return (_eval_poly_full(self._num, names, bindings)
                / _eval_poly_full(self._den, names, bindings))

    # ------------------------------------------------------------------
    # Symbolic substitution (build-time `compose`)
    # ------------------------------------------------------------------

    def compose(self, name: str, repl: RationalFunction) -> RationalFunction:
        """Substitute generator ``name`` with the rational function ``repl``.

        ``repl`` is a :class:`RationalFunction` over *other* generators.
        The result lives over a ring spanning
        ``(self.gens - {name}) ∪ repl.gens`` and equals ``self`` with every
        occurrence of ``name`` replaced by ``repl`` — computed by **ring
        arithmetic**, a generalization of :func:`_partial_substitute` where
        the bound generator's contribution ``name**exp`` is the RF power
        ``repl**exp`` (RF multiply / add) instead of a float.

        Numerator and denominator are each composed independently, so for
        ``num/den`` the answer is ``compose(num) / compose(den)``.  No
        :func:`sympy.subs` / :func:`sympy.simplify` is used.

        If ``name`` is not one of ``self.gens`` this is a semantic no-op and
        ``self`` is returned unchanged (its ring already excludes ``name``;
        any further ring join is resolved lazily by RF arithmetic).
        """
        src_names = self._ring.names
        if name not in src_names:
            return self
        name_pos = src_names.index(name)
        leftover_names = [n for n in src_names if n != name]
        target_names = tuple(sorted(set(leftover_names) | set(repl._ring.names)))
        target = _make_ring(target_names)
        repl_t = RationalFunction(
            _lift(repl._num, target), _lift(repl._den, target)
        )
        new_num = _compose_poly(self._num, src_names, name_pos, repl_t, target)
        new_den = _compose_poly(self._den, src_names, name_pos, repl_t, target)
        return new_num / new_den

    def compose_multi(
        self, mapping: Mapping[str, RationalFunction]
    ) -> RationalFunction:
        """Substitute several generators at once (see :meth:`compose`).

        Each ``repl`` must be over generators *disjoint* from ``mapping``'s
        keys (the substituted names), so the substitutions are independent
        and order-insensitive.  Names absent from ``self.gens`` are skipped.
        """
        result = self
        for name, repl in mapping.items():
            if name in result._ring.names:
                result = result.compose(name, repl)
        return result

    def _eval_constant_to_float(self) -> float:
        n = self._num if not coeff_zero(self._num) else self._ring.zero
        d = self._den
        # In a 0-generator ring, polynomials are ground constants.
        return _to_python_float(n.LC if not coeff_zero(n) else 0) / _to_python_float(d.LC)

    def _compile_eval(self, names: Sequence[str]):
        """Compile a Python evaluator for ``self`` over the named generators.

        Generates a closure ``fn(bindings) -> float`` that reads each
        generator value from ``bindings`` once, then evaluates the
        numerator and denominator as the term-sum
        ``Σ_i coeff_i · prod_j var_j ** exp_ij`` and returns
        ``num / den``.  ``compile`` + ``exec`` is a one-shot cost; the
        returned closure runs at native Python arithmetic speed without
        sympy's poly-iteration overhead.

        For the trivial shapes that dominate on Stmt-output atoms and
        constant cells (single-monomial numerator over ring-one
        denominator), skip ``compile``/``exec`` entirely and return a
        direct Python closure — measurably cheaper when each RF is
        evaluated only a handful of times (e.g. ~1ms saved per atom on
        the per-test build cost).
        """
        fast = self._maybe_compile_fast(names)
        if fast is not None:
            return fast
        src_lines = ["def __rf_eval(__b):"]
        # Read each generator out of bindings once into a local var.
        var_names = [_safe_local_name(n) for n in names]
        for name, var in zip(names, var_names):
            src_lines.append(f"    {var} = __b[{name!r}]")
        # Emit num/den as a series of ``__num += chunk`` statements
        # rather than one long ``+`` chain.  Long binary chains build a
        # left-associative AST whose compile depth is O(N) and overrun
        # CPython's recursion limit on polynomials with hundreds of
        # terms (Christoffel chains on 3D simplices routinely produce
        # these).  Chunking caps per-statement AST depth while keeping
        # each chunk an inlined ``BINARY_OP`` chain — no function-call
        # overhead per eval.
        src_lines.extend(_emit_poly_assign(self._num, var_names, "__num"))
        src_lines.extend(_emit_poly_assign(self._den, var_names, "__den"))
        src_lines.append("    return __num / __den")
        src = "\n".join(src_lines)
        ns: dict[str, Any] = {}
        exec(compile(src, f"<rf_eval:{id(self)}>", "exec"), ns)
        return ns["__rf_eval"]

    def _maybe_compile_fast(self, names: Sequence[str]):
        """Return a no-compile closure when ``self`` is structurally trivial.

        Trivial shapes (denominator equals ring-one and numerator is
        either constant, a pure atom, or a scaled atom) cover the
        dominant majority of cells produced by :meth:`atom`,
        :class:`Stmt` outputs, and constant DOF entries.  For these the
        ``compile`` + ``exec`` round-trip is pure overhead; returning a
        plain ``lambda`` is ~1ms per RF cheaper at first eval and runs
        at the same speed afterwards.

        Returns ``None`` if the RF is not in a recognised trivial
        shape; the caller falls through to the general codegen path.
        """
        den = self._den
        if den != self._ring.one:
            return None
        terms = self._num.terms()
        if len(terms) == 0:
            return lambda __b: 0.0
        if len(terms) > 1:
            return None
        ((monom, coeff),) = terms
        c = _coeff_to_float(coeff)
        total_exp = sum(monom)
        if total_exp == 0:
            return lambda __b, _c=c: _c
        if total_exp == 1 and max(monom) == 1:
            i = monom.index(1)
            name = names[i]
            # No ``float(...)`` cast: for a scalar binding the value is numerically identical (``float(x)==x``),
            # and dropping it makes the fast path ARRAY-safe — a ``(B,)`` batched binding broadcasts through,
            # which the vectorized (batched-program) evaluator relies on (the general codegen path is already
            # array-safe). ``eval_numeric_fast`` thus returns whatever dtype the binding carries.
            if c == 1.0:
                return lambda __b, _n=name: __b[_n]
            return lambda __b, _n=name, _c=c: _c * __b[_n]
        return None

    # ------------------------------------------------------------------
    # Degree / cleanup
    # ------------------------------------------------------------------

    def compute_degree(self, degrees: Mapping[str, int] | None = None) -> int:
        """Total-degree estimate.

        ``degrees`` maps generator names to assigned degrees (default 1
        per generator).  Returns
        ``max(deg(num), deg(den))`` under the weighted total-degree
        induced by ``degrees``.  This is the bound consulted by
        :class:`SymbolicBudget` for lane choice (§1.2.1).
        """
        if degrees is None:
            return max(_total_degree(self._num), _total_degree(self._den))
        names = self._ring.names
        weights = tuple(int(degrees.get(n, 1)) for n in names)
        return max(
            _weighted_total_degree(self._num, weights),
            _weighted_total_degree(self._den, weights),
        )

    def try_cancel(self, max_den_degree: int = 1) -> RationalFunction:
        """Cancel common factors when the denominator is "cheap enough".

        Runs :meth:`clean` only when the denominator's total degree is
        at most ``max_den_degree`` (default 1).  Above that threshold
        the polynomial gcd needed by :meth:`PolyElement.cancel` can be
        as expensive as the symbolic work that produced ``self``, so we
        return ``self`` unchanged and leave the call site to invoke
        :meth:`clean` deliberately at a boundary.
        """
        if _total_degree(self._den) <= max_den_degree:
            return self.clean()
        return self

    def clean(self, tol: float = 0.0) -> RationalFunction:
        """Cancel common factors of numerator and denominator.

        ``tol`` is reserved for a future numeric-zeroing pass; today it
        is only consulted to short-circuit "no-op" calls.  We never
        call :func:`sympy.simplify` here — the backend's ``cancel``
        (sympy: PolyElement.cancel; flint: gcd-based exact division) is
        sufficient for "drop a common gcd."  Per §1.2.2 this is a
        *boundary* operation and must not be invoked in inner loops.
        """
        del tol  # reserved for future numeric-zeroing
        if coeff_zero(self._num):
            return RationalFunction(self._ring.zero, self._ring.one)
        try:
            # num and den share a Ring (enforced in __init__) so they're
            # the same backend subtype; mypy sees only the union.
            n2, d2 = self._num.cancel(self._den)  # type: ignore[arg-type]
        except (sp.PolynomialError, NotImplementedError):
            return self
        return RationalFunction(n2, d2)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def as_expr(self) -> sp.Expr:
        """Return ``self`` as a sympy ``Expr`` (intended for display only)."""
        if self._den == self._ring.one:
            return self._num.as_expr()
        return self._num.as_expr() / self._den.as_expr()

    def __repr__(self) -> str:
        if self._den == self._ring.one:
            return f"RationalFunction({self._num.as_expr()})"
        return f"RationalFunction({self._num.as_expr()} / {self._den.as_expr()})"

    def __str__(self) -> str:
        return str(self.as_expr())


# ---------------------------------------------------------------------------
# Helpers used by RationalFunction
# ---------------------------------------------------------------------------

def _total_degree(poly: Poly) -> int:
    """Return the total degree of ``poly`` (``-1`` for the zero polynomial)."""
    if coeff_zero(poly):
        return -1
    return max(sum(m) for m in poly.monoms())


def _weighted_total_degree(poly: Poly, weights: tuple[int, ...]) -> int:
    if coeff_zero(poly):
        return -1
    return max(sum(w * e for w, e in zip(weights, m)) for m in poly.monoms())


def _to_python_float(value: Any) -> float:
    """Coerce a sympy / mpq / int / Python rational to a Python float.

    Bareiss elimination produces fraction-free intermediates whose
    numerators and denominators can overflow ``float`` even when their
    quotient is a normal-magnitude double.  We split via :class:`Fraction`
    when present (handles arbitrary-precision integer ratios) and fall
    back to :mod:`mpmath` for very large magnitudes.
    """
    if isinstance(value, float):
        return value
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        num, den = value.numerator, value.denominator
        try:
            return float(num) / float(den)
        except OverflowError:
            # Trim leading bits so the ratio survives float conversion.
            try:
                num_i = int(num)
                den_i = int(den)
                shift = max(num_i.bit_length(), den_i.bit_length()) - 900
                if shift > 0:
                    num_i >>= shift
                    den_i >>= shift
                if den_i == 0:
                    return 0.0
                return float(num_i) / float(den_i)
            except (TypeError, ValueError):
                import mpmath
                return float(mpmath.mpf(int(num)) / mpmath.mpf(int(den)))
    return float(value)


def _safe_local_name(gen_name: str) -> str:
    """Return a Python identifier safe to use as a local var for ``gen_name``.

    Generator names like ``V_0_0`` or ``_buildonce_xi_..._0`` are already
    valid Python identifiers, but anything fancy gets prefixed with
    ``_g_`` and sanitised to underscores so the generated source compiles.
    """
    if gen_name.isidentifier():
        return f"_g_{gen_name}"
    return "_g_" + "".join(ch if ch.isalnum() else "_" for ch in gen_name)


def _poly_to_pyexpr(
    poly: Poly,
    names: Sequence[str],
    var_names: Sequence[str],
) -> str:
    """Render ``poly`` as a Python expression over ``var_names``.

    Each term ``coeff · prod_j var_j ** exp_ij`` becomes ``c*v0**e0*...``;
    the term list is joined with ``+``.  Empty poly renders as ``0.0``.

    Use :func:`_poly_to_sum_pyexpr` for hot paths — long ``+`` chains
    blow CPython's compile recursion limit on Christoffel-chain
    polynomials.
    """
    terms = _poly_term_strings(poly, var_names)
    if not terms:
        return "0.0"
    return " + ".join(terms)


_TERM_CHUNK = 32


def _emit_poly_assign(
    poly: Poly,
    var_names: Sequence[str],
    target: str,
) -> list[str]:
    """Emit ``    target = ...`` / ``    target += ...`` lines.

    The first line initialises ``target`` to the first chunk (or
    ``0.0`` for empty polynomials); subsequent lines accumulate.
    Chunk size :data:`_TERM_CHUNK` caps per-line AST depth at ~32
    binary ops, well under CPython's compile recursion limit, while
    each chunk still inlines as a ``BINARY_OP`` chain — no function
    call per eval.
    """
    terms = _poly_term_strings(poly, var_names)
    if not terms:
        return [f"    {target} = 0.0"]
    chunks = [terms[i:i + _TERM_CHUNK] for i in range(0, len(terms), _TERM_CHUNK)]
    lines = [f"    {target} = " + " + ".join(chunks[0])]
    for ch in chunks[1:]:
        lines.append(f"    {target} += " + " + ".join(ch))
    return lines


def _poly_term_strings(
    poly: Poly,
    var_names: Sequence[str],
) -> list[str]:
    """Helper shared by :func:`_poly_to_pyexpr` and :func:`_poly_to_sum_pyexpr`."""
    terms: list[str] = []
    for monom, coeff in poly.terms():
        c = _coeff_to_float(coeff)
        if c == 0.0:
            continue
        factors: list[str] = []
        for src_pos, exp in enumerate(monom):
            if exp == 0:
                continue
            v = var_names[src_pos]
            if exp == 1:
                factors.append(v)
            else:
                factors.append(f"{v}**{exp}")
        if not factors:
            # pure constant term — the coefficient is the whole term.
            terms.append(repr(c))
        elif c == 1.0:
            # drop the redundant ``1.0*`` coefficient prefix.
            terms.append("*".join(factors))
        elif c == -1.0:
            # ``-1.0*x`` ⇒ ``-x`` (unary negate, matching existing ``+ -`` style).
            terms.append("-" + "*".join(factors))
        else:
            terms.append("*".join([repr(c), *factors]))
    return terms


def _eval_poly_full(poly: Poly, names: Sequence[str], bindings: Mapping[str, Any]) -> float:
    """Sum ``poly`` at a FULL numeric binding: Σ coeff · Π binding[var]**exp over its terms.

    The no-``compile`` numeric evaluator behind :meth:`RationalFunction.eval_numeric_direct`.
    Walks ``poly.terms()`` in the same order and with the same ``_coeff_to_float`` coercion as
    :func:`_poly_term_strings` (the source the compiled evaluator is built from), so the result
    is bit-identical to running that compiled evaluator — a skipped zero-coefficient term
    contributes ``0.0`` (a float no-op), so omitting it (as the string builder does) changes
    nothing."""
    total = 0.0
    for monom, coeff in poly.terms():
        c = _coeff_to_float(coeff)
        for src_pos, exp in enumerate(monom):
            if exp:
                c *= float(bindings[names[src_pos]]) ** int(exp)
        total += c
    return total


def _partial_substitute(
    poly: Poly,
    src_names: tuple[str, ...],
    bindings: Mapping[str, Any],
    target: Ring,
    leftover_names: list[str],
) -> Poly:
    """Partial-substitute bound generators into ``poly``, leaving the rest.

    Iterates terms once: each term's coefficient absorbs the bound-gen
    contributions (``v ** exp`` for each bound gen), and the residue
    monomial is reindexed into the target ring spanning ``leftover_names``.
    """
    leftover_idx_in_src = [src_names.index(n) for n in leftover_names]
    n_left = len(leftover_names)
    new_dict: dict[tuple[int, ...], float] = {}
    for monom, coeff in poly.terms():
        # Coerce to Python ``float`` up-front so the per-term math is
        # uniform across the supported coefficient types: ``mpf``
        # (sympy ``RR``), ``flint.fmpq`` (flint backend), and
        # ``PythonMPQ`` / ``sympy.Rational`` (legacy QQ-domain inputs).
        c = _coeff_to_float(coeff)
        for src_pos, exp in enumerate(monom):
            if exp == 0:
                continue
            name = src_names[src_pos]
            if name in bindings:
                c = c * (float(bindings[name]) ** int(exp))
        if not n_left:
            key: tuple[int, ...] = ()
        else:
            key = tuple(int(monom[src_pos]) for src_pos in leftover_idx_in_src)
        prev = new_dict.get(key)
        new_dict[key] = c if prev is None else prev + c
    if not n_left:
        const = new_dict.get((), 0.0)
        return target.ground_new(const)
    return target.from_dict(new_dict) if new_dict else target.zero


def _compose_poly(
    poly: Poly,
    src_names: tuple[str, ...],
    name_pos: int,
    repl_t: RationalFunction,
    target: Ring,
) -> RationalFunction:
    """Substitute the generator at ``name_pos`` with ``repl_t`` in ``poly``.

    The symbolic analogue of :func:`_partial_substitute`: iterate ``poly``'s
    terms once; each term is ``coeff · (leftover monomial) · name**exp``.  The
    leftover monomial (and the coefficient folded in) is built directly in the
    target ring, the bound generator's ``name**exp`` is the RF power
    ``repl_t**exp``, and the term RFs are summed.  ``repl_t`` is assumed already
    lifted into ``target``.
    """
    target_names = target.names
    n_tgt = len(target_names)
    tgt_index = {nm: i for i, nm in enumerate(target_names)}
    result = RationalFunction.constant(0, target)
    pow_cache: dict[int, RationalFunction] = {
        0: RationalFunction.constant(1, target)
    }

    def repl_pow(e: int) -> RationalFunction:
        p = pow_cache.get(e)
        if p is None:
            p = repl_t ** e
            pow_cache[e] = p
        return p

    for monom, coeff in poly.terms():
        c = _coeff_to_float(coeff)
        if c == 0.0:
            continue
        exp_name = int(monom[name_pos])
        if n_tgt == 0:
            mono = RationalFunction.constant(c, target)
        else:
            tgt_exp = [0] * n_tgt
            for src_pos, exp in enumerate(monom):
                if exp == 0 or src_pos == name_pos:
                    continue
                tgt_exp[tgt_index[src_names[src_pos]]] += int(exp)
            mono = RationalFunction.from_poly(target.from_dict({tuple(tgt_exp): c}))
        term = mono if not exp_name else mono * repl_pow(exp_name)
        result = result + term
    return result


# ---------------------------------------------------------------------------
# simple_zero predicate
# ---------------------------------------------------------------------------

def simple_zero(value: Any) -> bool:
    """Return ``True`` if ``value`` is structurally zero.

    Accepts a :class:`RationalFunction`, a backend :class:`Poly`, or a
    plain Python numeric.  The numeric branch is the only path that
    inspects a value — and even then it uses an exact equality check;
    callers that want a tolerance should wrap their own predicate.
    """
    if isinstance(value, RationalFunction):
        return coeff_zero(value.num)
    if isinstance(value, PolyTypes):
        return coeff_zero(value)
    if isinstance(value, (int, float, sp.Integer, sp.Rational)):
        return value == 0
    if isinstance(value, sp.Expr) and value.is_number:
        return value == 0
    return False


# ---------------------------------------------------------------------------
# Bareiss determinant (fraction-free elimination)
# ---------------------------------------------------------------------------

def bareiss_det(matrix: np.ndarray) -> RationalFunction | float:
    """Return ``det(matrix)`` over :class:`RationalFunction` cells.

    ``matrix`` is an ``(n, n)`` ndarray of cells (``RationalFunction``
    or numeric).  The implementation is the standard Bareiss
    fraction-free algorithm, which keeps intermediate entries
    polynomial whenever the input is polynomial.

    For an all-numeric input we short-circuit to :func:`numpy.linalg.det`.
    """
    arr = np.asarray(matrix, dtype=object)
    n, m = arr.shape
    if n != m:
        raise ValueError(f"bareiss_det requires a square matrix; got {arr.shape}")
    # Numeric short-circuit.
    if all(isinstance(c, (int, float)) or (isinstance(c, sp.Expr) and c.is_number) for c in arr.flat):
        return float(np.linalg.det(arr.astype(float)))
    if n == 0:
        return RationalFunction.constant(1)
    # Promote every cell to RationalFunction over a joint ring.
    cells: list[list[RationalFunction]] = []
    joint_ring: Ring | None = None
    for i in range(n):
        row: list[RationalFunction] = []
        for j in range(n):
            v = arr[i, j]
            if isinstance(v, RationalFunction):
                rf = v
            else:
                rf = RationalFunction.constant(v)
            if joint_ring is None:
                joint_ring = rf.ring
            elif rf.ring is not joint_ring:
                joint_ring = _join_rings(joint_ring, rf.ring)
            row.append(rf)
        cells.append(row)
    assert joint_ring is not None
    # Re-lift everything into joint_ring.
    cells = [
        [
            RationalFunction(_lift(c.num, joint_ring), _lift(c.den, joint_ring))
            for c in row
        ]
        for row in cells
    ]
    sign = 1
    prev = RationalFunction.constant(1, joint_ring)
    for k in range(n - 1):
        # Pivot: find a structurally non-zero entry in column k below the diagonal.
        if simple_zero(cells[k][k]):
            for r in range(k + 1, n):
                if not simple_zero(cells[r][k]):
                    cells[k], cells[r] = cells[r], cells[k]
                    sign = -sign
                    break
            else:
                return RationalFunction.constant(0, joint_ring)
        pivot = cells[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                cells[i][j] = (cells[i][j] * pivot - cells[i][k] * cells[k][j]) / prev
            cells[i][k] = RationalFunction.constant(0, joint_ring)
        prev = pivot
    result = cells[n - 1][n - 1]
    if sign < 0:
        result = -result
    # Bareiss is fraction-free in principle: every divide-by-prev should
    # be exact.  In our RationalFunction arithmetic the divisions stack
    # in the denominator until cleaned; so we run a single boundary
    # cancel pass on the way out.  This is cheaper than per-step
    # polynomial exact division because the gcd is taken once over the
    # whole numerator/denominator instead of after every elimination.
    return result.clean()


# ---------------------------------------------------------------------------
# Block-Schur inverse for small matrices
# ---------------------------------------------------------------------------

def cofactor_inverse(matrix: np.ndarray) -> np.ndarray:
    """Closed-form inverse of an ``(n, n)`` matrix of cells.

    Implementation is the cofactor expansion ``A^{-1} = adj(A) / det(A)``,
    suitable only for tiny ``n`` (≤ 4 in practice).  Returns an
    ndarray of :class:`RationalFunction` cells (or floats, in the
    numeric short-circuit).
    """
    arr = np.asarray(matrix, dtype=object)
    n, m = arr.shape
    if n != m:
        raise ValueError(f"cofactor_inverse requires square matrix; got {arr.shape}")
    if n == 0:
        return np.zeros((0, 0), dtype=object)
    det = bareiss_det(arr)
    if simple_zero(det):
        raise ZeroDivisionError("matrix is structurally singular")
    # Numeric short-circuit.
    if isinstance(det, float):
        return np.linalg.inv(arr.astype(float))
    # Otherwise compute adjugate.
    out = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(arr, i, axis=0), j, axis=1)
            cof = bareiss_det(minor)
            sign = 1 if (i + j) % 2 == 0 else -1
            entry = cof if sign > 0 else -_as_rf(cof)
            out[j, i] = _as_rf(entry) / det
    return out


def _as_rf(value: Any) -> RationalFunction:
    """Promote ``value`` to a :class:`RationalFunction`."""
    if isinstance(value, RationalFunction):
        return value
    return RationalFunction.constant(value)


def schur_inverse(matrix: np.ndarray, top_left: int) -> np.ndarray:
    """Block-``2x2`` Schur-complement inverse with ``top_left`` rows in the upper block.

    For
    ``M = [[A, B], [C, D]]`` with ``A`` square of size ``top_left``,
    returns
    ``M^{-1}`` via
    ``S = D - C A^{-1} B``,
    ``[[A^{-1} + A^{-1} B S^{-1} C A^{-1}, -A^{-1} B S^{-1}],
       [-S^{-1} C A^{-1}, S^{-1}]]``.
    Raises ``ZeroDivisionError`` if ``A`` is structurally singular.
    """
    arr = np.asarray(matrix, dtype=object)
    n, m = arr.shape
    if n != m:
        raise ValueError(f"schur_inverse requires square matrix; got {arr.shape}")
    k = int(top_left)
    if not (0 < k < n):
        raise ValueError(f"top_left must be strictly between 0 and n; got {k} of {n}")
    A = arr[:k, :k]
    B = arr[:k, k:]
    C = arr[k:, :k]
    D = arr[k:, k:]
    A_inv = cofactor_inverse(A)
    AB = A_inv @ B
    CA = C @ A_inv
    S = D - C @ AB
    S_inv = cofactor_inverse(S)
    AB_Sinv = AB @ S_inv
    out = np.empty((n, n), dtype=object)
    out[:k, :k] = A_inv + AB_Sinv @ CA
    out[:k, k:] = -AB_Sinv
    out[k:, :k] = -(S_inv @ CA)
    out[k:, k:] = S_inv
    return out


__all__ = [
    "RationalFunction",
    "bareiss_det",
    "cofactor_inverse",
    "coeff_zero",
    "schur_inverse",
    "simple_zero",
]
