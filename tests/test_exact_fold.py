"""The EXACT lane of `partial_eval_numeric` (modes exact | hybrid | probe).

The probe-and-freeze fold is probabilistic; since the ``P(T)=I`` affine-invariance
certificate rides on it, the exact lane certifies constancy by the rational normal
form of each entry (flint exact-rational arithmetic) and the default (hybrid) path
WARNS wherever a certificate is issued non-exactly.  These tests pin:

* an exactly-constant rational entry certifies with NO warning;
* a vertex-dependent entry that a COLLUDING probe set would freeze is REFUTED
  by the exact check (the unsoundness this lane closes);
* a non-normalizable (opaque-op) entry falls back to probes WITH the warning;
* probe-count configurability is honored;
* the entry-level fold certifies a cancellation no single statement exhibits.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from polyarray import (
    InvOp,
    NonExactFoldWarning,
    OutSpec,
    Program,
    Provenance,
    SymInput,
    TensordotOp,
    partial_eval_numeric,
    partial_eval_numeric_symarray,
)
from polyarray.ir import IdentityOp, OutputRef, SymArray


def _prov(name: str) -> Provenance:
    return Provenance("vertex", name, (), name)


def _inv_chain_program() -> Program:
    """The canonical invariant chain: C = A · inv(A) ≡ I (A symbolic everywhere)."""
    prog = Program("painv", inputs=[SymInput("A", (2, 2), _prov("A"))])
    prog.emit_stmt(InvOp(), [prog.input("A")], [OutSpec("B", (2, 2))], bulk=False)
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), OutputRef(0, 0)],
        [OutSpec("C", (2, 2))],
        bulk=False,
    )
    prog.add_output("C", C.cells)
    return prog


def test_exact_constant_certifies_without_warning() -> None:
    """A·inv(A): every entry's rational normal form is degree-0 (1 or 0) — the exact
    lane folds it with NO probe and NO warning, in both exact and hybrid modes."""
    for mode in ("exact", "hybrid"):
        prog = _inv_chain_program()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            folded = partial_eval_numeric(prog, mode=mode)
        assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)], mode
        rng = np.random.default_rng(7)
        A = rng.uniform(0.5, 1.5, (2, 2)) + np.eye(2)
        np.testing.assert_allclose(folded.run({"A": A})["C"], np.eye(2), atol=0.0)


def _colluding_program() -> tuple[Program, float]:
    """A stmt whose output is VERTEX-DEPENDENT yet agrees at every default probe.

    The probe lane (probes=3, seed=0) binds the single ``(1,)`` input to three
    deterministic draws a, b, c of ``default_rng(0).uniform(0.6, 1.6, (1,))``.
    The cell q(x) = (x−a)(x−b)(x−c) + 7 evaluates to EXACTLY 7.0 at all three
    probes (x−a is exact float zero at x==a), so probe-and-freeze folds it — a
    genuine false freeze.  Returns (program, q(1.0)) with q(1.0) ≠ 7."""
    prog = Program("collude", inputs=[SymInput("x", (1,), _prov("x"))])
    x_rf = np.asarray(prog.input_arrays["x"].cells)[0]
    rng = np.random.default_rng(0)
    a = float(rng.uniform(0.6, 1.6, (1,))[0])
    b = float(rng.uniform(0.6, 1.6, (1,))[0])
    c = float(rng.uniform(0.6, 1.6, (1,))[0])
    q = (x_rf - a) * (x_rf - b) * (x_rf - c) + 7.0
    sa = SymArray(np.array([q], dtype=object), program=prog)
    [Y] = prog.emit_stmt(IdentityOp(), [sa], [OutSpec("Y", (1,))], bulk=False)
    prog.add_output("Y", Y.cells)
    true_at_1 = (1.0 - a) * (1.0 - b) * (1.0 - c) + 7.0
    assert abs(true_at_1 - 7.0) > 1e-3
    return prog, true_at_1


def test_colluding_probes_would_freeze_wrongly_probe_mode() -> None:
    """mode='probe' (the legacy behavior) IS fooled by the colluding probe set —
    documenting the unsoundness the exact lane closes."""
    prog, true_at_1 = _colluding_program()
    folded = partial_eval_numeric(prog, mode="probe")
    assert len(folded.statements) == 0                      # wrongly frozen
    got = np.asarray(folded.run({"x": np.array([1.0])})["Y"], float)
    np.testing.assert_allclose(got, [7.0])                  # the frozen (WRONG) constant
    assert abs(got[0] - true_at_1) > 1e-3


def test_exact_check_refutes_colluding_probes() -> None:
    """exact/hybrid: the entry's rational normal form is NON-constant — refuted
    exactly, the statement survives, and hybrid does NOT hand it to the probe
    fallback (no warning: a refutation is exact, not a probe certificate)."""
    for mode in ("exact", "hybrid"):
        prog, true_at_1 = _colluding_program()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            folded = partial_eval_numeric(prog, mode=mode)
        assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)], mode
        assert len(folded.statements) == 1, mode            # NOT frozen
        got = np.asarray(folded.run({"x": np.array([1.0])})["Y"], float)
        np.testing.assert_allclose(got, [true_at_1])        # still the true value


class _OpaqueInvariant:
    """An op the exact lane cannot normalize whose value IS invariant on the probe
    box: sign(x) + 2 ≡ 3 on x ∈ [0.6, 1.6] (but not globally — sign is opaque)."""

    calls: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        type(self).calls += 1
        return np.sign(np.asarray(x, dtype=float)) + 2.0


def _opaque_program() -> Program:
    prog = Program("opq", inputs=[SymInput("x", (1,), _prov("x"))])
    [Y] = prog.emit_stmt(_OpaqueInvariant(), [prog.input("x")],
                         [OutSpec("Y", (1,))], bulk=False)
    prog.add_output("Y", Y.cells)
    return prog


def test_non_normalizable_entry_falls_back_with_warning() -> None:
    """hybrid: an opaque op the exact lane cannot execute is probe-frozen — and the
    fallback is LOUD (NonExactFoldWarning naming the site)."""
    prog = _opaque_program()
    with pytest.warns(NonExactFoldWarning, match="PROBE"):
        folded = partial_eval_numeric(prog, mode="hybrid")
    assert len(folded.statements) == 0                      # frozen (by probes)
    got = np.asarray(folded.run({"x": np.array([1.0])})["Y"], float)
    np.testing.assert_allclose(got, [3.0])


def test_exact_mode_refuses_non_normalizable_fold() -> None:
    """exact: the opaque statement is left symbolic — no probe, no warning."""
    prog = _opaque_program()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        folded = partial_eval_numeric(prog, mode="exact")
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]
    assert len(folded.statements) == 1                      # untouched


def test_probe_count_configurable() -> None:
    """``probes`` reaches the probe lane: the opaque op runs once per probe."""
    for mode in ("probe", "hybrid"):
        for probes in (2, 5):
            _OpaqueInvariant.calls = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                partial_eval_numeric(_opaque_program(), mode=mode, probes=probes)
            assert _OpaqueInvariant.calls == probes, (mode, probes)
    with pytest.raises(ValueError):
        partial_eval_numeric(_opaque_program(), probes=1)


def test_entry_level_cancellation_certifies_exactly() -> None:
    """A cancellation that completes only at the ENTRY (no statement invariant):
    y = x (an identity stmt whose output VARIES), entry = y/x ≡ 1.  The entry-level
    exact fold certifies the constant; the statement itself survives."""
    prog = Program("entry", inputs=[SymInput("x", (1,), _prov("x"))])
    [Y] = prog.emit_stmt(IdentityOp(), [prog.input("x")],
                         [OutSpec("Y", (1,))], bulk=False)
    prog.add_output("Y", Y.cells)
    x_rf = np.asarray(prog.input_arrays["x"].cells)[0]
    y_rf = np.asarray(Y.cells)[0]
    sa = SymArray(np.array([y_rf / x_rf], dtype=object), program=prog)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        folded = partial_eval_numeric_symarray(sa, mode="exact")
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]
    vals = np.asarray(folded.evaluate({}), float)           # collapses: entry is constant
    np.testing.assert_allclose(vals, [1.0])


def test_non_dyadic_constant_through_cancellation_is_folded_not_refuted() -> None:
    """REGRESSION (adversarial audit, 2026-07-30): the old two-point short-circuit
    evaluated cells by FLOAT term-summation, so a constant-through-cancellation cell
    with a non-dyadic constant — ``c·p/p`` with ``c = 1/3`` — produced two different
    floats and was declared "provably non-constant" WITHOUT the exact gcd running.
    A falsely-refuted statement is excluded from BOTH lanes in hybrid mode, silently
    gating a genuinely affine-invariant entry False.  The filter now evaluates in
    EXACT fmpq arithmetic: these cells must classify FOLDED."""
    from polyarray.exact_fold import _constant_value
    from polyarray.rational import RationalFunction as RF

    x, y = RF.atom("x"), RF.atom("y")
    p = x * x + y + x * y
    for c in (1.0 / 3.0, 7.0 / 3.0):
        cell = (RF.constant(c) * p) / p
        assert _constant_value(cell) == c, f"c={c}: constant-through-cancellation refuted"
    # and genuinely varying cells are still (exactly) refuted by the filter
    assert _constant_value(p) is None
    assert _constant_value(p / x) is None

    # statement-level: an IdentityOp over the (1/3)·p/p cell must FOLD in exact mode
    # (no probe, no warning), with the exact constant as the frozen value.
    c = 1.0 / 3.0
    prog = Program("nondyadic", inputs=[SymInput("x", (1,), _prov("x"))])
    x_rf = np.asarray(prog.input_arrays["x"].cells)[0]
    q = x_rf * x_rf + x_rf + 1.0                     # strictly positive ⇒ never singular
    sa = SymArray(np.array([(RF.constant(c) * q) / q], dtype=object), program=prog)
    [Y] = prog.emit_stmt(IdentityOp(), [sa], [OutSpec("Y", (1,))], bulk=False)
    prog.add_output("Y", Y.cells)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        folded = partial_eval_numeric(prog, mode="exact")
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]
    assert len(folded.statements) == 0, "constant-through-cancellation stmt must fold"
    got = np.asarray(folded.run({"x": np.array([2.0])})["Y"], float)
    np.testing.assert_allclose(got, [c], atol=0.0)


def test_time_budget_degrades_to_unresolved_not_hang() -> None:
    """REGRESSION (re-audit, 2026-07-30): the exact filter/classification must honor
    ``time_budget`` — a pathological cell degrades to *unresolved* (⇒ the warned
    probe fallback in hybrid), never a gate hang.  Pinned at three levels."""
    import time as _time

    from polyarray.exact_fold import _Timeout, _constant_value
    from polyarray.rational import RationalFunction as RF

    # (1) unit: an expired deadline raises _Timeout before any expensive step.
    x = RF.atom("x")
    rf = (RF.constant(1.0 / 3.0) * (x * x + 1.0)) / (x * x + 1.0)
    with pytest.raises(_Timeout):
        _constant_value(rf, deadline=_time.monotonic() - 1.0)
    # no deadline / generous deadline: same exact answer as before.
    assert _constant_value(rf) == 1.0 / 3.0
    assert _constant_value(rf, deadline=_time.monotonic() + 60.0) == 1.0 / 3.0

    # (2) end-to-end: time_budget=0 ⇒ the (foldable!) constant-through-cancellation
    # statement is NOT exact-folded; exact mode leaves it symbolic, silently.
    def _prog() -> Program:
        prog = Program("budget", inputs=[SymInput("x", (1,), _prov("x"))])
        x_rf = np.asarray(prog.input_arrays["x"].cells)[0]
        q = x_rf * x_rf + x_rf + 1.0
        sa = SymArray(np.array([(RF.constant(1.0 / 3.0) * q) / q], dtype=object),
                      program=prog)
        [Y] = prog.emit_stmt(IdentityOp(), [sa], [OutSpec("Y", (1,))], bulk=False)
        prog.add_output("Y", Y.cells)
        return prog

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        folded = partial_eval_numeric(_prog(), mode="exact", time_budget=0.0)
    assert len(folded.statements) == 1, "expired budget must leave the stmt unresolved"
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]

    # (3) hybrid: the unresolved statement goes to the LOUD probe fallback, and the
    # warning carries the time-budget reason.
    with pytest.warns(NonExactFoldWarning, match="time budget"):
        folded = partial_eval_numeric(_prog(), mode="hybrid", time_budget=0.0)
    assert len(folded.statements) == 0                  # probe-frozen (invariant)


def test_oversized_symbolic_op_is_rejected_before_it_runs() -> None:
    """REGRESSION (oracle lowering stall, 2026-07-30): the time budget alone cannot
    bound the exact lane — ONE ``np.einsum`` over object-dtype RF cells runs to
    completion inside a single deadline interval (Bell's degree-5 symbolic W stalled
    the oracle pyab-lowering gate >55 min under a nominal 10 s budget).  Operands
    above ``max_sym_mass`` must be rejected BEFORE the op runs ⇒ unresolved ⇒ the
    warned probe fallback."""
    from polyarray.exact_fold import _MAX_SYM_MASS, _sym_mass, exact_partial_eval
    from polyarray.rational import RationalFunction as RF

    # A cell with many monomials: (x + y + 1)^8 has 45 terms; mass counting early-exits.
    x, y = RF.atom("x"), RF.atom("y")
    big = (x + y + 1.0) ** 8
    assert _sym_mass([np.array([big], dtype=object)], 10) > 10        # early-exit works
    assert _sym_mass([np.array([big], dtype=object)], 10**6) == big.num.n_terms() + 1

    prog = Program("big", inputs=[SymInput("x", (1,), _prov("x"))])
    x_rf = np.asarray(prog.input_arrays["x"].cells)[0]
    cell = (x_rf + 1.0) ** 6
    sa = SymArray(np.array([cell], dtype=object), program=prog)
    [Y] = prog.emit_stmt(IdentityOp(), [sa], [OutSpec("Y", (1,))], bulk=False)
    prog.add_output("Y", Y.cells)

    # A cap BELOW this operand's mass ⇒ unresolved with the size reason (not executed).
    state = exact_partial_eval(prog, time_budget=60.0, max_sym_mass=2)
    assert 0 in state.unresolved
    assert "too large" in state.unresolved[0]
    assert not state.folded and not state.refuted
    # The production cap admits it (an ordinary small cell), and it is exactly refuted.
    state = exact_partial_eval(prog, time_budget=60.0, max_sym_mass=_MAX_SYM_MASS)
    assert state.refuted == {0} and not state.unresolved

    # hybrid: a size-rejected but genuinely INVARIANT statement (``(x+1)⁶/(x+1)⁶ ≡ 1``,
    # left uncancelled) reaches the LOUD probe fallback and is frozen there.
    prog2 = Program("bigconst", inputs=[SymInput("x", (1,), _prov("x"))])
    x2 = np.asarray(prog2.input_arrays["x"].cells)[0]
    ratio = ((x2 + 1.0) ** 6) / ((x2 + 1.0) ** 6)
    sa2 = SymArray(np.array([ratio], dtype=object), program=prog2)
    [Y2] = prog2.emit_stmt(IdentityOp(), [sa2], [OutSpec("Y", (1,))], bulk=False)
    prog2.add_output("Y", Y2.cells)
    with pytest.warns(NonExactFoldWarning, match="too large"):
        folded = partial_eval_numeric(prog2, mode="hybrid", max_sym_mass=2)
    assert len(folded.statements) == 0                      # probe-frozen
    np.testing.assert_allclose(
        np.asarray(folded.run({"x": np.array([3.0])})["Y"], float), [1.0])


def test_chained_compose_is_coefficient_exact() -> None:
    """REGRESSION for the root cause: ``_compose_poly`` used to round every source
    coefficient through ``float``, so the SECOND compose of a ``compose_multi`` chain
    rounded the non-double-representable coefficients the first compose produced
    (e.g. ``(1/3)·0.5625``).  Substitution must be exact on the exact backend."""
    from polyarray.rational import RationalFunction as RF

    c = 1.0 / 3.0
    x, y = RF.atom("x"), RF.atom("y")
    p = RF.constant(c) * (x * x + y + x * y)
    e = p.compose_multi({"x": RF.constant(0.75), "y": RF.constant(2.5)})
    # exact expectation: c·(0.75² + 2.5 + 0.75·2.5) = c·4.9375, cross-multiplied exactly
    assert e == RF.constant(c) * RF.constant(4.9375)


def test_probe_mode_is_silent_and_env_default_is_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode='probe' keeps the legacy silent behavior; the env var only moves the
    DEFAULT — an explicit ``mode`` argument always wins."""
    prog = _opaque_program()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        partial_eval_numeric(prog, mode="probe")
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]

    monkeypatch.setenv("POLYARRAY_PARTIAL_EVAL_MODE", "probe")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        partial_eval_numeric(_opaque_program())             # default from env: probe, silent
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]
    with pytest.warns(NonExactFoldWarning):
        partial_eval_numeric(_opaque_program(), mode="hybrid")   # explicit wins over env

    with pytest.raises(ValueError):
        partial_eval_numeric(_opaque_program(), mode="bogus")


# ---------------------------------------------------------------------------
# The vmap-closure descent + the front-end-op rational twins (2026-07-31).
#
# Before these, a vmap closure was OPAQUE to the exact lane and everything downstream of
# it fell back to probe-and-freeze (measured: 1120 of 1180 statements of the P⁻₂Λ¹(TET)
# symbolic Vandermonde, all inside `grass_dof` vmap bodies).
# ---------------------------------------------------------------------------

def _vmap_inv_chain_program(batch: int = 3, n: int = 2) -> Program:
    """``vmap(A ↦ A·inv(A))`` over a batched symbolic ``A`` — every output entry is
    identically ``I``, but ONLY the descent into the closure can see that."""
    from polyarray.ir import vmap

    body = Program("inv_body", inputs=[SymInput("A", (n, n), _prov("A"))])
    body.emit_stmt(InvOp(), [body.input("A")], [OutSpec("B", (n, n))], bulk=False)
    [C] = body.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [body.input("A"), OutputRef(0, 0)],
        [OutSpec("C", (n, n))],
        bulk=False,
    )
    body.add_output("C", C.cells)

    prog = Program("vmap_inv", inputs=[SymInput("Ab", (batch, n, n), _prov("Ab"))])
    [out] = prog.emit_stmt(
        vmap(body, in_axes=(0,), out_axes=0), [prog.input("Ab")],
        [OutSpec("R", (batch, n, n))], bulk=False,
    )
    prog.add_output("R", out.cells)
    return prog


def test_vmap_closure_descends_and_certifies_exactly() -> None:
    """The batched ``A·inv(A) ≡ I`` folds EXACTLY through the closure — no probe, no
    warning — and the folded program still returns ``I`` on real feeds."""
    prog = _vmap_inv_chain_program()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        folded = partial_eval_numeric(prog, mode="exact")
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]
    rng = np.random.default_rng(3)
    Ab = rng.uniform(0.5, 1.5, (3, 2, 2)) + np.eye(2)
    np.testing.assert_allclose(folded.run({"Ab": Ab})["R"],
                               np.broadcast_to(np.eye(2), (3, 2, 2)), atol=0.0)


def test_vmap_batch_cap_bounds_the_work_before_it_starts() -> None:
    """A batch over the cap is declared unresolved UP FRONT (⇒ the warned probe
    fallback), never descended: the Bell-stall rule — an uninterruptible op ignores a
    deadline, so the work is bounded before it starts, not interrupted after."""
    import polyarray.exact_fold as EF

    prog = _vmap_inv_chain_program(batch=4)
    st = EF.exact_partial_eval(prog, time_budget=30.0)
    assert st.folded == {0}, st.unresolved                  # under the cap: descends

    prog = _vmap_inv_chain_program(batch=4)
    st = EF.exact_partial_eval(prog, time_budget=30.0, max_sym_mass=EF._MAX_SYM_MASS)
    saved, EF._MAX_VMAP_BATCH = EF._MAX_VMAP_BATCH, 2
    try:
        st = EF.exact_partial_eval(_vmap_inv_chain_program(batch=4), time_budget=30.0)
    finally:
        EF._MAX_VMAP_BATCH = saved
    assert 0 in st.unresolved and "batch 4" in st.unresolved[0], st.unresolved


def test_pinv_twin_is_the_exact_generic_pseudo_inverse() -> None:
    """``pinv(A)·A ≡ I`` for a symbolic TALL ``A`` — the exact twin is the
    normal-equation form ``(AᵀA)⁻¹Aᵀ``, so the composition folds to the identity."""
    from polyarray.ir import EinsumStmtOp, PinvOp

    prog = Program("pinv_chain", inputs=[SymInput("A", (3, 2), _prov("A"))])
    prog.emit_stmt(PinvOp(), [prog.input("A")], [OutSpec("P", (2, 3))], bulk=False)
    [C] = prog.emit_stmt(
        EinsumStmtOp("kn,nj->kj"), [OutputRef(0, 0), prog.input("A")],
        [OutSpec("C", (2, 2))], bulk=False,
    )
    prog.add_output("C", C.cells)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        folded = partial_eval_numeric(prog, mode="exact")
    assert not [w for w in rec if issubclass(w.category, NonExactFoldWarning)]
    A = np.array([[1.0, 0.3], [0.2, 1.0], [0.4, 0.7]])
    np.testing.assert_allclose(folded.run({"A": A})["C"], np.eye(2), atol=1e-12)


def test_project_embed_axislen_twins_are_exact() -> None:
    """``Project`` / ``Embed`` / ``AxisLen`` thread exact cells: with an ORTHONORMAL
    (constant) frame ``P``, ``Pᵀ·(P·v) ≡ v`` — the round trip folds away entirely, and
    the axis length is exact even over a fully symbolic operand."""
    from polyarray.ir import AxisLenOp, Const, EmbedOp, ProjectOp

    P = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    prog = Program("proj_embed", inputs=[SymInput("v", (2,), _prov("v"))])
    [amb] = prog.emit_stmt(
        EmbedOp((3,)), [Const(P), prog.input("v")], [OutSpec("amb", (3,))], bulk=False)
    [back] = prog.emit_stmt(
        ProjectOp(), [Const(P), OutputRef(0, 0)], [OutSpec("back", (2,))], bulk=False)
    [n] = prog.emit_stmt(
        AxisLenOp(0), [OutputRef(0, 0)], [OutSpec("n", ())], bulk=False)
    prog.add_output("back", back.cells)
    prog.add_output("n", n.cells)
    import polyarray.exact_fold as EF
    st = EF.exact_partial_eval(prog, time_budget=30.0)
    assert st.folded == {2}, (st.folded, st.unresolved)     # AxisLen is CONSTANT (=3)
    assert st.refuted == {0, 1}, (st.refuted, st.unresolved)  # embed/project vary with v
    v = np.array([0.3, -1.25])
    out = prog.run({"v": v})
    np.testing.assert_allclose(out["back"], v, atol=0.0)
    assert float(np.asarray(out["n"])) == 3.0
    assert amb is not None


def test_assert_twin_passes_the_value_through_and_still_checks() -> None:
    """The ``AssertOp`` twin is value-transparent on a symbolic operand (so the guard's
    Stmt survives to run on real data) and still RAISES ⇒ opaque on a decidable failure."""
    from polyarray.ir import AssertOp
    import polyarray.exact_fold as EF

    prog = Program("assert_ok", inputs=[SymInput("A", (2, 2), _prov("A"))])
    [x] = prog.emit_stmt(
        AssertOp("square_full_rank", "guard"), [prog.input("A")],
        [OutSpec("X", (2, 2))], bulk=False)
    prog.add_output("X", x.cells)
    st = EF.exact_partial_eval(prog, time_budget=30.0)
    assert st.refuted == {0} and not st.unresolved          # value threaded, not opaque

    bad = Program("assert_bad", inputs=[SymInput("A", (2, 2), _prov("A"))])
    a = np.asarray(bad.input_arrays["A"].cells)
    sing = SymArray(np.array([[a[0, 0], a[0, 0]], [a[1, 0], a[1, 0]]], dtype=object),
                    program=bad)                            # structurally singular
    [y] = bad.emit_stmt(
        AssertOp("square_full_rank", "guard"), [sing], [OutSpec("Y", (2, 2))], bulk=False)
    bad.add_output("Y", y.cells)
    st = EF.exact_partial_eval(bad, time_budget=30.0)
    assert 0 in st.unresolved, st                            # decidable failure ⇒ opaque


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_kron_twins_are_exact_and_certify_a_constant_wedge() -> None:
    """``KronOp`` / ``KronFreeOp`` thread exact cells, and a Kron of CONSTANT operands folds.

    These were the FEEC `Λᵏ` blocker: both were simply ABSENT from ``_sym_apply``'s ladder, so they
    fell off the end into the opaque tail — declared un-normalizable by omission rather than by a
    decision.  A Kronecker product is field MULTIPLICATION, so it is exactly representable; leaving
    it opaque put the whole wedge DOF (whose two traced slots meet in a ``KronFreeOp``) on the
    probe lane."""
    from polyarray.ir import Const, KronFreeOp, KronOp

    A = np.array([[1.0, 2.0], [0.0, -1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    prog = Program("kron", inputs=[SymInput("v", (2,), _prov("v"))])
    prog.emit_stmt(KronOp(2), [Const(A), Const(B)], [OutSpec("k2", (4, 4))], bulk=False)
    prog.emit_stmt(KronFreeOp(0, 0), [Const(A), Const(B)], [OutSpec("kf", (4, 4))], bulk=False)

    import polyarray.exact_fold as EF
    st = EF.exact_partial_eval(prog, time_budget=30.0)
    assert st.folded == {0, 1}, (st.folded, st.unresolved)   # constant in, constant out
    assert not st.unresolved, st.unresolved                  # and NOT opaque


def test_kron_twin_threads_symbolic_cells_rather_than_going_opaque() -> None:
    """With a SYMBOLIC operand the Kron is still executed exactly — *refuted* (provably
    non-constant), never *unresolved*.  Only unresolved statements reach the probe fallback, so
    this is the difference between an exact certificate and a probed one."""
    from polyarray.ir import ColStackOp, Const, KronOp, OutputRef

    prog = Program("kron_sym", inputs=[SymInput("v", (2,), _prov("v"))])
    prog.emit_stmt(ColStackOp(), [prog.input("v"), Const(np.array([1.0, 0.0]))],
                   [OutSpec("m", (2, 2))], bulk=False)
    prog.emit_stmt(KronOp(2), [OutputRef(0, 0), Const(np.eye(2))],
                   [OutSpec("k", (4, 4))], bulk=False)

    import polyarray.exact_fold as EF
    st = EF.exact_partial_eval(prog, time_budget=30.0)
    assert 1 not in st.unresolved, st.unresolved             # executed, not opaque
    assert 1 in st.refuted, (st.refuted, st.folded)          # and PROVABLY non-constant


def test_kron_twin_backs_off_on_a_shape_it_cannot_honour() -> None:
    """A non-matrix ``KronOp`` operand falls through to the opaque tail instead of building a
    wrong-shaped result.  Silent shape corruption here does NOT fail locally — it surfaces much
    later as an unrelated einsum's arity error, so the twin verifies before committing."""
    from polyarray.ir import Const, KronOp

    prog = Program("kron_bad", inputs=[SymInput("v", (2,), _prov("v"))])
    prog.emit_stmt(KronOp(2), [Const(np.array([1.0, 2.0])), Const(np.eye(2))],
                   [OutSpec("k", (4,))], bulk=False)

    import polyarray.exact_fold as EF
    st = EF.exact_partial_eval(prog, time_budget=30.0)
    # UNRESOLVED, never a wrong-shaped fold.  The reason text is deliberately not pinned: the
    # declared-OutSpec check may fire before the twin's own ndim guard, and either way the
    # statement lands in `unresolved` (⇒ the warned probe fallback), which is the contract.
    assert 0 in st.unresolved, (st.unresolved, st.folded)
    assert 0 not in st.folded, st.folded


# --- SwitchOp (`select_x`) — the minimal σ-switch fold -------------------------------------------

def _switch_program(equal: bool):
    """A `select_x` over a run-time-only `IntAtom`, with branches equal or not."""
    from polyarray.ir import IntAtomRef, OutSpec, Program, SwitchOp, SymArray
    prog = Program(name="switchtest")
    prog.declare_int_atom("o_0", range(2))
    b0 = SymArray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object), program=prog)
    other = [[1.0, 2.0], [3.0, 4.0]] if equal else [[9.0, 9.0], [9.0, 9.0]]
    b1 = SymArray(np.array(other, dtype=object), program=prog)
    prog.emit_stmt(SwitchOp(2), [IntAtomRef("o_0"), b0, b1],
                   [OutSpec("sw", (2, 2))], note="sigma-switch")
    return prog


def test_switch_with_equal_branches_folds_through_a_symbolic_scrutinee():
    """A `SwitchOp` whose branches all carry the SAME value is the identity — the run-time-only
    scrutinee cannot change the result, so the exact lane must fold it rather than freeze.

    This is the σ-no-op case, and it is what makes design item B viable: B moves the σ switch from
    the folded OUTPUT to the geometry INPUTS, i.e. upstream of the whole body. Before this fold the
    statement aborted on the unresolved scrutinee ("operand depends on an unresolved statement"),
    which would have put an unfoldable node ahead of every σ-carrying expression and degraded the
    FEEC lanes back to probe fallback — trading away the `QrOp` removal, not adding to it."""
    from polyarray.exact_fold import exact_partial_eval
    st = exact_partial_eval(_switch_program(equal=True), time_budget=10.0)
    assert sorted(st.folded) == [0], f"equal branches must fold; got {st.unresolved}"
    assert not st.unresolved


def test_switch_with_differing_branches_is_unresolved_not_guessed():
    """Branches that genuinely differ under a run-time scrutinee stay UNRESOLVED — the fold must not
    pick one. The reason names the switch, so the hybrid-mode warning localizes it instead of
    blaming the generic unresolved-operand path."""
    from polyarray.exact_fold import exact_partial_eval
    st = exact_partial_eval(_switch_program(equal=False), time_budget=10.0)
    assert not st.folded
    assert len(st.unresolved) == 1
    assert "SwitchOp" in next(iter(st.unresolved.values()))


def test_switch_with_a_constant_scrutinee_selects_the_right_branch():
    """A build-time-CONSTANT scrutinee selects its branch, and the selected value flows on EXACTLY.

    Distinct from the all-numeric case, which the numeric path (`_exec_fn` running the real
    `SwitchOp.__call__`) already handled: here the branches are SYMBOLIC, so resolution goes through
    the exact twin. Pinned per-branch rather than merely "it resolved" — resolving to the WRONG
    branch would also look like success.

    The outcome is `refuted`, not `folded`: the selected branch `x+1` is genuinely vertex-dependent,
    so the exact lane proves NON-constancy. That is the sound classification (soundly excluded from
    any probe fallback), and it is the evidence the value was carried through exactly rather than
    frozen."""
    from polyarray.ir import Const, OutSpec, Program, SwitchOp, SymArray, SymInput
    from polyarray.exact_fold import exact_partial_eval

    for k, expect in ((0, "x_0 + 1"), (1, "x_0 + 5")):
        prog = Program("switchconst", inputs=[SymInput("x", (1,), _prov("x"))])
        x = np.asarray(prog.input_arrays["x"].cells)[0]
        b0 = SymArray(np.array([x + 1], dtype=object), program=prog)
        b1 = SymArray(np.array([x + 5], dtype=object), program=prog)
        prog.emit_stmt(SwitchOp(2), [Const(np.array(k)), b0, b1],
                       [OutSpec("sw", (1,))], note="sigma-switch")

        seen: list[str] = []
        import polyarray.exact_fold as _ef
        orig = _ef._sym_apply

        def spy(fn, args, *a, _o=orig, _s=seen, **kw):
            r = _o(fn, args, *a, **kw)
            if isinstance(fn, SwitchOp) and r is not None:
                _s.extend(str(v) for v in np.asarray(r[0]).ravel())
            return r

        _ef._sym_apply = spy
        try:
            st = exact_partial_eval(prog, time_budget=10.0)
        finally:
            _ef._sym_apply = orig
        assert seen == [expect], f"scrutinee={k}: selected {seen}, expected [{expect!r}]"
        assert sorted(st.refuted) == [0], f"scrutinee={k}: expected a refutation, got {st.unresolved}"
        assert not st.unresolved
