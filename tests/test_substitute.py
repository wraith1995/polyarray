"""P3 — symbolic argument substitution (`compose` + `substitute`)."""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import (
    OutSpec,
    Program,
    Provenance,
    SymInput,
    TensordotOp,
    bind_inputs,
    specialize,
    substitute,
)
from polyarray.ir import RationalRef, SymArray
from polyarray.rational import RationalFunction as RF


def _prov(name: str):
    return Provenance("vertex", name, (), name)


# ---------------------------------------------------------------------------
# RationalFunction.compose — unit
# ---------------------------------------------------------------------------

def test_compose_linear() -> None:
    x, y = RF.atom("x"), RF.atom("y")
    assert (x + 1).compose("x", y) == y + 1


def test_compose_multi_gen_repl() -> None:
    x, a, b = RF.atom("x"), RF.atom("a"), RF.atom("b")
    got = (x * x + x).compose("x", a + b)
    want = (a + b) * (a + b) + (a + b)
    assert got == want
    assert set(got.gens) == {"a", "b"}


def test_compose_absent_name_is_noop() -> None:
    x, y = RF.atom("x"), RF.atom("y")
    assert (x + 1).compose("z", y) == x + 1


def test_compose_denominator() -> None:
    x, y = RF.atom("x"), RF.atom("y")
    r = (RF.constant(1) / x).compose("x", y + 1)  # -> 1/(y+1)
    assert r.eval({"y": 3.0}) == pytest.approx(0.25)
    assert r.eval({"y": 0.0}) == pytest.approx(1.0)


def test_compose_to_constant() -> None:
    x = RF.atom("x")
    r = (x + 1).compose("x", RF.constant(5))
    assert r.eval({}) == pytest.approx(6.0)


def test_compose_multi() -> None:
    x, y, a = RF.atom("x"), RF.atom("y"), RF.atom("a")
    got = (x + y).compose_multi({"x": a + 1, "y": a - 1})
    assert got == (a + 1) + (a - 1)  # == 2a


# ---------------------------------------------------------------------------
# substitute — elementwise out = A + B, substitute B -> expr(A)
# ---------------------------------------------------------------------------

def _elementwise_add(n: int) -> Program:
    prog = Program(
        "add",
        inputs=[SymInput("A", (n,), _prov("A")), SymInput("B", (n,), _prov("B"))],
    )
    A = np.asarray(prog.input("A").cells)
    B = np.asarray(prog.input("B").cells)
    prog.add_output("out", A + B)
    return prog


def test_substitute_elementwise_drops_input() -> None:
    n = 4
    prog = _elementwise_add(n)
    A = np.asarray(prog.input("A").cells)
    expr = SymArray(np.array([a * a + 1 for a in A], dtype=object))

    sub = substitute(prog, {"B": expr})
    names = {inp.name for inp in sub.inputs}
    assert "B" not in names and "A" in names

    rng = np.random.default_rng(0)
    Av = rng.standard_normal(n)
    Bval = Av * Av + 1.0
    got = sub.run({"A": Av})["out"]
    want = prog.run({"A": Av, "B": Bval})["out"]
    np.testing.assert_allclose(got, want, rtol=1e-9)
    np.testing.assert_allclose(got, Av + Bval, rtol=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_substitute_exactness_seeds(seed: int) -> None:
    n = 5
    prog = _elementwise_add(n)
    A = np.asarray(prog.input("A").cells)
    # A nonlinear, denominator-bearing expression over A.
    expr = SymArray(np.array([(2 * a - 1) / (a * a + 1) for a in A], dtype=object))
    sub = substitute(prog, {"B": expr})

    rng = np.random.default_rng(seed)
    Av = rng.standard_normal(n)
    Bval = (2 * Av - 1) / (Av * Av + 1)
    got = sub.run({"A": Av})["out"]
    want = prog.run({"A": Av, "B": Bval})["out"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# substitute through a Stmt (SymArrayRef path): out = A @ B
# ---------------------------------------------------------------------------

def test_substitute_matmul_via_stmt() -> None:
    m, k, n = 2, 3, 2
    prog = Program(
        "mm",
        inputs=[SymInput("A", (m, k), _prov("A")), SymInput("B", (k, n), _prov("B"))],
    )
    [C] = prog.emit_stmt(
        TensordotOp.from_axes(([1], [0])),
        [prog.input("A"), prog.input("B")],
        [OutSpec("C", (m, n))],
        bulk=False,
    )
    prog.add_output("C", C.cells)

    A = np.asarray(prog.input("A").cells)  # (m, k) atoms
    # B[i,j] = 2*A[i, j] + 1   (shape (k, n) == (3, 2); reuse A[:k, :n] atoms)
    expr_cells = np.array(
        [[2 * A[i % m, j % k] + 1 for j in range(n)] for i in range(k)], dtype=object
    )
    expr = SymArray(expr_cells)

    sub = substitute(prog, {"B": expr})
    assert "B" not in {inp.name for inp in sub.inputs}

    rng = np.random.default_rng(11)
    Av = rng.standard_normal((m, k))
    Bval = np.array(
        [[2 * Av[i % m, j % k] + 1 for j in range(n)] for i in range(k)], dtype=float
    )
    got = sub.run({"A": Av})["C"]
    want = prog.run({"A": Av, "B": Bval})["C"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# substituted expr depends on ANOTHER still-symbolic input -> stays symbolic;
# then bind the remaining input -> pure number (compose + fold compose well)
# ---------------------------------------------------------------------------

def test_substitute_keeps_other_input_symbolic_then_bind() -> None:
    n = 3
    prog = Program(
        "addc",
        inputs=[
            SymInput("A", (n,), _prov("A")),
            SymInput("B", (n,), _prov("B")),
            SymInput("C", (n,), _prov("C")),
        ],
    )
    A = np.asarray(prog.input("A").cells)
    B = np.asarray(prog.input("B").cells)
    Cc = np.asarray(prog.input("C").cells)
    prog.add_output("out", A + B)  # note: C unused by output, only via subs

    # B -> A + C : depends on a still-symbolic input C.
    expr = SymArray(np.array([a + c for a, c in zip(A, Cc)], dtype=object))
    sub = substitute(prog, {"B": expr})
    names = {inp.name for inp in sub.inputs}
    assert "B" not in names and "A" in names and "C" in names

    rng = np.random.default_rng(7)
    Av = rng.standard_normal(n)
    Cv = rng.standard_normal(n)
    got = sub.run({"A": Av, "C": Cv})["out"]
    want = prog.run({"A": Av, "B": Av + Cv, "C": Cv})["out"]
    np.testing.assert_allclose(got, want, rtol=1e-9)

    # Now bind C numerically through specialize(subs=..., bind=...) ordering.
    sub2 = specialize(prog, subs={"B": expr}, bind={"C": Cv})
    assert "C" not in {inp.name for inp in sub2.inputs}
    got2 = sub2.run({"A": Av})["out"]
    np.testing.assert_allclose(got2, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# RationalRef input path is composed
# ---------------------------------------------------------------------------

def test_substitute_rationalref_path() -> None:
    n = 2
    prog = Program(
        "rr",
        inputs=[SymInput("A", (n,), _prov("A")), SymInput("B", (n,), _prov("B"))],
    )
    A = np.asarray(prog.input("A").cells)
    B = np.asarray(prog.input("B").cells)
    # A Stmt consuming a RationalRef (A[0] + B[0]) and emitting a scalar copy.
    rf = A[0] + B[0]
    [out] = prog.emit_stmt(
        lambda v: np.asarray(v),
        [RationalRef(rf)],
        [OutSpec("o", ())],
        bulk=False,
    )
    prog.add_output("o", out.cells)
    # confirm we actually emitted a RationalRef
    assert isinstance(prog.statements[0].in_[0], RationalRef)

    expr = SymArray(np.array([a * 3 for a in A], dtype=object))
    sub = substitute(prog, {"B": expr})
    assert "B" not in {inp.name for inp in sub.inputs}

    rng = np.random.default_rng(3)
    Av = rng.standard_normal(n)
    Bval = Av * 3
    got = sub.run({"A": Av})["o"]
    want = prog.run({"A": Av, "B": Bval})["o"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# the pass never disturbs the original program
# ---------------------------------------------------------------------------

def test_substitute_original_untouched() -> None:
    prog = _elementwise_add(3)
    A = np.asarray(prog.input("A").cells)
    expr = SymArray(np.array([a + 1 for a in A], dtype=object))
    _ = substitute(prog, {"B": expr})
    assert {inp.name for inp in prog.inputs} == {"A", "B"}
    rng = np.random.default_rng(0)
    Av, Bv = rng.standard_normal(3), rng.standard_normal(3)
    np.testing.assert_allclose(
        prog.run({"A": Av, "B": Bv})["out"], Av + Bv, rtol=1e-9
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
