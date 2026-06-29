"""P6 exactness tests: partial descent into nested sub-Program / CallOp bodies.

When a sub-Program / ``CallOp`` Stmt has SOME (not all) numeric operands, the
pass recurses into the body, binds the numeric operands to the body's inputs,
and replaces the Stmt's ``fn`` with the specialized (smaller) body — feeding it
only the still-symbolic operands.  Every test asserts the rebuilt program is
exact vs the original on random inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import (
    CallOp,
    DetOp,
    OutSpec,
    Program,
    Provenance,
    SymInput,
    TensordotOp,
    bind_inputs,
)


def _prov(name: str):
    return Provenance("vertex", name, (), name)


def _td(axes=([1], [0])):
    return TensordotOp.from_axes(axes)


# ---------------------------------------------------------------------------
# sub-Program Stmt with TWO inputs, bind ONE of the two that flow in
# ---------------------------------------------------------------------------

def test_subprogram_partial_descent_drops_one_input() -> None:
    # body(x, y) = (x @ y) @ y   — two inputs.
    body = Program(
        "body",
        inputs=[SymInput("x", (2, 2), _prov("x")), SymInput("y", (2, 2), _prov("y"))],
    )
    [xy] = body.emit_stmt(
        _td(), [body.input("x"), body.input("y")], [OutSpec("xy", (2, 2))], bulk=False
    )
    [xyy] = body.emit_stmt(
        _td(), [xy, body.input("y")], [OutSpec("xyy", (2, 2))], bulk=False
    )
    body.add_output("xyy", xyy.cells)

    top = Program(
        "top",
        inputs=[SymInput("P", (2, 2), _prov("P")), SymInput("Q", (2, 2), _prov("Q"))],
    )
    [out] = top.emit_stmt(
        body, [top.input("P"), top.input("Q")], [OutSpec("r", (2, 2))], bulk=False
    )
    top.add_output("r", out.cells)

    rng = np.random.default_rng(0)
    P = rng.standard_normal((2, 2))
    Q = rng.standard_normal((2, 2))

    # Bind P numeric; Q stays symbolic -> descend, body's "x" input is dropped.
    folded = bind_inputs(top, {"P": P})
    assert len(folded.statements) == 1
    surviving = folded.statements[0].fn
    assert isinstance(surviving, Program)
    # The specialized body has FEWER inputs than the original body (x dropped).
    assert len(surviving.inputs) < len(body.inputs)
    assert [inp.name for inp in surviving.inputs] == ["y"]
    # The Stmt now feeds only the still-symbolic operand (Q).
    assert len(folded.statements[0].in_) == 1

    got = folded.run({"Q": Q})["r"]
    want = top.run({"P": P, "Q": Q})["r"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_subprogram_partial_descent_bind_the_other_input() -> None:
    # Same body, but bind Q (-> body "y") instead.  "y" feeds both stmts, so
    # the specialized body keeps only "x".
    body = Program(
        "body",
        inputs=[SymInput("x", (2, 2), _prov("x")), SymInput("y", (2, 2), _prov("y"))],
    )
    [xy] = body.emit_stmt(
        _td(), [body.input("x"), body.input("y")], [OutSpec("xy", (2, 2))], bulk=False
    )
    [xyy] = body.emit_stmt(
        _td(), [xy, body.input("y")], [OutSpec("xyy", (2, 2))], bulk=False
    )
    body.add_output("xyy", xyy.cells)

    top = Program(
        "top",
        inputs=[SymInput("P", (2, 2), _prov("P")), SymInput("Q", (2, 2), _prov("Q"))],
    )
    [out] = top.emit_stmt(
        body, [top.input("P"), top.input("Q")], [OutSpec("r", (2, 2))], bulk=False
    )
    top.add_output("r", out.cells)

    rng = np.random.default_rng(1)
    P = rng.standard_normal((2, 2))
    Q = rng.standard_normal((2, 2))

    folded = bind_inputs(top, {"Q": Q})
    surviving = folded.statements[0].fn
    assert isinstance(surviving, Program)
    assert [inp.name for inp in surviving.inputs] == ["x"]
    got = folded.run({"P": P})["r"]
    want = top.run({"P": P, "Q": Q})["r"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# CallOp(fn=sub-Program) — partial descent through the typed call node
# ---------------------------------------------------------------------------

def test_callop_subprogram_partial_descent() -> None:
    # body(x, y) = det(x @ y)  (scalar out); bind x, keep y.
    body = Program(
        "body",
        inputs=[SymInput("x", (2, 2), _prov("x")), SymInput("y", (2, 2), _prov("y"))],
    )
    [xy] = body.emit_stmt(
        _td(), [body.input("x"), body.input("y")], [OutSpec("xy", (2, 2))], bulk=False
    )
    [d] = body.emit_stmt(DetOp(), [xy], [OutSpec("d", ())], bulk=False)
    body.add_output("d", d.cells)

    top = Program(
        "top",
        inputs=[SymInput("P", (2, 2), _prov("P")), SymInput("Q", (2, 2), _prov("Q"))],
    )
    [out] = top.emit_stmt(
        CallOp(fn=body), [top.input("P"), top.input("Q")], [OutSpec("r", ())], bulk=False
    )
    top.add_output("r", out.cells)

    rng = np.random.default_rng(2)
    P = rng.standard_normal((2, 2))
    Q = rng.standard_normal((2, 2))

    folded = bind_inputs(top, {"P": P})
    assert len(folded.statements) == 1
    surviving = folded.statements[0].fn
    # The descended fn stays a CallOp (typed wrapper preserved) over a smaller body.
    assert isinstance(surviving, CallOp)
    assert isinstance(surviving.fn, Program)
    assert [inp.name for inp in surviving.fn.inputs] == ["y"]
    assert len(folded.statements[0].in_) == 1

    got = folded.run({"Q": Q})["r"]
    want = top.run({"P": P, "Q": Q})["r"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# Nested two-levels-deep body, partial bind at the top, exactness holds
# ---------------------------------------------------------------------------

def test_nested_two_levels_partial_descent() -> None:
    # inner(a, b) = a @ b
    inner = Program(
        "inner",
        inputs=[SymInput("a", (2, 2), _prov("a")), SymInput("b", (2, 2), _prov("b"))],
    )
    [ab] = inner.emit_stmt(
        _td(), [inner.input("a"), inner.input("b")], [OutSpec("ab", (2, 2))], bulk=False
    )
    inner.add_output("ab", ab.cells)

    # mid(u, v) = inner(u, v) @ v  -> contains a sub-Program Stmt itself.
    mid = Program(
        "mid",
        inputs=[SymInput("u", (2, 2), _prov("u")), SymInput("v", (2, 2), _prov("v"))],
    )
    [iv] = mid.emit_stmt(
        inner, [mid.input("u"), mid.input("v")], [OutSpec("iv", (2, 2))], bulk=False
    )
    [ivv] = mid.emit_stmt(
        _td(), [iv, mid.input("v")], [OutSpec("ivv", (2, 2))], bulk=False
    )
    mid.add_output("ivv", ivv.cells)

    # top(P, Q) = mid(P, Q)
    top = Program(
        "top",
        inputs=[SymInput("P", (2, 2), _prov("P")), SymInput("Q", (2, 2), _prov("Q"))],
    )
    [out] = top.emit_stmt(
        mid, [top.input("P"), top.input("Q")], [OutSpec("r", (2, 2))], bulk=False
    )
    top.add_output("r", out.cells)

    rng = np.random.default_rng(3)
    P = rng.standard_normal((2, 2))
    Q = rng.standard_normal((2, 2))

    # Bind P: descend into mid (drops u), and mid's inner Stmt now has u bound
    # numeric + v symbolic -> recursive descent drops inner's "a".
    folded = bind_inputs(top, {"P": P})
    assert len(folded.statements) == 1
    spec_mid = folded.statements[0].fn
    assert isinstance(spec_mid, Program)
    assert [inp.name for inp in spec_mid.inputs] == ["v"]
    # The nested inner Stmt inside the specialized mid also shrank to one input.
    inner_stmts = [s for s in spec_mid.statements if isinstance(s.fn, Program)]
    assert inner_stmts, "expected the nested sub-Program Stmt to survive"
    assert len(inner_stmts[0].fn.inputs) == 1

    got = folded.run({"Q": Q})["r"]
    want = top.run({"P": P, "Q": Q})["r"]
    np.testing.assert_allclose(got, want, rtol=1e-9)


# ---------------------------------------------------------------------------
# Regression: NONE-numeric sub-Program Stmt stays fully symbolic (no descent)
# ---------------------------------------------------------------------------

def test_subprogram_no_numeric_stays_symbolic() -> None:
    body = Program(
        "body",
        inputs=[SymInput("x", (2, 2), _prov("x")), SymInput("y", (2, 2), _prov("y"))],
    )
    [xy] = body.emit_stmt(
        _td(), [body.input("x"), body.input("y")], [OutSpec("xy", (2, 2))], bulk=False
    )
    body.add_output("xy", xy.cells)

    top = Program(
        "top",
        inputs=[SymInput("P", (2, 2), _prov("P")), SymInput("Q", (2, 2), _prov("Q"))],
    )
    [out] = top.emit_stmt(
        body, [top.input("P"), top.input("Q")], [OutSpec("r", (2, 2))], bulk=False
    )
    top.add_output("r", out.cells)

    # No bind at all -> nothing numeric -> body untouched (still 2 inputs).
    from polyarray import fold_numeric

    folded = fold_numeric(top)
    assert len(folded.statements) == 1
    assert folded.statements[0].fn is body  # fn unchanged (not re-specialized)

    rng = np.random.default_rng(4)
    P = rng.standard_normal((2, 2))
    Q = rng.standard_normal((2, 2))
    np.testing.assert_allclose(
        folded.run({"P": P, "Q": Q})["r"], top.run({"P": P, "Q": Q})["r"], rtol=1e-9
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
