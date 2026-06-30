"""Tests for :func:`polyarray.to_numpy_source`.

Each test builds a small :class:`Program` — directly via the polyarray API or
(easier) via grassmann's ``G.compile`` — emits standalone numpy source, ``exec``s
it, calls the generated function on random numpy inputs, and asserts the result
matches ``program.run(same_inputs)["result"]``.
"""
from __future__ import annotations

import numpy as np
import pytest

import polyarray as pa
from polyarray.ir import (
    Program,
    Provenance,
    SymInput,
    SymbolicBudget,
)

grassmann = pytest.importorskip("grassmann")
import grassmann as G  # noqa: E402


# ---------------------------------------------------------------------------
# Renderers for grassmann's own Stmt ops (kept out of polyarray itself).
# Demonstrates the ``op_renderers`` extension point.
# ---------------------------------------------------------------------------

def _const_op(op, a):
    val = np.frombuffer(op.data_bytes, dtype=op.dtype).reshape(op.shape)
    return f"np.array({val.tolist()!r}, dtype=np.{op.dtype})"


def _axis_len_op(op, a):
    return f"np.asarray(int(np.asarray({a[0]}).shape[{op.axis}]))"


def _scale_by_op(op, a):
    return f"(np.asarray({a[1]}, dtype=float) * np.asarray({a[0]}, dtype=float))"


def _external_op(op, a):
    xs = [f"np.asarray({e}, dtype=float)" for e in a]
    table = {
        "sqrt": f"np.sqrt({xs[0]})",
        "abs": f"np.abs({xs[0]})",
        "sign": f"np.sign({xs[0]})",
        "reciprocal": f"(1.0 / {xs[0]})",
    }
    if op.name == "div":
        return f"({xs[0]} / {xs[1]})"
    if op.name not in table:
        raise AssertionError(f"unhandled _ExternalOp name {op.name!r}")
    return table[op.name]


GRASSMANN_RENDERERS = {
    "_ConstOp": _const_op,
    "_AxisLenOp": _axis_len_op,
    "_ScaleByOp": _scale_by_op,
    "_ExternalOp": _external_op,
}


def _roundtrip(prog, values, func_name="f", op_renderers=None):
    """Emit, exec, call, and compare against ``prog.run``.  Returns the source."""
    src = pa.to_numpy_source(prog, func_name, op_renderers=op_renderers)
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)
    got = ns[func_name](*[values[i.name] for i in prog.inputs])
    want = prog.run(values)["result"]
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-9, atol=1e-12)
    return src


def _prov(label="A"):
    return Provenance(kind="coeff", origin="t", index=(), label=label)


# ---------------------------------------------------------------------------
# grassmann-lowered programs
# ---------------------------------------------------------------------------

def test_worked_example_matvec():
    """The README worked example: y = f(x), f : V ⊸ W."""
    V, W = G.space("V", 2), G.space("W", 3)
    f = G.var("f", V >> W)
    x = G.var("x", V)
    prog = G.compile(f(x))
    fname, xname = prog.inputs[0].name, prog.inputs[1].name
    fv = np.random.rand(2, 3)
    xv = np.random.rand(2)
    src = _roundtrip(
        prog, {fname: fv, xname: xv},
        func_name="fApply", op_renderers=GRASSMANN_RENDERERS,
    )
    assert "import numpy as np" in src
    assert src.startswith("import numpy as np")


def test_inner_product():
    """G.dot(x, z) — exercises a _ConstOp + EinsumStmtOp, bulk scalar output."""
    Vsp = G.space("V", 4)
    x, z = G.var("x", Vsp), G.var("z", Vsp)
    prog = G.compile(G.dot(x, z))
    xn, zn = prog.inputs[0].name, prog.inputs[1].name
    _roundtrip(
        prog, {xn: np.random.rand(4), zn: np.random.rand(4)},
        func_name="fDot", op_renderers=GRASSMANN_RENDERERS,
    )


def test_normalize_nonlinear():
    """Normalize a vector: scale(reciprocal(sqrt(dot(x, x))), x).

    Exercises the nonlinear scalar ops (sqrt/reciprocal) and a scale.
    """
    Vsp = G.space("V", 3)
    x = G.var("x", Vsp)
    prog = G.compile(G.scale(G.reciprocal(G.sqrt(G.dot(x, x))), x))
    xn = prog.inputs[0].name
    _roundtrip(
        prog, {xn: np.array([3.0, 4.0, 12.0])},
        func_name="fNorm", op_renderers=GRASSMANN_RENDERERS,
    )


# ---------------------------------------------------------------------------
# directly-built polyarray programs (pure-rational lane, no front-end ops)
# ---------------------------------------------------------------------------

def test_det_closed_form():
    """3x3 determinant as a closed-form rational expression (no statements)."""
    prog = Program("det3", inputs=[SymInput("A", (3, 3), _prov())])
    d = prog.input("A").det()
    prog.add_output("result", d.cells)
    assert len(prog.statements) == 0  # pure rational lane
    _roundtrip(prog, {"A": np.random.rand(3, 3)}, func_name="fDet")


def test_inverse_closed_form_denominators():
    """2x2 closed-form inverse — rational cells WITH denominators."""
    prog = Program("inv2", inputs=[SymInput("A", (2, 2), _prov())])
    inv = prog.input("A").inverse()
    prog.add_output("result", inv.cells)
    assert len(prog.statements) == 0
    A = np.array([[2.0, 1.0], [1.0, 3.0]])  # invertible
    src = _roundtrip(prog, {"A": A}, func_name="fInv")
    assert "/" in src  # denominator path was exercised


def test_scalar_sqrt_stmt():
    """SqrtOp on a symbolic scalar emits a 0-d Stmt."""
    prog = Program("sq", inputs=[SymInput("s", (), _prov("s"))])
    out = prog.input("s").sqrt()
    prog.add_output("result", out.cells)
    assert [type(s.fn).__name__ for s in prog.statements] == ["SqrtOp"]
    _roundtrip(prog, {"s": 7.0}, func_name="fSqrt")


# ---------------------------------------------------------------------------
# over-budget linalg: typed Stmt ops (DetOp / InvOp) + bulk / per-cell outputs
# ---------------------------------------------------------------------------

def test_over_budget_det_stmt():
    """force_stmts routes det through a DetOp Stmt (+ IdentityOp freeze)."""
    budget = SymbolicBudget.force_stmts()
    prog = Program("detS", inputs=[SymInput("A", (3, 3), _prov())], budget=budget)
    d = prog.input("A").det()
    prog.add_output("result", d.cells)
    names = [type(s.fn).__name__ for s in prog.statements]
    assert "DetOp" in names
    _roundtrip(prog, {"A": np.random.rand(3, 3)}, func_name="fDetStmt")


def test_over_budget_inverse_stmt():
    """force_stmts routes inverse through an InvOp Stmt with a bulk (n,n) output."""
    budget = SymbolicBudget.force_stmts()
    prog = Program("invS", inputs=[SymInput("A", (4, 4), _prov())], budget=budget)
    inv = prog.input("A").inverse()
    prog.add_output("result", inv.cells)
    names = [type(s.fn).__name__ for s in prog.statements]
    assert "InvOp" in names
    A = np.random.rand(4, 4) + 4 * np.eye(4)  # well-conditioned
    _roundtrip(prog, {"A": A}, func_name="fInvStmt")


def test_solve_stmt():
    """A.solve(b) emits a SolveOp Stmt (always defers for symbolic operands)."""
    prog = Program(
        "solve",
        inputs=[SymInput("A", (3, 3), _prov("A")), SymInput("b", (3,), _prov("b"))],
    )
    x = prog.input("A").solve(prog.input("b"))
    prog.add_output("result", x.cells)
    assert "SolveOp" in [type(s.fn).__name__ for s in prog.statements]
    A = np.random.rand(3, 3) + 3 * np.eye(3)
    _roundtrip(prog, {"A": A, "b": np.random.rand(3)}, func_name="fSolve")


# ---------------------------------------------------------------------------
# unsupported op -> clear NotImplementedError
# ---------------------------------------------------------------------------

def test_unsupported_op_raises():
    from polyarray.ir import OutSpec

    class _MysteryOp:
        def __call__(self, x):
            return np.asarray(x)

    prog = Program("mystery", inputs=[SymInput("A", (2, 2), _prov())])
    (out,) = prog.emit_stmt(_MysteryOp(), [prog.input("A")], [OutSpec("m", (2, 2))])
    prog.add_output("result", out.cells)
    with pytest.raises(NotImplementedError, match="_MysteryOp"):
        pa.to_numpy_source(prog)


def test_grassmann_derivative_callop_emits_and_matches():
    """A grassmann derivative program lowers through a CallOp/sub-program (vmap full
    Jacobian); to_numpy_source must emit it and match Program.run (gap G12)."""
    import inspect

    import grassmann as G
    from grassmann.lower.numpy_renderers import op_renderers
    from grassmann.syntax.mcalc import build_problem

    for expr, decls, dims, wrt in [
        ("0.5*x'*A*x", {"x": "vector", "A": "matrix"}, {"n1": 3}, "x"),
        ("sin(x)'*y", {"x": "vector", "y": "vector"}, {"n0": 3}, "x"),
        ("A*exp(x)", {"A": "matrix", "x": "vector"}, {"n0": 3, "n1": 4}, "x"),
    ]:
        r = build_problem(expr, wrt=wrt, decls=decls)
        dprog = G.compile(r.instantiate(r.derivative, dims))
        src = pa.to_numpy_source(dprog, func_name="df", op_renderers=op_renderers())
        ns = {}
        exec(src, ns)  # noqa: S102
        params = list(inspect.signature(ns["df"]).parameters)
        rng = np.random.default_rng(0)
        shp = {si.name: tuple(int(d) for d in si.shape) for si in dprog.inputs}
        vals = {p: rng.standard_normal(shp[p]) for p in params}
        got = np.asarray(ns["df"](*[vals[p] for p in params]))
        ref = np.asarray(dprog.run(vals)["result"])
        assert np.allclose(got, ref), f"{expr}: emitted derivative != run()"
