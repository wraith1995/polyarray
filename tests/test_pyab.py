"""Tests for :mod:`polyarray.pyab` — lowering a Program to PyArrayBackend IR.

Ground truth is :meth:`Program.run`: for each constructed program we compile
through PyAB's numpy backend (and, when torch is available, its torch backend)
and assert the compiled function matches ``program.run`` on random inputs.

Small-QR interception is additionally checked *directly* against
``numpy.linalg.qr`` (sign convention included), and the ``place=`` call-site API
(plain / vmap / fuse) is exercised by embedding a lowered kernel inside a larger
PyAB program.  The suite skips cleanly when ``pyarraybackend`` (or torch) is not
installed — ``polyarray`` itself never depends on either.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import itertools

from polyarray.ir import (
    EinsumStmtOp,
    GSvdOp,
    MoveaxisOp,
    OutSpec,
    Program,
    Provenance,
    QrOp,
    SvdOp,
    SymbolicBudget,
    SymInput,
    TensordotOp,
    WhileOp,
    vmap,
)

pytest.importorskip("pyarraybackend")
from polyarray import pyab  # noqa: E402


def _prov(label="A"):
    return Provenance(kind="coeff", origin="t", index=(), label=label)


def _outputs_as_list(prog, values):
    want = prog.run(values)
    return [np.asarray(want[k]) for k in prog.outputs]


def _args(prog, values):
    return [values[i.name] for i in prog.inputs] + [values[a] for a in prog.int_atoms]


def _check_numpy(prog, values, label=""):
    cm = pyab.compile_numpy(prog, name="f")
    got = cm.module.f(*_args(prog, values))
    want = _outputs_as_list(prog, values)
    got_list = [np.asarray(got)] if len(want) == 1 else [np.asarray(g) for g in got]
    for g, w in zip(got_list, want):
        np.testing.assert_allclose(g, w, rtol=1e-8, atol=1e-10, err_msg=label)
    return cm


def _check_torch(prog, values, label=""):
    torch = pytest.importorskip("torch")
    cm = pyab.compile_torch(prog, name="f")
    targs = [
        torch.as_tensor(np.asarray(a, dtype=float)) if not np.isscalar(a) else a
        for a in _args(prog, values)
    ]
    got = cm.module.f(*targs)
    want = _outputs_as_list(prog, values)
    got_list = [np.asarray(got)] if len(want) == 1 else [np.asarray(g) for g in got]
    for g, w in zip(got_list, want):
        np.testing.assert_allclose(g, w, rtol=1e-7, atol=1e-8, err_msg=label)
    return cm


# ---------------------------------------------------------------------------
# rational cell lane (no Stmts) + op lane
# ---------------------------------------------------------------------------

def test_det_closed_form_cell_lane():
    p = Program("det3", inputs=[SymInput("A", (3, 3), _prov())])
    p.add_output("result", p.input("A").det().cells)
    assert not p.statements
    _check_numpy(p, {"A": np.random.rand(3, 3)}, "det3 cell lane")


def test_inverse_cell_lane_with_division():
    p = Program("inv2", inputs=[SymInput("A", (2, 2), _prov())])
    p.add_output("result", p.input("A").inverse().cells)
    src = pyab.compile_numpy(p, name="f").source
    assert "/" in src  # denominator path exercised
    _check_numpy(p, {"A": np.array([[2.0, 1.0], [1.0, 3.0]])}, "inv2 division")


@pytest.mark.parametrize("op", ["det", "inverse", "solve"])
def test_over_budget_linalg_stmts(op):
    budget = SymbolicBudget.force_stmts()
    if op == "solve":
        p = Program(
            "s",
            inputs=[SymInput("A", (3, 3), _prov("A")), SymInput("b", (3,), _prov("b"))],
            budget=budget,
        )
        p.add_output("result", p.input("A").solve(p.input("b")).cells)
        vals = {"A": np.random.rand(3, 3) + 3 * np.eye(3), "b": np.random.rand(3)}
    else:
        p = Program("m", inputs=[SymInput("A", (4, 4), _prov())], budget=budget)
        out = getattr(p.input("A"), op)()
        p.add_output("result", out.cells)
        vals = {"A": np.random.rand(4, 4) + 4 * np.eye(4)}
    _check_numpy(p, vals, f"{op} stmt")


def test_einsum_tensordot_moveaxis():
    p = Program("es", inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))])
    (out,) = p.emit_stmt(EinsumStmtOp(spec="ij,jk->ik"), [p.input("A"), p.input("B")], [OutSpec("C", (2, 4))])
    p.add_output("result", out.cells)
    _check_numpy(p, {"A": np.random.rand(2, 3), "B": np.random.rand(3, 4)}, "einsum")

    p = Program("td", inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 4), _prov("B"))])
    (out,) = p.emit_stmt(TensordotOp.from_axes(1), [p.input("A"), p.input("B")], [OutSpec("C", (2, 4))])
    p.add_output("result", out.cells)
    _check_numpy(p, {"A": np.random.rand(2, 3), "B": np.random.rand(3, 4)}, "tensordot")

    p = Program("mv", inputs=[SymInput("A", (2, 3, 4), _prov())])
    (out,) = p.emit_stmt(MoveaxisOp.from_spec(0, 2), [p.input("A")], [OutSpec("C", (3, 4, 2))])
    p.add_output("result", out.cells)
    _check_numpy(p, {"A": np.random.rand(2, 3, 4)}, "moveaxis")


def test_vmap_batched_det():
    body = Program("det2body", inputs=[SymInput("M", (2, 2), _prov())])
    body.add_output("result", body.input("M").det().cells)
    p = Program("vdet", inputs=[SymInput("batch", (5, 2, 2), _prov())])
    (out,) = p.emit_stmt(vmap(body, in_axes=(0,), out_axes=0), [p.input("batch")], [OutSpec("d", (5,))])
    p.add_output("result", out.cells)
    _check_numpy(p, {"batch": np.random.rand(5, 2, 2)}, "vmap det")


# ---------------------------------------------------------------------------
# small-QR interception (Householder) vs LAPACK sign convention
# ---------------------------------------------------------------------------

def _qr_program(m, n, mode="reduced"):
    p = Program(f"qr{m}{n}", inputs=[SymInput("A", (m, n), _prov())])
    q_shape = (m, n) if mode == "reduced" else (m, m)
    r_shape = (n, n) if mode == "reduced" else (m, n)
    Q, R = p.emit_stmt(QrOp(mode=mode), [p.input("A")], [OutSpec("Q", q_shape), OutSpec("R", r_shape)])
    p.add_output("Q", Q.cells)
    p.add_output("R", R.cells)
    return p


@pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4), (4, 3), (3, 2)])
def test_small_qr_householder_matches_lapack(m, n):
    p = _qr_program(m, n)
    # The small path emits scalar Householder — no linalg.qr call.
    assert "linalg.qr" not in pyab.compile_numpy(p, name="f").source
    A = np.random.default_rng(m * 10 + n).standard_normal((m, n))
    cm = pyab.compile_numpy(p, name="f")
    Q, R = cm.module.f(A)
    Qn, Rn = np.linalg.qr(A)  # LAPACK reduced QR
    np.testing.assert_allclose(np.asarray(Q), Qn, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(np.asarray(R), Rn, rtol=1e-9, atol=1e-11)
    # reconstruction A = Q R
    np.testing.assert_allclose(np.asarray(Q) @ np.asarray(R), A, rtol=1e-9, atol=1e-11)


def test_large_qr_falls_back_to_linalg():
    p = _qr_program(6, 5)
    assert "linalg.qr" in pyab.compile_numpy(p, name="f").source
    _check_numpy(p, {"A": np.random.default_rng(0).standard_normal((6, 5))}, "qr fallback")


def test_small_qr_threshold_configurable():
    p = _qr_program(3, 3)
    opts = pyab.LowerOpts(target="numpy", small_qr=pyab.SmallQrOpts(max_dim=2))
    # max_dim=2 < 3 -> fall back to linalg.qr even for 3x3
    assert "linalg.qr" in pyab.compile_numpy(p, name="f", opts=opts).source
    # disabled -> always linalg.qr
    opts0 = pyab.LowerOpts(target="numpy", small_qr=pyab.SmallQrOpts(enabled=False))
    assert "linalg.qr" in pyab.compile_numpy(p, name="f", opts=opts0).source


# ---------------------------------------------------------------------------
# torch backend
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,n", [(3, 3), (4, 3)])
def test_torch_small_qr(m, n):
    _check_torch(_qr_program(m, n), {"A": np.random.default_rng(m).standard_normal((m, n))}, "torch qr")


def test_torch_cell_lane_and_einsum():
    p = Program("d", inputs=[SymInput("A", (3, 3), _prov())])
    p.add_output("result", p.input("A").det().cells)
    _check_torch(p, {"A": np.random.rand(3, 3)}, "torch det cell lane")


# ---------------------------------------------------------------------------
# SVD / GSVD (composite; data-dependent rank runs eager) + WhileOp + nested vmap
# ---------------------------------------------------------------------------

def test_svd_numpy_exact_and_torch_reconstruction():
    p = Program("svd", inputs=[SymInput("A", (4, 3), _prov())])
    U, S, Vh, rank = p.emit_stmt(
        SvdOp(full_matrices=False), [p.input("A")],
        [OutSpec("U", (4, 3)), OutSpec("S", (3,)), OutSpec("Vh", (3, 3)), OutSpec("rank", ())],
    )
    for nm, sa in [("U", U), ("S", S), ("Vh", Vh), ("rank", rank)]:
        p.add_output(nm, sa.cells)
    A = np.random.default_rng(0).standard_normal((4, 3))
    _check_numpy(p, {"A": A}, "SVD numpy-exact")  # same LAPACK -> signs match

    torch = pytest.importorskip("torch")
    Ut, St, Vht, rt = (np.asarray(x) for x in pyab.compile_torch(p, name="f").module.f(torch.as_tensor(A)))
    np.testing.assert_allclose(Ut @ np.diag(St) @ Vht, A, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.sort(St), np.sort(np.linalg.svd(A, compute_uv=False)), rtol=1e-6)
    assert int(rt) == 3


def test_gsvd_numpy_exact_and_torch_reconstruction():
    nW, nV = 4, 3
    p = Program("gsvd", inputs=[
        SymInput("A", (nW, nV), _prov("A")),
        SymInput("MV", (nV, nV), _prov("MV")),
        SymInput("MW", (nW, nW), _prov("MW")),
    ])
    U, UI, V, VI, S, rank = p.emit_stmt(
        GSvdOp(), [p.input("A"), p.input("MV"), p.input("MW")],
        [OutSpec("U", (nW, nV)), OutSpec("UI", (nW, nW - nV)), OutSpec("V", (nV, nV)),
         OutSpec("VI", (nV, 0)), OutSpec("S", (nV,)), OutSpec("rank", ())],
    )
    for nm, sa in [("U", U), ("UI", UI), ("V", V), ("VI", VI), ("S", S), ("rank", rank)]:
        p.add_output(nm, sa.cells)
    rng = np.random.default_rng(1)

    def spd(n):
        M = rng.standard_normal((n, n))
        return M @ M.T + n * np.eye(n)

    vals = {"A": rng.standard_normal((nW, nV)), "MV": spd(nV), "MW": spd(nW)}
    _check_numpy(p, vals, "GSVD numpy-exact (incl. empty ker block)")

    torch = pytest.importorskip("torch")
    Ug, UIg, Vg, VIg, Sg, rg = (
        np.asarray(x)
        for x in pyab.compile_torch(p, name="f").module.f(
            torch.as_tensor(vals["A"]), torch.as_tensor(vals["MV"]), torch.as_tensor(vals["MW"])
        )
    )
    # GSVD reconstruction identity (docstring): A = U diag(S[:rank]) V^T M_V
    recon = Ug @ np.diag(Sg[: int(rg)]) @ Vg.T @ vals["MV"]
    np.testing.assert_allclose(recon, vals["A"], rtol=1e-6, atol=1e-7)


class _GtOneOp:
    def __call__(self, x):
        return np.asarray(np.asarray(x) > 1.0)


class _HalveOp:
    def __call__(self, x):
        return np.asarray(x) * 0.5


def _gt_one_low(op, builder, args, low):
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.GT, lhs=args[0], rhs=c.FloatLit(value=1.0))]


def _halve_low(op, builder, args, low):
    c = low.core
    return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=args[0], rhs=c.FloatLit(value=0.5))]


@pytest.mark.parametrize("target", ["numpy", "torch"])
def test_while_op(target):
    if target == "torch":
        pytest.importorskip("torch")
    body = Program("halve", inputs=[SymInput("x", (), _prov("x"))])
    (h,) = body.emit_stmt(_HalveOp(), [body.input("x")], [OutSpec("h", ())])
    body.add_output("x", h.cells)
    cond = Program("gt1", inputs=[SymInput("x", (), _prov("x"))])
    (b,) = cond.emit_stmt(_GtOneOp(), [cond.input("x")], [OutSpec("b", ())])
    cond.add_output("b", b.cells)
    p = Program("while", inputs=[SymInput("x0", (), _prov("x0"))])
    (xf,) = p.emit_stmt(WhileOp(cond=cond, body=body), [p.input("x0")], [OutSpec("xf", ())])
    p.add_output("result", xf.cells)

    opts = pyab.LowerOpts(target=target, op_lowerings={"_GtOneOp": _gt_one_low, "_HalveOp": _halve_low})
    want = float(np.asarray(p.run({"x0": 100.0})["result"]))
    if target == "numpy":
        got = float(np.asarray(pyab.compile_numpy(p, name="f", opts=opts).module.f(100.0)))
    else:
        import torch
        got = float(np.asarray(pyab.compile_torch(p, name="f", opts=opts).module.f(torch.tensor(100.0))))
    np.testing.assert_allclose(got, want, rtol=1e-9)


def _nested_vmap_closure(bodyprog, n_vars, n_free, outname):
    """Faithful stand-in for grassmann's multi-var LLam vmap closure."""
    def run(*args):
        eyes = [np.asarray(e) for e in args[:n_vars]]
        free = list(args[n_vars:])
        sizes = [e.shape[0] for e in eyes]
        grid = {}
        for c in itertools.product(*[range(s) for s in sizes]):
            sv = [eyes[v][c[v]] for v in range(n_vars)] + free
            names = [i.name for i in bodyprog.inputs]
            grid[c] = np.asarray(bodyprog.run(dict(zip(names, sv)))[outname])
        cod = next(iter(grid.values())).shape
        full = np.empty(tuple(sizes) + cod, dtype=float)
        for c, r in grid.items():
            full[c] = r
        return np.moveaxis(full, range(n_vars), range(-n_vars, 0))

    run._vmap_body = bodyprog
    run._nested_n_vars = n_vars
    run._nested_n_free = n_free
    return run


@pytest.mark.parametrize("target", ["numpy", "torch"])
def test_nested_vmap(target):
    if target == "torch":
        pytest.importorskip("torch")
    bodyp = Program("nb", inputs=[
        SymInput("e0", (3,), _prov("e0")), SymInput("e1", (3,), _prov("e1")), SymInput("w", (3,), _prov("w")),
    ])
    (dot0,) = bodyp.emit_stmt(EinsumStmtOp(spec="i,i->"), [bodyp.input("e0"), bodyp.input("w")], [OutSpec("d", ())])
    (out,) = bodyp.emit_stmt(EinsumStmtOp(spec=",i->i"), [dot0, bodyp.input("e1")], [OutSpec("r", (3,))])
    bodyp.add_output("r", out.cells)

    nv = _nested_vmap_closure(bodyp, n_vars=2, n_free=1, outname="r")
    p = Program("nvmap", inputs=[
        SymInput("E0", (2, 3), _prov("E0")), SymInput("E1", (2, 3), _prov("E1")), SymInput("w", (3,), _prov("w")),
    ])
    (o,) = p.emit_stmt(nv, [p.input("E0"), p.input("E1"), p.input("w")], [OutSpec("o", (3, 2, 2))])
    p.add_output("result", o.cells)
    vals = {"E0": np.eye(2, 3), "E1": np.eye(2, 3), "w": np.random.default_rng(2).standard_normal(3)}
    if target == "numpy":
        _check_numpy(p, vals, "nested vmap")
    else:
        _check_torch(p, vals, "nested vmap")


# ---------------------------------------------------------------------------
# placement API: call a lowered kernel from inside a larger PyAB program
# ---------------------------------------------------------------------------

def _det2_kernel():
    k = Program("det2", inputs=[SymInput("M", (2, 2), _prov())])
    k.add_output("result", k.input("M").det().cells)
    return k


@pytest.mark.parametrize("place", ["plain", "vmap", "fuse"])
def test_placement_embedding(place):
    torch = pytest.importorskip("torch")
    from pyarraybackend.ir import core, StmtBuilder
    from pyarraybackend.backends import torch_backend

    kernel = _det2_kernel()
    fb = StmtBuilder()
    if place == "vmap":
        param, shape = "batch", (7, 2, 2)
        kw = dict(place="vmap", in_dims=(0,), out_dim=0)
    else:
        param, shape = "M", (2, 2)
        kw = dict(place=place)
    module_defs, result = pyab.call_lowered(kernel, fb, [core.Var(param)], **kw)
    fb.ret(result)
    outer = core.FunctionDefStmt(name="outer", params=(core.Param(name=param),), body=fb.finish())
    cm = torch_backend.compile(module_defs + (outer,))

    rng = np.random.default_rng(3)
    if place == "vmap":
        batch = rng.standard_normal(shape)
        got = np.asarray(cm.module.outer(torch.as_tensor(batch)))
        want = np.array([np.linalg.det(b) for b in batch])
    else:
        M = rng.standard_normal(shape)
        got = np.asarray(cm.module.outer(torch.as_tensor(M)))
        want = np.linalg.det(M)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-8)


def test_batched_small_qr_vmapped_matches_lapack():
    """The flagship case: batched small-QR (Householder), vmapped, torch backend."""
    torch = pytest.importorskip("torch")
    from pyarraybackend.ir import core, StmtBuilder
    from pyarraybackend.backends import torch_backend

    kernel = _qr_program(3, 3)
    # kernel must carry NO linalg.qr (scalar Householder emitted):
    from pyarraybackend.backends.python_backend import codegen as pycg

    assert "linalg.qr" not in pycg.PythonCodegen().render(pyab.as_function_def(kernel, name="_k"))

    fb = StmtBuilder()
    mdefs, result = pyab.call_lowered(kernel, fb, [core.Var("batch")], place="vmap", in_dims=(0,), out_dim=0)
    fb.ret(result)
    outer = core.FunctionDefStmt(name="outer", params=(core.Param(name="batch"),), body=fb.finish())
    cm = torch_backend.compile(mdefs + (outer,))

    batch = np.random.default_rng(4).standard_normal((5, 3, 3))
    Qb, Rb = cm.module.outer(torch.as_tensor(batch))
    for i, b in enumerate(batch):
        qn, rn = np.linalg.qr(b)
        np.testing.assert_allclose(np.asarray(Qb[i]), qn, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(np.asarray(Rb[i]), rn, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# backend prep — collapse avoidable vmaps + fold identity-metric GSVD
# ---------------------------------------------------------------------------

def _batched_linalg_vmap(op_name, batch, tail, budget=True):
    """A vmap over a per-element linalg op (forced through a Stmt)."""
    b = SymbolicBudget.force_stmts() if budget else None
    if op_name == "solve":
        body = Program("b", inputs=[SymInput("A", tail, _prov("A")), SymInput("B", (tail[0], 2), _prov("B"))], budget=b)
        body.add_output("r", body.input("A").solve(body.input("B")).cells)
        outshape = batch + (tail[0], 2)
        p = Program("v", inputs=[SymInput("Ab", batch + tail, _prov("A")), SymInput("Bb", batch + (tail[0], 2), _prov("B"))])
        (o,) = p.emit_stmt(vmap(body, in_axes=(0, 0), out_axes=0), [p.input("Ab"), p.input("Bb")], [OutSpec("x", outshape)])
    else:
        body = Program("b", inputs=[SymInput("M", tail, _prov())], budget=b)
        body.add_output("r", getattr(body.input("M"), op_name)().cells)
        outshape = batch + (() if op_name == "det" else tail)
        p = Program("v", inputs=[SymInput("batch", batch + tail, _prov())])
        (o,) = p.emit_stmt(vmap(body, in_axes=(0,), out_axes=0), [p.input("batch")], [OutSpec("d", outshape)])
    p.add_output("result", o.cells)
    return p


@pytest.mark.parametrize("op_name,expect", [("det", "DetOp"), ("inverse", "InvOp"), ("solve", "SolveOp")])
def test_collapse_batchable_vmap(op_name, expect):
    p = _batched_linalg_vmap(op_name, (5,), (3, 3))
    _, report = pyab.prepare(p, opts=pyab.LowerOpts(target="numpy"))
    assert report["collapsed_vmap"] == [expect]
    assert "vmap" not in pyab.compile_numpy(p, name="f").source
    rng = np.random.default_rng(1)
    if op_name == "solve":
        vals = {"Ab": rng.standard_normal((5, 3, 3)) + 3 * np.eye(3), "Bb": rng.standard_normal((5, 3, 2))}
    else:
        vals = {"batch": rng.standard_normal((5, 3, 3)) + 3 * np.eye(3)}
    _check_numpy(p, vals, f"collapsed vmap({op_name})")


def test_collapse_einsum_vmap():
    body = Program("b", inputs=[SymInput("A", (2, 3), _prov("A")), SymInput("B", (3, 2), _prov("B"))])
    (cc,) = body.emit_stmt(EinsumStmtOp(spec="ij,jk->ik"), [body.input("A"), body.input("B")], [OutSpec("C", (2, 2))])
    body.add_output("r", cc.cells)
    p = Program("v", inputs=[SymInput("Ab", (5, 2, 3), _prov("A")), SymInput("Bb", (5, 3, 2), _prov("B"))])
    (o,) = p.emit_stmt(vmap(body, in_axes=(0, 0), out_axes=0), [p.input("Ab"), p.input("Bb")], [OutSpec("C", (5, 2, 2))])
    p.add_output("result", o.cells)
    _, report = pyab.prepare(p, opts=pyab.LowerOpts(target="numpy"))
    assert report["collapsed_vmap"] == ["EinsumStmtOp"]
    assert "vmap" not in pyab.compile_numpy(p, name="f").source
    _check_numpy(p, {"Ab": np.random.rand(5, 2, 3), "Bb": np.random.rand(5, 3, 2)}, "collapsed vmap(einsum)")


def test_collapse_rejects_nonbatchable_svd():
    """vmap(svd) is NOT collapsed — the rank output does not batch elementwise."""
    body = Program("b", inputs=[SymInput("M", (3, 3), _prov())])
    _, S, _, _ = body.emit_stmt(
        SvdOp(full_matrices=False), [body.input("M")],
        [OutSpec("U", (3, 3)), OutSpec("S", (3,)), OutSpec("Vh", (3, 3)), OutSpec("rank", ())],
    )
    body.add_output("r", S.cells)
    p = Program("v", inputs=[SymInput("batch", (4, 3, 3), _prov())])
    (o,) = p.emit_stmt(vmap(body, in_axes=(0,), out_axes=0), [p.input("batch")], [OutSpec("s", (4, 3))])
    p.add_output("result", o.cells)
    _, report = pyab.prepare(p, opts=pyab.LowerOpts(target="numpy"))
    assert report["collapsed_vmap"] == []


def test_collapse_can_be_disabled():
    p = _batched_linalg_vmap("det", (3,), (2, 2))
    assert "vmap" in pyab.compile_numpy(p, name="f", opts=pyab.LowerOpts(target="numpy", collapse_vmap=False)).source


def _gsvd_program(metrics):
    from polyarray.ir import Const
    nW, nV = 4, 3
    specs = {"identity": (Const(np.eye(nV)), Const(np.eye(nW))), "full": None}
    if metrics == "identity":
        mv, mw = Const(np.eye(nV)), Const(np.eye(nW))
        p = Program("g", inputs=[SymInput("A", (nW, nV), _prov("A"))])
        A = p.input("A")
    else:
        p = Program("g", inputs=[SymInput("A", (nW, nV), _prov("A")), SymInput("MV", (nV, nV), _prov("MV")), SymInput("MW", (nW, nW), _prov("MW"))])
        A, mv, mw = p.input("A"), p.input("MV"), p.input("MW")
    outs = p.emit_stmt(
        GSvdOp(), [A, mv, mw],
        [OutSpec("U", (nW, nV)), OutSpec("UI", (nW, nW - nV)), OutSpec("V", (nV, nV)),
         OutSpec("VI", (nV, 0)), OutSpec("S", (nV,)), OutSpec("rank", ())],
    )
    for nm, sa in zip(("U", "UI", "V", "VI", "S", "rank"), outs):
        p.add_output(nm, sa.cells)
    return p


def test_gsvd_identity_metrics_folds_to_svd():
    p = _gsvd_program("identity")
    src = pyab.compile_numpy(p, name="f").source
    assert "cholesky" not in src and "solve" not in src
    _check_numpy(p, {"A": np.random.default_rng(0).standard_normal((4, 3))}, "GSVD identity-fold")


def test_gsvd_nonidentity_metrics_kept():
    p = _gsvd_program("full")
    assert "cholesky" in pyab.compile_numpy(p, name="f").source
    rng = np.random.default_rng(2)
    spd = lambda n: (lambda M: M @ M.T + n * np.eye(n))(rng.standard_normal((n, n)))
    _check_numpy(p, {"A": rng.standard_normal((4, 3)), "MV": spd(3), "MW": spd(4)}, "GSVD non-identity kept")


# ---------------------------------------------------------------------------
# optionality — polyarray must not require pyarraybackend/torch
# ---------------------------------------------------------------------------

def test_import_polyarray_does_not_import_pyarraybackend_or_torch():
    code = (
        "import sys, polyarray\n"
        "assert 'pyarraybackend' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "from polyarray import pyab\n"          # importing the feature module...
        "assert 'torch' not in sys.modules\n"   # ...still does not import torch
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_numpy_lane_runs_with_torch_unimportable():
    code = (
        "import builtins\n"
        "_r = builtins.__import__\n"
        "def _n(name,*a,**k):\n"
        "    if name=='torch' or name.startswith('torch.'):\n"
        "        raise ModuleNotFoundError('sim')\n"
        "    return _r(name,*a,**k)\n"
        "builtins.__import__=_n\n"
        "import sys, numpy as np\n"
        "from polyarray.ir import Program, SymInput, Provenance\n"
        "from polyarray import pyab\n"
        "p=Program('d',inputs=[SymInput('A',(3,3),Provenance(kind='coeff',origin='t',index=(),label='A'))])\n"
        "p.add_output('result', p.input('A').det().cells)\n"
        "cm=pyab.compile_numpy(p, name='f')\n"
        "A=np.random.rand(3,3)\n"
        "np.testing.assert_allclose(cm.module.f(A), np.linalg.det(A), rtol=1e-9)\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
