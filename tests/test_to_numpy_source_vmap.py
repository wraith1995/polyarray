"""Tests for :func:`polyarray.to_numpy_source` over ``vmap`` and sub-``Program``
Stmt fns.

Each test builds a :class:`Program` directly (no grassmann needed), emits
standalone numpy source, ``exec``s it in a fresh namespace, calls the generated
entrypoint on random numpy inputs, and asserts the result matches
``program.run(same_inputs)`` exactly (allclose).  This is the acceptance bar:
the emitted code must equal what the program computes, including the byte-for-
byte semantics of ``vmap`` (in_axes ``None`` = broadcast, batched-axis stacking,
out_axes).
"""
from __future__ import annotations

import numpy as np
import pytest

import polyarray as pa
from polyarray.ir import (
    DetOp,
    OutSpec,
    Program,
    Provenance,
    SolveOp,
    SymArray,
    SymInput,
    vmap,
)
from polyarray.rational import RationalFunction


def _prov(label: str) -> Provenance:
    return Provenance(kind="vertex", origin=label, index=(), label=label)


def _roundtrip(prog, values, func_name="f"):
    """Emit, exec, call, and compare against ``prog.run``.  Returns the source."""
    src = pa.to_numpy_source(prog, func_name)
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)
    got = ns[func_name](*[values[i.name] for i in prog.inputs])
    want = prog.run(values)
    out_names = list(prog.outputs.keys())
    if len(out_names) == 1:
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want[out_names[0]]), rtol=1e-9, atol=1e-12
        )
    else:
        assert isinstance(got, tuple) and len(got) == len(out_names)
        for g, name in zip(got, out_names):
            np.testing.assert_allclose(
                np.asarray(g), np.asarray(want[name]), rtol=1e-9, atol=1e-12
            )
    return src


def _nonsingular(rng, shape):
    """Random batched square matrices, each made well-conditioned."""
    n = shape[-1]
    return rng.standard_normal(shape) + 3.0 * np.eye(n)


# ---------------------------------------------------------------------------
# 1. vmap over a batched leading axis, with a broadcast (in_axes=None) operand.
# ---------------------------------------------------------------------------

def test_vmap_solve_broadcast_rhs():
    """vmap(solve_body) over batched A (M,n,n); shared b (n,) via in_axes=None."""
    n = 3

    body = Program(
        "solve_body",
        inputs=[SymInput("A", (n, n), _prov("A")), SymInput("b", (n,), _prov("b"))],
    )
    (x,) = body.emit_stmt(
        SolveOp(),
        [body.input("A"), body.input("b")],
        [OutSpec("x", (n,))],
        bulk=True,
    )
    body.add_output("result", x)

    M = 5
    prog = Program(
        "vmap_solve",
        inputs=[
            SymInput("Ab", (M, n, n), _prov("Ab")),
            SymInput("b", (n,), _prov("b")),
        ],
    )
    vfn = vmap(body, in_axes=(0, None), out_axes=0)
    (out,) = prog.emit_stmt(
        vfn,
        [prog.input("Ab"), prog.input("b")],
        [OutSpec("result", (M, n))],
        bulk=True,
    )
    prog.add_output("result", out)

    rng = np.random.default_rng(0)
    values = {"Ab": _nonsingular(rng, (M, n, n)), "b": rng.standard_normal(n)}
    src = _roundtrip(prog, values, func_name="vmapped")
    assert "def _vmap" in src
    assert "np.stack(" in src
    # Sanity: matches an independent numpy reference too.
    ref = np.stack(
        [np.linalg.solve(values["Ab"][i], values["b"]) for i in range(M)], axis=0
    )
    np.testing.assert_allclose(prog.run(values)["result"], ref, atol=1e-9)


def test_vmap_batched_axis_nonzero_and_out_axis():
    """in_axes=1 (batch on a middle axis) and out_axes=1 — non-trivial axes."""
    n = 2
    body = Program("det_body", inputs=[SymInput("A", (n, n), _prov("A"))])
    (d,) = body.emit_stmt(
        DetOp(), [body.input("A")], [OutSpec("d", ())], bulk=True
    )
    body.add_output("result", d)

    M = 4
    # A laid out as (n, M, n): batch axis is 1.
    prog = Program("vmap_axis", inputs=[SymInput("A", (n, M, n), _prov("A"))])
    vfn = vmap(body, in_axes=1, out_axes=0)
    (out,) = prog.emit_stmt(
        vfn, [prog.input("A")], [OutSpec("result", (M,))], bulk=True
    )
    prog.add_output("result", out)

    rng = np.random.default_rng(1)
    A = _nonsingular(rng, (M, n, n)).transpose(1, 0, 2)  # -> (n, M, n)
    _roundtrip(prog, {"A": A})


# ---------------------------------------------------------------------------
# 2. raw sub-Program Stmt fn.
# ---------------------------------------------------------------------------

def test_subprogram_stmt_fn():
    """A Stmt whose fn is a sub-Program: dispatched by position, det output."""
    n = 3
    sub = Program("sub_det", inputs=[SymInput("A", (n, n), _prov("A"))])
    (d,) = sub.emit_stmt(DetOp(), [sub.input("A")], [OutSpec("d", ())], bulk=True)
    sub.add_output("det", d)

    prog = Program("callsub", inputs=[SymInput("A", (n, n), _prov("A"))])
    (out,) = prog.emit_stmt(
        sub, [prog.input("A")], [OutSpec("det", ())], bulk=True
    )
    prog.add_output("det", out)

    rng = np.random.default_rng(2)
    values = {"A": _nonsingular(rng, (n, n))}
    src = _roundtrip(prog, values, func_name="callsub")
    assert "def _sub" in src
    np.testing.assert_allclose(
        prog.run(values)["det"], np.linalg.det(values["A"]), atol=1e-9
    )


def test_subprogram_multi_output():
    """A sub-Program with two outputs returns a tuple, unpacked by the parent."""
    n = 3
    sub = Program(
        "sub_two",
        inputs=[SymInput("A", (n, n), _prov("A")), SymInput("b", (n,), _prov("b"))],
    )
    (d,) = sub.emit_stmt(DetOp(), [sub.input("A")], [OutSpec("d", ())], bulk=True)
    (x,) = sub.emit_stmt(
        SolveOp(), [sub.input("A"), sub.input("b")], [OutSpec("x", (n,))], bulk=True
    )
    sub.add_output("d", d)
    sub.add_output("x", x)

    prog = Program(
        "call_two",
        inputs=[SymInput("A", (n, n), _prov("A")), SymInput("b", (n,), _prov("b"))],
    )
    outs = prog.emit_stmt(
        sub,
        [prog.input("A"), prog.input("b")],
        [OutSpec("d", ()), OutSpec("x", (n,))],
        bulk=True,
    )
    prog.add_output("d", outs[0])
    prog.add_output("x", outs[1])

    rng = np.random.default_rng(3)
    values = {"A": _nonsingular(rng, (n, n)), "b": rng.standard_normal(n)}
    _roundtrip(prog, values, func_name="call_two")


# ---------------------------------------------------------------------------
# 3. nested: a multi-output vmap whose body itself calls a sub-Program.
# ---------------------------------------------------------------------------

def test_vmap_body_with_subprogram():
    n = 2
    inner = Program("inner_det", inputs=[SymInput("A", (n, n), _prov("A"))])
    (idet,) = inner.emit_stmt(
        DetOp(), [inner.input("A")], [OutSpec("d", ())], bulk=True
    )
    inner.add_output("d", idet)

    body = Program(
        "body_with_sub",
        inputs=[SymInput("A", (n, n), _prov("A")), SymInput("b", (n,), _prov("b"))],
    )
    (dout,) = body.emit_stmt(
        inner, [body.input("A")], [OutSpec("d", ())], bulk=True
    )
    (xout,) = body.emit_stmt(
        SolveOp(), [body.input("A"), body.input("b")], [OutSpec("x", (n,))], bulk=True
    )
    body.add_output("d", dout)
    body.add_output("x", xout)

    M = 4
    prog = Program(
        "nested",
        inputs=[
            SymInput("A", (M, n, n), _prov("A")),
            SymInput("b", (n,), _prov("b")),
        ],
    )
    vfn = vmap(body, in_axes=(0, None), out_axes=0)
    outs = prog.emit_stmt(
        vfn,
        [prog.input("A"), prog.input("b")],
        [OutSpec("d", (M,)), OutSpec("x", (M, n))],
        bulk=True,
    )
    prog.add_output("d", outs[0])
    prog.add_output("x", outs[1])

    rng = np.random.default_rng(4)
    values = {"A": _nonsingular(rng, (M, n, n)), "b": rng.standard_normal(n)}
    src = _roundtrip(prog, values, func_name="nested")
    assert "def _vmap" in src and "def _sub" in src


# ---------------------------------------------------------------------------
# 4. regression: an op-only program is unaffected (no helper defs emitted).
# ---------------------------------------------------------------------------

def test_op_only_program_regression():
    """A plain op program still emits & round-trips, with no vmap/sub helpers."""
    n = 3
    prog = Program("plain", inputs=[SymInput("A", (n, n), _prov("A"))])
    (d,) = prog.emit_stmt(DetOp(), [prog.input("A")], [OutSpec("d", ())], bulk=True)
    prog.add_output("det", d)

    src = pa.to_numpy_source(prog)
    assert "_vmap" not in src and "_sub" not in src
    # Determinism: identical re-emission.
    assert src == pa.to_numpy_source(prog)

    rng = np.random.default_rng(5)
    values = {"A": _nonsingular(rng, (n, n))}
    _roundtrip(prog, values)


# ---------------------------------------------------------------------------
# 5. free feed atoms: a generator bound by no input and no statement output.
# ---------------------------------------------------------------------------

def test_free_feed_atom_becomes_a_trailing_parameter():
    """An open program (a free vertex-style atom in its cells) renders as a function of it.

    The mid-pipeline programs savo hands the observability dump carry per-cell vertex atoms
    ``V_j_k`` that no input declares yet; they are the program's free variables, so the emitted
    function takes them as parameters after the declared inputs, sorted by generator name.
    ``Program.run`` binds the same atoms from an undeclared ``values`` entry, which is what makes
    the two comparable.
    """
    prog = Program("open", inputs=[SymInput("A", (2, 2), _prov("A"))])
    a = prog.input_arrays["A"].cells
    cells = np.empty((2, 2), dtype=object)
    cells[0, 0] = a[0, 0] + RationalFunction.atom("Vf_1")
    cells[0, 1] = a[0, 1]
    cells[1, 0] = a[1, 0]
    cells[1, 1] = a[1, 1] + RationalFunction.atom("Vf_0")
    (d,) = prog.emit_stmt(
        DetOp(), [SymArray(cells, program=prog)], [OutSpec("d", ())], bulk=True
    )
    prog.add_output("result", d)

    rng = np.random.default_rng(6)
    A = rng.standard_normal((2, 2)) + 3.0 * np.eye(2)
    v = rng.standard_normal(2)
    src = pa.to_numpy_source(prog)
    # Sorted generator-name order — Vf_0 before Vf_1, whatever order the cells mention them in.
    assert "def f(A, Vf_0, Vf_1):" in src
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)
    got = ns["f"](A, v[0], v[1])
    want = prog.run({"A": A, "Vf": v})["result"]
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(got), np.linalg.det(A + np.diag(v[::-1])), rtol=1e-9, atol=1e-12
    )


def test_free_feed_atom_threads_through_a_vmap_body():
    """A ``vmap`` body open over a free atom gains the same trailing parameter, passed at the call.

    Without the threading the body's ``def`` would grow a parameter its call site never supplies.
    The body is not runnable through ``Program.run`` (``_invoke`` maps operands to body inputs by
    position and knows nothing of a free atom), so the reference here is the arithmetic itself.
    """
    n, M = 2, 3
    body = Program("scale_body", inputs=[SymInput("x", (n,), _prov("x"))])
    xc = body.input_arrays["x"].cells
    out = np.empty((n,), dtype=object)
    out[0] = xc[0] * RationalFunction.atom("Vf_0")
    out[1] = xc[1] * RationalFunction.atom("Vf_1")
    body.add_output("result", SymArray(out, program=body))

    prog = Program("vmap_open", inputs=[SymInput("X", (M, n), _prov("X"))])
    (o,) = prog.emit_stmt(
        vmap(body, in_axes=(0,), out_axes=0),
        [prog.input("X")],
        [OutSpec("result", (M, n))],
        bulk=True,
    )
    prog.add_output("result", o)

    src = pa.to_numpy_source(prog)
    assert "def f(X, Vf_0, Vf_1):" in src
    assert "def _sub0(x, Vf_0, Vf_1):" in src
    assert "_sub0(*_sv, _ga0, _ga1)" in src

    rng = np.random.default_rng(7)
    X = rng.standard_normal((M, n))
    v = rng.standard_normal(n)
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)
    np.testing.assert_allclose(ns["f"](X, v[0], v[1]), X * v, rtol=1e-9, atol=1e-12)
