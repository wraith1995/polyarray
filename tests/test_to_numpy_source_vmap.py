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
    SymInput,
    vmap,
)


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
