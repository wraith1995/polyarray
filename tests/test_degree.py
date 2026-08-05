"""`polyarray.program_degree` — the whole-program polynomial-degree walker.

Native-op semantics only (front ends test their own category extensions — pointwise's
`test_compiler_mass` degree probes ride the same walker through its delegator):

* multilinear (EinsumStmtOp) → SUM of operand degrees;
* passthrough (MoveaxisOp) → MAX;
* rational/algebraic (SqrtOp) on a seed-dependent value → inf, on a constant → 0
  (the affine-constant short-circuit);
* CallOp (vmap) is unwrapped and recursed, seeding the body's inputs by position;
* a passthrough output (result IS an input) reads its seed, not inf.
"""
from __future__ import annotations

import numpy as np
import polyarray as pa

_INF = float("inf")


def _prov(name: str) -> pa.Provenance:
    return pa.Provenance("coeff", "x", (), name)


def _one_input(shape=()) -> pa.Program:
    return pa.Program("t", inputs=[pa.SymInput("u", shape, _prov("u"))])


def test_multilinear_sums_and_passthrough_maxes():
    p = pa.Program("m", inputs=[pa.SymInput("u", (), _prov("u")), pa.SymInput("v", (), _prov("v"))])
    u, v = p.input("u"), p.input("v")
    [prod] = p.emit_stmt(pa.EinsumStmtOp(spec=",->"), [u, v], [pa.OutSpec("p", ())],
                         note="uv", bulk=False)
    p.add_output("result", prod)
    # u:2, v:3 → product degree 5 (SUM)
    assert pa.program_degree(p, {"u": 2.0, "v": 3.0}) == 5.0

    p2 = _one_input((2, 3))
    [mv] = p2.emit_stmt(pa.MoveaxisOp(0, 1), [p2.input("u")], [pa.OutSpec("m", (3, 2))],
                        note="mv", bulk=False)
    p2.add_output("result", mv)
    assert pa.program_degree(p2, {"u": 4.0}) == 4.0              # passthrough → MAX


def test_rational_op_inf_on_seed_zero_on_constant():
    p = _one_input()
    [s] = p.emit_stmt(pa.SqrtOp(), [p.input("u")], [pa.OutSpec("s", ())], note="sq", bulk=False)
    p.add_output("result", s)
    assert pa.program_degree(p, {"u": 1.0}) == _INF              # algebraic in the seed

    p0 = _one_input()
    const = pa.SymArray(np.asarray(2.0), program=p0)
    [s0] = p0.emit_stmt(pa.SqrtOp(), [const], [pa.OutSpec("s", ())], note="sq0", bulk=False)
    p0.add_output("result", s0)
    assert pa.program_degree(p0, {"u": 1.0}) == 0.0              # constant short-circuit


def test_passthrough_output_reads_the_seed():
    p = _one_input()
    p.add_output("result", p.input("u"))                          # result IS an input
    assert pa.program_degree(p, {"u": 7.0}) == 7.0


def test_callop_vmap_body_is_recursed():
    # vmap a per-point body u ↦ u·u (einsum SUM inside the body): seed 1 → degree 2.
    body = pa.Program("b", inputs=[pa.SymInput("x", (), _prov("x"))])
    [sq] = body.emit_stmt(pa.EinsumStmtOp(spec=",->"), [body.input("x"), body.input("x")],
                          [pa.OutSpec("s", ())], note="x2", bulk=False)
    body.add_output("result", sq)

    outer = _one_input((4,))
    [out] = outer.emit_stmt(pa.ir.CallOp(fn=body), [outer.input("u")],
                            [pa.OutSpec("o", (4,))], note="call", bulk=False)
    outer.add_output("result", out)
    assert pa.program_degree(outer, {"u": 1.0}) == 2.0
    # a constant call is constant, whatever the body
    assert pa.program_degree(outer, {"u": 0.0}) == 0.0


def test_detop_is_polynomial_n_times_d():
    # det of a (2,2) matrix of degree-3 entries: a sum of 2-fold products → degree 6.
    p = _one_input((2, 2))
    [dt] = p.emit_stmt(pa.DetOp(), [p.input("u")], [pa.OutSpec("d", ())], note="det", bulk=False)
    p.add_output("result", dt)
    assert pa.program_degree(p, {"u": 3.0}) == 6.0
    assert pa.program_degree(p, {"u": 0.0}) == 0.0               # constant matrix → constant det
