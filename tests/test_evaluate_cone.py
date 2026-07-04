"""`evaluate_cone` — evaluate ONE SymArray via only its dependency cone.

The point: probe a target inside a big (possibly partially-built) program at a generic
feed WITHOUT running unrelated statements — so a singular / failing op elsewhere never
runs and never crashes the probe. Used by the covariant-`A₀` constant-fold in grassmann.
"""
import numpy as np
import pytest

import polyarray as pa
from polyarray.ir import Program, Provenance, SymInput
from polyarray.simplify import dependency_cone, evaluate_cone


def _prov(name):
    return Provenance(kind="input", origin=name, index=(), label=name)


def _prog():
    p = Program("t", inputs=[SymInput("x", (2, 2), _prov("x")),
                             SymInput("s", (2, 2), _prov("s"))])
    x, s = p.input_arrays["x"], p.input_arrays["s"]
    [y] = p.emit_stmt(pa.InvOp(), [x], [pa.OutSpec("y", (2, 2))], note="inv_x")          # stmt 0
    [bad] = p.emit_stmt(pa.InvOp(), [s], [pa.OutSpec("bad", (2, 2))], note="inv_s_POISON")  # stmt 1
    p.add_output("y", y)
    p.add_output("bad", bad)
    return p, x, s, y, bad


def test_cone_excludes_unrelated_statements():
    p, x, s, y, bad = _prog()
    cone_y = dependency_cone(p, y)
    assert cone_y == {0}                               # only inv_x; the poison inv_s is excluded
    assert dependency_cone(p, bad) == {1}


def test_evaluate_cone_skips_a_poison_op_a_full_run_would_hit():
    p, x, s, y, bad = _prog()
    xv = np.array([[2.0, 0.0], [0.0, 4.0]])
    sv = np.array([[1.0, 1.0], [1.0, 1.0]])            # SINGULAR ⇒ inv(s) raises
    vals = {"x": xv, "s": sv}

    with pytest.raises(Exception):                     # the full run hits the poison inv(s)
        p.run(vals)

    got = evaluate_cone(p, y, vals)                     # the cone of y never runs inv(s)
    np.testing.assert_allclose(got, np.linalg.inv(xv), atol=1e-12)


def test_evaluate_cone_matches_full_run_when_both_succeed():
    p, x, s, y, bad = _prog()
    xv = np.array([[2.0, 1.0], [0.0, 3.0]])
    sv = np.array([[1.0, 0.0], [0.0, 2.0]])            # non-singular ⇒ full run also succeeds
    vals = {"x": xv, "s": sv}
    full = np.asarray(p.run(vals)["y"], float)
    np.testing.assert_allclose(evaluate_cone(p, y, vals), full, atol=1e-12)


def test_cone_follows_a_chain_and_dynamic_rank():
    """A multi-stage chain: the cone is the transitive closure, and a dynamic (SVD-rank)
    output does not break the walk — evaluate_cone reproduces the full run."""
    p = Program("chain", inputs=[SymInput("m", (3, 2), _prov("m"))])
    m = p.input_arrays["m"]
    U, S, Vh, rank = p.emit_stmt(
        pa.SvdOp(full_matrices=True), [m],
        [pa.OutSpec("U", (3, 3)), pa.OutSpec("S", (2,)), pa.OutSpec("Vh", (2, 2)),
         pa.OutSpec("rank", ())], note="svd")
    [g] = p.emit_stmt(pa.EinsumStmtOp("ij,ik->jk"), [U, U], [pa.OutSpec("g", (3, 3))], note="UtU")
    p.add_output("g", g)
    mv = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 0.3]])
    full = np.asarray(p.run({"m": mv})["g"], float)
    np.testing.assert_allclose(evaluate_cone(p, g, {"m": mv}), full, atol=1e-10)
