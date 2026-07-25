"""`is_structurally_constant` — SOUND, sampling-free build-time-constancy test.

True iff `target`'s dependency cone is a CLOSED CONSTANT subprogram: no Stmt input is a
live feed (InputRef / IntAtomRef) and every cell generator in the cone (and in `target`
itself) has provenance kind "stmt_out". Conservative: anything not provably constant-safe
forces False. This is the exact check grassmann's `_cone_is_structurally_constant`
delegates to; it decides `pointwise.is_affine_invariant` without any eval/codegen.
"""
import numpy as np

import polyarray as pa
from polyarray.ir import InputRef, Program, Provenance, SymInput
from polyarray.simplify import _symarray_atoms


def _const_op(arr: np.ndarray) -> pa.ConstOp:
    a = np.ascontiguousarray(arr, dtype=float)
    return pa.ConstOp(key="k", data_bytes=a.tobytes(), shape=a.shape, dtype="float64")


def test_all_constant_cone_is_structurally_constant():
    """A cone of only nullary ConstOp + a downstream op over a prior stmt_out output —
    no live input, all-stmt_out generators — is a provable build-time constant."""
    p = Program("const", inputs=[])
    [c] = p.emit_stmt(_const_op(np.array([[2.0, 0.0], [0.0, 4.0]])), [],
                      [pa.OutSpec("c", (2, 2))], note="const")          # stmt 0 (nullary)
    [y] = p.emit_stmt(pa.InvOp(), [c], [pa.OutSpec("y", (2, 2))], note="inv")  # stmt 1
    p.add_output("y", y)
    assert pa.is_structurally_constant(p, y) is True


def test_cone_reading_an_input_ref_is_not_constant():
    """A Stmt whose input is an InputRef reads a value supplied at run time ⇒ False."""
    p = Program("live", inputs=[SymInput("x", (2, 2),
                                          Provenance(kind="input", origin="x", index=(), label="x"))])
    [y] = p.emit_stmt(pa.InvOp(), [InputRef("x")], [pa.OutSpec("y", (2, 2))], note="inv_x")
    p.add_output("y", y)
    assert pa.is_structurally_constant(p, y) is False


def test_target_carrying_a_feed_generator_is_not_constant():
    """If `target` itself reads a feed atom (here a 'vertex'-provenance input) it depends
    on the feed even with an empty/constant cone ⇒ False (the target-guard short-circuit)."""
    p = Program("feed", inputs=[SymInput("v", (2, 2),
                                         Provenance(kind="vertex", origin="v", index=(), label="v"))])
    v = p.input_arrays["v"]                     # per-cell SymArray; its atoms are vertex feeds
    atoms = _symarray_atoms(v)
    assert atoms and all(p.env.of(a).kind == "vertex" for a in atoms)  # live feeds, not stmt_out
    assert pa.is_structurally_constant(p, v) is False
