"""`SymArray.reshape` / `expand_dims` — the shape ops that do NOT force a bulk array.

Why these exist: a reshape wants no cell VALUES, only a shape. But `.cells` auto-unpacks a bulk
array — one `unpack` Stmt materialising every per-cell atom — so the idiom
`SymArray(sa.cells.reshape(...), program=sa.program)` forces a whole deferred tensor to answer a
question about its layout. Six sites across pointwise and savo were doing exactly that, and the
audit's `DEFER-FORCE` rule reported each as its own finding when in fact they shared one cause:
`ReshapeOp` has been in the IR (emitted by grassmann, rendered by pyab, degree-transparent) all
along, with no `SymArray` method exposing it.

The property that matters is the FIRST test: bulk in, bulk out, statement count grows by one, and
no `unpack` appears. The rest guard the arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyarray.ir import OutSpec, Program, Provenance, SymArray, SymInput


def _prov(label: str):
    return lambda idx: Provenance(kind="vertex", origin=label, index=(), label=label)


def _bulk_program(shape=(2, 3)):
    """A program whose single statement produces a BULK output — the deferred case."""
    prog = Program("bulk", inputs=[SymInput("a", shape, _prov("a"))])

    def _double(x):
        return x * 2.0

    (out,) = prog.emit_stmt(_double, [prog.input("a")], [OutSpec("r", shape)], bulk=True)
    return prog, out


def test_reshape_keeps_a_bulk_array_bulk() -> None:
    """The whole point: one more Stmt, still deferred, nothing materialised."""
    prog, out = _bulk_program()
    before = len(prog.statements)

    r = out.reshape((3, 2))

    assert r._bulk is not None, "reshape unpacked a bulk array — the force this method exists to avoid"
    assert tuple(r.shape) == (3, 2)
    assert len(prog.statements) == before + 1
    assert type(prog.statements[-1].fn).__name__ == "ReshapeOp"


def test_expand_dims_keeps_a_bulk_array_bulk() -> None:
    prog, out = _bulk_program()
    e = out.expand_dims(1)
    assert e._bulk is not None
    assert tuple(e.shape) == (2, 1, 3)


@pytest.mark.parametrize(("shape", "expected"), [((3, 2), (3, 2)), ((6,), (6,)), ((-1,), (6,)),
                                                 ((2, -1), (2, 3)), ((1, 6), (1, 6))])
def test_reshape_values_are_right(shape, expected) -> None:
    prog, out = _bulk_program()
    prog.add_output("out", out.reshape(shape))
    a = np.arange(6.0).reshape(2, 3)
    got = prog.run({"a": a})["out"]
    assert tuple(got.shape) == expected
    assert np.array_equal(got, (a * 2.0).reshape(expected))


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -3])
def test_expand_dims_values_are_right(axis) -> None:
    prog, out = _bulk_program()
    prog.add_output("out", out.expand_dims(axis))
    a = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(prog.run({"a": a})["out"], np.expand_dims(a * 2.0, axis))


def test_non_bulk_reshape_emits_no_statement() -> None:
    """Already-materialised cells reshape in place — no Stmt, and no force either."""
    prog = Program("plain")
    sa = SymArray(np.arange(6.0).reshape(2, 3), program=prog)
    assert tuple(sa.reshape((3, 2)).shape) == (3, 2)
    assert len(prog.statements) == 0


def test_size_mismatch_is_refused() -> None:
    """numpy would refuse too, but only after the operand was built — and the message would be
    about an ndarray, not about the array the caller is holding."""
    sa = SymArray(np.arange(6.0).reshape(2, 3), program=Program("p"))
    with pytest.raises(ValueError, match="cannot reshape size 6"):
        sa.reshape((4, 2))


def test_unresolvable_negative_one_is_refused() -> None:
    sa = SymArray(np.arange(6.0).reshape(2, 3), program=Program("p"))
    with pytest.raises(ValueError, match="resolve -1"):
        sa.reshape((4, -1))


def test_expand_dims_axis_is_range_checked() -> None:
    sa = SymArray(np.arange(6.0).reshape(2, 3), program=Program("p"))
    with pytest.raises(ValueError, match="out of range"):
        sa.expand_dims(3)
