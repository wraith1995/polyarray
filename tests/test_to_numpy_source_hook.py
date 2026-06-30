"""Tests for the ``__numpy_source__`` emit hook in :func:`polyarray.to_numpy_source`.

A ``Stmt.fn`` may be a plain Python callable (not an op class, sub-``Program``, or
``vmap`` closure).  Such a fn emits to standalone numpy IFF it carries a
``__numpy_source__(args) -> str`` attribute (``args`` = the rendered operand
expression strings; returns a numpy expression string).  A hookless plain fn
still raises the clear ``NotImplementedError``.  Each test builds a ``Program``
directly, emits, ``exec``s, and compares to ``program.run`` (the acceptance bar).
"""
from __future__ import annotations

import numpy as np
import pytest

import polyarray as pa
from polyarray.ir import OutSpec, Program, Provenance, SymInput


def _prov(label: str) -> Provenance:
    return Provenance(kind="vertex", origin=label, index=(), label=label)


def test_plain_fn_hook_emits_and_round_trips():
    """A plain ``np.einsum`` closure stamped with ``__numpy_source__`` emits + round-trips."""
    n = 4
    prog = Program(
        "hook_body",
        inputs=[SymInput("a", (n,), _prov("a")), SymInput("b", (n,), _prov("b"))],
    )

    def _dot(*arrs):
        return np.einsum("i,i->", *arrs)

    _dot.__numpy_source__ = lambda args: f"np.einsum('i,i->', {', '.join(args)})"
    (out,) = prog.emit_stmt(
        _dot, [prog.input("a"), prog.input("b")], [OutSpec("result", ())], bulk=True
    )
    prog.add_output("result", out)

    rng = np.random.default_rng(0)
    values = {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
    src = pa.to_numpy_source(prog, "f")
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 (test of generated source)
    got = ns["f"](values["a"], values["b"])
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(prog.run(values)["result"]), rtol=1e-9, atol=1e-12
    )
    assert "np.einsum('i,i->'" in src


def test_plain_fn_without_hook_still_raises():
    """A plain callable with no ``__numpy_source__`` falls through to the clear raise."""
    n = 3
    prog = Program("nohook", inputs=[SymInput("a", (n,), _prov("a"))])

    def _double(*arrs):
        return arrs[0] * 2.0

    (out,) = prog.emit_stmt(_double, [prog.input("a")], [OutSpec("result", (n,))], bulk=True)
    prog.add_output("result", out)
    with pytest.raises(NotImplementedError):
        pa.to_numpy_source(prog, "f")
