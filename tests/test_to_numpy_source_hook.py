"""Tests for the ``__numpy_source__`` emit hook in :func:`polyarray.to_numpy_source`.

A ``Stmt.fn`` — a front-end op instance, or a plain Python callable that is not an op
class, sub-``Program``, or ``vmap`` closure — emits to standalone numpy IFF it carries a
``__numpy_source__(args) -> str`` attribute (``args`` = the rendered operand
expression strings; returns a numpy expression string).  An explicit ``op_renderers=``
entry takes precedence over the hook; a hookless fn covered by neither still raises the
clear ``NotImplementedError``.  Each test builds a ``Program``
directly, emits, ``exec``s, and compares to ``program.run`` (the acceptance bar).
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class _ScaleOp:
    """A front-end-style typed op carrying its own ``__numpy_source__``."""

    factor: float = 2.0

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float) * self.factor

    def __numpy_source__(self, args: list[str]) -> str:
        """Emit ``a * factor``.

        Parameters
        ----------
        args
            The rendered operand expressions, in operand order.

        Returns
        -------
        str
            A numpy expression for the scaled operand.
        """
        return f"(np.asarray({args[0]}, dtype=float) * {self.factor!r})"


def _scale_prog(n: int) -> Program:
    prog = Program("scaled", inputs=[SymInput("a", (n,), _prov("a"))])
    (out,) = prog.emit_stmt(
        _ScaleOp(), [prog.input("a")], [OutSpec("result", (n,))], bulk=True
    )
    prog.add_output("result", out)
    return prog


def test_op_class_hook_emits_without_op_renderers():
    """A front-end OP (not just a plain callable) is discovered through its own hook.

    This is what lets an observability dump render a program carrying e.g. grassmann's
    ``_QrSignConventionOp`` with no ``op_renderers=`` threading at the call site.
    """
    n = 4
    prog = _scale_prog(n)
    rng = np.random.default_rng(1)
    values = {"a": rng.standard_normal(n)}
    src = pa.to_numpy_source(prog, "f")
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 (test of generated source)
    np.testing.assert_allclose(
        np.asarray(ns["f"](values["a"])), np.asarray(prog.run(values)["result"]),
        rtol=1e-9, atol=1e-12,
    )


def test_explicit_op_renderers_take_precedence_over_the_hook():
    """``op_renderers=`` wins over an op's own ``__numpy_source__``."""
    prog = _scale_prog(3)
    src = pa.to_numpy_source(
        prog, "f", op_renderers={"_ScaleOp": lambda op, args: f"({args[0]} * 100.0)"}
    )
    assert "* 100.0" in src and "* 2.0" not in src
