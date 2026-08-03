"""``polyarray.torch_batch.batched_torch`` — the torch backend (pyab → torch.vmap) matches batched_run.

Skipped unless BOTH ``torch`` and ``pyarraybackend`` are importable (optional deps; the core stack never
imports torch). torch's contraction kernels differ from numpy's by ~1 ULP, so the tolerance is loose."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarraybackend")

from dataclasses import dataclass

from polyarray import Program
from polyarray.ir import SymInput, Provenance, SymbolicBudget, OutSpec, SwitchOp
from polyarray.batch import batched_run
from polyarray.torch_batch import batched_torch, torch_available


def _prov(label="t"):
    return Provenance(kind="coeff", origin="t", index=(), label=label)


@pytest.mark.skipif(not torch_available(), reason="torch/pyarraybackend not available")
def test_batched_torch_matches_batched_run():
    # pinv (Stmt lane) composed with an einsum: result = pinv(A) @ x
    p = Program("f", inputs=[SymInput("A", (4, 4), _prov("A")), SymInput("x", (4,), _prov("x"))],
                budget=SymbolicBudget.force_stmts())
    p.add_output("result", p.input("A").pinv().einsum("ij,j->i", p.input("x")).cells)
    B = 32
    rng = np.random.default_rng(3)
    As = np.stack([rng.standard_normal((4, 4)) + 4 * np.eye(4) for _ in range(B)])
    xs = rng.standard_normal((B, 4))
    got = batched_torch(p, {"A": As, "x": xs})
    ref = batched_run(p, {"A": As, "x": xs})
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)


# Mocks of the grassmann-origin FEEC front-end ops carrying the canonical `__pyab_lower__` op-carried
# lowering hook (the twin of `numpy_source`'s `__numpy_source__`), which pyab's `_render_op` discovers by
# getattr — so plain `LowerOpts()` lowers them (no `op_lowerings` dict). The real ops are validated
# byte-identically on the FEEC residual; these give in-repo coverage without a grassmann dependency.
#
# ⚠ `Mock`-prefixed on purpose. These once carried the grassmann-era private names `_ReshapeOp` /
# `_ScaleOp` / `_AddOp` — which happened to be the (misspelled, therefore DEAD) keys `batch._apply`
# dispatched on, so `batched_run` recognized the mocks and nothing else. A local test double must not
# be able to answer to a production dispatch name; keep these distinct from anything in `polyarray.ir`.
@dataclass(frozen=True)
class _MockReshapeOp:
    shape: tuple
    def __call__(self, A): return np.asarray(A, float).reshape(self.shape)

    def __pyab_lower__(self, builder, args, low):
        c = low.core
        return [c.ReshapeExpr(a=args[0], shape=tuple(c.IntLit(value=int(d)) for d in self.shape))]


@dataclass(frozen=True)
class _MockScaleOp:
    factor: float
    def __call__(self, A): return self.factor * np.asarray(A, float)

    def __pyab_lower__(self, builder, args, low):
        c = low.core
        return [c.BinaryExpr(op=c.BinaryOp.MUL, lhs=c.FloatLit(value=float(self.factor)), rhs=args[0])]


@dataclass(frozen=True)
class _MockAddOp:
    def __call__(self, a, b): return np.asarray(a, float) + np.asarray(b, float)

    def __pyab_lower__(self, builder, args, low):
        c = low.core
        return [c.BinaryExpr(op=c.BinaryOp.ADD, lhs=args[0], rhs=args[1])]


@pytest.mark.skipif(not torch_available(), reason="torch/pyarraybackend not available")
def test_feec_op_lowerings_reshape_scale_add():
    # x -> reshape(2,3) -> scale 2.0 -> add(self) ; exercises the three `__pyab_lower__` hooks.
    # Reference is the per-element `Program.run` loop, NOT `batched_run`: this is a test of the TORCH
    # lowering, so its oracle must not route through the other batched backend.
    p = Program("g", inputs=[SymInput("x", (6,), _prov("x"))])
    (r,) = p.emit_stmt(_MockReshapeOp((2, 3)), [p.input("x")], [OutSpec("r", (2, 3))])
    (s,) = p.emit_stmt(_MockScaleOp(2.0), [r], [OutSpec("s", (2, 3))])
    (a,) = p.emit_stmt(_MockAddOp(), [s, r], [OutSpec("a", (2, 3))])
    p.add_output("result", a.cells)
    B = 16
    xs = np.random.default_rng(4).standard_normal((B, 6))
    got = batched_torch(p, {"x": xs})
    ref = np.stack([np.asarray(p.run({"x": xs[b]})["result"], float) for b in range(B)])
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)


# A canonical fold-ALL-operands lowering hook (the shape the real grassmann `_AddOp` uses): pyab must pass
# every operand to the hook so it can fold n > 2. Guards the consolidation's `_AddOp` fix — the old
# torch renderer summed only args[0]+args[1] and silently dropped args[2:] for n >= 3.
@dataclass(frozen=True)
class _SumOp:
    def __call__(self, *xs): return np.sum([np.asarray(x, float) for x in xs], axis=0)

    def __pyab_lower__(self, builder, args, low):
        c = low.core
        acc = args[0]
        for a in args[1:]:
            acc = c.BinaryExpr(op=c.BinaryOp.ADD, lhs=acc, rhs=a)
        return [acc]


@pytest.mark.skipif(not torch_available(), reason="torch/pyarraybackend not available")
def test_pyab_hook_folds_all_operands_n_gt_2():
    # a 3-operand sum through a __pyab_lower__ hook (plain LowerOpts, no op_lowerings dict).
    p = Program("sum3", inputs=[SymInput(n, (4,), _prov(n)) for n in ("x", "y", "z")])
    (s,) = p.emit_stmt(_SumOp(), [p.input("x"), p.input("y"), p.input("z")], [OutSpec("s", (4,))])
    p.add_output("result", s.cells)
    B = 8
    rng = np.random.default_rng(5)
    binds = {n: rng.standard_normal((B, 4)) for n in ("x", "y", "z")}
    got = batched_torch(p, binds)
    ref = binds["x"] + binds["y"] + binds["z"]                    # all three, not just x+y
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)


@pytest.mark.skipif(not torch_available(), reason="torch/pyarraybackend not available")
def test_switch_op_vmap_onehot_dynamic_scrutinee():
    # A per-lane (batched) integer scrutinee selects a branch under torch.vmap — the vmap-safe one-hot
    # lowering (the old `branches[int(scrutinee)]` .item()s a batched tensor and is illegal under vmap).
    p = Program("swd", inputs=[SymInput("s", (), _prov("s"))]
                + [SymInput(n, (2,), _prov(n)) for n in ("a", "b", "c")],
                budget=SymbolicBudget.force_stmts())
    (o,) = p.emit_stmt(SwitchOp(n_branches=3),
                       [p.input("s"), p.input("a"), p.input("b"), p.input("c")], [OutSpec("o", (2,))])
    p.add_output("result", o.cells)
    B = 6
    rng = np.random.default_rng(6)
    svals = np.array([0, 1, 2, 2, 1, 0], dtype=float)
    binds = {"s": svals, "a": rng.standard_normal((B, 2)),
             "b": rng.standard_normal((B, 2)), "c": rng.standard_normal((B, 2))}
    got = batched_torch(p, binds)
    branches = np.stack([binds["a"], binds["b"], binds["c"]], axis=0)     # (3, B, 2)
    ref = np.stack([branches[int(svals[i]), i] for i in range(B)])
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)
