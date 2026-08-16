"""Byte-identity tests for the general (nested / multi-op / broadcast-axis) vmap batching.

``pyab.collapse_general_vmap`` rewrites a whole ``vmap`` nest whose body is a chain of batchable
ops into ONE ``CallOp`` of a batched sub-Program, so the backend lowers batched einsums rather
than nested ``torch.vmap``.  These check the rewrite is numerically identical to the original
nested vmap on real inputs (not merely the collapse's internal random probe), across the shapes the
savo value block takes — a multilinear DOF grid ``vmap(Q)·vmap(N_u)·vmap(N_v)`` over a per-point
body, broadcast (``in_axes=None``) operands, static reshapes, and a leading-batch linalg op.
"""
from __future__ import annotations

import numpy as np
import pytest

from polyarray import pyab
from polyarray.ir import (
    EinsumStmtOp,
    InvOp,
    OutSpec,
    Program,
    Provenance,
    ReshapeOp,
    SymInput,
    vmap,
)


def _prog(name: str, *io: tuple[str, tuple[int, ...]]) -> Program:
    return Program(name=name, inputs=tuple(
        SymInput(name=n, shape=s, provenance=Provenance(kind="per_point", origin="t", index=(), label=n))
        for n, s in io))


def _frobenius_body() -> Program:
    """``x:(3,3), y:(3,3) -> <x, y>_F`` via reshape+reshape+einsum (mimics ``inner(D(u), D(v))``)."""
    b = _prog("body", ("x", (3, 3)), ("y", (3, 3)))
    (xr,) = b.emit_stmt(ReshapeOp((9,)), [b.input("x")], [OutSpec("xr", (9,))], bulk=False)
    (yr,) = b.emit_stmt(ReshapeOp((9,)), [b.input("y")], [OutSpec("yr", (9,))], bulk=False)
    (o,) = b.emit_stmt(EinsumStmtOp(spec="a,a->"), [xr, yr], [OutSpec("result", ())], bulk=False)
    b.add_output("result", o)
    return b


def _assert_collapsed_equal(top: Program, values: dict[str, np.ndarray], *,
                            expect_collapse: bool = True, tol: float = 1e-12) -> np.ndarray:
    new, collapsed = pyab.collapse_general_vmap(top)
    if expect_collapse:
        assert collapsed, "expected the nested vmap to collapse"
    else:
        assert not collapsed, "expected NO collapse"
    ref = next(iter(top.run(values).values()))
    got = next(iter(new.run(values).values()))
    assert np.shape(ref) == np.shape(got)
    assert np.allclose(ref, got, rtol=tol, atol=1e-14), float(np.abs(np.asarray(ref) - np.asarray(got)).max())
    return np.asarray(got)


def test_one_level_multi_op() -> None:
    b = _frobenius_body()
    m = 5
    top = _prog("top", ("X", (m, 3, 3)), ("Y", (m, 3, 3)))
    (o,) = top.emit_stmt(vmap(b, in_axes=(0, 0), out_axes=0),
                         [top.input("X"), top.input("Y")], [OutSpec("R", (m,))], bulk=True)
    top.add_output("R", o)
    rng = np.random.default_rng(0)
    _assert_collapsed_equal(top, {"X": rng.standard_normal((m, 3, 3)), "Y": rng.standard_normal((m, 3, 3))})


def test_broadcast_in_axis() -> None:
    """An ``in_axes=None`` operand is shared across the batch (einsum index-sharing)."""
    b = _frobenius_body()
    m = 5
    top = _prog("top", ("X", (m, 3, 3)), ("Y", (3, 3)))
    (o,) = top.emit_stmt(vmap(b, in_axes=(0, None), out_axes=0),
                         [top.input("X"), top.input("Y")], [OutSpec("R", (m,))], bulk=True)
    top.add_output("R", o)
    rng = np.random.default_rng(1)
    got = _assert_collapsed_equal(top, {"X": rng.standard_normal((m, 3, 3)), "Y": rng.standard_normal((3, 3))})
    assert got.shape == (m,)


def test_two_level_outer_product() -> None:
    b = _frobenius_body()
    nu, nv = 6, 4
    d0 = _prog("d0", ("x", (3, 3)), ("y", (nv, 3, 3)))
    (o0,) = d0.emit_stmt(vmap(b, in_axes=(None, 0), out_axes=0),
                         [d0.input("x"), d0.input("y")], [OutSpec("result", (nv,))], bulk=True)
    d0.add_output("result", o0)
    top = _prog("top", ("X", (nu, 3, 3)), ("Y", (nv, 3, 3)))
    (o,) = top.emit_stmt(vmap(d0, in_axes=(0, None), out_axes=0),
                         [top.input("X"), top.input("Y")], [OutSpec("R", (nu, nv))], bulk=True)
    top.add_output("R", o)
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((nu, 3, 3)), rng.standard_normal((nv, 3, 3))
    got = _assert_collapsed_equal(top, {"X": x, "Y": y})
    assert np.allclose(got, np.einsum("uab,vab->uv", x, y))


def test_three_level_value_block_shape() -> None:
    """``vmap(Q)·vmap(N_u)·vmap(N_v)`` — the savo velocity-block composite bind shape."""
    b = _frobenius_body()
    q, nu, nv = 7, 5, 4
    d0 = _prog("d0", ("x", (3, 3)), ("y", (nv, 3, 3)))
    (o0,) = d0.emit_stmt(vmap(b, in_axes=(None, 0), out_axes=0),
                         [d0.input("x"), d0.input("y")], [OutSpec("result", (nv,))], bulk=True)
    d0.add_output("result", o0)
    d1 = _prog("d1", ("x", (nu, 3, 3)), ("y", (nv, 3, 3)))
    (o1,) = d1.emit_stmt(vmap(d0, in_axes=(0, None), out_axes=0),
                         [d1.input("x"), d1.input("y")], [OutSpec("result", (nu, nv))], bulk=True)
    d1.add_output("result", o1)
    top = _prog("top", ("X", (q, nu, 3, 3)), ("Y", (q, nv, 3, 3)))
    (o,) = top.emit_stmt(vmap(d1, in_axes=(0, 0), out_axes=0),
                         [top.input("X"), top.input("Y")], [OutSpec("R", (q, nu, nv))], bulk=True)
    top.add_output("R", o)
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal((q, nu, 3, 3)), rng.standard_normal((q, nv, 3, 3))
    got = _assert_collapsed_equal(top, {"X": x, "Y": y})
    assert np.allclose(got, np.einsum("quab,qvab->quv", x, y))


def test_leading_batch_op_in_body() -> None:
    """A multi-op body with a leading-batch linalg op (inv) collapses."""
    b = _prog("body", ("A", (3, 3)), ("v", (3,)))
    (ai,) = b.emit_stmt(InvOp(), [b.input("A")], [OutSpec("Ai", (3, 3))], bulk=False)
    (o,) = b.emit_stmt(EinsumStmtOp(spec="ij,j->i"), [ai, b.input("v")], [OutSpec("result", (3,))], bulk=False)
    b.add_output("result", o)
    m = 6
    top = _prog("top", ("A", (m, 3, 3)), ("v", (m, 3)))
    (o,) = top.emit_stmt(vmap(b, in_axes=(0, 0), out_axes=0),
                         [top.input("A"), top.input("v")], [OutSpec("R", (m, 3))], bulk=True)
    top.add_output("R", o)
    rng = np.random.default_rng(4)
    a = rng.standard_normal((m, 3, 3)) + 3 * np.eye(3)
    v = rng.standard_normal((m, 3))
    got = _assert_collapsed_equal(top, {"A": a, "v": v})
    assert np.allclose(got, np.stack([np.linalg.inv(a[i]) @ v[i] for i in range(m)]))


def test_unbatchable_body_is_left_untouched() -> None:
    """A body op the interpreter cannot batch leaves the vmap in place (additive / safe)."""
    from polyarray.ir import SvdOp
    b = _prog("body", ("A", (3, 3)))
    # SvdOp is deliberately not batched by the interpreter -> bail.
    (u, s, vt, rk) = b.emit_stmt(SvdOp(), [b.input("A")],
                                 [OutSpec("U", (3, 3)), OutSpec("S", (3,)), OutSpec("Vt", (3, 3)),
                                  OutSpec("rank", ())], bulk=False)
    b.add_output("S", s)
    m = 4
    top = _prog("top", ("A", (m, 3, 3)))
    (o,) = top.emit_stmt(vmap(b, in_axes=(0,), out_axes=0), [top.input("A")], [OutSpec("R", (m, 3))], bulk=True)
    top.add_output("R", o)
    rng = np.random.default_rng(5)
    a = rng.standard_normal((m, 3, 3))
    new, collapsed = pyab.collapse_general_vmap(top)
    assert not collapsed
    # unchanged program still runs and equals the original
    assert np.allclose(next(iter(top.run({"A": a}).values())), next(iter(new.run({"A": a}).values())))


@pytest.mark.parametrize("collapse", [True, False])
def test_lowers_through_pyab_torch(collapse: bool) -> None:
    """The collapsed program lowers through pyab->torch to the SAME values as the nested-vmap path."""
    torch_batch = pytest.importorskip("polyarray.torch_batch")
    if not torch_batch.torch_available():
        pytest.skip("torch/pyarraybackend not available")
    import torch

    torch_batch.ensure_torch_pg()
    b = _frobenius_body()
    q, nu, nv = 3, 4, 2
    d0 = _prog("d0", ("x", (3, 3)), ("y", (nv, 3, 3)))
    (o0,) = d0.emit_stmt(vmap(b, in_axes=(None, 0), out_axes=0),
                         [d0.input("x"), d0.input("y")], [OutSpec("result", (nv,))], bulk=True)
    d0.add_output("result", o0)
    d1 = _prog("d1", ("x", (nu, 3, 3)), ("y", (nv, 3, 3)))
    (o1,) = d1.emit_stmt(vmap(d0, in_axes=(0, None), out_axes=0),
                         [d1.input("x"), d1.input("y")], [OutSpec("result", (nu, nv))], bulk=True)
    d1.add_output("result", o1)
    top = _prog("top", ("X", (q, nu, 3, 3)), ("Y", (q, nv, 3, 3)))
    (o,) = top.emit_stmt(vmap(d1, in_axes=(0, 0), out_axes=0),
                         [top.input("X"), top.input("Y")], [OutSpec("R", (q, nu, nv))], bulk=True)
    top.add_output("R", o)

    cm = pyab.compile_torch(top, name="f", opts=pyab.LowerOpts(collapse_vmap=collapse))
    rng = np.random.default_rng(6)
    x, y = rng.standard_normal((q, nu, 3, 3)), rng.standard_normal((q, nv, 3, 3))
    out = cm.module.f(torch.as_tensor(x), torch.as_tensor(y))
    got = np.asarray(out.detach() if hasattr(out, "detach") else out, dtype=float)
    assert np.allclose(got, np.einsum("quab,qvab->quv", x, y), rtol=1e-10, atol=1e-12)
