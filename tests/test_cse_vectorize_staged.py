"""Vectorized CSE emission and product-boundary (staged) fusion agree with the fused program."""
from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("REQUIRE_BENCHMARK_DEPS") == "1":
    import torch
    import grassmann as G
else:
    torch = pytest.importorskip("torch")
    G = pytest.importorskip("grassmann")

import polyarray as pa  # noqa: E402
from polyarray._staged import compile_torch_staged, fuse_staged  # noqa: E402
from polyarray.cse import compile_torch  # noqa: E402
from grassmann.stdlib.exterior import exterior_power, hodge_star, wedge  # noqa: E402


def _program(term):
    return pa.strip_asserts(G.compile(term))


def _feed(prog, n, rng):
    vals = {i.name: torch.as_tensor(rng.standard_normal((32, n))) for i in prog.inputs}
    return [vals[i.name] for i in prog.inputs]


def _exact(prog, args):
    """The program's own (unfused) value on each of the batched cells."""
    rows = []
    for c in range(args[0].shape[0]):
        out = prog.run({i.name: a[c].numpy() for i, a in zip(prog.inputs, args)})["result"]
        rows.append(np.asarray(out, dtype=float).ravel())
    return np.stack(rows)


def _terms(n):
    V = G.space(f"R{n}", n)
    a, b, c, d = (G.var(s, V) for s in "abcd")
    return {
        "wedge": wedge(a, b, 1, 1),
        "star_ab": hodge_star(wedge(a, b, 1, 1), 2),
        "gram": G.inner(exterior_power(V, 2)).at([0, 1])(wedge(a, b, 1, 1), wedge(c, d, 1, 1)),
        "triple": hodge_star(wedge(wedge(a, b, 1, 1), c, 2, 1), 3),
    }


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("kind", ["wedge", "star_ab", "gram", "triple"])
def test_vectorized_and_staged_match_program(n, kind):
    prog = _program(_terms(n)[kind])
    args = _feed(prog, n, np.random.default_rng(0))
    want = _exact(prog, args)
    scale = np.abs(want).max()
    for f in (compile_torch(pa.fuse_to_rational(prog), vectorize=True).f, compile_torch_staged(prog).f):
        got = f(*args).reshape(32, -1).numpy()
        assert np.abs(got - want).max() / scale < 1e-9   # CSE snaps √-constants to ~1e-10


def test_vectorized_emits_index_arrays_not_a_stack():
    prog = _program(_terms(3)["wedge"])
    src = compile_torch(pa.fuse_to_rational(prog), vectorize=True).source
    assert "torch.stack" not in src and "[..., [" in src


def test_boundary_keeps_minor_products_factored():
    """⟨a∧b, c∧d⟩ stays a sum of products of 2×2 minors: 3 + 3 stages, 24 FLOPs in ℝ³."""
    prog = _program(_terms(3)["gram"])
    stages, _ = fuse_staged(prog)
    assert len(stages) == 6
    assert sum(compile_torch_staged(prog).flops) == 24
