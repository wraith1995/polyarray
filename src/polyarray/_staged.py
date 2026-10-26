"""Fusion with a product boundary: keep a product of sums a product.

:func:`~polyarray.fuse_to_rational` collapses a program to one polynomial per output, which
multiplies out every product — a factored form the front end emitted (a Gram determinant of dot
products, a cofactor expansion) is expanded into its monomials and cannot be recovered by product
sharing alone.  :func:`fuse_staged` runs the same exact-fold engine but stops at one boundary:
before a multilinear statement (one ``_zero_absorbing`` op) that multiplies two or more
input-dependent operands, each operand cell that is a multi-term polynomial is bound to a fresh
*stage* generator ``__sN`` instead of being substituted.  Each stage stands for the operand's
*normalized* polynomial — its first coefficient scaled to 1 — so the normalization constant and
sign stay outside, where downstream products combine them, and a negated or rescaled copy of the
same polynomial reuses one stage.

The result is exact (stages are definitions, not approximations).  :func:`compile_torch_staged`
emits it as straight-line torch: the stages first, then the outputs over inputs and stages, with
the greedy product sharing of :mod:`polyarray.cse` applied to each block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import numpy as np

from . import cse
from . import exact_fold as ef
from .ir import Const, InputRef, IntAtomRef, OutputRef, Program, RationalRef, Ref, SymArrayRef, is_builtin_op
from .rational import RationalFunction


def _is_multi_term_poly(x: object) -> bool:
    if not isinstance(x, RationalFunction) or not x.gens:
        return False
    den = x.den.terms()
    return len(x.num.terms()) > 1 and len(den) == 1 and all(e == 0 for e in den[0][0])


def _depends_on_inputs(v: object) -> bool:
    return any(isinstance(c, RationalFunction) and c.gens
               for c in np.asarray(v, dtype=object).ravel())


def _stored_array(ref: Ref, env: ef._Env) -> np.ndarray | None:
    """The env array a statement operand reads, when it is a statement result; else ``None``."""
    match ref:
        case OutputRef():
            return env.outs.get((ref.stmt_idx, ref.out_idx))
        case SymArrayRef():
            return env.bulk.get(ref._bulk.name) if ref._bulk is not None else None
        case InputRef() | Const() | RationalRef() | IntAtomRef():
            return None                       # an input atom, a constant or a runtime integer
        case _ as unreachable:
            assert_never(unreachable)


def _canonical(cell: RationalFunction) -> tuple[float, RationalFunction, object]:
    """``cell = scale · prim`` with ``prim``'s first term's coefficient 1; a key for reuse."""
    terms = sorted(cell.num.terms(), key=lambda t: t[0])
    q = terms[0][1] / cell.den.terms()[0][1]
    scale = float(int(q.p)) / float(int(q.q))           # ring constants embed as floats anyway
    prim = cell * RationalFunction.constant(1.0 / scale)
    key = (cell.gens,
           tuple((m, round(float(c), 9)) for m, c in sorted(prim.num.terms(), key=lambda t: t[0])),
           round(float(prim.den.terms()[0][1]), 9))
    return scale, prim, key


@dataclass
class _Stages:
    defs: list[tuple[str, RationalFunction]]
    memo: dict[object, RationalFunction]

    def bind(self, arr: np.ndarray, env: ef._Env) -> None:
        """Replace each multi-term cell of ``arr`` (in place) by ``scale · __sN``."""
        for idx in (np.ndindex(*arr.shape) if arr.shape else [()]):
            cell = arr[idx]
            if not _is_multi_term_poly(cell):
                continue
            scale, prim, key = _canonical(cell)
            if key not in self.memo:
                name = f"__s{len(self.defs)}"
                self.defs.append((name, prim))
                self.memo[key] = RationalFunction.atom(name)
            staged = self.memo[key] * RationalFunction.constant(scale)
            for g, v in list(env.atom.items()):            # cell-based readers resolve via env.atom
                if v is cell:
                    env.atom[g] = staged
            arr[idx] = staged


def fuse_staged(program: Program) -> tuple[list[tuple[str, RationalFunction]], list[object]]:
    """Fuse ``program`` with a product boundary; return ``(stages, output cells)``.

    Output cells (all outputs, flattened in order) are polynomials over the program inputs and the
    stage generators; each stage is a polynomial over the inputs and earlier stages.
    """
    meter = ef._Meter(limit=ef._resolve_work_budget(None), wall_deadline=ef._wall_deadline(None))
    env, reason = ef._Env(), ef._Reason()
    stages = _Stages(defs=[], memo={})
    for i, stmt in enumerate(program.statements):
        if stmt.fn is not None and is_builtin_op(stmt.fn) and ef._zero_absorbing(stmt.fn):
            vals = [ef._resolve_ref(r, env, program) for r in stmt.in_]
            dependent = [j for j, v in enumerate(vals) if v is not ef._OPAQUE and _depends_on_inputs(v)]
            if len(dependent) >= 2:
                for j in dependent:
                    arr = _stored_array(stmt.in_[j], env)
                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        stages.bind(arr, env)
        reason.value = None
        if not ef._exec_stmt(program, i, stmt, env, meter, 0, reason, ef._MAX_SYM_MASS):
            fn = type(stmt.fn).__name__ if stmt.fn is not None else "fn-less"
            raise ef.FusionError(f"statement {i} ({fn}) is not finitely foldable: {reason.take(fn)}")
    for (i, k), val in list(env.outs.items()):              # as in fuse_to_rational
        arr = np.asarray(val)
        cells = np.asarray(program.statements[i].out[k].cells)
        if cells.shape != arr.shape:
            continue
        for idx in (np.ndindex(*cells.shape) if cells.shape else [()]):
            cell = cells[idx] if cells.shape else cells[()]
            if isinstance(cell, RationalFunction) and len(cell.gens) == 1:
                env.atom.setdefault(cell.gens[0], arr[idx] if arr.shape else arr[()])
    outs = [np.asarray(ef._fused_output(sa, env, program), dtype=object).ravel()
            for sa in program.outputs.values()]
    return stages.defs, [c for o in outs for c in o]


def _polys(cells: list[object], tol: float):
    atoms: list[str] = []
    seen: dict[str, int] = {}

    def atom_id(name: str) -> int:
        if name not in seen:
            seen[name] = len(atoms)
            atoms.append(name)
        return seen[name]

    polys = []
    for cell in cells:
        if not isinstance(cell, RationalFunction):
            polys.append([(float(cell), ())])
            continue
        dval = float(cell.den.terms()[0][1])
        terms = [(float(c) / dval, tuple(sorted(atom_id(cell.gens[k]) for k, e in enumerate(m) for _ in range(e))))
                 for m, c in cell.num.terms()]
        mx = max((abs(c) for c, _ in terms), default=1.0)
        polys.append([(c, a) for c, a in terms if abs(c) > tol * mx] or [(0.0, ())])
    return polys, atoms


def _emit_block(polys, atoms, leaf_src, lhs: list[str], prefix: str) -> list[str]:
    temps, bags = cse._greedy_share(polys)
    leaf = {i: leaf_src(atoms[i]) for i in range(len(atoms))}
    lines = []
    for tid, l1, l2 in temps:
        lines.append(f"    {prefix}{tid} = {leaf[l1]} * {leaf[l2]}")
        leaf[tid] = f"{prefix}{tid}"
    for name, bag in zip(lhs, bags):
        mags = sorted({round(abs(c), 9) for c, _ in bag})
        common = mags[0] if len(mags) == 1 and mags[0] != 0.0 else None
        pieces = []
        for k, (c, ids) in enumerate(bag):
            base = leaf[ids[0]] if ids else "1.0"
            if common is not None or abs(abs(c) - 1.0) < 1e-6:
                pieces.append((f"-{base}" if c < 0 else base) if k == 0 else f"{'-' if c < 0 else '+'} {base}")
            else:
                pieces.append(f"{'+ ' if k else ''}({c:+.17g}*{base})")
        body = " ".join(pieces)
        if common is not None and abs(common - 1.0) > 1e-6:
            body = f"{common:.17g}*({body})"
        lines.append(f"    {name} = {body}")
    return lines


def compile_torch_staged(program: Program, *, tol: float = 1e-9) -> cse.CseModule:
    """:func:`fuse_staged` then emit straight-line torch (stages, then outputs)."""
    import torch

    stages, outputs = fuse_staged(program)
    resolve = cse._atom_resolver(program)
    leaf_src = lambda n: n if n.startswith("__s") else resolve(n)  # noqa: E731
    lines: list[str] = []
    if stages:
        polys, atoms = _polys([rf for _, rf in stages], tol)
        lines += _emit_block(polys, atoms, leaf_src, [n for n, _ in stages], "p")
    polys, atoms = _polys(outputs, tol)
    outs = [f"o{k}" for k in range(len(outputs))]
    lines += _emit_block(polys, atoms, leaf_src, outs, "q")
    ret = f"torch.stack([{', '.join(outs)}], dim=-1)" if len(outs) > 1 else outs[0]
    src = (f"def f({', '.join(i.name for i in program.inputs)}):\n" + "\n".join(lines)
           + f"\n    return {ret}\n")
    ns: dict = {}
    exec(compile(src, "<cse-staged>", "exec"), {"torch": torch}, ns)  # noqa: S102 — our own code
    return cse.CseModule(f=ns["f"], source=src, n_temps=len(stages), flops=cse._count_flops(src))
