"""Common-subexpression emission for a fused, statement-free polynomial program.

:func:`polyarray.fuse_to_rational` collapses a program to one rational function per output cell,
but pyab renders each cell's monomials independently — a grassmann ``⋆(a∧b∧c)`` over ℝ⁴ emits 380
multiply-adds where its 60 real ones share sub-products, and the ℝ⁴ volume expands past what a
compiler will finish.  This module re-emits the polynomial outputs with common subexpressions
shared:

  1. read each output cell's monomials off its :class:`~polyarray.rational.RationalFunction`
     (Flint ``num.terms()`` / constant ``den``) as float-coefficient terms,
  2. drop terms whose coefficient is numerically zero (the √-normalisation constants reach the
     exact-rational lane as inexact giant fractions, leaving spurious ~1e-17 monomials),
  3. **factor** a per-output shared coefficient magnitude (the one √-constant), and
  4. **share** products greedily — the most-frequent co-occurring atom pair becomes a temporary,
     repeated until every monomial is a single leaf.

The result is straight-line torch code that batches over any leading axes (each atom is read as
``arg[..., component]``), so it needs no ``vmap``.  Only a *polynomial* program folds — a
non-constant denominator (a genuine rational function) raises :class:`CseError`.

This is a numeric pass: cleaning the inexact √-coefficients trades the exact-rational form for the
float form the inputs already carried, and returns a kernel that matches the fused program to
float precision.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .ir import Program
from .rational import RationalFunction


class CseError(Exception):
    """The program is not a finite polynomial (e.g. a non-constant denominator) — cannot CSE."""


@dataclass
class CseModule:
    """A compiled CSE kernel: ``module.f`` is the batched callable (accepts ``(n_cells, dim)`` per
    input), ``source`` the emitted straight-line code, ``flops`` its ``(mul, add/sub)`` count.

    Note: the CSE pass minimises the operation count and matches the hand-written kernel's FLOPs,
    but on a memory/dispatch-bound target that does not translate to speed — a structured
    broadcast kernel with more FLOPs can vectorise better than this share-heavy scalar form. CSE's
    payoff is op-count and compilability (it shrinks otherwise-uncompilable expanded polynomials),
    not runtime; see the benchmark notes.
    """

    f: object
    source: str
    n_temps: int
    flops: tuple[int, int]


def _output_polys(program: Program, tol: float):
    """Per output cell: ``[(coeff_float, sorted atom-index tuple)]`` over a shared atom list.

    Raises :class:`CseError` on a non-constant denominator or a non-``RationalFunction`` cell.
    """
    cells = []
    for sa in program.outputs.values():
        arr = np.asarray(sa.cells)
        cells.extend(arr.reshape(-1).tolist())
    atoms: list[str] = []
    seen: dict[str, int] = {}

    def atom_id(name: str) -> int:
        if name not in seen:
            seen[name] = len(atoms)
            atoms.append(name)
        return seen[name]

    polys: list[list[tuple[float, tuple[int, ...]]]] = []
    for cell in cells:
        if not isinstance(cell, RationalFunction):
            polys.append([(float(cell), ())])
            continue
        den = cell.den.terms()
        if len(den) != 1 or any(e != 0 for e in den[0][0]):
            raise CseError("output denominator is not constant — not a polynomial")
        dval = float(den[0][1])
        gens = cell.gens
        terms = [(float(coeff) / dval,
                  tuple(sorted(atom_id(gens[k]) for k, e in enumerate(monom) for _ in range(e))))
                 for monom, coeff in cell.num.terms()]
        m = max((abs(c) for c, _ in terms), default=1.0)
        polys.append([(c, at) for c, at in terms if abs(c) > tol * m] or [(0.0, ())])
    return polys, atoms


def _greedy_share(polys):
    """Greedy pairwise product sharing.  Returns ``(temps, termbags)``.

    ``temps`` are ``(id, leaf1, leaf2)`` in emit order; each output's ``termbags`` entry is a list
    of ``[coeff, [leaf-ids]]`` reduced until every product is a single leaf.
    """
    n_atoms = 1 + max((i for p in polys for _, at in p for i in at), default=-1)
    next_id = n_atoms
    termbags = [[[c, list(at)] for c, at in p] for p in polys]
    temps: list[tuple[int, int, int]] = []
    while True:
        cnt: Counter = Counter()
        pending = False
        for bag in termbags:
            for _, ids in bag:
                if len(ids) > 1:
                    pending = True
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            cnt[tuple(sorted((ids[i], ids[j])))] += 1
        if not pending:
            break
        (l1, l2), _ = max(cnt.items(), key=lambda kv: kv[1])
        tid = next_id
        next_id += 1
        temps.append((tid, l1, l2))
        for bag in termbags:
            for term in bag:
                ids = term[1]
                # a squared pair (l1 == l2) needs two copies present, not just one
                have = ids.count(l1) >= 2 if l1 == l2 else (l1 in ids and l2 in ids)
                if have:
                    ids.remove(l1)
                    ids.remove(l2)
                    ids.append(tid)
    return temps, termbags


def _atom_resolver(program: Program):
    """Map a generator name → a torch subscript on its input.

    Each generator is named ``<input>_<i0>[_<i1>…]`` — one integer index per input axis — so a
    vector input ``v`` yields ``v_0``.  Resolve against the declared inputs (longest name first,
    exact rank match) so the subscript matches the input's actual shape.
    """
    shapes = sorted(((i.name, tuple(i.shape)) for i in program.inputs),
                    key=lambda t: len(t[0]), reverse=True)

    def resolve(name: str) -> str:
        for inm, shape in shapes:
            if not shape and name == inm:
                return inm
            prefix = inm + "_"
            if name.startswith(prefix):
                rest = name[len(prefix):].split("_")
                if len(rest) == len(shape) and all(p.isdigit() for p in rest):
                    return f"{inm}[..., {', '.join(str(int(p)) for p in rest)}]"
        raise CseError(f"generator {name!r} does not resolve to an input component")

    return resolve


def _emit_source(program: Program, atoms, polys) -> str:
    argnames = [i.name for i in program.inputs]
    resolve = _atom_resolver(program)
    temps, termbags = _greedy_share(polys)
    leaf: dict[int, str] = {i: resolve(atoms[i]) for i in range(len(atoms))}
    lines = []
    for tid, l1, l2 in temps:
        lines.append(f"    t{tid} = {leaf[l1]} * {leaf[l2]}")
        leaf[tid] = f"t{tid}"

    def sign_only(c: float, base: str, first: bool) -> str:
        """Emit ``base`` with only its sign — magnitude carried elsewhere (a shared factor, or a
        snapped ±1). This is also what discards the ~1e-10 √-normalisation residue on unit
        coefficients, so it must never print the magnitude."""
        if first:
            return f"-{base}" if c < 0 else base
        return f"{'-' if c < 0 else '+'} {base}"

    outs = []
    for o, bag in enumerate(termbags):
        # A single shared magnitude groups the terms: factor it out front (unless it is 1) and emit
        # each term sign-only. With mixed magnitudes, emit explicit coefficients but still fold a
        # coefficient that is ±1 (to the √-residue tolerance) down to its sign.
        mags = sorted({round(abs(c), 9) for c, _ in bag})
        common = mags[0] if len(mags) == 1 and mags[0] != 0.0 else None
        pieces = []
        for k, (c, ids) in enumerate(bag):
            base = leaf[ids[0]] if ids else "1.0"
            if common is not None or abs(abs(c) - 1.0) < 1e-6:
                pieces.append(sign_only(c, base, first=(k == 0)))
            else:
                pieces.append(f"{'+ ' if k else ''}({c:+.17g}*{base})")
        body = " ".join(pieces)
        src = f"{common:.17g}*({body})" if common is not None and abs(common - 1.0) > 1e-6 else body
        lines.append(f"    o{o} = {src}")
        outs.append(f"o{o}")
    stack = f"torch.stack([{', '.join(outs)}], dim=-1)" if len(outs) > 1 else outs[0]
    return f"def f({', '.join(argnames)}):\n" + "\n".join(lines) + f"\n    return {stack}\n"


def _count_flops(src: str) -> tuple[int, int]:
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef))
    mul = add = 0
    for n in ast.walk(fn):
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Mult):
                mul += 1
            elif isinstance(n.op, (ast.Add, ast.Sub)):
                add += 1
    return mul, add


def _component_of(program: Program):
    """Map a generator name → ``(input name, component)`` for a vector input, else ``None``."""
    shapes = sorted(((i.name, tuple(i.shape)) for i in program.inputs),
                    key=lambda t: len(t[0]), reverse=True)

    def component(name: str) -> tuple[str, int] | None:
        for inm, shape in shapes:
            prefix = inm + "_"
            if len(shape) == 1 and name.startswith(prefix) and name[len(prefix):].isdigit():
                return inm, int(name[len(prefix):])
        return None

    return component


def _emit_slots(program: Program, atoms, polys) -> tuple[str, tuple[int, int]] | None:
    """Emit every output component at once over index arrays, or ``None`` when the shape forbids it.

    Applies when the outputs share one *term structure*: each output has the same number of
    monomials, and after ordering (positive terms first, then by input and component) the t-th
    monomial of every output reads the same inputs in the same order — true of minors, cross
    products and signed permutations.  The t-th monomials across outputs then form one *slot*,
    emitted as a product of index-array reads ``a[..., [i₀, i₁, …]] * b[..., [j₀, j₁, …]]`` whose
    trailing axis is the output components; slots combine by ``+``/``−``.  This performs the same
    per-component arithmetic as the scalar form but hands the compiler a component axis to
    vectorize instead of a ``stack`` of scalars.  Products are not shared *across* monomials here,
    so an output whose scalar form shares sub-products may cost more FLOPs in this form.

    Returns ``(source, (mul, add))`` with FLOPs counted per cell (per output component lane).
    """
    if len(polys) < 2:
        return None
    component = _component_of(program)
    order = {i.name: k for k, i in enumerate(program.inputs)}
    rows = []
    for poly in polys:
        terms = []
        for c, at in poly:
            parts = [component(atoms[a]) for a in at]
            if not parts or any(p is None for p in parts):
                return None
            parts.sort(key=lambda p: (order[p[0]], p[1]))
            terms.append((c, tuple(parts)))
        terms.sort(key=lambda t: (t[0] < 0, [(order[p[0]], p[1]) for p in t[1]]))
        rows.append(terms)
    n_slots = len(rows[0])
    if any(len(r) != n_slots for r in rows):
        return None
    pattern = [[p[0] for p in rows[0][t][1]] for t in range(n_slots)]
    if any([p[0] for p in r[t][1]] != pattern[t] for r in rows for t in range(n_slots)):
        return None

    lanes = len(rows)
    first = program.inputs[0].name
    mags = sorted({round(abs(c), 9) for r in rows for c, _ in r})
    common = mags[0] if len(mags) == 1 else None
    signs = [[1 if r[t][0] > 0 else -1 for r in rows] for t in range(n_slots)]
    uniform = common is not None and all(len(set(s)) == 1 for s in signs)
    mul = add = 0
    pieces = []
    for t in range(n_slots):
        reads = [f"{inm}[..., {[r[t][1][p][1] for r in rows]}]" for p, inm in enumerate(pattern[t])]
        mul += len(reads) - 1
        if uniform:
            sign = "-" if signs[t][0] < 0 else "+"
            pieces.append(f"{sign} {' * '.join(reads)}" if t else
                          f"{'-' if sign == '-' else ''}{' * '.join(reads)}")
            add += 1 if t else 0
        else:
            coeffs = [(s * common if common is not None else r[t][0])
                      for s, r in zip(signs[t], rows)]
            reads.append(f"{first}.new_tensor({[float(c) for c in coeffs]!r})")
            mul += 1
            pieces.append(f"{'+ ' if t else ''}{' * '.join(reads)}")
            add += 1 if t else 0
    body = " ".join(pieces)
    if uniform and abs(common - 1.0) > 1e-6:
        body = f"{common:.17g}*({body})"
        mul += 1
    src = (f"def f({', '.join(i.name for i in program.inputs)}):\n"
           f"    return {body}\n")
    return src, (mul * lanes, add * lanes)


def compile_torch(program: Program, *, tol: float = 1e-9, vectorize: bool = False) -> CseModule:
    """Emit a common-subexpression-shared torch kernel for a fused polynomial ``program``.

    ``program`` is typically the output of :func:`fuse_to_rational`.  Returns a :class:`CseModule`
    whose ``f`` batches over any leading axes and matches the program to float precision.

    ``vectorize`` emits all output components at once over index arrays (:func:`_emit_slots`)
    when the outputs share one term structure, falling back to the scalar shared form otherwise;
    ``flops`` then counts per cell across the component lanes.
    """
    import torch  # noqa: F401  (used by the emitted source)

    polys, atoms = _output_polys(program, tol)
    slots = _emit_slots(program, atoms, polys) if vectorize else None
    if slots is not None:
        src, flops = slots
    else:
        src = _emit_source(program, atoms, polys)
        flops = _count_flops(src)
    n_temps = sum(1 for ln in src.splitlines() if ln.lstrip().startswith("t") and "=" in ln)
    ns: dict = {}
    exec(compile(src, "<cse>", "exec"), {"torch": torch}, ns)  # noqa: S102 — our own generated code
    return CseModule(f=ns["f"], source=src, n_temps=n_temps, flops=flops)
