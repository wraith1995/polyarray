# PLAN — polyarray `simplify` branch

Worktree: /Users/teoc/projects/fem/polyarray-simplify   Branch: `simplify`
(off `feat/numpy-source-codegen` @3aa574d). Repo is git; remote = origin (GitHub).

Design docs: `plans/00-simplify-design.md` (mechanism — partial-eval lattice + API +
phases), `plans/01-budget-moderated-simplification.md` (control surface — post-build
`SimplifyBudget` moderating collapse↔expose, mirroring build-time `SymbolicBudget`).
Internals notes (scratch): `/private/tmp/.../scratchpad/polyarray-internals.md`.

## Goal
Post-build partial-evaluation pass on `Program`/`SymArray`: (1) reduce stmts → numeric
or expose-more-symbols, (2) replace an input with numeric or arg-dependent value,
(3) numeric-first then budget-gated, (4) propagate structural-zero sparsity (Vandermonde).
The pass is chartLib's long-specified, never-built "build-then-simplify". Additive only
(pointwise + external consumers depend on current API).

## Status
- [x] Research: polyarray internals, chartLib history, pointwise needs, oracle sparsity.
- [x] Clone + branch created.
- [x] Design docs written (`plans/00`, `plans/01`); `fold_numeric` spec sharpened.
- [x] **P0+P1+P2 first cut**: `src/polyarray/simplify.py` — `specialize`/`fold_numeric`/
      `bind_inputs`. Cascade via a `known: atom->float` map (NOT cell mutation — copy()
      shares cell arrays, so we rebuild fresh folded refs/cells). Drops folded Stmts +
      bound inputs; reindexes OutputRefs; conservative (keeps bulk/sub-Program/control-flow
      symbolic). `tests/test_simplify.py`: 4 pass (symbolic no-op, full collapse, partial
      residual-symbols, original-untouched). Full suite 57 passed / 1 skipped.
- [x] P1/P2 hardening (gaps pushed): **bulk-output folding** (record whole tensor under
      the bulk handle name; materialise folded bulk outputs), **all-numeric sub-Program /
      CallOp execution** (`_exec_fn` dispatches `.run` vs call), **dynamic-shape safety**
      (skip `is_dynamic` outputs), **OutputRef reindex** (+ hand-built test). simplify
      tests now 8 (added bulk, sub-Program, 3-stmt chain, OutputRef-remap). Full suite
      61 passed / 1 skipped.
- [ ] P6 (remaining, larger): **partial descent** into nested sub-Program / vmap CallOp
      bodies — specialize a body with the partial binds implied by its parent stmt's
      numeric inputs (today a body folds only when ALL its operands are numeric). This is
      the pointwise vmap-per-point case.
- [ ] P3 symbolic `subs` + `RationalFunction.compose`.
- [ ] P4 sparsity propagation + `block_zero_mask` (oracle M3 consumer).
- [ ] P5 `SimplifyBudget` + moderation procedure (collapse ceilings, `expose`,
      `den_degree_max`, `keep_provenance`, `analyze→budget→specialize` seam) — see plan 01.
- [ ] P6 nested-body descent + pointwise integration.

## Key anchors
- Extend, don't rebuild: `forward.partial_eval` (one-way collapse), `RF.eval` (partial
  numeric subst, `rational.py:452`), `_partial_substitute` (`rational.py:812`),
  `cells_sparsity` (`ir.py:218`), `iter_programs`/`_body_of` (`forward.py:35,47`).
- New code lands in `src/polyarray/simplify.py` (+ one `RationalFunction.compose`).
- Exactness invariant tested every phase: `E(p).run(x) == p.run({**subst, **x})`.
- Constraints: no sympy.subs at runtime; sparsity structural/rational-only/stops at
  opaque Stmt; keep `_body_of`, `run`, `evaluate`, `partial_eval(...)` stable.

## Consumers driving demand
- pointwise plan-36 (substitute-then-reduce per quad pt), plan-43 A1 (fold quad loop
  into one Program), plan-44 (default args from other args), grassmann_backend.py:369-503
  (hand-rolled walker to replace).
- oracle M3 symbolic Schur inverse (block-zero detection); tests in
  oracle/tests/test_vandermonde_m2.py.
