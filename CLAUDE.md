# polyarray — agent guide

Symbolic-numeric array IR: a `Program` of `Stmt`s over `SymArray`s whose cells are
**RationalFunctions | floats | Refs to opaque ops**. The lowering target for grassmann and
the value substrate for chartlib's symbolic geometry. Stack rules: `../../PHILOSOPHY.md`
(or `fem/PHILOSOPHY.md`). Authoritative API surface: **`PUBLIC_API.md`** (committed
contract — additions need it updated).

## Capability inventory — CHECK HERE BEFORE WRITING ANYTHING

| You are about to write… | It exists: |
|---|---|
| a constant-folder / partial evaluator | `simplify.specialize(program, bind=, subs=, budget=)`; wrappers `fold_numeric`, `bind_inputs`, `substitute` |
| an IR walker / cost report | `forward.analyze(program) -> IRReport` (walks nested sub-Programs + vmap bodies via `iter_programs`) |
| staged compile logging / dumps / "why did this blow up" | `observe.py` — **read `OBSERVABILITY.md`**. `FEM_OBSERVE` ∈ off\|warn\|info\|debug\|dump; Δ-waterfall report + `peak_stage`/`jump_stage`/`rational_stage`; `dump_dir_for(name)` feeds `pyab.compile_torch(dump_dir=)` so symbolic + generated-torch artifacts share one numbered dir. Instrument in pointwise/savo/oracle ONLY — never grassmann/chartLib |
| a "collapse big cells to numeric" pass | `forward.partial_eval(program, max_cell_size=)` — both eval families (`forward` and `simplify`) are current; use whichever fits (cost-driven collapse vs bind/subs-driven folding) |
| symbolic substitution | `RationalFunction.compose`/`compose_multi`; `substitute` / `specialize(subs=)` |
| a zero/sparsity mask | `sparsity.propagate_sparsity(program)`, `block_zero_mask` (structural; opaque ops reset the mask — false *positives* are bugs) |
| batching over points | `vmap(body, in_axes=, out_axes=)` — one Stmt for M points, not M calls |
| numpy code emission | `numpy_source.to_numpy_source(program, op_renderers=)` — front-end ops emit via `op_renderers` keyed by class name (deliberate: no reverse dependency) |
| a degree estimate (per cell) | `RationalFunction.compute_degree(degrees=)`; backend `total_degree()`; consumed by `SymbolicBudget.inverse_max_degree`/`naive_inverse_max_size` gates |
| a degree estimate (whole program) | `degree.program_degree(program, seed, gen_deg=, *_ops=)` (fem #9): statement-graph degree trace — multilinear SUM, passthrough MAX, det(n×n, deg d) = n·d, vmap `CallOp` recursed, rational/algebraic on a seed → ∞ (mathematically right for a POLYNOMIAL degree; over-estimation safe). Front ends extend the op categories per call; the FE seeding + affine geometry gate stay in pointwise `estimate_degree` (domain knowledge) |
| a size/complexity budget | build-time `SymbolicBudget` (`ir.py`; presets `legacy`, `build_big_symbols`, `force_stmts`) vs post-build `SimplifyBudget` (`budget.py`; presets `none`, `numeric_only`, `balanced`, `expose_symbols`) — two different knobs, don't conflate |
| a matrix inverse/det with fallback | `SymArray.det/inverse/solve/pinv` already do three-lane dispatch (closed-form Bareiss in budget → Stmt emission → numeric short-circuit) |
| control flow / dynamic shapes | `IntAtom`, `SwitchOp`, `WhileOp`, `CallOp`; `DimAtom` + `is_dynamic` for runtime ranks (e.g. SVD δ_f) |
| gcd cleanup | `RationalFunction.clean`/`try_cancel`; opt-in `eager_cancel` (added for the sparsity-mask lane) |

## Conventions & gotchas

- **One program, two lanes**: never fork numeric vs symbolic code paths; lane choice is a
  budget/value property.
- Op vocabulary is the ~20 frozen `Stmt.fn` dataclasses in `ir.py` (incl. `GSvdOp` with its
  documented reconstruction contract, `IdentityOp` = capture/freeze). New ops are rare
  events: update `PUBLIC_API.md`, sparsity handling (default: mask reset), and
  `to_numpy_source` rendering together.
- Poly backends selected by env: `CHARTLIB_POLY_BACKEND` ∈ sympy|flint|native_py|native_cpp,
  `CHARTLIB_POLY_COEFF` ∈ double|mpf|quad. `native_cpp` needs `make cython`. The
  `CHARTLIB_` prefix is **historical** (vendored from chartLib's `_symbolic`) — do NOT
  "fix" the names; both repos read them.
- **python-flint leak**: the global ring-context cache grows unboundedly on long symbolic
  runs — use `clear_ring_caches` between heavy passes; run big jobs under
  `~/.claude/bin/runjob`.
- `specialize(sparsity=)` is a no-op passthrough (API parity) — use
  `sparsity.propagate_sparsity` directly.
- `Program.fingerprint()` intentionally does not exist (PUBLIC_API.md "Not included").

## Layout & tests

`src/polyarray/`: `ir.py` (Program/SymArray/Stmt/ops/SymbolicBudget), `rational.py`,
`simplify.py`, `forward.py`, `sparsity.py`, `budget.py`, `numpy_source.py`,
`poly_backend.py` + native backends. Design docs: `plans/00-simplify-design.md`
(Numeric ⊏ Symbolic ⊏ Opaque lattice), `plans/01-budget-moderated-simplification.md`.
Tests: `pytest -n auto` in this repo's `.venv`. mypy: configured in pyproject (chartLib-model flags); baseline 52 errors (2026-07-03) — shrink, don't grow. Audit: `python3 savo/audit/workspace/run_audits.py
--repos polyarray` (from `fem/`; adjust if top-level checkout).
