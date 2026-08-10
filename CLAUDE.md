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
| a DEAD-INPUT elimination (an input no statement/output reads) | `simplify._drop_unread_inputs(prog)` — prunes them with the matching operand/`in_axes` entries and binds unbatched numeric operands into a vmap body. Written 2026-08-03: one dead input inside a vmap closure is enough to stop a whole Stmt folding |
| a contraction that must stay DEFERRED | `SymArray.einsum` routes to `runtime_einsum_multi` when any operand is bulk (one `EinsumStmtOp`, bulk out). ⚠ `matmul`/`matvec`/`det`/`inverse`/`pinv` have NO bulk branch — they read `.cells` and densify |
| a dense array CONSTANT in emitted code | pyab `ConstArrayExpr` (one node, base64 IEEE-754 payload, content-addressed so CSE dedupes). ⚠ codegen must HOIST it to a module binding — an inline decode puts `binascii` in the traced frame and Dynamo's unresumable graph break silently disables Inductor |
| numpy code emission | `numpy_source.to_numpy_source(program, op_renderers=)` — a front-end op emits via its own `__numpy_source__(self, args)` hook (twin of pyab's `__pyab_lower__`; found with no threading) or via `op_renderers` keyed by class name, which wins (deliberate: no reverse dependency). A program's FREE feed atoms (generators no input and no Stmt binds) become trailing parameters. A supplied key spelling one of OUR ops privately (`_AxisLenOp` for `AxisLenOp`) can never match and warns (`DeadOpKeyWarning`), on this entry point and on `pyab.LowerOpts(op_lowerings=)` |
| a degree estimate (per cell) | `RationalFunction.compute_degree(degrees=)`; backend `total_degree()`; consumed by `SymbolicBudget.inverse_max_degree`/`naive_inverse_max_size` gates |
| a degree estimate (whole program) | `degree.program_degree(program, seed, gen_deg=, *_ops=)` (fem #9): statement-graph degree trace — multilinear SUM, passthrough MAX, det(n×n, deg d) = n·d, vmap `CallOp` recursed, rational/algebraic on a seed → ∞ (mathematically right for a POLYNOMIAL degree; over-estimation safe). Front ends extend the op categories per call; the FE seeding + affine geometry gate stay in pointwise `estimate_degree` (domain knowledge) |
| a size/complexity budget | build-time `SymbolicBudget` (`ir.py`; presets `legacy`, `build_big_symbols`, `force_stmts`) vs post-build `SimplifyBudget` (`budget.py`; presets `none`, `numeric_only`, `balanced`, `expose_symbols`) — two different knobs, don't conflate |
| a bound on the EXACT lane | `exact_fold`'s `work_budget=` — deterministic WORK UNITS (`_Meter`, ~one monomial touched; ~128 000/second, default `_DEFAULT_WORK_BUDGET = 4_000_000`, `0` = unbounded, env `POLYARRAY_EXACT_WORK_BUDGET`), **not seconds**: a certificate must not record the load average. ⚠ `exact_partial_eval`/`exact_fold_cells` take `work_budget=` only; `partial_eval_numeric` still accepts `time_budget=` but it now sizes the loud wall-clock BACKSTOP (`NonDeterministicFoldWarning`), not what folds |
| a matrix inverse/det with fallback | `SymArray.det/inverse/solve/pinv` already do three-lane dispatch (closed-form Bareiss in budget → Stmt emission → numeric short-circuit) |
| control flow / dynamic shapes | `IntAtom`, `SwitchOp`, `WhileOp`, `CallOp`; `DimAtom` + `is_dynamic` for runtime ranks (e.g. SVD δ_f) |
| gcd cleanup | `RationalFunction.clean`/`try_cancel`; opt-in `eager_cancel` (added for the sparsity-mask lane) |

## Conventions & gotchas

- **One program, two lanes**: never fork numeric vs symbolic code paths; lane choice is a
  budget/value property.
- Op vocabulary is the **56** frozen `Stmt.fn` dataclasses in `ir.py` (incl. `GSvdOp` with
  its documented reconstruction contract, `IdentityOp` = capture/freeze), named by the
  closed union **`ir.StmtFn`**. New ops are rare events: add to `StmtFn`, then run mypy —
  every `match` closed by `assert_never` will name itself. Also update `PUBLIC_API.md`,
  `__init__` exports, `degree.DEFAULT_DEGREE_KINDS`, and the pyab / `to_numpy_source`
  render lanes; `tests/test_op_union.py` checks each mirror.
- **Never dispatch over ops with a bare isinstance ladder.** `Stmt.fn` is half-open: handle
  the open half (sub-`Program`, vmap closure, front-end op, plain callable) first, then
  `is_builtin_op(fn)` + exhaustive `match` + `assert_never` (`exact_fold._sym_apply_builtin`
  and `sparsity._apply_builtin_op` are the models). An op you deliberately do not handle
  gets its own arm stating why — "opaque by omission" is what cost a day three times
  (`KronOp`/`KronFreeOp` → FEEC Λ² at 0%; `SwitchOp` → every `select_x` frozen).
- `ir.Ref` is polyarray's other closed sum type (6 members) — same treatment applies.
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
