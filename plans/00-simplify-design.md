# polyarray `simplify` — partial evaluation, substitution & sparsity propagation

Branch `simplify` (worktree `polyarray-simplify`, off `feat/numpy-source-codegen`).

## Goal (from the request)

Given a `Program` / `SymArray` IR, be able to:

1. **Reduce** statements / reduce a value to a **pure numeric value**, *or* to a
   value that **exposes more symbols**.
2. **Replace an argument** (a program input) with a new value that is either
   **numeric** or **depends on other arguments**.
3. Do this **numeric-first**: propagate/evaluate the purely-numeric parts, *then*
   make further reduction judgements under a **symbolic budget**.
4. **Propagate sparsity** — thread a structural-zero mask through statements so a
   Vandermonde (and its inverse) can be reduced (oracle M3).

This is the long-specified, never-built **partial-evaluation / build-then-simplify**
pass. chartLib named all of it; nothing here is a new research direction, only the
implementation.

## What already exists (reuse, do not rebuild)

- **Three-lane dispatch + per-op numeric short-circuit** — numeric/rational/imperative.
  det/inv/matmul/solve already fold when operands are float (`ir.py:819,841,873`);
  einsum/tensordot/matvec already skip structural-zero cells (`ir.py:2128,2150`).
  This is *local* folding inside one op; we add *cross-statement* propagation.
- **`RationalFunction.eval(bindings)`** (`rational.py:452`) already does **partial
  numeric substitution**: bound gens fold into coeffs, residue returned over a fresh
  ring of leftover gens (float if all bound). → cell-level numeric folding is solved.
- **`_partial_substitute`** (`rational.py:812`) — the numeric term-fold kernel; our
  symbolic `compose` generalizes it (float fold → RF arithmetic).
- **`cells_sparsity(arr)`** (`ir.py:218`) — per-array structural-zero mask via
  `simple_zero`. We thread this *across* statements (currently nobody does).
- **`forward.partial_eval(program, *, max_cell_size)`** (`forward.py:189`) — one-way
  *collapse* of over-budget cells to atoms via `IdentityOp`, exactness-preserving,
  top-program only. Our pass is the **superset**: it also folds numerics, drops dead
  stmts, substitutes inputs, and runs the **inverse** (re-expand) direction.
- **`forward.iter_programs` / `_body_of`** (`forward.py:35,47`) — descent into nested
  sub-Programs + vmap `CallOp` bodies. pointwise imports `_body_of` directly
  (`grassmann_backend.py:448`) — **keep it stable.**
- **`IdentityOp`** — the capture/freeze primitive; **`RationalRef`** — a *reserved,
  currently-unconstructed* Ref for "a spliced RationalFunction over program atoms"
  (`ir.py:413`): the natural hook for argument substitution.

## Hard constraints (carried from chartLib)

- **No `sympy.simplify` / `sympy.subs` at *runtime*.** Our pass is *build-time*
  (compilation), so a one-time symbolic substitution is fine — but prefer ring
  arithmetic over `sympy.subs`, matching `_partial_substitute`.
- **Exactness.** For every entry point E and inputs x:
  `E(p).run(x) == p.run({**substituted, **x})`. Form changes; values never do.
- **Sparsity is structural, rational-lane only, and never crosses a Stmt boundary.**
  A mask entry is set only when the *algebra* guarantees zero (`simple_zero` /
  literal 0.0) — never because a value evaluates small (numeric-cancellation hazard).
  The imperative `Stmt.fn` is opaque: its output zero-pattern is unknown, so masks
  reset to "unknown" across an opaque op (`symbolic_interpreter.md:921-981`).
- **Additive only.** pointwise has external consumers. Do not change `run`/`evaluate`
  semantics, the `inputs/statements/outputs/_cells` shape, `CallOp`, `_body_of`, or
  the `partial_eval(program, max_cell_size=)` signature. Preserve value-space /
  valence metadata so pointwise's "metric rides free" stays correct.

## Core abstraction — an abstract interpreter over the Program

The pass is a **partial evaluator**: re-interpret the program against a *partial*
input environment, emitting a new program. Per Stmt-output / per output-cell we carry
an **abstract value** in a small lattice:

```
AVal = Numeric(ndarray)          # concrete floats — fully known
     | Symbolic(cells, mask)     # object-array of RFs over residual gens + zero-mask
     | Opaque(symarray)          # Stmt-output atoms we cannot fold (opaque op, sym in)
```

Lattice order: `Numeric ⊏ Symbolic ⊏ Opaque` ("more known" → "less known"). Join at
control-flow merges (`SwitchOp`) takes the upper bound.

Walk `statements` in order (the same order `run` uses), maintaining an env keyed like
`_resolve_ref`. For each Stmt, resolve `in_` refs to `AVal`s:

- **all `Numeric`** → execute `stmt.fn` at build time on the concrete arrays (reuse
  `_run_stmt`'s call path with a frozen binding) → outputs become `Numeric` →
  **drop the Stmt** (dead-stmt elimination) and record numeric outputs as `Const`.
- **mixed / `Symbolic`** → keep the Stmt, but rewrite refs to folded constants where a
  sub-input is numeric; fold each output cell whose gens are all bound (`RF.eval`);
  carry the sparsity mask through ops we model (below).
- **`Opaque` in / opaque op** → keep Stmt; outputs are `Opaque`; mask resets to unknown.

This unifies requirements (1)–(3): numeric propagation + DCE = the `Numeric` path;
"expose more symbols" = choosing *not* to collapse a `Symbolic`, or re-expanding an
`Opaque`/deferred numeric Stmt; the budget gate decides which.

### `fold_numeric`, precisely (the load-bearing floor)

"Constant-fold + DCE" is too thin a description — the mechanics matter and determine
whether the pass is correct. fold_numeric is **build-time partial execution against a
growing `known: dict[atom_name → float]` map**, seeded by `bind` (empty for a bare
`fold_numeric`, so a fully-symbolic program folds to a no-op copy — the interesting power
comes *paired with* `bind`):

1. **Resolve.** For each Stmt in `run` order, try to resolve every `in_` ref to a
   concrete float array using `known` + bound inputs + already-constant cells
   (`RF.is_constant` / float dtype / `Const`). A `SymArrayRef`'s cells resolve via
   `RF.eval(known)`; missing atoms ⇒ "not numeric".
2. **Fold + cascade.** If *all* inputs resolve numeric, call `stmt.fn` at build time
   (reusing `_run_stmt`'s call path), then **overwrite the Stmt's output SymArray cells
   in place with the resulting floats** and record each output `atom_name → value` in
   `known`. The in-place bake is what makes the cascade work: every downstream
   `SymArrayRef` to those cells now resolves numerically *with no run-time binding*, and
   the new `known` entries unlock the next Stmt. **This cascade — fold ⇒ output atoms
   become known ⇒ downstream folds — is the actual content of "numeric propagation",**
   not the per-cell `RF.eval` alone.
3. **Partial fold.** If inputs are mixed, keep the Stmt but rewrite its input refs by
   folding `known` into them (`RF.eval` partial binding) so it sees simplified inputs;
   its still-symbolic outputs stay `Symbolic`/`Opaque`.
4. **Rewrite outputs.** Fold `known` into every program-output cell (a fully-bound cell
   ⇒ float; a partially-bound cell ⇒ a smaller RF — the "leave residual symbols" case).
5. **Drop dead Stmts.** A fully-folded Stmt is now dead (its outputs are baked floats,
   referenced by nobody as a binding). Remove it. The *only* run-relevant fixup is
   reindexing any surviving `OutputRef` (rare — `SymArrayRef` is the workhorse and is
   index-free); `Provenance.origin` is descriptive and `run` never reads it, so it needs
   no fixup for correctness.

**Conservatism = safety.** Anything not confidently foldable (bulk/dynamic-shape
outputs, sub-`Program` fns, control-flow ops, an op we don't model) is *kept symbolic* —
worst case fold_numeric degrades to `copy()`. Exactness holds because every baked value
is exactly what `run` would have computed for those atoms:
`fold_numeric(p, bind=b).run(rest) == p.run({**b, **rest})`.

## Public API (additive, new module `polyarray/simplify.py`)

```python
def specialize(program, *, bind=None, subs=None,
               sparsity=False, budget=None) -> Program
```
The unified partial evaluator.
- `bind: Mapping[str, ndarray]` — replace an input with a **concrete numeric** array
  (requirement 2, numeric). Seeds the env as `Numeric`, then folds.
- `subs: Mapping[str, SymArray|RationalFunction]` — replace an input with an
  **expression over other inputs** (requirement 2, symbolic). Uses the new
  `compose` primitive.
- `sparsity: bool` — also compute & attach structural-zero masks (requirement 4).
- `budget: SimplifyBudget|None` — the **control surface** that moderates collapse↔expose
  (requirement 3). **`SimplifyBudget` is specified authoritatively in
  `01-budget-moderated-simplification.md`** (knobs, presets, moderation procedure,
  properties) — this doc does not re-spec it. `budget=None` defaults to
  `SimplifyBudget.none()`: numeric folding + dead-stmt elimination only (the
  unconditional floor), leaving all residual symbolic structure untouched. Any non-floor
  collapse / re-expansion happens **only** when a budget asks for it.

Composable sub-passes — each is a thin wrapper over `specialize` with a fixed `01`
preset (one vocabulary, not two), all exactness-preserving:
- `fold_numeric(program)` ≡ `specialize(program, budget=none())` — constant-fold +
  dead-stmt elimination, no subs. The floor every other entry point runs first.
- `bind_inputs(program, bind)` ≡ `specialize(program, bind=…)` — numeric arg replacement.
- `substitute(program, subs)` ≡ `specialize(program, subs=…)` — symbolic arg replacement.
- `expose(program, keep=…)` ≡ `specialize(program, budget=expose_symbols(keep=…))` — the
  inverse of `partial_eval`: re-expand deferred Stmts / extract intermediates so more
  symbols surface (requirement 1, "expose more symbols").
- `collapse(program, max_cell_mass=m)` ≡ `specialize(program, budget=balanced(m))` — the
  cost-bounded collapse that **subsumes** `partial_eval(program, max_cell_size=m)`.
- `propagate_sparsity(program) -> SparsityReport` — masks per SymArray, no rewrite (the
  one sub-pass that is not a `specialize` wrapper; see §P4 and `01` consumer table).

New rational primitive (the only `rational.py` addition):
```python
RationalFunction.compose(self, name: str, repl: RationalFunction) -> RationalFunction
```
Substitute generator `name` with the RF `repl` (over other inputs), via ring
arithmetic — a generalization of `_partial_substitute` (float fold → RF mul/add into a
target ring spanning `leftover ∪ repl.gens`). No `sympy.subs`.

## Sparsity rules (rational lane, structural)

Attach `mask: NDArray[bool]` (True = structurally zero) to each `Symbolic` AVal.
Propagation (from `symbolic_interpreter.md:921-981`):
- `A + B`: zero where **both** zero.
- `A * B` / Hadamard: zero where **either** zero.
- `A @ B` (matmul / `EinsumOp` contraction): `out[i,j]` zero iff **every** contributing
  pair is zero → a boolean matmul on the masks.
- `MoveaxisOp` / `IdentityOp` / reshape: permute/copy the mask.
- **Opaque Stmt** (Det/Inv/Pinv/Solve/Svd/Qr/Sqrt/...): mask resets to all-unknown on
  outputs — we do **not** claim the zero pattern of an opaque numeric op.

This is exactly what oracle's M3 Schur inverse needs as *input*: it reads the mask on
the symbolic Vandermonde `C` to "choose the `(p,q)` split that maximizes the zero
block" (`oracle/research/04:101-110`) and to skip recomputing zero blocks. The inverse
itself is opaque to *our* propagation (dense in general); the block structure is
oracle's own recursion, fed by our mask.

## Phasing (each phase ships with exactness tests vs `Program.run`)

- **P0 — skeleton.** Abstract-interpreter walk + AVal lattice; `fold_numeric` as the
  driver. Identity test: `fold_numeric(p)` with no numeric inputs ≡ `p.copy()`.
- **P1 — numeric propagation + DCE.** Fold fully-bound cells (`RF.eval`); execute
  fully-numeric Stmts at build time; drop them; rewrite downstream refs.
  Exactness: random inputs, `fold_numeric(p).run(x) == p.run(x)`.
- **P2 — numeric argument substitution (`bind`).** Seed env from `bind`, then P1.
  Serves pointwise plan-36 "collapse to numeric when all concrete, leave residual
  symbols otherwise" and "fix the point / a vertex" (`sampled.py:77-85`).
- **P3 — symbolic substitution (`subs`).** Add `RationalFunction.compose`; substitute
  an input's atoms with expressions over other inputs; re-fold. Serves default-args
  that depend on other args (`pointwise/plans/44:53-65`) and "expose more symbols."
- **P4 — sparsity propagation.** `SymArray.sparsity` masks + op rules + opaque reset;
  `propagate_sparsity` report; `block_zero_mask(symarray)` helper for oracle M3.
  Add an oracle test asserting the symbolic Vandermonde's block-zero pattern.
- **P5 — `SimplifyBudget` + moderation procedure** (full spec in `01`). Implement the
  collapse ceilings (`max_cell_mass`/`total_mass`, subsuming `partial_eval`'s
  `max_cell_size`), `expose` re-expansion (`chartLib todo.md:87`), `den_degree_max`
  intermediate-symbol extraction (`try_cancel`), `keep_provenance`, and the
  `analyze → budget → specialize` seam. The presets land here.
- **P6 — nested-body descent + pointwise integration.** Descend `CallOp`/vmap bodies
  (the slice `partial_eval` deferred, `forward.py:200`); offer to replace pointwise's
  hand-rolled degree/constant walker (`grassmann_backend.py:369-503`) and fold the
  quad Python loop (`pipeline.py:372-385`) into one points-axis Program (plan-43 A1).

## First-consumer mapping (so we build to real demand)

| Capability | pointwise need | oracle need |
|---|---|---|
| P1/P2 numeric fold + bind | plan-36 substitute-then-reduce per quad point | numeric Vandermonde fallback / `P(T)=I` affine case |
| P3 symbolic subs | default-arg = closed IPTEM feeding another arg | — |
| P4 sparsity | replace hand-rolled zero short-circuit (`grassmann_backend.py:433`) | M3 Schur: choose-pq on the zero block |
| P5 budget/expose | "leave residual symbols if input symbolic" | intermediate-var extraction when den-deg>1 |

## Open questions / risks

- **Folding structured Ops** (Det/Inv/Svd…) at build time = literally calling the numpy
  fn on concrete inputs. Fine and desirable; cost is one build-time numpy call.
- **Symbolic subs into an opaque Stmt's inputs**: we can rewrite the *ref*, but cannot
  see through the op — outputs stay `Opaque`. Acceptable.
- **`compose` ring growth**: substituting an arg-dependent expression enlarges each
  cell's gen set; guard via `SimplifyBudget.inherit_freeze` (`01`), which re-applies the
  source `Program.budget` freeze threshold so post-subst growth is capped consistently.
- **Sparsity false-negatives are safe; false-positives are not.** Default to "unknown"
  (not zero) whenever an op isn't in the modeled set.
