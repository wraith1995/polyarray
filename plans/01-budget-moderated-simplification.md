# Budget-moderated simplification

Companion to `00-simplify-design.md`. Where 00 describes the *mechanism* (an
abstract-interpreter partial evaluator over a `Numeric ⊏ Symbolic ⊏ Opaque` lattice),
this doc describes the *control surface*: a **post-build `SimplifyBudget`** that
moderates how far the pass simplifies — exactly mirroring how the build-time
`SymbolicBudget` moderates how symbolic a Program is built.

## Thesis

`SymbolicBudget` and `SimplifyBudget` are the two ends of one pipeline, bracketing
`analyze`:

```
  build_sampler(budget=SymbolicBudget)   →   analyze(p)   →   specialize(p, budget=SimplifyBudget)
  ── how symbolic to BUILD ──                ── measure ──     ── how far to SIMPLIFY ──
```

"build big, then decide what matters" (forward.py:12). `SymbolicBudget` decides the
first clause; `analyze` measures the result; `SimplifyBudget` decides the second. The
two budgets share a **currency** (symbolic mass = monomial count, `_cell_size`,
ir.py:1561) and a **vocabulary** (provenance kinds, the `cells_use_only_stmt_atoms`
gate), but act at different times: build-time, *structural/schema-driven*; post-build,
*data-driven* (it sees the actual numeric `bind` values).

## Two simplification directions the budget arbitrates

Simplification is not one-dimensional. There are two opposing moves, and the budget
slides between them:

- **Collapse** (↓ mass): fold to numeric, capture a heavy cell as a fresh atom
  (`IdentityOp` / `partial_eval`). Smaller, faster, **more opaque**.
- **Expose** (↑ legibility): keep a cell symbolic, re-expand a deferred Stmt, or extract
  a denominator as a *named intermediate symbol* (`try_cancel`, rational.py:626). Larger,
  **more legible** — surfaces structure (the parameterization, the Vandermonde blocks).

These trade off along a Pareto frontier (residual mass vs exposed-symbol count). A low
budget sits bottom-left (numeric, opaque); a high budget sits top-right (symbolic,
legible). `SimplifyBudget` is **the point on that frontier**.

## The one thing the budget does NOT moderate

**Numeric folding + dead-stmt elimination is unconditional** — done first, regardless of
budget. It is monotonically good: it never increases mass and never erases a symbol that
wasn't already numeric. The budget only governs the *discretionary* band — the
genuinely symbolic↔numeric choices where a real tradeoff exists. This is the "moderate
the simplification **where possible**" of the request: fold the free wins always; spend
budget only where a choice exists.

## `SimplifyBudget` — the post-build policy

Distinct type from `SymbolicBudget` (whose knobs — `defer_phi_jet`, `defer_covariant`,
`surface_frame` — are build-time op-gates meaningless to a finished Program). Precedent:
`partial_eval` already takes a bare `max_cell_size: int`, not a `SymbolicBudget` —
post-build cost knobs already live outside the build-time budget. `SimplifyBudget`
generalizes that single knob into a policy.

```python
@dataclass(frozen=True)
class SimplifyBudget:
    # --- collapse side (mass ceilings) ---
    max_cell_mass: int | None = None      # per-cell monomial ceiling; over ⇒ collapse to
                                           # atom. == partial_eval's max_cell_size (subsumed).
    total_mass: int | None = None         # whole-program ceiling; greedily collapse the
                                           # heaviest cells until under (global, vs partial_eval
                                           # per-cell). Fed by analyze's total_mass.
    # --- expose side (legibility) ---
    expose: Literal["never","if_under_budget","always"] = "never"
                                           # re-expand a deferred numeric Stmt back to symbolic.
                                           # "never" == today. bounded by the mass ceilings.
    den_degree_max: int = 1               # cell whose denominator degree exceeds this ⇒ extract
                                           # it as a fresh NAMED intermediate symbol (try_cancel),
                                           # NOT a numeric atom. The key "expose more symbols" lever.
    keep_provenance: frozenset[str] = frozenset()   # never collapse a cell while it still carries
                                           # a generator of these kinds (e.g. {"vertex","coeff"}).
                                           # Post-build analogue of defer_phi_jet's "keep the
                                           # parameterization". Enforced via cells_use_only_stmt_atoms.
    # --- consistency with how it was built ---
    inherit_freeze: bool = True           # after a symbolic subst grows a cell, re-apply the source
                                           # Program.budget freeze threshold rather than inventing one.
```

### Presets — end-state correspondence to `SymbolicBudget`

Mapped by *resulting form*, not mechanism (see 00 §"build_big_symbols vs expose"):

| `SimplifyBudget` preset | knobs | does | `SymbolicBudget` counterpart (end-state) |
|---|---|---|---|
| `none()` / `legacy()` | all off, `expose="never"` | numeric fold + DCE only; symbolic untouched | — (the always-safe core) |
| `numeric_only()` | `max_cell_mass=0` | collapse every symbolic cell to numeric/atom | `force_stmts` *result* / fully-collapsed legacy |
| `balanced(m)` | `max_cell_mass=m`, `expose="if_under_budget"` | keep cheap symbolic structure, collapse the heavy tail | `SymbolicBudget()` default middle |
| `expose_symbols(keep=…)` | `max_cell_mass=None`, `expose="always"`, `keep_provenance=keep` | maximize visible symbols, bounded only by `total_mass` | `build_big_symbols` |

`numeric_only()` is the "reduce to a pure numerical value" extreme of the request;
`expose_symbols()` is the "value that exposes more symbols" extreme; `balanced()` is the
moderated middle the framing is about.

## The moderation procedure

Per output cell / Stmt-output, after the unconditional numeric fold:

```
for each cell c in topological order:
    c = numeric_fold(c)                       # unconditional; uses RF.eval partial binding
    if c.is_numeric:            continue       # already a float — nothing to moderate

    if carries_kind(c, budget.keep_provenance):
        c = maybe_expose(c, budget); continue  # protected: never collapse; expose if asked

    if den_degree(c) > budget.den_degree_max:
        c = extract_intermediate(c)            # try_cancel ⇒ a NEW named symbol (expose)
        continue

    if budget.expose != "never" and c is a deferred-Stmt output and reexpandable(c):
        c2 = reexpand(c)                        # inverse of partial_eval
        if budget.expose=="always" or mass(c2) ≤ budget.max_cell_mass:
            c = c2

    if budget.max_cell_mass is not None and mass(c) > budget.max_cell_mass:
        c = collapse_to_atom(c)                 # IdentityOp — the partial_eval move

# global pass, if a whole-program ceiling is set:
while total_mass(prog) > budget.total_mass:
    collapse_to_atom(heaviest_remaining_cell())
```

Every branch is exactness-preserving (00 §"Hard constraints"): `collapse` captures the
cell for run-time eval; `extract_intermediate`/`reexpand` are algebraic identities;
`numeric_fold` is `RF.eval`. So for all budgets B and inputs x:

    specialize(p, budget=B).run(x) == p.run(x)            # form changes, value never does

### Properties to assert in tests

- **Numeric-fold floor**: `specialize(p, none())` ⊇ the numeric fold of every all-numeric
  subtree, independent of budget.
- **Monotone collapse**: tightening the budget (smaller `max_cell_mass`) only ever turns
  symbolic cells into atoms — never the reverse — so mass is monotone non-increasing in
  budget tightness.
- **Idempotence**: `specialize(specialize(p, B), B) == specialize(p, B)`.
- **`numeric_only()` ≡ `partial_eval(p, max_cell_size=0)`** on the symbolic-cell set
  (a compatibility anchor — we subsume the existing pass).
- **Frontier**: looser budget ⇒ exposed-symbol count non-decreasing, mass non-decreasing.

## Why the two budgets must stay separate types (but rhyme)

- **Timing/regime**: `SymbolicBudget` is structural (decides from input *schema* so the
  build/eval cache stays sound — chartLib `project_principles.md:220`). `SimplifyBudget`
  is data-driven (acts on concrete `bind` values), which is precisely why it can only run
  post-build. Same knob-name on both types would conflate two regimes.
- **Shared currency, not shared object**: both meter on `_cell_size` mass and respect the
  same provenance gate, so a report from `analyze` (built to feed the budget machinery,
  forward.py:11) parameterizes *either*. The clean seam is: `analyze` returns mass +
  provenance histogram → caller picks a `SimplifyBudget` → `specialize` applies it.
- **One soft coupling**: `inherit_freeze` lets `SimplifyBudget` read the source
  `Program.budget` freeze threshold so post-subst cell growth is capped consistently with
  how the Program was originally built. The only place the two budgets touch.

## Consumer mapping (which preset each need wants)

| Need | preset |
|---|---|
| pointwise plan-36: per quad point, collapse to numeric when all concrete | `numeric_only()` when `bind` is total; `balanced` when partial |
| pointwise plan-36: "leave residual symbols if any input symbolic" | `balanced(m)` — folds the numeric VA part, keeps the symbolic residue |
| pointwise: keep the parameterization over vertices visible | `expose_symbols(keep={"vertex","coeff"})` |
| oracle M3: intermediate variables when denominator degree > 1 | `den_degree_max=1` (default) — extracts the named symbol |
| oracle M3: numeric Vandermonde fallback / affine `P(T)=I` | `numeric_only()` |
| oracle M3: keep `C` near-block-triangular & legible for choose-pq | `balanced` + sparsity (00 §P4) |

## Where this lands in the 00 phasing

The budget is not a late add-on — its *hooks* thread through the whole pass:

- P1 numeric fold: the unconditional floor (no budget needed).
- P2 `bind`: seeds the data the budget reacts to.
- P4 sparsity: `balanced`/`expose_symbols` keep the structure sparsity needs to read.
- **P5 becomes "implement `SimplifyBudget` + the moderation procedure"** — the collapse
  ceilings, `expose`, `den_degree_max`, `keep_provenance`, and the `analyze→budget→specialize`
  seam. The presets land here.

So 00's `specialize(..., budget=…)` parameter is the entry point; this doc specifies what
that `budget` is and how it moderates each move.
