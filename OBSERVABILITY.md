# Compile observability — `polyarray.observe`

Watch a compile happen: per-stage IR size, polynomial degree, provenance, the delta from the
previous stage, proactive warnings, and — at the highest level — a directory of dumps running
from the symbolic IR through to the generated torch.

The problem it solves: every hard bug in this stack has been *"some stage blew the IR up and we
could not see which."* Localising the argyris degree-5 mass blow-up took hours of hand probes and
three wrong guesses. This turns that into reading a table.

---

## Quick start

```bash
FEM_OBSERVE=info  python your_assembly.py        # a staged table on stderr
FEM_OBSERVE=dump FEM_OBSERVE_DIR=/tmp/obs python your_assembly.py   # + a dump per stage
```

Nothing to change in your code — savo's `generic_assembly` and pointwise's `single_compile`
are already instrumented.

## Levels — `FEM_OBSERVE`

| level | what you get | cost |
|---|---|---|
| `off` | nothing | ~1 µs per instrumented boundary (0.01% of an argyris compile) |
| `warn` | **default** — proactive warnings only | +9% argyris, noise on smaller elements |
| `info` | + one line per stage, + the closing report | ≈ `warn` |
| `debug` | + per-sub-program `IRReport` breakdown; **measures every occurrence** | +230% argyris |
| `dump` | + `<FEM_OBSERVE_DIR>/<trace>/NN-<stage>/` per stage | +230%, and tens of MB |

`debug` and `dump` are expensive **by design**: below them, repeated occurrences of a stage are
sampled (see *Sampling* below), and there they are not.

Other env vars — warning thresholds, all optional:
`FEM_OBSERVE_MASS_CEILING` (250k), `FEM_OBSERVE_OPERAND_CEILING` (1M),
`FEM_OBSERVE_CELLS_CEILING` (16384), `FEM_OBSERVE_DEGREE_CEILING` (24).

---

## Reading the report

```
   stage                          n  meas       mass            Δ        cells   deg    time
--------------------------------------------------------------------------------------------
07 represent-matches              1     1          4                       0/0     2   0.44s
11     represent                192   192         12        (+12)          0/0     2   1.98s
12     chartlib-field           270   270        631   (+619, ×52.6)      0/12     —   0.31s
14   value-block                  1     1       1842   (+1838, ×460.5)    0/36     —   3.41s
16 value-kernel                   1     1       1842                   114/150   inf   1.23s

peak: stage 14 'value-block' at mass 1842
went rational: stage 16 'value-kernel' (degree inf — no longer polynomial in its inputs)
```

- **`mass`** — the headline cost: *operand* mass when there is any, else output mass. Operand mass
  is what predicts lowering cost (einsum/linalg statements defer their outputs to atoms, so output
  mass reads tiny while the operands expand into the codegen AST).
- **`Δ`** — change vs the previous stage at the same depth. **This is the column that names a
  culprit.** A ×10 jump in one stage is the blow-up fingerprint.
- **`n` / `meas`** — how many times the stage ran, and how many of those were measured.
- **`deg`** — polynomial degree in the program's own atoms. `inf` means rational/algebraic, i.e.
  no longer polynomial.
- **`cells`** — symbolic / total output cells. A large symbolic count is the eager per-cell
  scatter fingerprint.
- **`time`** — cumulative over all `n` occurrences.

Three summary lines: `peak` (where the IR is *biggest*), `largest jump` (where it *got* big —
usually the actual culprit, since the peak often just carries forward what an earlier stage
inflated), and `went rational` (where it stopped being polynomial).

> **The peak stage and the slow stage are often different.** On argyris the symbolic peak is in the
> element sampling entry, but the wall time is nearly all in `value-block`.

---

## Sampling — and when it will mislead you

Measuring a stage is O(program size). The instrumented boundaries sit in hot loops (argyris calls
`chartlib-field` 1170 times), so measuring every occurrence made the default level **3.3× slower
than no instrumentation at all**. Below `debug`, occurrences are therefore sampled geometrically —
1, 2, 4, 8, … — giving O(log n) measurements and full fidelity for small loops.

**The cost is real.** On argyris at `warn`/`info` the `chartlib-field` row reads mass 0, because
none of occurrences 1, 2, 4, …, 1024 happened to be the big one; at `debug` it reads 2,755,915.

> **Default is a smoke alarm; `debug` is the investigation.** When attributing a blow-up to a
> precise stage, re-run at `FEM_OBSERVE=debug`.

Repeated stages are also **rolled up** into one row keyed by `(depth, stage name)` — without that,
one Morley assembly produced 827 rows and would have written 827 dump directories.

---

## The dump

```
/tmp/obs/assembly-1iptem/
  report.txt                    the table above
  01-sample/  02-convert/  …    one directory per stage
  16-value-kernel/
    stage.txt                   mass / degree / provenance / the full IRReport
    detail.txt                  the caller's own description (see below)
    program.py                  THE POLYARRAY IR, rendered as source
    f/torch.py                  pyab's generated torch — same directory
    f/ir.txt                    pyab's lowered IR
    f/inductor/                 Inductor's debug trace
    f/dynamo_explain.txt
    cxx/**/*.cpp                the GENERATED C++ (with SAVO_TORCH_COMPILE=1)
```

The lowering stages hand their own directory to pyab (`CompileTrace.dump_dir_for` →
`pyab.compile_torch(dump_dir=)`), so the symbolic snapshot and the code generated from it sit
side by side. **One directory listing is the whole compile.**

`program.py` is rendered with no `op_renderers` threading, so a front-end op appears there iff it
carries a `__numpy_source__` hook (grassmann `_QrSignConventionOp`, chartlib `QrSignFixOp`, …); one
that does not is written as a `# to_numpy_source failed:` note rather than breaking the compile.
A mid-pipeline stage is usually *open* over savo's per-cell vertex atoms `V_j_k` — no input declares
them yet — so they render as the trailing parameters of the emitted function.

`detail.txt` is whatever the instrumented call site chose to describe. savo's `represent-matches`
writes, per enumerated match: the IPTEM and host entity, each argument's binding and the grassmann
**value basis** the body was represented consuming (`FieldInput.basis`), every geometry quantity
the kernel will demand, the value space and exterior degree, and the polyarray program's shape and
input-name map.

---

## Where is the C++? — `SAVO_TORCH_COMPILE=1`

pyab emits **eager** torch: it decorates a generated function with `@torch.compile` only when the
body contains a `FusionRegionExpr`, produced solely by `builder.region()` /
`polyarray.pyab.call_lowered(place="fuse")`. savo lowers with a plain `LowerOpts()` and wraps the
result in `torch.vmap`, so **by default Dynamo and Inductor never run and no C++ exists.**

`SAVO_TORCH_COMPILE=1` closes that: savo wraps the vmapped value kernel in `torch.compile`, so
Inductor compiles it on first call and (on CPU) emits C++.

```bash
SAVO_TORCH_COMPILE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/cxx python your_assembly.py
find /tmp/cxx -name '*.cpp'      # 26 files for a Morley mass assembly
```

Three things to know, all measured:

* **Default OFF, deliberately.** It changes how the kernel is built and run: first-call latency
  rises substantially, and a graph break or unsupported op can degrade or fail where eager worked.
* **It is NOT bit-identical to eager.** Fusion reassociates floating-point reductions, so results
  differ in the last bits — 6.9e-18 (P2), 1.4e-17 (Morley), i.e. in the structurally-zero entries.
  Symmetry, spectrum and the partition-of-unity total are unchanged. **A future eager-vs-compiled
  A/B gate must not demand bit-equality.** Covered by `savo/tests/test_torch_compile_kernel.py`.
* **Under `FEM_OBSERVE=dump` the C++ lands in the stage directory automatically** —
  `<dump>/NN-value-kernel/cxx/**/*.cpp`, beside the symbolic snapshot and the generated torch. No
  env var needed. Set `TORCHINDUCTOR_CACHE_DIR` yourself to override; the default (no dump, no
  override) is under `$TMPDIR` on macOS, **not** `/tmp`, and the trace records the resolved path.

That last point took two fixes, both worth knowing if it ever regresses. pyab's dump capture patches
Inductor with `force_disable_caches=True` so a warm cache cannot short-circuit its own debug trace —
which also stops Inductor writing any C++ at all (26 files → 0, with a perfectly correct assembly
and no error). savo now strips that capture before compiling. Stripping it from the entry point `f`
alone was **not** enough: the generated code calls its `_sub_*` helpers as module globals, and those
were wrapped too, so one of them still fired the patch. Guarded by
`test_cpp_lands_in_the_dump_stage_directory`.

---

## What is instrumented

**savo** — `generic_assembly` (opens the trace) · `_compile` · `represent-matches` · `entries` ·
`value-block` · `value-kernel` (+ torch dump hand-off) · `world-kernel` · `run`.

**pointwise** — `single_compile`'s spine (`sample` → `convert` → `combine` → `represent`), and the
two entries where pointwise calls INTO chartlib's samplers:

| stage | covers |
|---|---|
| `chartlib-field` | `bind_field` — the field/element sampler (world, reference and symbolic modes) |
| `chartlib-geom/<quantity>` | `eval_geometry`, per `GeomQuantity` — and everything reached from it: Tangents/Cotangents jets, the QR orthonormal frame, `restricted_param_jet`, `sample_normal`, `param_jet().measure()` |
| `chartlib-geom-sym/<quantity>` | `_symbolic_geometry` — the symbolic lane, where a quantity comes back as a `SymArray` in the point |
| `dof-table` | `vandermonde` — the DOF-table build (`evaluate_integral` is on the stage: it is the knob that separates the cheap build from the catastrophic one) |
| `dof-nodal-matrix` | `_world_nodal_matrix` on the affine-invariant fast path (`P(T)=I`, no `P(T)` built) |
| `dof-nodal-matrix-pt` | the same, on the NON-affine path where a vertex-rational `P(T)` IS built — the argyris/bell blow-up site |

**savo — the seam stages.** These carry a `detail.txt` rather than a big IR, because what matters
about them is *structure*, not size:

| stage | what its dump shows | the failure it catches |
|---|---|---|
| `enumerate` | the complex's entities, the candidate reps per argument, and every surviving `GMatch` (out entity, meet, storage, per-arg binding) | a missing or spurious match — silently too few/too many contributions, or contributions on the wrong entity. A `⚠ NO MATCHES` line is printed when an IPTEM contributes nothing |
| `sections` | the OUTPUT section schema and every INPUT section schema, in the same `SectionRow(label, T1=support, T2=storage)` vocabulary | output/input schema disagreement — an operator of plausible shape with its entries in the wrong places |
| `quadrature` | folded degree, point count, per-input degrees, geometry depth | an under-degree rule: no error, just silent inexact integration |
| `orientation-sigma` | how many σ atoms the compiled kernel carries, per iteration type | the `NOTE-orientation-sigma.md` failure — a direction-valued DOF whose `entity_orientation_matrices` came back EMPTY, so savo skips reconciliation with no error and assembles a wrong orientation-dependent operator. `reconciled: NO` is the tell |
| `run-binds` | the compile-DECLARED input names vs the run-BOUND ones, with shapes; **warns** on either-way mismatch | the compile/run seam — a channel declared but never bound (fails late and obscurely, or takes a stale value) or bound but never declared. A real collision bug already lived here |
| `index-block` | the emitted scatter coordinates next to the value blocks: block shape, and each binding arg's global-DOF index array, with a `⚠ MISMATCH` when the DOF-axis extents disagree | a transposed / mis-broadcast / mis-ordered axis — an operator of the right shape with its entries in the wrong places |
| `element-definition` | the Ciarlet triple `(V, P, Σ)` actually in play: cell type, space + degree, `dim P`, and every DOF functional with its support entity type and evaluation kind | a wrong or mis-ordered Σ — a perfectly well-formed but WRONG element, which the Vandermonde will happily invert |

**oracle.** oracle does no IR work of its own — it *drives* pointwise — so it gets attribution
rather than new measurement. `feec-element` groups a FEEC family/`r`/`k` build (measured: one
`pminus_element(TRI,1,1)` triggers 3 represents, 6 field samples and a `dof-table`, because
`p_minus` builds its Koszul kernel through `numeric_vandermonde` → `iptem_compiler` → represent),
which matters because the two FEEC families are mutually recursive and memoized — "how many element
builds actually happened" is the question a FEEC regression asks. `element-definition` is recorded
by savo for every element it is HANDED, since an element is normally constructed *before* the
traced region begins and the construction-time hook would never fire.

**Not instrumented:** oracle; and the Theme-A `integrate` / `build_assembly_program` path in
pointwise (`_symbolic_affine_rule`, `PointRule.rule`). These announce themselves — see *Off-path
detection* below.

> A savo assembly **does** reach `pointwise.integrate`, via the element Vandermonde
> (`_vandermonde_ref` → `numeric_vandermonde` → `iptem_compiler` → `integrate`) — the DOF-table
> build. This was found by the off-path marker after the code had asserted the opposite. It matters
> because the DOF-table build is where the value-kernel blow-up B lived, so that cost is currently
> attributed nowhere in the trace.

## Off-path detection

The instrumented boundaries describe *one* route through the stack. This stack's defining bug class
is a quantity taking a **different route in a different consumer**. A trace that silently omits the
route actually taken is worse than no trace — it reads like a complete account of the compile.

So an uninstrumented or non-canonical path calls `observe.off_path(name, why)`. When a trace is
active it records an `off-path/<name>` stage, warns once per route, and lists the holes at the
bottom of the report:

```
⚠ OFF-PATH — this trace does NOT account for:
    pointwise.integrate — the Theme-A quadrature/integration path — NOT staged. Reached from
    savo through the element VANDERMONDE …
```

When nobody is observing it is silent, so ordinary consumers of these APIs see nothing. That
asymmetry is the point: the warning means *"your trace has a hole here"*, which is only meaningful
to someone holding a trace.

Currently marked: `pointwise.integrate`, `pointwise.build_assembly_program`,
`pointwise.dumb_backend` (the numpy cross-check oracle — reaching it means the compile is not on
the grassmann path), and `savo.torch_compile_under_dump` (see below).

**Deliberately never instrumented: grassmann and chartLib.** They are the layers *below* the
compile driver; observability belongs at the entry points into pointwise/savo/oracle, and neither
of those repos imports `observe`.

---

## Instrumenting new code

```python
from polyarray import observe

def my_stage(...):
    tr = observe.trace()                       # never None — a null trace when off
    with tr.phase("my-stage", some="context") as box:
        result = do_work()
        box.append(result)                     # measured on exit
        return result
```

- `observe.scope(name)` at an entry point: reuses an enclosing trace if there is one, else opens
  its own — so wrapping a larger region still yields ONE trace.
- `detail=lambda: describe(x)` on `stage`/`phase`: a thunk called only at `dump`. It may close over
  locals not yet bound when the phase is entered — it runs at exit.
- `observe.dump_dir_for(name)` before a lowering call, passed as `compile_torch(dump_dir=...)`.
- `observe.active()` only to guard work you would do *purely* to hand to the tracer; `stage()` is
  already free when off.

Two invariants the call sites rely on, and which any change must preserve:

1. **Measurement never breaks a compile.** Every probe is failure-tolerant; the worst case is a
   snapshot carrying an error string.
2. **Nothing is forced.** Bulk (deferred) nodes are counted as deferred and never materialised —
   a tracer that materialises what it measures causes the blow-up it exists to report.

API reference: `PUBLIC_API.md` § *Compile observability*. Tests: `tests/test_observe.py`.
