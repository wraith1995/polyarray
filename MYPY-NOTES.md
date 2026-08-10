# mypy after the docs/typing sweep

Replacing `Any` with real types removed the silencer mypy had been running
under. The count moved 52 → 1422, then back down as the causes were fixed:

| after | errors | what changed |
|---|---|---|
| the sweep | 1422 | `Any` no longer hides anything |
| `batch.py` dispatch by `isinstance` | 580 | twelve name-dispatch lines, each fanning out over a 56-member union |
| `VmapClosure` / `NestedVmapClosure` protocols | 168 | four `getattr`-probed closures, narrowed once instead |

The 168 that remain are genuine type findings, not fan-out: an `npt.ArrayLike`
passed where an `np.ndarray` is required, a `Program | None` dereferenced
without a guard, a `Callable` slot assigned something narrower. None is a known
live bug — each is a place where the code relies on an invariant its signature
does not state — but each is worth reading.

    python -m mypy 2>&1 | grep -oE '\[[a-z-]+\]$' | sort | uniq -c | sort -rn

## What the dispatch conversion was

`batch._apply` compared `type(fn).__name__` against strings — the string-typed
dispatch `fem/CLAUDE.md` rule 3 forbids, and the reason seven rules once sat
dead for a year while `batched_run` silently fell back to the per-element loop.

Converting to `isinstance` is behaviour-preserving here, verified rather than
assumed: every dispatched name is one of polyarray's own ops, and no class in
grassmann, chartlib, pointwise, savo or oracle shares one of those names. It is
also strictly safer — a front end that later defines its own `ScaleOp` is no
longer mistaken for the builtin, and a misspelled class is now an ImportError
rather than a rule that never fires.
