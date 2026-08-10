# mypy after the docs/typing sweep

Replacing `Any` with real types removed the silencer mypy had been running
under, so the error count moved from 52 to 1422. The two populations are very
different and should be read separately.

**1282 errors — union fan-out over 28 source lines.** Every one is
`union-attr`: a site that dispatches on `type(fn).__name__` and then reads an
op-specific attribute (`fn.spec`, `fn._vmap_body`, `fn._nested_n_vars`). `StmtOp`
is a 56-member union, and mypy reports one error per member per access, so 28
lines produce 1282 messages. Nothing here is a new defect; the annotation is
accurate and the code is unchanged.

These sites are also the string-typed dispatch the workspace rules forbid
(`fem/CLAUDE.md` rule 3). Converting them to `isinstance` would both satisfy
mypy and satisfy the rule — but `batch.py`'s dispatch table deliberately covers
some *front-end* op names that this layer must not import, so the conversion is
not mechanical and is left for the maintainer to direct. The sites:

    python -m mypy 2>&1 | grep union-attr | cut -d: -f1,2 | sort -u

**140 errors — genuine type findings.** Real looseness the `Any` annotations
were hiding: `Program | None` dereferenced without a guard, `int | DimAtom`
passed to `int()`, `Mapping[str, float]` fed a value that may be an ndarray.
None is known to be a live bug — each is a place where the code relies on an
invariant the signature does not state — but each is worth reading.

    python -m mypy 2>&1 | grep error | grep -v union-attr
