# Provenance of the polyarray extraction

`polyarray` is a faithful, one-time **extraction** of chartlib's symbolic
array IR core (plan: `grassman/plans/10-ir-extraction.md`, Stage 1).

## Source

- **Repository:** `~/projects/fem/chartLib`
- **Source path:** `src/chartlib/_symbolic/`
- **Git commit:** `11d426a9213a252f49c22eeb25ef9aa07162c070`
- **Commit date:** 2026-06-15 11:18:34 -0400
- **Branch:** `main`
- **Extraction date:** 2026-06-15

chartlib was **not modified** by this extraction (Stage 1 is copy-only;
the chartlib refactor to depend on `polyarray` is a separate later
change — Stage 2).

## Files copied (MOVE-set, plan §1)

Copied verbatim from `src/chartlib/_symbolic/` into `src/polyarray/`:

| file | role |
|---|---|
| `ir.py` | `Program`, `Stmt`, `SymArray`, refs, Ops, `SymbolicBudget`, `Provenance`, `SymbolEnv`, `SymInput`, `Program.run` |
| `rational.py` | `RationalFunction`, `bareiss_det`, `cofactor_inverse`, ring helpers |
| `poly_backend.py` | `Poly`/`Ring` protocol + backend selection (sympy / flint / native_py / native_value / cython cpp) |
| `poly_native_py.py` | pure-Python polynomial backend |
| `poly_native_value.py` | value-handle backend support |
| `int_atom.py` | `IntAtom` integer selectors |
| `forward.py` | IR passes: `analyze`, `partial_eval`, `iter_programs` |
| `_poly_native_cpp_double.pyx`, `_poly_native_cpp_mpf.pyx` | C++ polynomial backends |
| `setup_cython.py` (from chartlib repo root) | Cython build glue |

All intra-package imports in the copied modules are already relative
(`.ir`, `.rational`, `.int_atom`, `.poly_backend`, `.poly_native_value`)
and work unchanged once the files sit together in `polyarray/`.

## Edits made to copied files

Faithful extraction (plan §4): **only** import paths / build paths were
touched. No new Ops, no API changes, no behavior changes, no new options.
`Program.fingerprint` was explicitly **not** added (out of scope).

1. **`setup_cython.py`**
   - `_PYX_DIR`: `os.path.join(_THIS, "src", "chartlib", "_symbolic")`
     → `os.path.join(_THIS, "src", "polyarray")`.
   - Docstring path reference `src/chartlib/_symbolic/` → `src/polyarray/`
     and "chartlib package" → "polyarray package" (comment text only).
   - `setup(name="chartlib-cython-ext")` → `name="polyarray-cython-ext"`
     (distribution label for the standalone ext build only).

2. **`ir.py`, `rational.py`, `poly_backend.py`, `poly_native_py.py`,
   `poly_native_value.py`, `int_atom.py`, `forward.py`,
   `_poly_native_cpp_double.pyx`, `_poly_native_cpp_mpf.pyx`**
   - **No edits.** Copied byte-for-byte. Their relative imports
     (`from .ir import …`, `from .rational import …`, etc.) resolve
     within `polyarray/` unchanged. Verified there are no chartlib-,
     interpreter-, per_point-, integrate-, geometry-, or topology-
     specific imports in any copied file.

## Notes

- `poly_backend.py` still reads the `CHARTLIB_POLY_BACKEND` /
  `CHARTLIB_POLY_COEFF` environment variables for backend selection.
  This was preserved **verbatim** (plan §4: "Backend selection
  unchanged ... preserved verbatim"). Renaming the env var would be a
  behavior change and is intentionally **not** done in Stage 1.
- The `flint` backend (default when `python-flint` is installed) is
  preserved verbatim. `python-flint` is an *optional* dependency
  (`[flint]` extra); it is auto-detected and falls back to sympy when
  absent — matching chartlib's existing behavior.
