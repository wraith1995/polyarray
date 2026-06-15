# polyarray

Standalone symbolic-numeric array IR — `Program` / `Stmt` / `SymArray` /
`RationalFunction` plus all polynomial backends (sympy, native_py,
native_value, and the Cython C++ `native_cpp` backends) and the
`analyze` / `partial_eval` IR passes.

Extracted faithfully from chartlib's `_symbolic` core; see `VENDORED.md`
for provenance and the exact (import-path-only) edits. The committed
public surface is enumerated in `PUBLIC_API.md`.

## Install

```sh
pip install -e .            # pure-Python backends work immediately
make cython                 # optional: build the native_cpp .so backends in place
```

`python-flint` (optional, `pip install -e '.[flint]'`) enables the fast
exact-rational `flint` backend, auto-detected at import.

## Use

```python
import numpy as np
from polyarray import Program, SymInput, Provenance

prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
prog.add_output("det", prog.input("A").det().cells)
prog.run({"A": np.eye(2)})   # -> {"det": array(1.0)}
```
