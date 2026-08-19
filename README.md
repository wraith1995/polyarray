# polyarray

[![CI](https://github.com/wraith1995/polyarray/actions/workflows/ci.yml/badge.svg)](https://github.com/wraith1995/polyarray/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://wraith1995.github.io/polyarray/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A symbolic-numeric array IR. A `Program` is an ordered list of statements over
`SymArray` values whose cells are exact rational functions, plain floats, or
references to the output of a deferred op. Computation that stays symbolic lives in
the cells as rational functions; computation deferred to a numeric or opaque
operation becomes a statement — so one program expresses symbolic and numeric work
together. A program executes as plain Python, and the same IR renders back to NumPy
source or compiles to a batched PyTorch kernel.

**If you want to symbolically simplify part of a program with a computer algebra
system and then lower the whole thing to an array language (NumPy, PyTorch), this
is the tool.** Symbolic representations can explode — a dense `n × n` symbolic
inverse by Cramer's rule runs to `n · n!` monomials — so polyarray parametrizes
how much of a computation is kept symbolic with a `SymbolicBudget`, trading
simplification power against size.

- **Use case:** <https://wraith1995.github.io/polyarray/use_case.html>
- **Docs:** <https://wraith1995.github.io/polyarray/>
- **Public API:** <https://wraith1995.github.io/polyarray/public_api.html>
- **Docstring style:** <https://wraith1995.github.io/polyarray/docstring_style.html>

## Install

```sh
pip install -e .            # the pure-Python backends work immediately
make cython                 # optional: build the native_cpp .so backends in place
```

`python-flint` (optional, `pip install -e '.[flint]'`) enables the fast exact-rational
`flint` backend, which is auto-detected at import and falls back to `sympy` when absent.

## Use

```python
import numpy as np
from polyarray import Program, SymInput, Provenance

prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
prog.add_output("det", prog.input("A").det().cells)
prog.run({"A": np.eye(2)})   # -> {"det": array(1.0)}
```

The polynomial ring behind a `RationalFunction` has four implementations — `sympy`,
`flint`, the pure-Python `native_py`, and the Cython/C++ `native_cpp` — selected at
import by the `CHARTLIB_POLY_BACKEND` environment variable, defaulting to `flint` when
python-flint is installed and `sympy` otherwise.

## Development

```sh
pip install -e ".[dev]"     # tests, ruff, mypy
pytest                      # run the suite (parallel by default)
ruff check src              # docstring + annotation lint
mypy                        # static types
pip install -e ".[docs]" && sphinx-build -W -b html docs/source docs/build/html   # build the docs
```
