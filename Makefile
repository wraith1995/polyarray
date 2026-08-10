# polyarray — local dev tasks.
#
# `make cython` builds the native_cpp polynomial backends in place
# (next to their .pyx siblings under src/polyarray/), matching the
# editable-install dev flow. Wheels build the same extensions via
# setup.py + cibuildwheel.

PYTHON ?= python

.PHONY: cython test clean lint lint-docs

cython:
	$(PYTHON) setup_cython.py build_ext --inplace

test:
	$(PYTHON) -m pytest tests -q

clean:
	rm -f src/polyarray/_poly_native_cpp_*.so
	rm -rf build
	find src/polyarray -name '*.c' -delete
	find src/polyarray -name '*.cpp' -delete

# Docstring + annotation gate (config in pyproject.toml): ruff's D/ANN rules under the
# numpy convention, then numpydoc's consistency checks on the sections a docstring
# declares. Both must be clean before committing.
lint-docs:
	$(PYTHON) -m ruff check src
	$(PYTHON) -m numpydoc lint src/polyarray/*.py

lint: lint-docs
	$(PYTHON) -m mypy
