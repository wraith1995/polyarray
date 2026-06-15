# polyarray — local dev tasks.
#
# `make cython` builds the native_cpp polynomial backends in place
# (next to their .pyx siblings under src/polyarray/), matching the
# editable-install dev flow. Wheels build the same extensions via
# setup.py + cibuildwheel.

PYTHON ?= python

.PHONY: cython test clean

cython:
	$(PYTHON) setup_cython.py build_ext --inplace

test:
	$(PYTHON) -m pytest tests -q

clean:
	rm -f src/polyarray/_poly_native_cpp_*.so
	rm -rf build
	find src/polyarray -name '*.c' -delete
	find src/polyarray -name '*.cpp' -delete
