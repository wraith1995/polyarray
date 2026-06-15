"""Standalone Cython build script for the ``native_cpp`` polynomial
backend (Track B of ``plans/poly_native_backend.md``).

Intentionally separate from ``pyproject.toml``'s build-backend
(``uv_build``) so the pure-Python install path stays simple.  Invoke
explicitly to build the extension modules in place::

    make cython
    # equivalently:
    python setup_cython.py build_ext --inplace

The produced ``.so`` files land next to their ``.pyx`` siblings inside
``src/polyarray/``; the polyarray package is editable-
installed via a ``.pth`` pointer at ``src/``, so the freshly built
extensions are picked up immediately on the next import.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


_THIS = os.path.dirname(os.path.abspath(__file__))
_PYX_DIR = os.path.join(_THIS, "src", "polyarray")


def _extensions() -> list[Extension]:
    out: list[Extension] = []
    for pyx in sorted(glob.glob(os.path.join(_PYX_DIR, "_poly_native_cpp_*.pyx"))):
        rel = os.path.relpath(pyx, os.path.join(_THIS, "src"))
        mod = rel.replace(os.sep, ".")[:-len(".pyx")]
        out.append(
            Extension(
                mod,
                sources=[pyx],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3", "-ffast-math"],
            )
        )
    return out


setup(
    name="polyarray-cython-ext",
    ext_modules=cythonize(
        _extensions(),
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
    package_dir={"": "src"},
    zip_safe=False,
)
