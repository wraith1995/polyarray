"""Wheel build entry point — compiles the Cython cpp polynomial backends.

``pyproject.toml`` declares ``setuptools.build_meta`` as the build
backend and lists ``Cython`` + ``numpy`` in ``build-system.requires``;
this ``setup.py`` supplies the ``ext_modules`` so ``pip wheel`` /
cibuildwheel ship the precompiled ``.so`` files.

For local editable dev, ``make cython`` (which runs ``setup_cython.py
build_ext --inplace``) builds the extensions in place next to their
``.pyx`` siblings instead. Both paths use the identical glob + compiler
directives; this file just shares them with the declarative build.
"""
from __future__ import annotations

import glob
import os

from setuptools import setup

_THIS = os.path.dirname(os.path.abspath(__file__))
_PYX_DIR = os.path.join(_THIS, "src", "polyarray")


def _ext_modules():
    """Cython Extensions for the cpp backends, or [] if Cython is absent.

    Returning [] (rather than raising) keeps a pure-Python source build
    working when no C++ toolchain / Cython is available; the sympy /
    native_py / native_value backends still function.
    """
    try:
        import numpy as np
        from Cython.Build import cythonize
        from setuptools import Extension
    except ImportError:
        return []

    exts = []
    for pyx in sorted(glob.glob(os.path.join(_PYX_DIR, "_poly_native_cpp_*.pyx"))):
        # Use a path relative to the project root for both the module name
        # and the source: setuptools' editable build rejects absolute
        # source paths.
        src_rel = os.path.relpath(pyx, _THIS)
        mod = os.path.relpath(pyx, os.path.join(_THIS, "src"))
        mod = mod.replace(os.sep, ".")[: -len(".pyx")]
        exts.append(
            Extension(
                mod,
                sources=[src_rel],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3", "-ffast-math"],
            )
        )
    if not exts:
        return []
    return cythonize(
        exts,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    )


setup(ext_modules=_ext_modules())
