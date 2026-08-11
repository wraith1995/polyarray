"""Sphinx configuration for the polyarray documentation.

The API reference is generated from the source: :mod:`sphinx.ext.autosummary`
walks the package recursively and :mod:`numpydoc` renders the docstrings, so
every documented function appears with its signature and its numpydoc sections.
Narrative pages are written by hand under ``docs/source``.

Build with ``make -C docs html``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "polyarray"
copyright = "Thea Collin"
author = "Thea Collin"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "numpydoc",
]

# The examples page is written as doctests, so ``make -C docs doctest`` runs the
# narrative and fails if the API drifts from what the docs claim.

# The narrative pages left for the maintainer mark their gaps with ``todo``;
# show them, so an unwritten section is visible in the built docs.
todo_include_todos = True

# Autosummary generates one stub page per module, class and function, so the
# reference tracks the source rather than a hand-maintained list.
autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# A frozen dataclass with several fields (e.g. SimplifyBudget) renders as one
# long signature line; wrap any signature past this width to one parameter per
# line so the reference stays readable.
python_maximum_signature_line_length = 72
autodoc_type_aliases = {
    "Cell": "polyarray.ir.Cell",
    "Ref": "polyarray.ir.Ref",
    "StmtFn": "polyarray.ir.StmtFn",
    "StmtOp": "polyarray.ir.StmtOp",
}

# numpydoc renders the sections; its own validation runs from `make lint-docs`,
# not from the docs build, so a build never fails on a docstring nit.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks: set[str] = set()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []
html_static_path = ["_static"]
html_theme = "alabaster"
html_title = "polyarray"

# Nothing is mocked: the workspace venv holds the whole stack, and mocking a
# module sympy probes for at import time breaks the build rather than helping.
autodoc_mock_imports: list[str] = []


# Internal structural-typing Protocols used only for isinstance dispatch — not
# part of the documented surface.
_INTERNAL_PROTOCOLS = {"VmapClosure", "NestedVmapClosure"}


def _skip_member(app, what, name, obj, skip, options):
    """Keep the reference to the public surface.

    Private members (leading underscore) are internal; a module-level type alias
    (e.g. ``Monom = tuple[int, ...]``) renders as a class whose ``tuple``
    signature autodoc cannot format, so it belongs inline in the module
    docstring rather than as its own stub; and the internal dispatch Protocols
    are implementation detail.
    """
    import types

    if name.startswith("_"):
        return True
    if name in _INTERNAL_PROTOCOLS:
        return True
    if isinstance(obj, types.GenericAlias):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_member)
