# Contributing

## Setup

```
uv sync --extra dev --extra flint
```

The `flint` extra provides exact rational arithmetic; a few tests assert
exact folds and are skipped without it.

## The checks CI runs

```
uv run ruff check src            # style, imports, docstrings, annotations
uv run pytest -q                 # torch- and consumer-dependent tests skip via importorskip
uv run sphinx-build -W -b html docs/source docs/build/html    # zero-warning docs
uv run sphinx-build -b doctest docs/source docs/build/doctest  # runnable examples
uv run mypy || true              # advisory until the baseline reaches zero
```

Run them before opening a pull request.

## Conventions

- **Docstrings** are the API reference — they are generated into the docs. Write
  them in the descriptive style documented in the docs' "Docstring style" page
  (`docs/source/docstring_style.md`): complete sentences, present tense,
  polyarray's own vocabulary, and a diagram where the shape is spatial.
- **numeric vs symbolic is a value property, not a code path.** Thread `SymArray`
  and its `Program`; do not fork on dtype or unwrap cells into object arrays.
- **Examples in the docs run as doctests**, so an example that drifts from the
  API fails the build. Add one when you add a user-facing capability.
