# polyarray documentation. `make html` writes docs/build/html; `make doctest`
# runs the examples page and fails if the API drifts from what it claims.

# Warnings are reported, not fatal: a handful of docstrings still carry prose
# reST docutils objects to. Run `make html SPHINXOPTS=-W` to make them fatal.
SPHINXOPTS    ?=
SPHINXBUILD   ?= $(PYTHON) -m sphinx
PYTHON        ?= python
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html doctest clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

doctest:
	$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
