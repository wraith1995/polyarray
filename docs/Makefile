# polyarray documentation. `make html` writes docs/build/html.

# Warnings are reported, not fatal: a handful of docstrings still carry prose
# reST docutils objects to. Run `make html SPHINXOPTS=-W` to make them fatal.
SPHINXOPTS    ?=
SPHINXBUILD   ?= $(PYTHON) -m sphinx
PYTHON        ?= python
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
