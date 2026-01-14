# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets
.PHONY: clean html livehtml linkcheck

clean:
	rm -rf $(BUILDDIR)/*
	@echo "Build directory cleaned."

html:
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--watch ../lrs \
		--ignore "*.pyc" \
		--ignore "*~" \
		--ignore "*.swp" \
		--ignore "*.swo" \
		--ignore "*/__pycache__/*" \
		--port 8000 \
		--open-browser
	@echo
	@echo "Live server running at http://localhost:8000"

linkcheck:
	@$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

spelling:
	@$(SPHINXBUILD) -b spelling $(SOURCEDIR) $(BUILDDIR)/spelling $(SPHINXOPTS)
	@echo
	@echo "Spell check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/spelling/output.txt."

coverage:
	@$(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage $(SPHINXOPTS)
	@echo
	@echo "Coverage report complete; open $(BUILDDIR)/coverage/python.txt"

doctest:
	@$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest $(SPHINXOPTS)
	@echo
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

latexpdf:
	@$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

epub:
	@$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub $(SPHINXOPTS)
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

# Development helpers
.PHONY: serve install check

serve: livehtml

install:
	pip install -r requirements.txt
	@echo "Documentation dependencies installed."

check: linkcheck spelling
	@echo "Documentation checks complete."

# Quick rebuild (useful during development)
.PHONY: quick
quick:
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) -W --keep-going
	@echo
	@echo "Quick build finished. The HTML pages are in $(BUILDDIR)/html."

