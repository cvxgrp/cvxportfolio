# This Makefile is used to automate some development tasks.
# Ideally this logic would be in pyproject.toml but it appears
# easier to do it this way for now.

PYTHON        = python
PROJECT       = cvxportfolio
TESTS         = $(PROJECT)/tests
BUILDDIR      = build
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin
EXTRA_SCRIPTS = bumpversion.py run_examples.py
EXAMPLES      = examples/*.py

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: env clean update test lint docs opendocs coverage fix release

env:
	$(PYTHON) -m venv $(ENVDIR)
	$(BINDIR)/python -m pip install --editable .
	$(BINDIR)/python -m pip install -r requirements.txt
	
clean:
	-rm -rf $(BUILDDIR)/*
	-rm -rf $(PROJECT).egg*
	-rm -rf $(ENVDIR)/*

update: clean env
	
test:
	$(BINDIR)/coverage run -m $(PROJECT).tests
	$(BINDIR)/coverage report
	$(BINDIR)/coverage xml
	$(BINDIR)/diff-cover coverage.xml --config-file pyproject.toml
	# disabled for now, we need to change pickle as default on-disk cache
	# $(BINDIR)/bandit $(PROJECT)/*.py $(PROJECT)/tests/*.py

lint:
	$(BINDIR)/pylint $(PROJECT)

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR)

opendocs: docs
	open build/index.html

coverage:
	$(BINDIR)/coverage html --fail-under=0 # overwrite pyproject.toml default
	open htmlcov/index.html

fix:
	# selected among many code auto-fixers, tweaked in pyproject.toml
	$(BINDIR)/autopep8 -i $(PROJECT)/*.py $(TESTS)/*.py
	$(BINDIR)/isort $(PROJECT)/*.py $(TESTS)/*.py
	# this is the best found for the purpose
	$(BINDIR)/docformatter --in-place $(PROJECT)/*.py $(TESTS)/*.py

release: update lint test
	$(BINDIR)/python bumpversion.py
	git push --no-verify
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*