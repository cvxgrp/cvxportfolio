# This Makefile is used to automate some development tasks.
# Ideally this logic would be in pyproject.toml but it appears
# easier to do it this way for now.

PYTHON        = python
PROJECT       = cvxportfolio
TESTS         = $(PROJECT)/tests
BUILDDIR      = build
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin
EXTRA_SCRIPTS = bumpversion.py
EXAMPLES      = examples

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: env clean update test lint docs opendocs coverage fix release examples

env:  ## create environment
	$(PYTHON) -m venv $(ENVDIR)
	$(BINDIR)/python -m pip install --editable .[docs,dev,examples]
	
clean:  ## clean environment
	-rm -rf $(BUILDDIR)/*
	-rm -rf $(PROJECT).egg*
	-rm -rf $(ENVDIR)/*

update: clean env  ## update environment
	
test:  ## run tests w/ cov report
	$(BINDIR)/coverage run -m $(PROJECT).tests
	$(BINDIR)/coverage report
	$(BINDIR)/coverage xml
	$(BINDIR)/diff-cover coverage.xml --config-file pyproject.toml
	# disabled for now, we need to change pickle as default on-disk cache
	# $(BINDIR)/bandit $(PROJECT)/*.py $(TESTS)/*.py

lint:  ## run linter
	$(BINDIR)/pylint $(PROJECT) $(EXTRA_SCRIPTS) # $(EXAMPLES)
	$(BINDIR)/diff-quality --violations=pylint --config-file pyproject.toml

docs:  ## build docs
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR)

opendocs: docs  ## open html docs
	open build/index.html

coverage:  ## open html cov report
	$(BINDIR)/coverage html --fail-under=0 # overwrite pyproject.toml default
	open htmlcov/index.html

fix:  ## auto-fix code
	# selected among many code auto-fixers, tweaked in pyproject.toml
	$(BINDIR)/autopep8 -i -r $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)
	$(BINDIR)/isort $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)
	# this is the best found for the purpose
	$(BINDIR)/docformatter -r --in-place $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)

release: update lint test  ## update version, publish to pypi
	$(BINDIR)/python bumpversion.py
	git push --no-verify
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*

examples:  ## run examples for docs
	for example in hello_world case_shiller universes dow30; \
		do env CVXPORTFOLIO_SAVE_PLOTS=1 $(BINDIR)/python -m examples."$$example" > docs/_static/"$$example"_output.txt; \
	done
	mv *.png docs/_static/

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
