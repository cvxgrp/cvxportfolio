# This Makefile is used to automate some development tasks.
# Ideally this logic would be in pyproject.toml but it appears
# easier to do it this way for now.

PYTHON        = python
PROJECT       = cvxportfolio
TESTS         = $(PROJECT)/tests
BUILDDIR      = build
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin
EXTRA_SCRIPTS =
EXAMPLES      = examples
VENV_OPTS     =

# Python venv on windows has different location
ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
# if you want to use (e.g., debian) packaged numpy/scipy/pandas, ...
# probably improves performance (on debian, at least); makes no difference
# if you're already using a virtualized Python installation;
# in the test suite in github we install everything from pip, including
# the last available dependencies versions for all platforms...
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		VENV_OPTS += --system-site-packages
	endif
endif

.PHONY: env clean update test lint docs opendocs coverage fix release publish

env:  ## create environment
	$(PYTHON) -m venv $(VENV_OPTS) $(ENVDIR)
	$(BINDIR)/python -m pip install --editable .[docs,dev,examples,test]

clean:  ## clean environment
	-rm -rf $(BUILDDIR)/*
	-rm -rf $(PROJECT).egg*
	-rm -rf $(ENVDIR)/*

update: clean env  ## update environment

test:  ## run tests w/ cov report
	$(BINDIR)/python -m coverage run -m $(PROJECT).tests
	$(BINDIR)/python -m coverage report
	$(BINDIR)/python -m coverage xml
	$(BINDIR)/diff-cover coverage.xml --config-file pyproject.toml

lint:  ## run linter
	$(BINDIR)/python -m pylint $(PROJECT) $(EXTRA_SCRIPTS) # $(EXAMPLES)
	$(BINDIR)/diff-quality --violations=pylint --config-file pyproject.toml

docs:  ## build docs
	$(BINDIR)/python -m sphinx build -E docs $(BUILDDIR)

opendocs: docs  ## open html docs
	open build/index.html

coverage:  ## open html cov report
	$(BINDIR)/python -m coverage html --fail-under=0 # overwrite pyproject.toml default
	open htmlcov/index.html

fix:  ## auto-fix code
	# selected among many code auto-fixers, tweaked in pyproject.toml
	$(BINDIR)/python -m autopep8 -i -r $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)
	$(BINDIR)/python -m isort $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)
	# this is the best found for the purpose
	$(BINDIR)/docformatter -r --in-place $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)

release: update lint test  ## tag new release; trigger publishing on repo
	@git diff --quiet && git diff --cached --quiet || { echo "Error: Git working directory is not clean."; exit 1; }
	$(BINDIR)/python -m rstcheck README.rst
	@echo "SetupTools SCM suggested new version is $$(env/bin/python -m setuptools_scm --strip-dev)"
	@read -p "enter the version tag you want: " version_tag; \
	echo "You entered: $$version_tag"; \
	git tag -a $$version_tag -em "version $$version_tag"; \
	git push; \
	git push --no-verify origin $$version_tag

publish: ## publish to PyPI from local using token
	$(BINDIR)/python -m build
	$(BINDIR)/python -m twine check dist/*
	$(BINDIR)/python -m twine upload --skip-existing dist/*

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
