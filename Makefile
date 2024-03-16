# This Makefile is used to automate some development tasks.
# Ideally this logic would be in pyproject.toml but it appears
# easier to do it this way for now.

PYTHON        = python #3.12
PROJECT       = cvxportfolio
TESTS         = $(PROJECT)/tests
BUILDDIR      = build
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin
EXTRA_SCRIPTS = bumpversion.py
EXAMPLES      = examples
# if you want to use (e.g., debian) packaged numpy/scipy/pandas, ...
# probably improves performance (on debian, at least)
# in the test suite in github we install everything from pip, including
# the last available dependencies versions for all platforms
VENV_OPTS     = --system-site-packages

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

# this way any Python command calls the venv interpreter with its own sys.path
export PATH   := $(BINDIR):$(PATH)

.PHONY: env clean update test lint docs opendocs coverage fix release examples

env:  ## create environment
	$(PYTHON) -m venv $(VENV_OPTS) $(ENVDIR)
	pip install --editable .[docs,dev,examples]
	
clean:  ## clean environment
	-rm -rf $(BUILDDIR)/*
	-rm -rf $(PROJECT).egg*
	-rm -rf $(ENVDIR)/*

update: clean env  ## update environment
	
test:  ## run tests w/ cov report
	coverage run -m $(PROJECT).tests
	coverage report
	coverage xml
	diff-cover coverage.xml --config-file pyproject.toml

lint:  ## run linter
	pylint $(PROJECT) $(EXTRA_SCRIPTS) # $(EXAMPLES)
	diff-quality --violations=pylint --config-file pyproject.toml

docs:  ## build docs
	sphinx-build -E docs $(BUILDDIR)

opendocs: docs  ## open html docs
	open build/index.html

coverage:  ## open html cov report
	coverage html --fail-under=0 # overwrite pyproject.toml default
	open htmlcov/index.html

fix:  ## auto-fix code
	# selected among many code auto-fixers, tweaked in pyproject.toml
	autopep8 -i -r $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)
	isort $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)
	# this is the best found for the purpose
	docformatter -r --in-place $(PROJECT) $(EXAMPLES) $(EXTRA_SCRIPTS)

release: update lint test  ## update version, publish to pypi
	python bumpversion.py
	git push --no-verify
	build
	twine check dist/*
	twine upload --skip-existing dist/*

examples:  ## run examples for docs
	for example in hello_world case_shiller universes dow30; \
		do env CVXPORTFOLIO_SAVE_PLOTS=1 python -m examples."$$example" > docs/_static/"$$example"_output.txt; \
	done
	mv *.png docs/_static/

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'
