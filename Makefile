PYTHON        = python
PROJECT       = cvxportfolio
TESTS         = $(PROJECT)/tests
COVERAGE      = 97  # target coverage score
DIFFCOVERAGE  = 99  # target coverage of new code
LINT          = 8  # target lint score
PYLINT_OPTS   = --good-names t,u,v,w,h,z --ignored-argument-names kwargs,universe,trading_calendar
BUILDDIR      = build
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin


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
	$(BINDIR)/coverage report --fail-under $(COVERAGE)
	$(BINDIR)/coverage xml
	$(BINDIR)/diff-cover --fail-under $(DIFFCOVERAGE) --compare-branch origin/master coverage.xml

lint:
	$(BINDIR)/pylint $(PYLINT_OPTS) --fail-under $(LINT) $(PROJECT)

# hardtest: test
#	$(BINDIR)/bandit $(PROJECT)/*.py $(PROJECT)/tests/*.py

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR)

opendocs: docs
	open build/index.html

coverage:
	$(BINDIR)/coverage html
	open htmlcov/index.html

fix:
	# selected among many popular python code auto-fixers
	$(BINDIR)/autopep8 --select W291,W293,W391,E231,E225,E303 -i $(PROJECT)/*.py $(TESTS)/*.py
	$(BINDIR)/isort $(PROJECT)/*.py $(TESTS)/*.py

release: update lint test
	$(BINDIR)/python bumpversion.py
	git push
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*