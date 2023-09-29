PYTHON        = python
PYTHON310     = python3.10
PYTHON309     = python3.9
PROJECT       = cvxportfolio
TESTS         = $(PROJECT)/tests
COVERAGE      = 97  # target coverage score
DIFFCOVERAGE  = 99  # target coverage of new code
LINT          = 8  # target lint score
PYLINT_OPTS   = --good-names t,u,v,w,h --ignored-argument-names kwargs
BUILDDIR      = build
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin
BINDIR310     = $(ENVDIR)310/bin
BINDIR309     = $(ENVDIR)309/bin
DEPS310       = numpy==1.23.4 scipy==1.9.3 cvxpy==1.2.3 pandas==1.5.0 osqp==0.6.2.post9 ecos==2.0.12 scs==3.2.2 requests==2.28.1
DEPS309       = numpy==1.21.5 scipy==1.7.3 cvxpy==1.1.17 pandas==1.4.0 osqp==0.6.2.post0 ecos==2.0.11 scs==2.1.4 requests==2.26.0


ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
    BINDIR310=$(ENVDIR)310/Scripts
    BINDIR309=$(ENVDIR)309/Scripts
endif

.PHONY: env test lint clean docs opendocs coverage release fix env309 test309 env310 test310

env:
	$(PYTHON) -m venv $(ENVDIR)
	$(BINDIR)/python -m pip install --editable .
	$(BINDIR)/python -m pip install -r requirements.txt
	
env310:
	$(PYTHON310) -m venv $(ENVDIR)310
	$(BINDIR310)/python -m pip install $(DEPS310)
	$(BINDIR310)/python -m pip install --editable .
	
env309:
	$(PYTHON309) -m venv $(ENVDIR)309
	$(BINDIR309)/python -m pip install $(DEPS309)
	$(BINDIR309)/python -m pip install --editable .
	
test310: env310
	$(BINDIR310)/python -m $(PROJECT).tests

test309: env309
	$(BINDIR309)/python -m $(PROJECT).tests
	
test:
	$(BINDIR)/coverage run -m $(PROJECT).tests
	$(BINDIR)/coverage report --fail-under $(COVERAGE)
	$(BINDIR)/coverage xml
	$(BINDIR)/diff-cover --fail-under $(DIFFCOVERAGE) --compare-branch origin/master coverage.xml

lint:
	$(BINDIR)/pylint $(PYLINT_OPTS) --fail-under $(LINT) $(PROJECT)

# hardtest: test
#	$(BINDIR)/bandit $(PROJECT)/*.py $(PROJECT)/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	-rm -rf $(PROJECT).egg*
	-rm -rf $(ENVDIR)/*

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

release: cleanenv env lint test test309 test310
	$(BINDIR)/python bumpversion.py
	git push
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*