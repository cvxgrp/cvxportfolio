BUILDDIR      = build
PYTHON        = python
PROJECT       = cvxportfolio
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: env docs clean test cleanenv

env:
	$(PYTHON) -m venv $(ENVDIR)
	$(BINDIR)/python -m pip install -r requirements.txt
	$(BINDIR)/python -m pip install --editable .
	
test:
	$(BINDIR)/python -m unittest $(PROJECT)/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	-rm -rf cvxportfolio.egg*

cleanenv:
	-rm -rf $(ENVDIR)/*

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR); open build/index.html

release: test
	$(BINDIR)/python bumpversion.py
	git push
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*