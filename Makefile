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
	$(BINDIR)/python -m pip install --editable .
	$(BINDIR)/python -m pip install -r requirements.txt
	
test:
	$(BINDIR)/pytest $(PROJECT)/tests/*.py
	
hardtest: cleanenv env test  

test8:
	flake8 --per-file-ignores='$(PROJECT)/__init__.py:F401,F403' $(PROJECT)/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	-rm -rf $(PROJECT).egg*

cleanenv:
	-rm -rf $(ENVDIR)/*

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR); open build/index.html

pep8:
	# use autopep8 to make innocuous fixes 
	$(BINDIR)/autopep8 -i $(PROJECT)/*.py $(PROJECT)/tests/*.py

release: hardtest
	$(BINDIR)/python bumpversion.py
	git push
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*