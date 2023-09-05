BUILDDIR      = build
PYTHON        = python
PROJECT       = cvxportfolio
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: env docs clean pytest test releasetest cleanenv

env:
	$(PYTHON) -m venv $(ENVDIR)
	$(BINDIR)/python -m pip install --editable .
	$(BINDIR)/python -m pip install -r requirements.txt

test:
	$(BINDIR)/coverage run -m unittest $(PROJECT)/tests/*.py
		
pytest:
	$(BINDIR)/pytest $(PROJECT)/tests/*.py
	
releasetest: cleanenv env pytest  

test8:
	flake8 --per-file-ignores='$(PROJECT)/__init__.py:F401,F403' $(PROJECT)/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	-rm -rf $(PROJECT).egg*

cleanenv:
	-rm -rf $(ENVDIR)/*

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR); open build/index.html

coverage: test
	$(BINDIR)/coverage html
	open htmlcov/index.html

pep8:
	# use autopep8 to make innocuous fixes 
	$(BINDIR)/autopep8 -i $(PROJECT)/*.py $(PROJECT)/tests/*.py

release: releasetest
	$(BINDIR)/python bumpversion.py
	git push
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*