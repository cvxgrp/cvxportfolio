BUILDDIR      = build
PYTHON        = python
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
	$(BIN)/python -m unittest cvxportfolio/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 

cleanenv:
	-rm -rf $(ENVDIR)/*

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR); open build/index.html

revision:
	$(BINDIR)/python bumpversion.py revision	
	git push
	$(BINDIR)/python setup.py sdist bdist_wheel
	$(BINDIR)/twine upload --skip-existing dist/*

minor:
	$(BINDIR)/python bumpversion.py minor	
	git push
	$(BINDIR)/python setup.py sdist bdist_wheel
	$(BINDIR)/twine upload --skip-existing dist/*

major:
	$(BINDIR)/python bumpversion.py major	
	git push
	$(BINDIR)/python setup.py sdist bdist_wheel
	$(BINDIR)/twine upload --skip-existing dist/*