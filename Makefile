BUILDDIR      = build
PYTHON		  = python
ENVDIR        = env


.PHONY: env docs clean test cleanenv
	
env:
	$(PYTHON) -m venv $(ENVDIR)
	$(ENVDIR)/bin/$(PYTHON) -m pip install -r requirements.txt
	$(ENVDIR)/bin/$(PYTHON) -m pip install --editable .
	
test: env
	$(ENVDIR)/bin/$(PYTHON) -m unittest cvxportfolio/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	
cleanenv:
	-rm -rf $(ENVDIR)/*

docs: env
	$(ENVDIR)/bin/sphinx-build -E docs $(BUILDDIR); open build/index.html