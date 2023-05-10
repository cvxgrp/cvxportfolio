BUILDDIR      = build
PYTHON		  = python
ENVDIR        = env


.PHONY: docs clean test env cleanenv
	
env:
	$(PYTHON) -m venv $(ENVDIR)
	$(ENVDIR)/bin/$(PYTHON) -m pip install -r requirements.txt
	
test:
	$(ENVDIR)/bin/$(PYTHON) -m unittest cvxportfolio/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	
cleanenv:
	-rm -rf $(ENVDIR)/*

docs:
	$(ENVDIR)/bin/sphinx-build -E docs $(BUILDDIR); open build/index.html