SPHINXBUILD   = sphinx-build -E
BUILDDIR      = build
PYTHON		  = python
ENVDIR        = env


.PHONY: docs clean test env
	
env:
	$(PYTHON) -m venv $(ENVDIR)
	$(ENVDIR)/bin/$(PYTHON) -m pip install -r requirements.txt
	
test:
	$(ENVDIR)/bin/$(PYTHON) -m unittest cvxportfolio/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	-rm -rf $(ENVDIR)/*

docs:
	$(SPHINXBUILD) docs $(BUILDDIR); open build/index.html