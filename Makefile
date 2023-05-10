SPHINXBUILD   = sphinx-build -E
BUILDDIR      = build
TESTRUNNER	  = poetry run python -m unittest 


.PHONY: docs clean test
	
tests:
	$(TESTRUNNER) cvxportfolio/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/*

docs:
	$(SPHINXBUILD) docs $(BUILDDIR); open build/index.html