BUILDDIR      = build
PYTHON        = python
ENVDIR        = env


.PHONY: env docs clean test cleanenv
	
env:
	$(PYTHON) -m venv $(ENVDIR)
	$(ENVDIR)/bin/python -m pip install -r requirements.txt
	$(ENVDIR)/bin/python -m pip install --editable .
	
test:
	$(ENVDIR)/bin/python -m unittest cvxportfolio/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	
cleanenv:
	-rm -rf $(ENVDIR)/*

docs:
	$(ENVDIR)/bin/sphinx-build -E docs $(BUILDDIR); open build/index.html
	
revision:
	$(ENVDIR)/bin/python bumpversion.py revision	
	git push
	$(ENVDIR)/bin/python setup.py sdist bdist_wheel
	$(ENVDIR)/bin/twine upload dist/*

minor:
	$(ENVDIR)/bin/python bumpversion.py minor	
	git push
	$(ENVDIR)/bin/python setup.py sdist bdist_wheel
	$(ENVDIR)/bin/twine upload dist/*

major:
	$(ENVDIR)/bin/python bumpversion.py major	
	git push
	$(ENVDIR)/bin/python setup.py sdist bdist_wheel
	$(ENVDIR)/bin/twine upload dist/*