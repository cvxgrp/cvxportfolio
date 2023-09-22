BUILDDIR      = build
PYTHON        = python
PROJECT       = cvxportfolio
ENVDIR        = env
BINDIR        = $(ENVDIR)/bin

ifeq ($(OS), Windows_NT)
    BINDIR=$(ENVDIR)/Scripts
endif

.PHONY: env test hardtest clean docs opendocs coverage fix hardfix release 

env:
	$(PYTHON) -m venv $(ENVDIR)
	$(BINDIR)/python -m pip install --editable .
	$(BINDIR)/python -m pip install -r requirements.txt

test:
	$(BINDIR)/coverage run -m unittest $(PROJECT)/tests/*.py
	$(BINDIR)/coverage report
	$(BINDIR)/coverage xml
	$(BINDIR)/diff-cover --compare-branch origin/master coverage.xml

hardtest:
	$(BINDIR)/pytest --cov --cov-report=xml -W error $(PROJECT)/tests/*.py
	$(BINDIR)/coverage report --fail-under 99
	$(BINDIR)/ruff --per-file-ignores='$(PROJECT)/__init__.py:F403' $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/isort --check-only $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/flake8 --per-file-ignores='$(PROJECT)/__init__.py:F401,F403' $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/docstr-coverage $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/bandit $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/pylint $(PROJECT)/*.py $(PROJECT)/tests/*.py

clean:
	-rm -rf $(BUILDDIR)/* 
	-rm -rf $(PROJECT).egg*
	-rm -rf $(ENVDIR)/*

docs:
	$(BINDIR)/sphinx-build -E docs $(BUILDDIR)
	
opendocs: docs
	open build/index.html

coverage: test
	$(BINDIR)/coverage html
	open htmlcov/index.html

fix:
	# (mostly) whitespace fixes
	$(BINDIR)/autopep8 --select W291,W293,W391,E231,E225,E303 -i $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/pydocstringformatter --write $(PROJECT)/*.py $(PROJECT)/tests/*.py

hardfix:
	# could be breaking fixes
	$(BINDIR)/ruff --fix $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/autopep8 --aggressive --aggressive --aggressive -i $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/pydocstringformatter --linewrap-full-docstring --write $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/autoflake $(PROJECT)/*.py $(PROJECT)/tests/*.py
	$(BINDIR)/isort $(PROJECT)/*.py $(PROJECT)/tests/*.py

release: cleanenv env test
	$(BINDIR)/python bumpversion.py
	git push
	$(BINDIR)/python -m build
	$(BINDIR)/twine check dist/*
	$(BINDIR)/twine upload --skip-existing dist/*