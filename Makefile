lint: ## run linter
	python -m flake8 cvxportfolio setup.py docs_source/conf.py

fix:  ## run black fix
	python -m black cvxportfolio/ setup.py docs_source/conf.py

test:  ## run unit tests
	python -m pytest -v cvxportfolio --cov=cvxportfolio --cov-report=term --cov-branch

clean: ## clean the repository
	find . -name "__pycache__" | xargs  rm -rf 
	find . -name "*.pyc" | xargs rm -rf 
	rm -rf .coverage cover htmlcov logs build dist *.egg-info
	make -C ./docs clean
	rm -rf ./docs/*.*.rst  # generated

docs:  ## make documentation
	make -C ./docs_source html
	cp docs_source/CNAME docs_source/html/
	rm -r docs
	mv docs_source/html docs/

install:  ## install to site-packages
	python -m pip install .

dev:
	python -m pip install .[dev]

dist:  ## create dists
	rm -rf dist build
	python setup.py sdist bdist_wheel
	python -m twine check dist/*
	
publish: dist  ## dist to pypi
	python -m twine upload dist/* --skip-existing

# Thanks to Francoise at marmelab.com for this
.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'

.PHONY: lint fix test clean docs install dev dist publish
