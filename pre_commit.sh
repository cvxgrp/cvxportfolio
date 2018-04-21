## TODO: move everything to docker 

source activate cvxportfolio-testing
nosetests
flake8 --exclude="*/__init__.py" .
source deactivate

source activate cvxportfolio-testing-py2
nosetests
flake8 --exclude="*/__init__.py" .
source deactivate