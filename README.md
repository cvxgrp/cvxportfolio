# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxportfolio/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)
[![Binder](http://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/cvxgrp/cvxportfolio/HEAD)


**Cvxportfolio is currently under development. We will freeze the user interface by end of 2023Q2 and release the first stable version by end of 2023Q3.**


`cvxportfolio` is a python library for portfolio optimization and simulation,
based on the paper [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).
It is written in Python, its major dependencies are [`cvxpy`](https://github.com/cvxgrp/cvxpy)
and [`pandas`](https://github.com/pandas-dev/pandas). 
The documentation of the package is at [cvxportfolio.readthedocs.io](https://cvxportfolio.readthedocs.io/en/latest/).




Installation
------------

```
pip install cvxportfolio
```

Testing
------------

To test it locally set up the development environment with [`poetry`](https://python-poetry.org/) (you will need to
install it first) and run [`pytest`](https://pytest.org/). 


```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
poetry install
poetry run pytest --cov
```

Examples
------------

You can see basic usage of the package in the [example notebooks](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/).
Currently we are working on simplifying the user interface and these may change.

To run them clone the repository, create the environment and the `cvxportfolio` kernel, and then start [`jupyter`](https://jupyter.org/).

```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
poetry install
bash create_kernel.sh
cd examples
poetry run jupyter notebook
```

The ones that run without isses (as of 2023-04-11) are [HelloWorld](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/HelloWorld.ipynb) and [MultiPeriodTCostOptimization](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/MultiPeriodTCostOptimization.ipynb).

The other example notebooks were used to develop the plots and results in [the paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf). We are keeping 
them for historical record but they don't currently run. 
We are doing our best to restore them. 


Citing
------------

If you wish to cite CVXPortfolio, please use:
```
@article{BBDKKNS:17,
    author       = {S. Boyd and E. Busseti and S. Diamond and R. Kahn and K. Koh and P. Nystrup and J. Speth},
    title        = {Multi-Period Trading via Convex Optimization},
    journal      = {Foundations and Trends in Optimization},
    year         = {2017},
    month        = {August},
    volume       = {3},
    number       = {1},
    pages        = {1--76},
    publisher    = {Now Publishers},
    url          = {http://stanford.edu/~boyd/papers/cvx_portfolio.html},
}
```


License
------------

Cvxportfolio is licensed under the [Apache 2.0](http://www.apache.org/licenses/) permissive
open source license.


