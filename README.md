# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-green.svg)](https://github.com/cvxgrp/cvxportfolio/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)

**The documentation of the package is given at [cvxportfolio.readthedocs.io](https://cvxportfolio.readthedocs.io/en/latest/).**

`cvxportfolio` is a python library for portfolio optimization and simulation,
based on the paper [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
It is written in Python, its major dependencies are [`cvxpy`](https://github.com/cvxgrp/cvxpy)
and [`pandas`](https://github.com/pandas-dev/pandas).


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

Installation
------------

To install the package from pypi:
```
pip install cvxportfolio
```

To test it:

```
poetry install
poetry run pytest
```

