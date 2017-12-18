CVXPortfolio
=============
[![Build Status](https://travis-ci.org/cvxgrp/cvxportfolio.png?branch=master)](https://travis-ci.org/cvxgrp/cvxportfolio)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master)

**The CVXPortfolio documentation is at [cvxportfolio.org](http://www.cvxportfolio.org/).**

CVXPortfolio (ConVeX Portfolio Optimization and Simulation)
is a toolset based on our paper [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
It is written in Python, its major dependencies are [CVXPY](https://github.com/cvxgrp/cvxpy)
and [Pandas](https://github.com/pandas-dev/pandas).

See the [examples](https://github.com/cvxgrp/cvxportfolio/tree/master/examples) for basic usage.

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

1. Install [Anaconda](https://docs.continuum.io/anaconda/install).
2. Install cvxpy with conda.
```
conda install -c cvxgrp cvxpy
```
3. Install CVXPortfolio with pip.
```
pip install cvxportfolio
```
4. Test the installation with nose.
```
conda install nose
nosetests cvxportfolio
```
