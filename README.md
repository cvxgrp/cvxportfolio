CVXPortfolio
=============

CVXPortfolio (ConVeX Portfolio Optimization and Simulation)
is a toolset based on our paper [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
It is written in Python, its major dependencies are [CVXPY](https://github.com/cvxgrp/cvxpy)
and [Pandas](https://github.com/pandas-dev/pandas).

See the [examples](https://github.com/cvxgrp/cvxportfolio/tree/master/examples) for basic usage.

If you wish to cite CVXPortfolio, please use:
```
@article{BBDKKNS:17
    author       = {S. Boyd and E. Busseti and S. Diamond and R. Kahn and K. Koh and P. Nystrup and J. Speth},
    title        = {Multi-Period Trading via Convex Optimization},
    journal      = {Foundations and Trends in Optimization},
    year         = {2017},
    pages        = {to appear}
    publisher    = {Now Publishers}
    url          = {http://stanford.edu/~boyd/papers/cvx_portfolio.html},
}
```

Installation
------------

To install the package run
```
pip install git+git://github.com/cvxgrp/cvxportfolio
```
