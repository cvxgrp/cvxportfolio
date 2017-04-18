Cvx_portfolio
=============

Cvx_portfolio is a toolset for (convex) portfolio optimization and simulation,
based on our paper [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).
It is written in Python 3, its major dependencies are [CVXPY](https://github.com/cvxgrp/cvxpy)
and [Pandas](https://github.com/pandas-dev/pandas).

See the [examples](examples/) for basic usage.

If you wish to cite cvx_portfolio, please use:
```
@article{BBDKKNS:17
    author       = {S. Boyd and E. Busseti and S. Diamond and R. Kahn and K. Koo and P. Nystrup and J. Speth},
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

To install the package (Linux/Mac) run
```
pip install git+git://github.com/cvxgrp/cvx_portfolio
```
