# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)


**Cvxportfolio is currently under development. We will freeze the user interface by end of 2023Q2 and release the first stable version by end of 2023Q3.**


`cvxportfolio` is a python library for portfolio optimization and simulation
based on the book [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).
It is written in Python, its main dependencies are [`cvxpy`](https://github.com/cvxgrp/cvxpy)
and [`pandas`](https://github.com/pandas-dev/pandas). 
The documentation of the package is at [cvxportfolio.readthedocs.io](https://cvxportfolio.readthedocs.io/en/latest/).


Roadmap
------------
Cvxportfolio is currently under fast development. (You can see the commit log.) We expect to have a working minimum viable product by end of Ramdan or, worst case, end of April 2023. In the meantime various things are not up to date. Please be patient. See the section below for comments on the status of the examples. Regarding the documentation, docstrings in the code are currently the best way to see what is happening. Here are the main aspects that are being developed:

- Data interface. Cvxportfolio versions 0.0.X had no logic to ingest and process data, and everything was done externally. We are building a modular interface that can be used with public or private data sources and databases. It is defined and documented in `cvxportfolio.data`. It is meant to also be used by `cvxportfolio` object to store data, *e.g.*, backtest results.
- Problem compilation. Cvxportfolio is now using `cvxpy.Parameters` as placeholders for data objects, it compiles the optimization problem at the start of each backtest, and updates the parameters as the backtest progesses. This, in combination with warm-startable solvers like `osqp`, provides huge speedups on backtest computation with respect to existising methods. The new mechanism is defined and documented in `cvxportfolio.estimator` and is being adopted across the rest of the classes.
- Simulator. It is being rewritten, with the goal of radical simplification of the user interface while keeping all existing features and adding new ones. More realistic costs are being added. Documentation is being written along with the new code in `cvxportfolio.simulator`
- Project management and testing. We migrated to `poetry` and `pytest`, we are still iterating on the best workflow automation tools. **We haven't broken any testfile and we don't intend to.** Tests currently run at a coverage of around 85%, we plan to make it close to 100%.

Installation
------------

```
pip install cvxportfolio
```

Testing
------------

To test it locally, for example, you can set up the development environment with [`poetry`](https://python-poetry.org/) and run [`pytest`](https://pytest.org/). 

```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
poetry install
poetry run pytest --cov
```

Releases
------------
With comments from the [git tags](https://github.com/cvxgrp/cvxportfolio/tags).

- [0.1.1](https://pypi.org/project/cvxportfolio/0.1.1/) Last version before 2023 internals change. If you have built code in
  2016-2023 that uses cvxportfolio versions 0.0.X and relies on internal methods you probably want this release. Subsequent ones will change the internal interfaces. User interfaces instead will mostly remain the same. In addition, this version adds a new module data.py (that here is not called by the rest of cvxportfolio) that simplifies getting data.
- 0.0.X Early development versions. Not tagged in git but [distributed on PyPI](https://pypi.org/project/cvxportfolio/).

Examples
------------

You can see basic usage of the package in the [examples](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/).
These are currently being reworked and simplified. At the moment we don't guarantee they run without issues.

Citing
------------

If you use `cvxportfolio` in your academic work please cite our book:
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


