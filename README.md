# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)



`cvxportfolio` is a python library for portfolio optimization and simulation
based on the book [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)
(also [available in print](https://www.amazon.com/Multi-Period-Trading-Convex-Optimization-Foundations/dp/1680833286/)).

The documentation of the package is kindly hosted by [Read the Docs](https://readthedocs.org) at [www.cvxportfolio.com](https://www.cvxportfolio.com). We also show some of our tutorials and examples on our
[youtube channel](https://www.youtube.com/@Cvxportfolio).


Installation
------------
All our source code and releases are kindly hosted by the [Python Package Index](https://pypi.org). You can install the latest one with

```
pip install -U cvxportfolio
```
You can see how this works on our [Installation and Hello World](https://youtu.be/1ThOKEu371M) youtube video.

Testing locally
------------
We ship our unit test suite with the pip package. After installing you can test in you local environment by

```
python -m unittest discover cvxportfolio
```


Simplest Example
----------------
In the following example market data is downloaded by a public source
(Yahoo finance) and the forecasts are computed iteratively, at each point in the backtest, from past data. 
That is, at each point in the backtest,
the policy object only operates on **past data**, and thus the result you get is a realistic simulation of what the strategy would have performed in the market.
The simulator by default includes holding and transaction costs, using the models described in the book, and default parameters that are typical for the US stock market.
The logic used
matches what is described in Chapter 7 of the book. For example, returns are forecasted as the historical mean returns 
and covariances as historical covariances (both ignoring `np.nan`'s). The logic used is detailed in the `forecast` module. Many optimizations
are applied to make sure the system works well with real data. 


```python
import cvxportfolio as cvx

gamma = 3       # risk aversion parameter (Chapter 4.2)
kappa = 0.05    # covariance forecast error risk parameter (Chapter 4.3)
objective = cvx.ReturnsForecast() - gamma * (
	cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.StocksTransactionCost()
constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

simulator = cvx.StockMarketSimulator(['AAPL', 'AMZN', 'TSLA', 'GM', 'CVX', 'NKE'])

result = simulator.backtest(policy, start_time='2020-01-01')

# print backtest result statistics
print(result)

# plot backtest results
result.plot()
```

Some Other Examples
-------------------
We show in the example on [user-provided forecasters](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/user_provided_forecasters.py) how the user can define custom classes to forecast
the expected returns and covariances. These provide callbacks that are
executed at each point in time during the backtest. The system enforces 
causality and safety against numerical errors. 
We recommend to always include 
the default forecasters that we provide in any analysis you may do, 
since they are very robust and well-tested. 

We show in the examples on [DOW30 components](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/dow30_example.py) and [wide assets-classes ETFs](https://github.com/cvxgrp/cvxportfolio/blob/master/examples/etfs_example.py) how a
simple sweep over hyper-parameters, taking advantage of our sophisticated parallel backtest machinery, quickly provides results on the best strategy
to apply to any given selection of assets.


Development
-----------
To set up a development environment locally you should

```
git clone https://github.com/cvxgrp/cvxportfolio.git
cd cvxportfolio
make env
```
This will replicate our [development environment](https://docs.python.org/3/library/venv.html). From there you can test with

```
make test
```

You activate the shell environment with one of scripts in `env/bin` (or `env\Scripts` on windows), for example if you use bash on POSIX
```
source env/bin/activate
```
and from the environment you can run any of the scripts in the examples (the cvxportfolio package is installed in [editable mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)). 
Or, if you don't want to activate the environment, you can just run scripts directly using `env/bin/python` or `env\Scripts\python` on windows, like we do in the Makefile.


Examples from the book
----------------------
In branch [0.0.X](https://github.com/cvxgrp/cvxportfolio/tree/0.0.X) you can find the original material used to generate plots
and results in the book. As you may see from those
ipython notebooks a lot of the logic that was implemented there, outside of cvxportfolio proper, is being included and made automatic
in newer versions of cvxportfolio. 


Academic
------------

If you use `cvxportfolio` in your academic work please cite our book:
```
@book{BBDKKNS:17,
    author       = {S. Boyd and E. Busseti and S. Diamond and R. Kahn and K. Koh and P. Nystrup and J. Speth},
    title        = {Multi-Period Trading via Convex Optimization},
    series       = {Foundations and Trends in Optimization},
    year         = {2017},
    month        = {August},
    publisher    = {Now Publishers},
    url          = {http://stanford.edu/~boyd/papers/cvx_portfolio.html},
}
```


License
------------

Cvxportfolio is licensed under the [Apache 2.0](http://www.apache.org/licenses/) permissive
open source license.


