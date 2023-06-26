# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)

**WORK IN PROGRESS. Cvxportfolio is currently under development. We will freeze the user interface by end of 2023Q2 and release the first stable version by end of 2023Q3.**


`cvxportfolio` is a python library for portfolio optimization and simulation
based on the book [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)
(also [available in print](https://www.amazon.com/Multi-Period-Trading-Convex-Optimization-Foundations/dp/1680833286/)).

The documentation of the package is at [cvxportfolio.readthedocs.io](https://cvxportfolio.readthedocs.io/en/latest/).


Installation
------------

```
pip install -U cvxportfolio
```

Testing locally
------------
We ship our unit test suite with the pip package. After installing you can test in you local environment by

```
python -m unittest discover cvxportfolio
```


Example
------------
To get a sneak preview of `cvxportfolio` you may try the following code. This is available in `examples/hello_world.py` and runs 
with `cvxportfolio >= 0.3.0`. All objects in `cvxportfolio` can either be provided data (in a variety of forms, but preferably pandas
series or dataframes) or infer/download it. For example in the following example, market data is downloaded by a public source
(Yahoo finance) and the forecasts are computed iteratively, at each point in the backtest, from past data. That is, at each point in the backtest,
the policy object only operates on **past data**, and thus the result you get is a realistic simulation of what the strategy would have performed in the market.
The simulator by default includes holding and transaction costs, using the models described in the book, and default parameters that are typical for the US stock market.
The logic used
matches what is described in Chapter 7 of the book. For example, returns are forecasted as the historical mean returns 
and covariances as historical covariances (both ignoring `np.nan`'s). The logic used is detailed in the `forecast` module. Many optimizations
are applied to make sure the system works well with real data. 


```python
import cvxportfolio as cvx
import matplotlib.pyplot as plt

gamma = 3       # risk aversion parameter (Chapter 4.2)
kappa = 0.05    # covariance forecast error risk parameter (Chapter 4.3)
objective = cvx.ReturnsForecast() - gamma * (
	cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.TransactionCost()
constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

simulator = cvx.MarketSimulator(['AAPL', 'AMZN', 'TSLA', 'GM', 'CVX', 'NKE'])

result = simulator.backtest(policy, start_time='2020-01-01')

print(result)

# plot value of the portfolio in time
result.v.plot(figsize=(12, 5), label='Multi Period Optimization')
plt.ylabel('USD')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
result.w.iloc[:, :-1].plot()
plt.title('Weights of the portfolio in time')
plt.show()
```

Development
-----------
Cvxportfolio is under development and things might change (quite fast), however you are (most) welcome to 
read the code, play with it, and contribute. To set up a development environment locally you should

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
and results in the book. Those are being restored, and (slowly) translated in the new framework. As you may see from those
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


