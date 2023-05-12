# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
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
with `cvxportfolio >= 0.3.0`


```python
import cvxportfolio as cvx
import matplotlib.pyplot as plt

objective = cvx.ReturnsForecast() - 3 * (cvx.FullCovariance() + \
	0.05 * cvx.RiskForecastError()) - cvx.TransactionCost()
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

print('\ntotal tcost ($)', result.tcost.sum())
print('total borrow cost ($)', result.hcost_stocks.sum())
print('total cash return + cost ($)', result.hcost_cash.sum())

```

Examples from the book
--------------------
In branch [0.0.X](https://github.com/cvxgrp/cvxportfolio/tree/0.0.X) you can find the original material used to generate plots
and results in the book. 


Academic
------------

If you use `cvxportfolio` in your academic work please cite our book:
```
@book{BBDKKNS:17,
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


