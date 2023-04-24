# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)


**WORK IN PROGRESS. Cvxportfolio is currently under development. We will freeze the user interface by end of 2023Q2 and release the first stable version by end of 2023Q3. The script `hello_world.py` now runs with the new interface (see below).**


`cvxportfolio` is a python library for portfolio optimization and simulation
based on the book [Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).
It is written in Python, its main dependencies are [`cvxpy`](https://github.com/cvxgrp/cvxpy)
and [`pandas`](https://github.com/pandas-dev/pandas). 

The documentation of the package is at [cvxportfolio.readthedocs.io](https://cvxportfolio.readthedocs.io/en/latest/).


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


Example
------------
To get a sneak preview of `cvxportfolio` you may try the following code. This is available in `examples/hello_world.py` and runs 
with `cvxportfolio` >= 0.2.0


```python
import cvxportfolio as cp
import matplotlib.pyplot as plt

# define a portfolio optimization policy
# with rolling window mean (~10 yrs) returns
# with forecast error risk on returns (see the book)
# rolling window mean (~10 yrs) covariance
# and forecast error risk on covariance (see the book)
policy = cp.SinglePeriodOptimization(objective = 
        cp.RollingWindowReturnsForecast(2500) -
        cp.RollingWindowReturnsForecastErrorRisk(2500) -
        5 * cp.RollingWindowFullCovariance(2500, forecast_error_kappa = 0.25), 
        constraints = [cp.LeverageLimit(3)]
        )
        
# define a market simulator, which downloads stock market data and stores it locally
# in ~/cvxportfolio/        
simulator = cp.MarketSimulator(["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM"])

# perform a backtest (by default it starts with 1E6 USD cash)
backtest = cp.BackTest(policy, simulator, '2023-01-01', '2023-04-21')

# plot value of the portfolio in time
backtest.v.plot(figsize=(12, 5), label='Single Period Optimization')
plt.ylabel('USD')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
backtest.w.iloc[:, :-1].plot()
plt.title('Weights of the portfolio in time')
plt.show()

print('total tcost', backtest.tcost.sum())
print('total borrow cost', backtest.hcost_stocks.sum())
print('total cash return + cost', backtest.hcost_cash.sum())

```

(*The other examples may currently have problems as we are changing various bits and pieces of `cvxportfolio`.*)


Academic
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


