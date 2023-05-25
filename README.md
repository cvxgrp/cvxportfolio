# Cvxportfolio

[![CVXportfolio on PyPI](https://img.shields.io/pypi/v/cvxportfolio.svg)](https://pypi.org/project/cvxportfolio/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxportfolio?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxportfolio)
[![Documentation Status](https://readthedocs.org/projects/cvxportfolio/badge/?version=latest)](https://cvxportfolio.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxportfolio/badge.svg?branch=master)](https://coveralls.io/github/cvxgrp/cvxportfolio?branch=master)


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
(Yahoo finance) and the forecasts are computed iteratively, at each point in the backtest, from past data. The logic used
matches what is described in Chapter 7 of the book. For example, returns are forecasted as the historical mean returns 
and covariances as historical covariances (both ignoring `np.nan`'s). The logic used is detailed in the `forecast` module. Many optimizations
are applied to make sure the system works well with real data. 


```python
import cvxportfolio as cvx
import matplotlib.pyplot as plt

gamma = 3  	    # risk aversion parameter (Chapter 4.2)
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

print('\ntotal tcost ($)', result.tcost.sum())
print('total borrow cost ($)', result.hcost_stocks.sum())
print('total cash return + cost ($)', result.hcost_cash.sum())

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
(Or, if you don't want to activate the environment, you can just run scripts directly using `env/bin/python` or `env\Scripts\python` on windows, like we do in the Makefile.)



Roadmap
-------
We plan to release the first stable version of cvxportfolio by the end of Summer 2023. Many new features are going to be added
and extensive testing will be performed by then. Here's a rough list of what we think `cvxportfolio 1.0` will implement:

- Automated hyperparameter search and tuning. Hyperparameters will be defined as cvxportfolio objects, with optional bounds and spacing,
	so that the simulator can iterate automatically through HPs combinations. Optionally the user can define a metric to be optimized
	(for example among the ones provided by BacktestResult) and let cvxportfolio use heuristics to get an (at least, local) optimal
	HP combination. We'll make sure that the optimization routine can be subclassed and substituted with custom ones easily, as 
	is the case with all other cvxportfolio objects. The same can also be done in a Pareto optimization fashion, so for example the user can 
	request a list of HP combinations that are Pareto optimal (in backtest) for excess return and risk.
- Online and offline caching. All expensive computations (including database accesses) are performed by cvxportfolio in a lazy fashion.
	For example risk model estimations are done online, during a backtest, with a view of the past market data that is provided by the market
	simulator. We will make sure that all expensive computations are done only once. Policy objects will provide an online cache 
	for their estimators (e.g., risk models) to share. For example, MPO policies have separate estimator objects for each MPO step so they benefit
	from this. Then, at the end of a backtest, the same data will be stored offline, in the `~/cvxportfolio_data/` folder, along with market
	data and the backtest result itself. This all is done safely with respect to parallel execution (we use Python native multiprocessing module
	for (potentially massive) parallelism). If the user asks for the same backtest twice, the saved result will be returned. If the user ask
	for a backtest performed yesterday, today, the backtest will only be updated with the new market data available (i.e., one day). 
- Documentation, auditability, code readability, ease of subclassing. With fast development come API breaks and lagging documentation. That's
	why we have fixed a date for the stable release and we mean to stick with it. Especially for a piece of software that aims to automate
	many functions of investment management we want everything to be clearly documented, but mostly readable and auditable. 
	We'll use Python logging functionalities to record every action performed in the simulation that is not documented in the book (for example,
	cancel a trade order because the market data for the day has null volumes, or add/remove one symbol during a backtest because their market
	data starts or ceases to exist) and those will be saved with the backtest result itself. The total number of lines of code will not exceed
	significantly the current value (we'll maybe even reduce it). The object model used now (estimators) will probably remain, but with added
	methods for caching, and be clearly documented for subclassing (so that custom classes can as well use all cvxportfolio functionalities). 


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


