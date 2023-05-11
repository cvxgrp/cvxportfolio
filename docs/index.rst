Cvxportfolio Documentation
==========================

**Cvxportfolio** is a Python library for portfolio optimization. It enables users
to quickly try optimization :doc:`policies <policies>` for financial portfolios
by testing their past performance with a sophisticated :doc:`market simulator <simulator>`.


Cvxportfolio is based on the book `Multi-Period Trading via Convex Optimization <https://www.amazon.com/Multi-Period-Trading-Convex-Optimization-Foundations/dp/1680833286>`_
(also `available in PDF <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_).
   
.. note::

   Cvxportfolio is under active development. We target the end of 2023 Q3 for 
   the first stable release.
   
   
Installation
------------

Cvxportolio is written in pure Python and depends on popular libraries such as
`Cvxpy <https://cvxpy.org/>`_, 
`Pandas <https://pandas.pydata.org/>`_, 
and `Yfinance <https://github.com/ranaroussi/yfinance>`_.
As such, it can be easily installed in your favorite environment by simple:

.. code-block:: console

	pip install -U cvxportfolio
	
	
Example
-------

The following example, `available in the repository <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/hello_world.py>`_,
shows how to define an optimization policy, initialize the market simulator (which downloads and stores stock market data behind the scenes),
run a backtest, and show its result. 

.. code-block:: python

	import cvxportfolio as cvx
	import matplotlib.pyplot as plt

	objective = cvx.ReturnsForecast() - 5 * (cvx.FullCovariance() + 0.1 * cvx.RiskForecastError()) - cvx.TransactionCost()
	constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]

	policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=2)

	simulator = cvx.MarketSimulator(['AAPL', 'AMZN', 'MSFT', 'TSLA'])

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
	
	
Licensing
---------

Cvxportfolio is licensed under the `Apache 2.0 <http://www.apache.org/licenses/>`_ permissive
open source license.

 
Academic
--------

If you use cvxportfolio for academic work you can cite the book it is based on:

.. code-block:: latex

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

   
Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   policies
   simulator
   returns
   costs
   risks
   constraints
   result
   data
   

   
   