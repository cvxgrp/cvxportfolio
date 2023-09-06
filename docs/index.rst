Cvxportfolio Documentation
==========================

**Cvxportfolio** is a Python library for portfolio optimization. It enables users
to quickly try optimization :doc:`policies <policies>` for financial portfolios
by testing their past performance with a sophisticated :doc:`market simulator <simulator>`.


Cvxportfolio is based on the book `Multi-Period Trading via Convex Optimization <https://www.amazon.com/Multi-Period-Trading-Convex-Optimization-Foundations/dp/1680833286>`_
(also `available in PDF <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_).


   
Installation
------------

Cvxportolio is written in pure Python and can be easily installed in your favorite environment by simple:

.. code-block:: console

	pip install -U cvxportfolio
	

We show how this is done on our `Installation and Hello World <https://youtu.be/1ThOKEu371M>`_ youtube video.
	
Testing locally
---------------
We ship our unit test suite with the software package, so after installing you can test with:

.. code-block:: console

	python -m unittest discover cvxportfolio
	

	
Example
-------

The following example, `available in the repository <https://github.com/cvxgrp/cvxportfolio/blob/master/examples/hello_world.py>`_,
shows how to define an optimization policy, initialize the market simulator (which downloads and stores stock market data behind the scenes),
run a backtest, and show its result. 

.. code-block:: python

	import cvxportfolio as cvx

	objective = cvx.ReturnsForecast() - 3 * (cvx.FullCovariance() + \
		0.05 * cvx.RiskForecastError()) - cvx.StocksTransactionCost()
	constraints = [cvx.LeverageLimit(3)]

	policy = cvx.MultiPeriodOptimization(
		objective, constraints, planning_horizon=2)

	simulator = cvx.StockMarketSimulator(
		['AAPL', 'AMZN', 'TSLA', 'GM', 'CVX', 'NKE'])

	result = simulator.backtest(policy, start_time='2020-01-01')

	# print backtest result statistics
	print(result)
	
	# plot backtest result
	result.plot()
	
	
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
   constraints
   result

