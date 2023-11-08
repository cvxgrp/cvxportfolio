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


Example
-------

Have a look at the :doc:`hello_world` to see Cvxportfolio in action.


Introduction
------------

Cvxportfolio is an object-oriented library for portfolio optimization and backtesting
which focuses on ease of use. It implements the models described 
in the `accompanying book <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_
and can be extended with user-defined objects and methods to accommodate
different data sources, custom cost models (both for simulation and optimization),
constraints, and so on.

The main abstractions used are the :class:`cvxportfolio.MarketSimulator`, which faithfully mimics
the trading activity of a financial market, the collection of 
:doc:`policies <policies>`, which include both simple policies such as
:class:`cvxportfolio.RankAndLongShort`, and the optimization-based policies :class:`cvxportfolio.SinglePeriodOptimization`
and :class:`cvxportfolio.MultiPeriodOptimization`.
For these two, the user specifies the objective function (which is maximized)
and a list of constraints which apply to the optimization. All these types
of objects can be customized in many ways, including by deriving or redefining them.

Then, we provide the :class:`cvxportfolio.data.MarketData` abstraction, which both serves historical
data during a backtest and real-time data in online usage. We implement the interface
to public data sources (`Yahoo finance <https://finance.yahoo.com>`_ 
and `FRED <https://fred.stlouisfed.org/>`_), as well as user-provided data (which
can also be passed to all other objects).

In addition, we provide logic to easily parallelize backtesting of many different policies,
or the same policy with different choices of hyperparameters, and cache on disk both
historical data (for reproducibility) and various expensive calculations, such as
estimates of covariance matrices. 

We present the results of each backtest with a clear interface, :class:`cvxportfolio.BacktestResult`,
which defines various metrics of backtest performance and the logic to both print
and plot them.

	
Testing locally
---------------
We ship our unit test suite with the software package, so after installing you can test 
in your local environment with:

.. code-block:: console

	python -m cvxportfolio.tests

We test against recent python versions (3.9, 3.10, 3.11) and recent versions of the main
dependencies (from pandas 1.4, cvxpy 1.1, ..., up to the current versions) on all major 
operating systems.
	
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
   
   hello_world
   manual
   policies
   simulator
   objective_terms
   constraints
   result
   data
   examples

