Cvxportfolio Documentation
==========================

Cvxportfolio is a Python library for portfolio optimization. It enables users
to quickly try optimization :doc:`policies <policies>` for asset management
by back-testing their past performance with a sophisticated :doc:`market simulator <simulator>`.


Most models implemented by Cvxportfolio, including the accounting methods,
naming conventions, and assumptions, are described
in the accompanying `paper`_.
This was written as a collaborative work by Stanford University researchers and
BlackRock Inc. investment professionals.


.. include:: ../README.rst    
    :start-after: .. Installation
    :end-before: .. Simple Example

Hello World Example
-------------------

We show in :doc:`hello_world` a minimal example.


Introduction
------------

Cvxportfolio is an object-oriented library for portfolio optimization and back-testing
which focuses on ease of use. It implements the models described 
in the accompanying `paper`_.
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
data during a back-test and real time data in online usage. We implement the interface
to public data sources (`Yahoo Finance`_ 
and `FRED`_), as well as user-provided data, which
can also be passed to all other objects (see :ref:`passing-data`).

In addition, we provide logic to easily parallelize back-testing of many different policies,
or the same policy with different choices of hyperparameters, and cache on disk both
historical data (for reproducibility) and various expensive calculations, such as
estimates of covariance matrices. 

We present the results of each back-test with a clear interface, :class:`cvxportfolio.BacktestResult`,
which defines various metrics of backtest performance and the logic to both print
and plot them.
	
.. include:: ../README.rst    
    :start-after: .. Versions

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   
   hello_world
   quickstart
   manual
   policies
   simulator
   objective_terms
   constraints
   result
   data
   internals
   examples
   contributing
   
.. _paper: https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf
.. _FRED: https://fred.stlouisfed.org/
.. _Yahoo Finance: https://finance.yahoo.com