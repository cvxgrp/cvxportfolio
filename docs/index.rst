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
   
Installation
------------

Cvxportolio is written in Python and can be easily installed in any environment by simple:

.. code-block:: console

	pip install -U cvxportfolio
	

We show how this is done on our `Installation and Hello World <https://youtu.be/1ThOKEu371M>`_ youtube video.
Its main dependencies are `Cvxpy <https://www.cvxpy.org>`_ for interfacing
with numerical solvers and `Pandas <https://pandas.pydata.org/>`_
for interfacing with databases.


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

	
Testing locally
---------------
We ship our unit test suite with the software library, so after installing you can test 
in your local environment with:

.. code-block:: console

	python -m cvxportfolio.tests

We test against recent python versions (3.9, 3.10, 3.11) and recent versions of the main
dependencies (from pandas 1.4, cvxpy 1.1, ..., up to the current versions) on all major 
operating systems. So, Cvxportfolio doesn't require any specific version of
any dependency, and should work in any pre-existing environment.
	
Licensing
---------

Cvxportfolio is licensed under the `Apache 2.0 <http://www.apache.org/licenses/>`_ permissive
open source license.

 
Citing
------------

If you use Cvxportfolio in work that leads to publication, you can cite the following:

.. code-block:: latex

    @misc{busseti2017cvx,
        author    = "Busseti, Enzo and Diamond, Steven and Boyd, Stephen",
        title     = "Cvxportfolio",
        month    = "January",
        year     = "2017",
        note     = "Portfolio Optimization and Back--{T}esting",
        howpublished = {\url{https://github.com/cvxgrp/cvxportfolio}},
    }

    @article{boyd2017multi,
      author  = "Boyd, Stephen and Busseti, Enzo and Diamond, Steven and Kahn, Ron and Nystrup, Peter and Speth, Jan",
      journal = "Foundations and Trends in Optimization",
      title   = "Multi--{P}eriod Trading via Convex Optimization",
      month   = "August",
      year    = "2017",
      number  = "1",
      pages   = "1--76",
      volume  = "3",
      url     = {\url{https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf}},
    }


The latter is also the first chapter of this PhD thesis:

.. code-block:: latex

    @phdthesis{busseti2018portfolio,
        author    = "Busseti, Enzo",
        title     = "Portfolio Management and Optimal Execution via Convex Optimization",
        school    = "Stanford University",
        address   = "Stanford, California, USA",
        month    = "May",
        year     = "2018",
        url     = {\url{https://stacks.stanford.edu/file/druid:wm743bj5020/thesis-augmented.pdf}},
    }


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
   
.. _paper: https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf
.. _FRED: https://fred.stlouisfed.org/
.. _Yahoo Finance: https://finance.yahoo.com