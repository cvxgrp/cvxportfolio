.. Copyright (C) 2023-2024 Enzo Busseti
.. Copyright (C) 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.

.. This file is part of Cvxportfolio.

.. Cvxportfolio is free software: you can redistribute it and/or modify it under
.. the terms of the GNU General Public License as published by the Free Software
.. Foundation, either version 3 of the License, or (at your option) any later
.. version.

.. Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
.. ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
.. FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
.. details.

.. You should have received a copy of the GNU General Public License along with
.. Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.

Cvxportfolio Documentation
==========================

Cvxportfolio is a Python library for portfolio optimization. It enables users
to quickly try optimization :doc:`policies <policies>` for asset management
by back-testing their past performance with a sophisticated
:doc:`market simulator <simulator>`.


Most models implemented by Cvxportfolio, including the accounting methods,
naming conventions, and assumptions, are described
in the accompanying `paper`_.
This was written as a collaborative work by Stanford University researchers and
BlackRock Inc. investment professionals.


.. include:: ../README.rst    
    :start-after: .. Installation
    :end-before: .. Simple Example


Introduction
------------

Cvxportfolio is an object-oriented library for portfolio optimization and
back-testing which focuses on ease of use. It implements the models described
in the accompanying `paper`_.
and can be extended with user-defined objects and methods to accommodate
different data sources, custom cost models (both for simulation and
optimization), constraints, and so on.

The main abstractions used are the :class:`cvxportfolio.MarketSimulator`, which
faithfully mimics the trading activity of a financial market, the collection of
:doc:`policies <policies>`, which include both simple policies such as
:class:`cvxportfolio.RankAndLongShort`, and the optimization-based policies
:class:`cvxportfolio.SinglePeriodOptimization`
and :class:`cvxportfolio.MultiPeriodOptimization`.
For these two, the user specifies the objective function (which is maximized)
and a list of constraints which apply to the optimization. All these types
of objects can be customized in many ways, including by deriving or redefining
them.

Then, we provide the :class:`cvxportfolio.data.MarketData` abstraction, which
both serves historical data during a back-test and real time data in online
usage. We implement the interface to public data sources (`Yahoo Finance`_
and `FRED`_), as well as user-provided data, which
can also be passed to all other objects, see :ref:`the manual section on
passing data <passing-data>`.

In addition, we provide logic to easily parallelize back-testing of many
different policies, or the same policy with different choices of
hyperparameters, and cache on disk both historical data (for reproducibility)
and various expensive calculations, such as
estimates of covariance matrices. 

We present the results of each back-test with a clear interface,
:class:`cvxportfolio.result.BacktestResult`, which defines various metrics of
backtest performance and the logic to both print and plot them.
	

Where to go next
----------------

You can see the :doc:`selection of examples <examples>` in this documentation
website (even more are available in the code repository).

Then, you can have a look at the :doc:`manual page <manual>` which explains
some of the more important aspects to understand when using Cvxportfolio.

Or, you can look directly at the documentation of each object the user
interacts with, like the :doc:`market simulator <simulator>`, the collection of
both :doc:`simple <simple_policies>` and :doc:`optimization-based policies
<optimization_policies>`, the objective terms (:doc:`return <returns>`,
:doc:`risk <risks>` and :doc:`cost <costs>` models, which all have their
specifities) or :doc:`constraints <constraints>` which apply to
optimization-based policies, and so on.

.. include:: ../README.rst    
    :start-after: .. Versions


Table of Contents
-----------------

.. toctree::
   :maxdepth: 4
   
   manual
   api
   examples
   contributing
   
.. _paper: _static/cvx_portfolio.pdf
.. _FRED: https://fred.stlouisfed.org/
.. _Yahoo Finance: https://finance.yahoo.com
