.. Copyright (C) 2023-2024 Enzo Busseti

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

Manual
======

.. py:module:: cvxportfolio
    :no-index:

Here we explain some concepts that apply throughout the library and are useful
to know. The code blocks in this document assume the following imports

.. code-block:: python

    import cvxportfolio as cvx
    import numpy as np
    import pandas as pd

.. note::

    This document is a work in progress. We are prioritizing topics which are
    more important to understand when using Cvxportfolio.


Quickstart
----------

Cvxportfolio is designed for ease of use and extension. With it, you can
quickly experiment with financial trading strategies and develop new ones.

For example, a classic Markowitz optimization strategy 

.. math::

    \begin{array}{ll}
         \text{maximize} & w^T \mathbf{E}[r] - \frac{1}{2} w^T \mathbf{Var}[r] w \\
         \text{subject to} & w \geq 0, w^T \mathbf{1} <= 1
    \end{array}

where the expected returns and covariances are the simple sample ones, is
specified as follows

.. code-block:: python
    
    objective = cvx.ReturnsForecast() - 0.5 * cvx.FullCovariance()
    constraints = [cvx.LongOnly(), cvx.LeverageLimit(1)]
    
    strategy = cvx.SinglePeriodOptimization(objective, constraints)
    
Here we instantiated an :ref:`optimization-based trading policy <optimization-policies-page>`.
Cvxportfolio defines two (the other is a simple extension of this). 
Optimization-based policies are defined by an objective function, which is 
maximized, and a list of constraints, that are imposed on the solution.
The objective function is specified as a linear combination of simple
terms, :doc:`we provide many <objective_terms>`, and it's easy to define new ones.
We provide as well :doc:`many constraints <constraints>` to choose from, and
it's even easier to define new ones.

Where are the assets' names?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cvxportfolio policies are symbolic, they only define the logic of a trading 
strategy. They work with any selection of assets. 
The two objective terms defined above, however, need to know the values of
the expected returns and the covariance matrix. 
You can :ref:`pass Pandas dataframes <passing-data>` for those, and in that
case your dataframes should contain the assets' names you want to trade. 
Or, if you want, or you can rely on :doc:`forecasters <forecasts>` to compute those
(iteratively, in back-testing) using past data. That is what happens in the 
code shown above, the default parameters of :class:`ReturnsForecast`
and :class:`FullCovariance` are forecasters that compute historical
means and historical covariances respectively, at each point in time if running in a back-test
or once, if running live, by using the policy's :meth:`execute` method.

.. _passing-data:

Passing Data
------------

Most Cvxportfolio objects, such as :doc:`policies <policies>`, 
:doc:`constraints <constraints>`, and :doc:`objective terms <objective_terms>`,
accept user-provided data. 
These data can either have values that are constant in time, varying in 
time, constant for all assets, or specific for each asset.
They are specified as follows.

- **Python scalars**. These represents values that are constant for all assets
  (if the object requires a value per each asset) and for all times. 
  For example

  .. code-block:: python

      cvx.LeverageLimit(3)
      
  is a constraint that requires the leverage of the portfolio to be less or 
  equal than three, at all times. Or,
  
  .. code-block:: python
      
      cvx.HoldingCost(short_fees=5.25)
      
  is a cost object that models 5.25% annual fees on short positions, for all
  assets and at all times.

- **Pandas series**. These represent either values that are constant in time 
  and vary for each asset, or values that vary in time and are constant for all
  assets. For example
  
  .. code-block:: python
      
      my_forecast = pd.Series([0.001, 0.0005], index=['AAPL', 'GOOG'])
      cvx.ReturnsForecast(r_hat=my_forecast)
  
  is an objective term that models market returns forecasts of 0.1% and 0.05%
  for the two stocks that are specified, over the trading period used. 
  The forecasts are constant in time.
  
  .. note:: 
  
      During a back-test the trading universe may change. Cvxportfolio objects
      are aware of the current trading universe at each point of a back-test.
      If you pass data that vary for each asset, Cvxportfolio objects will try 
      to slice it using the current universe. If they fail, they throw an
      error. So, you should always provide data for all assets that ever appear
      in a back-test.
      
  If instead the pandas series has datetime index it is assumed to contain 
  values that are varying in time. For example
  
  .. code-block:: python
      
      datetime_index_2020 = pd.date_range('2020-01-01', '2020-12-31')
      short_fees_2020 = pd.Series(5.0, index=datetime_index_2020)
      
      datetime_index_2021 = pd.date_range('2021-01-01', '2021-12-31')
      short_fees_2021 = pd.Series(5.25, index=datetime_index_2021)
      
      historical_short_fees = pd.concat([short_fees_2020, short_fees_2021])
      
      cvx.HoldingCost(short_fees=historical_short_fees)
      
  is a cost object that models annual fees on short positions, for all assets, 
  of 5% in 2020 and 5.25% in 2021.
  
  .. note:: 
  
      You should be careful and make sure that the timestamps used match the 
      timestamps used by the market data server: for example they must have the
      same timezone. To find the correct timestamps you can call the 
      :meth:`trading_calendar` method of a market data object.
   
- **Pandas dataframes**. The same conventions used for Pandas series apply, so 
  you should read the above, including the two notes. With dataframes you can
  specify data that varies both for each asset and in time, or 
  multi-dimensional data that varies in time, or for each assets, or both. If
  you provide data that varies in time, the datetime index should always be the
  index (not the columns) and in case of a multi-index it should be the first 
  level. For example
  
  .. code-block:: python
      
      my_forecast = pd.DataFrame(
          [[0.1, 0.05], [0.15, 0.06]],
          index=[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')],
          columns=['AAPL', 'GOOG'])
      cvx.ReturnsForecast(r_hat=my_forecast)
      
  is an objective term that models returns forecasts of 10% and 5% for the 
  first period, and 15% and 6% for the second period, for the two assets
  specified. In this case the two periods are one year each (you get that, for 
  example, by setting the ``trading_frequency`` attribute of a market 
  data server as ``'annual'``). Remember again that the timestamps must match
  those provided by the :meth:`trading_calendar` method of the market data 
  server used.
  
  Multi-dimensional data constant in time is modeled as follows
  
  .. code-block:: python
      
      exposures = pd.DataFrame(
          [[1, -.5], [-.25, .75]],
          index=['AAPL', 'GOOG'],
          columns=['factor_1', 'factor_2'])
      cvx.FactorNeutral(factor_exposure=exposures)
  
  so the resulting constraint requires neutrality of the portfolio with respect
  to those two factors. The index must contain all assets that appear in a 
  back-test, and it will be sliced if at some point in time of a back-test
  only a subset of those assets is traded (see the note above).
  
  Multi-dimensional data can also vary in time. It is modeled as a Pandas
  multi-indexed dataframe. If the data is time-varying, the first level
  of the multi-index should be a Pandas datetime index.
  
  .. code-block:: python
      
      multi_index = pd.MultiIndex.from_product(
          [[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')],
          ['AAPL', 'GOOG']])
      
      exposures = pd.DataFrame(
          [[1, -.5], [-.25, .75], [.9, -.3], [-.1, .9]],
          index=multi_index,
          columns=['factor_1', 'factor_2'])
      cvx.FactorNeutral(factor_exposure=exposures)
  
  All the conventions above apply (timestamps should match the ones provided by
  the :meth:`trading_calendar` method of the market data server, assets' names
  should include all the ones that are traded, ...).
  
  Another example are factor covariances that appear in low-rank factor model
  covariances. These are specified as follows
  
  .. code-block:: python
      
      multi_index = pd.MultiIndex.from_product(
          [[pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')],
          ['factor_1', 'factor_2']])
      
      factor_covariances = pd.DataFrame(
           # factor covariance at '2020-01-01'
          [[1, 0.25], 
           [0.25, 1], 
           # factor covariance at '2021-01-01'
           [1, .1], 
           [.1, 1]],
          index=multi_index,
          columns=['factor_1', 'factor_2'])
         
      cvx.FactorModelCovariance(
          F=exposures, Sigma_F=factor_covariances, d=0.01)

- **Numpy arrays**. These are not recommended but can be used in simple cases.
  One use-case is to model data that is constant in time and vary for the assets. 
  If the trading universe varies through a back-test these can't be used, an error 
  is thrown whenever the sizes of the trading universe and
  of the array don't match. For example
  
  .. code-block:: python
      
      my_forecast = np.array([0.001, 0.0005])
      cvx.ReturnsForecast(r_hat=my_forecast)
      
  models returns' forecasts of 0.1% for the first asset in the universe and
  0.05% for the second asset in the universe. The ordering is the one of the
  data provided by the :meth:`serve` method of the market data server.
  
  Another usecase, less problematic, is to model data that varies across other
  dimensions, such as risk factors. For example a constant factor covariance
  can be provided as follows

  .. code-block:: python
     
      factor_covariance = np.array( # constant in time
          [[1, 0.25], 
           [0.25, 1]])

      cvx.FactorModelCovariance(
          F=exposures, Sigma_F=factor_covariance, d=0.01)
         
  the ordering used here is that of the columns of the provided
  ``exposures`` dataframe.
  
Missing values
~~~~~~~~~~~~~~
When Cvxportfolio objects access user-provided data, after they locate the right
time and slice with the current trading universe (if applicable), they check
that the resulting data does not contain any ``np.nan`` missing value. If any is
found, they throw an error. Thus, you should make sure that no ``np.nan`` values
are contained in any data passed that will be accessed. It is fine to have 
``np.nan`` values for assets that are not traded at a certain time (for example,
because they didn't exist) because that data won't be accessed.
  
Cash account
~~~~~~~~~~~~
Many Cvxportfolio internal variables, such as the weights and holdings vectors
that you can access in a :class:`result.BacktestResult` object, include
the cash account as their last element. In most cases used-provided data is not
concerned with the cash account (such as all examples above) and so it can be
ignored. Exceptions are noted in the documentation of each object.


.. _multi-period:

Multi-Period Optimization
-------------------------
Multi-period optimization is the signature portfolio allocation model
defined by Cvxportfolio (although it is not the only one). It is discussed
in :paper:`section 5 of the paper <chapter.5>`, and it is based on
model-predictive control, the industrial engineering standard for dynamic
control. The relevant Cvxportfolio object we deal with here is the 
:class:`MultiPeriodOptimization` policy.

We note that the simpler single-period optimization model, implemented by 
:class:`SinglePeriodOptimization` and defined in
:paper:`section 4 of the paper <chapter.4>`, is in fact a special
case of multi-period optimization with the planning horizon equal to 1. So,
understanding the content of this section also helps you use the single-period
optimization model more effectively (and, in fact, there are various situations
in which single-period is preferable).

A general formulation of the model is given by the following optimization
problem, which is solved at time :math:`t`

.. math::

    \begin{array}{ll}
         \text{maximize} & \sum_{\tau = t}^{t + H - 1}
           \left(
            \hat {r}_{\tau | t}^T w_\tau^T 
            - \gamma^\text{risk} \psi_\tau(w_\tau)
            - \gamma^\text{hold} \hat{\phi}^\text{hold}_{\tau|t}(w_\tau)
            - \gamma^\text{trade} \hat{\phi}^\text{trade}_{\tau|t}(w_\tau - w_{\tau - 1})
            \right) \\
         \text{subject to} & 
            \mathbf{1}^T w  = 1, \ 
            w_\tau - w_{\tau -1} \in \mathcal{Z}_\tau, \ 
            w_\tau \in \mathcal{W}_\tau, \ 
            \text{for } \tau = t+1, \ldots, t+H 
    \end{array}

This is copied from the equations in the paper, and uses all the definitions
made there. We summarize them here for clarity:

- The time index :math:`\tau` spans the planning horizon :math:`H \geq 1`,
  and if :math:`H = 1` we have single-period optimization. The policy plans
  for allocations over a few time-steps in the future, and only the first
  time-step, :math:`\tau = t`, is used. (This is, in a nutshell, 
  model-predictive control.) The time indexes used here are natural numbers.
- The weight vector :math:`w_\tau` is the allocation of wealth among assets, at
  each step in the planning horizon. Its last element is used by the cash
  account, which is always there. The vector always sums to one, because of the
  :paper:`self-financing condition defined in the paper <section.2.5>`.
- The vector :math:`\hat {r}_{\tau | t}` are the forecasts of the returns, for all
  assets (and cash), made at (execution) time :math:`t` for (prediction) time :math:`\tau`.
  The return :math:`r_t` for a single asset is simply the ratio
  :math:`\frac{p_{t+1} - p_t}{p_t}` between consecutive prices. (Note that often returns by data
  vendors are defined differently, shifted by one.)
- The :math:`\gamma`'s are hyper-parameters, as discussed in :paper:`section 4.8 <section.4.8>`
  of the paper. These are positive numbers.
- The objective term :math:`\psi_{\tau|t}` is a risk model, like a factor model covariance.
  We can have different risk models at different planning steps.
- The terms :math:`\hat{\phi}^\text{hold}_{\tau|t}` and :math:`\hat{\phi}^\text{trade}_{\tau|t}`
  model the forecasts, made at time :math:`t`, for the holding and trading
  costs respectively that will be incurred at time :math:`\tau`.
- The sets :math:`\mathcal{Z}_\tau` and :math:`\mathcal{W}_\tau` represent the
  allowed space of trade allocation vectors at each planning step, and are
  defined by a selection of imposed constraints.

  A close translation of the above equation in Cvxportfolio code looks like this.
  Here we use :math:`H = 2`.

  .. code-block:: python

    same_period_returns_forecast = pd.DataFrame(...)
    next_period_returns_forecast = pd.DataFrame(...) # indexed by the time of execution, not of the forecast!

    gamma_risk = cvx.Gamma(initial_value = 0.5)
    gamma_hold = cvx.Gamma(initial_value = 1.0)
    gamma_trade = cvx.Gamma(initial_value = 1.0)

    objective_1 = cvx.ReturnsForecast(r_hat = same_period_returns_forecast) \
        - gamma_risk * cvx.FullCovariance() \
        - gamma_hold * cvx.HoldingCost(short_fees = 1.) \ 
        - gamma_trade * cvx.TransactionCost(a = 2E-4)

    objective_2 = cvx.ReturnsForecast(r_hat = next_period_returns_forecast) \
        - gamma_risk * cvx.FullCovariance() \
        - gamma_hold * cvx.HoldingCost(short_fees = 1.) \ 
        - gamma_trade * cvx.TransactionCost(a = 2E-4)

    constraints_1 = [cvx.LongOnly(applies_to_cash = True)]
    constraints_2 = [cvx.LongOnly(applies_to_cash = True)]

    policy = cvx.MultiPeriodOptimization(
        objective = [objective_1, objective_2],
        constraints = [constraints_1, constraints_2]
    )

Here we use a mixture of data provided by the user (for the returns' forecast
and the cost) and data estimated internally by :doc:`forecasters <forecasts>` (for the risk model).

One thing to note: If providing time-indexed data like returns forecast,
the time index always refers to the time of execution, not the time of the forecast.
When the policy is evaluated at time :math:`t`, every provided dataframe (if applicable) will be
searched for entries with index :math:`t`, regardless of which step in the planning
horizon it is at.

We note also that there's a simpler way to initialize :class:`cvxportfolio.MultiPeriodOptimization`,
by providing a single objective and a single list of constraints, and specifying
the ``planning_horizon`` argument to, for example, 2. This simply copies the terms
for each step of planning horizon.

.. _disk-access:

Disk Access
-----------

If you only use Cvxportfolio with user-provided data, meaning via the
:class:`UserProvidedMarketData` server, and are careful to specify the
``cash_key`` attribute so that the risk-free rates are not downloaded from the
FED website, Cvxportfolio will not access on-disk storage.

Objects in the :doc:`data` submodule, to be precise the :class:`data.SymbolData`
classes, may try to access on-disk storage.
These are objects that download single-symbol historical data from the Internet
and store the time series locally.
On subsequent runs, only the most recent data gets updated, and older
observations are kept to the values stored beforehand.
That is important for reproducibility.
The storage format can be controlled with the ``storage_backend`` option to
:class:`data.SymbolData` classes, which is also exposed by the
:class:`DownloadedMarketData` constructor.
In addition to the default ``'pickle'``, we provide a ``'csv'`` backend which
is very useful when manual inspection of the data is needed, and a ``'sqlite'``
one which doubles as a prototype for SQL backends.
In all cases the data is stored in a single folder, which is specified by the
``base_location`` argument to the :doc:`data interfaces <data>` objects, also
taken by the :doc:`market simulator <simulator>` objects. That is, by default,
a folder called ``'cvxportfolio_data'`` in your user home directory.
It contains subfolders named after each of the single symbol data interfaces,
which in turn contain the relevant files, with intuitive names.
You can delete that folder, or any of its subfolders, if you wish.
That will simply make Cvxportfolio objects re-download each series from the
start.

On-disk storage by the classes that download historical data from the Internet
can not be disabled, but if you wish you can simply remove the folder after
each Cvxportfolio run.

In addition, internal Cvxportfolio interfaces are used
to store back-test cache files, in the subdirectory with the same name of
the ``'cvxportfolio_data'`` folder, or whichever you specify as
``base_location`` to the :doc:`market simulator constructor <simulator>`. You
can see in the :doc:`back-test timing example <examples/timing>` how that is indeed very
useful in speeding up the work-flow of re-running a similar back-test multiple
times.

Back-tests executed with the :class:`UserProvidedMarketData` server do not use
on-disk caching (because that class does not define the
:meth:`data.MarketData.partial_universe_signature` method).

Lastly, some examples and the test suite use temporary file-system storage,
via the `temporary directory <https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory>`_
object from Python's Standard Library. That is otherwise not used by Cvxportfolio.


Network Access
--------------

The same objects in the :doc:`data interfaces <data>` submodule discussed in the previous section,
*i.e.,* the :class:`YahooFinance` and :class:`Fred` single-symbol data
interfaces, are the only two objects that access the Internet in the
Cvxportfolio library code.
They make standard HTTPS GET calls through
a HTTP client, `curl-cffi <https://curl-cffi.readthedocs.io/>`_ module,
which is one of our dependencies.

There are other internet calls in the examples, for example the
:doc:`script that downloads stock indexes components <examples/universes>`,
but the examples are not part of the Cvxportfolio library (they are not
included in the pip packages).

If you wish to use Cvxportfolio in an environment without Internet access, or
are otherwise concerned about accessing the Internet, you simply need to use
Cvxportfolio via the :class:`UserProvidedMarketData` object (and not
:class:`DownloadedMarketData`), being careful to specify the ``cash_key``
attribute so that the risk-free rates are not downloaded by :class:`Fred`.

The test-suite makes a limited number of calls to the two classes that require
internet access. If you run the test suite (by ``python -m cvxportfolio.tests``)
on a computer without internet access you should see a few tests, mostly
in the ``test_data.py`` module, failing, but most of the test suite will run.

Parallel back-testing
---------------------

You can run multiple back-tests in parallel with the :meth:`MarketSimulator.backtest_many`
method. It takes a list of policies and returns the corresponding list of
:class:`cvxportfolio.result.BacktestResult`. Also
:meth:`MarketSimulator.optimize_hyperparameters` uses the same approach,
to search over the space of hyper-parameters efficiently.

.. note::

    It is not recommended to run multiple Cvxportfolio programs at the same time,
    unless you are careful to not :ref:`access on-disk storage <disk-access>`. If
    you want to run many back-tests at the same time, you should run a single program
    with :meth:`MarketSimulator.backtest_many`.

.. note::

    If your Cvxportfolio program uses custom objects, for example
    :doc:`a forecaster <examples/user_provided_forecasters>`,
    and in that you call complex third party libraries, like machine-learning
    ones, parallel back-testing can be problematic. You should in those cases
    make sure to :ref:`initialize and finalize <execution-model>` all resources you use.
    Alternatively, Cvxportfolio supports the ``multiprocess`` `parallel execution
    library <https://multiprocess.readthedocs.io/en/latest/>`_, which may help in such
    cases. Simply install ``multiprocess`` in the Python environment to make
    Cvxportfolio use it.


CVXPY
-----

`CVXPY <https://cvxpy.org>`_ is an object-oriented Python library that offers
a simple user interface for the specification of optimization programs,
translates them into the formats required by high-performance
optimization solvers, runs a solver chosen by the user (or, heuristically,
by CVXPY itself, if the user doesn't require a specific one), and returns the
solution in the original format specified by the user. This is done
automatically, so the intricacies of numerical optimization are not exposed
to the end user, and the solver can be easily replaced (each solver typically
requires a different specification format, but those are handled by CVXPY).
CVXPY was born not very long before Cvxportfolio, and is now a very successful
library with lots of users and contributors, in both the applied and
theoretical optimization communities.

In the days before such high-level libraries were available, users typically
had to code their application programs against the APIs offered directly by the
solvers, resulting in complex, difficult to debug, and un-maintainable codes.
Today, the maintainers of the numerical solvers are themselves involved in
developing and maintaining the interfaces from CVXPY to their solvers, ensuring
best compatibility. Cvxportfolio users need not be expert or even familiar with
CVXPY, since Cvxportfolio offers an even higher level interface, automating the
definition and management of optimization objects like variables and
constraints.

One area however in which awareness of the underlying CVXPY process might be
useful is the choice and configuration of the numerical solver. Cvxportfolio
exposes the `relevant CVXPY API
<https://www.cvxpy.org/tutorial/solvers/index.html#solve>`_
through the constructor of the optimization-based policies. For example:

.. code-block:: python

    policy = cvx.SinglePeriodOptimization(
        objective = cvx.ReturnsForecast(),
        constraints = [
            cvx.FullCovariance() <= target_daily_vol**2,
            cvx.LongOnly(),
            cvx.LeverageLimit(1),
        ]
        # the following **kwargs are passed to cvxpy.Problem.solve
        solver='SCS',
        eps=1e-14,
        verbose=True,
    )

This policy object is instructed to solve its optimization program
with the CVXPY-interfaced solver called ``'SCS'``, which is
`a modern first-order conic solver <https://cvxgrp.org/scs>`_,
it also requires a target accuracy at convergence of :math:`10^{-14}`,
and will print verbose output from both CVXPY and SCS.

The `full list of solvers available, along with their options
<https://www.cvxpy.org/tutorial/solvers/index.html>`_, can be found on CVXPY's
documentation website, and it's always growing. Other options
to the `solve method <https://www.cvxpy.org/tutorial/solvers/index.html#solve>`_
can also be useful, like the ``ignore_dpp=True`` which is sometimes used in
the examples; that disables a form of matrix caching done by CVXPY which can
slow down execution in certain instances.

Familiarity with CVXPY syntax (and peculiarities!) is needed only if you wish
to extend Cvxportfolio with user-defined optimization terms, like custom
costs and constraints. Cvxportfolio is designed to make that process as simple
as possible, while maintaning a unified system that provides accurate
accounting, manages market and derived data, enables multi-processing for
parallel back-tests, ....
In the next section we explain how Cvxportfolio policy objects work,
which is useful to know when extending them.

.. _execution-model:

Policy execution model
----------------------

Cvxportfolio policy objects and their internal components inherit from the
:class:`cvxportfolio.estimator.Estimator` base class. That abstraction
provides three methods with both a recursive and a simple version. Then,
the :class:`cvxportfolio.estimator.CvxpyExpressionEstimator` subclass adds
one other method that is used to provide the CVXPY specific code for
optimization policies and objects.

All objects that inherit from these base classes typically inherit empty base
implementations for all methods. Then each object redefines what it needs
in order to function. In this way we manage to keep the infrastructural code
limited to :doc:`one module <estimators>`, and each specific object the user
interacts with contains the code that is strictly pertaining to it. (That is
very much unlike how CVXPY is designed, encapsulation is much less strict in
its codebase.)

Here's a brief discussion of these methods are and what they are or can be used
for:

- **Initialization.** :meth:`cvxportfolio.estimator.Estimator.initialize_estimator` and its
  recursive version :meth:`cvxportfolio.estimator.Estimator.initialize_estimator_recursive`
- **Evaluation.** :meth:`cvxportfolio.estimator.Estimator.values_in_time` and its recursive
  version :meth:`cvxportfolio.estimator.Estimator.values_in_time_recursive`
- **Finalization.** :meth:`cvxportfolio.estimator.Estimator.finalize_estimator` and its
  recursive version :meth:`cvxportfolio.estimator.Estimator.finalize_estimator_recursive`
- **CVXPY specific.** :meth:`cvxportfolio.estimator.CvxpyExpressionEstimator.compile_to_cvxpy`


*To be continued.*
