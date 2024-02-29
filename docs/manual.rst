Manual
======

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
code shown above, the default parameters of :class:`cvxportfolio.ReturnsForecast`
and :class:`cvxportfolio.FullCovariance` are forecasters that compute historical
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
that you can access in a :class:`cvxportfolio.BacktestResult` object, include
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
:class:`cvxportfolio.MultiPeriodOptimization` policy.

We note that the simpler single-period optimization model, implemented by 
:class:`cvxportfolio.SinglePeriodOptimization` and defined in
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

*To be continued.*
