Manual
======

Here we explain some concepts that apply throughout the library and are useful
to know. The code blocks in this document assume the following imports

.. code-block:: python

    import cvxportfolio as cvx
    import numpy as np
    import pandas as pd

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
  
      One should be careful and make sure that the timestamps used match the 
      timestamps used by the market data server: for example they must have the
      same timezone. To find the correct timestamps you can call the 
      :meth:`trading_calendar` method of a market data object.
   
- **Pandas dataframes**.
- **Numpy arrays**.
