.. _objective-terms-page:

Objective terms
===============

.. toctree::
	:maxdepth: 2
	
.. py:module:: cvxportfolio

.. _returns-page:

Return models
-------------

.. automodule:: cvxportfolio.returns

.. py:module:: cvxportfolio
    :noindex:

.. autoclass:: ReturnsForecast

.. autoclass:: CashReturn

.. _costs-page:

Cost models
-----------

.. automodule:: cvxportfolio.costs

.. py:module:: cvxportfolio
    :noindex:
    
.. autoclass:: HoldingCost

.. autoclass:: StocksHoldingCost

.. autoclass:: TransactionCost

.. autoclass:: StocksTransactionCost

.. autoclass:: SoftConstraint

.. autoclass:: TcostModel

.. autoclass:: HcostModel

.. _risks-page:

Risk models
-----------

.. automodule:: cvxportfolio.risks

.. py:module:: cvxportfolio
    :noindex:

.. autoclass:: DiagonalCovariance

.. autoclass:: FullCovariance

.. autoclass:: FactorModelCovariance

.. autoclass:: WorstCaseRisk

.. autoclass:: FullSigma

.. autoclass:: FactorModel


Forecast error models
---------------------

.. autoclass:: ReturnsForecastError

.. autoclass:: RiskForecastError

Base classes (for defining your own objective terms)
----------------------------------------------------

.. autoclass:: cvxportfolio.costs.Cost

.. autoclass:: cvxportfolio.costs.SimulatorCost