Data Interfaces 
===============

.. py:module:: cvxportfolio
     :noindex:
	 
.. automodule:: cvxportfolio.data

.. autoclass:: YahooFinance

.. autoclass:: Fred

.. autoclass:: UserProvidedMarketData

    .. automethod:: serve
    
    .. automethod:: trading_calendar

.. autoclass:: DownloadedMarketData

    .. automethod:: serve
    
    .. automethod:: trading_calendar

Base classes (for using other data sources)
-------------------------------------------

.. autoclass:: SymbolData

.. autoclass:: MarketData
    
    .. automethod:: serve
    
    .. automethod:: trading_calendar