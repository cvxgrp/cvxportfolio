Data Interfaces 
===============

.. automodule:: cvxportfolio.data

.. py:module:: cvxportfolio
     :noindex:

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

.. py:module:: cvxportfolio.data
     :noindex:

.. autoclass:: SymbolData

.. autoclass:: MarketData
    
    .. automethod:: serve
    
    .. automethod:: trading_calendar