Data Interfaces 
===============

.. automodule:: cvxportfolio.data

.. py:module:: cvxportfolio
     :noindex:

Single-symbol data download and storage
---------------------------------------

.. autoclass:: YahooFinance

.. autoclass:: Fred

Market data servers
-------------------

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

    .. automethod:: universe_at_time
    
    .. automethod:: trading_calendar
    
    .. autoproperty:: periods_per_year
    
    .. autoproperty:: full_universe
    
    .. automethod:: partial_universe_signature