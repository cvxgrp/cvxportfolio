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

Data Interfaces 
===============

.. automodule:: cvxportfolio.data

.. py:module:: cvxportfolio
     :no-index:

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
     :no-index:

.. autoclass:: SymbolData

.. autoclass:: MarketData
    
    .. automethod:: serve

    .. automethod:: universe_at_time
    
    .. automethod:: trading_calendar
    
    .. autoproperty:: periods_per_year
    
    .. autoproperty:: full_universe
    
    .. automethod:: partial_universe_signature