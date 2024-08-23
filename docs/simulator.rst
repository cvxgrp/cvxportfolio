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

Simulator
=========

.. automodule:: cvxportfolio.simulator

.. py:module:: cvxportfolio
     :no-index:

.. autoclass:: MarketSimulator

    .. automethod:: backtest
    
    .. method:: run_backtest
    
        Alias of :meth:`backtest`, as it was defined originally in 
        :paper:`section 6.1 <section.6.1>` of the paper. 
    
    .. automethod:: backtest_many
    
    .. method:: run_multiple_backtest

        Alias of :meth:`backtest_many`, as it was defined originally in 
        :paper:`section 6.1 <section.6.1>` of the paper.
	
    .. automethod:: optimize_hyperparameters

    
.. autoclass:: StockMarketSimulator

    .. automethod:: backtest
    
    .. method:: run_backtest
    
        Alias of :meth:`backtest`, as it was defined originally in 
        :paper:`section 6.1 <section.6.1>` of the paper. 
    
    .. automethod:: backtest_many
    
    .. method:: run_multiple_backtest

        Alias of :meth:`backtest_many`, as it was defined originally in 
        :paper:`section 6.1 <section.6.1>` of the paper.
	
    .. automethod:: optimize_hyperparameters



