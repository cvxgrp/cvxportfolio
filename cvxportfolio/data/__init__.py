# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
"""This module include classes that download, store, and serve market data.

The two main abstractions are :class:`SymbolData` and :class:`MarketData`.
Neither are exposed outside this module. Their derived classes instead are.
If you want to interface cvxportfolio with financial data source other
than the ones we provide, you should derive from either of those two classes.
"""

from .market_data import *
from .symbol_data import *

__all__ = [
    "YahooFinance", "Fred", "UserProvidedMarketData", "DownloadedMarketData"]
