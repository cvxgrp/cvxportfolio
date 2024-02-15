# Copyright 2023 Enzo Busseti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
