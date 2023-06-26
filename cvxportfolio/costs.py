# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
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

"""This module implements cost functions used by optimization-based policies.

Currently these are two: :class:`TransactionCost` and :class:`HoldingCost`.

The default parameters are chosen to approximate real market costs as well as
possible. 
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import copy
import inspect

from .estimator import CvxpyExpressionEstimator,  DataEstimator
from .utils import periods_per_year
__all__ = ["HoldingCost", "TransactionCost"]


class BaseCost(CvxpyExpressionEstimator):
    """Base class for cost objects (and also risks).

    Here there is some logic used to implement the algebraic operations.
    See also :class:`CombinedCost`.
    """

    def __mul__(self, other):
        """Multiply by constant."""
        if not np.isscalar(other):
            raise SyntaxError("You can only multiply cost by a scalar.")
        return CombinedCosts([self], [other])

    def __add__(self, other):
        """Add cost expression to another cost expression.

        Idea is to create a new CombinedCost class that
        implements `_compile_to_cvxpy` and _recursive_values_in_time
        by summing over costs.

        """
        if isinstance(other, CombinedCosts):
            return other + self
        return CombinedCosts([self, other], [1.0, 1.0])

    def __rmul__(self, other):
        """Multiply by constant."""
        return self * other

    def __radd__(self, other):
        """Add cost expression to another cost."""
        return self + other

    def __neg__(self):
        """Take negative of cost."""
        return self * -1

    def __sub__(self, other):
        """Subtract other cost."""
        return self.__add__(-other)

    def __rsub__(self, other):
        """Subtract from other cost."""
        return other.__add__(-self)


class CombinedCosts(BaseCost):
    """Algebraic combination of :class:`BaseCost` instances.

    :var costs: instances of :class:`BaseCost`
    :type var: list 
    :var multipliers: floats that multiply the ``costs``
    :type var: list
    """

    def __init__(self, costs, multipliers):
        for cost in costs:
            if not isinstance(cost, BaseCost):
                raise SyntaxError("You can only sum `BaseCost` instances to other `BaseCost` instances.")
        self.costs = costs
        self.multipliers = multipliers
        
    def __add__(self, other):
        """Add other (combined) cost to self."""
        if isinstance(other, CombinedCosts):
            return CombinedCosts(self.costs + other.costs, self.multipliers + other.multipliers)
        else:
            return CombinedCosts(self.costs + [other], self.multipliers + [1.0])

    def __mul__(self, other):
        """Multiply by constant."""
        return CombinedCosts(self.costs, [el * other for el in self.multipliers])

    def _recursive_pre_evaluation(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el._recursive_pre_evaluation(*args, **kwargs) for el in self.costs]

    def _recursive_values_in_time(self, **kwargs):
        """Iterate over constituent costs."""
        [el._recursive_values_in_time(**kwargs) for el in self.costs ]

    def _compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Iterate over constituent costs."""
        self.expression = 0
        for multiplier, cost in zip(self.multipliers, self.costs):
            self.expression += multiplier * cost._compile_to_cvxpy(w_plus, z, portfolio_value) 
        return self.expression


class HoldingCost(BaseCost):
    """A model for holding costs.

    :param spread_on_borrowing_stocks_percent: spread on top of cash return payed for borrowing assets,
        including cash. If ``None``, the default, it gets from :class:`Backtest` the
        value for the period.
    :type spread_on_borrowing_stocks_percent: float or pd.Series or pd.DataFrame or None
    :param spread_on_lending_cash_percent: spread that subtracts from the cash return for
        uninvested cash.
    :type spread_on_lending_cash_percent: float or pd.Series or None
    :param spread_on_borrowing_cash_percent: spread that adds to the cash return as rate payed for
        borrowed cash. This is not used as a policy optimization cost but is for the simulator cost.
    :type spread_on_borrowing_cash_percent: float or pd.Series or None
    :param dividends: dividends payed (expressed as fraction of the stock value)
        for each period.  If ``None``, the default, it gets from :class:`Backtest` the
        value for the period.
    :type dividends: pd.DataFrame or None
    :param periods_per_year: period per year (used in calculation of per-period cost). If None it is calculated
        automatically.
    :type periods_per_year: int or None
    :param cash_return_on_borrow: whether to add (negative of) cash return to borrow cost of assets
    :type cash_return_on_borrow: bool
    """

    def __init__(self, 
        spread_on_borrowing_stocks_percent=.5,
        spread_on_lending_cash_percent=.5,
        spread_on_borrowing_cash_percent=.5,
        periods_per_year=None,
        cash_return_on_borrow=True, #TODO revisit this plus spread_on_borrowing_stocks_percent syntax 
        dividends=0.):
        
        self.spread_on_borrowing_stocks_percent = None if spread_on_borrowing_stocks_percent is None else \
            DataEstimator(spread_on_borrowing_stocks_percent)
        if dividends is None:
            self.dividends = None
        else:
            self.dividends = DataEstimator(dividends, compile_parameter=True)
            #self.dividends_parameter = ParameterEstimator(dividends)
        
        self.spread_on_lending_cash_percent = None if spread_on_lending_cash_percent is None else \
            DataEstimator(spread_on_lending_cash_percent)        
        self.spread_on_borrowing_cash_percent = None if spread_on_borrowing_cash_percent is None else \
            DataEstimator(spread_on_borrowing_cash_percent)
            
        self.periods_per_year = periods_per_year
        self.cash_return_on_borrow = cash_return_on_borrow
        
    def _pre_evaluation(self, universe, backtest_times):
        
        if not (self.spread_on_borrowing_stocks_percent is None):
            self.borrow_cost_stocks = cp.Parameter(len(universe) - 1, nonneg=True)
        
        
    def _values_in_time(self, t, past_returns, **kwargs):
        """We use yesterday's value of the cash return here while in the simulator
        we use today's. In the US, updates to the FED rate are published outside
        of trading hours so we might as well use the actual value for today's. The difference
        is very small so for now we do this. 
        """
        ppy = periods_per_year(past_returns.index) if self.periods_per_year is None else \
            self.periods_per_year
                               
        cash_return = past_returns.iloc[-1,-1]

        if not (self.spread_on_borrowing_stocks_percent is None):
            self.borrow_cost_stocks.value = np.ones(past_returns.shape[1] - 1) * (
                    cash_return if self.cash_return_on_borrow else 0.) + \
                self.spread_on_borrowing_stocks_percent.current_value / (100 * ppy)
                
    def _simulate(self, t, h_plus, current_and_past_returns, **kwargs):
        
        ppy = periods_per_year(current_and_past_returns.index) if self.periods_per_year is None else \
            self.periods_per_year
        
        cash_return = current_and_past_returns.iloc[-1,-1]        
        multiplier = 1 / (100 * ppy)
        result = 0.
        borrowed_stock_positions = np.minimum(h_plus.iloc[:-1], 0.)
        result += np.sum(((cash_return if self.cash_return_on_borrow else 0.) + 
            self.spread_on_borrowing_stocks_percent._recursive_values_in_time(t) * multiplier) * borrowed_stock_positions)
        result += np.sum(h_plus[:-1] * self.dividends._recursive_values_in_time(t))
          
        # lending_spread = DataEstimator(spread_on_lending_cash_percent)._recursive_values_in_time(t) * multiplier
        # borrowing_spread = DataEstimator(spread_on_borrowing_cash_percent)._recursive_values_in_time(t) * multiplier

        # cash_return = current_and_past_returns.iloc[-1,-1]
        real_cash = h_plus.iloc[-1] + sum(np.minimum(h_plus.iloc[:-1], 0.))

        if real_cash > 0:
            result += real_cash * (max(cash_return - self.spread_on_lending_cash_percent._recursive_values_in_time(t) * multiplier, 0.) - cash_return)
        else:
            result += real_cash * self.spread_on_borrowing_cash_percent._recursive_values_in_time(t) * multiplier
            
        return result
            
            
    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression."""
        
        expression = 0. 
        
        if not (self.spread_on_borrowing_stocks_percent is None):
           expression += cp.multiply(self.borrow_cost_stocks, cp.neg(w_plus)[:-1])
        
        if not (self.dividends is None):
            expression -= cp.multiply(self.dividends.parameter, w_plus[:-1])
        assert cp.sum(expression).is_convex()
        return cp.sum(expression)


class TransactionCost(BaseCost):
    """A model for transaction costs.

    See pages 10-11 in `the book <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_.
    We don't include the short-term alpha term `c` here because it
    can be expressed with a separate `ReturnsForecast` object. 

    :param a:
    :type a: float or pd.Series or pd.DataFrame
    :param pershare_cost: per-share trade cost, as as in :class:`MarketSimulator`
    :type pershare_cost: float or pd.Series or pd.DataFrame
    :param b: coefficient of the second term
    :type b: float or pd.Series or pd.DataFrame
    :param window_sigma_est: length of the window standard deviation of past returns used to estimate :math:`\sigma`
    :type window_sigma_est: int
    :param window_volume_est: length of the window mean of past volumes used as volume estimate
    :type window_volume_est: int
    :param exponent: exponent of the non-linear term, default 1.5
    :type exponent: float
    """

    def __init__(self, a=0., pershare_cost=0.005, b=1.0, window_sigma_est=250, window_volume_est=250, exponent=1.5):

        self.a = None if a is None else DataEstimator(a)
        self.pershare_cost = None if pershare_cost is None else DataEstimator(pershare_cost)
        self.b = None if b is None else DataEstimator(b)
        self.window_sigma_est = window_sigma_est
        self.window_volume_est = window_volume_est
        self.exponent = exponent
            
    def _pre_evaluation(self, universe, backtest_times):
        self.first_term_multiplier = cp.Parameter(len(universe)-1, nonneg=True)
        if not (self.b is None):
            self.second_term_multiplier = cp.Parameter(len(universe)-1, nonneg=True)

    def _values_in_time(self, t,  current_portfolio_value, past_returns, past_volumes, current_prices, **kwargs):
                    
        self.first_term_multiplier.value = self.a.current_value + self.pershare_cost.current_value / current_prices.values
        if not (self.b is None):
            sigma_est = np.sqrt((past_returns.iloc[-self.window_sigma_est:, :-1]**2).mean()).values
            volume_est = past_volumes.iloc[-self.window_volume_est:].mean().values

            self.second_term_multiplier.value = self.b.current_value * sigma_est * \
                (current_portfolio_value / volume_est) ** (self.exponent - 1)
                
    def _simulate(self, t, u, current_and_past_returns, current_and_past_volumes, current_prices, **kwargs):
        
        sigma = np.std(current_and_past_returns.iloc[-self.window_sigma_est:, :-1], axis=0)

        result = 0.
        if not (self.pershare_cost is None):
            if current_prices is None:
                raise SyntaxError("If you don't provide prices you should set persharecost to None")
            result += self.pershare_cost._recursive_values_in_time(t) * int(sum(np.abs(u.iloc[:-1] + 1E-6) / current_prices.values))

        result += sum(self.a._recursive_values_in_time(t) * np.abs(u.iloc[:-1]))

        if not (self.b is None):
            if current_and_past_volumes is None:
                raise SyntaxError("If you don't provide volumes you should set b to None")
            # we add 1 to the volumes to prevent 0 volumes error (trades are cancelled on 0 volumes)
            result += (np.abs(u.iloc[:-1])**self.exponent) @ (self.b._recursive_values_in_time(t)  *
                sigma / ((current_and_past_volumes.iloc[-1] + 1) ** (self.exponent - 1)))

        assert not np.isnan(result)
        assert not np.isinf(result)

        return -result
        
        
    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):

        expression = cp.abs(z[:-1]).T @ self.first_term_multiplier
        assert expression.is_convex()
        if not (self.b is None):
            expression += (cp.abs(z[:-1]) ** self.exponent).T @ self.second_term_multiplier
            assert expression.is_convex()
        return expression
