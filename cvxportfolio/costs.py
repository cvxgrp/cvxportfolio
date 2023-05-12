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

import cvxpy as cvx
import numpy as np
import pandas as pd
import copy

from .estimator import CvxpyExpressionEstimator, ParameterEstimator, DataEstimator

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
        implements `compile_to_cvxpy` and values_in_time
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
                raise SyntaxError(
      "You can only sum `BaseCost` instances to other `BaseCost` instances.")
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

    def pre_evaluation(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el.pre_evaluation(*args, **kwargs) for el in self.costs]

    def values_in_time(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el.values_in_time(*args, **kwargs) for el in self.costs]

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Iterate over constituent costs."""
        self.expression = 0
        for multiplier, cost in zip(self.multipliers, self.costs):
            self.expression += multiplier * \
                cost.compile_to_cvxpy(w_plus, z, portfolio_value)

        return self.expression


class HoldingCost(BaseCost):
    """A model for holding costs.

    :param borrow_spread: spread on top of cash return payed for borrowing assets,
        including cash. If ``None``, the default, it gets from :class:`Backtest` the
        value for the period.
    :type borrow_spread: float or pd.Series or pd.DataFrame or None
    :param cash_lending_spread: spread that subtracts from the cash return for
        uninvested cash. If ``None``, the default, it gets from :class:`Backtest` the
        value for the period.
    :type cash_lending_spread: float or pd.Series or None
    :param dividends: dividends payed (expressed as fraction of the stock value)
        for each period.  If ``None``, the default, it gets from :class:`Backtest` the
        value for the period.
    :type dividends: pd.DataFrame or None
    """

    def __init__(self, 
        spread_on_borrowing_stocks_percent=.5,
        dividends=None,
        ):
        
        self.spread_on_borrowing_stocks_percent = None if spread_on_borrowing_stocks_percent is None else \
            DataEstimator(spread_on_borrowing_stocks_percent)
        self.dividends = None if dividends is None else \
            ParameterEstimator(dividends)

        
    def pre_evaluation(self, universe, backtest_times):
        super().pre_evaluation(universe=universe, backtest_times=backtest_times)
        
        if not (self.spread_on_borrowing_stocks_percent is None):
            self.borrow_cost_stocks = cvx.Parameter(len(universe) - 1, nonneg=True)
        
        
    def values_in_time(self, t, past_returns, **kwargs):
        """We use yesterday's value of the cash return here while in the simulator
        we use today's. In the US, updates to the FED rate are published outside
        of trading hours so we might as well use the actual value for today's. The difference
        is very small so for now we do this. 
        """
        super().values_in_time(t=t, past_returns=past_returns, **kwargs)
                               
        cash_return = past_returns.iloc[-1,-1]

        if not (self.spread_on_borrowing_stocks_percent is None):
            self.borrow_cost_stocks.value = np.ones(past_returns.shape[1] - 1) * (cash_return) + \
                self.spread_on_borrowing_stocks_percent.current_value / (100 * 252)
        

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression."""
        
        expression = 0. 
        
        if not (self.spread_on_borrowing_stocks_percent is None):
           expression += cvx.multiply(self.borrow_cost_stocks, cvx.neg(w_plus)[:-1])
        
        if not (self.dividends is None):
            expression -= cvx.multiply(self.dividends, w_plus[:-1])
        assert cvx.sum(expression).is_convex()
        return cvx.sum(expression)


class TransactionCost(BaseCost):
    """A model for transaction costs.

    See pages 10-11 in `the book <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_.
    We don't include the short-term alpha term `c` here because it
    can be expressed with a separate `ReturnsForecast` object. 

    :param spreads:
    :type spreads: float or pd.Series or pd.DataFrame
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

    def __init__(self, spreads=0., pershare_cost=0.005, b=1.0, window_sigma_est=250, window_volume_est=250, exponent=1.5):

        self.spreads = DataEstimator(spreads)
        self.pershare_cost = DataEstimator(pershare_cost)

        self.b = DataEstimator(b)
        self.window_sigma_est = window_sigma_est
        self.window_volume_est = window_volume_est
        self.exponent = exponent
            
    def pre_evaluation(self, universe, backtest_times):
        super().pre_evaluation(universe=universe, backtest_times=backtest_times)
        self.first_term_multiplier = cvx.Parameter(len(universe)-1, nonneg=True)
        self.second_term_multiplier = cvx.Parameter(len(universe)-1, nonneg=True)

    def values_in_time(self, t,  current_portfolio_value, past_returns, past_volumes, current_prices, **kwargs):
        
        super().values_in_time(t=t, current_portfolio_value=current_portfolio_value, 
            past_returns=past_returns, past_volumes=past_volumes, 
            current_prices=current_prices, **kwargs)
            
        self.first_term_multiplier.value = self.spreads.current_value/2. + self.pershare_cost.current_value / current_prices
        sigma_est = np.sqrt((past_returns.iloc[-self.window_sigma_est:, :-1]**2).mean()).values
        volume_est = past_volumes.iloc[-self.window_volume_est:].mean().values

        self.second_term_multiplier.value = self.b.current_value * sigma_est * \
            (current_portfolio_value / volume_est) ** (self.exponent - 1)
        
    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):

        expression = cvx.abs(z[:-1]).T @ self.first_term_multiplier
        assert expression.is_convex()
        expression += (cvx.abs(z[:-1]) ** self.exponent).T @ self.second_term_multiplier
        assert expression.is_convex()
        return expression
        
