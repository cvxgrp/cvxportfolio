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

Currently these are two: :class:`StocksTransactionCost` and :class:`StocksHoldingCost`.

The default parameters are chosen to approximate real costs for the stock market
as well as possible. 
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import copy
import inspect

from .estimator import CvxpyExpressionEstimator,  DataEstimator
from .utils import periods_per_year
from .hyperparameters import HyperParameter

__all__ = ["HoldingCost", "TransactionCost",
           "StocksTransactionCost", "StocksHoldingCost"]


class BaseCost(CvxpyExpressionEstimator):
    """Base class for cost objects (and also risks).

    Here there is some logic used to implement the algebraic operations.
    See also :class:`CombinedCost`.
    """

    def _simulate(self, *args, **kwargs):
        """Simulate cost, used by market simulator.

        Look at its invocation in ``MarketSimulator`` for its list of 
        arguments.

        Cost classes that are meant to be used in the simulator
        should implement this.
        """
        raise NotImplementedError

    def __mul__(self, other):
        """Multiply by constant."""
        if not (np.isscalar(other) or isinstance(other, HyperParameter)):
            raise SyntaxError("You can only multiply cost by a scalar "
                              + "or a HyperParameter instance. (Have you instantiated it?)")
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

    :param costs: instances of :class:`BaseCost`
    :type costs: list 
    :param multipliers: floats that multiply the ``costs``
    :type multipliers: list
    """

    def __init__(self, costs, multipliers):
        for cost in costs:
            if not isinstance(cost, BaseCost):
                raise SyntaxError(
                    "You can only sum cost instances to other cost instances.")
        self.costs = costs
        self.multipliers = multipliers

    def __add__(self, other):
        """Add other (combined) cost to self."""
        if isinstance(other, CombinedCosts):
            return CombinedCosts(self.costs + other.costs,
                                 self.multipliers + other.multipliers)
        else:
            return CombinedCosts(self.costs + [other], self.multipliers + [1.0])

    def __mul__(self, other):
        """Multiply by constant."""
        return CombinedCosts(self.costs,
                             [el * other for el in self.multipliers])

    def _recursive_pre_evaluation(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el._recursive_pre_evaluation(*args, **kwargs) for el in self.costs]

    def _recursive_values_in_time(self, **kwargs):
        """Iterate over constituent costs."""
        [el._recursive_values_in_time(**kwargs) for el in self.costs]

    def _compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Iterate over constituent costs."""
        self.expression = 0
        for multiplier, cost in zip(self.multipliers, self.costs):
            self.expression += multiplier * \
                cost._compile_to_cvxpy(w_plus, z, portfolio_value)
        return self.expression

    def _collect_hyperparameters(self):
        return sum([el._collect_hyperparameters() for el in self.costs], []) + \
            sum([el._collect_hyperparameters() for el in self.multipliers if
                hasattr(el, '_collect_hyperparameters')], [])

    def __repr__(self):
        """Pretty-print."""
        result = ''
        for i, (mult, cost) in enumerate(zip(self.multipliers, self.costs)):
            if mult == 0:
                continue
            if mult < 0:
                result += ' - ' if i > 0 else '-'
            else:
                result += ' + ' if i > 0 else ''
            result += (str(abs(mult)) + ' * ' if abs(mult) != 1 else '')
            result += cost.__repr__()
        return result


class HoldingCost(BaseCost):
    """This is a generic holding cost model.

    Currently it is not meant to be used directly. Look at
    :class:`StocksHoldingCost` for its version specialized to
    the stock market.
    """

    def __init__(self,
                 spread_on_borrowing_assets_percent=None,
                 spread_on_lending_cash_percent=None,
                 spread_on_borrowing_cash_percent=None,
                 periods_per_year=None,
                 # TODO revisit this plus spread_on_borrowing_stocks_percent syntax
                 cash_return_on_borrow=False,
                 dividends=None):

        self.spread_on_borrowing_assets_percent = None if spread_on_borrowing_assets_percent is None else \
            DataEstimator(spread_on_borrowing_assets_percent)
        self.dividends = None if dividends is None else DataEstimator(
            dividends, compile_parameter=True)
        self.spread_on_lending_cash_percent = None if spread_on_lending_cash_percent is None else \
            DataEstimator(spread_on_lending_cash_percent)
        self.spread_on_borrowing_cash_percent = None if spread_on_borrowing_cash_percent is None else \
            DataEstimator(spread_on_borrowing_cash_percent)

        self.periods_per_year = periods_per_year
        self.cash_return_on_borrow = cash_return_on_borrow

    def _pre_evaluation(self, universe, backtest_times):

        if self.spread_on_borrowing_assets_percent is not None or self.cash_return_on_borrow:
            self.borrow_cost_stocks = cp.Parameter(
                len(universe) - 1, nonneg=True)

    def _values_in_time(self, t, past_returns, **kwargs):
        """We use yesterday's value of the cash return here while in the simulator
        we use today's. In the US, updates to the FED rate are published outside
        of trading hours so we might as well use the actual value for today's. The difference
        is very small so for now we do this. 
        """
        ppy = periods_per_year(past_returns.index) if self.periods_per_year is None else \
            self.periods_per_year

        cash_return = past_returns.iloc[-1, -1]

        if self.spread_on_borrowing_assets_percent is not None or self.cash_return_on_borrow:
            self.borrow_cost_stocks.value = np.ones(past_returns.shape[1] - 1) * (
                cash_return if self.cash_return_on_borrow else 0.) + \
                self.spread_on_borrowing_assets_percent.current_value / \
                (100 * ppy)

    def _simulate(self, t, h_plus, current_and_past_returns, **kwargs):

        ppy = periods_per_year(current_and_past_returns.index) if self.periods_per_year is None else \
            self.periods_per_year

        cash_return = current_and_past_returns.iloc[-1, -1]
        multiplier = 1 / (100 * ppy)
        result = 0.
        borrowed_stock_positions = np.minimum(h_plus.iloc[:-1], 0.)
        result += np.sum(((cash_return if self.cash_return_on_borrow else 0.) +
                          (self.spread_on_borrowing_assets_percent._recursive_values_in_time(t) * multiplier if
                           self.spread_on_borrowing_assets_percent is not None else 0.))
                         * borrowed_stock_positions)

        if self.dividends is not None:
            result += np.sum(h_plus[:-1] *
                             self.dividends._recursive_values_in_time(t))

        # lending_spread = DataEstimator(spread_on_lending_cash_percent)._recursive_values_in_time(t) * multiplier
        # borrowing_spread = DataEstimator(spread_on_borrowing_cash_percent)._recursive_values_in_time(t) * multiplier

        # cash_return = current_and_past_returns.iloc[-1,-1]
        real_cash = h_plus.iloc[-1] + sum(np.minimum(h_plus.iloc[:-1], 0.))

        if real_cash > 0:
            if self.spread_on_lending_cash_percent is not None:
                result += real_cash * \
                    (max(cash_return - self.spread_on_lending_cash_percent._recursive_values_in_time(
                        t) * multiplier, 0.) - cash_return)
        else:
            if self.spread_on_borrowing_cash_percent is not None:
                result += real_cash * \
                    self.spread_on_borrowing_cash_percent._recursive_values_in_time(
                        t) * multiplier

        return result

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression."""

        expression = 0.

        if not (self.spread_on_borrowing_assets_percent is None):
            expression += cp.multiply(self.borrow_cost_stocks,
                                      cp.neg(w_plus)[:-1])

        if not (self.dividends is None):
            expression -= cp.multiply(self.dividends.parameter, w_plus[:-1])
        assert cp.sum(expression).is_convex()
        return cp.sum(expression)


class StocksHoldingCost(HoldingCost):
    """A model for holding cost of stocks.

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
                 # TODO revisit this plus spread_on_borrowing_stocks_percent syntax
                 cash_return_on_borrow=True,
                 dividends=0.):

        super().__init__(
            spread_on_borrowing_assets_percent=spread_on_borrowing_stocks_percent,
            spread_on_lending_cash_percent=spread_on_lending_cash_percent,
            spread_on_borrowing_cash_percent=spread_on_borrowing_cash_percent,
            periods_per_year=periods_per_year,
            cash_return_on_borrow=cash_return_on_borrow,
            dividends=dividends)


class TransactionCost(BaseCost):
    """This is a generic model for transaction cost of financial assets.

    Currently it is not meant to be used directly. Look at
    :class:`StocksTransactionCost` for its version specialized
    to the stock market.
    """

    def __init__(self, a=None, pershare_cost=None, b=0., window_sigma_est=None,
                 window_volume_est=None, exponent=None):

        self.a = None if a is None else DataEstimator(a)
        self.pershare_cost = None if pershare_cost is None else DataEstimator(
            pershare_cost)
        self.b = None if b is None else DataEstimator(b)
        self.window_sigma_est = window_sigma_est
        self.window_volume_est = window_volume_est
        self.exponent = exponent

    def _pre_evaluation(self, universe, backtest_times):
        if self.a is not None or self.pershare_cost is not None:
            self.first_term_multiplier = cp.Parameter(
                len(universe)-1, nonneg=True)
        if self.b is not None:
            self.second_term_multiplier = cp.Parameter(
                len(universe)-1, nonneg=True)

    def _values_in_time(self, t,  current_portfolio_value, past_returns,
                        past_volumes, current_prices, **kwargs):

        tmp = 0.

        if self.a is not None:
            _ = self.a.current_value
            tmp += _ * \
                np.ones(past_returns.shape[1]-1) if np.isscalar(_) else _
        if self.pershare_cost is not None:
            if current_prices is None:
                raise SyntaxError(
                    "If you don't provide prices you should set pershare_cost to None")
            tmp += self.pershare_cost.current_value / current_prices.values

        if self.a is not None or self.pershare_cost is not None:
            self.first_term_multiplier.value = tmp

        if self.b is not None:

            if (self.window_sigma_est is None) or \
                    (self.window_volume_est is None):
                ppy = periods_per_year(past_returns.index)
            windowsigma = ppy if (
                self.window_sigma_est is None) else self.window_sigma_est
            windowvolume = ppy if (
                self.window_volume_est is None) else self.window_volume_est

            # TODO refactor this with forecast.py logic
            sigma_est = np.sqrt(
                (past_returns.iloc[-windowsigma:, :-1]**2).mean()).values
            volume_est = past_volumes.iloc[-windowvolume:].mean().values

            self.second_term_multiplier.value = self.b.current_value * sigma_est * \
                (current_portfolio_value /
                 volume_est) ** ((2 if self.exponent is None else self.exponent) - 1)

    def _simulate(self, t, u, current_and_past_returns,
                  current_and_past_volumes, current_prices, **kwargs):

        if self.window_sigma_est is None:
            windowsigma = periods_per_year(current_and_past_returns.index)
        else:
            windowsigma = self.window_sigma_est

        exponent = (1.5 if self.exponent is None else self.exponent)

        sigma = np.std(
            current_and_past_returns.iloc[-windowsigma:, :-1], axis=0)

        result = 0.
        if self.pershare_cost is not None:
            if current_prices is None:
                raise SyntaxError(
                    "If you don't provide prices you should set pershare_cost to None")
            result += self.pershare_cost._recursive_values_in_time(t) * int(
                sum(np.abs(u.iloc[:-1] + 1E-6) / current_prices.values))

        if self.a is not None:
            result += sum(self.a._recursive_values_in_time(t)
                          * np.abs(u.iloc[:-1]))

        if self.b is not None:
            if current_and_past_volumes is None:
                raise SyntaxError(
                    "If you don't provide volumes you should set b to None")
            # we add 1 to the volumes to prevent 0 volumes error (trades are cancelled on 0 volumes)
            result += (np.abs(u.iloc[:-1])**exponent) @ (
                self.b._recursive_values_in_time(t) *
                sigma / ((current_and_past_volumes.iloc[-1] + 1) ** (exponent - 1)))

        assert not np.isnan(result)
        assert not np.isinf(result)

        return -result

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):

        expression = 0
        if self.a is not None or self.pershare_cost is not None:
            expression += cp.abs(z[:-1]).T @ self.first_term_multiplier
            assert expression.is_convex()
        if self.b is not None:
            expression += (cp.abs(z[:-1]) ** (
                2 if self.exponent is None else self.exponent)
            ).T @ self.second_term_multiplier
            assert expression.is_convex()
        return expression


class StocksTransactionCost(TransactionCost):
    """A model for transaction costs of stocks.

    See pages 10-11 in `the book <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_.
    We don't include the short-term alpha term `c` here because it
    can be expressed with a separate `ReturnsForecast` object. 

    :param a: linear cost, which multiplies the absolute value of each trade. This can model
        (half) the bid-ask spread, or any fee linear in the size of a trade.
    :type a: float or pd.Series or pd.DataFrame
    :param pershare_cost: per-share trade cost, amount of dollars paid for each share traded. 
    :type pershare_cost: float or pd.Series or pd.DataFrame
    :param b: coefficient of the non-linear term of the transaction cost model, which multiplies
        the estimated volatility for each stock (see the book).  
    :type b: float or pd.Series or pd.DataFrame
    :param window_sigma_est: we use an historical rolling standard deviation to estimate 
        the average size of the return on a stock on each day, and this multiplies the 
        second term of the transaction cost model.  See the paper for an explanation of the model. 
        Here you specify the length of the rolling window to use. If None (the default) it uses
        a length of 1 year (approximated with the data provided).
    :type window_sigma_est: int or None
    :param window_volume_est: length of the window for the mean of past volumes used as estimate
        of each period's volume. Has no effect on the simulator version of this which uses
        the actual volume. If None (the default) it uses a length of 1 year (approximated 
        with the data provided).
    :type window_volume_est: int
    :param exponent: exponent of the non-linear term, defaults (if set to ``None``) to 1.5 for
        the simulator version, and 2 for the optimization version (because it is more efficient
        numerically and the difference is small, you can change it if you want).
    :type exponent: float or None
    """

    def __init__(self, a=0., pershare_cost=0.005, b=1.0, window_sigma_est=None,
                 window_volume_est=None, exponent=1.5):

        super().__init__(a=a, pershare_cost=pershare_cost, b=b, window_sigma_est=window_sigma_est,
                         window_volume_est=window_volume_est, exponent=exponent)
