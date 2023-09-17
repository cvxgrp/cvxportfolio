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
from .constraints import EqualityConstraint, InequalityConstraint, CostInequalityConstraint
from .errors import *

__all__ = ["HoldingCost", "TransactionCost", "SoftConstraint",
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

    def __le__(self, other):
        """self <= other, return CostInequalityConstraint.

        For now we check here the type of "other"
        but it would be nicer to have CostInequalityConstraint's
        internal DataEstimator throw NotImplemented instead.
        """
        if isinstance(other, float) \
                or isinstance(other, int) \
                or isinstance(other, pd.Series):
            return CostInequalityConstraint(self, other)
        return NotImplemented

    def __ge__(self, other):
        """self >= other, return CostInequalityConstraint."""
        return (-self).__le__(other)


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
            add = (multiplier.current_value 
                if hasattr(multiplier, 'current_value') else multiplier) * \
                    cost._compile_to_cvxpy(w_plus, z, portfolio_value)
            if not add.is_dcp():
                raise ConvexSpecificationError(cost * multiplier)
            if not add.is_concave():
                raise ConvexityError(cost * multiplier)
            self.expression += add
        return self.expression

    def _collect_hyperparameters(self):
        return sum([el._collect_hyperparameters() for el in self.costs], []) + \
            sum([el._collect_hyperparameters() for el in self.multipliers if
                hasattr(el, '_collect_hyperparameters')], [])

    def __repr__(self):
        """Pretty-print."""
        result = ''
        for i, (mult, cost) in enumerate(zip(self.multipliers, self.costs)):
            if not isinstance(mult, HyperParameter):
                if mult == 0:
                    continue
                if mult < 0:
                    result += ' - ' if i > 0 else '-'
                else:
                    result += ' + ' if i > 0 else ''
                result += (str(abs(mult)) + ' * ' if abs(mult) != 1 else '')
            else:
                result += str(mult) + ' * '
            result += cost.__repr__()
        return result


class SoftConstraint(BaseCost):
    """Soft constraint cost.

    :param constraint: cvxportfolio constraint instance whose
        violation we penalize
    :type constraint: cvx.BaseConstraint
    """

    def __init__(self, constraint):
        self.constraint = constraint

    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression."""
        try:
            expr = (self.constraint._compile_constr_to_cvxpy(w_plus, z, w_plus_minus_w_bm)
                    - self.constraint._rhs())
        except AttributeError:
            raise SyntaxError(f"{self.__class__.__name__} can only be used with"
                              " EqualityConstraint or InequalityConstraint instances.")
        if isinstance(self.constraint, EqualityConstraint):
            return cp.sum(cp.abs(expr))
        if isinstance(self.constraint, InequalityConstraint):
            return cp.sum(cp.pos(expr))


def _annual_percent_to_per_period(value, ppy):
    """Transform annual percent to per-period return."""
    return np.exp(np.log(1 + value / 100) / ppy) - 1


class HoldingCost(BaseCost):
    r"""Generic holding cost model, as described in page 11 of the book.
    
    There are two ways to use this class. Either in the costs attribute
    of a :class:`MarketSimulator`, in which case the costs are evaluated
    on the post-trade dollar positions :math:`h^+_t`. Or, 
    as part of the objective function (or as a constraint!) 
    of a :class:`SinglePeriodOptimization` 
    or :class:`MultiPeriodOptimization` trading policy, in which case they
    are evaluated on the post-trade weights :math:`w_t + z_t`. The mathematical
    form is the same (see the discussion at pages 11-12 of the book).
    
    This particular implementation represents the following objective terms
    (expressed here in terms of the post-trade dollar positions):
    
    .. math::
 
        s^T_t {(h^+_t)}_- + l^T_t {(h^+_t)}_+ - d^T_t h^+_t
    
    where :math:`s_t` are the (short) borrowing fees, 
    :math:`l_t` are the fees on long positions,
    and :math:`d_t` are dividend rates (their sign is flipped because
    the costs are deducted from the cash account at each period). See
    below for their precise definition.
    
    Example usage as simulator cost:
    
    .. code-block:: python
    
        borrow_fees = pd.Series([5, 10], index=['AAPL', 'ZM'])
        simulator = cvx.MarketSimulator(['AAPL', 'ZM'], 
            costs=cvx.HoldingCost(short_fees=borrow_fees))
    
    Example usage as trading policy cost:
    
    .. code-block:: python
        
        objective = cvx.ReturnsForecast() - 5 * cvx.FullCovariance() \
            - cvx.HoldingCost(short_fees=10)
        constraints = [cvx.LeverageLimit(3)]    
        policy = cvx.SinglePeriodOptimization(objective, constraints)
    
    :param short_fees: Short borrowing fees expressed as annual percentage;
        you can provide them as a float (constant for all times and all 
        assets), a :class:`pd.Series` indexed by time (constant for all
        assets but varying in time) or by assets' names (constant in time
        but varying across assets), or a :class:`pd.DataFrame` indexed by 
        time and whose columns are the assets' names, if varying both
        in time and across assets. If you use a time-indexed pandas object
        be careful to include all times at which a backtest is evaluated
        (otherwise you'll get a :class:`MissingValueError` exception). If `None`,
        the term is ignored.
    :type short_fees: float, pd.Series, pd.DataFrame or None
    :param long_fees: Fees on long positions expressed as annual percentage;
        same convention as above applies.
    :type long_fees: float, pd.Series, pd.DataFrame or None
    :param dividends: Dividend rates per period. Dividends are already included in the market
        returns by the default data interface (based on Yahoo Finance "adjusted prices")
        and thus this parameter should not be used in normal circumstances.
    :type dividends: float, pd.Series, pd.DataFrame or None
    :param periods_per_year: How many trading periods are there in a year, for
        example 252 (for trading days in the US). This is
        only relevant when using this class as part of a trading policy. If you leave
        this to `None` the following happens.
        The value of periods per year are estimated at each period by looking at 
        the past market returns at that point in time: the number of past periods is divided by 
        the timestamp of the most recent period minus the timestamp of the
        first period (in years). That works well in most cases where there is enough history
        (say, a few years) and saves the user from having to manually enter this.
        If instead you use this object as a cost in a market simulator the parameter has no
        effect. (The length of each trading period in the simulation is known and so the per-period
        rates are evaluated exactly. For example, the rate over a weekend will be higher
        than overnight.)
    :type periods_per_year: float or None
    """

    def __init__(self, short_fees=None, long_fees=None, dividends=None,
                 periods_per_year=None):

        self.short_fees = None if short_fees is None else DataEstimator(short_fees)
        self.long_fees = None if long_fees is None else DataEstimator(long_fees)
        self.dividends = None if dividends is None else DataEstimator(dividends)
        self.periods_per_year = periods_per_year

    def _pre_evaluation(self, universe, backtest_times):
        """Initialize cvxpy parameters.
        
        We don't use the parameter from DataEstimator because we need to
        divide the value by periods_per_year.
        """

        if not (self.short_fees is None):
            self._short_fees_parameter = cp.Parameter(len(universe) - 1, nonneg=True)

        if not (self.long_fees is None):
            self._long_fees_parameter = cp.Parameter(len(universe) - 1, nonneg=True)

        if not (self.dividends is None):
            self._dividends_parameter = cp.Parameter(len(universe) - 1)

    def _values_in_time(self, t, past_returns, **kwargs):
        """Update cvxpy parameters.
        
        We compute the estimate of periods per year from past returns
        (if not provided by the user) and populate the cvxpy parameters
        with the current values of the user-provided data, transformed
        to per-period.
        """
        
        if not ((self.short_fees is None) 
                and (self.long_fees is None) 
                and (self.dividends is None)):
            ppy = periods_per_year(past_returns.index) if self.periods_per_year is None else \
                self.periods_per_year

        if not (self.short_fees is None):
            self._short_fees_parameter.value = \
                np.ones(past_returns.shape[1]-1) * \
                     _annual_percent_to_per_period(self.short_fees.current_value, ppy)

        if not (self.long_fees is None):
            self._long_fees_parameter.value = \
                np.ones(past_returns.shape[1]-1) * \
                     _annual_percent_to_per_period(self.long_fees.current_value, ppy)

        if not (self.dividends is None):
            self._dividends_parameter.value = \
                np.ones(past_returns.shape[1]-1) * self.dividends.current_value


    def _compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression."""

        expression = 0.
        
        if not (self.short_fees is None):
            expression += self._short_fees_parameter.T @ cp.neg(w_plus[:-1])
        
        if not (self.long_fees is None):
            expression += self._long_fees_parameter.T @ cp.pos(w_plus[:-1])

        if not (self.dividends is None):
            # we have a minus sign because costs are deducted from PnL
            expression -= self._dividends_parameter.T @ w_plus[:-1]
            
        assert expression.is_convex()
        
        return expression

    
    def _simulate(self, t, h_plus, current_and_past_returns, t_next, **kwargs):
        """
        TODO: make sure simulator cost sign convention is the same as optimization cost! OK
        TODO: make sure DataEstimator returns np.array of correct size! ~OK
        TODO: make sure simulator cost estimators are recursively evaluated! ~OK
        """
        
        year_divided_by_period = pd.Timedelta('365.24d') / (t_next - t)

        cost = 0. 
        
        # TODO this is a temporary fix, we should plug this into a recursive tree
        for est in [self.short_fees, self.long_fees, self.dividends]:
            if not (est is None):
                est._recursive_pre_evaluation(universe=h_plus.index, backtest_times=[t])
                est._recursive_values_in_time(t=t)

        if not (self.short_fees is None):
            cost += np.sum(_annual_percent_to_per_period(
                self.short_fees.current_value, year_divided_by_period) * (-np.minimum(h_plus[:-1], 0.)))
        
        if not (self.long_fees is None):
            cost += np.sum(_annual_percent_to_per_period(
                self.long_fees.current_value, year_divided_by_period) * np.maximum(h_plus[:-1], 0.))

        if not (self.dividends is None):
            # we have a minus sign because costs are deducted from PnL
            cost -= np.sum(self.dividends.current_value * h_plus[:-1])

        return cost


class StocksHoldingCost(HoldingCost):
    """A simplified version of :class:`HoldingCost` for the holding cost of stocks.
    
    This implements the simple model describe at page 11 of the book, *i.e.*
    the cost (in terms of the post-trade dollar positions):
    
    .. math::
 
        s^T_t {(h^+_t)}_-
    
    This class is a specialized version of :class:`HoldingCost`, and you should
    read its documentation for all details. Here we
    drop most of the parameters and use the default values explained above.
    We use a default value of :math:`5\%` annualized borrowing
    fee which is a rough (optimistic) approximation of the cost
    of shorting liquid US stocks. This cost is included **by default**
    in :class:`StockMarketSimulator`, the market simulator specialized
    to US (liquid) stocks.

    :param short_fees: Same as in :class:`HoldingCost`.
    :type short_fees: float, pd.Series, pd.DataFrame or None
    """

    def __init__(self, short_fees=5):

        super().__init__(short_fees=short_fees)


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
            volume_est = past_volumes.iloc[-windowvolume:].mean().values + 1E-8

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

        return result

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
