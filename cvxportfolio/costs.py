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

Currently these are two: :class:`StocksTransactionCost` and
:class:`StocksHoldingCost`.

The default parameters are chosen to approximate real costs for the stock
market as well as possible.
"""

import copy
from numbers import Number

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import (CostInequalityConstraint, EqualityConstraint,
                          InequalityConstraint)
from .errors import ConvexityError, ConvexSpecificationError
from .estimator import CvxpyExpressionEstimator, DataEstimator
from .hyperparameters import HyperParameter
from .utils import (average_periods_per_year,
                    periods_per_year_from_datetime_index, resample_returns)

__all__ = ["HoldingCost", "TransactionCost", "SoftConstraint",
           "StocksTransactionCost", "StocksHoldingCost", "TcostModel",
           "HcostModel"]


class Cost(CvxpyExpressionEstimator): # pylint: disable=abstract-method
    """Base class for cost objects (and also risks).

    Here there is some logic used to implement the algebraic operations.
    See also :class:`CombinedCost`.
    """

    def __mul__(self, other):
        """Multiply by constant."""
        if isinstance(other, (Number, HyperParameter)):
            return CombinedCosts([self], [other])
        raise SyntaxError(
            "You can only multiply a cost instance by a scalar "
            + "or a HyperParameter instance.")

    def __add__(self, other):
        """Add cost expression to another cost expression.

        Idea is to create a new CombinedCost class that
        implements `compile_to_cvxpy` and values_in_time_recursive
        by summing over costs.
        """
        if isinstance(other, CombinedCosts):
            return other.__radd__(self)
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
        """Self <= other, return CostInequalityConstraint.

        For now we check here the type of "other" but it would be nicer
        to have CostInequalityConstraint's internal DataEstimator throw
        NotImplemented instead.
        """
        if isinstance(other, (Number, pd.Series)):
            return CostInequalityConstraint(self, other)
        return NotImplemented

    def __lt__(self, other):
        """Self < other."""
        raise SyntaxError(
            'Strict inequalities are not allowed in convex programs.')

    def __gt__(self, other):
        """Self > other."""
        raise SyntaxError(
            'Strict inequalities are not allowed in convex programs.')

    def __ge__(self, other):
        """Self >= other, return CostInequalityConstraint."""
        return (-self).__le__(other)


class CombinedCosts(Cost):
    """Algebraic combination of :class:`Cost` instances.

    :param costs: instances of :class:`Cost`
    :type costs: list
    :param multipliers: floats that multiply the ``costs``
    :type multipliers: list
    """

    def __init__(self, costs, multipliers):
        for cost in costs:
            if not isinstance(cost, Cost):
                raise SyntaxError(
                    "You can only sum cost instances to other cost instances.")
        self.costs = costs
        self.multipliers = multipliers
        # this is changed by WorstCaseRisk before compiling to Cvxpy
        self.do_convexity_check = True

    def __add__(self, other):
        """Add self to other (combined) cost."""
        if isinstance(other, CombinedCosts):
            return CombinedCosts(self.costs + other.costs,
                                 self.multipliers + other.multipliers)
        return CombinedCosts(self.costs + [other], self.multipliers + [1.0])

    def __radd__(self, other):
        """Add other (combined) cost to self."""
        if isinstance(other, CombinedCosts):
            return other + self # pragma: no cover
        return CombinedCosts([other] + self.costs, [1.0] + self.multipliers)

    def __mul__(self, other):
        """Multiply by constant."""
        return CombinedCosts(self.costs,
                             [el * other for el in self.multipliers])

    def __neg__(self):
        """Take negative of cost."""
        return self * -1

    def initialize_estimator_recursive(self, **kwargs):
        """Initialize iterating over constituent costs.

        :param kwargs:  All parameters passed to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        for el in self.costs:
            el.initialize_estimator_recursive(**kwargs)

    def finalize_estimator_recursive(self, **kwargs):
        """Finalize iterating over constituent costs.

        :param kwargs:  All parameters passed to :meth:`finalize_estimator`.
        :type kwargs: dict
        """
        for el in self.costs:
            el.finalize_estimator_recursive(**kwargs)

    def values_in_time_recursive(self, **kwargs):
        """Evaluate estimators by iterating over constituent costs.

        :param kwargs: All parameters passed to :meth:`values_in_time`.
        :type kwargs: dict
        """
        for el in self.costs:
            el.values_in_time_recursive(**kwargs)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost by iterating over constituent costs.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :raises cvxportfolio.errors.ConvexSpecificationError: If the compiled
            Cvxpy expression doesn't follow the disciplined convex programming
            rules.
        :raises cvxportfolio.errors.ConvexityError: If the compiled Cvxpy
            expression is not concave (since we maximize it).

        :returns: Cvxpy expression of the combined cost.
        :rtype: cvxpy.Expression
        """
        expression = 0
        for multiplier, cost in zip(self.multipliers, self.costs):
            add = (multiplier.current_value
                if hasattr(multiplier, 'current_value') else multiplier) *\
                    cost.compile_to_cvxpy(w_plus, z, w_plus_minus_w_bm)
            if not add.is_dcp():
                raise ConvexSpecificationError(cost * multiplier)
            if self.do_convexity_check and (not add.is_concave()):
                raise ConvexityError(cost * multiplier)
            if (not self.do_convexity_check) and add.is_concave():
                raise ConvexityError(-cost * multiplier)
            expression += add
        return expression

    def collect_hyperparameters(self):
        """Collect hyper-parameters in the combined cost.

        :returns: List of :class:`cvxportfolio.hyperparameters.HyperParameter`
            instances.
        :rtype: list
        """
        return sum([el.collect_hyperparameters() for el in self.costs], []) +\
            sum([el.collect_hyperparameters() for el in self.multipliers if
                hasattr(el, 'collect_hyperparameters')], [])

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

    def _copy_keeping_multipliers(self):
        """This method is used when creating MPO policies.

        We want to deepcopy the constituent cost objects, but not the
        multipliers (which can be symbolic HPs).

        We will probably re-factor this, or make it public.
        """
        return CombinedCosts(
            costs = [el._copy_keeping_multipliers()
                if hasattr(el, '_copy_keeping_multipliers')
                    else copy.deepcopy(el)
                        for el in self.costs],
            multipliers = self.multipliers)


class SoftConstraint(Cost):
    """Soft constraint cost.

    :param constraint: Cvxportfolio constraint instance whose violation
        we penalize.
    :type constraint: cvxportfolio.constraints.EqualityConstraint or
        cvxportfolio.constraints.InequalityConstraint
    """

    def __init__(self, constraint):
        self.constraint = constraint

    def compile_to_cvxpy( # pylint: disable=inconsistent-return-statements
        self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :raises SyntaxError: If the constraint is not a EqualityConstraint or
             InequalityConstraint.

        :return: Cvxpy expression of the soft constraint.
        :rtype: cvxpy.Expression
        """

        try:
            expr = (self.constraint._compile_constr_to_cvxpy(
                w_plus, z, w_plus_minus_w_bm) - self.constraint._rhs())

        except AttributeError as exc:
            raise SyntaxError(
                f"{self.__class__.__name__} can only be used with"
                " EqualityConstraint or InequalityConstraint instances."
                    ) from exc

        if isinstance(self.constraint, EqualityConstraint):
            return cp.sum(cp.abs(expr))
        if isinstance(self.constraint, InequalityConstraint):
            return cp.sum(cp.pos(expr))


def _annual_percent_to_per_period(value, ppy):
    """Transform annual percent to per-period return.

    :param value: Annual percent return.
    :type value: float
    :param ppy: Periods per year.
    :type ppy: int

    :returns: Per-period return.
    :rtype: float
    """
    return resample_returns(returns=value/100, periods=ppy)


class SimulatorCost:
    """Cost class that can be used by a MarketSimulator."""

    def simulate(self, *args, **kwargs):
        """Simulate cost, used by market simulator.

        Look at its invocation in ``MarketSimulator`` for its list of
        arguments.

        Cost classes that are meant to be used in the simulator
        should implement this.

        :param args: Positional arguments.
        :type args: tuple
        :param kwargs: Keyword arguments.
        :type kwargs: dict
        """
        raise NotImplementedError # pragma: no cover


class HoldingCost(Cost, SimulatorCost):
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
        (otherwise you'll get a :class:`MissingValueError` exception). If
        `None`, the term is ignored.
    :type short_fees: float, pd.Series, pd.DataFrame or None
    :param long_fees: Fees on long positions expressed as annual percentage;
        same convention as above applies.
    :type long_fees: float, pd.Series, pd.DataFrame or None
    :param dividends: Dividend rates per period. Dividends are already
        included in the market returns by the default data interface
        (based on Yahoo Finance "adjusted prices") and thus this parameter
        should not be used in normal circumstances.
    :type dividends: float, pd.Series, pd.DataFrame or None
    :param periods_per_year: How many trading periods are there in a year, for
        example 252 (for trading days in the US). This is
        only relevant when using this class as part of a trading policy. If
        you leave this to `None` the following happens.
        The value of periods per year are estimated at each period by looking
        at  the past market returns at that point in time: the number of past
        periods is divided by the timestamp of the most recent period minus
        the timestamp of the first period (in years). That works well in most
        cases where there is enough history (say, a few years) and saves the
        user from having to manually enter this.
        If instead you use this object
        as a cost in a market simulator the parameter has no effect. (The
        length of each trading period in the simulation is known and so the
        per-period rates are evaluated exactly. For example, the rate over a
        weekend will be higher than overnight.)
    :type periods_per_year: float or None
    """

    def __init__(self, short_fees=None, long_fees=None, dividends=None,
                 periods_per_year=None):

        self.short_fees = None if short_fees is None else DataEstimator(
            short_fees)
        self.long_fees = None if long_fees is None else DataEstimator(
            long_fees)
        self.dividends = None if dividends is None else DataEstimator(
            dividends)
        self.periods_per_year = periods_per_year

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize cvxpy parameters.

        We don't use the parameter from
        :class:`cvxportfolio.estimator.DataEstimator` because we need to
        divide the value by periods_per_year.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """

        if self.short_fees is not None:
            self._short_fees_parameter = cp.Parameter(len(universe) - 1,
                nonneg=True)

        if self.long_fees is not None:
            self._long_fees_parameter = cp.Parameter(len(universe) - 1,
                nonneg=True)

        if self.dividends is not None:
            self._dividends_parameter = cp.Parameter(len(universe) - 1)

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Update cvxpy parameters.

        We compute the estimate of periods per year from past returns
        (if not provided by the user) and populate the cvxpy parameters
        with the current values of the user-provided data, transformed
        to per-period.

        :param past_returns: Past market returns (includes cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Other unused arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        if not ((self.short_fees is None)
                and (self.long_fees is None)
                and (self.dividends is None)):
            ppy = periods_per_year_from_datetime_index(
                past_returns.index) if self.periods_per_year is None else \
                    self.periods_per_year

        if self.short_fees is not None:
            self._short_fees_parameter.value = np.ones(
                past_returns.shape[1]-1) * _annual_percent_to_per_period(
                    self.short_fees.current_value, ppy)

        if self.long_fees is not None:
            self._long_fees_parameter.value = np.ones(
                past_returns.shape[1]-1) * _annual_percent_to_per_period(
                    self.long_fees.current_value, ppy)

        if self.dividends is not None:
            self._dividends_parameter.value =\
                np.ones(past_returns.shape[1]-1) * self.dividends.current_value

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression.
        :rtype: cvxpy.expression
        """

        expression = 0.

        if self.short_fees is not None:
            expression += self._short_fees_parameter.T @ cp.neg(w_plus[:-1])

        if self.long_fees is not None:
            expression += self._long_fees_parameter.T @ cp.pos(w_plus[:-1])

        if self.dividends is not None:
            # we have a minus sign because costs are deducted from PnL
            expression -= self._dividends_parameter.T @ w_plus[:-1]

        assert expression.is_convex()

        return expression

    def simulate(self, t, h_plus, t_next, **kwargs):
        """Simulate cost in a MarketSimulator.

        TODO: make sure simulator cost sign convention is
            the same as optimization cost! OK.
        TODO: make sure DataEstimator returns np.array of correct size! ~OK
        TODO: make sure simulator cost estimators are recursively evaluated!
            ~OK

        :param t: Current time.
        :type t: pandas.Timestamp
        :param h_plus: Post-trade holdings.
        :type h_plus: numpy.array or pandas.Series
        :param t_next: Next period's time.
        :type t_next: pandas.Timestamp
        :param kwargs: Unused arguments to
            :meth:`cvxportfolio.costs.SimulatorCost.simulate`.
        :type kwargs: dict

        :returns: Cost in units of value, positive (is subtracted from cash
            account).
        :rtype: float
        """

        year_divided_by_period = pd.Timedelta('365.24d') / (t_next - t)

        cost = 0.

        # TODO this is a temporary fix,
        # we should plug this into a recursive tree
        for est in [self.short_fees, self.long_fees, self.dividends]:
            if est is not None:
                est.initialize_estimator_recursive(universe=h_plus.index,
                                              trading_calendar=[t])
                est.values_in_time_recursive(t=t)

        if self.short_fees is not None:
            cost += np.sum(_annual_percent_to_per_period(
                self.short_fees.current_value, year_divided_by_period) * (
                    -np.minimum(h_plus[:-1], 0.)))

        if self.long_fees is not None:
            cost += np.sum(_annual_percent_to_per_period(
                self.long_fees.current_value,
                year_divided_by_period) * np.maximum(h_plus[:-1], 0.))

        if self.dividends is not None:
            # we have a minus sign because costs are deducted from PnL
            cost -= np.sum(self.dividends.current_value * h_plus[:-1])

        return cost


class StocksHoldingCost(HoldingCost, SimulatorCost):
    r"""Holding cost specialized to stocks.

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


class TransactionCost(Cost):
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

        # these are overwritten by parameters defined below
        self.first_term_multiplier = None
        self.second_term_multiplier = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize cvxpy parameters.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        if self.a is not None or self.pershare_cost is not None:
            self.first_term_multiplier = cp.Parameter(
                len(universe)-1, nonneg=True)
        if self.b is not None:
            self.second_term_multiplier = cp.Parameter(
                len(universe)-1, nonneg=True)

    def values_in_time( # pylint: disable=arguments-differ
            self, current_portfolio_value, past_returns, past_volumes,
            current_prices, **kwargs):
        """Update cvxpy parameters.

        :raises SyntaxError: If the prices are missing from the market data.

        :param current_portfolio_value: Current total value of the portfolio.
        :type current_portfolio_value: float
        :param past_returns: Past market returns (includes cash).
        :type past_returns: pandas.DataFrame
        :param past_volumes: Past market volumes.
        :type past_volumes: pandas.DataFrame
        :param current_prices: Current open prices.
        :type current_prices: pandas.Series
        :param kwargs: Other unused arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        tmp = 0.

        if self.a is not None:
            _ = self.a.current_value
            tmp += _ *\
                np.ones(past_returns.shape[1]-1) if np.isscalar(_) else _
        if self.pershare_cost is not None:
            if current_prices is None:
                raise SyntaxError("If you don't provide prices you",
                                  " should set pershare_cost to None")
            assert not np.any(current_prices.isnull())
            # assert not np.any(current_prices == 0.)
            tmp += self.pershare_cost.current_value / current_prices.values

        if self.a is not None or self.pershare_cost is not None:
            self.first_term_multiplier.value = tmp

        if self.b is not None:

            if (self.window_sigma_est is None) or\
                    (self.window_volume_est is None):
                ppy = periods_per_year_from_datetime_index(past_returns.index)
            windowsigma = ppy if (
                self.window_sigma_est is None) else self.window_sigma_est
            windowvolume = ppy if (
                self.window_volume_est is None) else self.window_volume_est

            # TODO refactor this with forecast.py logic
            sigma_est = np.sqrt(
                (past_returns.iloc[-windowsigma:, :-1]**2).mean()).values
            volume_est = past_volumes.iloc[-windowvolume:].mean().values + 1E-8

            self.second_term_multiplier.value =\
                self.b.current_value * sigma_est * (current_portfolio_value /
                     volume_est) ** (
                         (2 if self.exponent is None else self.exponent) - 1)

    def simulate(
            self, t, u, past_returns, current_returns, current_volumes,
            current_prices, **kwargs):
        """Simulate transaction cost in cash units.

        :raises SyntaxError: If the market returns are not available in the
            market data.

        :param t: Current timestamp.
        :type t: pandas.Timestamp
        :param u: Trades vector.
        :type u: pandas.Series
        :param past_returns: Dataframe of past market returns.
        :type past_returns: pandas.DataFrame
        :param current_returns: Current period's market returns.
        :type current_returns: pandas.Series
        :param current_volumes: Current market volumes.
        :type current_volumes: pandas.Series or None
        :param current_prices: Current market prices.
        :type current_prices: pandas.Series or None
        :param kwargs: Unused arguments passed by :class:`MarketSimulator`.
        :type kwargs: dict

        :returns: Transaction cost for this period in cash units.
        :rtype: float
        """

        result = 0.
        if self.pershare_cost is not None:
            if current_prices is None:
                raise SyntaxError(
                    "If you don't provide prices you should"
                    " set pershare_cost to None")
            result += self.pershare_cost.values_in_time_recursive(t=t) * int(
                sum(np.abs(u.iloc[:-1] + 1E-6) / current_prices.values))

        if self.a is not None:
            result += sum(self.a.values_in_time_recursive(t=t)
                          * np.abs(u.iloc[:-1]))

        if self.b is not None:

            if self.window_sigma_est is None:
                windowsigma = average_periods_per_year(
                    num_periods=len(past_returns)+1,
                    first_time=past_returns.index[0], last_time=t)
            else:
                windowsigma = self.window_sigma_est

            exponent = (1.5 if self.exponent is None else self.exponent)

            sigma = np.std(pd.concat(
                [past_returns.iloc[-windowsigma + 1:, :-1],
                pd.DataFrame(current_returns.iloc[:-1]).T], axis=0), axis=0)
            if current_volumes is None:
                raise SyntaxError(
                    "If you don't provide volumes you should set b to None"
                    f" in the {self.__class__.__name__} simulator cost")
            # we add 1E-8 to the volumes to prevent 0 volumes error
            # (trades are cancelled on 0 volumes)
            result += (np.abs(u.iloc[:-1])**exponent) @ (
                self.b.values_in_time_recursive(t=t) *
                sigma / ((current_volumes + 1E-8) ** (
                exponent - 1)))

        assert not np.isnan(result)
        assert not np.isinf(result)

        return result

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression.

        :param w_plus: Post-trade weights.
        :type w_plus: cvxpy.Variable
        :param z: Trade weights.
        :type z: cvxpy.Variable
        :param w_plus_minus_w_bm: Post-trade weights minus benchmark
            weights.
        :type w_plus_minus_w_bm: cvxpy.Variable

        :returns: Cvxpy expression.
        :rtype: cvxpy.expression
        """
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

    See pages 10-11 in
    `the book <https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf>`_.
    We don't include the short-term alpha term `c` here because it
    can be expressed with a separate `ReturnsForecast` object.

    :param a: linear cost, which multiplies the absolute value of each trade.
        This can model (half) the bid-ask spread, or any fee linear in the size
        of a trade.
    :type a: float or pd.Series or pd.DataFrame
    :param pershare_cost: per-share trade cost, amount of dollars paid for
         each share traded.
    :type pershare_cost: float or pd.Series or pd.DataFrame
    :param b: coefficient of the non-linear term of the transaction cost model,
        which multiplies the estimated volatility for each stock.
    :type b: float or pd.Series or pd.DataFrame
    :param window_sigma_est: we use an historical rolling standard deviation to
        estimate the average size of the return on a stock on each day, and
        this multiplies the second term of the transaction cost model.  See the
        paper for an explanation of the model. Here you specify the length of
        the rolling window to use. If None (the default) it uses a length of 1
        year (approximated with the data provided).
    :type window_sigma_est: int or None
    :param window_volume_est: length of the window for the mean of past volumes
        used as estimate of each period's volume. Has no effect on the
        simulator version of this which uses the actual volume. If None (the
        default) it uses a length of 1 year (approximated with the data
        provided).
    :type window_volume_est: int
    :param exponent: exponent of the non-linear term, defaults (if set to
        ``None``) to 1.5 for the simulator version, and 2 for the optimization
        version (because it is more efficient numerically and the difference is
        small, you can change it if you want).
    :type exponent: float or None
    """

    def __init__(self, a=0., pershare_cost=0.005, b=1.0, window_sigma_est=None,
                 window_volume_est=None, exponent=1.5):

        super().__init__(a=a, pershare_cost=pershare_cost, b=b,
                         window_sigma_est=window_sigma_est,
                         window_volume_est=window_volume_est,
                         exponent=exponent)

# Aliases

class TcostModel(TransactionCost):
    """Alias of :class:`TransactionCost`.

    As it was defined originally in :paper:`section 6.1 <section.6.1>` of the
    paper.
    """

class HcostModel(HoldingCost):
    """Alias of :class:`HoldingCost`.

    As it was defined originally in :paper:`section 6.1 <section.6.1>` of the
    paper.
    """
