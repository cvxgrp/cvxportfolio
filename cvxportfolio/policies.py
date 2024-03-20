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
"""This module contains trading policies that can be back-tested."""

import copy
import logging
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from .errors import (ConvexityError, ConvexSpecificationError, DataError,
                     MissingTimesError, PortfolioOptimizationError)
from .estimator import DataEstimator, Estimator
from .forecast import HistoricalMeanVolume
from .returns import CashReturn
from .utils import flatten_heterogeneous_list

__all__ = [
    "AllCash",
    "Hold",
    "SellAll",
    "FixedTrades",
    "PeriodicRebalance",
    "ProportionalRebalance",
    "AdaptiveRebalance",
    "SinglePeriodOptimization",
    "MultiPeriodOptimization",
    "MarketBenchmark",
    "ProportionalTradeToTargets",
    "RankAndLongShort",
    "FixedWeights",
    "Uniform",
    "SinglePeriodOpt",
    "MultiPeriodOpt",
]

logger = logging.getLogger(__name__)

class Policy(Estimator):
    """Base trading policy class, defines execute method."""

    def execute(self, h, market_data, t=None):
        """Execute trading policy at current or user-specified time.

        Return the (*e.g.*, dollar) trade vector :math:`u`, the timestamp
        of execution (for double check in case you don't pass it), and a Pandas
        Series of the number of shares to trade, if you pass a Market
        Data server which provides open prices (or None).

        :param h: Holdings vector, in dollars, including the cash account
            (the last element).
        :type h: pandas.Series
        :param market_data: :class:`MarketData` instance used to provide
            data to the policy
        :type market_data: cvxportfolio.MarketData instance
        :param t: Time at which we execute. If None (the default), the
            last timestamp in the trading calendar provided by the
            :class:`MarketData` instance is used. Note: if you use a default
            market data server, you probably want to set their ``online_usage``
            argument to ``True``.
        :type t: pandas.Timestamp or None

        :raises cvxportfolio.errors.DataError: Holdings vector sum to a
            negative value or don't match the market data server's universe.

        :returns: u, t, shares_traded
        :rtype: pandas.Series, pandas.Timestamp, pandas.Series
        """

        trading_calendar = market_data.trading_calendar()

        if t is None:
            t = trading_calendar[-1]

        if not t in trading_calendar:
            raise ValueError(f'Provided time {t} must be in the '
            + 'trading calendar implied by the market data server.')

        v = np.sum(h)

        if v < 0.:
            raise DataError(
                f"Holdings provided to {self.__class__.__name__}.execute "
                + " have negative sum.")

        past_returns, _, past_volumes, _, current_prices = market_data.serve(t)

        if sorted(h.index) != sorted(past_returns.columns):
            raise DataError(
                "Holdings provided don't match the universe"
                " implied by the market data server.")

        h = h[past_returns.columns]
        w = h / v

        # consider adding caching logic here
        self.initialize_estimator_recursive(
            universe=past_returns.columns,
            trading_calendar=trading_calendar[trading_calendar >= t])

        w_plus = self.values_in_time_recursive(
            t=t, past_returns=past_returns, past_volumes=past_volumes,
            current_weights=w, current_portfolio_value=v,
            current_prices=current_prices)

        # this could be optional, currently unused)
        self.finalize_estimator_recursive()

        z = w_plus - w
        u = z * v

        if current_prices is not None:
            shares_traded =  pd.Series(np.round(u.iloc[:-1] / current_prices),
                dtype=int)
        else:
            shares_traded = None

        return u, t, shares_traded


class Hold(Policy):
    """Hold initial portfolio, don't trade."""

    def values_in_time( # pylint: disable=arguments-differ
            self, current_weights, **kwargs):
        """Return current_weights.

        :param current_weights: Current weights.
        :type current_weights: pandas.Series
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Same current weights.
        :rtype: pandas.Series
        """
        return current_weights

class AllCash(Policy):
    """Allocate all weight to cash.

    This is the default benchmark used in :class:`SinglePeriodOptimization` and
    :class:`MultiPeriodOptimization` policies.
    """

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Return all cash weights.

        :param past_returns: Past market returns (used to infer universe).
        :type past_returns: pandas.DataFrame
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: All cash weights.
        :rtype: pandas.Series
        """
        result = pd.Series(0., past_returns.columns)
        result.iloc[-1] = 1.
        return result

class MarketBenchmark(Policy):
    """Allocation weighted by last year's average market traded volumes.

    This policy provides an approximation of a market capitalization-weighted
    allocation, by using the average of traded volumes in units of value (e.g.,
    USDOLLAR) over the previous year as proxy.

    .. versionadded:: 1.2.0

        We added the ``mean_volume_forecast`` parameter.

    :param mean_volume_forecast: Forecaster class that computes the average
        of historical volumes. You can also pass a DataFrame containing
        your own forecasts computed externally. Default is
        :class:`cvxportfolio.forecast.HistoricalMeanVolume` which is
        instantiated with parameter ``rolling=pd.Timedelta('365.24d')``
        (that's one solar year in number of days). If you
        want to provide a different forecaster, or change the parameters (like
        adding exponential smoothing) you should instantiate the forecaster
        class and pass the instance.
    :type mean_volume_forecast: pandas.DataFrame, cvx.forecast.BaseForecast
        class or instance
    """

    def __init__(self, mean_volume_forecast=HistoricalMeanVolume):
        if isinstance(mean_volume_forecast, type):
            mean_volume_forecast = mean_volume_forecast(
                rolling=pd.Timedelta('365.24d'))
        self.mean_volume_forecast = DataEstimator(
            mean_volume_forecast, data_includes_cash=False)

    def values_in_time( # pylint: disable=arguments-differ
            self, past_returns, **kwargs):
        """Return market benchmark weights.

        :param past_returns: Past market returns (used to infer universe with
            cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :raises cvxportfolio.errors.DataError: Market data does not include
            market volumes.

        :returns: Market benchmark weights.
        :rtype: pandas.Series
        """

        meanvolumes = self.mean_volume_forecast.current_value
        result = np.zeros(len(meanvolumes) + 1)
        result[:-1] = meanvolumes / sum(meanvolumes)
        return pd.Series(result, index=past_returns.columns)

class RankAndLongShort(Policy):
    """Rank assets by signal; long highest and short lowest.

    :param signal: time-indexed DataFrame of signal for all symbols
        excluding cash. At each point in time the num_long assets with
        highest signal will have equal positive weight, and the
        num_short assets with lower signal will have equal negative
        weight. If two or more assets have the same signal value and
        they are on the boundary of either the top or bottom set,
        alphanumerical ranking will prevail.
    :type signal: pd.DataFrame
    :param num_long: number of assets to long, default 1; if specified
        as Series it must be indexed by time.
    :type num_long: int or pd.Series
    :param num_short: Number of assets to short, default 1; if specified
        as Series it must be indexed by time.
    :type num_short: int or pd.Series)
    :param target_leverage: leverage of the resulting portfolio, default
        1; if specified as Series it must be indexed by time.
    :type target_leverage: float or pd.Series
    """

    def __init__(self, signal, num_long=1, num_short=1, target_leverage=1.):
        """Define sub-estimators at class attribute level."""
        self.num_long = DataEstimator(num_long)
        self.num_short = DataEstimator(num_short)
        self.signal = DataEstimator(signal)
        self.target_leverage = DataEstimator(target_leverage)

    def values_in_time( # pylint: disable=arguments-differ
            self, current_weights, **kwargs):
        """Get allocation weights.

        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Rank and long-short weights.
        :rtype: pandas.Series
        """

        sorted_ret = pd.Series(
            self.signal.current_value, current_weights.index[:-1]
        ).sort_values()
        short_positions = sorted_ret.index[: self.num_short.current_value]
        long_positions = sorted_ret.index[-self.num_long.current_value:]

        target_weights = pd.Series(0., index=current_weights.index)
        target_weights[short_positions] = -1.
        target_weights[long_positions] = 1.

        target_weights /= sum(abs(target_weights))
        target_weights *= self.target_leverage.current_value

        # cash is always 1.
        target_weights[current_weights.index[-1]] = 1.

        return target_weights


class ProportionalTradeToTargets(Policy):
    """Trade in equal proportion to match target weights in time.

    Initially, it loads the list of trading days and so at each day it
    knows how many are missing before the next target's day, and trades
    in equal proportions to reach those targets. If there are no targets
    remaining it defaults to not trading.

    :param targets: time-indexed DataFrame of target weight vectors at
        given points in time (e.g., start of each month).
    :type targets: pandas.DataFrame
    """

    def __init__(self, targets):
        self.targets = targets
        self.trading_days = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, trading_calendar, **kwargs):
        """Initialize policy instance with updated trading_calendar.

        :param trading_calendar: Future (including current) trading calendar.
        :type trading_calendar: pandas.DatetimeIndex
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        self.trading_days = trading_calendar

    def values_in_time( # pylint: disable=arguments-differ
            self, t, current_weights, **kwargs):
        """Get current allocation weights.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :raises ValueError: Some target weights provided do not sum to 1.

        :returns: Allocation weights.
        :rtype: pandas.Series
        """

        next_targets = self.targets.loc[self.targets.index >= t]
        if not np.allclose(next_targets.sum(1), 1.):
            raise ValueError(
                f"The target weights provided to {self.__class__.__name__} at"
                + f" time {t} do not sum to 1.")
        if len(next_targets) == 0:
            return current_weights
        next_target = next_targets.iloc[0]
        next_target_day = next_targets.index[0]
        trading_days_to_target = len(self.trading_days[(
            self.trading_days >= t) & (self.trading_days < next_target_day)])
        if trading_days_to_target == 0:
            return current_weights
        return current_weights + (
            next_target - current_weights) / trading_days_to_target


class SellAll(AllCash):
    """Sell all assets to cash.

    Alias of :class:`AllCash`.
    """


class FixedTrades(Policy):
    """Each day trade the provided trade weights vector.

    If there are no weights defined for the given day, default to no
    trades.

    :param trades_weights: target trade weights :math:`z_t` to trade at each
        period. If constant in time use a pandas Series indexed by the assets'
        names, including the cash account name (``cash_key`` option to
        :class:`MarketSimulator`). If varying in time, use a pandas DataFrame
        with datetime index and as columns the assets names including cash.
        If a certain time in the backtest is not present in the data provided
        the policy defaults to not trading in that period.
    :type trades_weights: pd.Series or pd.DataFrame
    """

    def __init__(self, trades_weights):
        """Trade fixed trade weigths in time."""
        self.trades_weights = DataEstimator(
            trades_weights, data_includes_cash=True)

    def values_in_time_recursive( # pylint: disable=arguments-differ
            self, t, current_weights, **kwargs):
        """Get current allocation weights.

        We redefine the recursive version of :meth:`values_in_time` because
        we catch an exception thrown by a sub-estimator.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Allocation weights.
        :rtype: pandas.Series
        """
        try:
            super().values_in_time_recursive(
                t=t, current_weights=current_weights, **kwargs)
            result = current_weights + pd.Series(
                self.trades_weights.current_value, current_weights.index)
        except MissingTimesError:
            logger.info("%s didn't trade at time %s because it couldn't find"
                + " trade weights among the provided ones.",
                self.__class__.__name__, t)
            result = current_weights
        self._current_value = result
        return result


class FixedWeights(Policy):
    """Each day trade to the provided trade weights vector.

    If there are no weights defined for the given day, default to no
    trades.

    :param target_weights: target weights :math:`w_t^+` to trade to at each
        period. If constant in time use a pandas Series indexed by the assets'
        names, including the cash account name (``cash_key`` option
        to the simulator). If varying in time, use a pandas DataFrame
        with datetime index and as columns the assets names including cash.
        If a certain time in the backtest is not present in the data provided
        the policy defaults to not trading in that period.
    :type target_weights: pd.Series or pd.DataFrame
    """

    def __init__(self, target_weights):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        self.target_weights = DataEstimator(
            target_weights, data_includes_cash=True)

    def values_in_time_recursive( # pylint: disable=arguments-differ
            self, t, current_weights, **kwargs):
        """Get current allocation weights.

        We redefine the recursive version of :meth:`values_in_time` because
        we catch an exception thrown by a sub-estimator.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param kwargs: Unused arguments to :meth:`values_in_time_recursive`.
        :type kwargs: dict

        :returns: Allocation weights.
        :rtype: pandas.Series
        """
        try:
            super().values_in_time_recursive(
                t=t, current_weights=current_weights, **kwargs)
            result = current_weights + pd.Series(
                self.target_weights.current_value, current_weights.index
                ) - current_weights
        except MissingTimesError:
            logger.info("%s didn't trade at time %s because it couldn't find"
                + " target weights among the provided ones.",
                self.__class__.__name__, t)
            result = current_weights
        self._current_value = result
        return result


class Uniform(FixedWeights):
    """Uniform allocation on non-cash assets.

    :param leverage: Leverage of the allocation.
    :type leverage: float
    """

    # pylint: disable=super-init-not-called
    def __init__(self, leverage=1.):
        self.leverage = leverage
        # then we re-define the target weights for each universe in the method
        # below, so we provide the same interface to the parent class'
        # values_in_time_recursive
        self.target_weights = None

    def initialize_estimator( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize this estimator.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Other unused arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """
        target_weights = pd.Series(1., universe)
        target_weights.iloc[-1] = 0
        target_weights /= sum(target_weights)
        target_weights *= self.leverage
        target_weights.iloc[-1] = 1. - target_weights.sum()
        self.target_weights = DataEstimator(target_weights)


class PeriodicRebalance(FixedWeights):
    """Track a target weight vector rebalancing at given times.

    This calls `FixedWeights`. If you want to change the target in time
    use that policy directly.

    :param target: Allocation weights to rebalance to.
    :type target: pandas.Series
    :param rebalancing_times: Times at which we rebalance.
    :type rebalancing_times: pandas.DateTimeIndex):
    """

    def __init__(self, target, rebalancing_times):
        target_weights = pd.DataFrame(
            {el: target for el in rebalancing_times}).T
        super().__init__(target_weights)


class ProportionalRebalance(ProportionalTradeToTargets):
    """Trade proportionally in time to track fixed target weights at times.

    This calls `ProportionalTradeToTargets`. If you want to change
    the target in time use that policy directly.

    :param target: Allocation weights to rebalance to.
    :type target: pandas.Series
    :param rebalancing_times: Times at which we rebalance.
    :type rebalancing_times: pandas.DateTimeIndex):
    """

    def __init__(self, target, target_matching_times):
        targets = pd.DataFrame({el: target for el in target_matching_times}).T
        super().__init__(targets)


class AdaptiveRebalance(Policy):
    """Rebalance portfolio when deviates too far from target.

    We use the 2-norm as trigger for rebalance. You may want to
    calibrate ``tracking_error`` for your application
    by backtesting this policy, *e.g.*, to get your desired turnover.

    :param target: target weights to rebalance to.
        It is assumed a constant if it is a Series. If it varies in
        time (you must specify it for every trading day) pass a
        DataFrame indexed by time.
    :type target: pd.Series or pd.DataFrame
    :param tracking_error: we trade to match the target
        weights whenever the 2-norm of our weights minus the
        target is larger than this. Pass a Series if you want to vary it in
        time.
    :type tracking_error: pd.Series or pd.DataFrame
    """

    def __init__(self, target, tracking_error):
        self.target = DataEstimator(target)
        self.tracking_error = DataEstimator(tracking_error)

    def values_in_time( # pylint: disable=arguments-differ
            self, current_weights, **kwargs):
        """Get target allocation weights.

        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param kwargs: Unused arguments to :meth:`values_in_time`.
        :type kwargs: dict

        :returns: Allocation weights.
        :rtype: pandas.Series
        """
        if np.linalg.norm(current_weights - self.target.current_value) >\
                self.tracking_error.current_value:
            return self.target.current_value
        return current_weights

# pylint: disable=too-many-instance-attributes
class MultiPeriodOptimization(Policy):
    r"""Multi Period Optimization policy.

    Implements the model developed in :paper:`chapter 5, page 49 <section.5.2>`
    of the paper.
    You specify the objective terms
    using classes such as ReturnsForecast and TcostModel, each
    multiplied by its multiplier. You also specify lists
    of constraints. There are two ways to do it. You either
    define the same objective terms and costraints for each
    step of the multi-period problem, or you define a different
    objective term and different list of constraints for each step.
    In addition we offer a `terminal_constraint` argument to
    simply impose that at the last step in the optimization the
    post-trade weights match the given weights (see page 51).

    When it computes the trajectory of weights for the future
    it only returns the first step (to the Simulator, typically).
    The future steps (planning horizon) are by default not returned.

    :param objective: These will be maximized;
        if you pass a single expression of Cost it is understood as the
        same for all steps; if it's a list you must also pass a list of lists
        for `constraints`, each term represents the cost for each step of the
        optimization (starting from the first, i.e., today) and the length of
        the list is used as planning_horizon (the value you pass there will be
        ignored)
    :type objective: algebra of Cost or list of
    :param constraints: These will be
        imposed on the optimization. Default []. Pass this as a list of
        lists of the same length as `objective` to specify different
        constraints at different time steps.
    :type constraints: list of Constraints or list of those
    :param planning_horizon:  How many steps in the future we
        plan for. Ignored if passing `objective` and `constraints` as lists.
        Default is None.
    :type planning_horizon: int or None
    :param terminal_constraint: If you pass a Series to this
        (default is None) it will impose that at the last step of the multi
        period optimization the post-trade weights are equal to this.
    :type terminal_constraint: pd.Series or None
    :param include_cash_return: Whether to automatically include the
        ``CashReturn`` term in the objective, with default parameters.
        Default is ``True``.
    :type include_cash_return: bool
    :param benchmark: Benchmark weights to use in the risk model and
        other terms that need it. You can use any policy here. Suggested ones
        ones are ``AllCash``, the default, ``Uniform``
        (uniform allocation on non-cash assets),
        and ``MarketBenchmark``, which approximates the market-weighted
        portfolio.
    :type benchmark: :class:`Policy` class or instance
    :param kwargs: Any extra argument will be passed to cvxpy.Problem.solve,
        so you can choose a solver and pass parameters to it.
    :type kwargs: dict

    :raises SyntaxError: If the format of provided objective and constraints
        is not right.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, objective, constraints=(), include_cash_return=True,
        planning_horizon=None, terminal_constraint=None,
        benchmark=AllCash, **kwargs):
        if hasattr(objective, '__iter__'):
            if not (hasattr(constraints, '__iter__') and len(constraints
                    ) and (hasattr(constraints[0], '__iter__') and len(
                    objective) == len(constraints))):
                raise SyntaxError(
                    'If you pass the objective as a list, constraints should'
                    + ' be a list of lists of the same length.')
            self._planning_horizon = len(objective)
            self.objective = objective
            self.constraints = constraints
        else:
            if not np.isscalar(planning_horizon):
                raise SyntaxError(
                    'If `objective` and `constraints` are the same for '
                    + 'all steps you must specify `planning_horizon`.')
            self._planning_horizon = planning_horizon
            self.objective = [objective.copy_keeping_multipliers()
                if hasattr(objective, 'copy_keeping_multipliers')
                    else copy.deepcopy(objective) for i in range(
                    planning_horizon)] if planning_horizon > 1 else [objective]
            self.constraints = [copy.deepcopy(constraints) for i in range(
                planning_horizon)] if planning_horizon > 1 else [constraints]

        self._include_cash_return = include_cash_return
        if self._include_cash_return:
            self.objective = [el + CashReturn() for el in self.objective]
        self.terminal_constraint = terminal_constraint

        if isinstance(benchmark, (pd.DataFrame, pd.Series)):
            self.benchmark = DataEstimator(benchmark, data_includes_cash=True)
        else:
            self.benchmark = benchmark() if isinstance(benchmark, type
                ) else benchmark

        self.cvxpy_kwargs = kwargs

        # redefined below
        self._cvxpy_objective = 0
        self._cvxpy_constraints = []
        self._problem = None
        self._w_bm = None
        self._w_current = None
        self._z_at_lags = None
        self._w_plus_at_lags = None
        self._w_plus_minus_w_bm_at_lags = None
        self._cache = {}

        # for recursive evaluation
        self.__subestimators__ = tuple(
            [self.benchmark] + self.objective + sum(
                [list(con_at_lag) for con_at_lag in self.constraints], []))

    def _compile_to_cvxpy(self):
        """Compile all cvxpy expressions and the problem."""
        self._cvxpy_objective = [
            el.compile_to_cvxpy(
                self._w_plus_at_lags[i], self._z_at_lags[i],
                self._w_plus_minus_w_bm_at_lags[i])
            for i, el in enumerate(self.objective)]
        for el, term in zip(self.objective, self._cvxpy_objective):
            if not term.is_dcp():
                raise ConvexSpecificationError(el)
            if not term.is_concave():
                raise ConvexityError(el)
        self._cvxpy_objective = sum(self._cvxpy_objective)

        def _compile_and_check_constraint(constr, i):
            result = constr.compile_to_cvxpy(
                self._w_plus_at_lags[i], self._z_at_lags[i],
                self._w_plus_minus_w_bm_at_lags[i])
            for el in (result if hasattr(result, '__iter__') else [result]):
                if not el.is_dcp():
                    raise ConvexSpecificationError(constr)
            return result

        self._cvxpy_constraints = [
            flatten_heterogeneous_list([
                _compile_and_check_constraint(constr, i) for constr in el])
            for i, el in enumerate(self.constraints)]

        self._cvxpy_constraints = sum(self._cvxpy_constraints, [])
        self._cvxpy_constraints += [cp.sum(z) == 0 for z in self._z_at_lags]
        w = self._w_current
        for i in range(self._planning_horizon):
            self._cvxpy_constraints.append(
                self._w_plus_at_lags[i] == self._z_at_lags[i] + w)
            self._cvxpy_constraints.append(
                self._w_plus_at_lags[i] - self._w_bm == \
                    self._w_plus_minus_w_bm_at_lags[i])
            w = self._w_plus_at_lags[i]
        if not self.terminal_constraint is None:
            self._cvxpy_constraints.append(w == self.terminal_constraint)
        self._problem = cp.Problem(cp.Maximize(
            self._cvxpy_objective), self._cvxpy_constraints)
        if not self._problem.is_dcp():  # dpp=True)
            raise SyntaxError(
              f"The optimization problem compiled by {self.__class__.__name__}"
                + " does not follow the convex optimization rules."
                + " This should not happen if you're using the default "
                + " cvxportfolio terms and is probably due to a"
                + " mis-specified custom term.")

    def initialize_estimator_recursive( # pylint: disable=arguments-differ
            self, universe, **kwargs):
        """Initialize the policy object with the trading universe.

        We redefine the recursive version of :meth:`initialize_estimator`
        because we initialize all objective and constraint objects.

        :param universe: Trading universe, including cash.
        :type universe: pandas.Index
        :param kwargs: Arguments to :meth:`initialize_estimator`.
        :type kwargs: dict
        """

        super().initialize_estimator_recursive(universe=universe, **kwargs)

        self._w_bm = cp.Parameter(len(universe))

        self._w_current = cp.Parameter(len(universe))
        self._z_at_lags = [cp.Variable(len(universe))
                          for i in range(self._planning_horizon)]
        self._w_plus_at_lags = [cp.Variable(
            len(universe)) for i in range(self._planning_horizon)]
        self._w_plus_minus_w_bm_at_lags = [cp.Variable(
            len(universe)) for i in range(self._planning_horizon)]

        # simulator will overwrite this with cache loaded from disk
        self._cache = {}

        self._compile_to_cvxpy()

    def values_in_time_recursive( # pylint: disable=arguments-differ
            self, t, current_weights, current_portfolio_value, **kwargs):
        """Update all cvxpy parameters, solve, and return allocation weights.

        We redefine the recursive version of :meth:`values_in_time`
        because we evaluate all objective and constraint objects.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param current_weights: Current allocation weights.
        :type current_weights: pandas.Series
        :param current_portfolio_value: Current total value of the portfolio.
        :type current_portfolio_value: float
        :param kwargs: Other arguments to :meth:`values_in_time_recursive`.
        :type kwargs: dict

        :raises cvxportfolio.errors.DataError: Current portfolio value is
            negative.
        :raises cvxportfolio.errors.PortfolioOptimizationError: The portfolio
            optimization failed: either the problem was infeasible on
            unbounded, or a numerical error occurred. In the latter case the
            original Cvxpy exception is re-raised.

        :returns: Allocation weights.
        :rtype: pandas.Series
        """

        if not current_portfolio_value > 0:
            raise DataError(
                f"Policy {self.__class__.__name__} was evaluated at "
                + f"{t} with negative portfolio value.")
        assert np.isclose(sum(current_weights), 1)

        for i, obj in enumerate(self.objective):
            obj.values_in_time_recursive(
                t=t, current_weights=current_weights,
                current_portfolio_value=current_portfolio_value,
                mpo_step=i, cache=self._cache,
                **kwargs)

        for i, constr_at_lag in enumerate(self.constraints):
            for constr in constr_at_lag:
                constr.values_in_time_recursive(
                    t=t, current_weights=current_weights,
                    current_portfolio_value=current_portfolio_value,
                    mpo_step=i, cache=self._cache, **kwargs)

        self.benchmark.values_in_time_recursive(
            t=t, current_weights=current_weights,
            current_portfolio_value=current_portfolio_value,
            **kwargs)

        self._w_bm.value = np.array(self.benchmark.current_value.values)\
             if hasattr(self.benchmark.current_value, 'values'
            ) else np.array(self.benchmark.current_value)
        self._w_current.value = current_weights.values

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message='Solution may be inaccurate')
                # suppress cvxpy 1.4 ECOS deprecation warnings
                if cp.__version__[:3] == '1.4':
                    warnings.filterwarnings("ignore", category=FutureWarning)
                self._problem.solve(**self.cvxpy_kwargs)
        except cp.SolverError as exc:
            raise PortfolioOptimizationError(
                f"Numerical solver for policy {self.__class__.__name__} at"
                + f" time {t} failed; try changing it, relaxing some"
                + " constraints, or removing costs.") from exc

        if self._problem.status in ["unbounded", "unbounded_inaccurate"]:
            raise PortfolioOptimizationError(
                f"Policy {self.__class__.__name__} at time "
                + f"{t} resulted in an unbounded problem.")

        if self._problem.status in ["infeasible", 'infeasible_inaccurate']:
            raise PortfolioOptimizationError(
                f"Policy {self.__class__.__name__} at time "
                + f"{t} resulted in an infeasible problem.")

        result = current_weights + pd.Series(
            self._z_at_lags[0].value, current_weights.index)
        self._current_value = result
        return result

class SinglePeriodOptimization(MultiPeriodOptimization):
    r"""Single Period Optimization policy.

    Implements the model developed in :paper:`chapter 4, page 43 <section.4.8>`
    of the paper.
    You specify the objective terms
    using classes such as ReturnsForecast and TcostModel, each
    multiplied by its multiplier. You also specify lists
    of constraints.

    :param objective: This algebraic combination of cvxportfolio cost objects
        will be maximized
    :type objective: CombinedCost
    :param constraints: These will be imposed on the optimization. Default [].
    :type constraints: list of Constraints
    :param include_cash_return: Whether to automatically include the
        ``CashReturn`` term in the objective, with default parameters.
        Default is ``True``.
    :type include_cash_return: bool
    :param benchmark: Benchmark weights to use in the risk model and
        other terms that need it. You can use any policy here. Suggested ones
        ones are ``AllCash``, the default, ``Uniform``
        (uniform allocation on non-cash assets),
        and ``MarketBenchmark``, which approximates the market-weighted
        portfolio.
    :type benchmark: :class:`Policy` class or instance
    :param kwargs: Any extra argument will be passed to cvxpy.Problem.solve,
        so you can choose a solver and pass parameters to it.
    :type kwargs: dict
    """

    def __init__(self, objective, constraints=(),
                include_cash_return=True, benchmark=AllCash, **kwargs):
        super().__init__(
            [objective], [constraints],
            include_cash_return=include_cash_return,
            benchmark=benchmark, **kwargs)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'objective=' + str(self.objective[0]) \
            + ', constraints=' + str(self.constraints[0])\
            + ', benchmark=' + str(self.benchmark)\
            + ', cvxpy_kwargs=' + str(self.cvxpy_kwargs)\
            + ')'

# Aliases


class SinglePeriodOpt(SinglePeriodOptimization):
    """Alias of :class:`SinglePeriodOptimization`.

    As it was defined originally in :paper:`section 6.1 <section.6.1>` of the
    paper.
    """

class MultiPeriodOpt(MultiPeriodOptimization):
    """Alias of :class:`MultiPeriodOptimization`.

    As it was defined originally in :paper:`section 6.1 <section.6.1>` of the
    paper.
    """
