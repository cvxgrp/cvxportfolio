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
"""This module contains trading policies that can be backtested."""

import datetime as dt
import copy
import logging
import warnings

import pandas as pd
import numpy as np
import cvxpy as cp

from .costs import BaseCost
from .returns import BaseReturnsModel
from .constraints import BaseConstraint
from .estimator import PolicyEstimator, DataEstimator
from .errors import *
from .returns import ReturnsForecast, CashReturn
from .benchmark import *
from .utils import *

__all__ = [
    "Hold",
    "SellAll",
    "FixedTrades",
    "PeriodicRebalance",
    "ProportionalRebalance",
    "AdaptiveRebalance",
    "SinglePeriodOptimization",
    "MultiPeriodOptimization",
    "ProportionalTradeToTargets",
    "RankAndLongShort",
    "FixedWeights",
    "Uniform",
]


class BaseTradingPolicy(PolicyEstimator):
    """Base class for a trading policy."""


class Hold(BaseTradingPolicy):
    """Hold initial portfolio, don't trade."""

    def _values_in_time(self, current_weights, **kwargs):
        """Update sub-estimators and produce current estimate."""
        return pd.Series(0., index=current_weights.index)


class RankAndLongShort(BaseTradingPolicy):
    """Rank assets by signal; long highest and short lowest.

    Args:
        signal (pd.DataFrame): time-indexed DataFrame of signal for all symbols
            excluding cash. At each point in time the num_long assets with
            highest signal will have equal positive weight, and the num_short assets with
            lower signal will have equal negative weight. If two or more assets have the same
            signal value and they are on the boundary of either the top or bottom set,
            alphanumerical ranking will prevail.
        num_long (int or pd.Series): number of assets to long, default 1; if specified as Series
            it must be indexed by time.
        num_short (int or pd.Series): number of assets to short, default 1; if specified as Series
            it must be indexed by time.
        target_leverage (float or pd.Series): leverage of the resulting portfolio, default 1;
            if specified as Series it must be indexed by time.
    """

    def __init__(self, signal, num_long=1, num_short=1, target_leverage=1.):
        """Define sub-estimators at class attribute level."""
        self.num_long = DataEstimator(num_long)
        self.num_short = DataEstimator(num_short)
        self.signal = DataEstimator(signal)
        self.target_leverage = DataEstimator(target_leverage)

    def _values_in_time(self, t, current_weights, **kwargs):
        """Update sub-estimators and produce current estimate."""

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

        return target_weights - current_weights


class ProportionalTradeToTargets(BaseTradingPolicy):
    """Given a DataFrame of target weights in time, trade in equal proportions to reach those.

    Initially, it loads the list of trading days and so at each day it knows
    how many are missing before the next target's day, and trades in equal proportions
    to reach those targets. If there are no targets remaining it defaults to
    not trading.

    Args:
        targets (pd.DataFrame): time-indexed DataFrame of target weight vectors at
            given points in time (e.g., start of each month).

    """

    def __init__(self, targets):
        self.targets = targets

    def _pre_evaluation(self, universe, backtest_times):
        """Get list of trading days."""
        self.trading_days = backtest_times

    def _values_in_time(self, t, current_weights, **kwargs):
        """Get current trade weights."""
        next_targets = self.targets.loc[self.targets.index >= t]
        if not np.allclose(next_targets.sum(1), 1.):
            raise ValueError(
                f"The target weights provided to {self.__class__.__name__} at time {t} do not sum to 1.")
        if not len(next_targets):
            return pd.Series(0., index=current_weights.index)
        next_target = next_targets.iloc[0]
        next_target_day = next_targets.index[0]
        trading_days_to_target = len(self.trading_days[(
            self.trading_days >= t) & (self.trading_days < next_target_day)])
        return (next_target - current_weights) / (trading_days_to_target + 1)


class SellAll(BaseTradingPolicy):
    """Sell all assets to cash.

    This is useful to check the tcost model in the simulator.
    """

    def _values_in_time(self, t, current_weights, **kwargs):
        """Get current trade weights."""
        target = np.zeros(len(current_weights))
        target[-1] = 1.
        return target - current_weights


class FixedTrades(BaseTradingPolicy):
    """Each day trade the provided trade weights vector.

    If there are no weights defined for the given day, default to no
    trades.

    :param trades_weights: target trade weights :math:`z_t` to trade at each period.
        If constant in time use a pandas Series indexed by the assets'
        names, including the cash account name (``cash_key`` option 
        to the simulator). If varying in time, use a pandas DataFrame
        with datetime index and as columns the assets names including cash.
        If a certain time in the backtest is not present in the data provided
        the policy defaults to not trading in that period.
    :type trades_weights: pd.Series or pd.DataFrame
    """

    def __init__(self, trades_weights):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        self.trades_weights = DataEstimator(trades_weights, data_includes_cash=True)

    def _recursive_values_in_time(self, t, current_weights, **kwargs):
        """We need to override recursion b/c we catch exception."""
        try:
            super()._recursive_values_in_time(t=t, current_weights=current_weights, **kwargs)
            return pd.Series(self.trades_weights.current_value, current_weights.index)
        except MissingTimesError:
            return pd.Series(0., current_weights.index)


class FixedWeights(BaseTradingPolicy):
    """Each day trade to the provided trade weights vector.

    If there are no weights defined for the given day, default to no
    trades.

    :param target_weights: target weights :math:`w_t^+` to trade to at each period.
        If constant in time use a pandas Series indexed by the assets'
        names, including the cash account name (``cash_key`` option 
        to the simulator). If varying in time, use a pandas DataFrame
        with datetime index and as columns the assets names including cash.
        If a certain time in the backtest is not present in the data provided
        the policy defaults to not trading in that period.
    :type target_weights: pd.Series or pd.DataFrame 
            
    """

    def __init__(self, target_weights):
        """Trade the tradevec vector (dollars) or tradeweight weights."""
        self.target_weights = DataEstimator(target_weights, data_includes_cash=True)

    def _recursive_values_in_time(self, t, current_weights, **kwargs):
        """We need to override recursion b/c we catch exception."""
        try:
            super()._recursive_values_in_time(t=t, current_weights=current_weights, **kwargs)
            return pd.Series(self.target_weights.current_value,
                             current_weights.index) - current_weights
        except MissingTimesError:
            return pd.Series(0., current_weights.index)


class Uniform(FixedWeights):
    """Uniform allocation on non-cash assets."""

    def __init__(self):
        pass

    def _pre_evaluation(self, universe, backtest_times):
        target_weights = pd.Series(1., universe)
        target_weights.iloc[-1] = 0
        target_weights /= sum(target_weights)
        self.target_weights = DataEstimator(target_weights)


class PeriodicRebalance(FixedWeights):
    """Track a target weight vector rebalancing at given times.

    This calls `FixedWeights`. If you want to change the target in time
    use that policy directly.


    Args:
        target (pd.Series): portfolio weights to rebalance to.
        rebalancing_times (pd.DateTimeIndex): after the open trading on these days
            portfolio is equal to target.
    """

    def __init__(self, target, rebalancing_times):
        target_weights = pd.DataFrame(
            {el: target for el in rebalancing_times}).T
        super().__init__(target_weights)


class ProportionalRebalance(ProportionalTradeToTargets):
    """Track a target weight exactly at given times, trading proportionally to it each period.

    This calls `ProportionalTradeToTargets`. If you want to change the target in time
    use that policy directly.

    Args:
        target (pd.Series): portfolio weights to rebalance to.
        target_matching_times (pd.DateTimeIndex): after the open trading on these days
            portfolio is equal to target.
    """

    def __init__(self, target, target_matching_times):
        targets = pd.DataFrame({el: target for el in target_matching_times}).T
        super().__init__(targets)


class AdaptiveRebalance(BaseTradingPolicy):
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

    def _values_in_time(self, t, current_weights, **kwargs):
        if np.linalg.norm(current_weights - self.target.current_value) > \
                self.tracking_error.current_value:
            return self.target.current_value - current_weights
        else:
            return pd.Series(0., current_weights.index)


class MultiPeriodOptimization(BaseTradingPolicy):
    r"""Multi Period Optimization policy.

    Implements the model developed in Chapter 5, in particular
    at page 49, of the book. You specify the objective terms
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

    :param objective: these will be maximized;
        if you pass a single expression of BaseCost it is understood as the 
        same for all steps; if it's a list you must also pass a list of lists
        for `constraints`, each term represents the cost for each step of the optimization
        (starting from the first, i.e., today) and the length of the list is 
        used as planning_horizon (the value you pass there will be ignored) 
    :type objective: algebra of BaseCost or list of
    :param constraints: these will be
        imposed on the optimization. Default []. Pass this as a list of
        lists of the same length as `objective` to specify different 
        constraints at different time steps.
    :type constraints: list of BaseConstraints or list of those
    :param planning_horizon:  how many steps in the future we 
        plan for. Ignored if passing `objective` and `constraints` as lists.
        Default is None.
    :type planning_horizon: int or None
    :param terminal_constraint: if you pass a Series to this
        (default is None) it will impose that at the last step of the multi
        period optimization the post-trade weights are equal to this.
    :type terminal_constraint: pd.Series or None
    :param include_cash_return: whether to automatically include the ``CashReturn`` term in the objective,
        with default parameters. Default is ``True``.
    :type include_cash_return: bool
    :param benchmark: benchmark weights to use in the risk model and other terms that need it. Implemented
        ones are ``CashBenchmark``, the default, ``UniformBenchmark`` (uniform allocation on non-cash assets),
        and ``MarketBenchmark``, which approximates the market-weighted portfolio.
    :type benchmark: BaseBenchmark class or instance
    :param \**kwargs: these will be passed to cvxpy.Problem.solve,
        so you can choose your own solver and pass
        parameters to it.
    """

    def __init__(self, objective, constraints=[], include_cash_return=True, planning_horizon=None, terminal_constraint=None, benchmark=CashBenchmark, **kwargs):
        if hasattr(objective, '__iter__'):
            if not (hasattr(constraints, '__iter__') and len(constraints) and (hasattr(constraints[0], '__iter__') and len(objective) == len(constraints))):
                raise SyntaxError(
                    'If you pass objective as a list, constraints should be a list of lists of the same length.')
            self._planning_horizon = len(objective)
            self.objective = objective
            self.constraints = constraints
        else:
            if not np.isscalar(planning_horizon):
                raise SyntaxError(
                    'If `objective` and `constraints` are the same for all steps you must specify `planning_horizon`.')
            self._planning_horizon = planning_horizon
            self.objective = [copy.deepcopy(objective) for i in range(
                planning_horizon)] if planning_horizon > 1 else [objective]
            self.constraints = [copy.deepcopy(constraints) for i in range(
                planning_horizon)] if planning_horizon > 1 else [constraints]

        self._include_cash_return = include_cash_return
        if self._include_cash_return:
            self.objective = [el + CashReturn() for el in self.objective]
        self.terminal_constraint = terminal_constraint
        self.benchmark = benchmark() if isinstance(benchmark, type) else benchmark
        self.cvxpy_kwargs = kwargs

    def _compile_to_cvxpy(self):  # , w_plus, z, value):
        """Compile all cvxpy expressions and the problem."""
        self.cvxpy_objective = [
            el._compile_to_cvxpy(
                self.w_plus_at_lags[i], self.z_at_lags[i], self.w_plus_minus_w_bm_at_lags[i])
            for i, el in enumerate(self.objective)]
        for el, term in zip(self.objective, self.cvxpy_objective):
            if not term.is_dcp():
                raise ConvexSpecificationError(el)
            if not term.is_concave():
                raise ConvexityError(el)
        self.cvxpy_objective = sum(self.cvxpy_objective)

        def compile_and_check_constraint(constr, i):
            result = constr._compile_to_cvxpy(
                self.w_plus_at_lags[i], self.z_at_lags[i], self.w_plus_minus_w_bm_at_lags[i])
            for el in (result if hasattr(result, '__iter__') else [result]):
                if not el.is_dcp():
                    raise ConvexSpecificationError(constr)
            return result

        self.cvxpy_constraints = [
            flatten_heterogeneous_list([
                compile_and_check_constraint(constr, i) for constr in el])
            for i, el in enumerate(self.constraints)]

        self.cvxpy_constraints = sum(self.cvxpy_constraints, [])
        self.cvxpy_constraints += [cp.sum(z) == 0 for z in self.z_at_lags]
        w = self.w_current
        for i in range(self._planning_horizon):
            self.cvxpy_constraints.append(
                self.w_plus_at_lags[i] == self.z_at_lags[i] + w)
            self.cvxpy_constraints.append(
                self.w_plus_at_lags[i] - self.w_bm == self.w_plus_minus_w_bm_at_lags[i])
            w = self.w_plus_at_lags[i]
        if not self.terminal_constraint is None:
            self.cvxpy_constraints.append(w == self.terminal_constraint)
        self.problem = cp.Problem(cp.Maximize(
            self.cvxpy_objective), self.cvxpy_constraints)
        if not self.problem.is_dcp():  # dpp=True)
            raise SyntaxError(f"The optimization problem compiled by {self.__class__.__name__}"
                              " does not follow the convex optimization rules. This should not happen"
                              " if you're using the default cvxportfolio terms and is probably due to a"
                              " mis-specified custom term.")

    def _recursive_pre_evaluation(self, universe, backtest_times):
        """No point in using recursive super() method."""

        for obj in self.objective:
            obj._recursive_pre_evaluation(
                universe=universe, backtest_times=backtest_times)
        for constr_at_lag in self.constraints:
            for constr in constr_at_lag:
                constr._recursive_pre_evaluation(
                    universe=universe, backtest_times=backtest_times)

        self.benchmark._recursive_pre_evaluation(
            universe=universe, backtest_times=backtest_times)
        self.w_bm = cp.Parameter(len(universe))

        # temporary
        # self.w_bm = np.zeros(len(universe))
        # self.w_bm[-1] = 1.

        # initialize the problem
        # self.portfolio_value = cp.Parameter(nonneg=True)
        self.w_current = cp.Parameter(len(universe))
        self.z_at_lags = [cp.Variable(len(universe))
                          for i in range(self._planning_horizon)]
        self.w_plus_at_lags = [cp.Variable(
            len(universe)) for i in range(self._planning_horizon)]
        self.w_plus_minus_w_bm_at_lags = [cp.Variable(
            len(universe)) for i in range(self._planning_horizon)]

        # simulator will overwrite this with cached loaded from disk
        self.cache = {}

        # self._compile_to_cvxpy()#self.w_plus, self.z, self.portfolio_value)

    def _recursive_values_in_time(self, t, current_weights, current_portfolio_value, past_returns, past_volumes, current_prices, **kwargs):
        """Update all cvxpy parameters and solve."""

        if not current_portfolio_value > 0:
            raise Bankruptcy(
                f"The backtest of policy:\n{self}\nat time {t} has resulted in bankruptcy.")
        assert np.isclose(sum(current_weights), 1)

        for i, obj in enumerate(self.objective):
            obj._recursive_values_in_time(t=t, current_weights=current_weights,
                                          current_portfolio_value=current_portfolio_value,
                                          past_returns=past_returns, past_volumes=past_volumes,
                                          current_prices=current_prices, mpo_step=i, cache=self.cache, **kwargs)

        for i, constr_at_lag in enumerate(self.constraints):
            for constr in constr_at_lag:
                constr._recursive_values_in_time(t=t, current_weights=current_weights,
                                                 current_portfolio_value=current_portfolio_value,
                                                 past_returns=past_returns, past_volumes=past_volumes,
                                                 current_prices=current_prices, mpo_step=i, cache=self.cache, **kwargs)

        self.benchmark._recursive_values_in_time(t=t, current_weights=current_weights,
                                                 current_portfolio_value=current_portfolio_value,
                                                 past_returns=past_returns, past_volumes=past_volumes,
                                                 current_prices=current_prices, mpo_step=i, cache=self.cache, **kwargs)

        self.w_bm.value = self.benchmark.current_value
        self.w_current.value = current_weights.values

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message='Solution may be inaccurate')
                self.problem.solve(**self.cvxpy_kwargs)
        except cp.SolverError:
            raise PortfolioOptimizationError(
                f"Numerical solver for policy {self.__class__.__name__} at time {t} failed;"
                "try changing it, relaxing some constraints, or dropping some costs.")
        if self.problem.status in ["unbounded", "unbounded_inaccurate"]:
            raise PortfolioOptimizationError(
                f"Policy {self.__class__.__name__} at time {t} resulted in an unbounded problem."
            )
        if self.problem.status in ["infeasible", 'infeasible_inaccurate']:
            raise PortfolioOptimizationError(
                f"Policy {self.__class__.__name__} at time {t} resulted in an infeasible problem."
            )

        return pd.Series(self.z_at_lags[0].value, current_weights.index)

    def _collect_hyperparameters(self):
        result = []
        for el in self.objective:
            result += el._collect_hyperparameters()
        for el in self.constraints:
            for constr in el:
                result += constr._collect_hyperparameters()
        return result


class SinglePeriodOptimization(MultiPeriodOptimization):
    r"""Single Period Optimization policy.

    Implements the model developed in Chapter 4, in particular
    at page 43, of the book. You specify the objective term
    using classes such as ReturnsForecast and TcostModel, each
    multiplied by its multiplier. You also specify a list
    of constraints.

    :param objective: this algebraic combination of cvxportfolio cost objects will be maximized
    :type objective: CombinedCost
    :param constraints: these will be imposed on the optimization. Default [].
    :type constraints: list of BaseConstraints
    :param include_cash_return: whether to automatically include the ``CashReturn`` term in the objective,
        with default parameters. Default is ``True``.
    :type include_cash_return: bool
    :param benchmark: benchmark weights to use in the risk model and other terms that need it. Implemented
        ones are ``CashBenchmark``, the default, ``UniformBenchmark`` (uniform allocation on non-cash assets),
        and ``MarketBenchmark``, which approximates the market-weighted portfolio.
    :type benchmark: BaseBenchmark class or instance
    :param \**kwargs: these will be passed to cvxpy.Problem.solve, so you can choose your own solver and pass
        parameters to it.
    """

    def __init__(self, objective, constraints=[], include_cash_return=True, benchmark=CashBenchmark, **kwargs):
        super().__init__([objective], [constraints], include_cash_return=include_cash_return,
                         benchmark=benchmark, **kwargs)

    # def __repr__(self):
    #     return self.__class__.__name__ + '(' \
    #         + 'objective=' + str(self.objective[0]) \
    #         + ', constraints=' + str(self.constraints[0])
    #         + ', benchmark=' + str(self.constraints[0])
    #         + ', cvxpy_kwargs=' + str(self.cvxpy_kwargs)
