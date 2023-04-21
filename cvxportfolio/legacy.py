# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
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
"""
This module contains classes that have been replaced by improved versions.
They are kept because they have some slight differences and are used by 
the original examples scripts to generate the book's plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import logging
import cvxpy as cvx
import numpy as np

from .policies import BaseTradingPolicy
from .returns import BaseReturnsModel
from .costs import BaseCost

logger = logging.getLogger(__name__)

__all__ = [
    "null_checker",
    "non_null_data_args",
    "values_in_time",
    "plot_what_if",
    "SinglePeriodOpt",
    "MultiPeriodOpt",
    "LegacyReturnsForecast",
    "MPOReturnsForecast",
    "MultipleReturnsForecasts"
    ]


def values_in_time(obj, t, tau=None):
    """Obtain value(s) of object at time t, or right before.

    Optionally specify time tau>=t for which we want a prediction,
    otherwise it is assumed tau = t.

    obj: callable, pd.Series, pd.DataFrame, or something else.

        If a callable, we return obj(t,tau).

        If obj has an index attribute,
        we try to return obj.loc[t],
        or obj.loc[t, tau], if the index is a MultiIndex.
        If not available, we return obj.

        Otherwise, we return obj.

    t: np.Timestamp (or similar). Time at which we want
        the value.

    tau: np.Timestamp (or similar), or None. Time tau >= t
        of the prediction,  e.g., tau could be tomorrow, t
        today, and we ask for prediction of market volume tomorrow,
        made today. If None, then it is assumed tau = t.

    """

    if hasattr(obj, "__call__"):
        return obj(t, tau)

    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        try:
            if not (tau is None) and isinstance(obj.index, pd.MultiIndex):
                return obj.loc[(t, tau)]
            else:
                return obj.loc[t]
        except KeyError:
            return obj

    return obj


def plot_what_if(time, true_results, alt_results):
    true_results.value.plot(label=true_results.pol_name)
    for result in alt_results:
        result.value.plot(label=result.pol_name, linestyle="--")
    plt.axvline(x=time, linestyle=":")


def null_checker(obj):
    """Check if obj contains NaN."""
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        if np.any(pd.isnull(obj)):
            raise ValueError("Data object contains NaN values", obj)
    elif np.isscalar(obj):
        if np.isnan(obj):
            raise ValueError("Data object contains NaN values", obj)
    else:
        raise TypeError("Data object can only be scalar or Pandas.")


def non_null_data_args(f):
    def new_f(*args, **kwds):
        for el in args:
            null_checker(el)
        for el in kwds.values():
            null_checker(el)
        return f(*args, **kwds)

    return new_f
    
    



class SinglePeriodOpt(BaseTradingPolicy):
    """Single-period optimization policy.

    Implements the model developed in chapter 4 of our paper
    https://stanford.edu/~boyd/papers/cvx_portfolio.html
    """

    def __init__(
            self,
            return_forecast,
            costs,
            constraints,
            solver=None,
            solver_opts=None):
        if not isinstance(return_forecast, BaseReturnsModel):
            null_checker(return_forecast)
        self.return_forecast = return_forecast

        super(SinglePeriodOpt, self).__init__()

        for cost in costs:
            assert isinstance(cost, BaseCost)
            self.costs.append(cost)

        for constraint in constraints:
            assert isinstance(constraint, BaseConstraint)
            self.constraints.append(constraint)

        self.solver = solver
        self.solver_opts = {} if solver_opts is None else solver_opts

    def get_trades(self, portfolio, t=None):
        """
        Get optimal trade vector for given portfolio at time t.

        Parameters
        ----------
        portfolio : pd.Series
            Current portfolio vector.
        t : pd.timestamp
            Timestamp for the optimization.
        """

        if t is None:
            t = dt.datetime.today()

        value = sum(portfolio)
        w = portfolio / value
        z = cvx.Variable(w.size)  # TODO pass index
        wplus = w.values + z

        if isinstance(self.return_forecast, BaseReturnsModel):
            alpha_term = self.return_forecast.weight_expr(t, wplus)
        else:
            alpha_term = cvx.sum(
                cvx.multiply(
                    values_in_time(
                        self.return_forecast,
                        t).values,
                    wplus))

        assert alpha_term.is_concave()

        costs, constraints = [], []

        for cost in self.costs:
            cost_expr, const_expr = cost.weight_expr(t, wplus, z, value)
            costs.append(cost_expr)
            constraints += const_expr

        for constr in self.constraints:
            constraints += constr.weight_expr(t, wplus, z, value)

        for el in costs:
            assert el.is_convex()

        for el in constraints:
            assert el.is_dcp()

        # raise Exception

        self.prob = cvx.Problem(cvx.Maximize(
            alpha_term - sum(costs)), [cvx.sum(z) == 0] + constraints)
        try:
            self.prob.solve(solver=self.solver, ignore_dpp=True, **self.solver_opts)

            if self.prob.status in ["unbounded", "unbounded_inaccurate"]:
                logging.error(
                    "The problem is unbounded. Defaulting to no trades")
                return self._nulltrade(portfolio)

            if self.prob.status in ["infeasible", 'infeasible_inaccurate']:
                logging.error(
                    "The problem is infeasible. Defaulting to no trades")
                return self._nulltrade(portfolio)

            return pd.Series(index=portfolio.index, data=(z.value * value))
        except (cvx.SolverError, TypeError):
            logging.error(
                "The solver %s failed. Defaulting to no trades" %
                self.solver)
            return self._nulltrade(portfolio)


# class LookaheadModel():
#     """Returns the planning periods for multi-period.
#     """
#     def __init__(self, trading_times, period_lens):
#         self.trading_times = trading_times
#         self.period_lens = period_lens
#
#     def get_periods(self, t):
#         """Returns planning periods.
#         """
#         periods = []
#         tau = t
#         for length in self.period_lens:
#             incr = length*pd.Timedelta('1 days')
#             periods.append((tau, tau + incr))
#             tau += incr
#         return periods


class MultiPeriodOpt(SinglePeriodOpt):
    def __init__(
            self,
            trading_times,
            terminal_weights,
            lookahead_periods=None,
            *args,
            **kwargs):
        """
        trading_times: list, all times at which get_trades will be called
        lookahead_periods: int or None. if None uses all remaining periods
        """
        # Number of periods to look ahead.
        self.lookahead_periods = lookahead_periods
        self.trading_times = trading_times
        # Should there be a constraint that the final portfolio is the bmark?
        self.terminal_weights = terminal_weights
        super(MultiPeriodOpt, self).__init__(*args, **kwargs)

    def get_trades(self, portfolio, t=dt.datetime.today()):
        value = sum(portfolio)
        assert value > 0.0
        w = cvx.Constant(portfolio.values / value)

        prob_arr = []
        z_vars = []

        # planning_periods = self.lookahead_model.get_periods(t)
        for tau in self.trading_times[
            self.trading_times.get_loc(t): self.trading_times.get_loc(t)
            + self.lookahead_periods
        ]:
            # delta_t in [pd.Timedelta('%d days' % i) for i in
            # range(self.lookahead_periods)]:

            #            tau = t + delta_t
            z = cvx.Variable(*w.shape)
            wplus = w + z
            obj = self.return_forecast.weight_expr_ahead(t, tau, wplus)

            costs, constr = [], []
            for cost in self.costs:
                cost_expr, const_expr = cost.weight_expr_ahead(
                    t, tau, wplus, z, value)
                costs.append(cost_expr)
                constr += const_expr

            obj -= sum(costs)
            constr += [cvx.sum(z) == 0]
            for single_constr in self.constraints:
                constr += single_constr.weight_expr(t, wplus, z, value)

            prob = cvx.Problem(cvx.Maximize(obj), constr)
            prob_arr.append(prob)
            z_vars.append(z)
            w = wplus

        # Terminal constraint.
        if self.terminal_weights is not None:
            # prob_arr[-1].constraints += [wplus == self.terminal_weights.values]
            prob_arr[-1] = cvx.Problem(cvx.Maximize(obj),
                                       constr + [wplus == self.terminal_weights.values])

        sum(prob_arr).solve(solver=self.solver, ignore_dpp=True)
        return pd.Series(index=portfolio.index, data=(z_vars[0].value * value))


# LEGACY CLASSES USED BY OLD TESTS. WILL BE REMOVED AS WE FINISH TRANSLATION


class LegacyReturnsForecast(BaseReturnsModel):
    """A single return forecast.

    STILL USED BY OLD PARTS OF CVXPORTFOLIO

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, returns, delta=0.0, gamma_decay=None, name=None):
        null_checker(returns)
        self.returns = returns
        null_checker(delta)
        self.delta = delta
        self.gamma_decay = gamma_decay
        self.name = name

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = cvx.multiply(values_in_time(self.returns, t), wplus)
        alpha -= cvx.multiply(values_in_time(self.delta, t), cvx.abs(wplus))
        return cvx.sum(alpha)

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """

        alpha = self.weight_expr(t, wplus)
        if tau > t and self.gamma_decay is not None:
            alpha *= (tau - t).days ** (-self.gamma_decay)
        return alpha


class MPOReturnsForecast(BaseReturnsModel):
    """A single alpha estimation.

    Attributes:
      alpha_data: A dict of series of return estimates.
    """

    def __init__(self, alpha_data):
        self.alpha_data = alpha_data

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        return self.alpha_data[(t, tau)].values.T * wplus


class MultipleReturnsForecasts(BaseReturnsModel):
    """A weighted combination of alpha sources.

    DEPRECATED: THIS SHOULD BE DONE BY MULTIPLYING BY HYPERPARAMETERS
    AND PASSING MULTIPLE RETURN MODELS LIKE WE DO FOR COSTS

    Attributes:
      alpha_sources: a list of alpha sources.
      weights: An array of weights for the alpha sources.
    """

    def __init__(self, alpha_sources, weights):
        self.alpha_sources = alpha_sources
        self.weights = weights

    def weight_expr(self, t, wplus, z=None, v=None):
        """Returns the estimated alpha.

        Args:
            t: time estimate is made.
            wplus: An expression for holdings.
            tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr(t, wplus) * self.weights[idx]
        return alpha

    def weight_expr_ahead(self, t, tau, wplus):
        """Returns the estimate at time t of alpha at time tau.

        Args:
          t: time estimate is made.
          wplus: An expression for holdings.
          tau: time of alpha being estimated.

        Returns:
          An expression for the alpha.
        """
        alpha = 0
        for idx, source in enumerate(self.alpha_sources):
            alpha += source.weight_expr_ahead(t,
                                              tau, wplus) * self.weights[idx]
        return alpha
