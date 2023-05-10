# Copyright 2016-2020 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.
# Copyright 2023- The Cvxportfolio Contributors
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

# from .expression import Expression
# from .utils import null_checker, values_in_time
from .estimator import CvxpyExpressionEstimator, ParameterEstimator, DataEstimator

__all__ = ["HoldingCost", "TransactionCost"]


class BaseCost(CvxpyExpressionEstimator):
    """Base class for cost objects (and also risks).

    Here there is some logic used to implement the algebraic operations.
    See also :class:`CombinedCost`.
    """

    # gamma = 1. # this will be removed
    #LEGACY = False # used by some methods that need to know if they run in legacy mode
    #INITIALIZED = False # used to interface w/ old cvxportfolio
    
    # # PLACEHOLDER METHOD TO USE OLD INTERFACE WITH NEW INTERFACE
    # def weight_expr(self, t, w_plus, z, value):
    #     cost, constr = self._estimate(t, w_plus, z, value)
    #     return cost, constr
    #
    # def _estimate(self, t, w_plus, z, value):
    #     """Temporary interface to old cvxportfolio."""
    #     #self.LEGACY = True
    #     #if not self.INITIALIZED:
    #     placehoder_returns = pd.DataFrame(np.zeros((1, w_plus.shape[0] if not w_plus is None else z.shape[0])))
    #     self.pre_evaluation(placehoder_returns, None, t, None)
    #     self.legacy_expression = self.compile_to_cvxpy(w_plus, z, value)
    #     #self.INITIALIZED = True
    #     self.values_in_time(t, None, value, None, None)
    #     return self.legacy_expression, []
    #
    # def weight_expr_ahead(self, t, tau, w_plus, z, value):
    #     return self.weight_expr(t, w_plus, z, value)

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
        
    # def _estimate(self, t, w_plus, z, value):
    #     """Temporary interface to old cvxportfolio."""
    #     #for cost in self.costs:
    #         #cost.LEGACY = True
    #     return super()._estimate(t, w_plus, z, value)

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
            # assert self.expression.is_dcp()#dpp=True)
        return self.expression
        # return sum([multiplier * cost.compile_to_cvxpy(w_plus, z, portfolio_value)
        #    for multiplier, cost in zip(self.multipliers, self.costs)])

    # TEMPORARY IN ORDER NOT TO BREAK TESTS
    # THESE METHODS ARE DEPRECATED

    # def optimization_log(self, t):
    #     return sum(
    #         [
    #             multiplier
    #             * (cost.expression.value if hasattr(cost, "expression") else 0.0)
    #             for multiplier, cost in zip(self.multipliers, self.costs)
    #         ]
    #     )
    #
    # def simulation_log(self, t):
    #     return sum(
    #         [
    #             multiplier * (cost.last_cost if hasattr(cost, "last_cost") else 0.0)
    #             for multiplier, cost in zip(self.multipliers, self.costs)
    #         ]
    #     )


class HoldingCost(BaseCost):
    """A model for holding costs.
    
    In normal use cases you should not pass any argument to the constructur, but
    rather to :class:`Backtest` (unless you're happy with the default values there).
    That will take care of populating the values for the various holding 
    costs in this class during each backtest. 
    Regarding dividends, by default they are already included in each stock's return. 
    Legacy applications might instead account for stock returns and dividends separately.
    That is not advised: it would introduce small biases in the estimation of historical
    mean returns and covariances.

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
        #spread_on_cash_percent = .5,
        # spread_on_long_positions_percent=None,
        dividends=None,
        # spread_on_lending_cash_percent=.5,
        # spread_on_borrowing_cash_percent=.5,
        # make_dqcp=False
        ):
        
        self.spread_on_borrowing_stocks_percent = None if spread_on_borrowing_stocks_percent is None else \
            DataEstimator(spread_on_borrowing_stocks_percent)
        #self.spread_on_cash_percent = None if spread_on_cash_percent is None else \
        #    DataEstimator(spread_on_cash_percent)
        # self.spread_on_long_positions_percent = None if spread_on_long_positions_percent is None else \
        #     DataEstimator(spread_on_long_positions_percent)
        self.dividends = None if dividends is None else \
            ParameterEstimator(dividends)
        # self.spread_on_lending_cash_percent = DataEstimator(0.) if spread_on_lending_cash_percent is None else \
        #     DataEstimator(spread_on_lending_cash_percent)
        # self.spread_on_borrowing_cash_percent = DataEstimator(0.) if spread_on_borrowing_cash_percent is None else \
        #     DataEstimator(spread_on_borrowing_cash_percent)
        #
        # self.make_dqcp=False
        
    def pre_evaluation(self, universe, backtest_times):
        super().pre_evaluation(universe=universe, backtest_times=backtest_times)
        
        # if not (self.spread_on_long_positions_percent is None):
        #     self.long_cost_stocks = cvx.Parameter(len(universe) - 1, nonneg=True)
        if not (self.spread_on_borrowing_stocks_percent is None):
            self.borrow_cost_stocks = cvx.Parameter(len(universe) - 1, nonneg=True)
        #if not (self.spread_on_cash_percent is None):
        #    self.cash_cost = cvx.Parameter(nonneg=True)
        # self.cash_lending_parameter = cvx.Parameter(nonneg=True)
        # self.cash_borrowing_parameter = cvx.Parameter(nonneg=True)
        # self.cash_return_parameter = cvx.Parameter()
        
        
    def values_in_time(self, t, past_returns, **kwargs):
        """We use yesterday's value of the cash return here while in the simulator
        we use today's. In the US, updates to the FED rate are published outside
        of trading hours so we might as well use the actual value for today's. The difference
        is very small so for now we do this. 
        """
        super().values_in_time(t=t, past_returns=past_returns, **kwargs)
                               
        cash_return = past_returns.iloc[-1,-1]
        # self.cash_return_parameter.value = cash_return
        
        # if not (self.spread_on_long_positions_percent is None):
        #     self.long_cost_stocks.value = np.ones(returns.shape[1] - 1) * cash_return + \
        #         self.spread_on_long_positions_percent.current_value / (100 * 252)
        if not (self.spread_on_borrowing_stocks_percent is None):
            self.borrow_cost_stocks.value = np.ones(past_returns.shape[1] - 1) * (cash_return) + \
                self.spread_on_borrowing_stocks_percent.current_value / (100 * 252)
        #if not (self.spread_on_cash_percent is None):
        #    self.cash_cost.value = self.spread_on_cash_percent.current_value / (100 * 252)
        # if not (self.spread_on_lending_cash_percent is None):
        #     self.cash_lending_parameter.value = np.maximum(cash_return -
        #         self.spread_on_lending_cash_percent.current_value / (100 * 252), 0.)
        # if not (self.spread_on_borrowing_cash_percent is None):
        #     self.cash_borrowing_parameter.value = cash_return + \
        #         self.spread_on_borrowing_cash_percent.current_value / (100 * 252)
        

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Compile cost to cvxpy expression."""
        
        # we need to take this out
        # because we re-add it below
        expression = 0. # - w_plus[-1] * self.cash_return_parameter 
        
        # the real cash position is without the shorts
        # however, when using it we get a QCQP program
        # approx_cash = cvx.pos(w_plus[-1])
        # real_cash = w_plus[-1] - cvx.sum(cvx.neg(w_plus)[:-1])
        
        # if not (self.spread_on_long_positions_percent is None):
        #     expression -= cvx.multiply(self.long_cost_stocks, cvx.pos(w_plus)[:-1])
        
        if not (self.spread_on_borrowing_stocks_percent is None):
           expression += cvx.multiply(self.borrow_cost_stocks, cvx.neg(w_plus)[:-1])
        
        #if not (self.spread_on_cash_percent is None):
        #    expression += self.cash_cost * cvx.abs(w_plus[-1])
        
        # expression += self.cash_lending_parameter * cvx.pos(real_cash if self.make_dqcp else approx_cash)
        # expression -= self.cash_borrowing_parameter * cvx.neg(real_cash)
        
        if not (self.dividends is None):
            expression -= cvx.multiply(self.dividends, w_plus[:-1])
        assert cvx.sum(expression).is_convex()
        return cvx.sum(expression)

    # def value_expr(self, t, h_plus, u):
    #     """Placeholder method as we update the rest of the stack to new interface."""
    #
    #     #if not self.INITIALIZED:
    #     self.pre_evaluation(None, None, t, None)
    #     wplus = cvx.Variable(len(h_plus))
    #     z = cvx.Variable(len(h_plus))
    #     v = cvx.Parameter(nonneg=True)
    #     self.compile_to_cvxpy(wplus, z, v)
    #     self.values_in_time(t, None, None, None, None)
    #
    #     self.last_cost = -np.minimum(0, h_plus.iloc[:-1]) * self.borrow_costs.value
    #     self.last_cost -= h_plus.iloc[:-1] * self.dividends.value
    #
    #     return sum(self.last_cost)
    #
    # def optimization_log(self, t):
    #     return self.expression.value
    #
    # def simulation_log(self, t):
    #     return self.last_cost


class TransactionCost(BaseCost):
    """A model for transaction costs.

    (See section pages 10-11 in the paper
    https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf).
    We don't include the short-term alpha term `c` here because it
    can be expressed with a separate `ReturnsForecast` object. If
    any term that appears in

    Args:
      half_spread (float or pd.Series or pd.DataFrame): Half the bid ask spread, either
        fixed per (non-cash) assets, or varying in time.
      nonlin_coeff (float or pd.Series or pd.DataFrame): Coefficients
            for the nonlinear cost term. This is the coefficient `b` in the paper.
            It can be constant, constant per-stock, or varying in time and stocks. Default 0.
      sigma (floar or pd.Series or pd.DataFrame): Daily volatilities. Default 0.
      volume (float or pd.Series or pd.DataFrame): Market volumes expressed in value (e.g., dollars).
            Default 1 to avoid NaN.
      power (float): The nonlinear tcost exponent. Default 1.5.
    """

    def __init__(self, spreads=0., pershare_cost=0.005, b=1.0, window_sigma_est=250, window_volume_est=250, exponent=1.5):
        # self.compile_first_term = not np.isscalar(half_spread) or half_spread > 0.0
        # self.compile_second_term = (not np.isscalar(nonlin_coeff) or nonlin_coeff > 0.0) and (not np.isscalar(sigma) or sigma > 0.0)
        # if self.compile_first_term:
        # if spreads is None:
        #     self.spreads = None
        # else:
        self.spreads = DataEstimator(spreads)#, non_negative=True)
        self.pershare_cost = DataEstimator(pershare_cost)
        # self.sigma = DataEstimator(sigma) #, non_negative=True)
        # self.volume = DataEstimator(volume) #, non_negative=True)
        # self.nonlin_coeff = DataEstimator(nonlin_coeff) #, non_negative=True)
        #if self.compile_second_term:
        self.b = DataEstimator(b)
        self.window_sigma_est = window_sigma_est
        self.window_volume_est = window_volume_est
        #self.base_second_term_multiplier = ParameterEstimator(sigma * nonlin_coeff / (volume**(power - 1)))#, non_negative=True)
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
        # self.second_term_multiplier.value *= current_portfolio_value ** (self.power - 1)
        #if self.compile_second_term:
        self.second_term_multiplier.value = self.b.current_value * sigma_est * \
            (current_portfolio_value / volume_est) ** (self.exponent - 1)
        
        #self.base_second_term_multiplier.value * current_portfolio_value ** (self.power - 1)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        #if self.compile_first_term:
        #first_term = (cvx.multiply(self.half_spread, cvx.abs(z[:-1])) if self.compile_first_term else 0.0)
        expression = cvx.abs(z[:-1]).T @ self.first_term_multiplier
        assert expression.is_convex()
        expression += (cvx.abs(z[:-1]) ** self.exponent).T @ self.second_term_multiplier
        assert expression.is_convex()
        return expression
        
        # if not (self.spreads is None):
        #     expression += cvx.multiply(self.spreads, cvx.abs(z[:-1]))/2.
        #     assert cvx.sum(expression).is_convex()
        # #else:
        # #    first_term = 0.
        # #if self.compile_second_term:
        #     # second_term = cvx.multiply(self.nonlin_coeff, self.sigma)
        #     # second_term = cvx.multiply(
        #     #     second_term, (value / self.volume) ** (self.power - 1)
        #     # )
        # expression += cvx.multiply(self.second_term_multiplier, cvx.abs(z[:-1]) ** self.exponent)
        # assert cvx.sum(expression).is_convex()
        # assert cvx.sum(expression).is_dcp(dpp=True)
        # #else:
        # #    second_term = 0.0
        #
        # # self.expression = first_term + second_term
        # # self.expression = cvx.sum(self.expression)
        # # assert self.expression.is_dcp(dpp=True)
        # # assert self.expression.is_convex()
        # return cvx.sum(self.expression)

    # def _estimate(self, t, w_plus, z, value):
    #     """Estimate tcosts given trades.
    #
    #     Args:
    #       t: time of estimate
    #       z: trades
    #       value: portfolio value
    #
    #     Returns:
    #       An expression for the tcosts.
    #     """
    #
    #     z = z[:-1]
    #
    #     constr = []
    #
    #     second_term = (
    #         values_in_time(self.nonlin_coeff, t)
    #         * values_in_time(self.sigma, t)
    #         * (value / values_in_time(self.volume, t)) ** (self.power - 1)
    #     )
    #
    #     # # no trade conditions
    #     # if np.isscalar(second_term):
    #     #     if np.isnan(second_term):
    #     #         constr += [z == 0]
    #     #         second_term = 0
    #     # else:  # it is a pd series
    #     #     no_trade = second_term.index[second_term.isnull()]
    #     #     second_term[no_trade] = 0
    #     #     constr += [z[second_term.index.get_loc(tick)] == 0 for tick in no_trade]
    #
    #     try:
    #         self.expression = cvx.multiply(
    #             values_in_time(self.half_spread, t), cvx.abs(z)
    #         )
    #     except TypeError:
    #         self.expression = cvx.multiply(
    #             values_in_time(self.half_spread, t).values, cvx.abs(z)
    #         )
    #     try:
    #         self.expression += cvx.multiply(second_term, cvx.abs(z) ** self.power)
    #     except TypeError:
    #         self.expression += cvx.multiply(
    #             second_term.values, cvx.abs(z) ** self.power
    #         )
    #
    #     return cvx.sum(self.expression), constr

    # THESE METHODS ARE DEPRECATED AND WILL BE REMOVED AS WE FINISH
    # TRANSLATING TO NEW INTERFACE
    def value_expr(self, t, h_plus, u):
        """Temporary placeholder, new simulators implement their own tcost."""

        #if not self.INITIALIZED:
        self.pre_evaluation(None, None, t, None)
        wplus = cvx.Variable(len(u))
        z = cvx.Variable(len(u))
        v = cvx.Parameter(nonneg=True)
        self.compile_to_cvxpy(wplus, z, v)
        self.values_in_time(t, None, 1., None, None)

        u_nc = u.iloc[:-1]
        self.tmp_tcosts = 0.
        #if self.compile_first_term:
        self.tmp_tcosts += np.abs(u_nc) * self.half_spread.value
        #if self.compile_second_term:
        self.tmp_tcosts += (
                self.second_term_multiplier.value
                # self.nonlin_coeff.value
                # * self.sigma.value
                * np.abs(u_nc) ** self.power
                # / (self.volume.value ** (self.power - 1))
            )

        return self.tmp_tcosts.sum()

    def optimization_log(self, t):
        try:
            return self.expression.value
        except AttributeError:
            return np.nan

    def simulation_log(self, t):
        # TODO find another way
        return self.tmp_tcosts

    def est_period(self, t, tau_start, tau_end, w_plus, z, value):
        """Returns the estimate at time t of tcost over given period."""
        K = (tau_end - tau_start).days
        tcost, constr = self.weight_expr(t, None, z / K, value)
        return tcost * K, constr
