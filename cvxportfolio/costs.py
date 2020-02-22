"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cvx
import numpy as np
import copy
from .expression import Expression
from .utils import null_checker, values_in_time


__all__ = ['HcostModel', 'TcostModel']


class BaseCost(Expression):

    def __init__(self):
        self.gamma = 1.  # it is changed by gamma * BaseCost()

    def weight_expr(self, t, w_plus, z, value):
        cost, constr = self._estimate(t, w_plus, z, value)
        return self.gamma * cost, constr

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        cost, constr = self._estimate_ahead(t, tau, w_plus, z, value)
        return self.gamma * cost, constr

    def __mul__(self, other):
        """Read the gamma parameter as a multiplication."""
        newobj = copy.copy(self)
        newobj.gamma *= other
        return newobj

    def __rmul__(self, other):
        """Read the gamma parameter as a multiplication."""
        return self.__mul__(other)


class HcostModel(BaseCost):
    """A model for holding costs.

    Attributes:
      borrow_costs: A dataframe of borrow costs.
      dividends: A dataframe of dividends.
    """

    def __init__(self, borrow_costs, dividends=0.):
        null_checker(borrow_costs)
        self.borrow_costs = borrow_costs
        null_checker(dividends)
        self.dividends = dividends
        super(HcostModel, self).__init__()

    def _estimate(self, t, w_plus, z, value):
        """Estimate holding costs.

        Args:
          t: time of estimate
          wplus: holdings
          tau: time to estimate (default=t)
        """
        try:
            w_plus = w_plus[w_plus.index != self.cash_key]
            w_plus = w_plus.values
        except AttributeError:
            w_plus = w_plus[:-1]  # TODO fix when cvxpy pandas ready

        try:
            self.expression = cvx.multiply(
                values_in_time(self.borrow_costs, t), cvx.neg(w_plus))
        except TypeError:
            self.expression = cvx.multiply(values_in_time(
                self.borrow_costs, t).values, cvx.neg(w_plus))
        try:
            self.expression -= cvx.multiply(
                values_in_time(self.dividends, t), w_plus)
        except TypeError:
            self.expression -= cvx.multiply(
                values_in_time(self.dividends, t).values, w_plus)

        return cvx.sum(self.expression), []

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        return self._estimate(t, w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        self.last_cost = -np.minimum(0, h_plus.iloc[:-1]) * values_in_time(
            self.borrow_costs, t)
        self.last_cost -= h_plus.iloc[:-1] * values_in_time(self.dividends, t)

        return sum(self.last_cost)

    def optimization_log(self, t):
        return self.expression.value

    def simulation_log(self, t):
        return self.last_cost


class TcostModel(BaseCost):
    """A model for transaction costs.

    (See figure 2.3 in the paper
    https://stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf)

    Attributes:
      volume: A dataframe of volumes.
      sigma: A dataframe of daily volatilities.
      half_spread: A dataframe of bid-ask spreads divided by 2.
      nonlin_coeff: A dataframe of coefficients for the nonlinear cost.
      power: The nonlinear tcost power.
    """

    def __init__(self, half_spread, nonlin_coeff=0., sigma=0., volume=1.,
                 power=1.5):
        null_checker(half_spread)
        self.half_spread = half_spread
        null_checker(sigma)
        self.sigma = sigma
        null_checker(volume)
        self.volume = volume
        null_checker(nonlin_coeff)
        self.nonlin_coeff = nonlin_coeff
        null_checker(power)
        self.power = power
        super(TcostModel, self).__init__()

    def _estimate(self, t, w_plus, z, value):
        """Estimate tcosts given trades.

        Args:
          t: time of estimate
          z: trades
          value: portfolio value

        Returns:
          An expression for the tcosts.
        """

        try:
            z = z[z.index != self.cash_key]
            z = z.values
        except AttributeError:
            z = z[:-1]  # TODO fix when cvxpy pandas ready

        constr = []

        second_term = values_in_time(self.nonlin_coeff, t) * values_in_time(
            self.sigma, t) * (value / values_in_time(self.volume, t)) ** (
            self.power - 1)

        # no trade conditions
        if np.isscalar(second_term):
            if np.isnan(second_term):
                constr += [z == 0]
                second_term = 0
        else:  # it is a pd series
            no_trade = second_term.index[second_term.isnull()]
            second_term[no_trade] = 0
            constr += [z[second_term.index.get_loc(tick)] == 0
                       for tick in no_trade]

        try:
            self.expression = cvx.multiply(
                values_in_time(self.half_spread, t), cvx.abs(z))
        except TypeError:
            self.expression = cvx.multiply(
                values_in_time(self.half_spread, t).values, cvx.abs(z))
        try:
            self.expression += cvx.multiply(second_term,
                                            cvx.abs(z) ** self.power)
        except TypeError:
            self.expression += cvx.multiply(
                second_term.values, cvx.abs(z) ** self.power)

        return cvx.sum(self.expression), constr

    def value_expr(self, t, h_plus, u):

        u_nc = u.iloc[:-1]
        self.tmp_tcosts = (
            np.abs(u_nc) * values_in_time(self.half_spread, t) +
            values_in_time(self.nonlin_coeff, t) * values_in_time(self.sigma,
                                                                  t) *
            np.abs(u_nc) ** self.power /
            (values_in_time(self.volume, t) ** (self.power - 1)))

        return self.tmp_tcosts.sum()

    def optimization_log(self, t):
        try:
            return self.expression.value
        except AttributeError:
            return np.nan

    def simulation_log(self, t):
        # TODO find another way
        return self.tmp_tcosts

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        """Returns the estimate at time t of tcost at time tau.
        """
        return self._estimate(t, w_plus, z, value)

    def est_period(self, t, tau_start, tau_end, w_plus, z, value):
        """Returns the estimate at time t of tcost over given period.
        """
        K = (tau_end - tau_start).days
        tcost, constr = self.weight_expr(t, None, z / K, value)
        return tcost * K, constr
