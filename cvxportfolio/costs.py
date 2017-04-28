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
import pandas as pd
import numpy as np
import copy
from .expression import Expression
from .utils.data_management import *

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

    def __mul__(self,other):
        """Read the gamma parameter as a multiplication."""
        newobj=copy.copy(self)
        newobj.gamma *= other
        return newobj

    def __rmul__(self,other):
        """Read the gamma parameter as a multiplication."""
        return self.__mul__(other)


class HcostModel(BaseCost):
    """A model for holding costs.

    Attributes:
      borrow_costs: A dataframe of borrow costs.
      dividends: A dataframe of dividends.
    """

    def __init__(self, borrow_costs, dividends=0.):
        self.borrow_costs = borrow_costs
        self.dividends = dividends
        if null_checker(self.borrow_costs) or null_checker(self.dividends):
            raise Exception('the arguments contain NaNs') ## TODO write decorator for this
        super(HcostModel, self).__init__()

    def _estimate(self, t, w_plus, z, value):
        ## TODO make expression a vector not a scalar (like tcost)
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

        n=len(w_plus)
        self.expression = vector_locator(self.borrow_costs,t,n).values.T*cvx.neg(w_plus)
        self.expression -= vector_locator(self.dividends,t,n).values.T*w_plus

        return self.expression, []

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        return self._estimate(t,w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        n=len(h_plus)-1
        self.last_cost= np.dot(-vector_locator(self.borrow_costs,t,n).values.T,
                                np.minimum(0,h_plus.values[:-1]))
        self.last_cost -= np.dot(vector_locator(self.dividends,t,n).values.T,
                                h_plus.values[:-1])
        return self.last_cost

    def optimization_log(self,t):
        return self.expression.value

    def simulation_log(self,t):
        return self.last_cost


class TcostModel(BaseCost):
    """A model for transaction costs.

    Attributes:
      volume: A dataframe of volumes.
      sigma: A dataframe of daily volatities.
      spread: A dataframe of bid-ask spreads.
      nonlin_coeff: A dataframe of coefficients for the nonlinear cost.
      power: The nonlinear tcost power.
    """
    def __init__(self, spread, nonlin_coeff=0., volume=1., sigma=0.,
                 power=1.5):
        self.spread = spread
        ## up to here...
        self.nonlin_term=nonlin_term
        self.cash_key = cash_key
        if volume is not None:
            self.volume = volume[volume.columns.difference([cash_key])]
        else:
            self.volume = None
        if self.nonlin_term:
            self.sigma = sigma[sigma.columns.difference([cash_key])]
            self.nonlin_coeff = nonlin_coeff[nonlin_coeff.columns.difference([cash_key])]
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

        z_abs = cvx.abs(z)

        constr = []

        if self.nonlin_term:
            tmp = self.nonlin_coeff.loc[t] * self.sigma.loc[t] * (value / self.volume.loc[t])**(self.power - 1)
            assert (z.size[0] == tmp.size)
            # if volume was 0 don't trade
            no_trade = tmp.index[tmp.isnull()]
            for ticker in no_trade:
                locator=tmp.index.get_loc(ticker)
                constr.append(z[locator]==0)
            tmp.loc[tmp.isnull()] = 0.
        assert (z.size[0] == self.spread.loc[t].size)

        self.expression = z_abs.T*self.spread.loc[t].values
        if self.nonlin_term:
            self.expression+=((z_abs)**self.power).T*tmp.values

        assert (self.expression)
        return self.expression, constr

    def value_expr(self, t, h_plus, u):
        # TODO figure out why calling weight_expr is buggy
        value=sum(h_plus)
        u_normalized = u/value
        abs_u = np.abs(u_normalized[:-1])

        tcosts = self.spread.loc[t]*abs_u
        if self.nonlin_term:
            tcosts+= self.nonlin_coeff.loc[t] * \
                 self.sigma.loc[t] * (abs_u**self.power)/((self.volume.loc[t]/value)**(self.power-1))

        self.tmp_tcosts=tcosts*value

        return self.tmp_tcosts.sum()

    def optimization_log(self,t):
        try:
            return self.expression.value.A1
        except AttributeError:
            return np.nan

    def simulation_log(self,t):
        ## TODO find another way
        return self.tmp_tcosts

    def _estimate_ahead(self, t, tau, w_plus, z, value):
        """Returns the estimate at time t of tcost at time tau.
        """
        return self._estimate(t,w_plus, z, value)

    def est_period(self, t, tau_start, tau_end, w_plus, z, value):
        """Returns the estimate at time t of tcost over given period.
        """
        K = (tau_end - tau_start).days
        tcost, constr= self.weight_expr(t, None, z / K, value)
        return tcost * K, constr
