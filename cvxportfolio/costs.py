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

Currently these are two: TransactionCostModel and HoldingCostModel. 

(In previous versions of Cvxportfolio these were used in the simulator as well,
but now instead we include their logic in the simulator itself.)

We do our best to include parameters that are accurate for the real market.
It should be easy to adjust these to other settings by following the description 
provided in the class definitions.
"""

import cvxpy as cvx
import numpy as np
import copy
# from .expression import Expression
from .utils import null_checker, values_in_time
from .estimator import CvxpyExpressionEstimator, ParameterEstimator

__all__ = ["HcostModel", "TcostModel"]


class BaseCost(CvxpyExpressionEstimator):
    """Base class for cost objects.
    
    It will use the CvxpyExpressionEstimator to compile the cost object to
    a cvxpy expression for optimization-based policies.
    
    It also overloads the values_in_time method to be used by simulator classes.
    """
    
    #gamma = 1. # this will be removed
    

    ## PLACEHOLDER METHOD TO USE OLD INTERFACE WITH NEW INTERFACE
    def weight_expr(self, t, w_plus, z, value):
        cost, constr = self._estimate(t, w_plus, z, value)
        return cost, constr
        
    def _estimate(self, t, w_plus, z, value):
        """Temporary interface to old cvxportfolio."""
        self.pre_evaluation(None, None, t, None)
        cost = self.compile_to_cvxpy(w_plus, z, value)
        self.values_in_time(t)
        return cost, []

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        return self.weight_expr(t, w_plus, z, value)

    def __mul__(self, other):
        """Multiply by constant."""
        if not np.isscalar(other):
            raise SyntaxError('You can only multiply cost by a scalar.')
        return CombinedCosts([self], [other])
        # copied = copy.deepcopy(self)
        # copied.gamma *= other
        # return copied

    def __rmul__(self, other):
        """Multiply by constant."""
        return self.__mul__(other)
        
    def __add__(self, other):
        """Add cost expression to another cost expression.
        
        Idea is to create a new CombinedCost class that
        implements `compile_to_cvxpy` and values_in_time
        by summing over costs.
        
        """
        return CombinedCosts([self, other], [1., 1.])
        
    def __radd__(self, other):
        """Add cost expression to another cost."""
        return self.__add__(other)
        
    def __neg__(self):
        """Take negative of cost."""
        return CombinedCosts([self], [-1.])
    
    def __sub__(self, other):
        """Subtract other cost."""
        return self.__add__(-other)
        
    def __rsub__(self, other):
        """Subtract from other cost."""
        return other.__add__(-self)


class CombinedCosts(BaseCost):
    """Class obtained by algebraic combination of Cost classes.
    
    Attributes:
        costs (list): a list of BaseCost instances
    """
    
    def __init__(self, costs, multipliers):
        for cost in costs:
            if not isinstance(cost, BaseCost):
                raise SyntaxError('You can only sum `BaseCost` instances to other `BaseCost` instances.')
        self.costs = costs
        self.multipliers = multipliers
        
    def pre_evaluation(self,  *args, **kwargs):
        """Iterate over constituent costs."""
        [el.pre_evaluation(*args, **kwargs) for el in self.costs]
        
    def values_in_time(self, *args, **kwargs):
        """Iterate over constituent costs."""
        [el.values_in_time(*args, **kwargs) for el in self.costs]
    
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Iterate over constituent costs."""
        return sum([multiplier * cost.compile_to_cvxpy(w_plus, z, portfolio_value) for multiplier, cost in zip(self.multipliers, self.costs)])
        
    ## TEMPORARY IN ORDER NOT TO BREAK TESTS
    ## THESE METHODS ARE DEPRECATED
        
    def optimization_log(self, t):
        return sum([multiplier * (cost.expression.value if hasattr(cost, 'expression') else 0.) for multiplier, cost in zip(self.multipliers, self.costs)])

    def simulation_log(self, t):
        return sum([multiplier * (cost.last_cost if hasattr(cost, 'last_cost') else 0.) for multiplier, cost in zip(self.multipliers, self.costs)])
        
    
        
class HcostModel(BaseCost):
    """A model for holding costs.

    Attributes:
      borrow_costs: A dataframe of borrow costs.
      dividends: A dataframe of dividends.
    """

    def __init__(self, borrow_costs, dividends=0.0):
        self.borrow_costs = ParameterEstimator(borrow_costs, non_negative=True)
        self.dividends = ParameterEstimator(dividends)
        
        
    def compile_to_cvxpy(self, w_plus, z, value):
        """Compile cost to cvxpy expression."""
        self.expression = cvx.multiply(self.borrow_costs.parameter, cvx.neg(w_plus[:-1]))
        self.expression -= cvx.multiply(self.dividends.parameter, w_plus[:-1])
        
        return cvx.sum(self.expression)
    

    def value_expr(self, t, h_plus, u):
        """Placeholder method as we update the rest of the stack to new interface."""
        
        self.pre_evaluation(None, None, t, None)
        self.values_in_time(t)
        
        self.last_cost = -np.minimum(0, h_plus.iloc[:-1]) * self.borrow_costs.parameter.value
        self.last_cost -= h_plus.iloc[:-1] * self.dividends.parameter.value

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
      power (float): The nonlinear tcost power.
    """

    def __init__(self, half_spread, nonlin_coeff=0.0, sigma=0.0, volume=1.0, power=1.5):
        null_checker(half_spread)
        self.half_spread = half_spread
        null_checker(sigma)
        self.sigma = sigma
        null_checker(volume)
        self.volume = volume
        null_checker(nonlin_coeff)
        self.nonlin_coeff = nonlin_coeff
        null_checker(power)
        self.power: float = power

    def _estimate(self, t, w_plus, z, value):
        """Estimate tcosts given trades.

        Args:
          t: time of estimate
          z: trades
          value: portfolio value

        Returns:
          An expression for the tcosts.
        """

        z = z[:-1] 

        constr = []

        second_term = (
            values_in_time(self.nonlin_coeff, t)
            * values_in_time(self.sigma, t)
            * (value / values_in_time(self.volume, t)) ** (self.power - 1)
        )

        # no trade conditions
        if np.isscalar(second_term):
            if np.isnan(second_term):
                constr += [z == 0]
                second_term = 0
        else:  # it is a pd series
            no_trade = second_term.index[second_term.isnull()]
            second_term[no_trade] = 0
            constr += [z[second_term.index.get_loc(tick)] == 0 for tick in no_trade]

        try:
            self.expression = cvx.multiply(
                values_in_time(self.half_spread, t), cvx.abs(z)
            )
        except TypeError:
            self.expression = cvx.multiply(
                values_in_time(self.half_spread, t).values, cvx.abs(z)
            )
        try:
            self.expression += cvx.multiply(second_term, cvx.abs(z) ** self.power)
        except TypeError:
            self.expression += cvx.multiply(
                second_term.values, cvx.abs(z) ** self.power
            )

        return cvx.sum(self.expression), constr

    def value_expr(self, t, h_plus, u):
        u_nc = u.iloc[:-1]
        self.tmp_tcosts = np.abs(u_nc) * values_in_time(
            self.half_spread, t
        ) + values_in_time(self.nonlin_coeff, t) * values_in_time(
            self.sigma, t
        ) * np.abs(
            u_nc
        ) ** self.power / (
            values_in_time(self.volume, t) ** (self.power - 1)
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
