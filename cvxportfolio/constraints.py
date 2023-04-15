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
"""This module implements realistic constraints to be used with SinglePeriodOptimization
and MultiPeriodOptimization policies, or other Cvxpy-based policies.
"""


import cvxpy as cvx
import numpy as np

# from .utils import values_in_time
from .estimator import CvxpyExpressionEstimator, ParameterEstimator


__all__ = [
    "LongOnly",
    "LeverageLimit",
    "LongCash",
    "DollarNeutral",
    "ParticipationRateLimit",
    "MaxWeights",
    "MinWeights",
    "FactorMaxLimit",
    "FactorMinLimit",
    "FixedFactorLoading",
]


class BaseConstraint(CvxpyExpressionEstimator):
    """Base cvxpy constraint class."""
    pass
    # def __init__(self, **kwargs):
    #     self.w_bench = kwargs.pop("w_bench", 0.0)
    #
    
    ## DEFINED TEMPORARILY TO INTERFACE WITH OLD CVXPORTFOLIO
    def weight_expr(self, t, w_plus, z, v):
        
        self.pre_evaluation(None, None, t, None)
        result = self.compile_to_cvxpy(wplus, z, v)
        self.values_in_time(t)
        return result
        
        
        # """Returns a list of trade constraints.
        #
        # Args:
        #   t: time
        #   w_plus: post-trade weights
        #   z: trade weights
        #   v: portfolio value
        # """
        # if w_plus is None:
        #     return self._weight_expr(t, None, z, v)
        # return self._weight_expr(t, w_plus - self.w_bench, z, v)
    #
    # def _weight_expr(self, t, w_plus, z, v):
    #     raise NotImplementedError
    

class BaseTradeConstraint(BaseConstraint):
    """Base class for constraints that operate on trades."""
    pass
    
class BaseWeightConstraint(BaseConstraint):
    """Base class for constraints that operate on weights.
    
    Here we can implement a method to pass benchmark weights 
    and make the constraint relative to it rather than to the null
    portfolio.
    """
    pass
    


class ParticipationRateLimit(BaseTradeConstraint):
    """A limit on maximum trades size as a fraction of market volumes.
    
    Attributes:
        self.volumes (ParameterEstimator): ParameterEstimator with point-in-time market volumes estimations
        self.max_fraction_of_volumes (ParameterEstimator): ParameterEstimator with point-in-time,
             and also possibly per-stock, requirements of maximum participation rate
        
    """

    def __init__(self, volumes, max_fraction_of_volumes=0.05):
        self.volumes = ParameterEstimator(volumes)
        self.max_participation_rate = ParameterEstimator(max_fraction_of_volumes)
        # self.children = [self.volumes, self.max_participation_rate]


    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [
            cvx.multiply(cvx.abs(z[:-1]), portfolio_value) <= cvx.multiply(self.volumes.parameter, self.max_participation_rate.parameter)
        ]
        
    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of trade constraints.
    #
    #     Args:
    #       t: time
    #       w_plus: post-trade weights
    #       z: trade weights
    #       v: portfolio value
    #     """
    #     return [
    #         cvx.abs(z[:-1]) * v <= self.volumes.parameter * self.max_fraction_of_volumes.parameter
    #     ]


class LongOnly(BaseWeightConstraint):
    """A long only constraint."""

    # def __init__(self, **kwargs):
    #     super(LongOnly, self).__init__(**kwargs)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [w_plus[:-1] >= 0]


class LeverageLimit(BaseWeightConstraint):
    """A limit on leverage.

    Attributes:
      limit: A (time) series or scalar giving the leverage limit.
    """

    def __init__(self, limit):#, **kwargs):
        self.limit = ParameterEstimator(limit)
        # self.children = [self.limit]
        #super(LeverageLimit, self).__init__(**kwargs)
        
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [cvx.norm(w_plus[:-1], 1) <= self.limit.parameter]

    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #       t: time
    #       w_plus: holdings
    #     """
    #     return [cvx.norm(w_plus[:-1], 1) <= values_in_time(self.limit, t)]


class LongCash(BaseWeightConstraint):
    """Requires that cash be non-negative."""

    # def __init__(self, **kwargs):
    #     super(LongCash, self).__init__(**kwargs)
    
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [w_plus[-1] >= 0]

    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #       t: time
    #       w_plus: holdings
    #     """
    #     return [w_plus[-1] >= 0]


class DollarNeutral(BaseWeightConstraint):
    """Long-short dollar neutral strategy."""

    # def __init__(self, **kwargs):
    #     super(DollarNeutral, self).__init__(**kwargs)
    
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        # return [w_plus[-1] == 1] # check that this one is equivalent
        return [sum(w_plus[:-1]) == 0]

    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #       t: time
    #       w_plus: holdings
    #     """
    #     return [sum(w_plus[:-1]) == 0]


class MaxWeights(BaseWeightConstraint):
    """A max limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit):#, **kwargs):
        self.limit = ParameterEstimator(limit)
        #self.children = [self.limit]
        #super(MinWeights, self).__init__(**kwargs)
        
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [w_plus[:-1] <= self.limit.parameter]

    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #       t: time
    #       w_plus: holdings
    #     """
    #     return [w_plus[:-1] <= values_in_time(self.limit, t)]


class MinWeights(BaseWeightConstraint):
    """A min limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit):#, **kwargs):
        self.limit = ParameterEstimator(limit)
        #self.children = [self.limit]
        #super(MinWeights, self).__init__(**kwargs)
        
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [w_plus[:-1] >= self.limit.parameter]

    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #       t: time
    #       w_plus: holdings
    #     """
    #     return [w_plus[:-1] >= values_in_time(self.limit, t)]


class FactorMaxLimit(BaseWeightConstraint):
    """A max limit on portfolio-wide factor (e.g. beta) exposure.

    Attributes:
        factor_exposure: An (n * r) matrix giving the factor exposure per asset
        per factor, where n represents # of assets and r represents # of factors
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit):#, **kwargs):
        #super(FactorMaxLimit, self).__init__(**kwargs)
        self.factor_exposure = ParameterEstimator(factor_exposure)
        self.limit = ParameterEstimator(limit)
        #self.children = [self.limit, self.factor_exposure]
        
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [self.factor_exposure.parameter.T @ w_plus[:-1] <= self.limit.parameter]

    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #         t: time
    #         w_plus: holdings
    #     """
    #     return [
    #         values_in_time(self.factor_exposure, t).T @ w_plus[:-1]
    #         <= values_in_time(self.limit, t)
    #     ]


class FactorMinLimit(BaseWeightConstraint):
    """A min limit on portfolio-wide factor (e.g. beta) exposure.

    Attributes:
        factor_exposure: An (n * r) matrix giving the factor exposure per asset
        per factor, where n represents # of assets and r represents # of factors
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit):#, **kwargs):
        #super(FactorMaxLimit, self).__init__(**kwargs)
        self.factor_exposure = ParameterEstimator(factor_exposure)
        self.limit = ParameterEstimator(limit)
        #self.children = [self.limit, self.factor_exposure]
        
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [self.factor_exposure.parameter.T @ w_plus[:-1] >= self.limit.parameter]
        
    # def __init__(self, factor_exposure, limit, **kwargs):
    #     super(FactorMinLimit, self).__init__(**kwargs)
    #     self.factor_exposure = factor_exposure
    #     self.limit = limit
    #
    # def _weight_expr(self, t, w_plus, z, v):
    #     """Returns a list of holding constraints.
    #
    #     Args:
    #         t: time
    #         w_plus: holdings
    #     """
    #     return [
    #         values_in_time(self.factor_exposure, t).T @ w_plus[:-1]
    #         >= values_in_time(self.limit, t)
    #     ]


class FixedFactorLoading(BaseWeightConstraint):
    """A constraint to fix portfolio loadings to a set of factors. 
    
    This can be used to impose market neutrality, a certain portfolio-wide alpha, ....

    Attributes:
        factor_exposure: An (n * r) matrix giving the factor exposure on each
        factor
        target: A series or number giving the targeted factor loading
    """
    
    def __init__(self, factor_exposure, target):#, **kwargs):
        #super(FactorMaxLimit, self).__init__(**kwargs)
        self.factor_exposure = ParameterEstimator(factor_exposure)
        self.target = ParameterEstimator(target)
        #self.children = [self.target, self.factor_exposure]
        
    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a list of Cvxpy constraints."""
        return [self.factor_exposure.parameter.T @ w_plus[:-1] == self.target.parameter]
    
    #
    # def __init__(self, return_forecast, alpha_target, **kwargs):
    #     super(FixedAlpha, self).__init__(**kwargs)
    #     self.return_forecast = return_forecast
    #     self.alpha_target = alpha_target
    #
    # def _weight_expr(self, t, w_plus, z, v):
    #     return [
    #         values_in_time(self.return_forecast, t).T @ w_plus[:-1]
    #         == values_in_time(self.alpha_target, t)
    #     ]