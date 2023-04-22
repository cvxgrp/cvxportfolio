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
    
    #INITIALIZED = False # used to interface w/ old cvxportfolio

    # interface to old cvxportfolio
    def weight_expr(self, t, w_plus, z, v):
        #if not self.INITIALIZED:
        self.pre_evaluation(None, None, t, None)
        self.legacy_expression = self.compile_to_cvxpy(w_plus, z, v)
        #self.INITIALIZED = True
        self.values_in_time(t, None, None, None, None)
        if hasattr(self.legacy_expression, "__iter__"):
            return self.legacy_expression
        else:
            return [self.legacy_expression]


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
        self.max_participation_rate = ParameterEstimator(
            max_fraction_of_volumes)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return cvx.multiply(cvx.abs(z[:-1]), portfolio_value) <= cvx.multiply(
            self.volumes, self.max_participation_rate
        )


class LongOnly(BaseWeightConstraint):
    """A long only constraint."""

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return w_plus[:-1] >= 0


class LeverageLimit(BaseWeightConstraint):
    """A limit on leverage.

    Attributes:
      limit: A (time) series or scalar giving the leverage limit.
    """

    def __init__(self, limit):
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return cvx.norm(w_plus[:-1], 1) <= self.limit


class LongCash(BaseWeightConstraint):
    """Requires that cash be non-negative."""

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return w_plus[-1] >= 0


class DollarNeutral(BaseWeightConstraint):
    """Long-short dollar neutral strategy."""

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return w_plus[-1] == 1


class MaxWeights(BaseWeightConstraint):
    """A max limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit):
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return w_plus[:-1] <= self.limit


class MinWeights(BaseWeightConstraint):
    """A min limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit):
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return w_plus[:-1] >= self.limit


class FactorMaxLimit(BaseWeightConstraint):
    """A max limit on portfolio-wide factor (e.g. beta) exposure.

    Args:
        factor_exposure: An (n * r) matrix giving the factor exposure per asset
        per factor, where n represents # of assets and r represents # of factors
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit):
        self.factor_exposure = ParameterEstimator(factor_exposure)
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return self.factor_exposure.T @ w_plus[:-1] <= self.limit


class FactorMinLimit(BaseWeightConstraint):
    """A min limit on portfolio-wide factor (e.g. beta) exposure.

    Args:
        factor_exposure: An (n * r) matrix giving the factor exposure per asset
        per factor, where n represents # of assets and r represents # of factors
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit):
        self.factor_exposure = ParameterEstimator(factor_exposure)
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return self.factor_exposure.T @ w_plus[:-1] >= self.limit


class FixedFactorLoading(BaseWeightConstraint):
    """A constraint to fix portfolio loadings to a set of factors.

    This can be used to impose market neutrality, a certain portfolio-wide alpha, ....

    Attributes:
        factor_exposure: An (n * r) matrix giving the factor exposure on each
        factor
        target: A series or number giving the targeted factor loading
    """

    def __init__(self, factor_exposure, target):
        self.factor_exposure = ParameterEstimator(factor_exposure)
        self.target = ParameterEstimator(target)

    def compile_to_cvxpy(self, w_plus, z, portfolio_value):
        """Return a Cvxpy constraint."""
        return self.factor_exposure.T @ w_plus[:-1] == self.target
