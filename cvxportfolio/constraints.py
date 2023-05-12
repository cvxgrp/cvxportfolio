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
"""This module implements realistic constraints to be used with SinglePeriodOptimization
and MultiPeriodOptimization policies, or other Cvxpy-based policies.
"""


import cvxpy as cvx
import numpy as np

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
    

    :param volumes: per-stock and per-day market volume estimates, or constant in time
    :type volumes: pd.Series or pd.DataFrame
    :param max_fraction_of_volumes: max fraction of market volumes that we're allowed to trade
    :type max_fraction_of_volumes: float, pd.Series, pd.DataFrame
    """

    def __init__(self, volumes, max_fraction_of_volumes=0.05):
        self.volumes = ParameterEstimator(volumes)
        self.max_participation_rate = ParameterEstimator(
            max_fraction_of_volumes)
        self.portfolio_value = cvx.Parameter(nonneg=True)

    def values_in_time(self, current_portfolio_value, **kwargs):
        self.portfolio_value.value = current_portfolio_value
        super().values_in_time(current_portfolio_value=current_portfolio_value, **kwargs)
        
    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return cvx.multiply(cvx.abs(z[:-1]), self.portfolio_value) <= cvx.multiply(
            self.volumes, self.max_participation_rate
        )


class LongOnly(BaseWeightConstraint):
    """A long only constraint.
    
    Imposes that at each point in time the post-trade
    weights are non-negative.
    """

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return w_plus[:-1] >= 0


class LeverageLimit(BaseWeightConstraint):
    """A limit on leverage.
    
    Leverage is defined as the :math:`\ell_1` norm of non-cash
    post-trade weights. Here we require that it is smaller than
    a given value

    :param limit: constant or varying in time leverage limit
    :type limit: float or pd.Series
    """

    def __init__(self, limit):
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return cvx.norm(w_plus[:-1], 1) <= self.limit


class LongCash(BaseWeightConstraint):
    """Requires that cash be non-negative."""

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return w_plus[-1] >= 0


class DollarNeutral(BaseWeightConstraint):
    """Long-short dollar neutral strategy."""

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return w_plus[-1] == 1


class MaxWeights(BaseWeightConstraint):
    """A max limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit):
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return w_plus[:-1] <= self.limit


class MinWeights(BaseWeightConstraint):
    """A min limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit):
        self.limit = ParameterEstimator(limit)

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
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

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
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

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
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

    def compile_to_cvxpy(self, w_plus, z, w_plus_minus_w_bm):
        """Return a Cvxpy constraint."""
        return self.factor_exposure.T @ w_plus[:-1] == self.target
