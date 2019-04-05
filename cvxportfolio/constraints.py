"""
Copyright (C) Enzo Busseti 2016-2019

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Code written before September 2016 is copyrighted to 
Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.,
and is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from abc import ABCMeta, abstractmethod

import cvxpy as cvx
import numpy as np
import pandas as pd

from .risks import locator
from .utils.data_management import time_locator

__all__ = ['LongOnly', 'LeverageLimit', 'LongCash', 'DollarNeutral', 'MaxTrade',
           'MaxWeights', 'MinWeights', 'FactorMaxLimit', 'FactorMinLimit',
           'FixedAlpha']


class BaseConstraint(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.w_bench = kwargs.pop('w_bench', 0.)

    def weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: post-trade weights
          z: trade weights
          v: portfolio value
        """
        if w_plus is None:
            return self._weight_expr(t, None, z, v)
        return self._weight_expr(t, w_plus - self.w_bench, z, v)

    @abstractmethod
    def _weight_expr(self, t, w_plus, z, v):
        pass


class MaxTrade(BaseConstraint):
    """A limit on maximum trading size.
    """

    def __init__(self, ADVs, max_fraction=0.05, **kwargs):
        self.ADVs = ADVs
        self.max_fraction = max_fraction
        super(MaxTrade, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of trade constraints.

        Args:
          t: time
          w_plus: post-trade weights
          z: trade weights
          v: portfolio value
        """
        return cvx.abs(z[:-1]) * v <= \
            np.array(locator(self.ADVs, t)) * self.max_fraction

        # TODO fix the locator for this usecase


class LongOnly(BaseConstraint):
    """A long only constraint.
    """

    def __init__(self, **kwargs):
        super(LongOnly, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus >= 0


class LeverageLimit(BaseConstraint):
    """A limit on leverage.

    Attributes:
      limit: A (time) series or scalar giving the leverage limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(LeverageLimit, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        if isinstance(self.limit, pd.Series):
            limit = self.limit.loc[t]
        else:
            limit = self.limit
        return cvx.norm(w_plus[:-1], 1) <= limit


class LongCash(BaseConstraint):
    """Requires that cash be non-negative.
    """

    def __init__(self, **kwargs):
        super(LongCash, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return w_plus[-1] >= 0


class DollarNeutral(BaseConstraint):
    """Long-short dollar neutral strategy.
    """

    def __init__(self, **kwargs):
        super(DollarNeutral, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        return sum(w_plus[:-1]) == 0


class MaxWeights(BaseConstraint):
    """A max limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(MaxWeights, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        if isinstance(self.limit, pd.Series):
            limit = self.limit.loc[t]
        else:
            limit = self.limit
        return w_plus[:-1] <= limit


class MinWeights(BaseConstraint):
    """A min limit on weights.

    Attributes:
      limit: A series or number giving the weights limit.
    """

    def __init__(self, limit, **kwargs):
        self.limit = limit
        super(MinWeights, self).__init__(**kwargs)

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
          t: time
          w_plus: holdings
        """
        if isinstance(self.limit, pd.Series):
            limit = self.limit.loc[t]
        else:
            limit = self.limit
        return w_plus[:-1] >= limit


class FactorMaxLimit(BaseConstraint):
    """A max limit on portfolio-wide factor (e.g. beta) exposure.

    Attributes:
        factor_exposure: An (T * N) matrix giving the specific one factor exposure by N assets
        and T periods, where T represents # of periods and N represents # of assets
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit, **kwargs):
        super(FactorMaxLimit, self).__init__(**kwargs)
        self.factor_exposure = factor_exposure
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
            t: time
            w_plus: holdings
        """
        if isinstance(self.limit, pd.Series):
            limit = self.limit.loc[t]
        else:
            limit = self.limit
        return time_locator(self.factor_exposure, t).values.T * w_plus[:-1] <= limit


class FactorMinLimit(BaseConstraint):
    """A min limit on portfolio-wide factor (e.g. beta) exposure.

    Attributes:
        factor_exposure: An (T * N) matrix giving the specific one factor exposure by N assets
        and T periods, where T represents # of periods and N represents # of assets
        limit: A series of list or a single list giving the factor limits
    """

    def __init__(self, factor_exposure, limit, **kwargs):
        super(FactorMinLimit, self).__init__(**kwargs)
        self.factor_exposure = factor_exposure
        self.limit = limit

    def _weight_expr(self, t, w_plus, z, v):
        """Returns a list of holding constraints.

        Args:
            t: time
            w_plus: holdings
        """
        if isinstance(self.limit, pd.Series):
            limit = self.limit.loc[t]
        else:
            limit = self.limit
        return time_locator(self.factor_exposure, t).values.T * w_plus[:-1] >= limit


class FixedAlpha(BaseConstraint):
    """A constraint to fix portfolio-wide alpha

    Attributes:
        forecast_returns: An (n * 1) vector giving the return forecast on each
        asset
        alpha_target: A series or number giving the targeted portfolio return
    """

    def __init__(self, return_forecast, alpha_target, **kwargs):
        super(FixedAlpha, self).__init__(**kwargs)
        self.return_forecast = return_forecast
        self.alpha_target = alpha_target

    def _weight_expr(self, t, w_plus, z, v):
        if isinstance(self.alpha_target, pd.Series):
            alpha_target = self.alpha_target.loc[t]
        else:
            alpha_target = self.alpha_target
        return self.return_forecast.T * w_plus[:-1] == alpha_target
