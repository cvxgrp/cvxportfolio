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
from cvxportfolio.expression import Expression
__all__ = ['AlphaSource', 'MPOAlphaSource', 'AlphaStream']


class BaseAlphaModel(Expression):
    pass


class AlphaSource(BaseAlphaModel):
    """A single alpha estimateion.

    Attributes:
      alpha_data: A dataframe of return estimates.
      delta_data: A confidence interval around the estimates.
      half_life: Number of days for alpha auto-correlation to halve.
    """

    def __init__(self, alpha_data, delta_data=None, gamma_decay=None, name=None):
        self.alpha_data = alpha_data
        # TODO input check goes here
        assert (not self.alpha_data.isnull().values.any())
        self.delta_data = delta_data
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
        # idx = self.alpha_data.index.get_loc(t, method='pad')
        # alpha_vec = self.alpha_data.iloc[idx]
        alpha = self.alpha_data.loc[t].values.T*wplus
        if self.delta_data is not None:
            alpha -= self.delta_data.loc[t].values.T*cvx.abs(wplus)
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
        # if isinstance(tau, tuple):
        #     tau_start, tau_end = tau
        # else:
        #     tau_start = tau
        #     tau_end = tau + pd.Timedelta('1 days')
        alpha = self.weight_expr(t, wplus)
        if tau > t  and self.gamma_decay is not None:
            alpha *= (tau-t).days**(-self.gamma_decay)
            # decay_init = 2**(-(tau_start - t).days/self.half_life)
            # K = (tau_end - tau_start).days ## in all our calls K = 1 because tau is not a tuple
            # decay_factor = 2**(-1/self.half_life)
            # decay = decay_init*(1 - decay_factor**K)/(1 - decay_factor)
            # alpha *= decay
        return alpha
    
class MPOAlphaSource(BaseAlphaModel):
    """A single alpha estimateion.

    Attributes:
      alpha_data: A dict of serieses of return estimates.
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
        return self.alpha_data[(t,tau)].values.T*wplus


class AlphaStream(BaseAlphaModel):
    """A weighted combination of alpha sources.

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
            alpha += source.weight_expr_ahead(t, tau, wplus) * self.weights[idx]
        return alpha
