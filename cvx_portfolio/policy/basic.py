"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, Blackrock Inc.

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


import numpy as np
from .base_policy import BasePolicy


class Hold(BasePolicy):
    """Hold initial portfolio.
    """
    def get_trades(self, portfolio, t):
        return self._nulltrade(portfolio)


class BaseRebalance(BasePolicy):

    def _rebalance(self, portfolio):
        return sum(portfolio) * self.target - portfolio


class PeriodicRebalance(BaseRebalance):
    """Track a target portfolio, rebalancing at given times.
    """
    def __init__(self, target, rebalancing_times, name="PeriodicRebalance"):
        """

        Args:
            target: target weights, n+1 vector
            rebalancing_times: iterable/set of datetime objects, times at which we want to rebalance
        """
        self.target = target
        self.rebalancing_times = rebalancing_times
        self.name = name

    def get_trades(self, portfolio, t):
        if t in self.rebalancing_times:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)


class AdaptiveRebalance(BaseRebalance):
    """ Rebalance portfolio when deviates too far from target.
    """
    def __init__(self, target, tracking_error):
        self.target = target
        self.tracking_error = tracking_error

    def get_trades(self, portfolio, t):
        weights=portfolio/sum(portfolio)
        diff = (weights - self.target).values

        if np.linalg.norm(diff, 2) > self.tracking_error:
            return self._rebalance(portfolio)
        else:
            return self._nulltrade(portfolio)
