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
"""This module defines benchmark portfolio weights in time that are used,
for example, by risk terms of optimization-based policies.
"""

import numpy as np
import pandas as pd

from .estimator import PolicyEstimator, DataEstimator

__all__ = ['Benchmark', 'CashBenchmark', 'UniformBenchmark', 'MarketBenchmark']


class BaseBenchmark(PolicyEstimator):
    """Base class for cvxportfolio benchmark weights."""
    pass


class Benchmark(BaseBenchmark, DataEstimator):
    """User-provided benchmark.

    :param benchmark_weights: benchmark weights, either constant in
        time (pd.Series indexed by assets) or varying in time
        (pd.DataFrame indexed by time and whose columns are the assets).
    :type benchmark_weights: pd.Series or pd.DataFrame

    """

    def __init__(self, benchmark_weights):
        DataEstimator.__init__(self, 
            benchmark_weights, 
            data_includes_cash=True)


class CashBenchmark(BaseBenchmark):
    """Default benchmark weights for cvxportfolio risk models.
    """

    def _pre_evaluation(self, universe, backtest_times):
        """Define current_value as a constant."""
        self.current_value = np.zeros(len(universe))
        self.current_value[-1] = 1.


class UniformBenchmark(BaseBenchmark):
    """Benchmark weights uniform on non-cash assets.
    """

    def _pre_evaluation(self, universe, backtest_times):
        """Define current_value as a constant."""
        self.current_value = np.ones(len(universe))
        self.current_value[-1] = 0.
        self.current_value /= np.sum(self.current_value[:-1])


class MarketBenchmark(BaseBenchmark):
    """Portfolio weighted by last year's total volumes.
    """

    def _values_in_time(self, past_volumes, **kwargs):
        """Update current_value using past year's volumes."""
        sumvolumes = past_volumes.loc[past_volumes.index >= (
            past_volumes.index[-1] - pd.Timedelta('365d'))].mean()
        result = np.zeros(len(sumvolumes) + 1)
        result[:-1] = sumvolumes / sum(sumvolumes)
        return result
