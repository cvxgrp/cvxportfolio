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

from __future__ import print_function
import collections
import numpy as np
import pandas as pd
import copy

from .estimator import Estimator
from .utils import periods_per_year
import matplotlib.pyplot as plt

__all__ = ['BacktestResult']


def getFiscalQuarter(dt):
    """Convert a time to a fiscal quarter."""
    year = dt.year
    quarter = (dt.month - 1) // 3 + 1
    return "Q%i %s" % (quarter, year)


class BacktestResult(Estimator):
    """Holds the data from a Backtest and producs metrics and plots."""

    def __init__(self, universe, backtest_times, costs):
        """Initialization of backtest result."""
        self.h = pd.DataFrame(index=backtest_times,
                              columns=universe, dtype=float)
        self.u = pd.DataFrame(index=backtest_times,
                              columns=universe, dtype=float)
        self.z = pd.DataFrame(index=backtest_times,
                              columns=universe, dtype=float)
        self.costs = {cost.__class__.__name__: pd.Series(
            index=backtest_times, dtype=float) for cost in costs}
        self.policy_times = pd.Series(index=backtest_times, dtype=float)
        self.simulator_times = pd.Series(index=backtest_times, dtype=float)

    @property
    def PPY(self):
        return periods_per_year(self.h.index)

    @property
    def v(self):
        """The value of the portfolio over time."""
        return self.h.sum(axis=1)

    @property
    def profit(self):
        """The profit made, in dollars."""
        return self.v[-1] - self.v[0]

    @property
    def w(self):
        """The weights of the portfolio in time."""
        return (self.h.T / self.v).T

    @property
    def leverage(self):
        """Portfolio leverage"""
        return np.abs(self.w.iloc[:, :-1]).sum(1)

    @property
    def volatility(self):
        """The annualized, realized portfolio volatility."""
        return np.sqrt(self.PPY) * np.std(self.returns)

    @property
    def mean_return(self):
        """The annualized mean portfolio return."""
        return self.PPY * np.mean(self.returns)

    @property
    def returns(self):
        """The returns R_t = (v_{t+1}-v_t)/v_t"""
        val = self.v
        return pd.Series(
            data=val.values[1:] / val.values[:-1] - 1, index=val.index[:-1]
        )

    @property
    def excess_returns(self):
        return self.returns - self.cash_returns

    @property
    def growth_rates(self):
        """The growth rate log(v_{t+1}/v_t)"""
        return np.log(self.returns + 1)

    @property
    def excess_growth_rates(self):
        """The growth rate log(v_{t+1}/v_t)"""
        return np.log(self.excess_returns + 1)

    # @property
    # def annual_growth_rate(self):
    #     """The annualized growth rate PPY/T sum_{t=1}^T log(v_{t+1}/v_t)"""
    #     return self.growth_rates.mean() * self.PPY
    #
    # @property
    # def annual_return(self):
    #     """The annualized return in percent."""
    #     ret = self.growth_rates
    #     return self._growth_to_return(ret.mean())
    #
    # def _growth_to_return(self, growth_rates):
    #     """Convert growth to annualized percentage return."""
    #     return 100 * (np.exp(self.PPY * growth) - 1)

    # def get_quarterly_returns(self, benchmark=None):
    #     """The annualized returns for each fiscal quarter."""
    #     ret = self.growth_rates
    #     quarters = ret.groupby(getFiscalQuarter).aggregate(np.mean)
    #     return self._growth_to_return(quarters)
    #
    # def get_best_quarter(self, benchmark=None):
    #     ret = self.get_quarterly_returns(benchmark)
    #     return (ret.argmax(), ret.max())
    #
    # def get_worst_quarter(self, benchmark=None):
    #     ret = self.get_quarterly_returns(benchmark)
    #     return (ret.argmin(), ret.min())

    @property
    def sharpe_ratio(self):
        return (
            np.sqrt(self.PPY)
            * np.mean(self.excess_returns)
            / np.std(self.excess_returns)
        )

    @property
    def turnover(self):
        """Turnover ||u_t||_1/(2*v_t)"""
        return np.abs(self.u.iloc[:, :-1]).sum(axis=1) / (2*self.v.loc[self.u.index])

    @property
    def trading_days(self):
        """The fraction of days with nonzero turnover."""
        return (self.turnover.values > 0).sum() / self.turnover.size

    @property
    def drawdown(self):
        return -(1 - (self.v / self.v.cummax()))

    def plot(self, show=True, how_many_weights=7):
        """Make plots."""

        # value
        self.v.plot(figsize=(12, 5), label='Multi Period Optimization')
        plt.ylabel('USD')
        plt.yscale('log')
        plt.title('Total value of the portfolio in time')

        # weights
        biggest_weights = np.abs(self.w).mean(
        ).sort_values().iloc[-how_many_weights:].index
        self.w[biggest_weights].plot()
        plt.title('Largest weights of the portfolio in time')

        if show:
            plt.show()

    def __repr__(self):
        data = collections.OrderedDict({
            "Number of periods": self.u.shape[0],
            "Initial timestamp": self.h.index[0],
            "Universe size": self.h.shape[1],
            "Final timestamp": self.h.index[-1],
            "Total profit (PnL)": self.profit,
            "Annualized portfolio return (%)": self.returns.mean() * 100 * self.PPY,
            "Annualized excess return (%)": self.excess_returns.mean() * 100 * self.PPY,
            "Annualized excess risk (%)": self.excess_returns.std() * 100 * np.sqrt(self.PPY),
            "Sharpe ratio": self.sharpe_ratio,
            "Worst drawdown (%)": self.drawdown.min() * 100,
            "Average drawdown (%)": self.drawdown.mean() * 100,
            "Daily Turnover (%)": self.turnover.mean() * 100,
            "Annualized Turnover (%)": self.turnover.mean() * 100 * self.PPY,

            "Average leverage (%)": self.leverage.mean() * 100,
            "Max leverage (%)": self.leverage.max() * 100})

        data.update(collections.OrderedDict({f'Average of {cost} per period (bp)': (
            self.costs[cost]/self.v).mean()*1E4 for cost in self.costs}))

        data.update(collections.OrderedDict(
            {"Average policy time (sec)": self.policy_times.mean(),
             "Average simulator time (sec)": self.simulator_times.mean()
             }))

        return 'Backtest Result:\n' + pd.Series(data=data).to_string(float_format="{:,.3f}".format)
