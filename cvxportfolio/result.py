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


# def getFiscalQuarter(dt):
#     """Convert a time to a fiscal quarter."""
#     year = dt.year
#     quarter = (dt.month - 1) // 3 + 1
#     return "Q%i %s" % (quarter, year)


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
    def cash_key(self):
        return self.h.columns[-1]

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
        return self.v.iloc[-1] - self.v.iloc[0]

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
    def sharpe_ratio(self):
        return (np.sqrt(self.PPY)
                * np.mean(self.excess_returns)
                / np.std(self.excess_returns)
                )

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

    @staticmethod
    def _print_growth_rate(gr):
        """Transform growth rate into return and pretty-print it.

        Either prints in basis points, percentage, or
        multiplication.
        """
        ret = np.exp(gr)-1
        if np.abs(ret) < 0.005:
            return f'{int(ret*1E4):d}bp'
        if np.abs(gr) < 1:
            return f'{ret*100:.2f}%'
        return f'{1+ret:.1f}X'

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
        
         # US Letter size
        fig, axes = plt.subplots(3, figsize=(8.5, 11), layout='constrained')
        fig.suptitle('Back-test result')

        # value
        self.v.plot(label='Portfolio value', ax=axes[0])
        axes[0].set_ylabel(self.cash_key)
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', which="both")

        # weights
        biggest_weights = np.abs(self.w).mean(
            ).sort_values().iloc[-how_many_weights:].index
        self.w[biggest_weights].plot(ax=axes[1])
        axes[1].set_ylabel(f'Largest {how_many_weights} weights')
        axes[1].grid(True, linestyle='--')
        
        # leverage / turnover
        self.leverage.plot(ax=axes[2], linestyle='--', color='k', label='Leverage')
        self.turnover.plot(ax=axes[2], linestyle='-', color='r', label='Turnover')
        axes[2].legend()
        axes[2].grid(True, linestyle='--')
        
        if show:
            fig.show()
            
    def __repr__(self):
        
        stats = collections.OrderedDict({

            "Universe size": self.h.shape[1],
            "Initial timestamp": self.h.index[0],
            "Final timestamp": self.h.index[-1],
            "Number of periods": self.u.shape[0],
            
            ' '*3:'',
            f"Initial value ({self.cash_key})": f"{self.v.iloc[0]:.3e}",
            f"Final value ({self.cash_key})": f"{self.v.iloc[-1]:.3e}",
            f"Profit ({self.cash_key})": f"{self.profit:.3e}",
            
            ' '*4:'',
            "Absolute return (annualized)": f"{100 * self.mean_return:.1f}%",
            "Absolute risk (annualized)": f"{100 * self.volatility:.1f}%",
            "Excess return (annualized)": f"{self.excess_returns.mean() * 100 * self.PPY:.1f}%",
            "Excess risk (annualized)": f"{self.excess_returns.std() * 100 * np.sqrt(self.PPY):.1f}%",

            ' '*5:'',
            "Avg. growth rate (absolute)": self._print_growth_rate(self.growth_rates.mean()),
            "Avg. growth rate (excess)": self._print_growth_rate(self.excess_growth_rates.mean()),

        })
        
        if len(self.costs):
            stats[' '*6]=''
        for cost in self.costs:
            stats[f'Avg. {cost}'] = f"{(self.costs[cost]/self.v).mean()*1E4:.0f}bp"
            stats[f'Max. {cost}'] = f"{(self.costs[cost]/self.v).max()*1E4:.0f}bp"
        
        stats.update(collections.OrderedDict({
            
            ' '*7:'',
            "Sharpe ratio": f"{self.sharpe_ratio:.2f}",
            
            ' '*8:'',
            
            "Avg. drawdown": f"{self.drawdown.mean() * 100:.1f}%",
            "Min. drawdown": f"{self.drawdown.min() * 100:.1f}%",
            
            "Avg. leverage": f"{self.leverage.mean() * 100:.1f}%",
            "Max. leverage": f"{self.leverage.max() * 100:.1f}%",
            
            "Avg. turnover": f"{self.turnover.mean() * 100:.1f}%",
            "Max. turnover": f"{self.turnover.max() * 100:.1f}%",
            
            ' '*9:'',

            "Avg. policy time": f"{self.policy_times.mean():.3f}s",
            "Avg. simulator time": f"{self.simulator_times.mean():.3f}s",
            "Total time": f"{self.simulator_times.sum() + self.policy_times.sum():.3f}s",
            }))

        content = pd.Series(stats).to_string()
        lenline = len(content.split('\n')[0])

        return '\n' + '#'*lenline + '\n' + content + '\n' + '#'*lenline + '\n' 
        
