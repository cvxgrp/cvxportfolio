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

from __future__ import print_function
import collections
import numpy as np
import pandas as pd
import copy
from .policies import MultiPeriodOpt


def getFiscalQuarter(dt):
    """Convert a time to a fiscal quarter.
    """
    year = dt.year
    quarter = (dt.month - 1) // 3 + 1
    return "Q%i %s" % (quarter, year)


class SimulationResult():
    """A container for the result of a simulation.

    Attributes:
        h_next: A dataframe of holdings over time.
        u: A dataframe of trades over time.
        tcosts: A series of transaction costs over time.
        borrow_costs: A series of borrow costs over time.
    """
    def __init__(self, initial_portfolio, policy, cash_key, simulator,
                 simulation_times=None, PPY=252,
                 timedelta=pd.Timedelta("1 days")):
        """
        Initialize the result object.

        Args:
            initial_portfolio:
            policy:
            simulator:
            simulation_times:
            PPY:
        """
        self.PPY = PPY
        self.timedelta = timedelta
        self.initial_val = sum(initial_portfolio)
        self.initial_portfolio = copy.copy(initial_portfolio)
        self.cash_key = cash_key
        self.simulator = simulator
        self.policy = policy

    def summary(self):
        print(self._summary_string())

    def _summary_string(self):
        data = collections.OrderedDict({
            'Number of periods':
                self.u.shape[0],
            'Initial timestamp':
                self.h.index[0],
            'Final timestamp':
                self.h.index[-1],
            'Portfolio return (%)':
                self.returns.mean() * 100 * self.PPY,
            'Excess return (%)':
                self.excess_returns.mean() * 100 * self.PPY,
            'Excess risk (%)':
                self.excess_returns.std() * 100 * np.sqrt(self.PPY),
            'Sharpe ratio':
                self.sharpe_ratio,
            'Max. drawdown':
                self.max_drawdown,
            'Turnover (%)':
                self.turnover.mean() * 100 * self.PPY,
            'Average policy time (sec)':
                self.policy_time.mean(),
            'Average simulator time (sec)':
                self.simulation_time.mean(),
        })

        return (pd.Series(data=data).
                to_string(float_format='{:,.3f}'.format))

    def log_data(self, name, t, entry):
        try:
            getattr(self, name).loc[t] = entry
        except AttributeError:
            setattr(self, name,
                    (pd.Series if np.isscalar(entry) else
                     pd.DataFrame)(index=[t], data=[entry]))

    def log_policy(self, t, exec_time):
        self.log_data("policy_time", t, exec_time)
        # TODO mpo policy requires changes in the optimization_log methods
        if not isinstance(self.policy, MultiPeriodOpt):
            for cost in self.policy.costs:
                self.log_data("policy_" + cost.__class__.__name__,
                              t, cost.optimization_log(t))

    def log_simulation(self, t, u, h_next, risk_free_return, exec_time):
        self.log_data("simulation_time", t, exec_time)
        self.log_data("u", t, u)
        self.log_data("h_next", t, h_next)
        self.log_data("risk_free_returns", t, risk_free_return)
        for cost in self.simulator.costs:
            self.log_data("simulator_" + cost.__class__.__name__,
                          t, cost.simulation_log(t))

    @property
    def h(self):
        """
        Concatenate initial portfolio and h_next dataframe.

        Infers the timestamp of last element by increasing the final timestamp.
        """
        tmp = self.h_next.shift(1)
        tmp.iloc[0] = self.initial_portfolio
        # TODO fix
        # tmp.loc[self.h_next.index[-1] + self.timedelta]=self.h_next.iloc[-1]
        return tmp

    @property
    def v(self):
        """The value of the portfolio over time.
        """
        return self.h.sum(axis=1)

    @property
    def profit(self):
        """The profit made, in dollars."""
        return self.v[-1] - self.v[0]

    @property
    def w(self):
        """The weights of the portfolio over time."""
        return (self.h.T / self.v).T

    @property
    def leverage(self):
        """Portfolio leverage"""
        return np.abs(self.w).sum(1)

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
        """The returns R_t = (v_{t+1}-v_t)/v_t
        """
        val = self.v
        return pd.Series(data=val.values[1:] / val.values[:-1] - 1,
                         index=val.index[:-1])

    @property
    def growth_rates(self):
        """The growth rate log(v_{t+1}/v_t)"""
        return np.log(self.returns + 1)

    @property
    def annual_growth_rate(self):
        """The annualized growth rate PPY/T \sum_{t=1}^T log(v_{t+1}/v_t)
        """
        return self.growth_rates.sum() * self.PPY / self.growth_rates.size

    @property
    def annual_return(self):
        """The annualized return in percent.
        """
        ret = self.growth_rates
        return self._growth_to_return(ret.mean())

    def _growth_to_return(self, growth):
        """Convert growth to annualized percentage return.
        """
        return 100 * (np.exp(self.PPY * growth) - 1)

    def get_quarterly_returns(self, benchmark=None):
        """The annualized returns for each fiscal quarter.
        """
        ret = self.growth_rates
        quarters = ret.groupby(getFiscalQuarter).aggregate(np.mean)
        return self._growth_to_return(quarters)

    def get_best_quarter(self, benchmark=None):
        ret = self.get_quarterly_returns(benchmark)
        return (ret.argmax(), ret.max())

    def get_worst_quarter(self, benchmark=None):
        ret = self.get_quarterly_returns(benchmark)
        return (ret.argmin(), ret.min())

    @property
    def excess_returns(self):
        return self.returns - self.risk_free_returns

    @property
    def sharpe_ratio(self):
        return np.sqrt(self.PPY) * np.mean(self.excess_returns) / \
            np.std(self.excess_returns)

    @property
    def turnover(self):
        """Turnover ||u_t||_1/v_t
        """
        noncash_trades = self.u.drop(self.cash_key, axis=1)
        return np.abs(noncash_trades).sum(axis=1) / self.v

    @property
    def trading_days(self):
        """The fraction of days with nonzero turnover.
        """
        return (self.turnover.values > 0).sum() / self.turnover.size

    @property
    def max_drawdown(self):
        """The maximum peak to trough drawdown in percent.
        """
        val_arr = self.v.values
        max_dd_so_far = 0
        cur_max = val_arr[0]
        for val in val_arr[1:]:
            if val >= cur_max:
                cur_max = val
            elif 100 * (cur_max - val) / cur_max > max_dd_so_far:
                max_dd_so_far = 100 * (cur_max - val) / cur_max
        return max_dd_so_far
