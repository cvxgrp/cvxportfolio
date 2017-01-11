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
import pandas as pd
import copy


def getFiscalQuarter(dt):
    """Convert a time to a fiscal quarter.
    """
    year = dt.year
    quarter = (dt.month-1) // 3 + 1
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
                simulation_times=None, PPY=252):
        """
        Initialize the result object.

        Args:
            initial_portfolio:
            policy:
            simulator:
            simulation_times:
            PPY:
        """

        # constants
        self.PPY = PPY
        self.initial_val = sum(initial_portfolio)
        self.initial_portfolio = copy.copy(initial_portfolio)
        self.cash_key = cash_key

        # data
        self._data = pd.Panel(items=['u','h_next'],
                              major_axis=simulation_times,
                              minor_axis=initial_portfolio.index)

        # simulator
        self.simulator = simulator
        self.simulation_time = pd.Series(index=simulation_times)
        for cost in self.simulator.costs:
            self._data['simulation.'+cost.__class__.__name__] = np.nan

        # policy
        self.policy = policy
        self.pol_name = policy.name
        self.policy_time = pd.Series(index=simulation_times)
        for cost in self.policy.costs:
            self._data['policy.'+cost.__class__.__name__] = np.nan

    @property
    def u(self):
        return self._data.u

    @property
    def h_next(self):
        return self._data.h_next

    def log_policy(self, t, exec_time):
        self.policy_time.loc[t] = exec_time
        for cost in self.policy.costs:
            self._data['policy.'+cost.__class__.__name__].loc[t] = \
                cost.optimization_log(t)

    def log_simulation(self, t, u, next_h, exec_time):
        self.simulation_time.loc[t] = exec_time
        self._data['u'].loc[t] = u
        self._data['h_next'].loc[t] = next_h
        for cost in self.simulator.costs:
            self._data['simulation.'+cost.__class__.__name__].loc[t] = \
                cost.simulation_log(t)

    @property
    def market_returns(self):
        # TODO log them?
        raise NotImplemented

    @property
    def h(self):
        """
        Concatenate initial portfolio and h_next dataframe.

        Infers the timestamp of last element by increasing the final timestamp.
        """
        return self.h_next  # TODO implement
        # raise NotImplemented

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

    def _summary_dataframes(self):
        result = {}
        result['Basics'] = pd.DataFrame(index=['Date', 'Value', 'Cash'], #, 'Cash (%)', 'Leverage'],
                               columns=['Start', 'End', 'Delta'],
                               data=[[self.h_next.index[0], self.h_next.index[-1],
                                      len(self.h_next.index)],
                                     [self.value[0], self.value[-1], self.value[-1] - self.value[0]],
                                         [self.h_next.iloc[0][self.cash_key],
                                          self.h_next.iloc[-1][self.cash_key],
                                          self.h_next.iloc[-1][self.cash_key] -
                                          self.h_next.iloc[0][self.cash_key]],
                                    ])

        result['Performance'] = pd.DataFrame(index=['Ret. (Ann. %)',
                                                    'Vol. (Ann. %)',
                                                    'Sharpe',
                                                    'IR',
                                                    'Turnover (Ann. %)',
                                                    'Max Drawdown (%)',
                                                    'P&L ($)',
                                                    'TCosts ($)',
                                                    'HCosts ($)'],
                                             columns=[self.pol_name],
                                             data=np.array([self.annual_return,
                                                     self.realized_volatility,
                                                     self.sharpe_ratio,
                                                     self.information_ratio,
                                                     self.turnover.mean()*100*self.PPY,
                                                     self.max_drawdown,
                                                     self.profit,
                                                     sum(self.opt_TcostModel),
                                                     sum(self.opt_HcostModel)]).T)
        return result

    @property
    def tcosts(self):
        return self.opt_TcostModel.sum(axis=1)  # TODO this is tmp hack

    def summary(self):
        """Pretty print summary."""
        # TODO this doesn't work when outside ipython
        try:
            from IPython.core import display as ICD
            ICD.display(self._summary_dataframes())
        except ImportError:
            print(self.__repr__())

    def summary_line(self):
        return self._summary_dataframes()['Performance']

    def __repr__(self):
        """Print basic statistics."""
        result = ''
        sum_dict = self._summary_dataframes()
        for k in sum_dict.keys():
            result += k + '\n'
            result += str(sum_dict[k]) + '\n'
        return result

    @property
    def realized_volatility(self):
        """The annualized, realized portfolio volatility, in percent."""
        return 100 * np.sqrt(self.PPY) * np.std(self.active_returns)

    # @property
    # def active_weights(self):
    #     if self.benchmark is None:
    #         return None
    #     else:
    #         return self.weights - self.benchmark

    @property
    def cash(self):
        return self.h_next[self.cash_key]

    @property
    def noncash(self):
        return self.h_next.drop(self.cash_key, axis=1)

    @property
    def returns(self):
        """The returns (v_{t+1}-v_t)/v_t
        """
        val = self.value
        return pd.Series(data=val.values[1:]/val.values[:-1] - 1, index=val.index[1:])

    @property
    def growth_rates(self):
        """The growth rate log(v_{t+1}/v_t)"""
        return np.log(self.active_returns + 1)

    @property
    def annual_growth_rate(self):
        """The annualized growth rate PPY/T \sum_{t=1}^T log(v_{t+1}/v_t)
        """
        return self.growth_rates.sum()*self.PPY/self.growth_rates.size

    @property
    def annual_return(self):
        """The annualized return in percent.
        """
        ret = self.growth_rates
        return self._growth_to_return(ret.mean())

    def _growth_to_return(self, growth):
        """Convert growth to annualized percentage return.
        """
        return 100*(np.exp(self.PPY*growth) - 1)

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

    # @property
    # def active_returns(self):
    #     return self.returns - self.benchmark_returns

    @property
    def excess_returns(self):
        return self.returns - self.market_returns.ix[:, self.cash_key]

    # TODO maybe annualize stuff
    @property
    def sharpe_ratio(self):
        return np.sqrt(self.PPY) * np.mean(self.excess_returns) / np.std(self.excess_returns)

    # TODO maybe annualize stuff
    @property
    def information_ratio(self):
        return np.sqrt(self.PPY) * np.mean(self.active_returns) / np.std(self.active_returns)

    @property
    def turnover(self):
        """Turnover ||u_t||_1/v_t
        """
        noncash_trades = self.u.drop(self.cash_key, axis=1)
        return np.abs(noncash_trades).sum(axis=1)/self.value

    @property
    def trading_days(self):
        """The fraction of days with nonzero turnover.
        """
        return (self.turnover.values > 0).sum()/self.turnover.size

    #TODO
    # Best/worst quarter.
    # Logger calculate all this. log.best_quarter -> (Q1 2012, 10%)

    @property
    def max_drawdown(self):
        """The maximum peak to trough drawdown in percent.
        """
        val_arr = self.value.values
        max_dd_so_far = 0
        cur_max = val_arr[0]
        for val in val_arr[1:]:
            if val >= cur_max:
                cur_max = val
            elif 100*(cur_max - val)/cur_max > max_dd_so_far:
                max_dd_so_far = 100*(cur_max - val)/cur_max
        return max_dd_so_far
