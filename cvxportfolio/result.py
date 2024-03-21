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
"""This module defines :class:`BacktestResult`.

This is the object that is returned by the
:meth:`cvxportfolio.MarketSimulator.backtest`
method, and also by the same method in derived classes of
:class:`cvxportfolio.MarketSimulator`.
It contains all relevant information from a back-test and implements the logic
to compute various performance metrics, in addition to the
:meth:`BacktestResult.plot` method for producing plots
and ``__repr__`` magic method, which is invoked when the user
prints an instance.

.. versionadded:: 1.1.0
    The :attr:`BacktestResult.log` property, which returns the logs produced
    during the back-test, at level ``INFO`` or higher. It works also for
    back-tests run in parallel!
"""


from __future__ import annotations, print_function

import collections
import logging
import time
from io import StringIO
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import periods_per_year_from_datetime_index

__all__ = ['BacktestResult']

# Module level constants, should be exposed to user (move to configuration.py?)
RECORD_LOGS = 'INFO'
LOG_FORMAT = '| %(asctime)s | %(levelname)s | process:%(process)d | %(pathname)s:%(lineno)s | %(message)s '


logger = logging.getLogger(__name__)

# def getFiscalQuarter(dt):
#     """Convert a time to a fiscal quarter."""
#     year = dt.year
#     quarter = (dt.month - 1) // 3 + 1
#     return "Q%i %s" % (quarter, year)


# pylint: disable=too-many-public-methods
class BacktestResult:
    """Store the data from a back-test and produce metrics and plots.

    Additionally, record all logs produced by the simulator, market data
    server, and policy object during the back-test. These are stored in the
    ``logs`` attribute as a newline separated string. This is done in a
    multi-process safe manner, so that if you run parallel back-tests with
    :meth:`cvxportfolio.MarketSimulator.backtest_many`, only the logs from the
    right process are recorded.

    :param universe: Best initial guess of the trading universe.
    :type universe: pandas.Index
    :param trading_calendar: Trading calendar. Can be a best guess, but the
        first timestamp **must be** the actual.
    :type trading_calendar: pd.DateTimeIndex
    :param costs: Simulator cost objects whose value is logged. Note: we use
        only the classes' **names** here.
    :type costs: list

    .. note::

        The initializer of this class is still experimental, we might still
        change its signature without the guarantee of semantic versioning.
    """

    def __init__(self, universe, trading_calendar, costs):
        """Initialization of back-test result."""
        timer = time.time()
        self._h = pd.DataFrame(index=trading_calendar,
                              columns=universe, dtype=float)
        self._u = pd.DataFrame(index=trading_calendar,
                              columns=universe, dtype=float)
        self._z = pd.DataFrame(index=trading_calendar,
                              columns=universe, dtype=float)
        self.costs = {cost.__class__.__name__: pd.Series(
            index=trading_calendar, dtype=float) for cost in costs}
        self._policy_times = pd.Series(index=trading_calendar, dtype=float)
        self._simulator_times = pd.Series(index=trading_calendar, dtype=float)
        self._market_data_times = pd.Series(
            index=trading_calendar, dtype=float)
        self._result_times = pd.Series(
            index=trading_calendar, dtype=float)
        self._simulator_times = pd.Series(
            index=trading_calendar, dtype=float)
        self._cash_returns = pd.Series(index=trading_calendar, dtype=float)
        self._benchmark_returns = pd.Series(
            index=trading_calendar, dtype=float)
        self._current_universe = pd.Index(universe)
        self._indexer = np.arange(len(universe), dtype=int)
        self._init_timer = time.time() - timer

    def __enter__(self):
        """Set up logging context to record back-test logs."""
        # we do this because you can also use the class without logging
        # pylint: disable=attribute-defined-outside-init

        # record logs
        self._log = ''
        self._root_logger = logging.getLogger()

        # We modify the root logger to filter at our chosen level (or lower
        # if its was lower) and pre-existing handlers (notably the
        # stderr one) to filter at the level of the original logger, or theirs
        # if higher. We put them back to their initial state at the end.
        self._orig_rootlogger_level = self._root_logger.level
        try:
            _ = logging.getLevelNamesMapping()[RECORD_LOGS]
        # for Py < 3.11
        except AttributeError: # pragma: no cover
            _ = logging._nameToLevel[RECORD_LOGS] # pragma: no cover
        self._root_logger.setLevel(min(_, self._root_logger.level))
        self._orig_loghandlers_levels = []
        for pre_existing_handler in self._root_logger.handlers:
            self._orig_loghandlers_levels.append(pre_existing_handler.level)
            pre_existing_handler.setLevel(
                max(self._orig_rootlogger_level, pre_existing_handler.level))

        # add stream handler that we use
        self._log_stream = StringIO()
        self._log_stream_handler = logging.StreamHandler(
            stream=self._log_stream)
        self._log_stream_handler.setLevel(RECORD_LOGS)
        self._root_logger.addHandler(self._log_stream_handler)

        # logs formatting
        formatter = logging.Formatter(LOG_FORMAT)
        self._log_stream_handler.setFormatter(formatter)

        return self

    @property
    def _current_full_universe(self):
        """Helper property used by ``_change_universe``.

        :returns: Current full universe (including assets that were seen
            in the past but have been dropped).
        :rtype: pandas.Index
        """
        return self._h.columns

    def _change_universe(self, new_universe):
        """Change current universe (columns of dataframes) during back-test."""

        # print('new universe')
        # print(new_universe)
        # print('old universe')
        # print(self._current_universe)

        # if necessary, expand columns of dataframes
        if not new_universe.isin(self._current_universe).all():

            # check that cash key didn't change!
            assert new_universe[-1] == self._current_universe[-1]

            # if new universe is larger we use it as ordering
            # this is the default situation with yfinance data
            # careful (thanks gh PR #114) because we need the
            # _current_full_universe, otherwise we drop assets that
            # are not traded any more
            if self._current_full_universe.isin(new_universe).all():
                joined = new_universe

            # otherwise we lose the ordering :(
            else:
                logger.info(
                    "%s joining new universe with old",
                    self.__class__.__name__)
                joined = pd.Index(
                    # need to join with full, not current!
                    sorted(set(self._current_full_universe[:-1]
                        ).union(new_universe[:-1])))
                joined = joined.append(new_universe[-1:])

            self._h = self._h.reindex(columns = joined)
            self._u = self._u.reindex(columns = joined)
            self._z = self._z.reindex(columns = joined)

        assert new_universe.isin(self._h.columns).all()
        self._current_universe = new_universe
        self._indexer = self._h.columns.get_indexer(new_universe)

    #pylint: disable=too-many-arguments
    def log_trading(self, t: pd.Timestamp,
        h: pd.Series[float], u: pd.Series[float],
        z: pd.Series[float], costs: Dict[str, float],
        cash_return: float, benchmark_return: float or None,
        policy_time: float, simulator_time: float, market_data_time: float):
        """Log one trading period.

        :param t: Timestamp of execution.
        :type t: pd.Timestamp
        :param h: Initial holdings.
        :type h: pd.Series
        :param u: Trade vectors in (*e.g.*) dollars.
        :type u: pd.Series
        :param z: Trade weight vectors requested by the policy. Can be
            different from the actual trades because of rounding or any other
            filtering applied by the simulator. They are recorded but not used
            in accounting.
        :type z: pd.Series
        :param costs: Dictionary indexed by the cost class names with the
            current values of each. They are recorded but not used for
            accounting, that is done directly in the simulator and already
            included in the computed holdings.
        :type costs: dict
        :param cash_return: Current return of the cash account (interest rate),
            used for excess metrics (*e.g.*, Sharpe ratio).
        :type cash_return: float
        :param benchmark_return: Current return of the benchmark, if defined,
            otherwise None.
        :type benchmark_return: float or None
        :param policy_time: Time spent inside the policy object in this period.
        :type policy_time: float
        :param simulator_time: Time spent to back-test this period outside of
            the policy object.
        :type simulator_time: float
        :param market_data_time: Time spent inside
            :meth:`cvxportfolio.data.MarketData` for this period (also already
            included in ``simulator_time``).
        :type market_data_time: float

        .. note::

            This method is still experimental, we might change its signature
            without the guarantee of semantic versioning.
        """
        timer = time.time()
        if not h.index.equals(self._current_universe):
            self._change_universe(h.index)

        tidx = self._h.index.get_loc(t)

        self._h.iloc[tidx, self._indexer] = h
        self._u.iloc[tidx, self._indexer] = u
        self._z.iloc[tidx, self._indexer] = z

        for cost in costs:
            self.costs[cost].iloc[tidx] = costs[cost]

        self._simulator_times.iloc[tidx] = simulator_time
        self._policy_times.iloc[tidx] = policy_time
        self._market_data_times.iloc[tidx] = market_data_time
        self._cash_returns.iloc[tidx] = cash_return
        if benchmark_return is not None:
            self._benchmark_returns.iloc[tidx] = benchmark_return
        self._result_times.iloc[tidx] = time.time() - timer + self._init_timer
        self._init_timer = 0.

    def log_final(self, t, t_next, h, extra_simulator_time):
        """Log final elements and (if necessary) clean up.

        Also clean up if the back-test finishes before it was expected to
        (*e.g.*, bankruptcy).

        :param t: Last execution time.
        :type t: pd.Timestamp
        :param t_next: Next execution time, at which we don't run the policy
            but we do record the holdings.
        :type t_next: pd.Timestamp
        :param h: Last value of the holdings, at time ``t_next``.
        :type h: pd.Series
        :param extra_simulator_time: Any extra time, in seconds, spent in the
            simulator to propagate the holdings from t to t_next (other times
            are already accounted for).
        :type extra_simulator_time: float

        .. note::

            This method is still experimental, we might change its signature
            without the guarantee of semantic versioning.
        """
        self._h.loc[t_next] = h
        self._simulator_times.loc[t] += extra_simulator_time
        # in case of bankruptcy
        if t_next < self._h.index[-1]:
            tidx = self._h.index.get_loc(t_next)
            self._h = self._h.iloc[:tidx+1]
            self._u = self._u.iloc[:tidx]
            self._z = self._z.iloc[:tidx]
            self._simulator_times = self._simulator_times.iloc[:tidx]
            self._policy_times = self._policy_times.iloc[:tidx]
            self._market_data_times = self._market_data_times.iloc[:tidx]
            self._result_times = self._result_times.iloc[:tidx]

            self._cash_returns = self._cash_returns.iloc[:tidx]
            self._benchmark_returns = self._benchmark_returns.iloc[:tidx]

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up logging context manager.

        TODO: we'll also do something with exceptions
        """

        # logging output
        self._log = self._log_stream.getvalue()

        # put back root logger and handlers to initial state
        self._root_logger.setLevel(self._orig_rootlogger_level)
        self._root_logger.removeHandler(self._log_stream_handler)
        for i, handler in enumerate(self._root_logger.handlers):
            handler.setLevel(self._orig_loghandlers_levels[i])

        # delete logging helpers (unnecessary?)
        del self._log_stream_handler
        self._log_stream.close()
        del self._log_stream

    #
    # General backtest information
    #

    @property
    def logs(self):
        """Logs from the policy, simulator, market data server, ....

        :return: Logs produced during the back-test, newline separated.
        :rtype: str
        """
        return self._log

    @property
    def policy_times(self):
        """The computation time of the policy object at each period.

        :returns: Policy time in seconds at each period.
        :rtype: pandas.Series
        """
        return pd.Series(self._policy_times)

    @property
    def simulator_times(self):
        """The computation time of the simulator object at each period.

        :returns: Simulator time in seconds at each period.
        :rtype: pandas.Series
        """
        return pd.Series(self._simulator_times)

    @property
    def market_data_times(self):
        """The computation time of the market data server at each period.

        This is already included in ``simulator_times`` !

        :returns: Market data server time in seconds at each period.
        :rtype: pandas.Series
        """
        return pd.Series(self._market_data_times)

    @property
    def result_times(self):
        """The computation time of the back-test result (this) at each period.

        This is already included in ``simulator_times`` !

        :returns: Back-test result time in seconds at each period.
        :rtype: pandas.Series
        """
        return pd.Series(self._result_times)

    @property
    def cash_returns(self):
        """Per-period returns on cash (*i.e.*, the risk-free rate).

        :returns: All cash returns.
        :rtype: pandas.Series
        """
        return pd.Series(self._cash_returns)

    @property
    def benchmark_returns(self):
        """Benchmark returns per period (if the policy has a benchmark).

        :returns: All benchmark returns, if defined, else ``nan``.
        :rtype: pandas.Series
        """
        return pd.Series(self._benchmark_returns)

    @property
    def cash_key(self):
        """The name of the cash unit used (e.g., USDOLLAR).

        :returns: Name of the cash accounting unit.
        :rtype: str
        """
        return self._h.columns[-1]

    @property
    def periods_per_year(self):
        """Average trading periods per year in this backtest (rounded).

        :returns: Average periods per year.
        :rtype: int
        """
        return periods_per_year_from_datetime_index(self._h.index)

    #
    # Basic portfolio variables, defined in Chapter 2
    #

    @property
    def h(self):
        """The portfolio (holdings) at each trading period (including the end).

        :returns: Holdings at each period, includes cash account.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self._h)

    @property
    def u(self):
        """The portfolio trade vector at each trading period.

        :returns: Trades at each period.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self._u)

    @property
    def z(self):
        """The portfolio trade weights at each trading period.

        :returns: Trades weights at each period.
        :rtype: pandas.DataFrame
        """
        return (self.u.T / self.v).T

    @property
    def z_policy(self):
        r"""The trade weights requested by the policy at each trading period.

        This is different from the trade weights :math:`z` because the
        :class:`MarketSimulator` instance may change it by enforcing
        the self-financing condition (recalculates cash value), rounding
        trades to integer number of shares, canceling trades on assets whose
        volume is zero for the day, :math:`\ldots`.

        :returns: Trades weights requested by the policy at each period.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self._z)

    @property
    def v(self):
        """The total value (or NAV) of the portfolio at each period.

        :returns: Total value at each period.
        :rtype: pandas.Series
        """
        return self.h.sum(axis=1)

    @property
    def initial_value(self):
        """The initial value (or NAV) of the portfolio.

        :returns: Value at the start of the back-test.
        :rtype: float
        """
        return self.v.iloc[0]

    @property
    def final_value(self):
        """The portfolio value (or NAV) at the end of the back-test.

        :returns: Final value of the portfolio.
        :rtype: float
        """
        return self.v.iloc[-1]

    @property
    def profit(self):
        """The total profit (PnL) in this backtest.

        :returns: Total profit.
        :rtype: float
        """
        return self.v.iloc[-1] - self.v.iloc[0]

    @property
    def w(self):
        """The weights of the portfolio at each period.

        :returns: Portfolio weights at each period.
        :rtype: pandas.DataFrame
        """
        return (self.h.T / self.v).T

    @property
    def h_plus(self):
        """The post-trade portfolio (holdings) at each period.

        :returns: Post-trade holdings at each period.
        :rtype: pandas.DataFrame
        """
        return self.h.loc[self.u.index] + self.u

    @property
    def w_plus(self):
        """The post-trade weights of the portfolio at each period.

        :returns: Post-trade weights at each period.
        :rtype: pandas.DataFrame
        """
        return (self.h_plus.T / self.v).T

    @property
    def leverage(self):
        r"""Leverage of the portfolio at each period.

        This is defined as:

        .. math::

            \| {(h_t)}_{1:n} \|_1 / v_t,

        where :math:`h_t` is the portfolio (the holdings) at time :math:`t`,
        we exclude the cash account from the :math:`\ell_1` norm,
        and :math:`v_t` is the total value (NAV) of the portfolio
        at time :math:`t`.

        :returns: Leverage at each period.
        :rtype: pandas.Series
        """
        return np.abs(self.w.iloc[:, :-1]).sum(1)

    @property
    def turnover(self):
        r"""The turnover of the portfolio at each period.

        This is defined as:

        .. math::

            \| {(u_t)}_{1:n} \|_1 / (2 v_t),

        where :math:`u_t` are the portfolio trades at time :math:`t`,
        we exclude the cash account from the :math:`\ell_1` norm,
        and :math:`v_t` is the total value (NAV) of the portfolio
        at time :math:`t`.

        :returns: Turnover at each period.
        :rtype: pandas.Series
        """
        return np.abs(self.u.iloc[:, :-1]).sum(axis=1) / (
            2*self.v.loc[self.u.index])

    @property
    def returns(self):
        r"""The portfolio returns at each period.

        This is defined as:

        .. math::

            R_t^\text{p} = \frac{v_{t+1} - v_t}{v_t}

        in terms of the portfolio value (NAV).

        :returns: Portfolio returns at each period.
        :rtype: pandas.Series
        """
        val = self.v
        return pd.Series(
            data=val.values[1:] / val.values[:-1] - 1, index=val.index[:-1]
        )

    #
    # Absolute metrics, defined in Chapter 3 Section 1
    #

    @property
    def average_return(self):
        r"""The average realized return :math:`\overline{R^\text{p}}`.

        :returns: Average portfolio return.
        :rtype: float
        """
        return np.mean(self.returns)

    @property
    def annualized_average_return(self):
        r"""The average realized return, annualized.

        :returns: Average portfolio return, annualized.
        :rtype: float
        """
        return self.average_return * self.periods_per_year

    @property
    def growth_rates(self):
        r"""The growth rate (or log-return) of the portfolio at each period.

        This is defined as:

        .. math::

            G^\text{p}_t = \log (v_{t+1} / v_t) = \log(1 + R^\text{p}_t).

        :returns: Growth rate of the portfolio value at each period.
        :rtype: pandas.Series
        """
        return np.log(self.returns + 1)

    @property
    def average_growth_rate(self):
        r"""The average portfolio growth rate :math:`\overline{G^\text{p}}`.

        :returns: Average growth rate.
        :rtype: float
        """
        return np.mean(self.growth_rates)

    @property
    def annualized_average_growth_rate(self):
        r"""The average portfolio growth rate, annualized.

        :returns: Average growth rate, annualized.
        :rtype: float
        """
        return self.average_growth_rate * self.periods_per_year

    @property
    def volatility(self):
        """Realized volatility (standard deviation of the portfolio returns).

        :returns: Volatility.
        :rtype: float
        """
        return np.std(self.returns)

    @property
    def annualized_volatility(self):
        """Realized volatility, annualized.

        :returns: Volatility, annualized.
        :rtype: float
        """
        return self.volatility * np.sqrt(self.periods_per_year)

    @property
    def quadratic_risk(self):
        """Quadratic risk, square of the realized volatility.

        :returns: Quadratic risk.
        :rtype: float
        """
        return self.volatility ** 2

    @property
    def annualized_quadratic_risk(self):
        """Quadratic risk, annualized.

        :returns: Quadratic risk, annualized.
        :rtype: float
        """
        return self.quadratic_risk * self.periods_per_year

    #
    # Metrics relative to benchmark, defined in Chapter 3 Section 2
    #

    @property
    def active_returns(self):
        """Portfolio returns minus benchmark returns (if defined by policy).

        :returns: Active returns at each period if benchmark is defined,
            else ``nan``.
        :rtype: pandas.Series
        """
        return self.returns - self.benchmark_returns

    @property
    def average_active_return(self):
        r"""The average active return :math:`\overline{R^\text{a}}`.

        :returns: Average active portfolio return if benchmark is defined,
            else ``nan``.
        :rtype: float
        """
        return np.mean(self.active_returns)

    @property
    def annualized_average_active_return(self):
        """The average active return, annualized.

        :returns: Average active portfolio return, annualized. If benchmark is
            not defined, ``nan``.
        :rtype: float
        """
        return self.average_active_return * self.periods_per_year

    @property
    def active_volatility(self):
        """Active volatility (standard deviation of the active returns).

        :returns: Average active volatility if benchmark is defined,
            else ``nan``.
        :rtype: float
        """
        return np.std(self.active_returns)

    @property
    def annualized_active_volatility(self):
        """Annualized active volatility.

        :returns: Average active volatility, annualized. If benchmark is
            not defined, ``nan``.
        :rtype: float
        """
        return self.active_volatility * np.sqrt(self.periods_per_year)

    @property
    def excess_returns(self):
        """Excess portfolio returns with respect to the cash returns.

        :returns: Excess returns at each period.
        :rtype: pandas.Series
        """
        return self.returns - self.cash_returns

    @property
    def average_excess_return(self):
        r"""The average excess return :math:`\overline{R^\text{e}}`.

        :returns: Average excess portfolio return.
        :rtype: float
        """
        return np.mean(self.excess_returns)

    @property
    def annualized_average_excess_return(self):
        """The average excess return, annualized.

        :returns: Average excess portfolio return, annualized.
        :rtype: float
        """
        return self.average_excess_return * self.periods_per_year

    @property
    def excess_volatility(self):
        """Excess volatility (standard deviation of the excess returns).

        :returns: Average excess volatility.
        :rtype: float
        """
        return np.std(self.excess_returns)

    @property
    def annualized_excess_volatility(self):
        """Annualized excess volatility.

        :returns: Average excess volatility, annualized.
        :rtype: float
        """
        return self.excess_volatility * np.sqrt(self.periods_per_year)

    @property
    def sharpe_ratio(self):
        r"""Sharpe ratio (using annualized excess portfolio returns).

        This is defined as

        .. math::

            \text{SR} = \overline{R^\text{e}}/\sigma^\text{e}

        where :math:`\overline{R^\text{e}}` is the average excess portfolio
        return and :math:`\sigma^\text{e}` its standard deviation. Both are
        annualized.

        :returns: Sharpe Ratio.
        :rtype: float
        """
        return self.annualized_average_excess_return / (
            self.annualized_excess_volatility + 1E-8)

    @property
    def information_ratio(self):
        r"""Information ratio (using annualized active portfolio returns).

        This is defined as

        .. math::

            \text{IR} = \overline{R^\text{a}}/\sigma^\text{a}

        where :math:`\overline{R^\text{a}}` is the average active portfolio
        return and :math:`\sigma^\text{a}` its standard deviation. Both are
        annualized.

        :returns: Information Ratio, ``nan`` if benchmark is not defined.
        :rtype: float
        """
        return self.annualized_average_active_return / (
            self.annualized_active_volatility + 1E-8)

    @property
    def excess_growth_rates(self):
        r"""The growth rate of the portfolio, relative to cash.

        This is defined as:

        .. math::

            G^\text{e}_t = \log(1 + R^\text{e}_t)

        where :math:`R^\text{e}_t` are the excess portfolio returns.

        :returns: Excess growth rates at each period.
        :rtype: pandas.Series
        """
        return np.log(self.excess_returns + 1)

    @property
    def active_growth_rates(self):
        r"""The growth rate of the portfolio, relative to benchmark.

        This is defined as:

        .. math::

            G^\text{a}_t = \log(1 + R^\text{a}_t)

        where :math:`R^\text{a}_t` are the active portfolio returns.

        :returns: Active growth rates at each period. If benchmark is
            not defined, ``nan``.
        :rtype: pandas.Series
        """
        return np.log(self.active_returns + 1)

    @property
    def average_excess_growth_rate(self):
        r"""The average excess growth rate :math:`\overline{G^\text{e}}`.

        :returns: Average excess portfolio growth rates.
        :rtype: float
        """
        return np.mean(self.excess_growth_rates)

    @property
    def annualized_average_excess_growth_rate(self):
        """The average excess growth rate, annualized.

        :returns: Average excess portfolio growth rates, annualized.
        :rtype: float
        """
        return self.average_excess_growth_rate * self.periods_per_year

    @property
    def average_active_growth_rate(self):
        r"""The average active growth rate :math:`\overline{G^\text{a}}`.

        :returns: Average active portfolio growth rates. If benchmark is not
            defined, ``nan``.
        :rtype: float
        """
        return np.mean(self.active_growth_rates)

    @property
    def annualized_average_active_growth_rate(self):
        """The average active growth rate, annualized.

        :returns: Average active portfolio growth rates, annualized. If
            benchmark is not defined, ``nan``.
        :rtype: float
        """
        return self.average_active_growth_rate * self.periods_per_year

    @property
    def drawdown(self):
        """The drawdown of the portfolio value over time.

        :returns: Drawdown of portfolio value at each period.
        :rtype: pandas.Series
        """
        return -(1 - (self.v / self.v.cummax()))

    # TODO: decide if keeping any of these or throw

    # @staticmethod
    # def _print_growth_rate(gr):
    #     """Transform growth rate into return and pretty-print it.
    #
    #     Either prints in basis points, percentage, or multiplication.
    #     """
    #     ret = np.exp(gr)-1
    #     if np.abs(ret) < 0.005:
    #         return f'{int(ret*1E4):d}bp'
    #     if np.abs(gr) < 1:
    #         return f'{ret*100:.2f}%'
    #     return f'{1+ret:.1f}X'

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

    # @property
    # def trading_days(self):
    #     """The fraction of days with nonzero turnover."""
    #     return (self.turnover.values > 0).sum() / self.turnover.size

    #
    # Results presentation
    #

    def plot(self, show=True, how_many_weights=7):
        """Make plot and show it.

        :param show: if True, call ``matplotlib.Figure.show``, helpful when
            running in the interpreter.
        :type show: bool
        :param how_many_weights: How many assets' weights are shown in the
            weights plots. The ones with largest average absolute value
            are chosen.
        :type how_many_weights: int

        :returns: Resulting matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """

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
        self.leverage.plot(ax=axes[2], linestyle='--',
                           color='k', label='Leverage')
        self.turnover.plot(ax=axes[2], linestyle='-',
                           color='r', label='Turnover')
        axes[2].legend()
        axes[2].grid(True, linestyle='--')

        if show:
            fig.show() # pragma: no cover

        return fig

    def times_plot(self, show=True):
        """Plot all execution times of the back-test.

        :param show: if True, call ``matplotlib.Figure.show``, helpful when
            running in the interpreter.
        :type show: bool

        :returns: Resulting matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """

        fig, ax = plt.subplots(1, figsize=(8.5, 8.5/1.618))

        fig.suptitle('Back-Test Times')
        self.policy_times.plot(label='Policy', fig=fig)
        self.simulator_times.plot(label='Simulator', fig=fig)
        self.market_data_times.plot(label='Of which: market data', fig=fig)
        self.result_times.plot(label='Of which: result', fig=fig)
        ax.set_ylabel('Seconds')
        ax.legend()

        if show:
            fig.show() # pragma: no cover

        return fig

    def __repr__(self):
        """Print the class instance."""

        stats = collections.OrderedDict({
            "Universe size": self.h.shape[1],
            "Initial timestamp": self.h.index[0],
            "Final timestamp": self.h.index[-1],
            "Number of periods": self.u.shape[0],
            f"Initial value ({self.cash_key})": f"{self.initial_value:.3e}",
            f"Final value ({self.cash_key})": f"{self.final_value:.3e}",
            f"Profit ({self.cash_key})": f"{self.profit:.3e}",
            ' '*4: '',
            "Avg. return (annualized)":
                f"{100 * self.annualized_average_return:.1f}%",
            "Volatility (annualized)":
                f"{100 * self.annualized_volatility:.1f}%",
            "Avg. excess return (annualized)":
                f"{100 * self.annualized_average_excess_return:.1f}%",
            "Avg. active return (annualized)":
                f"{100 * self.annualized_average_active_return:.1f}%",
            "Excess volatility (annualized)":
                f"{100 * self.annualized_excess_volatility:.1f}%",
            "Active volatility (annualized)":
                f"{100 * self.annualized_active_volatility:.1f}%",
            ' '*5: '',
            "Avg. growth rate (annualized)":
                f"{100*self.annualized_average_growth_rate:.1f}%",
            "Avg. excess growth rate (annualized)":
                f"{100*self.annualized_average_excess_growth_rate:.1f}%",
            "Avg. active growth rate (annualized)":
                f"{100*self.annualized_average_active_growth_rate:.1f}%",
        })

        if len(self.costs):
            stats[' '*6] = ''
        for cost in self.costs:
            stats[f'Avg. {cost}'] = \
                f"{(self.costs[cost]/self.v).mean()*1E4:.0f}bp"
            stats[f'Max. {cost}'] = \
                f"{(self.costs[cost]/self.v).max()*1E4:.0f}bp"

        stats.update(collections.OrderedDict({
            ' '*7: '',
            "Sharpe ratio": f"{self.sharpe_ratio:.2f}",
            "Information ratio": f"{self.information_ratio:.2f}",
            ' '*8: '',
            "Avg. drawdown": f"{self.drawdown.mean() * 100:.1f}%",
            "Min. drawdown": f"{self.drawdown.min() * 100:.1f}%",
            "Avg. leverage": f"{self.leverage.mean() * 100:.1f}%",
            "Max. leverage": f"{self.leverage.max() * 100:.1f}%",
            "Avg. turnover": f"{self.turnover.mean() * 100:.1f}%",
            "Max. turnover": f"{self.turnover.max() * 100:.1f}%",
            ' '*9: '',
            "Avg. policy time": f"{self.policy_times.mean():.3f}s",
            "Avg. simulator time": f"{self.simulator_times.mean():.3f}s",
            "    Of which: market data":
                f"{self.market_data_times.mean():.3f}s",
            # "    Of which: result":
            #     f"{self.result_times.mean():.3f}s",
            "Total time":
                f"{self.simulator_times.sum() + self.policy_times.sum():.3f}s",
            }))

        if np.all(np.isnan(self.active_returns)):
            del stats["Avg. active return (annualized)"]
            del stats["Active volatility (annualized)"]
            del stats["Avg. active growth rate (annualized)"]
            del stats["Information ratio"]

        content = pd.Series(stats).to_string()
        lenline = len(content.split('\n')[0])

        return '\n' + '#'*lenline + '\n' + content + '\n' + '#'*lenline + '\n'
