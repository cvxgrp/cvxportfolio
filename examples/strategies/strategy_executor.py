# Copyright 2023 Enzo Busseti
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
"""This module contains code used to run each individual strategy.

Each strategy defines two functions: one that back-tests and optimizes over
hyper-parameters, returning their chosen values, and one that takes such values
and runs the strategy for the day.

Here we define the logic to handle those two functions, save everything in json
files, store in git, and check for problems and errors. The final product is a
strategy script that is run by one line of crontab.

.. note::

    We may eventually move this in the main library.
"""

import json
import sys
from functools import cached_property
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from cvxportfolio.result import RECORD_LOGS, LOG_FORMAT
import cvxportfolio as cvx

INITIAL_VALUE = 1E6 # initial value (in cash)

logging.basicConfig(level=RECORD_LOGS, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def main(
    hyperparameter_optimize, execute_strategy, universe, cash_key='USDOLLAR'):
    """Executor for each strategy's script.

    :param hyperparameter_optimize: Function that returns the current best
        choice of hyper-parameters, as a dictionary.
    :type hyperparameter_optimize: callable
    :param execute_strategy: Function that takes the current holdings,
        market data object, and the choices of hyper-parameters, and returns
        the output of :meth:`cvxportfolio.Policy.execute`.
    :type execute_strategy: callable
    :param universe: Current universe of the strategy.
    :type universe: iterable
    :param cash_key: Name of cash account.
    :type cash_key: str
    """
    if len(sys.argv) < 2 or sys.argv[1] not in ['hyperparameters', 'strategy']:
        print('Usage (from root directory):')
        stratfile = hyperparameter_optimize.__code__.co_filename
        print()
        print('\tpython -m', __package__ + '.' + Path(stratfile).stem,
            'hyperparameters')
        print()
        print('\t\tRun hyper-parameter optimization.')
        print()
        print('\tpython -m', __package__ + '.' + Path(stratfile).stem,
            'strategy')
        print()
        print('\t\tRun strategy.')
        print()

        sys.exit(0)

    runner = _Runner(
        hyperparameter_optimize, execute_strategy, universe, cash_key)

    if sys.argv[1] == 'hyperparameters':
        runner.run_hyperparameters()
    if sys.argv[1] == 'strategy':
        runner.run_execute_strategy()


class _Runner:

    def __init__(
        self, hyperparameter_optimize, execute_strategy, universe, cash_key):
        self.hyperparameter_optimize = hyperparameter_optimize
        self.execute_strategy = execute_strategy
        self.universe = universe
        self.cash_key = cash_key
        # self.today = str(datetime.datetime.now().date())
        self.stratfile = self.hyperparameter_optimize.__code__.co_filename
        assert self.stratfile == \
            self.execute_strategy.__code__.co_filename
        self.all_hyper_params = self.load_json(self.file_hyper_parameters)
        self.all_target_weights = self.load_json(self.file_target_weights)
        self.all_holdings = self.load_json(self.file_holdings)

    @cached_property
    def today(self):
        """Get today's open timestamp.

        :returns: Open timestamp.
        :rtype: pandas.Timestamp
        """
        logger.info('Getting last time-stamp from market data.')
        _today = cvx.DownloadedMarketData(self.universe[:1]).returns.index[-1]
        logger.info('Last timestamp is %s', _today)
        return _today

    @property
    def file_hyper_parameters(self):
        """File that stores optimized hyper-parameters.

        :returns: Hyper-parameters file.
        :rtype: pathlib.Path
        """
        return Path(self.stratfile).parent \
            / (Path(self.stratfile).stem + '_hyper_parameters.json')

    @property
    def file_holdings(self):
        """File that stores realized (paper) initial holdings.

        :returns: Holdings file.
        :rtype: pathlib.Path
        """
        return Path(self.stratfile).parent \
            / (Path(self.stratfile).stem + '_initial_holdings.json')

    @property
    def file_target_weights(self):
        """File that stores target allocation weights.

        :returns: Target weights file.
        :rtype: pathlib.Path
        """
        return Path(self.stratfile).parent \
            / (Path(self.stratfile).stem + '_target_weights.json')

    def load_json(self, filename):
        """Load json, return empty dict if not existent.

        :param filename: Json file to load.
        :type filename: str

        :return: Loaded content (always dict in our files).
        :rtype: dict
        """
        logger.info('Loading json file %s', filename)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                result = json.load(f)
        except FileNotFoundError:
            logger.info("File not found, returning empty dict.")
            result = {}
        return {pd.Timestamp(k): result[k] for k in result}

    def store_json(self, filename, content):
        """Store content to json, with our formatting.

        :param filename: Json file to store to.
        :type filename: str
        :param content: Content to store.
        :type content: dict
        """
        logger.info('Storing json file %s', filename)
        content = {str(k): dict(content[k]) for k in content}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, sort_keys=True, indent=4)

    def run_hyperparameters(self):
        """Run hyper-parameter optimization, store result."""

        logger.info('Running hyper-parameter optimization')
        if self.today in self.all_hyper_params:
            logger.info('Today already ran hyper-parameter optimization.')
            return

        self.all_hyper_params[self.today] = self.hyperparameter_optimize()

        print('Hyper-parameters optimized today:')
        print(self.all_hyper_params[self.today])

        self.store_json(self.file_hyper_parameters, self.all_hyper_params)

    def backtest_from_last_point(self):
        """Backtest from the last point to get new holdings.

        .. note::

            If some stock was delisted between the last execution and today's
            open this will fail.

        :return: New holdings up to today's open.
        :rtype: pandas.DataFrame
        """
        last_day = sorted(self.all_holdings.keys())[-1]
        logger.info("Back-testing from %s to %s to get today's holdings",
            last_day, self.today)

        last_day_holdings = pd.Series(self.all_holdings[last_day])
        last_day_target_weigths = pd.Series(self.all_target_weights[last_day])
        last_day_universe = [
            el for el in last_day_holdings.index if not el == self.cash_key]
        sim = cvx.StockMarketSimulator(
            last_day_universe, cash_key=self.cash_key)

        # This should be done by MarketSimulator, but for safety.
        last_day_holdings = last_day_holdings[sim.market_data.returns.columns]
        last_day_target_weigths = last_day_target_weigths[
            sim.market_data.returns.columns]

        # For safety also recompute cash of target weights
        assert np.isclose(last_day_target_weigths.iloc[-1],
            1-last_day_target_weigths.iloc[:-1].sum())
        last_day_target_weigths.iloc[-1] = \
            1-last_day_target_weigths.iloc[:-1].sum()

        result = sim.backtest(
                policy=cvx.FixedWeights(last_day_target_weigths),
                h=last_day_holdings, start_time=last_day)
        return result.h.loc[result.h.index > last_day]

    def adjust_universe(self):
        """If today's universe changed, liquidate unused stocks first."""
        today_ts = sorted(self.all_holdings.keys())[-1]
        initial_holdings = self.all_holdings[today_ts]
        holdings_universe = [
            el for el in initial_holdings if not el == self.cash_key]

        if not set(holdings_universe) == set(self.universe):
            print('Universe from last time is not same as todays. Adjusting.')

            # liquidate
            stocks_to_liquidate = set(holdings_universe).difference(
                self.universe)
            initial_holdings[self.cash_key] += sum(
                initial_holdings[k] for k in stocks_to_liquidate)
            for stock in stocks_to_liquidate:
                del initial_holdings[stock]

            # add zeros
            for stock in set(self.universe).difference(holdings_universe):
                initial_holdings[stock] = 0.

    def get_current_holdings(self):
        """Get current (today's open) holdings.

        :returns: Holdings vector
        :rtype: pandas.Series
        """

        logger.info('Getting holdings for today %s', self.today)
        if len(self.all_holdings) < 1:
            logger.info('Defaulting to %s all in cash', INITIAL_VALUE)
            self.all_holdings[self.today] = pd.Series(0., self.universe)
            self.all_holdings[self.today][self.cash_key] = INITIAL_VALUE
        else:
            if self.today in self.all_holdings:
                logger.info(
                    'Today already in the initial holdings file, returning.')
                return self.all_holdings[self.today]
            new_holdings = self.backtest_from_last_point()
            for t in new_holdings.index:
                self.all_holdings[t] = dict(new_holdings.loc[t])
            self.adjust_universe()

        self.store_json(self.file_holdings, self.all_holdings)

        today_ts = sorted(self.all_holdings.keys())[-1]
        return self.all_holdings[today_ts]

    def run_execute_strategy(self):
        """Run strategy's execution, store result.

        :raises ValueError: If there are no hyper-parameters available.
        """

        if len(self.all_hyper_params) < 1:
            raise ValueError('Empty hyper-parameters file!')
        hp_index = sorted(self.all_hyper_params.keys())[-1]
        logger.info('Using hyper-parameters optimized on %s', hp_index)

        todays_holdings = self.get_current_holdings()

        h = pd.Series(todays_holdings)
        v = sum(h)
        w = h / v

        logger.info('Running strategy execution for day %s', self.today)
        u, t, _ = self.execute_strategy(
            current_holdings=h,
            market_data=cvx.DownloadedMarketData(
                self.universe, online_usage=True,
                min_history=pd.Timedelta('0d')),
            **self.all_hyper_params[hp_index])

        assert t == self.today

        # u is trades in dollars without rounding, so we can
        # reconstruct exact w_plus from strategy

        z = u / v
        w_plus = w + z

        self.all_target_weights[self.today] = w_plus

        self.store_json(self.file_target_weights, self.all_target_weights)
