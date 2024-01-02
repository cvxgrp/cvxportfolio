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

Each strategy defines only one function, which defines its chosen trading
policy, and a few constants: trading universe, start of the hyper-parameter
optimization period, and objective function.

Here we define the logic to handle those, both for online execution and
hyper-parameter optimization. Also we save everything in json
files and (in the future) store in git and check for problems and errors.
This is currently run in a cron job from a shell script
(``strategies_runner.sh``) in the root of the repository, which also stores in
git.

.. note::

    We may eventually move this in the main library.
"""

import inspect
import json
import logging
import sys
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.result import LOG_FORMAT, RECORD_LOGS

INITIAL_VALUE = 1E6 # initial value (in cash)

logging.basicConfig(level=RECORD_LOGS, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def main(
    policy, hyperparameter_opt_start, objective, universe,
    cash_key='USDOLLAR', initial_values=None):
    """Executor for each strategy's script.

    :param policy: Function that returns the policy object and a dictionary
        mapping its hyper-parameter names (which are the arguments it takes)
        to their respective objects (used only for hyper-parameter
        optimization).
    :type policy: callable
    :param hyperparameter_opt_start: Start of hyper-parameter optimization
        back-test period.
    :type hyperparameter_opt_start: str or pandas.Timestamp
    :param objective: Objective used for hyper-parameter optimization (
        attribute of :class:`cvxportfolio.BacktestResult`).
    :type objective: str
    :param universe: Current universe of the strategy.
    :type universe: iterable
    :param cash_key: Name of cash account.
    :type cash_key: str
    :param initial_values: Initial hyper-parameter choices.
    :type initial_values: dict or None
    """
    if len(sys.argv) < 2 or sys.argv[1] not in ['hyperparameters', 'strategy']:
        print('Usage (from root directory):')
        stratfile = policy.__code__.co_filename
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
        policy=policy, hyperparameter_opt_start=hyperparameter_opt_start,
        objective=objective, universe=universe, cash_key=cash_key,
        initial_values=initial_values)

    if sys.argv[1] == 'hyperparameters':
        runner.run_hyperparameters()
    if sys.argv[1] == 'strategy':
        runner.run_execute_strategy()


def hyperparameter_optimize(
    universe, policy, hyperparameter_opt_start, objective='sharpe_ratio',
    initial_values=None):
    """Optimize hyper-parameters of a policy over back-test.

    :param universe: Trading universe for the policy.
    :type universe: iterable
    :param policy: Policy creation function.
    :type policy: callable
    :param hyperparameter_opt_start: When to start the hyper-parameter
        optimization.
    :type hyperparameter_opt_start: str or pandas.Timestamp
    :param objective: Which hyper-parameter optimization objective to use
        (must be an attribute of :class:`cvxportfolio.BacktestResult`).
        Default ``'sharpe_ratio'``.
    :type objective: str
    :param initial_values: Initial hyper-parameter choices.
    :type initial_values: dict or None

    :return: Choice of gamma risk and gamma trade.
    :rtype: dict
    """
    hyper_parameter_names = inspect.getfullargspec(policy).args
    if initial_values is None:
        initial_values = {k: 1. for k in hyper_parameter_names}

    sim = cvx.StockMarketSimulator(universe)
    policy, hyperpar_handles = policy(**initial_values)

    # check to be sure
    assert set(hyper_parameter_names) == set(hyperpar_handles)

    sim.optimize_hyperparameters(
        policy, start_time=hyperparameter_opt_start,
        objective=objective)

    return {k: hyperpar_handles[k].current_value
        for k in hyperpar_handles}

def execute_strategy(
    current_holdings, market_data, policy, hyper_parameters):
    """Execute this strategy.

    :param current_holdings: Current holdings in dollars.
    :type current_holdings: pandas.Series
    :param market_data: Market data server.
    :type market_data: cvxportfolio.data.MarketData
    :param policy: Policy constructor function.
    :type policy: callable
    :param hyper_parameters: Current choice of hyper-parameters.
    :type hyper_parameters: dict

    :return: Output of the execute method of a Cvxportfolio policy.
    :rtype: tuple
    """
    _policy, _ = policy(**hyper_parameters)
    return _policy.execute(h=current_holdings, market_data=market_data)

class _Runner:

    def __init__(
        self, policy, hyperparameter_opt_start, objective, universe,
            cash_key='USDOLLAR', initial_values=None):
        self.policy = policy
        self.hyperparameter_opt_start = hyperparameter_opt_start
        self.objective = objective
        self.universe = universe
        self.cash_key = cash_key
        self.initial_values = initial_values
        # self.today = str(datetime.datetime.now().date())
        self.stratfile = self.policy.__code__.co_filename
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

        self.all_hyper_params[self.today] = hyperparameter_optimize(
            universe=self.universe, policy=self.policy,
            hyperparameter_opt_start=self.hyperparameter_opt_start,
            objective=self.objective, initial_values=self.initial_values)

        print('Hyper-parameters optimized today:')
        print(self.all_hyper_params[self.today])

        self.store_json(self.file_hyper_parameters, self.all_hyper_params)

    def backtest_from_day(self, day):
        """Back-test from day with stored weights to get next days holdings.

        .. note::

            If some stock was delisted between the day and the next this will
            fail.

        :param day: Which day to back-test.
        :type day: pandas.Timestamp

        :return: New holdings up to today's open.
        :rtype: pandas.DataFrame
        """
        # last_day = sorted(self.all_holdings.keys())[-1]
        logger.info("Back-testing day %s to get next day's holdings", day)

        day_init_holdings = pd.Series(self.all_holdings[day])
        day_target_weigths = pd.Series(self.all_target_weights[day])
        day_universe = [
            el for el in day_init_holdings.index if not el == self.cash_key]
        sim = cvx.StockMarketSimulator(
            market_data=cvx.DownloadedMarketData(
            day_universe, min_history=pd.Timedelta('0d'), 
            cash_key=self.cash_key))

        # This should be done by MarketSimulator, but for safety.
        day_init_holdings = day_init_holdings[sim.market_data.returns.columns]
        last_day_target_weigths = day_target_weigths[
            sim.market_data.returns.columns]

        # For safety also recompute cash of target weights
        assert np.isclose(day_target_weigths.iloc[-1],
            1-day_target_weigths.iloc[:-1].sum())
        day_target_weigths.iloc[-1] = \
            1-day_target_weigths.iloc[:-1].sum()

        result = sim.backtest(
                policy=cvx.FixedWeights(last_day_target_weigths),
                h=day_init_holdings, start_time=day)
        return result.h.loc[result.h.index > day]

    def adjust_universe(self, day, new_universe):
        """If day universe changed, liquidate unused stocks first.

        :param day: Time period at which we adjust.
        :type day: pandas.Timestamp
        :param new_universe: New universe to adjust to.
        :type new_universe: iterable
        """

        initial_holdings = self.all_holdings[day]
        holdings_universe = [
            el for el in initial_holdings if not el == self.cash_key]

        if not set(holdings_universe) == set(new_universe):
            logging.info(
                'Universe from last time is not same as day %s. Adjusting.',
                day)

            # liquidate
            stocks_to_liquidate = set(holdings_universe).difference(
                new_universe)
            initial_holdings[self.cash_key] += sum(
                initial_holdings[k] for k in stocks_to_liquidate)
            for stock in stocks_to_liquidate:
                del initial_holdings[stock]

            # add zeros
            for stock in set(new_universe).difference(holdings_universe):
                initial_holdings[stock] = 0.

    def reconcile_and_get_todays_holdings(self):
        """Reconcile yesterday's holdings and get today's."""

        last_run_day = sorted(self.all_holdings.keys())[-1]

        # reconciliation
        if len(self.all_holdings) > 1:

            # first get the universe we were using yesterday
            yesterdays_holdings = self.all_holdings[last_run_day]
            yesterdays_universe = [
                el for el in yesterdays_holdings if not el == self.cash_key]

            # now back-test from day before yesterday
            day_before_last_run = sorted(self.all_holdings.keys())[-2]
            new_holdings = self.backtest_from_day(day_before_last_run)

            # update including today (which will be overwritten next)
            for t in new_holdings.index:
                self.all_holdings[t] = dict(new_holdings.loc[t])

            # adjust yesterday universe
            self.adjust_universe(
                last_run_day, new_universe=yesterdays_universe)

        # now back-test from yesterday to get today holdings
        new_holdings = self.backtest_from_day(last_run_day)
        for t in new_holdings.index:
            self.all_holdings[t] = dict(new_holdings.loc[t])
        self.adjust_universe(self.today, self.universe)

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

            self.reconcile_and_get_todays_holdings()

        self.store_json(self.file_holdings, self.all_holdings)

        today_ts = sorted(self.all_holdings.keys())[-1]
        return self.all_holdings[today_ts]

    def run_execute_strategy(self):
        """Run strategy's execution, store result.

        :raises ValueError: If there are no hyper-parameters available.
        """

        if self.today in self.all_target_weights:
            logger.info(
                "Already computed weights for last trading day, aborting.")
            return

        if len(self.all_hyper_params) < 1:
            raise ValueError('Empty hyper-parameters file!')
        hp_index = sorted(self.all_hyper_params.keys())[-1]
        logger.info('Using hyper-parameters optimized on %s', hp_index)

        todays_holdings = self.get_current_holdings()

        h = pd.Series(todays_holdings)
        v = sum(h)
        w = h / v

        logger.info('Running strategy execution for day %s', self.today)
        u, t, _ = execute_strategy(
            current_holdings=h,
            market_data=cvx.DownloadedMarketData(
                self.universe, online_usage=True,
                min_history=pd.Timedelta('0d')),
            policy=self.policy,
            hyper_parameters=self.all_hyper_params[hp_index])

        assert t == self.today

        # u is trades in dollars without rounding, so we can
        # reconstruct exact w_plus from strategy

        z = u / v
        w_plus = w + z

        self.all_target_weights[self.today] = w_plus

        self.store_json(self.file_target_weights, self.all_target_weights)
