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
"""Build covariance matrix by regression.

This is a prototype for a planned extension of the forecast module in the main
library. However we're not sure yet which precise form it will take.


.. note::

    The internal interface methods used here **may not be public**
    and may not covered by the semantic versioning agreement (they might change
    without notice). In general, methods that are shown in the examples are
    considered public, but for this one there is no such guarantee.
"""

import numpy as np
import pandas as pd

import cvxportfolio as cvx
from cvxportfolio.estimator import Estimator

# from cvxportfolio.forecast import BaseForecast

def _striptime(df):
    res = pd.DataFrame(df)
    res.index = pd.to_datetime(df.index.date)
    return res

def _covariance(returns):
    _ = returns.fillna(0.)
    num = _.T @ _
    _ = (~returns.isnull()) * 1.
    den = _.T @ _
    return num / den

def _covariance_weigh(returns, weighter):
    tmp = returns.multiply(weighter, axis=0)
    left = tmp.fillna(0.)
    right = returns.multiply(~(weighter.isnull())*1., axis=0).fillna(0.)
    num = left.T @ right
    _ = (~tmp.isnull()) * 1.
    den = _.T @ _
    return num / den

def _decorrelate(XtX, obs, gamma):
    return np.linalg.solve(
        XtX + np.diag(([gamma] * (len(obs)-1)) + [0.]),
        obs)

class RegressionBase(Estimator):
    """Base for linear regression forecasters."""

    def __init__(self, regressors, gamma=1e-1):
        """Initialize with regressors.

        For simplicity here we drop time (and keep date) from the index.

        :param regressors: Regressors used.
        :type regressors: pandas.DataFrame
        :param gamma: Tikhonov regularizer, default 0.
        :type gamma: float
        """
        self.regressors = _striptime(regressors)
        self.regressors['intercept'] = 1.
        self.gamma = gamma

    def _obtain_weighted(self, used_regressors, **kwargs):
        """Obtain objects weighted by the regressors.

        :param used_regressors: Used regressors (only past).
        :type used_regressors: pandas.DataFrame
        :param kwargs: All arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        raise NotImplementedError

    def values_in_time(self, t, past_returns, **kwargs):
        """Produce regressed object for the period.

        :param t: Current time.
        :type t: pandas.Timestamp
        :param past_returns: Past returns (includes cash).
        :type past_returns: pandas.DataFrame
        :param kwargs: Other arguments.
        :type kwargs: dict

        :return: Regressed covariance.
        :rtype: pandas.DataFrame
        """
        # raise Exception
        used_regressors = pd.DataFrame(
            self.regressors.loc[
                self.regressors.index <= pd.Timestamp(t.date())], copy=True)

        # to extend to index
        used_regressors['intercept'] = past_returns.iloc[:, -1]

        used_regressors = used_regressors.ffill()
        used_regressors -= used_regressors.mean()
        used_regressors /= np.sqrt((used_regressors**2).mean())

        used_regressors['intercept'] = 1.

        # print('used_regressors')
        # print(used_regressors)

        _ = _covariance(used_regressors.iloc[:-1])
        # print('XtX', _)
        lhs = _decorrelate(_, used_regressors.iloc[-1], gamma=self.gamma)
        # print('lhs', lhs)

        _ = self._obtain_weighted(used_regressors,
            t=t, past_returns=past_returns, **kwargs)

        return sum(el * _[i] for i, el in enumerate(lhs))


class RegressionCovariance(RegressionBase):
    """Covariance matrix that is obtained by linear regression."""

    def _obtain_weighted(self, used_regressors, past_returns, **kwargs):
        """Obtain objects weigthed by the regressors.

        :param used_regressors: Used regressors (only past).
        :type used_regressors: pandas.DataFrame
        :param past_returns: Past market returns.
        :type past_returns: pandas.DataFrame
        :param kwargs: All other arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        # creates new object, doesn't damage original one
        past_returns = _striptime(past_returns)

        return [_covariance_weigh(
                past_returns.iloc[:, :-1], used_regressors.iloc[:-1][col])
            for col in used_regressors]

class RegressionReturns(RegressionBase):
    """Returns forecast that is obtained by linear regression."""

    def _obtain_weighted(self, used_regressors, past_returns, **kwargs):
        """Obtain objects weighted by the regressors.

        :param used_regressors: Used regressors (only past).
        :type used_regressors: pandas.DataFrame
        :param past_returns: Past market returns.
        :type past_returns: pandas.DataFrame
        :param kwargs: All other arguments to :meth:`values_in_time`.
        :type kwargs: dict
        """

        # creates new object, doesn't damage original one
        past_returns = _striptime(past_returns)

        return [ past_returns.iloc[:, :-1].multiply(
                    used_regressors.iloc[:-1][col], axis=0).mean().values
            for col in used_regressors]

def example_regressors():
    """Vix regressors and others.

    The open value of VIX is avalaible at market open US, so this is a valid
    (no look-ahead) set of regressors.

    :return: Various regressors, with different timescales and availability.
    :rtype: pandas.DataFrame
    """
    vix = cvx.YahooFinance('^VIX').data.open
    rate = cvx.DownloadedMarketData(['GE']).returns.USDOLLAR
    #wage_change = cvx.Fred('CES0500000003').data.pct_change()
    #gdp_change = cvx.Fred('GDP').data.pct_change()
    base_regressors = {
        'vix': vix,
        'rate': rate,
        #'wage_change': wage_change,
        #'gdp_change': gdp_change
        }
    regressors = pd.DataFrame(base_regressors)

    # for col in base_regressors:
    #     if np.all(regressors[col].dropna() > 0):
    #         regressors['log_' + col] = np.log(regressors[col])
    #     regressors['square_'+ col] = regressors[col]**2
    #     regressors['cube_' + col] = regressors[col]**3

    return _striptime(regressors).ffill()

if __name__ == '__main__':

    from .universes import DOW30

    REGRESSORS = example_regressors()
    print(REGRESSORS)

    # RETURNS_REGRESSORS = REGRESSORS[
    #     [col for col in REGRESSORS.columns
    #         if not 'vix' in col]
    #     ]

    # market_data = cvx.DownloadedMarketData(['AAPL', 'GOOG', 'TSLA'])

    cov = RegressionCovariance(REGRESSORS, gamma = 1e-6)
    # ret = RegressionReturns(RETURNS_REGRESSORS, gamma = 1000)

    # tmp = pd.DataFrame(market_data.returns.iloc[:-10],copy=True)
    # tmp.iloc[-1] = np.nan
    #print(cov.values_in_time(
    #    t = market_data.returns.index[-10],
    #    past_returns=tmp,))

    constraints = [
        #cvx.LongOnly(),
        cvx.LeverageLimit(7)
    ]

    # result_plain, result_covregr, result_retregr, result_retcovregr = \
    result_plain, result_covregr = \
            cvx.MarketSimulator(DOW30).backtest_many(
        [
        cvx.SinglePeriodOptimization(cvx.ReturnsForecast()
            - 3 * cvx.FullCovariance(),
            constraints,
            ),
        cvx.SinglePeriodOptimization(cvx.ReturnsForecast()
            - 3 * cvx.FullCovariance(cov),
            constraints,
            ),
        # cvx.SinglePeriodOptimization(cvx.ReturnsForecast(ret)
        #     - 3 * cvx.FullCovariance(),
        #     constraints,
        #     ),
        # cvx.SinglePeriodOptimization(cvx.ReturnsForecast(ret)
        #     - 3 * cvx.FullCovariance(cov),
        #     constraints,
        #     ),
        ],
        start_time='2010-01-01') # can't do earlier than ~2010
        # because regressor not available,
        # need to add logic to include them after some min_history

    result_plain.plot()
    result_covregr.plot()
    # result_retregr.plot()
    # result_retcovregr.plot()

    print('Result with no regression:')
    print(result_plain)

    print('Result with covariance regression:')
    print(result_covregr)

    # print('Result with returns regression:')
    # print(result_retregr)

    # print('Result with returns and covariance regression:')
    # print(result_retcovregr)