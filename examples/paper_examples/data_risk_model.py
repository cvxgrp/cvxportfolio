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
"""This is the restored version of the IPython notebook
used to download data for the original 2016 examples:

https://github.com/cvxgrp/cvxportfolio/blob/0.0.X/examples/DataEstimatesRiskModel.ipynb

The main reasons why the original can't be used any more:

- the data source we used, "Quandl wiki" historical stock market data,
    is defunct. Fortunately we saved the data in this repo, which is the
    reason why it's quite big, so we can reproduce the original results.
- Pandas dropped its Panel data structure (after which it is named!); we
    used it to store the risk models. We now use multi-indexed Dataframes.

The data files that we use from the notebook above in 2017 are:

- ``returns.csv.gz``: Historical close-to-close total returns; note that today
    :class:`cvxportfolio.DownloadedMarketData` instead computes
    historical open-to-open total returns, matching the model
    from the paper.
- ``volumes.csv.gz``: historical market volumes in USD, computed as volumes
    in number of shares times the adjusted close price.

These are for a selection of stocks from the components of SP500 in 2016. 
The data cleaning logic is shown in the notebook. We don't use the
following files from that notebook:

- ``sigmas.csv.gz``: now computed automatically by 
    :class:`cvxportfolio.TransactionCost`; note that intraday returns were 
    used in the notebook, while now we use open-to-open returns. This may 
    cause some noticeable difference in the market impact term of the
    transaction cost estimates. In practice, the market impact term is either 
    negligible for small to medium investors, or needs to be tuned for the
    given assets (using historical realized costs) in case of a large investor.
- ``prices.csv.gz``: unused
- ``sigma_estimate.csv.gz``: now computed automatically by 
    :class:`cvxportfolio.TransactionCost`
- ``volume_estimate.csv.gz``: now computed automatically by 
    :class:`cvxportfolio.TransactionCost`
"""

from pathlib import Path

import numpy as np
import pandas as pd

import cvxportfolio as cvx


def paper_market_data():
    """Build market data server for the paper's examples.

    The returns dataframe already includes the cash returns column
    (as its last), we point the ``cash_key`` argument to its name.

    We also use the ``min_history`` argument, which, howerver ,in this case is
    not needed (since we start our back-tests after the default minimum
    history of one year).

    :return: Market data for the paper.
    :rtype: :class:`cvxportfolio.UserProvidedMarketData`
    """
    returns = pd.read_csv(
        Path(__file__).parent / 'returns.csv.gz', index_col=0, parse_dates=[0])
    volumes = pd.read_csv(
        Path(__file__).parent / 'volumes.csv.gz', index_col=0, parse_dates=[0])
    # print(returns)
    return cvx.UserProvidedMarketData(
        returns=returns, volumes=volumes,
        cash_key='USDOLLAR', min_history=pd.Timedelta('0d'))


def paper_risk_model():
    """Build low-rank risk model for the paper's examples.

    This is mostly a copy-paste of the last cell of the original IPython
    notebook. The differences are that we reshape the data into multi-indexed
    dataframes and we explicitely skip the cash column.

    :return: Factor exposures, factor covariances, idyosincratic risks.
    :rtype: pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
    """

    k = 15

    start_t = "2012-01-01"
    end_t = "2016-12-31"

    returns = pd.read_csv(
        Path(__file__).parent / 'returns.csv.gz',
        index_col=0, parse_dates=[0]).iloc[:, :-1]  # skip cash column

    first_days_month = pd.date_range(
        start=returns.index[next(
            i for (i, el) in enumerate(returns.index >= start_t) if el)-1],
        end=returns.index[-1], freq='MS')

    print('Computing risk model...')

    factor_exposures = pd.DataFrame(
        index = pd.MultiIndex.from_product([first_days_month, range(k)]),
        columns = returns.columns,
    )

    factor_sigma = pd.DataFrame(
        index = pd.MultiIndex.from_product([first_days_month, range(k)]),
        columns = range(k),
    )

    idyosincratic = pd.DataFrame(
        index = first_days_month, columns = returns.columns)

    for day in first_days_month:
        used_returns = returns.loc[
            (returns.index < day) & (returns.index >= day-pd.Timedelta(
                "730 days"))]
        second_moment = (
            used_returns.values.T @ used_returns.values
            / used_returns.values.shape[0])
        eival, eivec = np.linalg.eigh(second_moment)
        factor_sigma.loc[day] = np.diag(eival[-k:])
        factor_exposures.loc[day] = eivec[:, -k:].T
        idyosincratic.loc[day] = np.diag(
            eivec[:, :-k] @ np.diag(eival[:-k]) @ eivec[:, :-k].T)

    return factor_exposures, factor_sigma, idyosincratic


if __name__ == '__main__':

    md = paper_market_data()

    print('Trading calendar:')
    print(md.trading_calendar())

    print('Returns:')
    print(md.returns)

    print('Volumes:')
    print(md.volumes)

    factor_exposures, factor_sigma, idyosincratic = paper_risk_model()

    print("Factor exposures:")
    print(factor_exposures)

    print("Factor covariances:")
    print(factor_sigma)

    print("Idyosincratic risks:")
    print(idyosincratic)
