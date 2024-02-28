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
"""This script is used to show the data cleaning applied to Yahoo Finance data.

It is not really an example, and some of the methods shown here are not public,
so not covered by the semantic versioning agreeement (they could change
without notice).

You run it (from the root of the development environment) by

.. code-block::

    python -m examples.data_cleaning

"""

import shutil
import tempfile
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxportfolio as cvx

# Uncomment the following lines to get information about the cleaning procedure
# import logging
# logging.basicConfig(level=logging.INFO)
# log=logging.getLogger('=>')

# Here put any number of stocks for which you wish to analyze the cleaning
TEST_UNIVERSE = ['AAPL', 'GOOG', 'TSLA']

# Some names with known issues:
# TEST_UNIVERSE = ['SMT.L', 'NVR', 'HUBB', 'NWG.L', 'BA.L']

# Or, pick a larger universe
from .universes import *
TEST_UNIVERSE = FTSE100

class TestYahooFinance(cvx.YahooFinance):
    """Example of subclass of YahooFinance for tuning cleaning parameters.

    You can change any of the parameters to see how they affect cleaning.

    We copied them from the data/symbol_data.py module. You find their
    documentation there. The commented values below are their defaults.
    """
    # FILTERING_WINDOWS = (10, 20, 50, 100, 200)
    # THRESHOLD_OPEN_TO_CLOSE = 20
    # THRESHOLD_LOWHIGH_TO_CLOSE = 20
    # THRESHOLD_WARN_EXTREME_LOGRETS = 40
    # EXCLUDE_EXACT_ZEROS_FROM_FILTERING = False
    # MAX_CONTIGUOUS_MISSING_ADJCLOSES = 20
    # THRESHOLD_BAD_ADJCLOSE = 50
    # THRESHOLD_FALSE_LOG10RETS = .5
    # ASSUME_FALSE_BEFORE = pd.Timestamp('2000-01-01', tz='UTC')
    # UPDATE_OVERLAP = 5
    # DIVIDEND_THRESHOLD = .2


ALL_DROPPED_ROWS_PCT = pd.Series(dtype=float)
ALL_MIN_LR = pd.Series(dtype=float)
ALL_MAX_LR = pd.Series(dtype=float)

PLOT = True
SLEEP = 1

for stock in TEST_UNIVERSE:
    sleep(SLEEP)
    print(f'\n\t{stock}:')

    # This method is not public:
    raw_yfinance = cvx.YahooFinance._get_data_yahoo(stock)
    print(f'{stock}: YAHOO FINANCE RAW')
    print(raw_yfinance)

    tmpdir = Path(tempfile.mkdtemp())
    cvx_cleaned = TestYahooFinance(stock, base_location=tmpdir).data
    shutil.rmtree(tmpdir)
    print(f'{stock}: CVXPORTFOLIO CLEANED (WITH TEST PARAMETERS)')
    print(cvx_cleaned)

    yf_log10r = np.log10(raw_yfinance.adjclose).diff().shift(-1)
    cvx_log10r = np.log10(1 + cvx_cleaned['return'])

    if PLOT:
        fig, axes = plt.subplots(
            3, figsize=(10/1.62, 10), layout='constrained')

        raw_yfinance.iloc[:, :5].plot(ax=axes[0])
        axes[0].set_yscale('log')
        axes[0].set_title(f'{stock}: RAW YAHOO FINANCE')

        cvx_cleaned.iloc[:, :4].plot(ax=axes[1])
        axes[1].set_title(f'{stock}: CVX CLEANED (W/ TEST PARAMETERS)')
        axes[1].set_yscale('log')

        (yf_log10r.cumsum() - yf_log10r.sum()).plot(
            label='Yahoo Finance total close-to-close', ax=axes[2])
        (cvx_log10r.cumsum() - cvx_log10r.sum()).plot(
            label='Cvx cleaned total open-to-open', ax=axes[2])
        axes[2].set_title(f'{stock}: CUMULATIVE LOG10 RETURNS (SCALED)')
        axes[2].legend()

        plt.show()

    assert cvx_cleaned.index[-1] == raw_yfinance.index[-1]

    print()
    dropped_rows = len(raw_yfinance) - len(cvx_cleaned)
    dropped_rows_pct = dropped_rows / len(raw_yfinance)
    ALL_DROPPED_ROWS_PCT.loc[stock] = dropped_rows_pct*100
    print(f'Cvxportfolio dropped {int(dropped_rows_pct*100)}% of rows')

    ALL_MIN_LR.loc[stock] = np.log(1+cvx_cleaned['return']).min()
    ALL_MAX_LR.loc[stock] = np.log(1+cvx_cleaned['return']).max()

    print('Max Cvxportfolio logreturn:', ALL_MAX_LR.loc[stock])
    print('Min Cvxportfolio logreturn:', ALL_MIN_LR.loc[stock] )
    print('How many zero volumes:', (cvx_cleaned['valuevolume'] == 0.).mean())

print('\nCvxportfolio dropped rows %:')
print(ALL_DROPPED_ROWS_PCT.sort_values().tail())

print('\nCvxportfolio min logreturns:')
print(ALL_MIN_LR.sort_values().head())

print('\nCvxportfolio max logreturns:')
print(ALL_MAX_LR.sort_values().tail())
