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

The procedure has many steps, with a lot of heuristics. You can see it
by inspecting the code of :class:`cvxportfolio.YahooFinance` (and its parent
classes). All the heuristic steps are configurable with class-level constants,
which can be easily overridden by subclassing for your specific usecase. None
of this matters if you use Cvxportfolio with user-provided market data, but
it's crucial for our example strategies (which use this interface, and rely
on the data cleaning).

This is not really an example, and one method shown here
(:meth:`cvxportfolio.YahooFinance._get_data_yahoo`) is not public,
so not covered by the semantic versioning agreeement (it could change
without notice).
"""

# You run this from the root directory of the development environment by:
# python -m examples.data_cleaning

import os

# Uncomment the following lines to get information about the cleaning procedure
# import logging
# logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    import shutil
    import tempfile
    from pathlib import Path
    from time import sleep

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import cvxportfolio as cvx

    # Here you can put any stocks for which you wish to analyze the cleaning;
    # Some names with known issues:
    TEST_UNIVERSE = ['SMT.L', 'NVR', 'HUBB', 'NWG.L', 'BA.L']

    # Or, pick a larger universe
    # from .universes import *
    # TEST_UNIVERSE = DOW30

    ALL_DROPPED_ROWS_PCT = pd.Series(dtype=float)
    ALL_MIN_LR = pd.Series(dtype=float)
    ALL_MAX_LR = pd.Series(dtype=float)

    PLOT = True
    SLEEP = 1

    figures = {}

    for stock in TEST_UNIVERSE:
        sleep(SLEEP)
        print(f'\n\t{stock}:')

        # This method is not public, it only downloads historical data from
        # Yahoo Finance; all the cleaning is applied downstream
        # pylint: disable=protected-access
        raw_yfinance = cvx.YahooFinance._get_data_yahoo(stock)
        print(f'{stock}: YAHOO FINANCE RAW')
        print(raw_yfinance)

        tmpdir = Path(tempfile.mkdtemp())
        cvx_cleaned = cvx.YahooFinance(stock, base_location=tmpdir).data
        shutil.rmtree(tmpdir)
        print(f'{stock}: CVXPORTFOLIO CLEANED')
        print(cvx_cleaned)

        yf_log10r = np.log10(raw_yfinance.adjclose).diff().shift(-1)
        cvx_log10r = np.log10(1 + cvx_cleaned['return'])

        if PLOT:
            figures[stock], axes = plt.subplots(
                3, figsize=(10/1.62, 10), layout='constrained')

            raw_yfinance.iloc[:, :5].plot(ax=axes[0])
            axes[0].set_yscale('log')
            axes[0].set_title(f'{stock}: RAW YAHOO FINANCE')

            cvx_cleaned.iloc[:, :4].plot(ax=axes[1])
            axes[1].set_title(f'{stock}: CVXPORTFOLIO CLEANED DATA')
            axes[1].set_yscale('log')

            (yf_log10r.cumsum() - yf_log10r.sum()).plot(
                label='Yahoo Finance total close-to-close', ax=axes[2])
            (cvx_log10r.cumsum() - cvx_log10r.sum()).plot(
                label='Cvxportfolio total open-to-open', ax=axes[2])
            axes[2].set_title(f'{stock}: CUMULATIVE LOG10 RETURNS (SCALED)')
            axes[2].legend()

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
        print('How many zero volumes:',
            (cvx_cleaned['valuevolume'] == 0.).mean())

    print('\nCvxportfolio dropped rows %:')
    print(ALL_DROPPED_ROWS_PCT.sort_values().tail())

    print('\nCvxportfolio min logreturns:')
    print(ALL_MIN_LR.sort_values().head())

    print('\nCvxportfolio max logreturns:')
    print(ALL_MAX_LR.sort_values().tail())

    # we use this to save the plots for the documentation
    if 'CVXPORTFOLIO_SAVE_PLOTS' in os.environ:
        for stock, fig in figures.items():
            fig.savefig(f'{stock}_data_cleaning.png')
    else:
        plt.show()
