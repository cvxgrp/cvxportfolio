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
"""This is a simple example of back-tests with Cvxportfolio.

This is a close translation of what was done in `this notebook
<https://github.com/cvxgrp/cvxportfolio/blob/0.0.X/examples/HelloWorld.ipynb>`_.
In fact, you can see that the results are identical.

The approach used here is not recommended; in particular we download
data externally (it is done better now by the automatic data download
and cleaning code we include in Cvxportfolio). The returns used here
are close-to-close total returns, while our interface computes correctly
the open-to-open total returns.

In this example returns and covariances are forecasted externally,
while today this can be done automatically using the default forecasters
used by :class:`cvxportfolio.ReturnsForecast` and
:class:`cvxportfolio.FullCovariance`.

Nevertheless, you can see by running this that we are still able to
reproduce exactly the behavior of the early development versions
of the library.

.. note::

    To run this, you need to install ``yfinance`` and
    ``pandas_datareader``.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import yfinance

import cvxportfolio as cvx

# Download market data
tickers = ['AMZN', 'GOOGL', 'TSLA', 'NKE']

returns = pd.DataFrame(dict([(ticker,
    yfinance.download(ticker)['Adj Close'].pct_change())
                for ticker in tickers]))

returns["USDOLLAR"] = pdr.get_data_fred(
    'DFF', start="1900-01-01",
    end=pd.Timestamp.today())['DFF']/(252*100)

returns = returns.fillna(method='ffill').iloc[1:]

print('Returns')
print(returns)

# Create market data server.
market_data = cvx.UserProvidedMarketData(
    returns = returns,
    cash_key = 'USDOLLAR')


# Today we'd do all the above by (no external packages needed):
# market_data = cvx.DownloadedMarketData(
#     universe = ['AMZN', 'GOOGL', 'TSLA', 'NKE'],
#     cash_key = 'USDOLLAR')


print('Historical returns:')
print(market_data.returns)

# Build forecasts of expected returns and covariances.
# Note that we shift so that each day we use ones built
# using past returns only. This is done automatically
# by the forecasters used by default in the stable versions
# of Cvxportfolio.
r_hat_with_cash = market_data.returns.rolling(
    window=250).mean().shift(1).dropna()
Sigma_hat_without_cash = market_data.returns.iloc[:, :-1
    ].rolling(window=250).cov().shift(4).dropna()

r_hat = r_hat_with_cash.iloc[:, :-1]
r_hat_cash = r_hat_with_cash.iloc[:, -1]
print('Expected returns forecast:')
print(r_hat_with_cash)

# Define transaction and holding cost models.

# Half spread.
HALF_SPREAD = 10E-4

# In the 2016 development code borrow fees were expressed per-period.
# In the stable version we require annualized percent.
# This value corresponds to 1 basis point per period, which was in the
# original example.
BORROW_FEE = 2.552

tcost_model = cvx.TcostModel(a=HALF_SPREAD, b=None)
hcost_model = cvx.HcostModel(short_fees=BORROW_FEE)

# As risk model, we use the historical covariances computed above.
# Note that the stable version of Cvxportfolio requires the covariance
# matrix to not include cash (as it shouldn't). In the development versions
# it was there. It doesn't make any difference in numerical terms.
risk_model = cvx.FullSigma(Sigma_hat_without_cash)

# Constraint.
leverage_limit = cvx.LeverageLimit(3)

# Define a single-period optimization policy; its objective function is
# maximized.
gamma_risk, gamma_trade, gamma_hold = 5., 1., 1.
spo_policy = cvx.SinglePeriodOpt(
    objective = cvx.ReturnsForecast(r_hat) + cvx.CashReturn(r_hat_cash)
        - gamma_risk * risk_model
        - gamma_trade * tcost_model
        - gamma_hold * hcost_model,
    constraints=[leverage_limit],
    include_cash_return=False)

# Define the market simulator.
market_sim = cvx.MarketSimulator(
    market_data = market_data,
    costs = [
        cvx.TcostModel(a=HALF_SPREAD, b=None),
        cvx.HcostModel(short_fees=BORROW_FEE)])

# Initial portfolio, uniform on non-cash assets.
init_portfolio = pd.Series(
    index=market_data.returns.columns, data=250000.)
init_portfolio.USDOLLAR = 0

# Run two back-tests.
results = market_sim.run_multiple_backtest(
    h=[init_portfolio]*2,
    start_time='2013-01-03',  end_time='2016-12-31',
    policies=[spo_policy, cvx.Hold()])

print('Back-test result, single-period optimization policy:')
print(results[0])

print('Back-test result, Hold policy:')
print(results[1])

results[0].v.plot(label='SPO')
results[1].v.plot(label='Hold policy')
plt.title('Portfolio total value in time (USD)')
plt.legend()
plt.show()

results[0].w.plot()
plt.title('SPO weights in time')
plt.show()
