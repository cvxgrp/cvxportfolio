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
"""

import cvxportfolio as cvx
import pandas as pd
import matplotlib.pyplot as plt


# Download market data.
market_data = cvx.DownloadedMarketData(
    universe = ['AMZN', 'GOOGL', 'TSLA', 'NKE'],
    cash_key = 'USDOLLAR')

print('Historical open-to-open total returns:')
print(market_data.returns)

# Define transaction and holding cost models.
HALF_SPREAD = 10E-4
BORROW_FEE = 1E-4 * (252 / 100) # in annualized percentage
tcost_model = cvx.TcostModel(a=HALF_SPREAD)
hcost_model = cvx.HcostModel(short_fees=BORROW_FEE)

# As returns forecast, we simply take the historical means
# computed at each point in the back-test (looking only
# at past returns). That's the default behavior; we
# may as well pass a dataframe here with different predictions.
r_hat = cvx.ReturnsForecast()

# As risk model, we choose the full historical covariance.
# It is computed every day using the full past historical 
# returns at that point. (We may as well provide it as a 
# dataframe.)
risk_model = cvx.FullSigma()

# Constraint.
leverage_limit = cvx.LeverageLimit(3)

# Define a single-period optimization policy; its
# objective function is maximized.
gamma_risk, gamma_trade, gamma_hold = 5., 1., 1.
spo_policy = cvx.SinglePeriodOpt(
    objective = r_hat
        - gamma_risk * risk_model
        - gamma_trade * tcost_model 
        - gamma_hold * hcost_model,
    constraints=[leverage_limit])

# Define the market simulator.
market_sim = cvx.MarketSimulator(
    market_data = market_data,
    costs = [
        cvx.TcostModel(a=HALF_SPREAD),
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
