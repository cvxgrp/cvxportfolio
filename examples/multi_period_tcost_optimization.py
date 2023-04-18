import pandas as pd
import cvxportfolio as cp
import yfinance
import pandas_datareader
import numpy as np
import matplotlib.pyplot as plt

# Download the stock data from yfinance
tickers = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA"]
data = {ticker: yfinance.download(ticker) for ticker in tickers}
returns = pd.DataFrame({ticker: data[ticker]['Adj Close'].pct_change() for ticker in tickers})
volumes = pd.DataFrame({ticker: data[ticker]['Volume'] * data[ticker]['Close'] for ticker in tickers})


# get the return on cash from FRED
returns[["USDOLLAR"]] =  pandas_datareader.get_data_fred("DFF", start='2000-01-01') / (250 * 100)
returns = returns.fillna(method="ffill").dropna()

# We compute rolling estimates of the first and second moments of the returns using a window of 1000 days. We shift them by one unit (so at every day we present the optimizer with only past data).
r_hat = returns.rolling(window=1000).mean().shift(1).dropna()
Sigma_hat = returns.shift(1).rolling(window=1000).cov().dropna()

# For the cash return instead we simply use the previous day's return.
r_hat['USDOLLAR'] = returns['USDOLLAR'].shift(1)

# We compute per-stock standard deviations by rolling window
sigma = returns.iloc[:,:-1].rolling(window=1000).std().shift(1).dropna()

# Here we define the transaction cost and holding cost model (sections 2.3 and 2.4 [of the paper](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)). 
# The data can be expressed as:
# - a scalar the same value for all assets and all time periods;
# - a Pandas Series indexed by the asset names, for asset-specific values; 
# - a Pandas DataFrame indexed by timestamps with asset names as columns, 
# for values that vary by asset and in time.
tcost_model = cp.TcostModel(half_spread=10e-4, nonlin_coeff=1., sigma=sigma, volume=volumes.dropna())
hcost_model = cp.HcostModel(borrow_costs=1e-4)

# We define a multi period optimization policy
risk_model = cp.FullCovariance(Sigma_hat)
gamma_risk, gamma_trade, gamma_hold = 5.0, 1.0, 1.0
leverage_limit = cp.LeverageLimit(3)

terminal_weights = pd.Series(0., returns.columns)
terminal_weights['USDOLLAR'] = 1.

trading_times = returns.index[returns.index >= '2023-03-01']
trading_times = trading_times[trading_times <= '2023-03-31']


mpo_policy = cp.MultiPeriodOpt(
    trading_times = trading_times,
    terminal_weights=terminal_weights, 
    lookahead_periods=25, # a month
    return_forecast=cp.LegacyReturnsForecast(r_hat),
    costs=[
        gamma_risk * risk_model,
        gamma_trade * tcost_model,
        gamma_hold * hcost_model,
    ],
    constraints=[leverage_limit],
    solver='ECOS'
)

# And a fixed trade policy
tw = np.ones(len(tickers)+1)
tw[:-1] = -1/(22 * len(tickers))
tw[-1] = 0 - sum(tw[:-1])

fixedtrade = cp.FixedTrades(trades_weights=pd.Series(tw, returns.columns))

# We run a backtest, which returns a result object.
# By calling its summary method we get some basic statistics.



market_sim = cp.MarketSimulator(
    returns, [tcost_model, hcost_model], cash_key="USDOLLAR"
)
init_portfolio = pd.Series(index=returns.columns, data=1E9)
init_portfolio.USDOLLAR = 0
results = market_sim.run_multiple_backtest(
    init_portfolio,
    start_time="2023-03-01",
    end_time="2023-03-31",
    policies=[mpo_policy, 
             fixedtrade,
            ],
)

print('Multi Period Optimization results')
print(results[0].summary())
print('Fixed trades results')
print(results[1].summary())

results[0].h_next.plot()
plt.show()

results[0].w.plot()
plt.show()