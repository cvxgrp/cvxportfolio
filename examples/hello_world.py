import pandas as pd
import cvxportfolio as cp
import yfinance
import pandas_datareader
import matplotlib.pyplot as plt

# download stock returns with yfinance
tickers = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM"]
returns = pd.DataFrame(dict([(ticker, yfinance.download(ticker)["Adj Close"].pct_change()) for ticker in tickers]))

# download return on cash from FRED
returns[["USDOLLAR"]] = pandas_datareader.get_data_fred("DFF", start='2000-01-01') / (250 * 100)
returns = returns.fillna(method="ffill").dropna()

# create forecasts by rolling means 
r_hat = returns.rolling(window=1000).mean().shift(1).dropna()
Sigma_hat = returns.shift(1).rolling(window=1000).cov().dropna()

# for the cash return forecast instead simply use yesterday's return
r_hat['USDOLLAR'] = returns['USDOLLAR'].shift(1)

# create transaction cost and holding cost models
tcost_model = cp.TcostModel(half_spread=10e-4)
hcost_model = cp.HcostModel(borrow_costs=1e-4)

# define a single period optimization policy (see Chapter 4 of the book)
risk_model = cp.FullCovariance(Sigma_hat)
gamma_risk, gamma_trade, gamma_hold = 1.0, 1.0, 1.0
leverage_limit = cp.LeverageLimit(3)

spo_policy = cp.SinglePeriodOpt(
    return_forecast=r_hat,
    costs=[gamma_risk * risk_model,
           gamma_trade * tcost_model,
           gamma_hold * hcost_model],
    constraints=[leverage_limit],
    solver='ECOS',
)

# initialize the market simulator
market_sim = cp.MarketSimulator(
    returns, [tcost_model, hcost_model], cash_key="USDOLLAR"
)

# and the initial portfolio (uniform on the non-cash assets)
init_portfolio = pd.Series(index=returns.columns, data=250000.0)
init_portfolio.USDOLLAR = 0

# run backtests for the single period optimization policy 
# and a Hold() policy which never trades
results = market_sim.legacy_run_multiple_backtest(
    init_portfolio,
    start_time="2020-01-01",
    end_time="2023-04-01",
    policies=[spo_policy,
             cp.Hold()
            ],
)

# print summaries of the two backtests
print('Single Period Optimization results')
print(results[0].summary())
print('Hold policy results')
print(results[1].summary())

# plot value of the portfolio in time
results[0].v.plot(figsize=(12, 5), label='Single Period Optimization')
results[1].v.plot(figsize=(12, 5), label='Hold')
plt.ylabel('USD')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
results[0].w.iloc[:, :-1].plot()
plt.title('Weights of the portfolio in time')
plt.show()
