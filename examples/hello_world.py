import cvxportfolio as cp
import matplotlib.pyplot as plt

# define a portfolio optimization policy
# with rolling window mean (~10 yrs) returns
# with forecast error risk on returns (see the book)
# rolling window mean (~10 yrs) covariance
# and forecast error risk on covariance (see the book)

LOOKBACK = 2500

policy = cp.SinglePeriodOptimization(objective = 
        cp.ReturnsForecast() - #rolling=LOOKBACK) -
        .25 * cp.ReturnsForecastErrorRisk() -
        2.5 * cp.FullCovariance(kappa=.12, addmean=True), 
        # 5 * cp.RollingWindowFactorModelRisk(LOOKBACK, num_factors=5, forecast_error_kappa = 0.5), 
        constraints = [cp.LeverageLimit(3)]
        )
        
# define a market simulator, which downloads stock market data and stores it locally
# in ~/cvxportfolio/        
simulator = cp.MarketSimulator(["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM", 'NKE', 'MCD', 'GE', 'CVX', 
                                'XOM', 'MMM', 'UNH', 'HD', 'WMT', 'ORCL', 'INTC', 'JPM', 'BLK', 'BA', 'NVDA', 
                                'F', 'GS', 'AMD', 'CSCO', 'KO', 'HON', 'DIS',# 'DOW',
                                # 'V', 'ADBE'
                            ])

#policy = cp.Uniform()
# perform a backtest (by default it starts with 1E6 USD cash)
backtest = cp.BackTest(policy, simulator, '2018-01-01', '2023-04-21')

# plot value of the portfolio in time
backtest.v.plot(figsize=(12, 5), label='Single Period Optimization')
plt.ylabel('USD')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
backtest.w.iloc[:, :-1].plot()
plt.title('Weights of the portfolio in time')
plt.show()

print('total tcost', backtest.tcost.sum())
print('total borrow cost', backtest.hcost_stocks.sum())
print('total cash return + cost', backtest.hcost_cash.sum())

print('sharpe ratio', backtest.sharpe_ratio)

print('mean excess lret', backtest.excess_growth_rates.mean())
print('std excess lret', backtest.excess_growth_rates.std())
