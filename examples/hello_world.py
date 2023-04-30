import cvxportfolio as cp
import matplotlib.pyplot as plt

# from cvxportfolio.returns import Kelly

# define a portfolio optimization policy
# with rolling window mean (~10 yrs) returns
# with forecast error risk on returns (see the book)
# rolling window mean (~10 yrs) covariance
# and forecast error risk on covariance (see the book)

#LOOKBACK = 2500

policy = cp.SinglePeriodOptimization(objective = 
        #Kelly(2500)
        cp.ReturnsForecast()  #rolling=LOOKBACK) -
       # -.75 * cp.ReturnsForecastError() 
        #10 * cp.FullCovariance(#halflife=10000, addmean=True
        #),#, kappa=.25), #addmean=True
       # - 10 * cp.FullCovariance(#halflife=10000, addmean=True
        #)
        
        
        - 10 * cp.FactorModelCovariance(num_factors=5)
       # - 10 * cp.DiagonalCovariance() 
        # - 10 * cp.FullCovariance()
        
        
        #- 1 * cp.RiskForecastError()
        
        ,#, kappa=.25), #addmean=True
        #)
        
        # 5 * cp.RollingWindowFactorModelRisk(LOOKBACK, num_factors=5, forecast_error_kappa = 0.5), 
        constraints = [#cp.LeverageLimit(.1), cp.MaxWeights(0.02), #cp.MinWeights(-0.02), #cp.DollarNeutral(),
    ],verbose=True
        )
        
# define a market simulator, which downloads stock market data and stores it locally
# in ~/cvxportfolio/        
simulator = cp.MarketSimulator(sorted(set(["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM", 'NKE', 'MCD', 'GE', 'CVX', 
                              'XOM', 'MMM', 'UNH', 'HD', 'WMT',# 'ORCL', 'INTC', 'JPM', 'BLK', 'BA', 'NVDA', 
                               'F', 'GS', 'AMD', 'CSCO', 'KO', 'HON', 'DIS', 'FRC', # 'DOW',
                                'V', 'ADBE', 'AMGN', 'CAT', 'BA', 'HON', 'JNJ', 'AXP', 'PG', 'JPM', 
                           'IBM', 'MRK', 'MMM', 'VZ', 'WBA', 'INTC', 'PEP', 'AVGO',
                            'COST', 'TMUS', 'CMCSA', 'TXN', 'NFLX', 'SBUX', 'GILD',
                            'ISRG', 'MDLZ', 'BKNG', 'AMAT', 'ADI', 'ADP', 'VRTX', 
                            'REGN', #'PYPL',
                             'FISV', 'LRCX'
                        
                        ]
                            )),
                             rolling_window_sigma_estimator=250)

#signal = simulator.returns.data.iloc[:, :-1].ewm(halflife=10000).mean().shift(1)
#policy = cp.RankAndLongShort(signal, num_long=10, num_short=0, target_leverage=.1)
#policy = cp.Uniform()
# perform a backtest (by default it starts with 1E6 USD cash)
backtest = cp.BackTest(policy, simulator, '2012-01-01', '2023-04-21')

# plot value of the portfolio in time
backtest.v.plot(figsize=(12, 5), label='Single Period Optimization')
plt.ylabel('USD')
plt.yscale('log')
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
