import numpy as np
import pandas as pd

import cvxportfolio as cvx

prices = pd.DataFrame(index = [pd.Timestamp('2021-01-01', tz = 'utc'), pd.Timestamp('2021-01-02', tz = 'utc'), pd.Timestamp('2021-01-03', tz = 'utc')], columns = ['STOCK1', 'STOCK2', 'USDOLLAR'])
volume = pd.DataFrame(index = [pd.Timestamp('2021-01-01', tz = 'utc'), pd.Timestamp('2021-01-02', tz = 'utc'), pd.Timestamp('2021-01-03', tz = 'utc'), pd.Timestamp('2021-01-04', tz = 'utc'), pd.Timestamp('2021-01-05', tz = 'utc'), pd.Timestamp('2021-01-06', tz = 'utc'), pd.Timestamp('2021-01-07', tz = 'utc')], columns = ['STOCK1', 'STOCK2'])
prices.loc[pd.Timestamp('2021-01-01', tz = 'utc'), 'STOCK1'] = 100
prices.loc[pd.Timestamp('2021-01-02', tz = 'utc'), 'STOCK1'] = 10
prices.loc[pd.Timestamp('2021-01-03', tz = 'utc'), 'STOCK1'] = 100
prices.loc[pd.Timestamp('2021-01-04', tz = 'utc'), 'STOCK1'] = 10
prices.loc[pd.Timestamp('2021-01-05', tz = 'utc'), 'STOCK1'] = 100
prices.loc[pd.Timestamp('2021-01-06', tz = 'utc'), 'STOCK1'] = 101
prices.loc[pd.Timestamp('2021-01-07', tz = 'utc'), 'STOCK1'] = 100


prices.loc[pd.Timestamp('2021-01-01', tz = 'utc'), 'STOCK2'] = 10
prices.loc[pd.Timestamp('2021-01-02', tz = 'utc'), 'STOCK2'] = 100
prices.loc[pd.Timestamp('2021-01-03', tz = 'utc'), 'STOCK2'] = 10
prices.loc[pd.Timestamp('2021-01-04', tz = 'utc'), 'STOCK2'] = 100
prices.loc[pd.Timestamp('2021-01-05', tz = 'utc'), 'STOCK2'] = 10
prices.loc[pd.Timestamp('2021-01-06', tz = 'utc'), 'STOCK2'] = 100
prices.loc[pd.Timestamp('2021-01-07', tz = 'utc'), 'STOCK2'] = 101

prices.loc[pd.Timestamp('2021-01-01', tz = 'utc'), 'USDOLLAR'] = 1
prices.loc[pd.Timestamp('2021-01-02', tz = 'utc'), 'USDOLLAR'] = 1
prices.loc[pd.Timestamp('2021-01-03', tz = 'utc'), 'USDOLLAR'] = 1
prices.loc[pd.Timestamp('2021-01-04', tz = 'utc'), 'USDOLLAR'] = 1
prices.loc[pd.Timestamp('2021-01-05', tz = 'utc'), 'USDOLLAR'] = 1
prices.loc[pd.Timestamp('2021-01-06', tz = 'utc'), 'USDOLLAR'] = 1
prices.loc[pd.Timestamp('2021-01-07', tz = 'utc'), 'USDOLLAR'] = 1

# using constant large volume bc we don't really care about this for dummy test
volume['STOCK1'] = 1_000_000
volume['STOCK2'] = 1_000_000
returns = prices.pct_change(fill_method=None).dropna()
prices = prices.drop(['USDOLLAR'], axis =1)

returns = returns.astype(float)
volume = volume.astype(float)
prices = prices.astype(float)

r_hat = returns.copy()
ret_forecast1 = cvx.ReturnsForecast(r_hat = r_hat)

def make_policy_single(ret_forecast1, leverage_limit = 1):
    return  cvx.SinglePeriodOptimization(
        objective =
        ret_forecast1,

        constraints= [cvx.LeverageLimit(leverage_limit)],

        solver='ECOS', ignore_dpp=True, include_cash_return= True)

policy = make_policy_single(ret_forecast1)

sim = cvx.MarketSimulator(returns = returns, volumes = volume, prices = prices, cash_key= 'USDOLLAR',
                                   min_history=pd.Timedelta('1d'))

bt_result = sim.backtest(policy, initial_value = 1_500_000, start_time = pd.Timestamp('2021-01-02', tz = 'utc'), end_time=pd.Timestamp('2021-01-07', tz = 'utc'))
