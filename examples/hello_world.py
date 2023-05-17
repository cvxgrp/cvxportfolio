import cvxportfolio as cvx
import matplotlib.pyplot as plt


#if __name__ == '__main__':
    
policies = []

for risk_multiplier in [2]:#,1,2,3,#4,5,6]:
    #7,8,9,10]:
    
    

    objective = cvx.ReturnsForecast() -.0 * cvx.ReturnsForecastError() \
    - risk_multiplier * (
    cvx.FactorModelCovariance(num_factors=20) 
    # cvx.FullCovariance() 
    + 0.1 * cvx.RiskForecastError()) - cvx.TransactionCost(exponent=2)  - cvx.HoldingCost()
    constraints = [cvx.MarketNeutral()]#cvx.LongOnly(),cvx.LeverageLimit(1)]

    policies.append(cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=1))#, solver='ECOS')


universe = sorted(set(["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM", 
                                    'NKE', 'MCD', 'GE', 'CVX', 
                                  'XOM', 'MMM', 'UNH', 'HD', 'WMT', 'ORCL', 'INTC', 'JPM', 'BLK', 'BA', 'NVDA', 
                                   'F', 'GS', 'AMD', 'CSCO', 'KO', 'HON', 'DIS', 
                                   'V', 
                                    'ADBE', 'AMGN', 'CAT', 'BA', 'HON', 'JNJ', 'AXP', 'PG', 'JPM', 
                               'IBM', 'MRK', 'MMM', 'VZ', 'WBA', 'INTC', 'PEP', 'AVGO',
                                'COST', 'TMUS', 
                                'CMCSA', 'TXN', 'NFLX', 'SBUX', 'GILD',
                                'ISRG', 'MDLZ', 'BKNG', 'AMAT', 'ADI', 'ADP', 'VRTX', 
                                'REGN', 'PYPL',
                                 'FISV', 'LRCX', 'PYPL',
                                  'MU', 'CSX', 'MELI', 
                                  'MNST', 'ATVI',
                                 'PANW', 
                                 'ORLY', 'ASML', 'SNPS', 'CDNS', 'MAR', 'KLAC', 'FTNT', 'CHTR', 
                                'CHTR', 'MRNA', 
                                 'KHC', 
                                 'CTAS', 'AEP', 'DXCM', 'LULU', 'KDP', 
                                 'AZN',
                                 'BIIB', 'ABNB', 
                                 'NXPI', 
                                 'ADSK', 'EXC', 'MCHP', 'IDXX', 'CPRT', 'PAYX',
                                 'PCAR', 'XEL', 'PDD', 
                                 'WDAY', 
                                 'SGEN', 'ROST', 'DLTR', 'RA', 
                                 #'MRVL',
                                 'ODFL', 'VRSK',
                                  'ILMN', 'CTSH', 'FAST', 'CSGP', 'WBD', 'GFS', 
                                 'CRWD', 
                             'BKR', 'WBA', 'CEG', 
                             'ANSS', 'DDOG', 
                             'EBAY', 'FANG', 
                             'ENPH',
                              'ALGN', 'TEAM',
                             'ZS', 
                             'JD', 'ZM' ,
                             'SIRI','LCID', 'RIVN', 'SGEN', 'MDLZ', 'NFLX', 'GOOGL', 'DXCM']))
   
#universe = universe[:3]                          
simulator = cvx.MarketSimulator(universe)

result = simulator.backtest(policies, start_time='2020-01-01', initial_value=1E9)

# import numpy as np
# x = [el.excess_returns.std() * 100 * np.sqrt(el.PPY) for el in results]
# y = [el.excess_returns.mean() * 100 * (el.PPY) for el in results]
# plt.plot(x,y, '.-')
# plt.show()
# raise Exception

print(result)
import numpy as np
q = cvx.YfinanceTimeSeries('QQQ', base_location=cvx.BASE_LOCATION)
q.pre_evaluation()
lq = np.log(1 + q.data.Return)
lb = result.growth_rates
lb.cumsum().plot(label='BT'); lq.loc[lb.index].cumsum().plot(label='QQQ'); plt.legend(); plt.show()


# plot value of the portfolio in time
result.v.plot(figsize=(12, 5), label='Multi Period Optimization')
plt.ylabel('USD')
plt.yscale('log')
plt.title('Total value of the portfolio in time')
plt.show()

# plot weights of the (non-cash) assets for the SPO policy
import numpy as np
biggest_weights = np.abs(result.w.iloc[:, :-1]).max().sort_values().iloc[-10:].index
result.w[biggest_weights].plot()
plt.title('Largest 10 weights of the portfolio in time')
plt.show()

result.leverage.plot(); plt.show()

result.drawdown.plot(); plt.show()

print('\ntotal tcost ($)', result.tcost.sum())
print('total borrow cost ($)', result.hcost_stocks.sum())
print('total cash return + cost ($)', result.hcost_cash.sum())
