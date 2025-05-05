import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cvxportfolio as cvx

strats = ['dow30_daily', 'ndx100_daily', 'sp500_daily', 'ftse100_daily', 'marketneutral_daily']
etfs = ['DIA', 'QQQ', 'SPY', 'ISF.L', None]

for i, strat in enumerate(strats):
    print('\nSTRATEGY', strat)
    h = pd.read_json(strat + '_initial_holdings.json').T
    v = h.sum(1)
    lrets = np.log(v).diff().shift(-1)
    print(np.exp(lrets)-1)
    plt.figure()
    lrets.cumsum().plot(label=strat)
    if etfs[i] is not None:
        etf_ret = cvx.YahooFinance(etfs[i]).data['return'].loc[lrets.index]
        print(etf_ret)
        print(etfs[i], 'mean logreturn', np.log(1+etf_ret).mean())
        np.log(1+etf_ret).cumsum().plot(label=etfs[i])
    print(strat, 'mean logreturn', lrets.mean())
    plt.legend()


plt.show()
