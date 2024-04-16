import sys

import pandas as pd

import cvxportfolio as cvx

strat = 'dow30_daily' if len(sys.argv) < 2 else sys.argv[1]

w = pd.read_json(strat + '_target_weights.json').T
h = pd.read_json(strat + '_initial_holdings.json').T

uni = [el for el in w.columns if el != 'USDOLLAR']

sim = cvx.StockMarketSimulator(uni)
r = sim.backtest(
    cvx.FixedWeights(w), start_time=w.index[0])

print(h)
print(r.h)
print(h - r.h)

## performance
