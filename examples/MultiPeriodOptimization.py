#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# In[8]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxportfolio as cp

plotdir = "../portfolio/plots/"
datadir = "../data/"

sigmas = pd.read_csv(datadir + "sigmas.csv.gz", index_col=0, parse_dates=[0]).iloc[
    :, :-1
]
returns = pd.read_csv(datadir + "returns.csv.gz", index_col=0, parse_dates=[0])
volumes = pd.read_csv(datadir + "volumes.csv.gz", index_col=0, parse_dates=[0]).iloc[
    :, :-1
]

w_b = pd.Series(index=returns.columns, data=1)
w_b.USDOLLAR = 0.0
w_b /= sum(w_b)

start_t = "2012-01-01"
end_t = "2016-12-29"

simulated_tcost = cp.TcostModel(
    half_spread=0.0005 / 2.0, nonlin_coeff=1.0, sigma=sigmas, volume=volumes
)
simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
simulator = cp.MarketSimulator(
    returns,
    costs=[simulated_tcost, simulated_hcost],
    market_volumes=volumes,
    cash_key="USDOLLAR",
)

return_estimate = pd.read_csv(
    datadir + "return_estimate.csv.gz", index_col=0, parse_dates=[0]
).dropna()
volume_estimate = pd.read_csv(
    datadir + "volume_estimate.csv.gz", index_col=0, parse_dates=[0]
).dropna()
sigma_estimate = pd.read_csv(
    datadir + "sigma_estimate.csv.gz", index_col=0, parse_dates=[0]
).dropna()

optimization_tcost = cp.TcostModel(
    half_spread=0.0005 / 2.0,
    nonlin_coeff=1.0,
    sigma=sigma_estimate,
    volume=volume_estimate,
)
optimization_hcost = cp.HcostModel(borrow_costs=0.0001)

risk_data = pd.HDFStore(datadir + "risk_model.h5")
risk_model = cp.FactorModelSigma(
    risk_data.exposures, risk_data.factor_sigma, risk_data.idyos
)


# In[9]:


all_return_estimates = {}
n_p1 = returns.shape[1]
T = returns.shape[0]

for i, t in enumerate(returns.index[:-1]):
    all_return_estimates[(t, t)] = return_estimate.loc[t]
    tp1 = returns.index[i + 1]
    all_return_estimates[(t, tp1)] = return_estimate.loc[tp1]

returns_forecast = cp.MPOReturnsForecast(all_return_estimates)
results_MPO = {}


# ## MPO Coarse search 

# In[ ]:


import cvxpy as cvx

policies = {}
gamma_risks_coarse = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
gamma_tcosts_coarse = [1, 2, 5, 10, 20]
for gamma_risk in gamma_risks_coarse:
    for gamma_tcost in gamma_tcosts_coarse:
        policies[(gamma_risk, gamma_tcost)] = cp.MultiPeriodOpt(
            return_forecast=returns_forecast,
            costs=[
                gamma_risk * risk_model,
                gamma_tcost * optimization_tcost,
                optimization_hcost,
            ],
            constraints=[cp.LeverageLimit(3)],
            trading_times=list(
                returns.index[(returns.index >= start_t) & (returns.index <= end_t)]
            ),
            lookahead_periods=2,
            terminal_weights=None,
        )

results_MPO.update(
    {
        k: v
        for k, v in zip(
            policies.keys(),
            simulator.run_multiple_backtest(
                w_b * 1e8,
                start_time=start_t,
                end_time=end_t,
                policies=policies.values(),
                parallel=True,
            ),
        )
    }
)


# In[27]:


result_df_coarse = pd.DataFrame()
for k in results_MPO:
    if k[0] in gamma_risks_coarse and k[1] in gamma_tcosts_coarse:
        result_df_coarse.loc[k[0], k[1]] = results_MPO[k]

result_df = result_df_coarse.loc[
    sorted(result_df_coarse.index), sorted(result_df_coarse.columns)
]


# In[28]:


plt.figure(figsize=(8, 5))
for gamma_tcost in result_df.columns:
    x = [el.excess_returns.std() * 100 * np.sqrt(250) for el in result_df[gamma_tcost]]
    y = [el.excess_returns.mean() * 100 * 250 for el in result_df[gamma_tcost]]
    plt.plot(
        np.array(x),
        np.array(y),
        ".-",
        label="$\gamma^\mathrm{trade} = %g$" % gamma_tcost,
    )
plt.legend(loc="lower right")
plt.xlabel("Risk")
plt.ylabel("Return")
plt.xlim([0, 20])
plt.ylim([0, 30])


import matplotlib.ticker as mtick

ax = plt.gca()
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

plt.savefig(plotdir + "mpo_riskrewardfrontier.png")


# # MPO Pareto search 

# In[17]:


results_pareto = {}


# In[ ]:


policies = {}
# gamma_risks_pareto=[int(round(el)) for el in np.logspace(0,3,13)]
gamma_risks_pareto = [1, 2, 3, 6, 10, 18, 32, 56, 100, 178, 316, 562, 1000]
gamma_tcosts_pareto = [7, 8, 9, 10, 11, 12]
gamma_holdings = [0.1, 1.0, 10.0, 100.0, 1000.0]
for gamma_risk in gamma_risks_pareto:
    for gamma_tcost in gamma_tcosts_pareto:
        for gamma_holding in gamma_holdings:
            policies[(gamma_risk, gamma_tcost, gamma_holding)] = cp.MultiPeriodOpt(
                alpha_model=returns_forecast,
                costs=[
                    gamma_risk * risk_model,
                    gamma_tcost * optimization_tcost,
                    gamma_holding * optimization_hcost,
                ],
                constraints=[cp.LeverageLimit(3)],
                trading_times=list(
                    returns.index[(returns.index >= start_t) & (returns.index <= end_t)]
                ),
                lookahead_periods=2,
                terminal_weights=None,
            )

import warnings

warnings.filterwarnings("ignore")
results_pareto.update(
    dict(
        zip(
            policies.keys(),
            simulator.run_multiple_backtest(
                1e8 * w_b,
                start_time=start_t,
                end_time=end_t,
                policies=policies.values(),
                parallel=True,
            ),
        )
    )
)


# In[55]:


table = pd.DataFrame()
table[r"$\gamma^\mathrm{risk}$"] = [el[0] for el in results_pareto.keys()]
table[r"$\gamma^\mathrm{trade}$"] = [el[1] for el in results_pareto.keys()]
table[r"$\gamma^\mathrm{hold}$"] = ["%g" % el[2] for el in results_pareto.keys()]
table["Return"] = [
    (results_pareto[k].excess_returns.mean() * 100 * 250) for k in results_pareto.keys()
]
table["Risk"] = [
    (results_pareto[k].excess_returns.std() * 100 * np.sqrt(250))
    for k in results_pareto.keys()
]

table = table.sort_values("Risk", ascending=False).reset_index()
del table["index"]
is_pareto = lambda i: table.loc[i, "Return"] >= max(table.ix[i:].Return)
table["is_pareto"] = [is_pareto(i) for i in range(len(table))]
table.to_csv(datadir + "mpo_pareto_results.csv", float_format="%g")


# In[56]:


plt.figure(figsize=(8, 5))
plt.scatter(table.Risk.values, table.Return.values)
plt.plot(
    table[table.is_pareto].Risk,
    table[table.is_pareto].Return,
    "C1.-",
    label="Pareto optimal frontier",
)
plt.legend(loc="lower right")
plt.xlabel("Risk")
plt.ylabel("Return")
plt.xlim([0, 20])
plt.ylim([0, 30])

import matplotlib.ticker as mtick

ax = plt.gca()
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

plt.savefig(plotdir + "mpo_pareto.png")


# In[21]:


xlim = 20
ylim = 30
tableprint = table[table.is_pareto]
tableprint = tableprint[tableprint.Risk <= xlim]
tableprint = tableprint[tableprint.Return <= ylim]
del tableprint["is_pareto"]
tableprint.Risk = tableprint.Risk.apply(lambda x: "%.2f%%" % x)
tableprint.Return = tableprint.Return.apply(lambda x: "%.2f%%" % x)
print(
    tableprint.iloc[::-1]
    .to_latex(float_format="%.2f", escape=False, index=False)
    .replace("%", r"\%")
)


# # SPO vs MPO 

# In[57]:


table_spo = pd.read_csv("spo_pareto_results.csv", index_col=0)
plt.figure(figsize=(8, 5))

plt.plot(
    table[table.is_pareto].Risk,
    table[table.is_pareto].Return,
    "C3.-",
    label="MPO pareto frontier",
)
plt.plot(
    table_spo[table_spo.is_pareto].Risk,
    table_spo[table_spo.is_pareto].Return,
    "C2.-",
    label="SPO pareto frontier",
)
plt.legend(loc="lower right")
plt.xlabel("Risk")
plt.ylabel("Return")
plt.xlim([0, 20])
plt.ylim([0, 30])

import matplotlib.ticker as mtick

ax = plt.gca()
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))


plt.savefig(plotdir + "spo_vs_mpo_pareto.png")

