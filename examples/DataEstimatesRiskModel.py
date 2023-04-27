import pandas as pd
import numpy as np
import datetime as dt

RISK_FREE_SYMBOL = "USDOLLAR"
datadir = "../data/"
assets = pd.read_csv(datadir + "SP500.txt", comment="#").set_index("Symbol")

SAVE_ESTIMATES = False
SAVE_RISK_MODEL = True

## NOTE (April 2023): The original plots in the paper were produced with data
# downloaded from `quandl`, which is now defunct. We have the data stored
# in the repository and here is the code that was used to download it,
# but it doesn't run any more.
#
# import quandl
#
# QUANDL = {
#     ## Get a key (free) from quandl.com and copy it here
#     "authtoken": "skyJ_Rgz8mgzTuZJ7Biy",
#     "start_date": dt.date(2007, 1, 1),
#     "end_date": dt.date(2016, 12, 31),
# }
#
# data = {}
#
#
# from time import sleep
#
# for ticker in assets.index:
#     if ticker in data:
#         continue
#     print(
#         "downloading %s from %s to %s"
#         % (ticker, QUANDL["start_date"], QUANDL["end_date"])
#     )
#     try:
#         data[ticker] = quandl.get(assets.Quandlcode[ticker], **QUANDL)
#         # sleep(0.1)
#     except quandl.NotFoundError:
#         print("\tInvalid asset code")
#
#
# #### Computation 
#
# keys = [el for el in assets.index if not el in (set(assets.index) - set(data.keys()))]
#
#
# def select_first_valid_column(df, columns):
#     for column in columns:
#         if column in df.columns:
#             return df[column]
#
#
# # extract prices
# prices = pd.DataFrame.from_items(
#     [
#         (k, select_first_valid_column(data[k], ["Adj. Close", "Close", "Value"]))
#         for k in keys
#     ]
# )
#
# # compute sigmas
# open_price = pd.DataFrame.from_items(
#     [(k, select_first_valid_column(data[k], ["Open"])) for k in keys]
# )
# close_price = pd.DataFrame.from_items(
#     [(k, select_first_valid_column(data[k], ["Close"])) for k in keys]
# )
# sigmas = np.abs(np.log(open_price.astype(float)) - np.log(close_price.astype(float)))
#
# # extract volumes
# volumes = pd.DataFrame.from_items(
#     [(k, select_first_valid_column(data[k], ["Adj. Volume", "Volume"])) for k in keys]
# )
#
# # fix risk free
# prices[RISK_FREE_SYMBOL] = (
#     10000 * (1 + prices[RISK_FREE_SYMBOL] / (100 * 250)).cumprod()
# )
#
#
# # filter NaNs - threshold at 2% missing values
# bad_assets = prices.columns[prices.isnull().sum() > len(prices) * 0.02]
# if len(bad_assets):
#     print("Assets %s have too many NaNs, removing them" % bad_assets)
#
# prices = prices.loc[:, ~prices.columns.isin(bad_assets)]
# sigmas = sigmas.loc[:, ~sigmas.columns.isin(bad_assets)]
# volumes = volumes.loc[:, ~volumes.columns.isin(bad_assets)]
#
# nassets = prices.shape[1]
#
# # days on which many assets have missing values
# bad_days1 = sigmas.index[sigmas.isnull().sum(1) > nassets * 0.9]
# bad_days2 = prices.index[prices.isnull().sum(1) > nassets * 0.9]
# bad_days3 = volumes.index[volumes.isnull().sum(1) > nassets * 0.9]
# bad_days = pd.Index(set(bad_days1).union(set(bad_days2)).union(set(bad_days3))).sort_values()
# print("Removing these days from dataset:")
# print(pd.DataFrame(
#         {"nan price": prices.isnull().sum(1)[bad_days],
#          "nan volumes": volumes.isnull().sum(1)[bad_days],
#          "nan sigmas": sigmas.isnull().sum(1)[bad_days],
#         }
#     )
# )
#
# prices = prices.loc[~prices.index.isin(bad_days)]
# sigmas = sigmas.loc[~sigmas.index.isin(bad_days)]
# volumes = volumes.loc[~volumes.index.isin(bad_days)]
#
# # extra filtering
# print(pd.DataFrame(
#         {"remaining nan price": prices.isnull().sum(),
#          "remaining nan volumes": volumes.isnull().sum(),
#          "remaining nan sigmas": sigmas.isnull().sum(),
#         }
#     )
# )
# prices = prices.fillna(method="ffill")
# sigmas = sigmas.fillna(method="ffill")
# volumes = volumes.fillna(method="ffill")
# print(pd.DataFrame(
#         {"remaining nan price": prices.isnull().sum(),
#          "remaining nan volumes": volumes.isnull().sum(),
#          "remaining nan sigmas": sigmas.isnull().sum(),
#         }
#     )
# )


# # #### Save
#
#
# # make volumes in dollars
# volumes = volumes * prices
#
# # compute returns
# returns = (prices.diff() / prices.shift(1)).fillna(method="ffill").ix[1:]
#
# bad_assets = returns.columns[((-0.5 > returns).sum() > 0) | ((returns > 2.0).sum() > 0)]
# if len(bad_assets):
#     print("Assets %s have dubious returns, removed" % bad_assets)
#
# prices = prices.loc[:, ~prices.columns.isin(bad_assets)]
# sigmas = sigmas.loc[:, ~sigmas.columns.isin(bad_assets)]
# volumes = volumes.loc[:, ~volumes.columns.isin(bad_assets)]
# returns = returns.loc[:, ~returns.columns.isin(bad_assets)]
#
# # remove USDOLLAR
# prices = prices.iloc[:, :-1]
# sigmas = sigmas.iloc[:, :-1]
# volumes = volumes.iloc[:, :-1]
#
#
# # save data
# prices.to_csv(datadir + "prices.csv.gz", compression="gzip", float_format="%.3f")
# volumes.to_csv(datadir + "volumes.csv.gz", compression="gzip", float_format="%d")
# returns.to_csv(datadir + "returns.csv.gz", compression="gzip", float_format="%.3e")
# sigmas.to_csv(datadir + "sigmas.csv.gz", compression="gzip", float_format="%.3e")

## LOAD DATA
prices = pd.read_csv(datadir + "prices.csv.gz", index_col=0, parse_dates=True)
volumes = pd.read_csv(datadir + "volumes.csv.gz", index_col=0, parse_dates=True)
returns = pd.read_csv(datadir + "returns.csv.gz", index_col=0, parse_dates=True)
sigmas = pd.read_csv(datadir + "sigmas.csv.gz", index_col=0, parse_dates=True)

## Estimates 

print("typical variance of returns: %g" % returns.var().mean())

sigma2_n = 0.02
sigma2_r = 0.0005

np.random.seed(0)
noise = pd.DataFrame(
    index=returns.index,
    columns=returns.columns,
    data=np.sqrt(sigma2_n) * np.random.randn(*returns.values.shape),
)
return_estimate = (returns + noise) * sigma2_r / (sigma2_r + sigma2_n)
return_estimate.USDOLLAR = returns.USDOLLAR

if SAVE_ESTIMATES:
    return_estimate.to_csv(datadir + "return_estimate.csv.gz", compression="gzip", float_format="%.3e")

agree_on_sign = np.sign(returns.iloc[:, :-1]) == np.sign(return_estimate.iloc[:, :-1])
print("Return predictions have the right sign %.1f%% of the times" % (100 * agree_on_sign.sum().sum() / (agree_on_sign.shape[0] * (agree_on_sign.shape[1] - 1))))

if SAVE_ESTIMATES:
    volume_estimate = volumes.rolling(window=10, center=False).mean().dropna()
    volume_estimate.to_csv(datadir + "volume_estimate.csv.gz", compression="gzip", float_format="%d")
    sigmas_estimate = sigmas.rolling(window=10, center=False).mean().dropna()
    sigmas_estimate.to_csv(datadir + "sigma_estimate.csv.gz", compression="gzip", float_format="%.3e")

## Risk model 

start_t = "2012-01-01"
end_t = "2016-12-31"

first_days_month = pd.date_range(start=returns.index[next(i for (i, el) in enumerate(returns.index >= start_t) if el) - 1
    ], end=returns.index[-1], freq="MS")

k = 15
exposures, factor_sigma, idyos = {}, {}, {}
for day in first_days_month:
    used_returns = returns.loc[
        (returns.index < day) & (returns.index >= day - pd.Timedelta("730 days"))
    ]
    second_moment = (
        used_returns.values.T @ used_returns.values / used_returns.values.shape[0]
    )
    eival, eivec = np.linalg.eigh(second_moment)
    factor_sigma[day] = np.diag(eival[-k:])
    exposures[day] = pd.DataFrame(data=eivec[:, -k:], index=returns.columns)
    idyos[day] = pd.Series(
        data=np.diag(eivec[:, :-k] @ np.diag(eival[:-k]) @ eivec[:, :-k].T),
        index=returns.columns,
    )
    

# build multiindexed dataframe of exposures
major_index = pd.DatetimeIndex(exposures.keys())
minor_index = pd.DataFrame(list(exposures.values())[0]).columns
columns = pd.DataFrame(list(exposures.values())[0]).index
exposures_df = pd.DataFrame(np.vstack([exposures[k].values.T for k in exposures]),
    index=pd.MultiIndex.from_product([major_index, minor_index]), 
    columns = columns)

# build multiindexed dataframe of factor sigmas 
major_index = pd.DatetimeIndex(factor_sigma.keys())
minor_index = pd.DataFrame(list(factor_sigma.values())[0]).index
factor_sigma_df = pd.DataFrame(np.vstack([factor_sigma[k] for k in factor_sigma]),
    index=pd.MultiIndex.from_product([major_index, minor_index]), columns = minor_index)


if SAVE_RISK_MODEL:
    #pd.Panel(exposures).swapaxes(1, 2).to_hdf(datadir + "risk_model.h5", "exposures")
    #pd.DataFrame(idyos).T.to_hdf(datadir + "risk_model.h5", "idyos")
    exposures_df.to_csv(datadir + "exposures.csv.gz", compression="gzip", float_format="%.3e")
    pd.DataFrame(idyos).T.to_csv(datadir + "idyo_vars.csv.gz", compression="gzip", float_format="%.3e")
    factor_sigma_df.to_csv(datadir + "factor_sigma.csv.gz", compression="gzip", float_format="%.3e")
    





