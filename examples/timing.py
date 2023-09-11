import cvxportfolio as cvx
import matplotlib.pyplot as plt
import time
import pandas as pd

SP500 = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ATVI', 'ADM', 'ADBE', 'ADP',
        'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE',
        'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN',
        'AMCR', 'AMD', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK',
        'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA',
        'AAPL', 'AMAT', 'APTV', 'ACGL', 'ANET', 'AJG', 'AIZ', 'T', 'ATO',
        'ADSK', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BBWI',
        'BAX', 'BDX', 'WRB', 'BRK-B', 
        'BBY', 'BIO', 'TECH', 'BIIB', 'BLK',
        'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR',
        'BRO', 'BF-B',
         'BG', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF',
        'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW',
        'CE', 'CNC', 'CNP', 'CDAY', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX',
        'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
        'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG',
        'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'CTVA', 'CSGP',
        'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI',
        'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS',
        'DISH', 'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE',
        'DUK', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA',
        'ELV', 'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX',
        'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC',
        'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT',
        'FDX', 'FITB', 'FSLR', 'FE', 'FIS', 'FLT', 'FMC', 'F',
        'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GEHC',
        'GEN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN',
        'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES',
        'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ',
        'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY',
        'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG',
        'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 'JCI',
        'JPM', 'JNPR', 'K', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI',
        'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS',
        'LEN', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB',
        'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH',
        'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM',
        'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ',
        'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP',
        'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN',
        'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI',
        'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OGN', 'OTIS',
        'PCAR', 'PKG', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP',
        'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG',
        'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA',
        'PHM', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O',
        'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'RHI', 'ROK', 'ROL',
        'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE',
        'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SNA', #'SEDG', 
        'SO',
        'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SYF', 'SNPS',
        'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY',
        'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT',
        'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA',
        'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VRSN',
        'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'VICI', 'V', 'VMC', 'WAB',
        'WBA', 'WMT', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST',
        'WDC', 'WRK', 'WY', 'WHR', 'WMB', #'WTW',
         'GWW', 'WYNN', 'XEL',
        'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

# we test the typical time it takes
# to solve and simulate an SPO policy,
# in the same way as we did in Figure
# 7.8 of the book

# NOTE: the first time you run this, it will 
# compute and cache the risk model terms (for
# each day). The second time you run you should
# see faster runtime.

# changing these may have some effect
# on the solver time, but small
GAMMA_RISK = 1.
GAMMA_TRADE = 1.
GAMMA_HOLD = 1. 

# the solve time grows linearly
# with this. 15 is the same number
# we had in the book examples
NUM_RISK_FACTORS = 15

# if you change this to 2 (quadratic model)
# the resulting problem is a QP and can be
# solved faster
TCOST_EXPONENT = 1.5

# you can add any constraint or objective
# term to see how it affects execution time
policy = cvx.SinglePeriodOptimization(
    objective = cvx.ReturnsForecast() 
        - GAMMA_RISK * cvx.FactorModelCovariance(num_factors=NUM_RISK_FACTORS)
        - GAMMA_TRADE * cvx.StocksTransactionCost(exponent=TCOST_EXPONENT)
        - GAMMA_HOLD * cvx.StocksHoldingCost(),
    constraints = [
        cvx.LeverageLimit(3),
    ],
    
    # You can select any CVXPY
    # solver here to see how it
    # affects performance of your
    # particular problem. This one
    # is the default for this type
    # of problems.
    solver='ECOS',
    
    # this is a CVXPY compilation flag, it is 
    # recommended for large optimization problems
    # (like this one) but not for small ones
    ignore_dpp=True,
    
    # you can add any other cvxpy.Problem.solve option
    # here, see https://www.cvxpy.org/tutorial/advanced/index.html
)

# this downloads data for all the sp500
simulator = cvx.StockMarketSimulator(SP500)

# execution and timing, 5 years backtest
s = time.time()
result = simulator.backtest(policy, start_time=pd.Timestamp.today() - pd.Timedelta(f'{365*5}d'))
print('BACKTEST TOOK', time.time() - s)
print('SIMULATOR + POLICY TIMES', result.simulator_times.sum() + result.policy_times.sum())
print('AVERAGE TIME PER ITERATION', result.simulator_times.mean() + result.policy_times.mean())

# plot
result.policy_times.plot(label='policy times')
result.simulator_times.plot(label='simulator times')
plt.legend()
plt.show()

