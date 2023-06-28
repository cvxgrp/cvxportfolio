import cvxportfolio as cvx
import matplotlib.pyplot as plt

## This is an example of a very large backtest, ~600 names and ~6000 days
## with multi period optimization. It shows that all parts of the system scale
## to such usecases.

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
        'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CTVA', 'CSGP',
        'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI',
        'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS',
        'DISH', 'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE',
        'DUK', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA',
        'ELV', 'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX',
        'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC',
        'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT',
        'FDX', 'FITB', 'FSLR', 'FE', 'FIS', 'FISV', 'FLT', 'FMC', 'F',
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
        
        
NDX100 = ["AMZN", "AAPL", "MSFT", "GOOGL", "TSLA", "GM", 
          'NKE', 'MCD', 'GE', 'CVX', 
          'XOM', 'MMM', 'UNH', 'HD', 'WMT', 'ORCL', 'INTC', 'JPM', 'BLK', 'BA', 'NVDA', 
          'F', 'GS', 'AMD', 'CSCO', 'KO', 'HON', 'DIS', 
          'V', 'ADBE', 'AMGN', 'CAT', 'BA', 'HON', 'JNJ', 'AXP', 'PG', 'JPM', 
          'IBM', 'MRK', 'MMM', 'VZ', 'WBA', 'INTC', 'PEP', 'AVGO',
          'COST', 'TMUS', 'CMCSA', 'TXN', 'NFLX', 'SBUX', 'GILD', 'ISRG', 'MDLZ', 'BKNG', 'AMAT', 
          'ADI', 'ADP', 'VRTX', 'REGN', 'PYPL',  'FISV', 'LRCX', 'PYPL',
          'MU', 'CSX', 'MELI', 'MNST', 'ATVI', 'PANW', 'ORLY', 'ASML', 'SNPS', 
          'CDNS', 'MAR', 'KLAC', 'FTNT', 'CHTR', 'CHTR', 'MRNA', 
          'KHC', 'CTAS', 'AEP', 'DXCM', 'LULU', 'KDP', 
          'AZN', 'BIIB', 'ABNB', 'NXPI', 'ADSK', 'EXC', 'MCHP', 'IDXX', 'CPRT', 'PAYX',
          'PCAR', 'XEL', 'PDD', 'WDAY', 'SGEN', 'ROST', 'DLTR', 'RA', #'MRVL',
          'ODFL', 'VRSK', 'ILMN', 'CTSH', 'FAST', 'CSGP', 'WBD', 'GFS', 'CRWD', 
          'BKR', 'WBA', 'CEG', 'ANSS', 'DDOG', 'EBAY', 'FANG','ENPH', 'ALGN', 'TEAM',
          'ZS','JD', 'ZM','SIRI','LCID', 'RIVN', 'SGEN', 'MDLZ', 'NFLX', 'GOOGL', 'DXCM']


objective = cvx.ReturnsForecast() -.05 * cvx.ReturnsForecastError() \
     - 5 * (cvx.FactorModelCovariance(num_factors=50) + 0.1 * cvx.RiskForecastError()) \
     - cvx.StocksTransactionCost(exponent=2)  - cvx.StocksHoldingCost()

constraints = [cvx.LeverageLimit(3)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=3, ignore_dpp=True)
 
universe = sorted(set(SP500 + NDX100))
simulator = cvx.StockMarketSimulator(universe)

result = simulator.backtest(policy, start_time='2000-01-01', initial_value=1E9)

# print result backtest statistics
print(result)

# plot value and weights of the portfolio in time
result.plot() 
