# Copyright (C) 2023-2024 Enzo Busseti
#
# This file is part of Cvxportfolio.
#
# Cvxportfolio is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Cvxportfolio is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Cvxportfolio. If not, see <https://www.gnu.org/licenses/>.
"""This module contains up-to-date universes of stock tickers.

If you run it attempts to download updated lists from the relevant
Wikipedia pages and it rewrites itself. Be careful when you run it
and check that the results make sense.

We could also save each universe in a ``json`` file.
"""

# This was generated on 2025-04-27 10:40:05.022120+00:00

SP500 = \
['A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB',
 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT',
 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APTV',
 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC',
 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG', 'BIIB', 'BK', 'BKNG', 'BKR',
 'BLDR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BX', 'BXP', 'C', 'CAG',
 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW',
 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA',
 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST',
 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX',
 'CTAS', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DASH',
 'DAY', 'DD', 'DE', 'DECK', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN',
 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS',
 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG',
 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FOX',
 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN',
 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC',
 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG',
 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY',
 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC',
 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY',
 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE',
 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LII', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT',
 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR',
 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM',
 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC',
 'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD',
 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW',
 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI',
 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW',
 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG',
 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD',
 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PYPL', 'QCOM',
 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST',
 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI',
 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
 'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG',
 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO', 'TMUS', 'TPL',
 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO',
 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS',
 'URI', 'USB', 'V', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX',
 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDAY', 'WDC', 'WEC',
 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WSM', 'WST', 'WTW', 'WY', 'WYNN',
 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS']

NDX100 = \
['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
 'AMZN', 'ANSS', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'AZN', 'BIIB', 'BKNG',
 'BKR', 'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD',
 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DXCM', 'EA', 'EXC',
 'FANG', 'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX',
 'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR',
 'MCHP', 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRVL', 'MSFT', 'MSTR', 'MU',
 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD',
 'PEP', 'PLTR', 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SNPS', 'TEAM',
 'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL',
 'ZS']

DOW30 = \
['AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS',
 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT']

FTSE100 = \
['AAF.L', 'AAL.L', 'ABF.L', 'ADM.L', 'AHT.L', 'ALW.L', 'ANTO.L', 'AUTO.L',
 'AV.L', 'AZN.L', 'BA.L', 'BAB.L', 'BARC.L', 'BATS.L', 'BEZ.L', 'BKG.L',
 'BNZL.L', 'BP.L', 'BT-A.L', 'BTRW.L', 'CCEP.L', 'CCH.L', 'CNA.L', 'CPG.L',
 'CRDA.L', 'CTEC.L', 'DCC.L', 'DGE.L', 'DPLM.L', 'EDV.L', 'ENT.L', 'EXPN.L',
 'EZJ.L', 'FCIT.L', 'FRES.L', 'GAW.L', 'GLEN.L', 'GSK.L', 'HIK.L', 'HLMA.L',
 'HLN.L', 'HSBA.L', 'HSX.L', 'HWDN.L', 'IAG.L', 'ICG.L', 'IHG.L', 'III.L',
 'IMB.L', 'IMI.L', 'INF.L', 'ITRK.L', 'JD.L', 'KGF.L', 'LAND.L', 'LGEN.L',
 'LLOY.L', 'LMP.L', 'LSEG.L', 'MKS.L', 'MNDI.L', 'MNG.L', 'MRO.L', 'NG.L',
 'NWG.L', 'NXT.L', 'PCT.L', 'PHNX.L', 'PRU.L', 'PSH.L', 'PSN.L', 'PSON.L',
 'REL.L', 'RIO.L', 'RKT.L', 'RMV.L', 'RR.L', 'RTO.L', 'SBRY.L', 'SDR.L',
 'SGE.L', 'SGRO.L', 'SHEL.L', 'SMIN.L', 'SMT.L', 'SN.L', 'SPX.L', 'SSE.L',
 'STAN.L', 'STJ.L', 'SVT.L', 'TSCO.L', 'TW.L', 'ULVR.L', 'UTG.L', 'UU.L',
 'VOD.L', 'WEIR.L', 'WPP.L', 'WTB.L']

NIKKEI225 = \
['1332.T', '1605.T', '1721.T', '1801.T', '1802.T', '1803.T', '1808.T',
 '1812.T', '1925.T', '1928.T', '1963.T', '2002.T', '2269.T', '2282.T',
 '2413.T', '2432.T', '2501.T', '2502.T', '2503.T', '2768.T', '2801.T',
 '2802.T', '2871.T', '2914.T', '3086.T', '3092.T', '3099.T', '3289.T',
 '3382.T', '3401.T', '3402.T', '3405.T', '3407.T', '3436.T', '3659.T',
 '3861.T', '4004.T', '4005.T', '4021.T', '4042.T', '4043.T', '4061.T',
 '4063.T', '4151.T', '4183.T', '4188.T', '4208.T', '4307.T', '4324.T',
 '4385.T', '4452.T', '4502.T', '4503.T', '4506.T', '4507.T', '4519.T',
 '4523.T', '4543.T', '4568.T', '4578.T', '4661.T', '4689.T', '4704.T',
 '4751.T', '4755.T', '4901.T', '4902.T', '4911.T', '5019.T', '5020.T',
 '5101.T', '5108.T', '5201.T', '5214.T', '5233.T', '5301.T', '5332.T',
 '5333.T', '5401.T', '5406.T', '5411.T', '5631.T', '5706.T', '5711.T',
 '5713.T', '5714.T', '5801.T', '5802.T', '5803.T', '5831.T', '6098.T',
 '6103.T', '6113.T', '6146.T', '6178.T', '6273.T', '6301.T', '6302.T',
 '6305.T', '6326.T', '6361.T', '6367.T', '6471.T', '6472.T', '6473.T',
 '6479.T', '6501.T', '6503.T', '6504.T', '6506.T', '6526.T', '6532.T',
 '6594.T', '6645.T', '6674.T', '6701.T', '6702.T', '6723.T', '6724.T',
 '6752.T', '6753.T', '6758.T', '6762.T', '6770.T', '6841.T', '6857.T',
 '6861.T', '6902.T', '6920.T', '6952.T', '6954.T', '6971.T', '6976.T',
 '6981.T', '6988.T', '7004.T', '7011.T', '7012.T', '7013.T', '7186.T',
 '7201.T', '7202.T', '7203.T', '7205.T', '7211.T', '7261.T', '7267.T',
 '7269.T', '7270.T', '7272.T', '7453.T', '7731.T', '7733.T', '7735.T',
 '7741.T', '7751.T', '7752.T', '7762.T', '7832.T', '7911.T', '7912.T',
 '7951.T', '7974.T', '8001.T', '8002.T', '8015.T', '8031.T', '8035.T',
 '8053.T', '8058.T', '8233.T', '8252.T', '8253.T', '8267.T', '8304.T',
 '8306.T', '8308.T', '8309.T', '8316.T', '8331.T', '8354.T', '8411.T',
 '8591.T', '8601.T', '8604.T', '8630.T', '8697.T', '8725.T', '8750.T',
 '8766.T', '8795.T', '8801.T', '8802.T', '8804.T', '8830.T', '9001.T',
 '9005.T', '9007.T', '9008.T', '9009.T', '9020.T', '9021.T', '9022.T',
 '9064.T', '9101.T', '9104.T', '9107.T', '9147.T', '9201.T', '9202.T',
 '9432.T', '9433.T', '9434.T', '9501.T', '9502.T', '9503.T', '9531.T',
 '9532.T', '9602.T', '9613.T', '9735.T', '9766.T', '9843.T', '9983.T',
 '9984.T']

if __name__ == '__main__':

    # import json
    from pprint import pprint

    import bs4 as bs
    import pandas as pd
    from curl_cffi import get

    universes = {
        'sp500': {
            'page': "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            'table_number': 0,
            'column_number': 0,
            'ignore_class_bsoup_filtering': True,
        },
        'ndx100': {
            'page': "https://en.wikipedia.org/wiki/Nasdaq-100",
            'table_number': -1,
            'column_number': 1,
        },
        'dow30': {
         'page': "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            'table_number': 0,
            'column_number': 1,
        },
        'ftse100': {
            'page': 'https://en.wikipedia.org/wiki/FTSE_100_Index',
            'table_number': -1,
            'column_number': 1,
            'suffix': '.L',
        },
        'nikkei225': {
            'getter': 'get_nikkei',
            'suffix': '.T',
        }
    }

    def get_column_wikipedia_page(
            page, table_number, column_number,
            ignore_class_bsoup_filtering=False, **kwargs):
        """Get a column as list of strings from a table on wikipedia.

        This is adapted from:

        https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/

        :param page: Wikipedia URL.
        :type page: str
        :param table_number: Which table on the page.
        :type table_number: int
        :param column_number: Which column to extract.
        :type column_number: int
        :param kwargs: Unused arguments.
        :type kwargs: dict

        :returns: Sorted strings of the column.
        :rtype: list
        """
        resp = get(page, impersonate='chrome', timeout=10)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = (
            soup.find_all('table') if ignore_class_bsoup_filtering
            else soup.find_all('table', {'class': 'wikitable sortable'})
                )[table_number]
        column = []
        for row in table.findAll('tr')[1:]:
            element = row.findAll('td')[column_number].text
            column.append(element.strip())
        return sorted(column)

    def get_nikkei():
        """Get nikkei components.

        This is a simple scraping of the official webpage.

        :returns: Nikkei225 components.
        :rtype: list
        """
        url = 'https://indexes.nikkei.co.jp/en/nkave/index/component'
        resp = get(url, impersonate='chrome', timeout=10)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        components = []
        for table in soup.find_all('table'):
            for row in table.findAll('tr')[1:]:
                element = row.findAll('td')[0].text
                components.append(element.strip())
        return sorted(components)

    def adapt_for_yahoo_finance(tickers_list, suffix='', **kwargs):
        """Change tickers to match the spelling of Yahoo Finance.

        :param tickers_list: Tickers from Wikipedia.
        :type tickers_list: list
        :param suffix: Suffix to add to each ticker, default empty string.
        :type suffix: str
        :param kwargs: Unused arguments.
        :type kwargs: dict

        :returns: Adapted tickers.
        :rtype: list
        """

        # this is
        return [el.replace(
                    '.', '-').rstrip('-') + suffix for el in tickers_list]

    # re-write this file

    with open(__loader__.path, 'r', encoding='utf-8') as f:
        this_file_content = f.readlines()

    code_index = this_file_content.index("if __name__ == '__main__':\n")

    with open(__loader__.path, 'w', encoding='utf-8') as f:

        # header
        f.writelines(this_file_content[:16])

        # docstring
        f.write('"""' + __doc__ + '"""\n')

        # timestamp
        f.write("\n# This was generated on " + str(pd.Timestamp.utcnow()) + "\n")

        # universes lists
        for key, value in universes.items():

            tickers = adapt_for_yahoo_finance(
                globals()[value['getter']]() if 'getter' in value
                else get_column_wikipedia_page(**value), **value)
            f.write(f'\n{key.upper()} = \\\n')
            pprint(tickers, compact=True, width=79, stream=f)

            # # also save in json
            # with open(key + '.json', 'w', encoding='utf-8') as f1:
            #     json.dump(tickers, f1)

        # copy everything in the if __name__ == '__main__' clause
        f.write('\n')
        f.writelines(this_file_content[code_index:])
