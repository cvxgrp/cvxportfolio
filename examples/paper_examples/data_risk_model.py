# Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This is the restored version of the IPython notebook
used to download data for the original 2016 examples:

https://github.com/cvxgrp/cvxportfolio/blob/0.0.X/examples/DataEstimatesRiskModel.ipynb

The main reasons why the original can't be used any more:

- the data source we used, "Quandl wiki" historical stock market data,
    is defunct. Fortunately we saved the data in this repo, which is the
    reason why it's quite big, so we can reproduce the original results.
- Pandas dropped its Panel data structure (after which it is named!); we
    used it to store the risk models. We now use multi-indexed Dataframes.

The data files that we use from the notebook above in 2017 are:

- returns.csv.gz: Historical close-to-close total returns; note that today
    :class:`cvxportfolio.DownloadedMarketData` instead computes
    historical open-to-open total returns, matching the model
    from the paper.
- volumes.csv.gz: historical market volumes in USD, computed as volumes
    in number of shares times the adjusted close price.

These are for a selection of stocks from the components of SP500 in 2016. 
The data cleaning logic is shown in the notebook. We don't use the
following files from that notebook:

- sigmas.csv.gz: now computed automatically by 
    :class:`cvxportfolio.TransactionCost`; note that intraday returns were 
    used in the notebook, while now we use open-to-open returns. This may 
    cause some noticeable difference in the market impact term of the
    transaction cost estimates. In practice, the market impact term is either 
    negligible for small to medium investors, or needs to be tuned for the
    given assets (using historical realized costs) in case of a large investor.
- prices.csv.gz: unused
- sigma_estimate.csv.gz: now computed automatically by 
    :class:`cvxportfolio.TransactionCost`
- volume_estimate.csv.gz: now computed automatically by 
    :class:`cvxportfolio.TransactionCost`
"""
