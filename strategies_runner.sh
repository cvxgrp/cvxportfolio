#! /usr/bin/env bash
# 
# We run this in a cron job, from the root of repo in the development
# environment, at **10am New York time**. So, we give time to the
# data provider to update their open prices. If we had a real-time data
# provider we could run this exactly at market open.
#
# Since the open prices by the data provider are in any case not
# finalized, we also re-calculate the holdings of the previous day and commit
# their changes to git too; **we do not change** the weights that we calculated
# the previous day (as certified by git), so there is still no look-ahead.

env/bin/python -m examples.strategies.dow30_daily strategy &>> examples/strategies/dow30_daily.log
git add examples/strategies/dow30_daily*.json
git commit -m '[auto commit] dow30_daily reconciliation & execution'

env/bin/python -m examples.strategies.ndx100_daily strategy &>> examples/strategies/ndx100_daily.log
git add examples/strategies/ndx100_daily*.json
git commit -m '[auto commit] ndx100_daily reconciliation & execution'

env/bin/python -m examples.strategies.sp500_daily strategy &>> examples/strategies/sp500_daily.log
git add examples/strategies/sp500_daily*.json
git commit -m '[auto commit] sp500_daily reconciliation & execution'

git push
