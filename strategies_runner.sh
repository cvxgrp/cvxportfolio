#! /usr/bin/env bash
# 
# run this from root of repo, at ~9.35am New York Time
#
# note that the open prices provided by the data provider are not finalized,
# they may change on the next day. we handle this case by re-calculating the
# holdings of yesterday as well, and committing the change to git too; 
# **we do not change** the weights that we calculated yesterday, so there
# is still no look-ahead

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
