#! /usr/bin/env bash
# 
# run this from root of repo, at ~9.50am New York Time

env/bin/python -m examples.strategies.dow30_daily strategy &>> examples/strategies/dow30_daily.log
git add examples/strategies/dow30_daily*.json
git commit -m '[auto commit]: dow30_daily daily run'

env/bin/python -m examples.strategies.ndx100_daily strategy &>> examples/strategies/ndx100_daily.log
git add examples/strategies/ndx100_daily*.json
git commit -m '[auto commit]: ndx100_daily daily run' 

git push