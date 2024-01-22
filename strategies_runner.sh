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

date=$(date '+%Y-%m-%d')

for strat in dow30_daily ndx100_daily sp500_daily; do
    retry_counter=0
    until env/bin/python -m examples.strategies."$strat" strategy &>> examples/strategies/"$strat".log
        do
            if [ $retry_counter -gt 10 ]; then
                break
            fi
            sleep 10
            ((retry_counter++))
        done
    git add examples/strategies/"$strat"*.json
    git commit -m '[auto commit] '"$strat"' reconciliation & execution on '"$date"
done
git push
