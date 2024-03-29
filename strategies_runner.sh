#! /usr/bin/env bash
# 
# We run this in a cron job, from the root of repo in the development environment:
# - at 8:30 London time with arguments ftse100_daily
# - at 10:00 New York time with arguments dow30_daily ndx100_daily sp500_daily
#
# These are about 30 minutes after each market opens, so we give time to the
# data provider to update their open prices. If we had a real-time data
# provider we could run this exactly at market open.
#
# Since the open prices by the data provider are in any case not
# finalized, we also re-calculate the holdings of the previous day and commit
# their changes to git too; **we do not change** the weights that we calculated
# the previous day (as certified by git), so there is still no look-ahead.

date=$(date '+%Y-%m-%d')

git pull
for strat in "$@"; do
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
