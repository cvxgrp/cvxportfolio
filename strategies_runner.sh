#! /usr/bin/env bash
#
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
#
#
# 
# We run this in a cron job, from the root of repository in the development
# environment (master branch):
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
