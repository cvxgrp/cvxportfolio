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
"""This module contains examples of simple strategies built with Cvxportfolio.

These are executed every day (or less often, depending on the strategy) at
around the open time of the market (currently, only US). The results are
then added to the data files that accompany each strategy and committed to
the git repository. So, we time-stamp each execution and thus certify
each strategy's performance; there is no look-ahead.

If you like some strategy's returns, as they are certified by this procedure,
you're welcome to subscribe to the updates and model your portfolio after our
target allocations (or, directly copy our trades). This strenghtens the future
returns of the strategy, as a self-fulfilling prophecy; You can also run them
on your computer.
"""
