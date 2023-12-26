# Copyright 2023 Enzo Busseti
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
