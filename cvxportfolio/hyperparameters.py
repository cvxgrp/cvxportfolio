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
"""This module defines hyperparameter objects.
These are used currently as symbolic multipliers
of cost terms in Single and Multi Period Optimization policies
and can be iterated (and optimized over) automatically.
"""


class HyperParameter:
    """Base Hyper Parameter class."""


class RangeHyperParameter(HyperParameter):
    """Range Hyper Parameter.

    This is not meant to be used directly, look at
    its subclasses for ones that you can use.
    """

    def __init__(self, values):
        self.values = values


class GammaRisk(RangeHyperParameter):
    """Multiplier of a risk term."""

    def __init__():
        super().__init__([.5, 1., 2., 5., 10.])


class GammaTrade(RangeHyperParameter):
    """Multiplier of a transaction cost term."""

    def __init__():
        super().__init__([0., .1, .2, .5, 1., 2., 5., 10.])


class GammaHold(RangeHyperParameter):
    """Multiplier of a holding cost term."""

    def __init__():
        super().__init__([0., .1, .2, .5, 1., 2., 5., 10.])
