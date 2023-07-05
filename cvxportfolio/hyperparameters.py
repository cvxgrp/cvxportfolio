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

GAMMA_RISK_RANGE = [.5, 1., 2., 5., 10.]
GAMMA_COST_RANGE = [0., .1, .2, .5, 1., 2., 5., 10.]

import copy

import numpy as np

class HyperParameter:
    """Base Hyper Parameter class."""
    
    def __mul__(self, other):
        if np.isscalar(other):
            return CombinedHyperParameter([self], [other])
        return NotImplemented
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __div__(self, other):
        if np.isscalar(other):
            return CombinedHyperParameter([self], [1./other])
        return NotImplemented
        
    def __truediv__(self, other):
        return self.__div__(other)
        
    def __add__(self, other):
        if isinstance(other, HyperParameter):
            return CombinedHyperParameter([self, other], [1., 1.])
        return NotImplemented
        
    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        return self + (-other)
        
    def __neg__(self):
        return self * (-1)
        
class CombinedHyperParameter(HyperParameter):
    """Algebraic combination of HyperParameters."""
    
    def __init__(self, hyperparameters, multipliers):
        self.hyperparameters = hyperparameters
        self.multipliers = multipliers
        
    @property
    def current_value(self):
        return sum([h.current_value * m 
            for h,m in zip(self.hyperparameters, self.multipliers)])
    


class RangeHyperParameter(HyperParameter):
    """Range Hyper Parameter.

    This is not meant to be used directly, look at
    its subclasses for ones that you can use.
    """

    def __init__(self, values_range, initial_value):
        if not (initial_value in values_range):
            raise SyntaxError('Initial value must be in the provided range')
        self.values_range = values_range
        self.current_value = initial_value


class GammaRisk(RangeHyperParameter):
    """Multiplier of a risk term."""

    def __init__(self, values_range = GAMMA_RISK_RANGE, initial_value = 1.):
        super().__init__(values_range, initial_value)


class GammaTrade(RangeHyperParameter):
    """Multiplier of a transaction cost term."""

    def __init__(self, values_range = GAMMA_COST_RANGE, initial_value = 1.):
        super().__init__(values_range, initial_value)


class GammaHold(RangeHyperParameter):
    """Multiplier of a holding cost term."""

    def __init__(self, values_range = GAMMA_COST_RANGE, initial_value = 1.):
        super().__init__(values_range, initial_value)
