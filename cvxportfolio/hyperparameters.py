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

import numpy as np
import copy
GAMMA_RISK_RANGE = [.5, 1., 2., 5., 10.]
GAMMA_COST_RANGE = [0., .1, .2, .5, 1., 2., 5., 10.]


__all__ = ['GammaRisk', 'GammaTrade', 'GammaHold']


class HyperParameter:
    """Base Hyper Parameter class.

    Implements arithmetic operations between hyper parameters.

    You can sum and multiply HPs between themselves and with scalars,
    and divide by a scalar. Arbitrary algebraic combination of these
    operations are supported.
    """

    def __mul__(self, other):
        if np.isscalar(other) or isinstance(other, HyperParameter):
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

    def _collect_hyperparameters(self):
        return [self]


class CombinedHyperParameter(HyperParameter):
    """Algebraic combination of HyperParameters."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def current_value(self):
        return sum([
            (le.current_value if hasattr(le, 'current_value') else le)
            * (ri.current_value if hasattr(ri, 'current_value') else ri)
            for le, ri in zip(self.left, self.right)])

    def _collect_hyperparameters(self):
        """Collect (not combined) hyperparameters."""
        result = []
        for el in self.left + self.right:
            if hasattr(el, '_collect_hyperparameters'):
                result += el._collect_hyperparameters()
        return result
        
    def __repr__(self):
        result = ''
        for le, ri in zip(self.left, self.right):
            result += str(le) + ' * ' + str(ri)
        return result


class RangeHyperParameter(HyperParameter):
    """Range Hyper Parameter.

    This is not meant to be used directly, look at
    its subclasses for ones that you can use.
    """

    def __init__(self, values_range, current_value):
        if not (current_value in values_range):
            raise SyntaxError('Initial value must be in the provided range')
        self.values_range = values_range
        self._index = self.values_range.index(current_value)
    
    @property
    def current_value(self):
        return self.values_range[self._index]
        
    def __repr__(self):
        return self.__class__.__name__ \
            + f'(current_value={self.current_value})'
            #+ f'(values_range={self.values_range}'\
            
    def _increment(self):
        if self._index == len(self.values_range) - 1:
            raise IndexError
        self._index += 1
    
    def _decrement(self):
        if self._index == 0:
            raise IndexError
        self._index -= 1


class GammaRisk(RangeHyperParameter):
    """Multiplier of a risk term."""

    def __init__(self, values_range=GAMMA_RISK_RANGE, current_value=1.):
        super().__init__(values_range, current_value)


class GammaTrade(RangeHyperParameter):
    """Multiplier of a transaction cost term."""

    def __init__(self, values_range=GAMMA_COST_RANGE, current_value=1.):
        super().__init__(values_range, current_value)


class GammaHold(RangeHyperParameter):
    """Multiplier of a holding cost term."""

    def __init__(self, values_range=GAMMA_COST_RANGE, current_value=1.):
        super().__init__(values_range, current_value)
