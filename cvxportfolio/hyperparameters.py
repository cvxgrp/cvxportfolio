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

These are used currently as symbolic multipliers of cost terms in Single
and Multi Period Optimization policies and can be iterated (and
optimized over) automatically.
"""

from numbers import Number

import numpy as np
import pandas as pd

# GAMMA_RISK_RANGE = [.5, 1., 2., 5., 10.]
# GAMMA_COST_RANGE = [0., .1, .2, .5, 1., 2., 5., 10.]


__all__ = ['Gamma', 'RangeHyperParameter']


def _resolve_hyperpar(possible_hyperpar):
    """Return current value if input is hyper-parameter, or input itself."""
    if isinstance(possible_hyperpar, HyperParameter):
        return possible_hyperpar.current_value
    return possible_hyperpar

class HyperParameter:
    """Base Hyper Parameter class.

    Implements arithmetic operations between hyper parameters.

    You can sum and multiply HPs between themselves and with scalars,
    and divide by a scalar. Arbitrary algebraic combination of these
    operations are supported.
    """

    def __mul__(self, other):
        if np.isscalar(other) or isinstance(other, HyperParameter) \
                or isinstance(other, pd.Timedelta):
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

    def collect_hyperparameters(self):
        """Collect hyper-parameters.

        :returns: Hyperparameters found (self).
        :rtype: list
        """
        return [self]

    @property
    def current_value(self):
        """Current value of the hyper-parameter.

        :returns: Current value.
        :rtype: int, float, pd.Timedelta
        """
        raise NotImplementedError # pragma: no cover

    def __repr__(self):
        return self.__class__.__name__\
            + f'(current_value={self.current_value})'

class CombinedHyperParameter(HyperParameter):
    """Algebraic combination of HyperParameters."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def current_value(self):
        """Current value of the hyper-parameter.

        :returns: Current value.
        :rtype: int or float
        """
        # we unroll the sum() to support non-numeric (Timedeltas, ...)
        summands = list(
            (le.current_value if hasattr(le, 'current_value') else le)
            * (ri.current_value if hasattr(ri, 'current_value') else ri)
            for le, ri in zip(self.left, self.right))
        result = summands[0]
        for el in summands[1:]:
            result += el
        return result

    def collect_hyperparameters(self):
        """Collect (not combined) hyper-parameters.

        :returns: List of found hyper-parameters.
        :rtype: list
        """
        result = []
        for el in self.left + self.right:
            if hasattr(el, 'collect_hyperparameters'):
                result += el.collect_hyperparameters()
        return result

    def __repr__(self):
        """Pretty print.

        .. note::
            This could be improved a lot, it does pretty-printing of algebraic
            expressions in general. It's not perfect but readable.
        """

        # TODO this gives wrong string repr with nested expressions like
        # ``-(cvx.Gamma() * -3 * (cvx.Gamma() - cvx.Gamma()))``
        # internal algebra is correct, though

        def _minus_repr(obj):
            rawrepr = str(obj).lstrip()
            if rawrepr[0] == '-':
                return ' + ' + rawrepr[1:].lstrip()
            if rawrepr[0] == '+':
                return ' - ' + rawrepr[1:].lstrip() # pragma: no cover
            return ' - ' + rawrepr

        def _plus_repr(obj):
            rawrepr = str(obj).lstrip()
            if rawrepr[0] == '-':
                return ' - ' + rawrepr[1:].lstrip()
            if rawrepr[0] == '+':
                return ' + ' + rawrepr[1:].lstrip()
            return ' + ' + rawrepr

        result = ''

        def _with_possible_number(num, other):
            if num == -1.:
                return _minus_repr(other)
            if num == 1.:
                return _plus_repr(other)
            return str(num).rstrip() + ' * ' + str(other).lstrip()

        for left, right in zip(self.left, self.right):
            if isinstance(left, Number):
                result += _with_possible_number(left, right) # pragma: no cover
            else:
                result += _with_possible_number(right, left)

        return result.strip()

class RangeHyperParameter(HyperParameter):
    """Range Hyper Parameter.

    This is not meant to be used directly, look at its subclasses for
    ones that you can use.
    """

    def __init__(self, values_range, current_value):
        if not current_value in values_range:
            raise SyntaxError('Initial value must be in the provided range')
        self.values_range = list(values_range)
        self._index = self.values_range.index(current_value)

    @property
    def current_value(self):
        """Current value of the hyper-parameter.

        :returns: Current value.
        :rtype: int or float
        """
        return self.values_range[self._index]

    def _increment(self):
        if self._index == len(self.values_range) - 1:
            raise IndexError
        self._index += 1

    def _decrement(self):
        if self._index == 0:
            raise IndexError
        self._index -= 1


class Gamma(HyperParameter):
    """Generic multiplier."""

    def __init__(self, initial_value = 1., increment = 1.1):
        self._initial_value = initial_value
        self._spacing = increment
        self._index = 0

    @property
    def current_value(self):
        """Current value of the hyper-parameter.

        :returns: Current value.
        :rtype: int or float
        """
        return self._initial_value * (self._spacing ** self._index)

    def _increment(self):
        # if self._index == len(self.values_range) - 1:
        #     raise IndexError
        self._index += 1

    def _decrement(self):
        # if self._index == 0:
        #     raise IndexError
        self._index -= 1
#
# class GammaRisk(RangeHyperParameter):
#     """Multiplier of a risk term."""
#
#     def __init__(self, values_range=GAMMA_RISK_RANGE, current_value=1.):
#         super().__init__(values_range, current_value)
#
#
# class GammaTrade(RangeHyperParameter):
#     """Multiplier of a transaction cost term."""
#
#     def __init__(self, values_range=GAMMA_COST_RANGE, current_value=1.):
#         super().__init__(values_range, current_value)
#
#
# class GammaHold(RangeHyperParameter):
#     """Multiplier of a holding cost term."""
#
#     def __init__(self, values_range=GAMMA_COST_RANGE, current_value=1.):
#         super().__init__(values_range, current_value)
