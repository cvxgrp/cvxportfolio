"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABCMeta, abstractmethod


class Expression(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def weight_expr(self, t, w_plus, z, value):
        """Returns the estimate of cost at time t."""
        pass

    def weight_expr_ahead(self, t, tau, w_plus, z, value):
        """Returns the estimate at time t of cost at time tau.
        """
        return self.weight_expr(t, w_plus, z, value)

    def value_expr(self, t, h_plus, u):
        """Returns the expression at time t, using value representation.

        This should be overridden if the term is used in the simulator.
        """
        return sum(h_plus) * self.weight_expr(t, h_plus / sum(h_plus),
                                              u / sum(u), sum(h_plus))
