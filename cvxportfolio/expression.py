"""
Copyright (C) Enzo Busseti 2016-2019 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Code written before September 2016 is copyrighted to 
Stephen Boyd, Enzo Busseti, Steven Diamond, BlackRock Inc.,
and is licensed under the Apache License, Version 2.0 (the "License");
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
