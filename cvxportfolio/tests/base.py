# # Copyright 2016 Enzo Busseti, Stephen Boyd, Steven Diamond, BlackRock Inc.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #    http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# import unittest
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import cvxpy as cp
# import cvxportfolio as cvx
#
#
# class BaseTestClass(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         """Load the data and initialize cvxpy vars."""
#         cls.sigma = pd.read_csv(
#             Path(__file__).parent / "sigmas.csv", index_col=0, parse_dates=[0])
#         cls.returns = pd.read_csv(
#             Path(__file__).parent / "returns.csv", index_col=0, parse_dates=[0])
#         cls.volumes = pd.read_csv(
#             Path(__file__).parent / "volumes.csv", index_col=0, parse_dates=[0])
#         cls.w_plus = cp.Variable(cls.returns.shape[1])
#         cls.w_plus_minus_w_bm = cp.Variable(cls.returns.shape[1])
#         cls.z = cp.Variable(cls.returns.shape[1])
#         cls.N = cls.returns.shape[1]
#
#     def boilerplate(self, model):
#         model._recursive_pre_evaluation(
#             universe=self.returns.columns, backtest_times=self.returns.index)
#         return model._compile_to_cvxpy(self.w_plus, self.z, self.w_plus_minus_w_bm)
