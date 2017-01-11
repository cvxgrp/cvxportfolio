"""
Copyright 2016 Stephen Boyd, Enzo Busseti, Steven Diamond, Blackrock Inc.

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

from abc import abstractmethod

import cvxpy as cvx
import numpy as np
import pandas as pd

from .costs import BaseCost


def locator(obj, t):
    """Picks last element before t."""
    try:
        return obj.iloc[obj.axes[0].get_loc(t, method='pad')]
    except AttributeError:  # obj not pandas
        return obj


class BaseRiskModel(BaseCost):

    def __init__(self, **kwargs):
        self.gamma_half_life = kwargs.pop('gamma_half_life', np.inf)
        self.gamma = kwargs.pop('gamma')

    def weight_expr(self, t, w_plus, w_bench, z, value):

        expression = self._estimate(t, w_plus, w_bench, z, value)
        return self.gamma * expression

    @abstractmethod
    def _estimate(self, t, w_plus, w_bench, z, value):
        pass

    def weight_expr_ahead(self, t, tau, w_plus, w_bench, z, value):
        """Estimate risk model at time tau in the future, while t is present."""
        if self.gamma_half_life == np.inf:
            gamma_multiplier = 1.
        else:
            decay_factor = 2**(-1/self.gamma_half_life)
            gamma_init = decay_factor**((tau - t).days)  # TODO not dependent on days
            gamma_multiplier = gamma_init*(1 - decay_factor)/(1 - decay_factor)

        return gamma_multiplier * self.weight_expr(t, w_plus, w_bench, z, value)

    def result_data_type(self, portfolio):
        return pd.Series, {}


class FullSigma(BaseRiskModel):
    def __init__(self, Sigma, **kwargs):
        """Sigma is either a matrix or a pd.Panel"""
        self.Sigma = Sigma
        try:
            assert(not pd.isnull(Sigma).values.any())
        except AttributeError:
            assert (not pd.isnull(Sigma).any())
        super(FullSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        try:
            self.expression = cvx.quad_form(wplus - wbench, locator(self.Sigma, t))
        except TypeError:
            self.expression = cvx.quad_form(wplus - wbench, locator(self.Sigma.values, t))
        return self.expression

    def post_optimization_log(self, t):
        getattr(self.logger, self.destination).loc[t] = self.expression.value


class EmpSigma(BaseRiskModel):
    def __init__(self, returns, lookback, **kwargs):
        """returns is dataframe, lookback is int"""
        self.returns = returns
        self.lookback = lookback
        assert(not np.any(pd.isnull(returns)))
        super(EmpSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        idx = self.returns.index.get_loc(t)
        R = self.returns.iloc[idx-1-self.lookback:idx-1]
        # TODO make sure pandas + cvxpy works
        self.expression = cvx.sum_squares(R.values*(wplus - wbench))/self.lookback
        return self.expression

    def post_optimization_log(self, t):
        getattr(self.logger, self.destination).loc[t] = self.expression.value

    def result_data_type(self, portfolio):
        return pd.Series, {}


class SqrtSigma(BaseRiskModel):
    def __init__(self, sigma_sqrt, **kwargs):
        """returns is dataframe, lookback is int"""
        self.sigma_sqrt = sigma_sqrt
        assert(not np.any(pd.isnull(sigma_sqrt)))
        super(SqrtSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        # TODO make sure pandas + cvxpy works
        self.expression = cvx.sum_squares(self.sigma_sqrt.values @ (wplus - wbench))
        return self.expression

    def post_optimization_log(self, t):
        getattr(self.logger, self.destination).loc[t] = self.expression.value

    def result_data_type(self, portfolio):
        return pd.Series, {}


class FactorModelSigma(BaseRiskModel):
    def __init__(self, exposures, factor_Sigma, idiosync, **kwargs):
        """Each is a pd.Panel (or ) or a vector/matrix"""
        self.exposures = exposures
        assert (not exposures.isnull().values.any())
        self.factor_Sigma = factor_Sigma
        assert (not factor_Sigma.isnull().values.any())
        self.idiosync = idiosync
        assert(not idiosync.isnull().values.any())
        super(FactorModelSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        self.expression = cvx.sum_squares(cvx.mul_elemwise(np.sqrt(locator(self.idiosync, t).values),
                                             wplus - wbench)) + \
                             cvx.quad_form(locator(self.exposures, t).values @ (wplus - wbench),
                                                 locator(self.factor_Sigma, t).values)
        return self.expression

    def post_optimization_log(self, t):
        getattr(self.logger, self.destination).loc[t] = self.expression.value

    def result_data_type(self, portfolio):
        return pd.Series, {}


class RobustSigma(BaseRiskModel):
    """Implements covariance forecast error risk."""
    def __init__(self, Sigma, epsilon, **kwargs):
        self.Sigma = Sigma  # pd.Panel or matrix
        self.epsilon = epsilon  # pd.Series or scalar
        super(RobustSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        self.expression = cvx.quad_form(wplus - wbench, locator(self.Sigma, t)) + \
                             locator(self.epsilon, t) * (cvx.abs(wplus - wbench).T * np.diag(locator(self.Sigma, t)))**2

        return self.expression

    def post_optimization_log(self, t):
        getattr(self.logger, self.destination).loc[t] = self.expression.value

    def result_data_type(self, portfolio):
        return pd.Series, {}


class RobustFactorModelSigma(BaseRiskModel):
    """Implements covariance forecast error risk."""
    def __init__(self, exposures, factor_Sigma, idiosync, epsilon, **kwargs):
        """Each is a pd.Panel (or ) or a vector/matrix"""
        self.exposures = exposures
        assert (not exposures.isnull().values.any())
        self.factor_Sigma = factor_Sigma
        assert (not factor_Sigma.isnull().values.any())
        self.idiosync = idiosync
        assert(not idiosync.isnull().values.any())
        self.epsilon = epsilon
        super(RobustFactorModelSigma, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        F = locator(self.exposures, t)
        f = ((wplus - wbench).T * F.T).T
        Sigma_F = locator(self.factor_Sigma, t)
        D = locator(self.idiosync, t)
        self.expression = cvx.sum_squares(cvx.mul_elemwise(np.sqrt(D), wplus - wbench)) + \
                               cvx.quad_form(f, Sigma_F) + \
                               self.epsilon * (cvx.abs(f).T * np.sqrt(np.diag(Sigma_F)))**2

        return self.expression

    def post_optimization_log(self, t):
        getattr(self.logger, self.destination).loc[t] = self.expression.value

    def result_data_type(self, portfolio):
        return pd.Series, {}


class WorstCaseRisk(BaseRiskModel):
    def __init__(self, riskmodels, **kwargs):
        self.riskmodels = riskmodels
        super(WorstCaseRisk, self).__init__(**kwargs)

    def _estimate(self, t, wplus, wbench, z, value):
        self.risks = [risk.weight_expr(t, wplus, wbench) for risk in self.riskmodels]
        return cvx.max_elemwise(*self.risks)

    def post_optimization_log(self, t):
        """Return data to log in the result object."""
        getattr(self.logger, self.destination).loc[t] = [risk.value[0, 0] for risk in self.risks]

    def result_data_type(self, portfolio):
        return pd.DataFrame, {'columns': [model.__class__.__name__ for model in self.riskmodels]}
