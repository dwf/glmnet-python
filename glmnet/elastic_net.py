import numpy as np
from glmnet import elastic_net

class ElasticNet(object):
    """ElasticNet based on GLMNET"""
    def __init__(self, alpha, rho=0.2):
        super(ElasticNet, self).__init__()
        self.alpha = alpha
        self.rho = rho
        self.coef_ = None
        self.rsquared_ = None

    def fit(self, X, y):
        n_lambdas, intercept_, coef_, _, _, rsquared_, lambdas, _, jerr \
        = elastic_net(X, y, self.rho, lambdas=[self.alpha])
        # elastic_net will fire exception instead
        # assert jerr == 0
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.rsquared_ = rsquared_
        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def __str__(self):
        n_non_zeros = (np.abs(self.coef_) != 0).sum()
        return ("%s with %d non-zero coefficients (%.2f%%)\n" + \
                " * Intercept = %.7f, Lambda = %.7f\n" + \
                " * Training r^2: %.4f") % \
                (self.__class__.__name__, n_non_zeros,
                 n_non_zeros / float(len(self.coef_)) * 100,
                 self.intercept_[0], self.alpha, self.rsquared_[0])


def elastic_net_path(X, y, rho, **kwargs):
    """return full path for ElasticNet"""
    n_lambdas, intercepts, coefs, _, _, _, lambdas, _, jerr \
    = elastic_net(X, y, rho, **kwargs)
    return lambdas, coefs, intercepts

def Lasso(alpha):
    """Lasso based on GLMNET"""
    return ElasticNet(alpha, rho=1.0)

def lasso_path(X, y, **kwargs):
    """return full path for Lasso"""
    return elastic_net_path(X, y, rho=1.0, **kwargs)
