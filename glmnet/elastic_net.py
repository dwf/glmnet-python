import numpy as np
from glmnet import elastic_net

class ElasticNet(object):
    """ElasticNet based on GLMNET"""
    def __init__(self, alpha, rho=0.2):
        super(ElasticNet, self).__init__()
        self.alpha = alpha
        self.rho = rho
        self.coef_ = None

    def fit(self, X, y):
        n_lambdas, intercept_, coef_, _, _, _, lambdas, _, jerr \
        = elastic_net(X, y, self.rho, lambd=self.alpha)
        assert jerr == 0
        self.coef_ = coef_
        self.intercept_ = intercept_
        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

def elastic_net_path(X, y, rho):
    """return full path for ElasticNet"""
    n_lambdas, intercepts, coefs, _, _, _, lambdas, _, jerr \
    = elastic_net(X, y, rho)
    return lambdas, coefs, intercepts

def Lasso(alpha):
    """Lasso based on GLMNET"""
    return ElasticNet(alpha, rho=1.0)

def lasso_path(X, y):
    """return full path for Lasso"""
    return elastic_net_path(X, y, rho=1.0)
