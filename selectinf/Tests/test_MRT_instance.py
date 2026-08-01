import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statsmodels.api as sm

#from .instance import gaussian_instance
from .MRT_instance_older import MRT_instance
from ..lasso import lasso
from ..Utils.base import selected_targets, selected_targets_WCLS
from ..exact_reference import exact_grid_inference

def test_inst(N=900):

    while True:

        inst, const = MRT_instance, lasso.gaussian

        X, Y, beta, A = MRT_instance(N=N)[:4]

        n1, p = X.shape

        n = int(n1/30)

        sigma_ = np.std(Y)

        eps = np.random.standard_normal((n1, 2000)) * Y.std()
        W = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))

        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale= 1)


        signs = conv.fit()
        nonzero = beta != 0

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        # GEE
        Xf = np.array(A.iloc[:, 2:-1])
        Xf = Xf[:, nonzero]
        yf = A['Y']
        groups = A['id']

        # fit the GEE model
        model = sm.GEE(yf, Xf, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
        result = model.fit(cov_type='robust')

        GEE_intervals = np.array(result.conf_int(alpha=0.1))
        lci1 = GEE_intervals[:, 0]
        uci1 = GEE_intervals[:, 1]
        coverage1 = (lci1 < beta_target) * (uci1 > beta_target)
        length1 = uci1 - lci1

        GEE_est = np.array(result.params)

        print(GEE_est)

        robust_covariance = result.cov_robust

        print(np.sqrt(np.diag(robust_covariance)))

        bread = result.cov_naive
        sandwich = np.dot(np.dot(np.linalg.inv(bread), robust_covariance), np.linalg.inv(bread))
        print(sandwich)
        print(np.sqrt(np.diag(bread)))

        # Naive
        # olsn = sm.OLS(Y, X[:, nonzero])
        # olsfit = olsn.fit()
        #
        # scale = olsfit.scale
        #
        # beta_est = olsfit.params
        # cov_est = np.sqrt(np.diag(olsfit.cov_params()))

        # naive_intervals = olsfit.conf_int(alpha=0.1, cols=None)
        # lci1 = naive_intervals[:, 0]
        # uci1 = naive_intervals[:, 1]
        # coverage1 = (lci1 < beta_target) * (uci1 > beta_target)
        # length1 = uci1 - lci1


        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X[:, nonzero].dot(np.linalg.pinv(X[:, nonzero]).dot(Y))) ** 2 / (n1 - p)
        else:
            dispersion = sigma_ ** 2


        target_spec = selected_targets(conv.loglike,
                                       beta,
                                       dispersion=dispersion)

        e1 = target_spec.observed_target
        c1 = target_spec.cov_target

        print(e1)
        print(np.sqrt(np.diag(c1)))


        target_spec2 = selected_targets_WCLS(conv.loglike,
                                            beta,
                                            K = conv.K,
                                            dispersion = 1)

        e2 = target_spec2.observed_target
        c2 = target_spec2.cov_target

        sandwich2 = np.dot(np.dot(np.linalg.inv(c1), c2), np.linalg.inv(c1))
        print(sandwich2)

        print(e2)
        print(np.sqrt(np.diag(robust_covariance))/np.sqrt(np.diag(c2)))

        # np.testing.assert_allclose(beta_est, e2, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True)


        lower, upper = [], []
        ntarget = nonzero.sum()
        for m in range(ntarget):
            observed_target_uni = target_spec.observed_target[m]
            cov_target_uni = np.sqrt(np.diag(target_spec.cov_target))[m]
            level=0.9
            u = norm.ppf(1 - (1-level)/2)
            l = norm.ppf((1-level)/2)
            lower.append(l * cov_target_uni + observed_target_uni)
            upper.append(u * cov_target_uni + observed_target_uni)

        lci2 = np.asarray(lower)
        uci2 = np.asarray(upper)

        coverage2 = (lci2 < beta_target) * (uci2 > beta_target)
        length2 = uci2 - lci2


        return np.mean(coverage2), np.mean(length2), np.mean(coverage1), np.mean(length1)

print(test_inst())

# nsim = 0
# bcoverage = []
# blength = []
# bcoverage1 = []
# blength1 = []
#
# for i in range(nsim):
#     coverage2, length2, coverage1, length1 = test_inst()
#     bcoverage.append(coverage2)
#     blength.append(length2)
#     bcoverage1.append(coverage1)
#     blength1.append(length1)
#
# print(np.mean(bcoverage))
# print(np.mean(bcoverage1))
# print(np.mean(blength))
# print(np.mean(blength1))

def naive_test(N=300,
               p=50,
               trueP = 5):

    while True:

        inst, const = MRT_instance, lasso.gaussian

        X, Y, beta, A = MRT_instance(N=N)

        nonzero = beta != 0

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        #Naive
        olsn = sm.OLS(Y, X[:, nonzero])
        olsfit = olsn.fit()
        naive_intervals = olsfit.conf_int(alpha=0.1, cols=None)
        lci = naive_intervals[:, 0]
        uci = naive_intervals[:, 1]
        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        return np.mean(coverage), np.mean(length)

# nsim = 50
# coverage = []
# length = []
#
# for i in range(nsim):
#     coverage.append(naive_test()[0])
#     length.append(naive_test()[1])
#
# print(np.mean(coverage))
# print(np.mean(length))