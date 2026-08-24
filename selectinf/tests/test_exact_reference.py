import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import norm, uniform
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import statsmodels.api as sm

#from .instance import gaussian_instance
from .MRT_instance_older import MRT_instance
from ..lasso import lasso
from ..Utils.base import selected_targets, selected_targets_WCLS
from ..grid_inference import grid_inference

# from selective-inference.selectinf.algorithms.api import lasso

import warnings
# suppress warnings
warnings.filterwarnings('ignore')

def test_inf(N=300,
             beta_11=4.4,
             randomizer_scale=1.,
             equicorrelated=False,
             CI=True):

    while True:

        inst, const = MRT_instance, lasso.gaussian

        X, Y, beta, A = MRT_instance(N=N, beta_11=beta_11)[:4]

        n, p = X.shape
        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n, 2000)) * Y.std()
        W = 0.7 * np.median(np.abs(X.T.dot(eps)).max(1))


        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        #print("size of selected set ", nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            conv.setup_inference(dispersion=dispersion)


            target_spec = selected_targets_WCLS(conv.loglike,
                                                A,
                                                conv.observed_soln,
                                                K = conv.K,
                                                dispersion= dispersion)

            result_exact = conv.inference(target_spec)


            if CI is False:
                pvals = result_exact['pvalue']
                #print("check pvalue ", pvals)
                return pvals

            else:
                intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
                lci = intervals[:, 0]
                uci = intervals[:, 1]
                coverage = (lci < beta_target) * (uci > beta_target)
                length = uci - lci

                # print("check intervals ", lci, uci)
                return np.mean(coverage), np.mean(length)

#print(test_inf(CI=False))

## Coverage

# nsim = 500
# coverage = []
# length = []
# for i in tqdm(range(nsim)):
#     coverage.append(test_inf(beta_11=4.4,CI=True)[0])
#     length.append(test_inf(beta_11=4.4,CI=True)[1])
#
# print(np.mean(coverage))
# print(np.mean(length))


### Plots of Pivots/p-values

# from statsmodels.distributions.empirical_distribution import ECDF
#
# nsim = 100
# pivots = []
# for i in tqdm(range(nsim)):
#     for j in test_inf(CI=False):
#         pivots.append(j)
#         # pivots.append(test_inf(CI=False)[j])
#
# plt.clf()
# ecdf_pivot = ECDF(np.asarray(pivots))
# grid = np.linspace(0, 1, nsim)
# plt.plot(grid, ecdf_pivot(grid), c='blue', marker='^')
# plt.plot(grid, grid, 'k--')
# plt.show()



### Functions for Naive and Data Splitting


def compare_inf(N=900,
                beta_11 = 4.4,
                randomizer_scale=1.):

    while True:

        inst, const = MRT_instance, lasso.gaussian

        X, Y, beta, A = MRT_instance(N=N, beta_11 = beta_11)[:4]

        n, p = X.shape

        sigma_ = np.std(Y)

        if n > (2 * p):
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        eps = np.random.standard_normal((n, 2000)) * Y.std()
        W = 0.4 * np.median(np.abs(X.T.dot(eps)).max(1))

        conv = const(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * np.sqrt(dispersion))

        signs = conv.fit()
        nonzero = signs != 0
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        #Sel-Inf
        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            conv.setup_inference(dispersion=dispersion)

            # target_spec2 = selected_targets(conv.loglike,
            #                                conv.observed_soln,
            #                                dispersion=dispersion)

            target_spec = selected_targets_WCLS(conv.loglike,
                                                A,
                                                conv.observed_soln,
                                                K = conv.K,
                                                dispersion=dispersion)

            result_exact = conv.inference(target_spec)
            sel_intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
            lci = sel_intervals[:, 0]
            uci = sel_intervals[:, 1]
            coverage1 = (lci < beta_target) * (uci > beta_target)
            length1 = uci - lci
            #print("Sel Inf coverage and length ", np.mean(coverage1), np.mean(length1))
            #print("check selective intervals ", lci, uci)

            # # without sandwiched cov
            # result_exact2 = conv.inference(target_spec2)
            # sel_intervals2 = np.asarray(result_exact2[['lower_confidence', 'upper_confidence']])
            # lci2 = sel_intervals2[:, 0]
            # uci2 = sel_intervals2[:, 1]
            # coverage1w = (lci2 < beta_target) * (uci2 > beta_target)
            # length1w = uci2 - lci2

        #Naive

        # olsn = sm.OLS(Y, X[:, nonzero])
        # olsfit = olsn.fit()
        # naive_intervals = olsfit.conf_int(alpha=0.1, cols=None)
        # lci = naive_intervals[:, 0]
        # uci = naive_intervals[:, 1]
        # coverage2 = (lci < beta_target) * (uci > beta_target)
        # length2 = uci - lci
        #print("Naive coverage and length ", np.mean(coverage2), np.mean(length2))

        # Xf = np.array(A.iloc[:, 2:-1])
        # Xf = Xf[:, nonzero]
        # yf = A['Y']
        # groups = A['id']
        #
        # # fit the GEE model
        # model = sm.GEE(yf, Xf, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
        # result = model.fit(cov_type='robust')
        #
        # GEE_intervals = np.array(result.conf_int(alpha=0.1))
        # lci = GEE_intervals[:, 0]
        # uci = GEE_intervals[:, 1]
        # coverage2 = (lci < beta_target) * (uci > beta_target)
        # length2 = uci - lci

        #yaha se#
        lower, upper = [], []
        ntarget = nonzero.sum()
        for m in range(ntarget):
            observed_target_uni = target_spec.observed_target[m]
            cov_target_uni = np.sqrt(np.diag(target_spec.cov_target))[m]
            level = 0.9
            u = norm.ppf(1 - (1 - level) / 2)
            l = norm.ppf((1 - level) / 2)
            lower.append(l * cov_target_uni + observed_target_uni)
            upper.append(u * cov_target_uni + observed_target_uni)

        lci = np.asarray(lower)
        uci = np.asarray(upper)

        coverage2 = (lci < beta_target) * (uci > beta_target)
        length2 = uci - lci
        #yaha tak#


        #Split
        unique_ids = A['id'].unique()
        shuffled_ids = pd.Series(unique_ids).sample(frac=1, random_state=42019).tolist()

        # Split the IDs into train and test sets
        test_size = 0.5
        train_size = int((1 - test_size) * len(shuffled_ids))
        train_ids = shuffled_ids[:train_size]
        test_ids = shuffled_ids[train_size:]

        # Split the dataframe based on these IDs
        A_train = A[A['id'].isin(train_ids)]
        A_test = A[A['id'].isin(test_ids)]
        X_train = np.array(A_train.iloc[:, 2:p+2])
        X_test = np.array(A_test.iloc[:, 2:p+2])
        Y_train = np.array(A_train.iloc[:, p + 2])
        Y_test = np.array(A_test.iloc[:, p + 2])

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        eps1 = np.random.standard_normal((n_train, 2000)) * Y_train.std()
        W_train = 0.2 *  n_train/(n_train+ n_test) * np.median(np.abs(X_train.T.dot(eps1)).max(1))
        conv2 = const(X_train,
                      Y_train,
                      W_train,
                      ridge_term=0.,
                      randomizer_scale=0.)

        signs2 = conv2.fit()
        nonzero2 = signs2 != 0
        beta_target2 = np.linalg.pinv(X_test[:, nonzero2]).dot(X_test.dot(beta))


        Xf1 = np.array(A_test.iloc[:, 2:-1])
        Xf1 = Xf1[:, nonzero2]
        yf1 = A_test['Y']
        groups = A_test['id']

        ##yaha se##

        # conv3 = const(X_test,
        #               Y_test,
        #               W_test,
        #               ridge_term=0.,
        #               randomizer_scale=randomizer_scale * np.sqrt(dispersion))
        # s = conv3.fit()
        #
        # target_spec3 = selected_targets_WCLS(conv3.loglike,
        #                                      A_test,
        #                                      conv2.observed_soln,
        #                                      K = conv3.K,
        #                                      dispersion = 1)
        #
        # lower, upper = [], []
        # ntarget = nonzero2.sum()
        # for m in range(ntarget):
        #     observed_target_uni = target_spec3.observed_target[m]
        #     cov_target_uni = np.sqrt(np.diag(target_spec3.cov_target))[m]
        #     level = 0.9
        #     u = norm.ppf(1 - (1 - level) / 2)
        #     l = norm.ppf((1 - level) / 2)
        #     lower.append(l * cov_target_uni + observed_target_uni)
        #     upper.append(u * cov_target_uni + observed_target_uni)
        #
        # lci = np.asarray(lower)
        # uci = np.asarray(upper)
        #
        # coverage3 = (lci < beta_target2) * (uci > beta_target2)
        # length3 = uci - lci

        ##yaha tak##

        # fit the GEE model
        model2 = sm.GEE(yf1, Xf1, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
        result_split = model2.fit(cov_type='robust')

        split_intervals = np.array(result_split.conf_int(alpha=0.1))

        # olss = sm.OLS(Y_test, X_test[:, nonzero2])
        # olsfit2 = olss.fit()
        # split_intervals = olsfit2.conf_int(alpha=0.1, cols=None)

        lci = split_intervals[:, 0]
        uci = split_intervals[:, 1]
        coverage3 = (lci < beta_target2) * (uci > beta_target2)
        length3 = uci - lci

        # print("size of selected set ", nonzero.sum())
        #print("Split coverage and length ", np.mean(coverage3), np.mean(length3))

        return coverage1, coverage2, coverage3, length1, length2, length3

# print(compare_inf(300))

# Simulation vary N

# nsim = 500
# bcoverage1_1 = []
# bcoverage1w_1 = []
# bcoverage2_1 = []
# bcoverage3_1 = []
#
# bcoverage1_2 = []
# bcoverage1w_2 = []
# bcoverage2_2 = []
# bcoverage3_2 = []
#
# bcoverage1_3 = []
# bcoverage1w_3 = []
# bcoverage2_3 = []
# bcoverage3_3 = []
#
# blength1_1 = []
# blength1w_1 = []
# blength2_1 = []
# blength3_1 = []
#
# blength1_2 = []
# blength1w_2 = []
# blength2_2 = []
# blength3_2 = []
#
# blength1_3 = []
# blength1w_3 = []
# blength2_3 = []
# blength3_3 = []
#
# for i in tqdm(range(nsim)):
#     coverage1_1, coverage2_1, coverage3_1, length1_1, length2_1, length3_1 = compare_inf(30,  0.04)
#     coverage1_2, coverage2_2, coverage3_2, length1_2, length2_2, length3_2 = compare_inf(90, 0.04)
#     coverage1_3, coverage2_3, coverage3_3, length1_3, length2_3, length3_3 = compare_inf(300,  0.04)
#
#     bcoverage1_1.append(np.mean(coverage1_1))
#     bcoverage2_1.append(np.mean(coverage2_1))
#     bcoverage3_1.append(np.mean(coverage3_1))
#
#     bcoverage1_2.append(np.mean(coverage1_2))
#     bcoverage2_2.append(np.mean(coverage2_2))
#     bcoverage3_2.append(np.mean(coverage3_2))
#
#     bcoverage1_3.append(np.mean(coverage1_3))
#     bcoverage2_3.append(np.mean(coverage2_3))
#     bcoverage3_3.append(np.mean(coverage3_3))
#
#     blength1_1.append(np.mean(length1_1))
#     blength2_1.append(np.mean(length2_1))
#     blength3_1.append(np.mean(length3_1))
#
#     blength1_2.append(np.mean(length1_2))
#     blength2_2.append(np.mean(length2_2))
#     blength3_2.append(np.mean(length3_2))
#
#     blength1_3.append(np.mean(length1_3))
#     blength2_3.append(np.mean(length2_3))
#     blength3_3.append(np.mean(length3_3))
#
#
# Coverage_Data = pd.DataFrame({"Selective Coverage (n=300)": bcoverage1_1,  "Naive Coverage (n=300)": bcoverage2_1, "Data Splitting Coverage (n=300)": bcoverage3_1,
#                               "Selective Coverage (n=900)": bcoverage1_2, "Naive Coverage (n=900)": bcoverage2_2, "Data Splitting Coverage (n=900)": bcoverage3_2,
#                               "Selective Coverage (n=3000)": bcoverage1_3, "Naive Coverage (n=3000)": bcoverage2_3, "Data Splitting Coverage (n=3000)": bcoverage3_3
#                               })
# Length_Data = pd.DataFrame({"Selective CI Lengths (n=300)": blength1_1, "Naive CI Lengths (n=300)": blength2_1, "Data Splitting CI Lengths (n=300)": blength3_1,
#                             "Selective CI Lengths (n=900)": blength1_2, "Naive CI Lengths (n=900)": blength2_2, "Data Splitting CI Lengths (n=900)": blength3_2,
#                             "Selective CI Lengths (n=3000)": blength1_3,"Naive CI Lengths (n=3000)": blength2_3, "Data Splitting CI Lengths (n=3000)": blength3_3})
#
#
# print(Coverage_Data[list(Coverage_Data.columns)].mean())
# print(Length_Data[list(Length_Data.columns)].mean())

# Coverage_Data.to_csv('Coverage_Data.csv')
# Length_Data.to_csv('Length_Data.csv')
#
# print(np.mean(np.asarray(bcoverage1_2)))
# print(np.mean(np.asarray(bcoverage1w_2)))
# print(np.mean(np.asarray(bcoverage2_2)))
# print(np.mean(np.asarray(bcoverage3_2)))
# print(np.mean(np.asarray(blength1_2)))
# print(np.mean(np.asarray(blength1w_2)))
# print(np.mean(np.asarray(blength2_2)))
# print(np.mean(np.asarray(blength3_2)))



#Simulation vary signal

nsim = 500

bcoverage1_1 = []
bcoverage1w_1 = []
bcoverage2_1 = []
bcoverage3_1 = []

bcoverage1_2 = []
bcoverage1w_2 = []
bcoverage2_2 = []
bcoverage3_2 = []

bcoverage1_3 = []
bcoverage1w_3 = []
bcoverage2_3 = []
bcoverage3_3 = []

blength1_1 = []
blength1w_1 = []
blength2_1 = []
blength3_1 = []

blength1_2 = []
blength1w_2 = []
blength2_2 = []
blength3_2 = []

blength1_3 = []
blength1w_3 = []
blength2_3 = []
blength3_3 = []

for i in tqdm(range(nsim)):
    coverage1_1, coverage2_1, coverage3_1, length1_1, length2_1, length3_1 = compare_inf(150, 0.00000000000004)
    coverage1_2,  coverage2_2, coverage3_2, length1_2, length2_2, length3_2 = compare_inf(150,0.00000008 )
    coverage1_3, coverage2_3, coverage3_3, length1_3, length2_3, length3_3 = compare_inf(150,  2.2)

    bcoverage1_1.append(np.mean(coverage1_1))
    bcoverage2_1.append(np.mean(coverage2_1))
    bcoverage3_1.append(np.mean(coverage3_1))

    bcoverage1_2.append(np.mean(coverage1_2))
    bcoverage2_2.append(np.mean(coverage2_2))
    bcoverage3_2.append(np.mean(coverage3_2))

    bcoverage1_3.append(np.mean(coverage1_3))
    bcoverage2_3.append(np.mean(coverage2_3))
    bcoverage3_3.append(np.mean(coverage3_3))

    blength1_1.append(np.mean(length1_1))
    blength2_1.append(np.mean(length2_1))
    blength3_1.append(np.mean(length3_1))

    blength1_2.append(np.mean(length1_2))
    blength2_2.append(np.mean(length2_2))
    blength3_2.append(np.mean(length3_2))

    blength1_3.append(np.mean(length1_3))
    blength2_3.append(np.mean(length2_3))
    blength3_3.append(np.mean(length3_3))



Coverage_Data_s = pd.DataFrame({"Selective Coverage (1)": bcoverage1_1,  "Naive Coverage (1)": bcoverage2_1, "Data Splitting Coverage (1)": bcoverage3_1,
                              "Selective Coverage (2)": bcoverage1_2, "Naive Coverage (2)": bcoverage2_2, "Data Splitting Coverage (2)": bcoverage3_2,
                              "Selective Coverage (3)": bcoverage1_3, "Naive Coverage (3)": bcoverage2_3, "Data Splitting Coverage (3)": bcoverage3_3
                              })
Length_Data_s = pd.DataFrame({"Selective CI Lengths (1)": blength1_1, "Naive CI Lengths (1)": blength2_1, "Data Splitting CI Lengths (1)": blength3_1,
                            "Selective CI Lengths (2)": blength1_2, "Naive CI Lengths (2)": blength2_2, "Data Splitting CI Lengths (2)": blength3_2,
                            "Selective CI Lengths (3)": blength1_3,  "Naive CI Lengths (3)": blength2_3, "Data Splitting CI Lengths (3)": blength3_3})


print(Coverage_Data_s[list(Coverage_Data_s.columns)].mean())
print(Length_Data_s[list(Length_Data_s.columns)].mean())

# Coverage_Data_s.to_csv('Coverage_Data_signal.csv')
# Length_Data_s.to_csv('Length_Data_signal.csv')

Coverage_Data_s.to_csv('Coverage_Data_signal_Lap.csv')
Length_Data_s.to_csv('Length_Data_signal_Lap.csv')


# ON REAL DATA
#
# data = pd.read_csv(r'~/Documents/realdata/stepsdata.csv')
#
#
#
#
# # np.c_[np.ones(15641), X]
#
# names = np.asarray(pd.read_csv(r'~/Documents/realdata/names.csv'))
#
# Y = data['Y']
# X = data.iloc[:,:-2]
# n, p = X.shape
#
# # from sklearn.preprocessing import StandardScaler
# # scaler = StandardScaler()
# # scaler.fit(X)
# # X = scaler.transform(X)
#
# sigma_ = np.std(Y)
#
# if n > (2 * p):
#     dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
# else:
#     dispersion = sigma_ ** 2
#
# eps = np.random.standard_normal((n, 2000)) * Y.std()
# W = 50 * np.median(np.abs(X.T.dot(eps)).max(1))
#
# const = lasso.gaussian
# randomizer_scale = 1.
# conv = const(X,
#              Y,
#              W,
#              ridge_term=0.,
#              randomizer_scale=randomizer_scale * np.sqrt(dispersion))
#
# signs = conv.fit()
# nonzero = signs != 0
# print("size of selected set ", nonzero.sum())
# print("potential moderators ", names[nonzero])
#
# conv.setup_inference(dispersion=dispersion)

# target_spec = selected_targets(conv.loglike,
#                                conv.observed_soln,
#                                dispersion=dispersion)

# target_spec = selected_targets_WCLS(conv.loglike,
#                                     data,
#                                     conv.observed_soln,
#                                     K = conv.K,
#                                     dispersion= 1)

#
# result_exact = conv.inference(target_spec)
#
# intervals = np.asarray(result_exact[['lower_confidence', 'upper_confidence']])
# lci = intervals[:, 0]
# uci = intervals[:, 1]
# print("check intervals ", lci, uci)


#means and sds of selected columns

# print(np.mean(X[:,nonzero], axis=0))
# print(np.std(X[:,nonzero], axis=0))




