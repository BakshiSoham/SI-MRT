import numpy as np

import regreg.api as rr
import regreg.affine as ra

import statsmodels.api as sm


def restricted_estimator(loss, active, solve_args={'min_its': 50, 'tol': 1.e-10}):
    """
    Fit a restricted model using only columns `active`.
    Parameters
    ---------
    Mest_loss : objective function
        A GLM loss.
    active : ndarray
        Which columns to use.
    solve_args : dict
        Passed to `solve`.
    Returns
    -------
    soln : ndarray
        Solution to restricted problem.
    """
    X, Y = loss.data

    if not loss._is_transform and hasattr(loss, 'saturated_loss'):  # M_est is a glm
        X_restricted = X[:, active]
        loss_restricted = rr.affine_smooth(loss.saturated_loss, X_restricted)
    else:
        I_restricted = ra.selector(active, ra.astransform(X).input_shape[0], ra.identity((active.sum(),)))
        loss_restricted = rr.affine_smooth(loss, I_restricted.T)
    beta_E = loss_restricted.solve(**solve_args)

    return beta_E


from typing import NamedTuple
class TargetSpec(NamedTuple):
    observed_target: np.ndarray
    cov_target: np.ndarray
    regress_target_score: np.ndarray
    alternatives: list

def selected_targets(loglike,
                     solution,
                     features=None,
                     sign_info={},
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 100}):

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    linpred = X[:, features].dot(observed_target)

    Hfeat = _compute_hessian(loglike,
                             solution,
                             features)[1]
    Qfeat = Hfeat[features]
    _score_linear = -Hfeat

    cov_target = np.linalg.inv(Qfeat)
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                observed_target.shape[0])

    regress_target_score = np.zeros((cov_target.shape[0], p))
    regress_target_score[:, features] = cov_target

    return TargetSpec(observed_target,
                      cov_target,
                      regress_target_score,
                      alternatives)


def selected_targets_WCLS(loglike,
                          MRT,
                          solution,
                          K,
                          features=None,
                          sign_info={},
                          dispersion=None,
                          solve_args={'tol': 1.e-12, 'min_its': 100}):
    """
    Target specification for WCLS selective inference.

    Key difference from selected_targets:
        cov_target comes from GEE robust (sandwich) covariance,
        which is ALREADY the full variance estimate of the target.
        It should NOT be multiplied by dispersion.

    The regress_target_score uses the Hessian-based Qinv,
    which maps score_E → target correctly because the
    Hessian weights (from loglike coef=1/sigma^2) are
    already built into Qinv.
    """

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    linpred = X[:, features].dot(observed_target)

    # GEE robust covariance — already the full sandwich estimate
    Xf = X[:, features]
    yf = MRT['Y']
    groups = MRT['id']

    model = sm.GEE(yf, Xf, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence())
    result = model.fit(cov_type='robust')
    cov_target = result.cov_robust

    Hfeat = _compute_hessian(loglike,
                             solution,
                             features)[1]

    # Qfeat = H_{E,E} = (1/sigma^2) X_E^T X_E  (for Gaussian loglike)
    Qfeat = Hfeat[features]
    Qinv = np.linalg.inv(Qfeat)

    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    # regress_target_score[:, E] = Qinv = sigma^2 (X_E^T X_E)^{-1}
    # This correctly maps score_E to the target:
    #   Qinv @ score_E = sigma^2 (X_E^T X_E)^{-1} @ (1/sigma^2) X_E^T r = (X_E^T X_E)^{-1} X_E^T r = beta_hat_E
    regress_target_score = np.zeros((Qinv.shape[0], p))
    regress_target_score[:, features] = Qinv

    # FIX: cov_target from GEE is already the full sandwich variance.
    # Do NOT multiply by dispersion (old code had cov_target * dispersion).
    return TargetSpec(observed_target,
                      cov_target,
                      regress_target_score,
                      alternatives)


def selected_targets_logistic(loglike,
                          solution,
                          K,
                          features=None,
                          sign_info={},
                          dispersion=None,
                          solve_args={'tol': 1.e-12, 'min_its': 100}):

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    linpred = X[:, features].dot(observed_target)

    Hfeat = _compute_hessian(loglike,
                             solution,
                             features)[1]

    # Qfeat = H_{E,E}
    Qfeat = Hfeat[features]
    Qinv = np.linalg.inv(Qfeat)
    _score_linear = -Hfeat

    # Kfeat = K_{E,E}
    # Kfeat = K
    Kfeat = K[:, features]
    Kfeat = Kfeat[features]

    # cov_target = np.linalg.inv(Qfeat)
    # cov_target = Qinv @ Kfeat @ Qinv
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                observed_target.shape[0])

    # if n > (2 * p):
    #     dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    # else:
    #     dispersion = sigma_ ** 2

    regress_target_score = np.zeros((Qinv.shape[0], p))
    regress_target_score[:, features] = Qinv


    return TargetSpec(observed_target,
                      cov_target * dispersion,
                      regress_target_score,
                      alternatives)



def _compute_hessian(loglike,
                     beta_bar,
                     *bool_indices):
    X, y = loglike.data
    linpred = X.dot(beta_bar)
    n = linpred.shape[0]

    if hasattr(loglike.saturated_loss, "hessian"):  # a GLM -- all we need is W
        W = loglike.saturated_loss.hessian(linpred)
        parts = [np.dot(X.T, X[:, bool_idx] * W[:, None]) for bool_idx in bool_indices]
        _hessian = np.dot(X.T, X * W[:, None])  # CAREFUL -- this will be big
    elif hasattr(loglike.saturated_loss, "hessian_mult"):
        parts = []
        for bool_idx in bool_indices:
            _right = np.zeros((n, bool_idx.sum()))
            for i, j in enumerate(np.nonzero(bool_idx)[0]):
                _right[:, i] = loglike.saturated_loss.hessian_mult(linpred,
                                                                   X[:, j],
                                                                   case_weights=loglike.saturated_loss.case_weights)
            parts.append(X.T.dot(_right))
        _hessian = np.zeros_like(X)
        for i in range(X.shape[1]):
            _hessian[:, i] = loglike.saturated_loss.hessian_mult(linpred,
                                                                 X[:, i],
                                                                 case_weights=loglike.saturated_loss.case_weights)
        _hessian = X.T.dot(_hessian)
    else:
        raise ValueError('saturated_loss has no hessian or hessian_mult method')

    if bool_indices:
        return (_hessian,) + tuple(parts)
    else:
        return _hessian

#
# def _compute_gradient(loglike,
#                      beta_bar,
#                      *bool_indices):
#
#     X, y = loglike.data
#     linpred = X.dot(beta_bar)
#     n = linpred.shape[0]
#
#     # W = loglike.smooth_objective(beta_bar, 'grad')
#     #
#     # print(W.shape)                        #shouldn't its shape be (n,)
#     #
#     # parts = [np.dot(np.dot(X[:, bool_idx].T, np.diagflat(W[:, None]**p)), X[:, bool_idx])  for bool_idx in bool_indices]
#     # _gradient = np.dot(X.T, W[:, None])
#
#
#     if hasattr(loglike, "gradient"): #loglike.saturated_loss.gradient doesnt work
#         W = loglike.gradient(beta_bar)
#         print(W.shape)
#         # Active idx, then unpenalized idx
#         parts = [np.dot(np.dot(X[:, bool_idx].T, np.diagflat(W[:, None]**2)), X[:, bool_idx]) for bool_idx in bool_indices]
#         _gradient = np.dot(X.T, W[:, None])
#     else:
#         raise ValueError('not a smooth loglikelihood')
#
#
#     if bool_indices:
#         # Returns _gradient, _gradient_active, _gradient_unpen
#         return (_gradient,) + tuple(parts)
#     else:
#         return _gradient
#


def _pearsonX2(y,
               linpred,
               loglike,
               df_fit):
    W = loglike.saturated_loss.hessian(linpred)
    n = y.shape[0]
    resid = y - loglike.saturated_loss.mean_function(linpred)
    return (resid ** 2 / W).sum() / (n - df_fit)


def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):
    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(U1)))
    U5 = U1.T.dot(QS.M1.dot(QS.opt_linear))

    return U1, U2, U3, U4, U5