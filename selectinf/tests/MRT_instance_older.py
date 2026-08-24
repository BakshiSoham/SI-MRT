import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import statsmodels.api as sm

# OLDER GENERATION
# # Define constants
# T = 30
# N = 150
# P = 50
# trueP = 5
# sigma_residual = 1.5
# sigma_randint = 1.5
# rho = 0.5
# txt_intercept = -0.2
# beta_logit = np.concatenate(([-1], 0.8 * np.ones(P) / P))
# beta_11 = 4.4
# theta1 = 0.8
#
# # Generate AR(1) process
# def arima_sim(rho, n, sd=1):
#     ar = np.array([1, -rho])
#     ma = np.array([1])
#     ARMA = ArmaProcess(ar, ma)
#     return ARMA.generate_sample(nsample=n, scale=sd)
#
#
# # Generate individual data
# def generate_individual(id=1, beta_11=beta_11, rho=rho):
#     all_states = np.column_stack([arima_sim(rho, T) for _ in range(P)])
#
#     all_actions = np.zeros(T)
#     all_probabilities = np.zeros(T)
#
#     current_action = 0
#     for t in range(T):
#         current_states = all_states[t, :]
#         current_action = all_actions[t - 1] if t != 0 else 0
#         prob_action = 1 / (1 + np.exp(-np.dot(np.concatenate(([current_action], current_states)), beta_logit)))
#         current_action = np.random.binomial(n=1, p=prob_action)
#         all_probabilities[t] = prob_action
#         all_actions[t] = current_action
#
#     treatment_effect = np.sum(all_states[:, :trueP], axis=1) * beta_11 / trueP + txt_intercept
#     main_effect = theta1 * np.sum(all_states, axis=1)
#     meanY = main_effect + (all_actions - all_probabilities) * treatment_effect
#     errorY = arima_sim(rho, T, sd=sigma_residual)
#     txterrorY = arima_sim(rho, T, sd=sigma_residual) + np.random.normal(0, sigma_randint, T)
#     # txterrorY = arima_sim(rho, T, sd=sigma_residual) + np.random.laplace(0, sigma_randint, T)
#     # txterrorY = arima_sim(rho, T, sd=sigma_residual) + np.random.exponential(sigma_randint, T)
#     # obsY = meanY + errorY + txterrorY * (all_actions - all_probabilities)
#     obsY = meanY + txterrorY * (all_actions - all_probabilities)
#
#     df_individual = pd.DataFrame({
#         "id": id,
#         "decision_point": np.arange(1, T + 1),
#         **{f"state{i}": all_states[:, i] for i in range(P)},
#         "prob": all_probabilities,
#         "action": all_actions,
#         "outcome": obsY
#     })
#     return df_individual
#
# def MRT_instance(N=N, beta_11=beta_11, rho=rho):
#
#     individual_data_frames = []
#
#     # Generate individual data and collect them in the list
#     for n in range(1, N + 1):
#           fake_individual = generate_individual(n, beta_11, rho)  # Call the generate_individual function
#           individual_data_frames.append(fake_individual)
#
#     MRT_data = pd.concat(individual_data_frames, ignore_index=True)
#     n1 = int(2 * N / 3)
#
#     #Nuisance Parameter estimation
#
#     X1 = MRT_data[MRT_data["id"] > n1].iloc[:, 2:P + 2]
#     Y1 = MRT_data[MRT_data["id"] > n1].iloc[:, P + 4]
#     alphahat = np.array(sm.OLS(Y1, X1).fit().params)
#
#     # alphahat = theta1 * np.ones(P)
#     Y = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 4]) - np.dot(np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, 2:P + 2]), alphahat)
#     At_Pt = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 3]) - np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, P + 2])
#     X = np.array(MRT_data[MRT_data["id"] < n1 + 1].iloc[:, 2:P + 2].multiply(At_Pt, axis="index"))
#     # X -= X.mean(0)[None, :] #centering
#     # scaling = X.std(0) * np.sqrt(N)
#     # X /= np.sqrt(N) #scaling
#
#     beta = (beta_11/trueP) * np.concatenate((np.ones(trueP), np.zeros(P - trueP)))
#     A = MRT_data[MRT_data["id"] < n1 + 1].iloc[:, :2]
#     A = A.join(pd.DataFrame(X, columns = ['State'+str(i) for i in range(1,P+1)]))
#     A['Y'] = Y.tolist()
#
#
#     # active = np.zeros(P, bool)
#     # active[beta != 0] = True
#
#     # scaling = Y.std(0) * np.sqrt(n)
#     # Y /= scaling
#     return X, Y, beta, A
"""
MRT Data Generating Process for Selective Inference on Time-Varying Effect Moderation.

Matches the simulation design in Section 5 of:
    "Selective Inference for Time-Varying Moderated Effects"
    Bakshi, Dempsey, Panigrahi (JASA revision)

DGP Summary (minimal version: gamma1=0, beta10=0):
─────────────────────────────────────────────────────
    1. Innovations:   e_t ~ N(0, sigma_state^2 * Sigma_corr)   [= centered states S_tilde]
    2. Raw states:    S_t = rho_state * S_{t-1} + e_t           [used only for treatment]
    3. Treatment:     logit(p_t) = eta1 * A_{t-1} + eta2 * mean(S_t)
    4. Outcome:       Y_t = theta1^T e_t + (A_t - p_t)(beta^T e_t) + epsilon_t
    5. Error:         epsilon_t ~ AR(1) with rho_error, marginal SD = sigma_error

Two independent correlation structures:
    - Cross-moderator:  Sigma_corr[j,k] = corr^|j-k|   (Toeplitz, same time point)
    - Longitudinal:     Corr(eps_t, eps_s) = rho_error^|t-s|  (same individual, across time)

Why S_tilde (= e_t) as WCLS features, not raw S:
    The paper defines the causal excursion effect as linear in the centered
    moderators S_tilde_t = S_t - E[S_t | H_{t-1}, A_{t-1}]. With gamma1=0,
    S_tilde_t = S_t - rho * S_{t-1} = e_t (the innovation). The WCLS with
    features S_tilde is correctly specified: the excursion effect IS beta^T S_tilde.
    Using raw S instead would create misspecification (the lagged state term
    rho * S_{t-1} contaminates the treatment effect projection).

Signal scaling:
    beta_11 is the per-coordinate treatment effect for active moderators.
    Low/Medium/High signal: beta_11 in {0.2, 0.4, 0.8}.
    For varying n: scale as beta_11 * sqrt(30/n) to maintain O(n^{-1/2}) rate.

References:
    - Boruvka et al. (2018), JASA 113: original MRT simulation
    - Paper Section 5: simulation design
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ─── Utilities ─────────────────────────────────────────────────────────────────


def expit(x):
    """Logistic sigmoid, numerically stable."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def toeplitz_cov(p, corr):
    """
    Toeplitz correlation matrix: Sigma[j,k] = corr^|j-k|.
    Diagonal = 1, so this is a correlation matrix.
    """
    idx = np.arange(p)
    return corr ** np.abs(idx[:, None] - idx[None, :])


def iid_centered_noise(T, sd=1.0, dist="gaussian", rng=None):
    """
    Mean-zero iid noise with specified marginal SD.

    dist options:
        "gaussian"    : N(0, sd^2)
        "laplace"     : Laplace(0, sd/sqrt(2)), heavier tails, same variance
        "exponential" : sd * (Exp(1) - 1), right-skewed, mean 0, variance sd^2
    """
    if rng is None:
        rng = np.random.default_rng()
    if dist == "gaussian":
        return rng.normal(0.0, sd, size=T)
    if dist == "laplace":
        return rng.laplace(0.0, sd / np.sqrt(2.0), size=T)
    if dist == "exponential":
        return sd * (rng.exponential(1.0, size=T) - 1.0)
    raise ValueError(f"Unknown dist: {dist}")


def ar1_noise(rho, T, marginal_sd=1.0, dist="gaussian", rng=None):
    """
    AR(1) outcome noise with specified marginal SD and error distribution.

        epsilon_t = rho * epsilon_{t-1} + eta_t

    where eta_t has SD = marginal_sd * sqrt(1 - rho^2), so that
    Var(epsilon_t) = marginal_sd^2 stationarily.

    Paper: Corr(epsilon_t, epsilon_s) = rho^|t-s|, rho = 0.5.
    """
    if rng is None:
        rng = np.random.default_rng()

    innov_sd = marginal_sd * np.sqrt(1.0 - rho ** 2)
    eps = np.zeros(T)

    # Stationary initialization: eps_0 has marginal SD
    eps[0] = iid_centered_noise(1, sd=marginal_sd, dist=dist, rng=rng)[0]

    for t in range(1, T):
        eta = iid_centered_noise(1, sd=innov_sd, dist=dist, rng=rng)[0]
        eps[t] = rho * eps[t - 1] + eta

    return eps


# ─── Individual trajectory ─────────────────────────────────────────────────────


def generate_individual(
    id,
    *,
    T=30,
    p=50,
    true_p=5,
    rho_state=0.5,
    rho_error=0.5,
    corr=0.0,
    theta1,  # (p,) dense nuisance coefficient
    beta,  # (p,) sparse treatment effect coefficient
    sigma_state=1.5,
    sigma_error=1.0,
    eta1=-0.8,
    eta2=0.8,
    misspec_strength=0.0,
    error_dist="gaussian",
    rng=None,
):
    """
    Generate one individual's MRT trajectory.

    Paper Section 5 mapping
    ───────────────────────
    Step 1 — Innovations (= centered moderator states S_tilde):
        e_t ~ N(0, sigma_state^2 * Sigma_corr),  t = 1, ..., T
        Sigma_corr[j,k] = corr^|j-k|

        These are the WCLS moderator features.  Independent across time.

    Step 2 — Raw states (for treatment assignment only):
        S_0 ~ N(0, Sigma_stationary)       [stationary initialization]
        S_t = rho_state * S_{t-1} + e_t    [AR(1)]

        Only used in logit(p_t).  NOT used in WCLS.

    Step 3 — Treatment (Boruvka-style):
        logit(p_t) = eta1 * A_{t-1} + eta2 * mean(S_t)
        A_t | H_t ~ Bernoulli(p_t)

        mean(S_t) uses ALL p coordinates — no knowledge of E*.

    Step 4 — Outcome (Paper Section 5, with beta10=0, theta2=0):
        Y_t = theta1^T e_t                 [nuisance baseline]
            + misspec_strength * (mean(e_t^2) - sigma_state^2)
                                            [non-linear misspec term]
            + (A_t - p_t)(beta^T e_t)      [causal excursion effect]
            + epsilon_t                     [AR(1) noise]

        When misspec_strength > 0, the OLS nuisance model (Y ~ S_tilde)
        is misspecified because it cannot capture the quadratic term.
        The term is centered (E[e_j^2] = sigma_state^2) so it has mean 0.

    Step 5 — Error:
        Corr(epsilon_t, epsilon_s) = rho_error^|t-s|
        Marginal SD(epsilon_t) = sigma_error
        Distribution: gaussian, laplace, or exponential (centered)

    Returns
    -------
    DataFrame with columns:
        id, decision_point,
        state1..p   (raw S_t — for diagnostics / treatment assignment),
        stilde1..p  (innovations e_t = S_tilde_t — WCLS moderator features),
        prob, action, outcome
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Step 1: Innovations e_t = S_tilde_t ────────────────────────────────
    #
    # Paper: e_t ~ N(0, sigma^2 I_p).  We generalize to Toeplitz cross-corr.
    # With corr=0 (paper default), innovations are independent across coords.
    if corr == 0.0:
        innovations = rng.normal(0.0, sigma_state, size=(T, p))
    else:
        Sigma_corr = toeplitz_cov(p, corr)
        innov_cov = sigma_state ** 2 * Sigma_corr
        innovations = rng.multivariate_normal(np.zeros(p), innov_cov, size=T)

    # ── Step 2: Raw states S_t (for treatment probabilities) ───────────────
    #
    # S_t = rho_state * S_{t-1} + e_t
    # Stationary marginal: Var(S_j) = sigma_state^2 / (1 - rho_state^2)
    states = np.zeros((T, p))

    # Stationary initialization of S_0
    if corr == 0.0:
        stationary_sd = sigma_state / np.sqrt(1.0 - rho_state ** 2)
        S_prev = rng.normal(0.0, stationary_sd, size=p)
    else:
        Sigma_S = (sigma_state ** 2 / (1.0 - rho_state ** 2)) * toeplitz_cov(
            p, corr
        )
        S_prev = rng.multivariate_normal(np.zeros(p), Sigma_S)

    for t in range(T):
        states[t] = rho_state * S_prev + innovations[t]
        S_prev = states[t]

    # ── Step 3: Treatment assignment ───────────────────────────────────────
    #
    # logit(p_t) = eta1 * A_{t-1} + eta2 * mean(S_t)
    # Following Boruvka: eta1=-0.8, eta2=0.8
    # mean(S_t) uses all p coordinates — no knowledge of true support E*
    actions = np.zeros(T)
    probs = np.zeros(T)

    for t in range(T):
        lag_a = actions[t - 1] if t > 0 else 0.0
        logit_p = eta1 * lag_a + eta2 * np.mean(states[t])
        probs[t] = expit(logit_p)
        actions[t] = rng.binomial(1, probs[t])

    # ── Step 4: Outcome ────────────────────────────────────────────────────
    #
    # Y_t = theta1^T e_t + (A_t - p_t)(beta^T e_t) + epsilon_t
    #
    # Nuisance:  theta1^T e_t   (dense — all moderators affect baseline)
    # Signal:    (A-p)(beta^T e_t)  (sparse — only E* moderators matter)
    nuisance_component = innovations @ theta1
    centered_treatment = actions - probs
    treatment_effect = centered_treatment * (innovations @ beta)

    # Non-linear misspecification term:
    #   misspec_strength * (mean(e_t^2) - sigma_state^2)
    # Centered so E[term] = 0.  OLS on S_tilde can't capture this.
    if misspec_strength != 0.0:
        nonlinear_term = misspec_strength * (
            np.mean(innovations ** 2, axis=1) - sigma_state ** 2
        )
    else:
        nonlinear_term = 0.0

    # ── Step 5: Error ──────────────────────────────────────────────────────
    epsilon = ar1_noise(rho_error, T, sigma_error, error_dist, rng)

    outcome = nuisance_component + nonlinear_term + treatment_effect + epsilon

    # ── Build DataFrame ────────────────────────────────────────────────────
    data = {"id": id, "decision_point": np.arange(1, T + 1)}
    for j in range(p):
        data[f"state{j + 1}"] = states[:, j]
    for j in range(p):
        data[f"stilde{j + 1}"] = innovations[:, j]
    data["prob"] = probs
    data["action"] = actions
    data["outcome"] = outcome

    return pd.DataFrame(data)


# ─── Full MRT instance ─────────────────────────────────────────────────────────


def MRT_instance(
    *,
    N=30,
    T=30,
    p=50,
    true_p=5,
    rho_state=0.5,
    rho_error=0.5,
    corr=0.0,
    theta1_value=0.5,
    beta_11=0.4,
    sigma_state=1.5,
    sigma_error=1.5,
    misspec_strength=0.0,
    eta1=-0.5,
    eta2=0.5,
    error_dist="gaussian",
    nuisance="ols",
    n_nuisance=None,
    seed=None,
):
    """
    Generate MRT data and return WCLS design for the lasso / selective inference.

    Parameter mapping
    ─────────────────
    theta1 = theta1_value * 1_p     Dense nuisance.  Paper: 0.8 * 1_p.
    beta_j = beta_11, j <= true_p   Sparse signal (per-coordinate).
    beta_j = 0,       j >  true_p

    beta_11 is the raw per-coordinate signal strength:
        Low signal:    beta_11 = 0.2
        Medium signal: beta_11 = 0.4
        High signal:   beta_11 = 0.8
        Varying n:     beta_11 = beta_11_base * sqrt(30/n)  for O(n^{-1/2}) scaling

    Nuisance estimation (on held-out split)
    ───────────────────────────────────────
    "oracle":  mu_hat = theta1^T S_tilde  using true theta1.
               sigma_hat = sigma_error (known).
    "ols":     mu_hat = theta_hat^T S_tilde, theta_hat from OLS(Y ~ S_tilde)
               on the nuisance split.  Correctly specified: E[(A-p)|e_t]=0,
               so the treatment effect does not bias the OLS.
               sigma_hat = residual SD from nuisance OLS.

    WCLS design
    ───────────
    After residualizing Y_res = Y - mu_hat:
        Y_res ≈ (A - p)(beta^T S_tilde) + epsilon + estimation_error

    Design matrix: X = (A - p) * S_tilde,  shape (n_analysis * T,  p)
    Target:        beta,                    shape (p,)

    Returns
    -------
    X :          ndarray (n_analysis*T, p)   WCLS design
    Y :          ndarray (n_analysis*T,)     residualized outcome
    beta :       ndarray (p,)                true sparse coefficient
    A_df :       DataFrame                   id, decision_point, design cols, Y
    sigma_hat :  float                       estimated noise SD
    """
    rng = np.random.default_rng(seed)

    # ── Coefficient vectors ────────────────────────────────────────────────
    theta1 = theta1_value * np.ones(p)
    beta = np.zeros(p)
    beta[:true_p] = beta_11

    # ── Generate all individuals ───────────────────────────────────────────
    dfs = [
        generate_individual(
            i,
            T=T,
            p=p,
            true_p=true_p,
            rho_state=rho_state,
            rho_error=rho_error,
            corr=corr,
            theta1=theta1,
            beta=beta,
            sigma_state=sigma_state,
            sigma_error=sigma_error,
            eta1=eta1,
            eta2=eta2,
            misspec_strength=misspec_strength,
            error_dist=error_dist,
            rng=rng,
        )
        for i in range(1, N + 1)
    ]
    MRT_data = pd.concat(dfs, ignore_index=True)

    # ── Split: nuisance vs analysis ────────────────────────────────────────
    #
    # Paper Section 6.1: "30% ... nuisance estimator ... remaining 70%"
    # We use ~1/3 for nuisance (at least 5 individuals).
    if n_nuisance is None:
        n_nuisance = max(int(np.ceil(N / 3)), 5)

    n_analysis = N - n_nuisance

    analysis_mask = MRT_data["id"] <= n_analysis
    nuisance_mask = MRT_data["id"] > n_analysis

    stilde_cols = [f"stilde{j + 1}" for j in range(p)]

    Stilde_analysis = MRT_data.loc[analysis_mask, stilde_cols].to_numpy()
    Y_analysis = MRT_data.loc[analysis_mask, "outcome"].to_numpy()
    A_analysis = MRT_data.loc[analysis_mask, "action"].to_numpy()
    p_analysis = MRT_data.loc[analysis_mask, "prob"].to_numpy()

    # ── Nuisance estimation ────────────────────────────────────────────────
    #
    # Target: theta1 in  Y_t = theta1^T e_t + (A-p)(beta^T e_t) + eps_t
    #
    # OLS of Y on S_tilde (= e_t) is consistent for theta1 because
    # E[A_t - p_t | H_t] = 0  =>  E[(A-p)(beta^T e_t) | e_t] = 0.
    # So the treatment effect term is mean-zero conditional on e_t and
    # does not bias the OLS.
    if nuisance == "oracle":
        theta_hat = theta1.copy()
        sigma_hat = sigma_error
    elif nuisance == "ols":
        Stilde_nuis = MRT_data.loc[nuisance_mask, stilde_cols].to_numpy()
        Y_nuis = MRT_data.loc[nuisance_mask, "outcome"].to_numpy()

        ols_fit = sm.OLS(Y_nuis, Stilde_nuis).fit()
        theta_hat = np.asarray(ols_fit.params)

        # sigma_hat: residual SD after removing nuisance.
        # Residuals ≈ (A-p)(beta^T e_t) + eps — includes treatment effect.
        # This estimates SD(Y_res), which is what the paper uses for
        # lambda (Negahban) and randomization Omega = sigma_hat^2 * I_p.
        # Paper Section 6.1: "tau^2 = sigma_hat^2 where sigma_hat^2 is
        # the estimated variance of the response."
        df_resid = max(float(ols_fit.df_resid), 1.0)
        sigma_hat = np.sqrt(np.sum(ols_fit.resid ** 2) / df_resid)
    else:
        raise ValueError("nuisance must be 'oracle' or 'ols'")

    # ── Residualize outcome ────────────────────────────────────────────────
    #
    # Y_res = Y - theta_hat^T S_tilde
    #       ≈ (A - p)(beta^T S_tilde) + epsilon + estimation_error
    Y = Y_analysis - Stilde_analysis @ theta_hat

    # ── WCLS design: X = (A - p) * S_tilde ────────────────────────────────
    #
    # Column j of X:  (A_t - p_t) * S_tilde_{t,j}
    # The lasso on (X, Y) selects moderators; beta is the target.
    At_pt = A_analysis - p_analysis
    X = Stilde_analysis * At_pt[:, None]

    # ── Output DataFrame ───────────────────────────────────────────────────
    A_df = MRT_data.loc[analysis_mask, ["id", "decision_point"]].copy()
    X_df = pd.DataFrame(
        X, columns=[f"State{j + 1}" for j in range(p)], index=A_df.index
    )
    A_df = A_df.join(X_df)
    A_df["Y"] = Y

    return X, Y, beta, A_df, sigma_hat