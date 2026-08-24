# Selective inference for M-estimators and applications in time-varying causal effect moderation

Reference implementation and reproduction code for:
> **Selective Inference for Time-Varying Effect Moderation** (under revision JASA)
> Soham Bakshi, Walter Dempsey, Snigdha Panigrahi
> [arXiv:2411.15908](https://arxiv.org/abs/2411.15908)

Micro-randomized trials (MRTs) collect many time-varying context features, any
of which might moderate the causal excursion effect of a just-in-time
intervention. Fitting all of them is uninterpretable; picking a few by eye or by
lasso and then reporting standard intervals is invalid. This repository
implements a two-step alternative: select a small moderator set with a
**Gaussian-randomized lasso** on the weighted-and-centered least squares (WCLS)
design, then condition on the selection event to build a pivot that yields
uniformly asymptotically valid semi-parametric intervals **in the selected
model**.

The headline empirical claims — valid coverage where polyhedral and
sample-splitting methods fail, and finite (rather than occasionally infinite)
interval lengths — are reproduced by
[`selectinf/tests/output_simulations.ipynb`](selectinf/tests/output_simulations.ipynb).

---

## Methods compared

| Method | Label in code | What it does |
|---|---|---|
| **Randomized SI** (ours) | `randomized_si` | Randomized lasso, then conditional selective inference via an exact grid pivot |
| Polyhedral | `polyhedral` | Lee et al. (2016), through R's `selectiveInference::fixedLassoInf` |
| Naive GEE | `naive_gee` | Robust sandwich intervals on the selected columns, no selection correction |
| Data splitting | `data_splitting` | Lasso on a train split by individual, GEE inference on the held-out split |

All four run on the *same* simulated dataset within each Monte Carlo
replication, so coverage and length are directly comparable.

---

## Installation

### Python

**Use Python 3.9–3.11.** This is not a preference — `regreg`, which supplies the
convex solver behind `selectinf.lasso`, fails to build on 3.12+ (its `setup.py`
calls `configparser.SafeConfigParser`, removed in that release).

```bash
git clone https://github.com/BakshiSoham/SI-MRT.git
cd SI-MRT

python3.11 -m venv .venv && source .venv/bin/activate    # or conda

# regreg first: its latest PyPI release (0.1.3) ships wheels only up to
# CPython 3.6 and has no source distribution, so `pip install regreg` fails.
pip install git+https://github.com/regreg/regreg.git

pip install -r requirements.txt
```

Alternatively, conda-forge carries a regreg build:
`conda install -c conda-forge regreg`.

Everything else is standard: `numpy`/`scipy` do the heavy lifting and
`statsmodels` supplies the GEE sandwich estimator.

`selectinf` is pure Python and needs no build/install step: run everything
from the repository root (or add the root to `PYTHONPATH`) so that
`import selectinf` resolves. There is no `pip install -e .` here — `setup.py`
is vestigial, inherited from an upstream package that builds Cython/C
extensions this repository doesn't include, and will fail if you try to run
it (`import versioneer` / `cythexts` / `setup_helpers`, none of which are
vendored here).

### R dependency

The **polyhedral** comparison calls R through `rpy2`. Install R, then:

```r
install.packages("selectiveInference")
```

If R is unavailable, everything else still runs — drop `rpy2` from
`requirements.txt` and remove the `polyhedral` arm from the method list in the
notebook's simulation cell.

---

## Repository layout

```
selectinf/                  the inference package
├── lasso.py                randomized lasso (gaussian / logistic / poisson /
│                           coxph / sqrt_lasso / WCLS constructors)
├── query.py                gaussian_query: the conditional law of the optimization
│                           variables given the selection event
├── randomization.py        randomization densities (isotropic Gaussian, Laplace, …)
├── grid_inference.py       grid-based construction of the selective pivot,
│                           confidence intervals and p-values
├── exact_reference.py      exact_grid_inference: the truncated-Gaussian reference
│                           measure used by the pivot
├── Utils/
│   ├── base.py             target specifications — selected_targets,
│   │                       selected_targets_WCLS (the MRT/GEE sandwich target)
│   └── discrete_family.py  exponential-family tilting, interval inversion
├── PoSI.R                  R-side helpers
└── tests/
    ├── MRT_instance.py           MRT data-generating process (paper Section 5)
    ├── MRT_instance_older.py     superseded DGP, kept for reference
    ├── instance.py               generic Gaussian design instances
    ├── output_simulations.ipynb  ← all simulations, figures and tables
    ├── test_MRT_instance.py      script-style driver for the MRT simulation
    ├── test_exact_reference.py   script-style driver for the pivot
    └── test_lasso.py             script-style driver for the randomized lasso
```

This repository does **not** include the VALENTINE trial data or any
notebooks/files derived from it — those data are not publicly shareable (see
[VALENTINE data application](#valentine-data-application), below). A prior,
private version of this repository had a `realdata/` directory and a data
exploration notebook; both were removed from this repo's history entirely
(not just deleted in a later commit) before it was made public.

**On the `tests/` directory.** Despite the name and the `test_*.py` filenames,
these are simulation drivers, not a `pytest` suite — they execute long
Monte Carlo loops at import time and have no assertions. Do not point `pytest`
at them expecting a green run. The notebook supersedes all three.

---

## The data-generating process

`MRT_instance` (in `selectinf/tests/MRT_instance.py`) implements the design in
Section 5 of the paper:

```
innovations   e_t  ~  N(0, sigma_state^2 * Sigma_corr),  Sigma_corr[j,k] = corr^|j-k|
states        S_t  =  rho_state * S_{t-1} + e_t
treatment     logit(p_t) = eta1 * A_{t-1} + eta2 * mean(S_t)
outcome       Y_t  =  theta1' e_t + (A_t - p_t)(beta' e_t) + eps_t
error         eps_t ~ AR(1) with rho_error, marginal SD sigma_error
```

The WCLS features are the *centered* moderators `S_tilde_t = S_t - E[S_t | H_{t-1}, A_{t-1}]`,
which equal the innovations `e_t` when `gamma1 = 0`. This keeps the excursion
effect exactly linear in the features; using raw states instead would introduce
misspecification through the lagged term. The design matrix is
`X = (A - p) * S_tilde` and the response is `Y` residualized against a nuisance
fit estimated on a held-out third of individuals.

Key knobs: `N` (individuals), `T` (decision points), `p` (candidate moderators),
`beta_11` (per-coordinate signal, {0.2, 0.4, 0.8} for low/medium/high), `corr`
(cross-moderator correlation), `rho_error` (longitudinal correlation),
`error_dist` (`gaussian` / `laplace` / `exponential`), `misspec_strength`, and
`nuisance` (`ols` or `oracle`).

---

## Minimal usage

```python
import numpy as np
from selectinf.tests.MRT_instance import MRT_instance
from selectinf.lasso import lasso
from selectinf.Utils.base import selected_targets_WCLS

# 1. Simulate an MRT and build the WCLS design
X, Y, beta, A_df, sigma_hat = MRT_instance(N=60, T=30, p=50, beta_11=0.2, seed=0)

# 2. Penalty: Negahban et al. (2012) score prescription
eps = np.random.standard_normal((X.shape[0], 2000)) * sigma_hat
W = np.median(np.max(np.abs(X.T @ eps), axis=0))

# 3. Randomized lasso; tau matches the gradient scale
tau = np.sqrt(np.mean((X ** 2).sum(0))) * sigma_hat
conv = lasso.gaussian(X, Y, W, ridge_term=0.0, randomizer_scale=tau)
nonzero = conv.fit() != 0

# 4. Conditional selective inference on the selected moderators
conv.setup_inference(dispersion=1)
target = selected_targets_WCLS(conv.loglike, A_df, conv.observed_soln,
                               K=conv.K, dispersion=1)
result = conv.inference(target)
print(result[["lower_confidence", "upper_confidence"]])
```

The interval target is the projection
`beta_target = pinv(X[:, E]) @ X @ beta` — the best linear approximation of the
true moderated effect *within the selected model* `E`. Coverage is defined
against this data-dependent target, which is the estimand each method actually
addresses.

---

## Reproducing the paper

Open the notebook and run the two setup cells first; every figure cell after
that is independent.

```bash
cd selectinf/tests
jupyter notebook output_simulations.ipynb
```

`B` (Monte Carlo replications per setting) is set in its own cell near the top.
`B = 500` reproduces the paper; drop it to 20–50 for a smoke test. The full grid
at `B = 500` takes hours — the randomized-lasso fit and grid pivot dominate.

| Notebook section | Output | Paper location |
|---|---|---|
| Vary signal strength β₁₁ | `fig_vary_signal.pdf` | Simulations |
| Vary sample size N | `fig_vary_N.pdf` | Simulations |
| Vary randomization level τ | `fig_vary_tau.pdf` | Simulations |
| Cross-moderator correlation | `fig_vary_signal_corr.pdf` | Appendix F |
| Nuisance misspecification | `fig_vary_N_misspecified.pdf` | Appendix F |
| Laplace errors | `fig_vary_signal_laplace.pdf` | Appendix F |
| Exponential errors | `fig_vary_signal_exponential.pdf` | Appendix F |
| Conditional coverage | `fig_conditional_coverage.pdf`, `conditional_coverage_table.csv` | Section AE.4 |
| Seed stability (K = 20 draws, fixed data) | `fig_seed_stability.pdf`, `seed_stability_raw.csv` | Section R1.3 |
| VALENTINE application | three CI plots + LaTeX significance table | Section 6 |

Each simulation panel reports three things side by side: selection quality
(precision / recall / F1 against the true active set), coverage of the selected
target, and the distribution of *bounded* interval lengths — the last of these
matters because polyhedral intervals are frequently infinite.

---

## VALENTINE data application

The application uses the VALENTINE micro-randomized trial: transformed step
count as the response, time-varying context features as candidate moderators.
`load_valentine_design` mirrors `MRT_instance` step for step —

1. one-hot encode categoricals and split the enrollment-time trend by phase;
2. optionally add Phase × moderator and demographic × behavioral interactions;
3. split individuals 1/3 (nuisance estimation) vs 2/3 (analysis);
4. residualize `Y` against the OLS nuisance fit and form `X = (A - p) · covariates`;
5. standardize the design columns to unit SD.

Three moderator sets are analyzed: main effects (25 candidates), phase
interactions (58), and extended interactions (94).

**The VALENTINE trial data are not included in this repository and cannot be
made public.** They were collected and analyzed under a University of
Michigan IRB; access requires a data-use agreement with the study PI. If you
have obtained the data this way, the notebook reads `realdata/AW_df.csv`
relative to the repository root (create that directory yourself — it isn't
tracked here). Point it elsewhere with an environment variable:

```bash
export VALENTINE_CSV=/path/to/AW_df.csv
```

Without access to the restricted data, the VALENTINE-application cells in the
notebook won't run, but everything else will: `selectinf/tests/MRT_instance.py`
generates synthetic data matching the same design (see [The
data-generating process](#the-data-generating-process), above), so the
selection and selective-inference steps of the pipeline can still be exercised
end to end on simulated data.

---


## Citation

```bibtex
@article{bakshi2024selective,
  title   = {Selective Inference for Time-Varying Effect Moderation},
  author  = {Bakshi, Soham and Dempsey, Walter and Panigrahi, Snigdha},
  journal = {arXiv preprint arXiv:2411.15908},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.15908}
}
```

## References

- Bakshi, Dempsey & Panigrahi (2024). *Selective inference for time-varying effect moderation.* arXiv:2411.15908.
- Boruvka, Almirall, Witkiewitz & Murphy (2018). *Assessing time-varying causal effect moderation in mobile health.* JASA 113(523), 1112–1121.
- Lee, Sun, Sun & Taylor (2016). *Exact post-selection inference, with application to the lasso.* Annals of Statistics 44(3).
- Negahban, Ravikumar, Wainwright & Yu (2012). *A unified framework for high-dimensional analysis of M-estimators with decomposable regularizers.* Statistical Science 27(4).
