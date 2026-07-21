"""Estimation, uncertainty quantification, and forecasting.

These three deliverables mirror the three benefits the 2026 abstract claims the
framework provides: "improves parameter estimation, uncertainty quantification,
and short-term forecasting."
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from model import (
    Epidemic,
    ModelParams,
    Policy,
    alarm_path,
    log_likelihood,
    policy_signal,
)

# ---------------------------------------------------------------------------
# Maximum-likelihood estimation
# ---------------------------------------------------------------------------

# Parameters we estimate, with (lower, upper) bounds for the optimizer.
FIT_BOUNDS = {
    "beta": (0.05, 3.0),
    "gamma": (0.02, 0.9),
    "endog_delta": (0.0, 1.0),
    "endog_x0": (10.0, 5000.0),
    "endog_nu": (0.5, 8.0),
    "policy_weight": (0.0, 1.0),
    "adapt_rate": (0.02, 1.0),
}


def fit_mle(
    epi: Epidemic,
    fit_names: list[str],
    p_template: ModelParams,
    n_restarts: int = 8,
    seed: int = 0,
) -> dict:
    """Maximum-likelihood estimate via multi-start L-BFGS-B.

    Returns a dict with the estimate, the log-likelihood, and the numerically
    approximated standard errors (from the inverse observed-information matrix).
    """
    rng = np.random.default_rng(seed)
    bounds = [FIT_BOUNDS[n] for n in fit_names]

    def neg_ll(x):
        return -log_likelihood(dict(zip(fit_names, x)), epi, p_template)

    best = None
    for r in range(n_restarts):
        x0 = np.array([lo + (hi - lo) * rng.random() for lo, hi in bounds]) \
            if r else np.array([getattr(p_template, n) for n in fit_names])
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
        if res.success or np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res

    est = dict(zip(fit_names, best.x))
    se = _standard_errors(best.x, fit_names, epi, p_template, bounds)
    return {"estimate": est, "loglik": -best.fun, "se": dict(zip(fit_names, se)),
            "fit_names": fit_names}


def _standard_errors(x, fit_names, epi, p_template, bounds):
    """SEs from a finite-difference Hessian of the negative log-likelihood."""
    n = len(x)
    H = np.zeros((n, n))
    steps = np.array([max(1e-4, abs(v) * 1e-3) for v in x])

    def nll(v):
        return -log_likelihood(dict(zip(fit_names, v)), epi, p_template)

    f0 = nll(x)
    for i in range(n):
        for j in range(i, n):
            xi = x.copy(); xi[i] += steps[i]; xi[j] += steps[j]
            xij = x.copy(); xij[i] += steps[i]
            xji = x.copy(); xji[j] += steps[j]
            fpp, fp_i, fp_j = nll(xi), nll(xij), nll(xji)
            H[i, j] = H[j, i] = (fpp - fp_i - fp_j + f0) / (steps[i] * steps[j])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)
    return se


# ---------------------------------------------------------------------------
# Bayesian uncertainty quantification: random-walk Metropolis
# ---------------------------------------------------------------------------


def fit_mcmc(
    epi: Epidemic,
    fit_names: list[str],
    p_template: ModelParams,
    n_iter: int = 20_000,
    burn: int = 5_000,
    thin: int = 5,
    proposal_sd: dict | None = None,
    init: dict | None = None,
    seed: int = 1,
) -> dict:
    """Random-walk Metropolis with flat (bounded) priors -> posterior draws.

    The complete-data likelihood is very informative for large populations, so
    the posterior is sharply peaked.  Seed ``init`` at the MLE and set
    ``proposal_sd`` to (a fraction of) the MLE standard errors to keep the
    acceptance rate in a workable range.
    """
    rng = np.random.default_rng(seed)
    bounds = {n: FIT_BOUNDS[n] for n in fit_names}
    if proposal_sd is None:
        proposal_sd = {n: 0.03 * (hi - lo) for n, (lo, hi) in bounds.items()}

    def logpost(theta):
        for n, v in theta.items():
            lo, hi = bounds[n]
            if not (lo <= v <= hi):
                return -np.inf
        return log_likelihood(theta, epi, p_template)

    cur = dict(init) if init else {n: getattr(p_template, n) for n in fit_names}
    cur_lp = logpost(cur)
    draws, accept = [], 0
    for it in range(n_iter):
        prop = {n: cur[n] + rng.normal(0, proposal_sd[n]) for n in fit_names}
        prop_lp = logpost(prop)
        if np.log(rng.random() + 1e-300) < prop_lp - cur_lp:
            cur, cur_lp = prop, prop_lp
            accept += 1
        if it >= burn and (it - burn) % thin == 0:
            draws.append([cur[n] for n in fit_names])

    samples = np.array(draws)
    summ = {}
    for i, n in enumerate(fit_names):
        col = samples[:, i]
        summ[n] = {"mean": float(col.mean()), "sd": float(col.std()),
                   "q2.5": float(np.percentile(col, 2.5)),
                   "q97.5": float(np.percentile(col, 97.5))}
    return {"samples": samples, "fit_names": fit_names,
            "accept_rate": accept / n_iter, "summary": summ}


# ---------------------------------------------------------------------------
# Short-term forecasting with posterior predictive uncertainty
# ---------------------------------------------------------------------------


def forecast(
    epi_obs: Epidemic,
    mcmc: dict,
    p_template: ModelParams,
    policies: list[Policy],
    horizon: int,
    n_paths: int = 400,
    seed: int = 2,
) -> dict:
    """Posterior-predictive forecast of daily incidence over `horizon` days.

    Continues the epidemic forward from the last observed state, drawing a
    parameter vector from the posterior for each simulated path so the fan
    reflects both parameter and process (chain-binomial) uncertainty.
    """
    from model import target_alarm

    rng = np.random.default_rng(seed)
    fit_names = mcmc["fit_names"]
    samples = mcmc["samples"]
    N = epi_obs.N
    T = len(epi_obs.S)
    n_days = T + horizon
    p_sig = policy_signal(policies, n_days)

    paths = np.zeros((n_paths, horizon))
    for k in range(n_paths):
        theta = dict(zip(fit_names, samples[rng.integers(len(samples))]))
        p = ModelParams(**{**p_template.__dict__, **theta})
        pi_IR = 1.0 - np.exp(-p.gamma)

        # Reconstruct alarm state at the forecast origin from observed incidence.
        hist_inf = list(epi_obs.new_infections[:T])
        S, I = int(epi_obs.S[T - 1]), int(epi_obs.I[T - 1])
        a_prev = alarm_path(epi_obs.new_infections, epi_obs.policy, p)[T - 1]

        for h in range(horizon):
            t = T + h
            window = np.array(hist_inf[max(0, t - p.m):t])
            avg_inc = window.mean() if window.size else 0.0
            a_star = target_alarm(avg_inc, p_sig[t], p)
            a_prev = (1 - p.adapt_rate) * a_prev + p.adapt_rate * a_star
            pi_SI = 1.0 - np.exp(-p.beta * (1.0 - a_prev) * I / N)
            i_star = rng.binomial(S, pi_SI)
            r_star = rng.binomial(I, pi_IR)
            paths[k, h] = i_star
            hist_inf.append(i_star)
            S -= i_star
            I += i_star - r_star

    return {
        "median": np.median(paths, axis=0),
        "lo": np.percentile(paths, 2.5, axis=0),
        "hi": np.percentile(paths, 97.5, axis=0),
        "paths": paths,
    }
