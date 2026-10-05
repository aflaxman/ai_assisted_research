"""Migration-adjusted power prior of Dupuis (2026), Chapter 2 (binomial) and
Chapter 4 (multinomial).

For focal area j with baseline (migration-naive) posterior approximated by
Beta(alpha_j, beta_j), sampled data (y_k positives of n_k) in every area k, and
post-migration composition counts C[j, k] (= residents of j who originated in k,
C[j, j] = stayers), one draw of the migration-adjusted posterior is

    a_j. ~ independent Beta(C[j,k], N_j - C[j,k]) for k with C[j,k] > 0, renormalized to sum to 1
    theta_j ~ Beta(alpha_j + sum_k a_jk y_k,  beta_j + sum_k a_jk (n_k - y_k))

(the k = j term carries the focal area's own likelihood, scaled by the stayer
proportion a_jj, exactly as in the dissertation). The scalars are drawn from
their prior at every iteration, never updated from the data.
"""
from __future__ import annotations

import numpy as np


def beta_moments(draws, floor=1e-9):
    """Method-of-moments Beta(alpha, beta) fit, per column, to posterior draws (S, r)."""
    mean = np.clip(draws.mean(0), 1e-6, 1 - 1e-6)
    var = np.maximum(draws.var(0), floor)
    common = np.maximum(mean * (1 - mean) / var - 1.0, 1e-3)
    return mean * common, (1 - mean) * common


def draw_scalars(C, rng, size, scheme="beta"):
    """Draw migration scalars a (size, r, r) from the prior implied by composition counts C."""
    C = np.asarray(C, dtype=np.float64)
    r = C.shape[0]
    N = C.sum(1, keepdims=True)
    if scheme == "dirichlet":
        g = rng.gamma(np.where(C > 0, C, 1e-12)[None, :, :], 1.0, size=(size, r, r))
        g = np.where(C[None] > 0, g, 0.0)
        return g / g.sum(2, keepdims=True)
    # independent Beta(C_jk, N_j - C_jk), then renormalize each row (Dupuis Ch. 2)
    a = np.zeros((size, r, r))
    mask = C > 0
    alpha = C[mask]
    beta = np.maximum((N * np.ones_like(C))[mask] - C[mask], 1e-6)
    a[:, mask] = rng.beta(alpha[None, :], beta[None, :], size=(size, mask.sum()))
    return a / a.sum(2, keepdims=True)


def power_prior_binomial(baseline_draws, y_s, n_s, C, rng, n_draws=1000, scheme="beta",
                         double_count=True):
    """Return (n_draws, r) migration-adjusted posterior draws of the area proportions.

    double_count=True reproduces the dissertation (the focal area's sample enters
    both through the Beta prior fitted to the baseline posterior and through its
    own a_jj-scaled likelihood). double_count=False is a variant that tempers the
    baseline Beta prior by a_jj instead of re-using the focal area's likelihood.
    """
    alpha, beta = beta_moments(np.asarray(baseline_draws))
    y_s = np.asarray(y_s, dtype=np.float64)
    n_s = np.asarray(n_s, dtype=np.float64)
    a = draw_scalars(C, rng, n_draws, scheme)          # (M, r, r)
    if double_count:
        shape1 = alpha[None, :] + a @ y_s
        shape2 = beta[None, :] + a @ (n_s - y_s)
    else:
        ajj = np.einsum("mjj->mj", a)
        a_off = a.copy()
        idx = np.arange(C.shape[0])
        a_off[:, idx, idx] = 0.0
        shape1 = ajj * (alpha[None, :] - 1) + 1 + a_off @ y_s
        shape2 = ajj * (beta[None, :] - 1) + 1 + a_off @ (n_s - y_s)
    return rng.beta(np.maximum(shape1, 1e-6), np.maximum(shape2, 1e-6))


def power_prior_multinomial(baseline_draws, Y_s, C, rng, n_draws=1000, scheme="dirichlet"):
    """Multinomial version (Dupuis Ch. 4): baseline_draws (S, r, H) -> Dirichlet prior per
    area by moment matching; Y_s (r, H) sampled category counts; returns (n_draws, r, H)."""
    draws = np.asarray(baseline_draws)
    mean = np.clip(draws.mean(0), 1e-6, 1)                      # (r, H)
    mean = mean / mean.sum(1, keepdims=True)
    var = np.maximum(draws.var(0), 1e-9)
    # Dirichlet precision s from the first H-1 categories: Var = m(1-m)/(s+1)
    s = np.median(mean * (1 - mean) / var - 1.0, axis=1)
    s = np.maximum(s, 1e-3)
    alpha0 = mean * s[:, None]
    a = draw_scalars(C, rng, n_draws, scheme)                    # (M, r, r)
    conc = alpha0[None] + np.einsum("mjk,kh->mjh", a, np.asarray(Y_s, np.float64))
    g = rng.gamma(np.maximum(conc, 1e-6))
    return g / g.sum(2, keepdims=True)
