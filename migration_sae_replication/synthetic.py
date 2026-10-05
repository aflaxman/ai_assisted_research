"""Synthetic TB-registry populations, migration scenarios and the three
data-generating processes (DGPs) used in Dupuis (2026).

Everything is a stand-in for Ukraine's eTB Manager registry (2015-2018), which
is not public: district case loads are proportional to the population proxy,
covariate mixes and effects are plausible values, and the "Ukraine" base
pattern of the dissertation is replaced by a south-east gradient.
"""
from __future__ import annotations

import numpy as np

from geography import Geography

CASES_PER_CAPITA = 0.0024          # ~60 notified TB cases /100k/yr over four years
P_MALE, P_PREV = 0.68, 0.22        # share male; share previously treated
P_AGE = np.array([0.25, 0.25, 0.22, 0.28])          # 18-34, 35-44, 45-54, 55+
# log-odds effects on rifampicin resistance: intercept (set per district), male,
# age 35-44, 45-54, 55+ (ref 18-34), previously treated
BETA_TRUE = np.array([0.0, 0.10, 0.10, 0.0, -0.30, 1.20])
PI_KNOWN = {0: 0.08, 1: 0.16}      # P(RR status known) for new / previously treated cases
LEVELS = {"low": 0.10, "medium": 0.40, "high": 0.70}
PATTERNS = ["crisis_idp", "crisis_dist", "into_hotcold", "into_urban",
            "neighbors", "out_hotcold", "out_urban", "distance"]
PATTERN_LABELS = {"crisis_idp": "Crisis-IDPs", "crisis_dist": "Crisis-Dist", "into_hotcold": "Into-Hot-Cold",
                  "into_urban": "Into-Urban", "neighbors": "Neighbors", "out_hotcold": "Out-Hot-Cold",
                  "out_urban": "Out-Urban", "distance": "Distance"}
BASES = ["block", "hotcold", "random", "gradient"]
BASE_LABELS = {"block": "Block", "hotcold": "Hot-Cold", "random": "Random", "gradient": "SE-gradient"}


# --------------------------------------------------------------------------- covariates
def covariate_cells():
    """16 covariate patterns: returns X (16, 6), cell probabilities (16,), prev-treated flag (16,)."""
    rows, probs, prev = [], [], []
    for sex in (0, 1):
        for age in range(4):
            for pt in (0, 1):
                x = np.zeros(6)
                x[0] = 1.0
                x[1] = sex
                if age > 0:
                    x[1 + age] = 1.0
                x[5] = pt
                rows.append(x)
                probs.append((P_MALE if sex else 1 - P_MALE) * P_AGE[age] * (P_PREV if pt else 1 - P_PREV))
                prev.append(pt)
    return np.array(rows), np.array(probs), np.array(prev)


def calibrate_intercepts(target, X, cell_p, beta=BETA_TRUE, iters=30):
    """Per-district intercept mu_j with  sum_c cell_p[c] * logistic(mu_j + X_c beta) = target_j."""
    xb = X @ beta
    mu = np.log(target / (1 - target)) - cell_p @ xb
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(mu[:, None] + xb[None, :])))
        f = (p * cell_p[None, :]).sum(1) - target
        df = (p * (1 - p) * cell_p[None, :]).sum(1)
        mu = mu - f / df
    return mu


# --------------------------------------------------------------------------- base patterns
def base_pattern(name, geo: Geography, seed=2026):
    rng = np.random.default_rng(seed)
    x, y = geo.coords[:, 0], geo.coords[:, 1]
    if name == "block":
        terciles = np.quantile(x, [1 / 3, 2 / 3])
        return np.where(x < terciles[0], 0.10, np.where(x < terciles[1], 0.25, 0.40))
    if name == "hotcold":
        anchors = np.where(geo.hot | geo.cold)[0]
        values = np.where(geo.hot[anchors], 0.40, 0.10)
        d = geo.dist[:, anchors]
        w = 1.0 / np.maximum(d, 1.0) ** 2
        base = (w * values[None, :]).sum(1) / w.sum(1)
        base[anchors] = values
        return base
    if name == "random":
        return rng.uniform(0.10, 0.40, geo.r)
    if name == "gradient":          # stand-in for the registry's observed pattern: higher in the south-east
        score = (x - x.mean()) / x.std() - (y - y.mean()) / y.std() + rng.normal(0, 0.6, geo.r)
        score = (score - score.min()) / (score.max() - score.min())
        return 0.10 + 0.30 * score
    raise ValueError(name)


def spatial_noise(geo: Geography, rng, sd=0.10):
    """Draw N(0, Sigma) with Sigma = (L + I)^{-1}, rescaled so the mean marginal sd is `sd`."""
    L = np.diag(geo.adjacency.sum(1)) - geo.adjacency
    Sigma = np.linalg.inv(L + np.eye(geo.r))
    Sigma = Sigma * (sd ** 2 / np.mean(np.diag(Sigma)))
    return rng.multivariate_normal(np.zeros(geo.r), Sigma)


def district_logits(base, geo, rng, X, cell_p, noise_sd=0.10):
    """Spatially smoothed target proportions and calibrated district intercepts."""
    logit = np.log(base / (1 - base)) + spatial_noise(geo, rng, noise_sd)
    target = 1 / (1 + np.exp(-logit))
    return target, calibrate_intercepts(target, X, cell_p)


# --------------------------------------------------------------------------- population
def generate_population(geo: Geography, rng, cell_p):
    """District and covariate-cell index for every synthetic case at t0."""
    N = np.maximum(np.round(geo.pop * CASES_PER_CAPITA).astype(int), 30)
    district = np.repeat(np.arange(geo.r), N)
    cell = rng.choice(len(cell_p), size=district.size, p=cell_p)
    return district, cell


def individual_probs(mu_by_area, area, cell, X, beta=BETA_TRUE):
    return 1 / (1 + np.exp(-(mu_by_area[area] + X[cell] @ beta)))


def sample_known_status(cell, prev_flag, rng):
    """Informative sampling: RR status known with probability depending on case type."""
    pi = np.where(prev_flag[cell] == 1, PI_KNOWN[1], PI_KNOWN[0])
    sampled = rng.uniform(size=cell.size) < pi
    w = 1.0 / pi
    w = w * sampled.sum() / w[sampled].sum()          # normalize weights to the sample size
    return sampled, w


# --------------------------------------------------------------------------- migration
def migration_structure(pattern, geo: Geography):
    """Eligible-origin mask (r,) and destination matrix P (r, r) with rows summing to one."""
    r = geo.r
    inv_d = 1.0 / np.maximum(geo.dist, 1.0)
    np.fill_diagonal(inv_d, 0.0)
    anchors = geo.hot | geo.cold
    if pattern == "neighbors":
        E, P = np.ones(r, bool), geo.contiguity.copy()
    elif pattern == "distance":
        E, P = np.ones(r, bool), inv_d.copy()
    elif pattern == "crisis_dist":
        E, P = geo.conflict.copy(), inv_d * (~geo.conflict)[None, :]
    elif pattern == "crisis_idp":
        E, P = geo.conflict.copy(), np.tile(geo.idp_weight, (r, 1))
    elif pattern == "into_urban":
        E, P = ~geo.urban, np.tile(geo.pop * geo.urban, (r, 1))
    elif pattern == "out_urban":
        E, P = geo.urban.copy(), inv_d * (~geo.urban)[None, :]
    elif pattern == "into_hotcold":
        E, P = ~anchors, np.tile(anchors.astype(float), (r, 1))
    elif pattern == "out_hotcold":
        E, P = anchors.copy(), inv_d * (~anchors)[None, :]
    else:
        raise ValueError(pattern)
    P = P.astype(float)
    np.fill_diagonal(P, 0.0)
    rows = P.sum(1)
    P[rows > 0] /= rows[rows > 0, None]
    E = E & (rows > 0)
    return E, P


def simulate_moves(district, eligible, P, level, rng):
    """Destination district for every case: eligible-origin cases move with probability `level`."""
    dest = district.copy()
    movers = eligible[district] & (rng.uniform(size=district.size) < level)
    idx = np.where(movers)[0]
    cum = np.cumsum(P[district[idx]], axis=1)
    u = rng.uniform(size=idx.size)
    dest[idx] = np.minimum((u[:, None] > cum).sum(1), P.shape[0] - 1)
    return dest


def composition_counts(origin, dest, r):
    """C[j, k] = number of post-migration residents of j who originated in k."""
    C = np.zeros((r, r))
    np.add.at(C, (dest, origin), 1.0)
    return C


# --------------------------------------------------------------------------- aggregation
def build_cells(area, cell, sampled, w, z, X, r):
    """Aggregate sampled individuals into district x covariate cells for the model, plus the
    population-level quantities needed for finite-population prediction."""
    K = X.shape[0]
    cid = area * K + cell
    n_all = np.bincount(cid, minlength=r * K).astype(float)
    n_s = np.bincount(cid[sampled], minlength=r * K).astype(float)
    y_s = np.bincount(cid[sampled], weights=z[sampled], minlength=r * K)
    w_cell = np.zeros(r * K)
    np.add.at(w_cell, cid[sampled], w[sampled])
    w_cell = np.where(n_s > 0, w_cell / np.maximum(n_s, 1), 1.0)
    cells = dict(X=np.tile(X, (r, 1)), area=np.repeat(np.arange(r), K), n=n_s, y=y_s, w=w_cell)
    pop_cells = dict(X=cells["X"], area=cells["area"], m=n_all - n_s,
                     y_obs=np.bincount(area[sampled], weights=z[sampled], minlength=r),
                     N=np.bincount(area, minlength=r).astype(float))
    return cells, pop_cells


def district_sample_counts(area, sampled, z, r):
    """Sampled positives and sample sizes per district (inputs to the power prior)."""
    y = np.bincount(area[sampled], weights=z[sampled], minlength=r)
    n = np.bincount(area[sampled], minlength=r).astype(float)
    return y, n
