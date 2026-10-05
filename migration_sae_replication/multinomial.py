"""Multinomial extension (Dupuis 2026, Chapter 4): four TB drug-resistance categories.

Model: multinomial-logit unit-level SAE with reference category DS-TB, category-specific
fixed effects and spatial random effects sharing one variance, optionally with the
migration matrix A applied to every category's random effects (A is shared across
categories, as in the dissertation). The dissertation's stick-breaking Gibbs sampler is
replaced by NUTS on the softmax likelihood, which needs no stick-breaking.

Data generation follows Section 4.4.1: the DS-TB share equals the binomial DS share and the
resistant remainder is split 0.6 / 0.2 / 0.2 into RR-TB, other DR-TB and pre-XDR/XDR-TB.

Usage: uv run python multinomial.py --study headtohead --reps 5 --bases block hotcold
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import itertools
import time
from runner import run_tasks
from pathlib import Path

import numpy as np
import pandas as pd

from geography import load_geography
from synthetic import (BASES, PATTERNS, LEVELS, covariate_cells, base_pattern, district_logits,
                       generate_population, individual_probs, sample_known_status,
                       migration_structure, simulate_moves, composition_counts)
from power_prior import power_prior_multinomial, draw_scalars

CATS = ["DS", "RR", "OtherDR", "preXDR_XDR"]
SPLIT = np.array([0.6, 0.2, 0.2])
H = len(CATS)
MCMC = dict(num_warmup=400, num_samples=400, num_chains=2)
PP_DRAWS = 1000
STUDY_ID = {"prediction": 11, "estimation": 12, "headtohead": 13}
_GEO = None


def geo_and_eig():
    global _GEO
    if _GEO is None:
        from models import laplacian_eigen
        geo = load_geography()
        _GEO = (geo, laplacian_eigen(geo.adjacency))
    return _GEO


def seed_for(*parts):
    return np.random.default_rng(list(parts))


# ----------------------------------------------------------------------------- data
def category_probs(p_rr):
    """(N,) resistant probability -> (N, H) category probabilities."""
    return np.column_stack([1 - p_rr] + [p_rr * s for s in SPLIT])


def draw_categories(P, rng):
    u = rng.uniform(size=P.shape[0])
    return np.minimum((u[:, None] > np.cumsum(P, 1)).sum(1), P.shape[1] - 1)


def build_cells_multi(area, cell, sampled, w, zcat, X, r):
    K = X.shape[0]
    cid = area * K + cell
    n_all = np.bincount(cid, minlength=r * K).astype(float)
    n_s = np.bincount(cid[sampled], minlength=r * K).astype(float)
    Y = np.zeros((r * K, H))
    np.add.at(Y, (cid[sampled], zcat[sampled]), 1.0)
    w_cell = np.zeros(r * K)
    np.add.at(w_cell, cid[sampled], w[sampled])
    w_cell = np.where(n_s > 0, w_cell / np.maximum(n_s, 1), 1.0)
    Yobs = np.zeros((r, H))
    np.add.at(Yobs, (area[sampled], zcat[sampled]), 1.0)
    cells = dict(X=np.tile(X, (r, 1)), area=np.repeat(np.arange(r), K), Y=Y, w=w_cell)
    pop = dict(X=cells["X"], area=cells["area"], m=n_all - n_s, Yobs=Yobs,
               N=np.bincount(area, minlength=r).astype(float))
    return cells, pop


def district_sample_counts_multi(area, sampled, zcat, r):
    Y = np.zeros((r, H))
    np.add.at(Y, (area[sampled], zcat[sampled]), 1.0)
    return Y


# ----------------------------------------------------------------------------- model
def multinomial_sae(X, area, Y, w, eigvals, eigvecs, A=None, hyper=None):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from models import HYPER_DEFAULT
    h = dict(HYPER_DEFAULT, **(hyper or {}))
    p = X.shape[1]
    r = eigvecs.shape[0]
    sigma_eta2 = numpyro.sample("sigma_eta2", dist.InverseGamma(h["a"], h["b"]))
    sigma_beta2 = numpyro.sample("sigma_beta2", dist.InverseGamma(h["c"], h["d"]))
    beta = numpyro.sample("beta", dist.Normal(0.0, jnp.sqrt(sigma_beta2)).expand([H - 1, p]).to_event(2))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([H - 1, r]).to_event(2))
    scale = jnp.sqrt(sigma_eta2) / jnp.sqrt(eigvals + 1.0 / sigma_eta2)
    eta = numpyro.deterministic("eta", (z * scale) @ eigvecs.T)          # (H-1, r)
    area_eff = eta if A is None else eta @ A.T
    psi = X @ beta.T + area_eff[:, area].T                               # (C, H-1)
    logits = jnp.concatenate([jnp.zeros((psi.shape[0], 1)), psi], axis=1)
    numpyro.factor("pseudo_loglik", jnp.sum(w[:, None] * Y * jax.nn.log_softmax(logits, axis=1)))


def fit_multinomial(cells, eig, A_draws=None, seed=0, **mcmc):
    import jax
    import jax.numpy as jnp
    eigvals, eigvecs = eig
    data = dict(X=jnp.asarray(cells["X"], jnp.float32), area=jnp.asarray(cells["area"], jnp.int32),
                Y=jnp.asarray(cells["Y"], jnp.float32), w=jnp.asarray(cells["w"], jnp.float32),
                eigvals=jnp.asarray(eigvals), eigvecs=jnp.asarray(eigvecs))
    from models import _cached_mcmc
    m = _cached_mcmc(multinomial_sae, "amatrix" if A_draws is not None else "naive",
                     mcmc["num_warmup"], mcmc["num_samples"])
    out = {"beta": [], "eta": [], "sigma_eta2": [], "diverging": [], "A": []}
    rng = jax.random.PRNGKey(seed)
    for c in range(mcmc["num_chains"]):
        rng, sub = jax.random.split(rng)
        A_c = None if A_draws is None else jnp.asarray(A_draws[c], jnp.float32)
        m.run(sub, A=A_c, **data)
        s = m.get_samples()
        for k in ("beta", "eta", "sigma_eta2"):
            out[k].append(np.asarray(s[k]))
        out["diverging"].append(int(np.asarray(m.get_extra_fields()["diverging"]).sum()))
        out["A"].append(None if A_draws is None else np.asarray(A_draws[c]))
    for k in ("beta", "eta", "sigma_eta2"):
        out[k] = np.stack(out[k])
    return out


def predict_multinomial(fit, pop, rng, A_draws=None):
    """Posterior predictive district category proportions, (chains*samples, r, H)."""
    X = np.asarray(pop["X"], np.float32)
    area = np.asarray(pop["area"])
    m = np.rint(np.asarray(pop["m"])).astype(np.int64)
    r = len(pop["N"])
    draws = []
    for c in range(fit["beta"].shape[0]):
        beta, eta = fit["beta"][c], fit["eta"][c]                        # (S, H-1, p), (S, H-1, r)
        A_c = fit["A"][c] if A_draws is None else A_draws[c]
        area_eff = eta if A_c is None else eta @ np.asarray(A_c).T       # (S, H-1, r)
        psi = np.einsum("cp,shp->sch", X, beta) + np.transpose(area_eff[:, :, area], (0, 2, 1))  # (S, C, H-1)
        logits = np.concatenate([np.zeros(psi.shape[:2] + (1,)), psi], axis=2)
        logits -= logits.max(2, keepdims=True)
        theta = np.exp(logits)
        theta /= theta.sum(2, keepdims=True)
        # sequential binomial draws for a multinomial with cell totals m
        remaining = np.broadcast_to(m, theta.shape[:2]).copy()
        counts = np.zeros(theta.shape)
        cum = 1.0 - np.cumsum(theta, 2) + theta
        for h in range(H - 1):
            ph = np.clip(theta[:, :, h] / np.maximum(cum[:, :, h], 1e-12), 0, 1)
            counts[:, :, h] = rng.binomial(remaining, ph)
            remaining = remaining - counts[:, :, h].astype(np.int64)
        counts[:, :, H - 1] = remaining
        dist_counts = np.zeros((theta.shape[0], r, H))
        for h in range(H):
            np.add.at(dist_counts[:, :, h].T, area, counts[:, :, h].T)
        draws.append((pop["Yobs"][None] + dist_counts) / pop["N"][None, :, None])
    return np.concatenate(draws, 0)


def predict_new_population_multi(fit, X, cell_counts, A_draws=None):
    r, K = cell_counts.shape
    pop = dict(X=np.tile(X, (r, 1)), area=np.repeat(np.arange(r), K), m=cell_counts.reshape(-1),
               Yobs=np.zeros((r, H)), N=np.maximum(cell_counts.sum(1), 1.0))
    return predict_multinomial(fit, pop, np.random.default_rng(0), A_draws=A_draws)


def records_multi(draws, truth, **meta):
    """Long rows per district x category, plus TVD per district (category = 'TVD')."""
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    est = draws.mean(0)
    r = truth.shape[0]
    frames = []
    for h, cat in enumerate(CATS):
        frames.append(pd.DataFrame(dict(district=np.arange(r), category=cat, truth=truth[:, h], est=est[:, h],
                                        lo=lo[:, h], hi=hi[:, h], sd=draws[:, :, h].std(0))))
    tvd = 0.5 * np.abs(est - truth).sum(1)
    frames.append(pd.DataFrame(dict(district=np.arange(r), category="TVD", truth=0.0, est=tvd, lo=np.nan, hi=np.nan, sd=np.nan)))
    df = pd.concat(frames, ignore_index=True)
    for k, v in meta.items():
        df[k] = v
    return df


def diag_of(fit, **meta):
    from models import rhat
    return dict(meta, divergences=int(sum(fit["diverging"])), rhat_eta_max=float(rhat(fit["eta"]).max()),
                rhat_beta_max=float(rhat(fit["beta"]).max()), sigma_eta2=float(fit["sigma_eta2"].mean()))


# ----------------------------------------------------------------------------- studies
def _population(base, rng, geo, X, cell_p):
    district, cell = generate_population(geo, rng, cell_p)
    target, mu = district_logits(base_pattern(base, geo), geo, rng, X, cell_p)
    return district, cell, mu


def prediction_task(base, rep, patterns, levels):
    geo, eig = geo_and_eig()
    r = geo.r
    X, cell_p, prev = covariate_cells()
    b = BASES.index(base)
    rng = seed_for(STUDY_ID["prediction"], b, rep)
    district, cell, mu = _population(base, rng, geo, X, cell_p)
    P = category_probs(individual_probs(mu, district, cell, X))
    z0 = draw_categories(P, rng)
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop = build_cells_multi(district, cell, sampled, w, z0, X, r)
    t = time.time()
    fit = fit_multinomial(cells, eig, seed=int(rng.integers(2**31)), **MCMC)
    base_draws = predict_multinomial(fit, pop, rng)
    diags = [diag_of(fit, study="prediction", base=base, rep=rep, method="naive", seconds=time.time() - t)]
    Y_s = district_sample_counts_multi(district, sampled, z0, r)
    Y_full = pop["N"][:, None] * base_draws.mean(0)
    out = []
    for pattern, level in itertools.product(patterns, levels):
        E, Pm = migration_structure(pattern, geo)
        rs = seed_for(STUDY_ID["prediction"], b, rep, PATTERNS.index(pattern), list(LEVELS).index(level))
        dest = simulate_moves(district, E, Pm, LEVELS[level], rs)
        z1 = draw_categories(P, rs)
        N1 = np.maximum(np.bincount(dest, minlength=r), 1)
        truth = np.zeros((r, H))
        np.add.at(truth, (dest, z1), 1.0)
        truth /= N1[:, None]
        C = composition_counts(district, dest, r)
        meta = dict(study="prediction", base=base, pattern=pattern, level=level, rep=rep)
        out.append(records_multi(base_draws, truth, method="naive", **meta))
        out.append(records_multi(power_prior_multinomial(base_draws, Y_full, C, rs, PP_DRAWS), truth, method="pp", **meta))
        out.append(records_multi(power_prior_multinomial(base_draws, Y_s, C, rs, PP_DRAWS), truth, method="pp_sample", **meta))
    return pd.concat(out, ignore_index=True), diags


def estimation_task(base, pattern, level, rep):
    geo, eig = geo_and_eig()
    r = geo.r
    X, cell_p, prev = covariate_cells()
    b, pi, li = BASES.index(base), PATTERNS.index(pattern), list(LEVELS).index(level)
    rng = seed_for(STUDY_ID["estimation"], b, pi, li, rep)
    district, cell, mu = _population(base, rng, geo, X, cell_p)
    E, Pm = migration_structure(pattern, geo)
    dest = simulate_moves(district, E, Pm, LEVELS[level], rng)
    C = composition_counts(district, dest, r)
    A = C / C.sum(1, keepdims=True)
    P = category_probs(individual_probs(A @ mu, dest, cell, X))
    z = draw_categories(P, rng)
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop = build_cells_multi(dest, cell, sampled, w, z, X, r)
    truth = np.zeros((r, H))
    np.add.at(truth, (dest, z), 1.0)
    truth /= pop["N"][:, None]
    meta = dict(study="estimation", base=base, pattern=pattern, level=level, rep=rep)
    out, diags = [], []
    for method, A_counts in (("naive", None), ("amatrix", C)):
        t = time.time()
        A_draws = None if A_counts is None else draw_scalars(A_counts, rng, MCMC["num_chains"], scheme="dirichlet")
        fit = fit_multinomial(cells, eig, A_draws=A_draws, seed=int(rng.integers(2**31)), **MCMC)
        out.append(records_multi(predict_multinomial(fit, pop, rng), truth, method=method, **meta))
        diags.append(diag_of(fit, **meta, method=method, seconds=time.time() - t))
    return pd.concat(out, ignore_index=True), diags


def headtohead_task(base, rep, patterns, levels):
    geo, eig = geo_and_eig()
    r = geo.r
    X, cell_p, prev = covariate_cells()
    K = X.shape[0]
    b = BASES.index(base)
    rng = seed_for(STUDY_ID["headtohead"], b, rep)
    district, cell, mu = _population(base, rng, geo, X, cell_p)
    P = category_probs(individual_probs(mu, district, cell, X))
    z0 = draw_categories(P, rng)
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop = build_cells_multi(district, cell, sampled, w, z0, X, r)
    t = time.time()
    fit = fit_multinomial(cells, eig, seed=int(rng.integers(2**31)), **MCMC)
    base_draws = predict_multinomial(fit, pop, rng)
    diags = [diag_of(fit, study="headtohead", base=base, rep=rep, method="naive", seconds=time.time() - t)]
    Y_s = district_sample_counts_multi(district, sampled, z0, r)
    Y_full = pop["N"][:, None] * base_draws.mean(0)
    out = []
    for pattern, level in itertools.product(patterns, levels):
        E, Pm = migration_structure(pattern, geo)
        rs = seed_for(STUDY_ID["headtohead"], b, rep, PATTERNS.index(pattern), list(LEVELS).index(level))
        dest = simulate_moves(district, E, Pm, LEVELS[level], rs)
        C = composition_counts(district, dest, r)
        order = np.argsort(dest, kind="stable")
        starts = np.searchsorted(dest[order], np.arange(r + 1))
        truth = np.zeros((r, H))
        for j in range(r):
            res = order[starts[j]:starts[j + 1]]
            if res.size:
                inherited = z0[res][rs.integers(0, res.size, size=res.size)]
                truth[j] = np.bincount(inherited, minlength=H) / res.size
        cell_counts = np.zeros((r, K))
        np.add.at(cell_counts, (dest, cell), 1.0)
        meta = dict(study="headtohead", base=base, pattern=pattern, level=level, rep=rep)
        out.append(records_multi(base_draws, truth, method="naive", **meta))
        out.append(records_multi(power_prior_multinomial(base_draws, Y_full, C, rs, PP_DRAWS), truth, method="pp", **meta))
        t = time.time()
        A_draws = draw_scalars(C, rs, MCMC["num_chains"], scheme="dirichlet")
        fitA = fit_multinomial(cells, eig, A_draws=A_draws, seed=int(rs.integers(2**31)), **MCMC)
        diags.append(diag_of(fitA, **meta, method="amatrix", seconds=time.time() - t))
        out.append(records_multi(predict_new_population_multi(fitA, X, cell_counts), truth, method="amatrix", **meta))
        out.append(records_multi(predict_new_population_multi(fit, X, cell_counts, A_draws=[A_draws[0]] * MCMC["num_chains"]),
                                 truth, method="amatrix_proj", **meta))
    return pd.concat(out, ignore_index=True), diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDY_ID), required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bases", nargs="*", default=BASES)
    ap.add_argument("--patterns", nargs="*", default=PATTERNS)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reps = range(args.reps)
    if args.study == "prediction":
        tasks = [(prediction_task, (b, k, args.patterns, args.levels)) for b in args.bases for k in reps]
    elif args.study == "estimation":
        tasks = [(estimation_task, (b, p, l, k)) for b in args.bases for p in args.patterns for l in args.levels for k in reps]
    else:
        tasks = [(headtohead_task, (b, k, args.patterns, args.levels)) for b in args.bases for k in reps]
    out = Path(args.out or f"results/raw/multinomial_{args.study}.parquet")
    chunk = {'prediction': 10, 'estimation': 40, 'headtohead': 10}[args.study]
    run_tasks(tasks, out, args.workers, chunk)


if __name__ == "__main__":
    main()
