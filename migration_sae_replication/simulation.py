"""Simulation runner for the binomial studies of Dupuis (2026), re-implemented with numpyro.

Studies
  prediction  : Chapter 2 DGP. Cases get an RR probability at t0, migrate, and outcomes are
                redrawn at t1. Methods: migration-naive SAE prediction (the t0 finite-population
                posterior) vs. the power-prior adjustment (plus two variants).
  estimation  : Chapter 3 DGP. District effects are mixed through the realized composition
                matrix A before outcomes are drawn once. Methods: naive SAE vs. A-matrix SAE.
  headtohead  : binomial analogue of the method-neutral comparison in Chapter 4: t1 cases inherit
                the RR status of a random t0 case living in their post-migration district.

Each task writes long-format rows (district-level posterior mean, 95% interval, truth).
Usage: uv run python simulation.py --study prediction --reps 10 --workers 4
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import itertools
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from geography import load_geography
from synthetic import (BASES, PATTERNS, LEVELS, covariate_cells, base_pattern, district_logits,
                       generate_population, individual_probs, sample_known_status, build_cells,
                       migration_structure, simulate_moves, composition_counts, district_sample_counts)
from power_prior import power_prior_binomial, draw_scalars

MCMC = dict(num_warmup=400, num_samples=400, num_chains=2)
PP_DRAWS = 1000
STUDY_ID = {"prediction": 1, "estimation": 2, "headtohead": 3}

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


def records(draws, truth, **meta):
    """District-level summaries of posterior draws (S, r) against the truth (r,)."""
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    df = pd.DataFrame(dict(district=np.arange(len(truth)), truth=truth, est=draws.mean(0),
                           lo=lo, hi=hi, sd=draws.std(0)))
    for k, v in meta.items():
        df[k] = v
    return df


def fit_and_predict(cells, pop_cells, eig, rng, seed, A_counts=None):
    from models import fit_sae, predict_district_proportions, rhat
    A_draws = None if A_counts is None else draw_scalars(A_counts, rng, MCMC["num_chains"], scheme="dirichlet")
    fit = fit_sae(cells, eig, A_draws=A_draws, seed=int(seed), **MCMC)
    draws = predict_district_proportions(fit, pop_cells, rng)
    diag = dict(rhat_eta_max=float(rhat(fit["eta"]).max()), rhat_beta_max=float(rhat(fit["beta"]).max()),
                divergences=int(sum(fit["diverging"])), sigma_eta2=float(fit["sigma_eta2"].mean()))
    return fit, draws, diag


def predict_new_population(fit, X, cell_counts, A_draws=None):
    """Posterior predictive district proportions for a population with no observed outcomes.
    cell_counts (r, K): residents per district and covariate cell."""
    from models import predict_district_proportions
    r, K = cell_counts.shape
    pop = dict(X=np.tile(X, (r, 1)), area=np.repeat(np.arange(r), K), m=cell_counts.reshape(-1),
               y_obs=np.zeros(r), N=np.maximum(cell_counts.sum(1), 1.0))
    return predict_district_proportions(fit, pop, np.random.default_rng(0), A_draws=A_draws)


# ----------------------------------------------------------------------------- studies
def prediction_task(base, rep, patterns, levels):
    geo, eig = geo_and_eig()
    r = geo.r
    X, cell_p, prev = covariate_cells()
    b = BASES.index(base)
    rng = seed_for(STUDY_ID["prediction"], b, rep)
    district, cell = generate_population(geo, rng, cell_p)
    target, mu = district_logits(base_pattern(base, geo), geo, rng, X, cell_p)
    p = individual_probs(mu, district, cell, X)
    z0 = (rng.uniform(size=p.size) < p).astype(float)
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop_cells = build_cells(district, cell, sampled, w, z0, X, r)
    t = time.time()
    fit, base_draws, diag = fit_and_predict(cells, pop_cells, eig, rng, rng.integers(2**31))
    diag.update(study="prediction", base=base, rep=rep, method="naive", seconds=time.time() - t)
    y_s, n_s = district_sample_counts(district, sampled, z0, r)
    N = pop_cells["N"]
    y_full = N * base_draws.mean(0)                     # observed + imputed positives in the whole area
    out = []
    for pattern, level in itertools.product(patterns, levels):
        E, P = migration_structure(pattern, geo)
        rs = seed_for(STUDY_ID["prediction"], b, rep, PATTERNS.index(pattern), list(LEVELS).index(level))
        dest = simulate_moves(district, E, P, LEVELS[level], rs)
        z1 = rs.uniform(size=p.size) < p
        N1 = np.bincount(dest, minlength=r)
        truth = np.bincount(dest, weights=z1, minlength=r) / np.maximum(N1, 1)
        C = composition_counts(district, dest, r)
        meta = dict(study="prediction", base=base, pattern=pattern, level=level, rep=rep)
        out.append(records(base_draws, truth, method="naive", **meta))
        out.append(records(power_prior_binomial(base_draws, y_full, N, C, rs, PP_DRAWS), truth, method="pp", **meta))
        out.append(records(power_prior_binomial(base_draws, y_s, n_s, C, rs, PP_DRAWS), truth, method="pp_sample", **meta))
        out.append(records(power_prior_binomial(base_draws, y_full, N, C, rs, PP_DRAWS, double_count=False),
                           truth, method="pp_tempered", **meta))
    return pd.concat(out, ignore_index=True), [diag]


def estimation_task(base, pattern, level, rep):
    geo, eig = geo_and_eig()
    r = geo.r
    X, cell_p, prev = covariate_cells()
    b, pi, li = BASES.index(base), PATTERNS.index(pattern), list(LEVELS).index(level)
    rng = seed_for(STUDY_ID["estimation"], b, pi, li, rep)
    district, cell = generate_population(geo, rng, cell_p)
    target, mu = district_logits(base_pattern(base, geo), geo, rng, X, cell_p)
    E, P = migration_structure(pattern, geo)
    dest = simulate_moves(district, E, P, LEVELS[level], rng)
    C = composition_counts(district, dest, r)
    A = C / C.sum(1, keepdims=True)
    p = individual_probs(A @ mu, dest, cell, X)          # migration-smoothed district effects
    z = (rng.uniform(size=p.size) < p).astype(float)
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop_cells = build_cells(dest, cell, sampled, w, z, X, r)
    truth = np.bincount(dest, weights=z, minlength=r) / pop_cells["N"]
    meta = dict(study="estimation", base=base, pattern=pattern, level=level, rep=rep)
    diags, out = [], []
    for method, A_counts in (("naive", None), ("amatrix", C)):
        t = time.time()
        fit, draws, diag = fit_and_predict(cells, pop_cells, eig, rng, rng.integers(2**31), A_counts)
        diag.update(meta, method=method, seconds=time.time() - t)
        diags.append(diag)
        out.append(records(draws, truth, method=method, **meta))
    return pd.concat(out, ignore_index=True), diags


def headtohead_task(base, rep, patterns, levels):
    geo, eig = geo_and_eig()
    r = geo.r
    X, cell_p, prev = covariate_cells()
    K = X.shape[0]
    b = BASES.index(base)
    rng = seed_for(STUDY_ID["headtohead"], b, rep)
    district, cell = generate_population(geo, rng, cell_p)
    target, mu = district_logits(base_pattern(base, geo), geo, rng, X, cell_p)
    p = individual_probs(mu, district, cell, X)
    z0 = (rng.uniform(size=p.size) < p).astype(float)
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop_cells = build_cells(district, cell, sampled, w, z0, X, r)
    t = time.time()
    fit, base_draws, diag = fit_and_predict(cells, pop_cells, eig, rng, rng.integers(2**31))
    diag.update(study="headtohead", base=base, rep=rep, method="naive", seconds=time.time() - t)
    diags = [diag]
    y_s, n_s = district_sample_counts(district, sampled, z0, r)
    N = pop_cells["N"]
    y_full = N * base_draws.mean(0)
    out = []
    for pattern, level in itertools.product(patterns, levels):
        E, P = migration_structure(pattern, geo)
        rs = seed_for(STUDY_ID["headtohead"], b, rep, PATTERNS.index(pattern), list(LEVELS).index(level))
        dest = simulate_moves(district, E, P, LEVELS[level], rs)
        C = composition_counts(district, dest, r)
        # method-neutral truth: each t1 case in district j inherits the status of a random t0 case
        # now living in j (random infector pairing within the post-migration population)
        order = np.argsort(dest, kind="stable")
        starts = np.searchsorted(dest[order], np.arange(r + 1))
        truth = np.zeros(r)
        for j in range(r):
            res = order[starts[j]:starts[j + 1]]
            if res.size:
                truth[j] = z0[res][rs.integers(0, res.size, size=res.size)].mean()
        cell_counts = np.zeros((r, K))
        np.add.at(cell_counts, (dest, cell), 1.0)
        meta = dict(study="headtohead", base=base, pattern=pattern, level=level, rep=rep)
        out.append(records(base_draws, truth, method="naive", **meta))
        out.append(records(power_prior_binomial(base_draws, y_full, N, C, rs, PP_DRAWS), truth, method="pp", **meta))
        out.append(records(power_prior_binomial(base_draws, y_s, n_s, C, rs, PP_DRAWS), truth, method="pp_sample", **meta))
        # A-matrix model fitted to the t0 sample, predicting the post-migration population
        t = time.time()
        A_draws = draw_scalars(C, rs, MCMC["num_chains"], scheme="dirichlet")
        from models import fit_sae, rhat
        fitA = fit_sae(cells, eig, A_draws=A_draws, seed=int(rs.integers(2**31)), **MCMC)
        diags.append(dict(meta, method="amatrix", seconds=time.time() - t, divergences=int(sum(fitA["diverging"])),
                          rhat_eta_max=float(rhat(fitA["eta"]).max()), rhat_beta_max=float(rhat(fitA["beta"]).max()),
                          sigma_eta2=float(fitA["sigma_eta2"].mean())))
        out.append(records(predict_new_population(fitA, X, cell_counts), truth, method="amatrix", **meta))
        # projection variant: naive fit at t0, random effects pushed through A for the t1 population
        out.append(records(predict_new_population(fit, X, cell_counts, A_draws=[A_draws[0]] * MCMC["num_chains"]),
                           truth, method="amatrix_proj", **meta))
    return pd.concat(out, ignore_index=True), diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDY_ID), required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bases", nargs="*", default=BASES)
    ap.add_argument("--patterns", nargs="*", default=PATTERNS)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reps = range(args.rep_start, args.rep_start + args.reps)
    if args.study == "prediction":
        tasks = [(prediction_task, (b, k, args.patterns, args.levels)) for b in args.bases for k in reps]
    elif args.study == "estimation":
        tasks = [(estimation_task, (b, p, l, k)) for b in args.bases for p in args.patterns for l in args.levels for k in reps]
    else:
        tasks = [(headtohead_task, (b, k, args.patterns, args.levels)) for b in args.bases for k in reps]
    out = Path(args.out or f"results/raw/{args.study}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames, diags = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(fn, *a) for fn, a in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            df, dg = f.result()
            frames.append(df)
            diags.extend(dg)
            if i % max(1, len(futs) // 20) == 0 or i == len(futs):
                print(f"{i}/{len(futs)} tasks done, {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(out, index=False)
    pd.DataFrame(diags).to_csv(out.with_suffix(".diagnostics.csv"), index=False)
    print(f"wrote {out} ({len(res)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
