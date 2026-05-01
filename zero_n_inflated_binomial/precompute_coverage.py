"""Precompute the coverage simulation that notebook 04 visualises.

Writes a pickle to coverage_sim.pkl so the notebook itself runs fast.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zni_rct import (
    fit_naive,
    simulate_rct_heterogeneous,
    simulate_rct_inflation_shift,
    simulate_rct_matched,
    simulate_rct_overdisp,
)

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
ro.r('suppressPackageStartupMessages(library(mcount))')

N = 30


def fit_naive_diff(y, tx, N):
    f = fit_naive(y, tx, N)
    p0 = 1.0 / (1.0 + np.exp(-f["intercept"]))
    p1 = 1.0 / (1.0 + np.exp(-(f["intercept"] + f["tx"])))
    se = f["tx_se"]
    g = p1 * (1 - p1)
    half = 1.96 * float(g * se)
    return p1 - p0, (p1 - p0 - half, p1 - p0 + half)


def fit_mznib_diff(y, tx, N, R=50, seed=1):
    df = pd.DataFrame({
        "y": y.astype(int),
        "tx": tx.astype(int),
        "N_i": np.full(len(y), N, dtype=int),
    })
    ro.globalenv["df"] = df
    ro.r(f"fit <- mznib(y ~ tx, data=df, N_i=df$N_i, R={R}, seed={seed}L)")
    alpha = np.asarray(ro.r("fit$alpha_estimates"))
    p0 = 1.0 / (1.0 + np.exp(-alpha[:, 0]))
    p1 = 1.0 / (1.0 + np.exp(-(alpha[:, 0] + alpha[:, 1])))
    diff = p1 - p0
    return float(diff.mean()), (float(np.quantile(diff, .025)),
                                float(np.quantile(diff, .975)))


SIMS = [
    ("matched", simulate_rct_matched, {}),
    ("π_0 shift", simulate_rct_inflation_shift, dict(pi0_ctl=0.20, pi0_tx=0.05)),
    ("non-responders", simulate_rct_heterogeneous,
        dict(p_tx_responder=0.75, p_tx_nonresp=0.40, prop_resp=0.5)),
    ("overdispersed", simulate_rct_overdisp, dict(kappa=4.0)),
]


def coverage_sim(stage_name, sim_fn, kwargs, n_reps=30, R=50):
    rows = []
    t0 = time.time()
    for r in range(n_reps):
        if r % 5 == 0:
            print(
                f"  {stage_name}: rep {r}/{n_reps}  "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
        rng = np.random.default_rng(1000 + r)
        y, tx = sim_fn(rng, **kwargs)
        emp = (y[tx == 1] / N).mean() - (y[tx == 0] / N).mean()
        nd, nci = fit_naive_diff(y, tx, N)
        md_, mci = fit_mznib_diff(y, tx, N, R=R, seed=r + 1)
        rows.append({
            "stage": stage_name,
            "naive est": nd, "naive lo": nci[0], "naive hi": nci[1],
            "mznib est": md_, "mznib lo": mci[0], "mznib hi": mci[1],
            "truth": emp,
        })
    return pd.DataFrame(rows)


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    print(f"Running coverage_sim with n_reps={n_reps}, R={R}", flush=True)
    pieces = []
    for name, sim, kw in SIMS:
        df = coverage_sim(name, sim, kw, n_reps=n_reps, R=R)
        pieces.append(df)
    all_reps = pd.concat(pieces, ignore_index=True)

    print("\nComputing population-level truth on a large draw per stage...",
          flush=True)
    truths = {}
    for name, sim, kw in SIMS:
        big = np.random.default_rng(987654321)
        y_big, tx_big = sim(big, n_per_arm=20000, **kw)
        truths[name] = (
            (y_big[tx_big == 1] / N).mean()
            - (y_big[tx_big == 0] / N).mean()
        )
        print(f"  {name:>16}: Δ = {truths[name]:+.4f}", flush=True)

    out = Path("coverage_sim.pkl")
    with out.open("wb") as f:
        pickle.dump({"reps": all_reps, "truths": truths,
                     "n_reps": n_reps, "R": R}, f)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
