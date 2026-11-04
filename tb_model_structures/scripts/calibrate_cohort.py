#!/usr/bin/env python3
"""Calibrate structures to the cohort likelihood (MAP + Laplace) and save results.

NUTS is reliable for the spectrum models but struggles with the non-identifiable
classic structures (a posterior ridge) and is slow; MAP optimisation with a
Laplace approximation and weakly-informative regularising priors is fast, robust,
and yields a direct identifiability diagnostic (tbmodels/calibrate_map.py).  The
SP-full MAP reproduces the NUTS/Horton 2023 result.

Usage:
    uv run python scripts/calibrate_cohort.py --all
    uv run python scripts/calibrate_cohort.py SP-full C E

Per structure -> data/derived/posteriors/<name>/:
    samples.npz         Laplace draws (constrained space)
    summary.csv         median + 95% interval per parameter
    identifiability.csv Hessian eigenvalues + per-parameter prior->posterior contraction
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derived" / "posteriors"


def run(name: str) -> dict:
    from tbmodels.registry import get
    from tbmodels.calibrate_map import calibrate_map, _spec
    s = get(name)
    r = calibrate_map(s, n_restarts=10)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "samples.npz", **r["draws"])

    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "map", "median", "q2.5", "q97.5"])
        for k in r["names"]:
            v = r["draws"][k]
            med, lo, hi = np.percentile(v, [50, 2.5, 97.5])
            w.writerow([k, round(r["map"][k], 5), round(med, 5), round(lo, 5), round(hi, 5)])

    # identifiability: per-parameter prior->posterior contraction (unconstrained sd)
    cov = r["cov"]
    with open(d / "identifiability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "prior_sd", "posterior_sd", "contraction"])
        for i, k in enumerate(r["names"]):
            prior_sd = _spec(k)[3]
            post_sd = float(np.sqrt(max(cov[i, i], 0.0)))
            contraction = 1.0 - min(post_sd / prior_sd, 1.0)
            w.writerow([k, round(prior_sd, 4), round(post_sd, 4), round(contraction, 4)])
    # global diagnostics
    evals = r["hessian_evals"]
    cond = float(evals.max() / max(evals.min(), 1e-9))
    with open(d / "identifiability.csv", "a", newline="") as f:
        f.write(f"# hessian_eigenvalues,{','.join(str(round(e,4)) for e in evals)}\n")
        f.write(f"# condition_number,{round(cond,1)}\n")
        f.write(f"# neg_log_post,{round(r['neg_log_post'],3)}\n")
    print(f"{name}: MAP done, cond={cond:.1f}, "
          f"min_contraction={min(1 - min(np.sqrt(cov[i,i])/_spec(k)[3],1.0) for i,k in enumerate(r['names'])):.2f}")
    return r


def main(argv):
    if "--all" in argv:
        from tbmodels.registry import all_structures
        names = list(all_structures())
    else:
        names = [a for a in argv[1:] if not a.startswith("--")]
    for name in names:
        run(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
