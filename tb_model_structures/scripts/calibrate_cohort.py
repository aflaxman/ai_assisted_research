#!/usr/bin/env python3
"""Calibrate structures to the cohort likelihood and save posteriors.

Usage:
    uv run python scripts/calibrate_cohort.py SP-full            # one structure
    uv run python scripts/calibrate_cohort.py --all             # every structure
    uv run python scripts/calibrate_cohort.py --all --quick     # short chains

Outputs per structure to data/derived/posteriors/<name>/:
    samples.npz          posterior draws
    summary.csv          median + 95% CrI + rhat per parameter
    pointwise_loglik.npz per-observation log-likelihoods (for LOO/WAIC)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "derived" / "posteriors"


def run(name: str, quick: bool) -> dict:
    import arviz as az
    import numpyro
    numpyro.set_host_device_count(2)
    from tbmodels.registry import get
    from tbmodels.calibrate import calibrate, pointwise_loglik

    s = get(name)
    nw, ns = (300, 300) if quick else (800, 800)
    mcmc = calibrate(s, num_warmup=nw, num_samples=ns, num_chains=2, seed=1)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    idata = az.from_numpyro(mcmc)
    rhat = az.rhat(idata)

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "samples.npz", **samples)

    rows = []
    for k in s.param_names:
        v = samples[k]
        med, lo, hi = np.percentile(v, [50, 2.5, 97.5])
        rh = float(rhat[k].max()) if k in rhat else np.nan
        rows.append((k, med, lo, hi, rh))
    import csv
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "median", "q2.5", "q97.5", "rhat"])
        w.writerows(rows)

    pw = pointwise_loglik(s, samples)
    np.savez(d / "pointwise_loglik.npz", **pw)

    max_rhat = float(max(float(rhat[v].max()) for v in rhat.data_vars))
    print(f"{name}: done, max rhat={max_rhat:.3f} -> {d}")
    return {"name": name, "max_rhat": max_rhat}


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    if "--all" in argv:
        from tbmodels.registry import all_structures
        names = list(all_structures())
    else:
        names = [a for a in argv[1:] if not a.startswith("--")]
    for name in names:
        run(name, quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
