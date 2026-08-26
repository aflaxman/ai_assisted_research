"""Post-fit pipeline: extract each city's IF2 scout estimate, forward-simulate
an ensemble at it, and tabulate the fitted parameters.

Run after the four `camdl fit run fit_<city>.toml --label <city>` jobs:
    uv run python run_postfit.py
"""

import glob
import subprocess
import tomllib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SIMS = HERE / "results" / "sims_obs"
SIMS.mkdir(parents=True, exist_ok=True)

CITIES = ["london", "liverpool", "newyork", "baltimore"]
N_SIMS = 20
ESTIMATED = ["R0", "amplitude", "rho", "s0"]


def best_final_params(city):
    """Final-iteration parameters from the best-loglik chain."""
    fit_dirs = sorted(glob.glob(str(HERE / "results" / "fits" / f"fit_{city}-*")))
    assert fit_dirs, f"no fit directory for {city}"
    traces = glob.glob(fit_dirs[-1] + "/01-scout-*/seed_*/chain_*/parameter_traces.tsv")
    best = None
    for tr in traces:
        df = pd.read_csv(tr, sep="\t", comment="#")
        last = df.iloc[-1]
        if best is None or last["loglik"] > best["loglik"]:
            best = last
    return best


def main():
    rows = []
    for city in CITIES:
        best = best_final_params(city)
        with open(HERE / f"fit_{city}.toml", "rb") as f:
            fit_cfg = tomllib.load(f)
        params = dict(fit_cfg["fixed"])
        for p in ESTIMATED:
            params[p] = float(best[p])

        ptoml = HERE / "results" / f"{city}_fitted_params.toml"
        with open(ptoml, "w") as f:
            f.write(f"# {city}: IF2 scout point estimate (best of 4 chains, final iteration)\n")
            f.write(f"# loglik = {best['loglik']:.1f}\n")
            for k, v in params.items():
                f.write(f"{k} = {v}\n")

        for seed in range(1, N_SIMS + 1):
            out = SIMS / f"{city}_seed{seed}.tsv"
            if out.exists():
                continue
            subprocess.run(
                ["camdl", "simulate", f"{city}_seir.camdl",
                 "--params", str(ptoml),
                 "--backend", "chain_binomial", "--dt", "1.0",
                 "--seed", str(seed), "--obs-only", str(out)],
                cwd=HERE, check=True, capture_output=True,
            )

        rows.append({"city": city, "loglik": float(best["loglik"]),
                     **{p: float(best[p]) for p in ESTIMATED}})
        print(f"{city}: loglik={best['loglik']:.1f} "
              + " ".join(f"{p}={best[p]:.3g}" for p in ESTIMATED))

    pd.DataFrame(rows).to_csv(HERE / "results" / "fit_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
