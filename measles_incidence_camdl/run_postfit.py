"""Post-fit pipeline: take each city's IF2 scout MLE (camdl's own
mle_params.toml, best chain by particle-filter loglik), forward-simulate
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


def mle_params_path(city):
    fit_dirs = sorted(glob.glob(str(HERE / "results" / "fits" / f"fit_{city}-*")))
    assert fit_dirs, f"no fit directory for {city}"
    mles = glob.glob(fit_dirs[-1] + "/01-scout-*/seed_*/mle_params.toml")
    assert len(mles) == 1, mles
    return Path(mles[0])


def main():
    rows = []
    for city in CITIES:
        ptoml = mle_params_path(city)
        with open(ptoml, "rb") as f:
            mle = tomllib.load(f)
        loglik = mle["provenance"]["log_likelihood"]

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

        rows.append({"city": city, "loglik": loglik,
                     **{p: float(mle[p]) for p in ESTIMATED}})
        print(f"{city}: loglik={loglik:.1f} "
              + " ".join(f"{p}={mle[p]:.3g}" for p in ESTIMATED))

    pd.DataFrame(rows).to_csv(HERE / "results" / "fit_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
