"""Post-fit pipeline: take each city's IF2 scout MLE (camdl's own
mle_params.toml, best chain by particle-filter loglik), forward-simulate
an ensemble at it, and tabulate the fitted parameters.

Run after the four `camdl fit run fit_<city>.toml --label <city>` jobs:
    uv run python run_postfit.py
"""

import glob
import shutil
import subprocess
import tomllib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SIMS = HERE / "results" / "sims_obs"
SIMS.mkdir(parents=True, exist_ok=True)

CITIES = ["london", "liverpool", "newyork", "baltimore"]
N_SIMS = 20
# sigma_se is estimated for the US cities, fixed for the UK ones; the MLE
# file carries its value either way.
ESTIMATED = ["R0", "amplitude", "rho", "s0", "sigma_se"]


def mle_params_path(city):
    # Best particle-filter loglik across all fit rounds for this city.
    mles = glob.glob(
        str(HERE / "results" / "fits" / f"fit_{city}-*" /
            "01-scout-*" / "seed_*" / "mle_params.toml")
    )
    assert mles, f"no fit results for {city}"

    def loglik(p):
        with open(p, "rb") as f:
            return tomllib.load(f)["provenance"]["log_likelihood"]

    return Path(max(mles, key=loglik))


def main():
    import sys

    cities = sys.argv[1:] or CITIES
    rows = []
    if (HERE / "results" / "fit_summary.tsv").exists():
        prev = pd.read_csv(HERE / "results" / "fit_summary.tsv", sep="\t")
        rows = [r for r in prev.to_dict("records") if r["city"] not in cities]
    for city in cities:
        ptoml = mle_params_path(city)
        with open(ptoml, "rb") as f:
            mle = tomllib.load(f)
        loglik = mle["provenance"]["log_likelihood"]
        # Keep the selected MLE in results/ (fit dirs are gitignored).
        shutil.copy(ptoml, HERE / "results" / f"{city}_mle_params.toml")

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
