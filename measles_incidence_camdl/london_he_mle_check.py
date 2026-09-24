"""Does the biennial attractor need He et al.'s full-search MLE?

The IF2 scout on London walks to R0 ~ 25, s0 ~ 0.065 — a ridge point with
the same effective reproduction number R0*s0 as He et al.'s published MLE
(R0 = 56.8, s0 = 0.0297; 56.8 x 0.0297 = 1.69 vs 25 x 0.065 = 1.6) but a
longer simulated inter-epidemic period. This script simulates London at
the published MLE (same covariates, same model) and compares the two
regimes against the observed series.

Run:  uv run python london_he_mle_check.py
"""

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_camdl_fig import DAYS_PER_YEAR, dominant_period

HERE = Path(__file__).parent
SIMS = HERE / "results" / "sims_he_mle"
SIMS.mkdir(parents=True, exist_ok=True)
N_SIMS = 20

HE_MLE = {
    # He, Ionides & King (2010) published London MLE (see
    # ../camdl_replication/params/he2010_london_mle.toml), with this
    # project's N0 (registry population at t0, Jan 1944).
    "R0": 56.8, "alpha": 0.976, "amplitude": 0.554, "iota": 2.9,
    "sigma_se": 2.816, "sigma": 0.0791, "gamma": 0.0832,
    "mu": 0.0000548, "cohort": 0.557, "rho": 0.488, "psi": 0.116,
    "N0": 2481706, "s0": 0.0297, "e0": 0.0000517, "i0": 0.0000514,
}


def simulate():
    ptoml = HERE / "results" / "london_he_mle_params.toml"
    with open(ptoml, "w") as f:
        f.write("# He et al. (2010) published London MLE\n")
        for k, v in HE_MLE.items():
            f.write(f"{k} = {v}\n")
    for seed in range(1, N_SIMS + 1):
        out = SIMS / f"london_seed{seed}.tsv"
        if out.exists():
            continue
        subprocess.run(
            ["camdl", "simulate", "london_seir.camdl", "--params", str(ptoml),
             "--backend", "chain_binomial", "--dt", "1.0",
             "--seed", str(seed), "--obs-only", str(out)],
            cwd=HERE, check=True, capture_output=True,
        )


def load_ensemble(directory, pattern):
    sims = []
    for f in sorted(Path(directory).glob(pattern)):
        s = pd.read_csv(f, sep="\t")
        sims.append(s[s["time"] > 0].set_index("time")["weekly_cases"])
    sim = pd.concat(sims, axis=1)
    sim.columns = range(sim.shape[1])
    return sim


def main():
    simulate()

    meta = pd.read_csv(HERE / "data" / "city_meta.tsv", sep="\t").set_index("city")
    t0 = meta.loc["london", "t0_decyear"]
    obs = pd.read_csv(HERE / "data" / "london_cases.tsv", sep="\t")
    obs["decyear"] = t0 + obs["time"] / DAYS_PER_YEAR
    dt_years = 7 / DAYS_PER_YEAR

    scout = load_ensemble(HERE / "results" / "sims_obs", "london_seed*.tsv")
    he = load_ensemble(SIMS, "london_seed*.tsv")

    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    panels = [
        ("observed (London 1944-58)", None, obs["weekly_cases"], "black"),
        ("IF2 scout estimate (R0=25, s0=0.064)", scout, None, "tab:blue"),
        ("He et al. published MLE (R0=56.8, s0=0.0297)", he, None, "tab:red"),
    ]
    rows = []
    for (title, sim, series, color), (ax_ts, ax_pg) in zip(panels, axes):
        if sim is not None:
            dy = t0 + sim.index.to_numpy() / DAYS_PER_YEAR
            lo_, hi_ = (sim.quantile(q, axis=1) for q in (0.05, 0.95))
            ax_ts.fill_between(dy, lo_ / 1000, hi_ / 1000, color=color, alpha=0.2, lw=0)
            ax_ts.plot(dy, sim[0] / 1000, color=color, lw=0.7)
            periods = [dominant_period(sim[c], dt_years)[1] for c in sim.columns]
            _, _, f, _ = dominant_period(sim[0], dt_years)
            pgs = [dominant_period(sim[c], dt_years)[3] for c in sim.columns]
            pgs = [p / p.max() for p in pgs]
            ax_pg.plot(1 / f, np.median(pgs, axis=0), color=color, lw=1)
            med = float(np.median(periods))
            q05, q95 = np.quantile(periods, [0.05, 0.95])
            rows.append((title, med, q05, q95))
            ax_pg.axvline(med, color=color, ls=":", lw=0.8)
        else:
            ax_ts.plot(obs["decyear"], series / 1000, color=color, lw=0.7)
            p_all, p_multi, f, p = dominant_period(series, dt_years)
            ax_pg.plot(1 / f, p / p.max(), color=color, lw=1)
            ax_pg.axvline(p_multi, color=color, ls=":", lw=0.8)
            rows.append((title, p_multi, np.nan, np.nan))
        ax_ts.set_title(title, fontsize=10)
        ax_ts.set_ylabel("cases ($\\times 10^{-3}$)", fontsize=8)
        ax_ts.set_ylim(0, 6)
        ax_pg.set_xlim(0.5, 5)
        ax_pg.set_ylabel("power", fontsize=8)
    axes[-1, 0].set_xlabel("year")
    axes[-1, 1].set_xlabel("period (years)")
    fig.suptitle("London: the biennial attractor and the R0·s0 ridge", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "results" / "london_he_mle_check.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for title, med, q05, q95 in rows:
        print(f"{title}: multiannual period {med:.2f}y ({q05:.2f}-{q95:.2f})")


if __name__ == "__main__":
    main()
