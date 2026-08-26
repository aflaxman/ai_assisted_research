"""Final analysis figure: for each Fig 1.4 city, observed vs camdl-simulated
incidence at the IF2 scout estimate, plus periodograms of both.

Also writes results/periodicity.tsv with each city's dominant inter-epidemic
period, observed and simulated (median over the ensemble).

Run after run_postfit.py:  uv run python plot_camdl_fig.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram

HERE = Path(__file__).parent
SIMS = HERE / "results" / "sims_obs"
DAYS_PER_YEAR = 365.25

CITIES = [
    ("london", "London", "weekly_cases", 7),
    ("liverpool", "Liverpool", "weekly_cases", 7),
    ("newyork", "New York", "biweekly_cases", 14),
    ("baltimore", "Baltimore", "biweekly_cases", 14),
]


def dominant_period(x, dt_years, lo=0.5, hi=8.0):
    """Dominant period (years) from a linear-detrended sqrt-scale periodogram.

    Returns (overall peak period, multiannual peak period (>1.2 y), f, p).
    The multiannual peak separates inter-epidemic spacing from the annual
    seasonal component, which otherwise dominates cities (like London
    1944-50) that mix annual and multiannual regimes in one window.
    """
    x = np.sqrt(np.asarray(x, float).clip(min=0))
    f, p = periodogram(x, fs=1.0 / dt_years, detrend="linear")
    keep = (f > 1.0 / hi) & (f < 1.0 / lo)
    f, p = f[keep], p[keep]
    multi = f < 1.0 / 1.2
    return (
        1.0 / f[np.argmax(p)],
        1.0 / f[multi][np.argmax(p[multi])],
        f,
        p,
    )


def main():
    meta = pd.read_csv(HERE / "data" / "city_meta.tsv", sep="\t").set_index("city")
    fitsum = pd.read_csv(HERE / "results" / "fit_summary.tsv", sep="\t").set_index("city")

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), gridspec_kw={"width_ratios": [2.2, 1]})
    rows = []

    for (city, label, col, emit), (ax_ts, ax_pg) in zip(CITIES, axes):
        obs = pd.read_csv(HERE / "data" / f"{city}_cases.tsv", sep="\t")
        t0 = meta.loc[city, "t0_decyear"]
        obs["decyear"] = t0 + obs["time"] / DAYS_PER_YEAR
        dt_years = emit / DAYS_PER_YEAR

        sims = []
        for f in sorted(SIMS.glob(f"{city}_seed*.tsv")):
            s = pd.read_csv(f, sep="\t")
            s = s[s["time"] > 0]
            sims.append(s.set_index("time")[col])
        sim = pd.concat(sims, axis=1)
        sim.columns = range(len(sims))
        sim_dy = t0 + sim.index.to_numpy() / DAYS_PER_YEAR

        # -- time series panel
        lo_, med, hi_ = (sim.quantile(q, axis=1) for q in (0.05, 0.5, 0.95))
        ax_ts.fill_between(sim_dy, lo_ / 1000, hi_ / 1000, color="tab:blue", alpha=0.25, lw=0)
        ax_ts.plot(sim_dy, med / 1000, color="tab:blue", lw=0.7, label="camdl SEIR (5-95%)")
        ax_ts.plot(obs["decyear"], obs[col] / 1000, color="black", lw=0.7, label="observed")
        r0, amp, rho = (fitsum.loc[city, k] for k in ("R0", "amplitude", "rho"))
        ax_ts.set_title(
            f"{label} — R0={r0:.0f}, amplitude={amp:.2f}, rho={rho:.2f}", fontsize=10
        )
        ax_ts.set_ylabel(f"{col.split('_')[0]} cases ($\\times 10^{{-3}}$)", fontsize=8)
        ax_ts.set_ylim(bottom=0)
        if city == "london":
            ax_ts.legend(fontsize=7, loc="upper right")

        # -- periodogram panel
        pd_obs, pd_obs_multi, f_obs, p_obs = dominant_period(obs[col], dt_years)
        ax_pg.plot(1 / f_obs, p_obs / p_obs.max(), color="black", lw=1, label="observed")
        sim_periods, sim_multi, pgs = [], [], []
        for c in sim.columns:
            pd_sim, pd_sim_multi, f_sim, p_sim = dominant_period(sim[c], dt_years)
            sim_periods.append(pd_sim)
            sim_multi.append(pd_sim_multi)
            pgs.append(p_sim / p_sim.max())
        ax_pg.plot(1 / f_sim, np.median(pgs, axis=0), color="tab:blue", lw=1, label="simulated")
        ax_pg.set_xlim(0.5, 5)
        ax_pg.set_xlabel("period (years)", fontsize=8)
        ax_pg.set_ylabel("relative power", fontsize=8)
        ax_pg.axvline(pd_obs_multi, color="black", ls=":", lw=0.8)
        ax_pg.axvline(np.median(sim_multi), color="tab:blue", ls=":", lw=0.8)
        if city == "london":
            ax_pg.legend(fontsize=7)

        rows.append({
            "city": city,
            "observed_dominant_yr": round(pd_obs, 2),
            "observed_multiannual_yr": round(pd_obs_multi, 2),
            "sim_dominant_median_yr": round(float(np.median(sim_periods)), 2),
            "sim_multiannual_median_yr": round(float(np.median(sim_multi)), 2),
            "sim_multiannual_q05": round(float(np.quantile(sim_multi, 0.05)), 2),
            "sim_multiannual_q95": round(float(np.quantile(sim_multi, 0.95)), 2),
        })

    axes[-1, 0].set_xlabel("year")
    fig.suptitle(
        "Recurrent measles epidemics, observed vs seasonally forced stochastic SEIR\n"
        "fit with camdl (IF2 scout) — cities of Bjornstad's Epidemics Fig. 1.4",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE / "results" / "fig14_camdl_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    per = pd.DataFrame(rows)
    per.to_csv(HERE / "results" / "periodicity.tsv", sep="\t", index=False)
    print(per.to_string(index=False))


if __name__ == "__main__":
    main()
