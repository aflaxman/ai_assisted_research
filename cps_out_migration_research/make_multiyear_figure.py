"""Figures for the multi-year extension.

figure_multiyear.png:
  Panel A -- national gross emigration per ASEC pair 2019->20 ... 2024->25,
             with full-uncertainty error bars and the Van Hook circa-2000
             benchmark band.
  Panel B -- states: pooled 2019->24 baseline vs 2024->25, for states with
             enough sample.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_STATE_N = 100


def main():
    nat = pd.read_csv("outputs/multiyear_summary.csv")
    st = pd.read_csv("outputs/state_rates.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.8),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- Panel A: national time series ------------------------------------
    x = np.arange(len(nat))
    g = nat["gross_e"] * 100
    se = nat["gross_se_full"] * 100
    axA.axhspan(2.9, 3.8, color="#cfe3f0", alpha=0.6, zorder=0,
                label="Van Hook circa-2000 (net-gross band)")
    axA.axhline(0, color="#333", lw=0.8)
    axA.errorbar(x, g, yerr=1.96 * se, fmt="o-", color="#4c78a8", lw=2,
                 capsize=4, markersize=7, label="Gross emigration ±95% CI")
    axA.plot(x, nat["net_e"] * 100, "s--", color="#e45756", markersize=6,
             lw=1.2, label="Net emigration")
    for xi, gi in zip(x, g):
        axA.text(xi, gi + 0.35, f"{gi:.1f}", ha="center", fontsize=8.5,
                 fontweight="bold", color="#2a5480")
    axA.set_xticks(x)
    axA.set_xticklabels([p.replace("->", "→") for p in nat["pair"]],
                        fontsize=8.5, rotation=20)
    covid = nat.index[nat["pair"] == "2020->2021"]
    if len(covid):
        axA.annotate("pandemic-era files:\ncollection disrupted",
                     xy=(covid[0], g.iloc[covid[0]]),
                     xytext=(covid[0] + 0.3, g.max() * 0.85), fontsize=7.5,
                     color="#666", arrowprops=dict(arrowstyle="->", color="#999"))
    axA.set_ylabel("Annual foreign-born emigration rate (%)")
    axA.set_title("A. National, six ASEC pairs\n(CPS matching method, "
                  "standardized)", fontsize=11, fontweight="bold")
    axA.legend(fontsize=7.5, loc="lower left")

    # ---- Panel B: states baseline vs 2024->25 ------------------------------
    cur = st[(st["pair"] == "2024->2025") & (st["n"] >= MIN_STATE_N)].copy()
    base = st[st["pair"] != "2024->2025"].copy()
    base_agg = (base.groupby("state")
                .apply(lambda d: np.average(d["e_adj"], weights=d["n"]),
                       include_groups=False)
                .rename("e_base"))
    cur = cur.merge(base_agg, on="state", how="left")
    cur = cur.sort_values("e_adj", ascending=True)
    y = np.arange(len(cur))
    axB.axvline(0, color="#333", lw=0.8)
    for yi, (_, r) in zip(y, cur.iterrows()):
        axB.plot([r["e_base"] * 100, r["e_adj"] * 100], [yi, yi],
                 color="#bbb", lw=1.5, zorder=1)
    axB.scatter(cur["e_base"] * 100, y, s=45, color="#9ecae9", zorder=2,
                label="2019→24 pooled baseline")
    axB.scatter(cur["e_adj"] * 100, y, s=55, color="#f58518", zorder=3,
                label="2024→25")
    axB.set_yticks(y)
    axB.set_yticklabels([f"{r.state}  (n={r.n})" for r in cur.itertuples()],
                        fontsize=8.5)
    axB.set_xlabel("Gross emigration rate (%), standardized")
    axB.set_title(f"B. States with n ≥ {MIN_STATE_N} foreign-born records "
                  "in 2024→25", fontsize=11, fontweight="bold")
    axB.legend(fontsize=8, loc="lower right")
    axB.grid(axis="x", alpha=0.25)

    fig.suptitle("Foreign-born emigration by the CPS matching method, "
                 "2019–2025 — more years, more states",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "Standardized (composition-adjusted) rates; national 95% CIs are "
             "±3–4pp per pair, state estimates noisier still. The 2024→25 "
             "elevation mixes real departures with immigrant survey-response "
             "decline (see README caveats).",
             ha="center", fontsize=8, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("outputs/figure_multiyear.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_multiyear.png")


if __name__ == "__main__":
    main()
