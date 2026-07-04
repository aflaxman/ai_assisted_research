"""figure_smallarea.png: empirical-Bayes state estimates of foreign-born
gross emigration, 2024->25, for every state with >=20 eligible foreign-born
adult records. Shrunk estimates with 95% intervals; direct estimates shown
as open circles to display how much shrinkage each state gets.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    d = pd.read_csv("outputs/state_small_area.csv")
    cur = d[d["window"] == "2024->2025"].sort_values("e_shrunk")
    base = d[d["window"] == "baseline_2019_24"].set_index("state")

    fig, ax = plt.subplots(figsize=(9, 0.28 * len(cur) + 2.2))
    y = np.arange(len(cur))
    mu = np.average(cur["e_shrunk"], weights=1 / cur["post_sd"] ** 2)

    ax.axvline(0, color="#333", lw=0.8)
    ax.axvline(mu * 100, color="#4c78a8", lw=1.2, ls="--",
               label=f"national (μ ≈ {mu*100:.1f}%)")
    # baseline shrunk (2019-24) as small grey squares
    bvals = [100 * base["e_shrunk"].get(s, np.nan) for s in cur["state"]]
    ax.scatter(bvals, y, marker="s", s=18, color="#b8b8b8", zorder=2,
               label="2019→24 baseline (shrunk)")
    # direct 2024->25 as open circles
    ax.scatter(cur["e_direct"] * 100, y, facecolors="none",
               edgecolors="#f58518", s=40, zorder=3, label="2024→25 direct")
    # shrunk with 95% intervals
    ax.errorbar(cur["e_shrunk"] * 100, y, xerr=1.96 * cur["post_sd"] * 100,
                fmt="o", color="#d1495b", markersize=5, capsize=2.4, lw=1.1,
                zorder=4, label="2024→25 shrunk ± 95%")
    labels = []
    for r in cur.itertuples():
        w = f"{r.shrink_weight_own*100:.0f}%"
        labels.append(f"{r.state}  (n={r.n_adults}, own {w})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    hl = cur.reset_index(drop=True).index[cur["state"].values == "WA"]
    for yi in hl:
        ax.get_yticklabels()[yi].set_fontweight("bold")
        ax.axhspan(yi - 0.45, yi + 0.45, color="#fff3cd", zorder=0)
    ax.set_xlabel("Gross foreign-born emigration rate (%), 2024→25")
    ax.set_title("Small-area (empirical-Bayes) state estimates\n"
                 "direct state attrition shrunk toward the national mean",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.005,
             "Direct estimate = national rate + state deviation in raw "
             "foreign-born adult attrition / (1 − m). 'own' = weight on the "
             "state's own signal; the rest shrinks to μ. Assumes state "
             "attrition deviations reflect emigration, not local response "
             "culture.",
             ha="center", fontsize=7.2, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig("outputs/figure_smallarea.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_smallarea.png")


if __name__ == "__main__":
    main()
