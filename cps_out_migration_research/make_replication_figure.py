"""Figure: Van Hook et al. (2006) circa-2000 benchmarks vs. this replication
on ASEC 2023->24 and 2024->25.

Panel A: gross and net foreign-born emigration rates, three eras.
Panel B: the duration-of-residence gradient, circa 2000 vs 2024->25 --
         the 2000 pattern (recent arrivals leave most) has inverted.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    r = pd.read_csv("outputs/replication_summary.csv")
    p1 = r[r["pair"] == "2023->2024"].iloc[0]
    p2 = r[r["pair"] == "2024->2025"].iloc[0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # ---- Panel A: gross & net across eras --------------------------------
    eras = ["Van Hook et al.\ncirca 2000\n(matched CPS 1996-2003)",
            "This replication\n2023 " + "→" + " 2024",
            "This replication\n2024 " + "→" + " 2025"]
    gross = [3.8, 100 * p1["gross_e"], 100 * p2["gross_e"]]
    net = [2.9, 100 * p1["net_e"], 100 * p2["net_e"]]
    x = np.arange(3)
    w = 0.36
    bars1 = axA.bar(x - w / 2, gross, w, label="Gross emigration",
                    color="#4c78a8", edgecolor="white")
    bars2 = axA.bar(x + w / 2, net, w, label="Net emigration",
                    color="#e45756", edgecolor="white")
    for bars in (bars1, bars2):
        for b in bars:
            v = b.get_height()
            axA.text(b.get_x() + b.get_width() / 2,
                     v + (0.12 if v >= 0 else -0.32),
                     f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axA.axhline(0, color="#333", lw=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels(eras, fontsize=8.5)
    axA.set_ylabel("Annual rate (% of foreign-born)")
    axA.set_title("A. The method replicates — and the answer moved",
                  fontsize=11, fontweight="bold")
    axA.legend(fontsize=8.5, loc="upper left")
    axA.text(0.5, -0.24,
             "Honest SEs ≈ ±2pp per pair. 2023→2024: ≈0 during the "
             "immigration surge.\n2024→2025: elevated — but foreign-born "
             "attrition itself was flat (34.9% both pairs); the signal is the\n"
             "foreign-born diverging from the second-generation control, "
             "mixing real departures with response shifts.",
             transform=axA.transAxes, ha="center", fontsize=8,
             style="italic", color="#444")

    # ---- Panel B: duration gradient --------------------------------------
    cats = ["0-4 yrs\nin U.S.", "5-9 yrs", "10+ yrs"]
    circa2000 = [6.5, 5.0, 2.5]
    now = [100 * p2["e_in_us_0_4"], 100 * p2["e_in_us_5_9"],
           100 * p2["e_in_us_10plus"]]
    x = np.arange(3)
    b1 = axB.bar(x - w / 2, circa2000, w, label="circa 2000 (gross)",
                 color="#9ecae9", edgecolor="white")
    b2 = axB.bar(x + w / 2, now, w, label="2024→2025 (gross)",
                 color="#f58518", edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            axB.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                     f"{b.get_height():.1f}%", ha="center", fontsize=9,
                     fontweight="bold")
    axB.set_xticks(x)
    axB.set_xticklabels(cats)
    axB.set_ylabel("Annual gross emigration rate (%)")
    axB.set_title("B. The standardized duration gradient flipped\n"
                  "(suggestive, not conclusive — see text)",
                  fontsize=11, fontweight="bold")
    axB.legend(fontsize=8.5)
    axB.text(0.5, -0.26,
             "In 2000, recently arrived immigrants left at 2.6× the rate of "
             "settled ones (circular migration).\nIn 2024→2025 the "
             "standardized rates flip — but raw panel attrition is still far "
             "higher for recent arrivals\n(52% vs 31%), and the contrast is "
             "within the ~±2pp error bars.",
             transform=axB.transAxes, ha="center", fontsize=8,
             style="italic", color="#444")

    fig.suptitle("Replicating the CPS matching method for foreign-born "
                 "emigration (Van Hook et al. 2006)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig("outputs/figure_replication.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_replication.png")


if __name__ == "__main__":
    main()
