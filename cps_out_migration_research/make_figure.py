"""Build the headline figure from the pooled CPS results.

Panel A: observed monthly household-departure rate, native vs foreign-born (WA).
Panel B: the magnitude problem -- the observed annual foreign-born household-
         departure rate dwarfs the published foreign-born *emigration* rate, so
         emigration is an unidentifiable sliver inside ordinary mobility churn.

The emigration band (~1.0-1.5%/yr of the foreign-born) is from the published
literature (Van Hook et al. 2006; U.S. Census Bureau residual estimates), not
from the CPS panel itself -- which is exactly the point.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def annualize(m):
    return 1.0 - (1.0 - m) ** 12


# Published foreign-born emigration rate band (per year), from the literature:
# residual methods give ~0.9-1.5%/yr; the matched-CPS method (Van Hook et al.)
# gives higher, ~2.9%/yr on average and up to ~3.8%/yr for recent arrivals.
EMIG_LO, EMIG_HI = 0.010, 0.029


def main():
    d = pd.read_csv("outputs/pooled_summary.csv").set_index("label")
    wa = d.loc["Washington"]

    m_nat = wa["monthly_leaver_rate_wt_native"]
    m_fb = wa["monthly_leaver_rate_wt_foreign_born"]
    a_nat, a_fb = annualize(m_nat), annualize(m_fb)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.8),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- Panel A: observed monthly departure rate by nativity -------------
    bars = axA.bar(["U.S.-born", "Foreign-born"],
                   [m_nat * 100, m_fb * 100],
                   color=["#4c78a8", "#e45756"], edgecolor="white", width=0.6)
    for b, m, a in zip(bars, [m_nat, m_fb], [a_nat, a_fb]):
        axA.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                 f"{m*100:.2f}%/mo\n(~{a*100:.0f}%/yr)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axA.set_ylabel("Monthly rate of leaving a continuing household (%)")
    axA.set_ylim(0, m_fb * 100 * 1.4)
    axA.set_title("A. What the CPS panel actually sees\n"
                  "WA, 2024 (continuing households)",
                  fontsize=11, fontweight="bold")
    axA.text(0.5, -0.16,
             "Every one of these departures has an UNKNOWN destination.\n"
             "The CPS records no 'moved abroad' field.",
             transform=axA.transAxes, ha="center", fontsize=8.5,
             style="italic", color="#555")

    # ---- Panel B: churn vs emigration, on an annual basis -----------------
    # Stacked: of the observed foreign-born annual departure, at most the
    # emigration band could be international; the rest is domestic/death/etc.
    churn = a_fb * 100
    axB.bar(["Foreign-born\nhousehold departures\n(CPS panel, observed)"],
            [churn], color="#e45756", edgecolor="white", width=0.5,
            label="Observed departures (destination unknown)")
    # overlay the emigration band near the bottom
    axB.bar(["Foreign-born\nhousehold departures\n(CPS panel, observed)"],
            [EMIG_HI * 100], color="#54a24b", edgecolor="white", width=0.5,
            label="Plausibly international (published emigration rate)")
    axB.annotate(f"~{churn:.0f}% / yr leave the household\n"
                 "(mostly domestic moves, plus deaths,\n"
                 "institutional entry, family change)",
                 xy=(0, churn), xytext=(0.55, churn * 0.92),
                 fontsize=9, color="#a33",
                 arrowprops=dict(arrowstyle="->", color="#a33"))
    axB.annotate(f"Published foreign-born EMIGRATION:\n"
                 f"~{EMIG_LO*100:.1f}-{EMIG_HI*100:.1f}% / yr\n"
                 "(residual ~1-1.5%; matched-CPS ~2.9%)",
                 xy=(0, EMIG_HI * 100), xytext=(0.40, churn * 0.45),
                 fontsize=9, color="#264d1f",
                 arrowprops=dict(arrowstyle="->", color="#264d1f"))
    axB.set_ylabel("Annual rate (% of foreign-born)")
    axB.set_ylim(0, churn * 1.15)
    axB.set_title("B. Why the panel can't isolate emigration\n"
                  "The signal is ~8-20x too big",
                  fontsize=11, fontweight="bold")
    axB.set_xticks([0])
    axB.set_xticklabels(["Foreign-born household\ndepartures (observed)"],
                        fontsize=9)
    axB.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("The CPS sees who leaves a household — never where they go",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(top=0.80, wspace=0.28)
    plt.savefig("outputs/figure_departure.png", dpi=140, bbox_inches="tight")
    print(f"WA foreign-born: {m_fb*100:.2f}%/mo -> {a_fb*100:.1f}%/yr; "
          f"native: {m_nat*100:.2f}%/mo -> {a_nat*100:.1f}%/yr")
    print("wrote outputs/figure_departure.png")


if __name__ == "__main__":
    main()
