"""figure_strata.png: standardized gross emigration by subgroup, 2024->25
vs the pooled 2019->24 baseline, for the stratifiers the ASEC supports:
citizenship, region of birth, education (25+), age, sex, household income
quartile, race/ethnicity, and duration in the U.S.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    ("citizenship", "Citizenship", ["naturalized", "noncitizen"]),
    ("region_birth", "Region of birth",
     ["mexico", "other_americas", "asia", "europe", "africa"]),
    ("educ4", "Education (age 25+)", ["lt_hs", "hs", "some_col", "ba_plus"]),
    ("agegrp", "Age group",
     ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]),
    ("sex", "Sex", ["male", "female"]),
    ("income_q", "Household income quartile (FB)",
     ["q1_low", "q2", "q3", "q4_high"]),
    ("race_eth", "Race / ethnicity",
     ["hispanic", "nh_white", "nh_black", "nh_asian"]),
    ("duration", "Years in the U.S.", ["0-4 yrs", "5-9 yrs", "10+ yrs"]),
]

LABELS = {
    "naturalized": "Naturalized\ncitizen", "noncitizen": "Not a\ncitizen",
    "mexico": "Mexico", "other_americas": "Other\nAmericas", "asia": "Asia",
    "europe": "Europe", "africa": "Africa",
    "lt_hs": "< HS", "hs": "HS", "some_col": "Some\ncollege", "ba_plus": "BA+",
    "q1_low": "Q1\n(low)", "q2": "Q2", "q3": "Q3", "q4_high": "Q4\n(high)",
    "hispanic": "Hispanic", "nh_white": "NH\nwhite", "nh_black": "NH\nblack",
    "nh_asian": "NH\nAsian",
}


def main():
    s = pd.read_csv("outputs/strata_rates.csv")
    cur = s[s["pair"] == "2024->2025"]
    base = s[s["pair"] != "2024->2025"]
    base_agg = (base.groupby(["stratifier", "level"])
                .apply(lambda d: np.average(d["e_adj"], weights=d["n"]),
                       include_groups=False)
                .rename("e_base").reset_index())

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6))
    for ax, (key, title, order) in zip(axes.flat, PANELS):
        c = cur[cur["stratifier"] == key].set_index("level")
        b = base_agg[base_agg["stratifier"] == key].set_index("level")
        levels = [l for l in order if l in c.index]
        xs = np.arange(len(levels))
        w = 0.38
        eb = [100 * b["e_base"].get(l, np.nan) for l in levels]
        ec = [100 * c.loc[l, "e_adj"] for l in levels]
        ns = [int(c.loc[l, "n"]) for l in levels]
        ax.bar(xs - w / 2, eb, w, color="#9ecae9", label="2019→24 pooled")
        ax.bar(xs + w / 2, ec, w, color="#f58518", label="2024→25")
        ax.axhline(0, color="#333", lw=0.7)
        for x, v, n in zip(xs, ec, ns):
            ax.text(x + w / 2, v + (0.25 if v >= 0 else -0.6), f"{v:.1f}",
                    ha="center", fontsize=7.5, fontweight="bold",
                    color="#8a4a0c")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS.get(l, l) for l in levels], fontsize=7.5)
        ax.set_title(f"{title}\n(2024→25 n: {', '.join(map(str, ns))})",
                     fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        if ax is axes.flat[0]:
            ax.legend(fontsize=7.5)
        if ax in axes[:, 0]:
            ax.set_ylabel("Gross emigration (%)", fontsize=8.5)

    fig.suptitle("Standardized foreign-born emigration by subgroup — "
                 "2024→25 vs the 2019→24 pooled baseline",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
             "Composition-standardized rates from the CPS matching method; "
             "subgroups not in the standardization models are mediated by "
             "age/sex/ethnicity/education/tenure composition. National 95% "
             "CIs are ±3–4pp; subgroup CIs wider. Interpret contrasts, not "
             "levels.",
             ha="center", fontsize=8.5, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("outputs/figure_strata.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_strata.png")


if __name__ == "__main__":
    main()
