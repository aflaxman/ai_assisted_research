"""figure_occind.png: standardized gross emigration by major occupation and
industry group (employed foreign-born adults), 2024->25 vs pooled 2019->24.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_N = 50

TITLES = {"occ_major": "Major occupation group",
          "ind_major": "Major industry group"}
LBL = {
    "mgmt_biz_fin": "Mgmt/business/finance", "professional": "Professional",
    "service": "Service", "sales": "Sales", "office_admin": "Office/admin",
    "farm_fish_forest": "Farming/fish/forestry",
    "construction": "Construction", "install_repair": "Install/repair",
    "production": "Production", "transport_moving": "Transport/moving",
    "agriculture": "Agriculture", "mining": "Mining",
    "manufacturing": "Manufacturing", "trade": "Wholesale/retail",
    "transport_utilities": "Transport/utilities",
    "information": "Information", "financial": "Financial",
    "prof_biz_services": "Prof/business svcs",
    "educ_health": "Education/health",
    "leisure_hospitality": "Leisure/hospitality",
    "other_services": "Other services", "public_admin": "Public admin",
}


def main():
    s = pd.read_csv("outputs/strata_rates.csv")
    cur = s[s["pair"] == "2024->2025"]
    base = s[s["pair"] != "2024->2025"]
    base_agg = (base.groupby(["stratifier", "level"])
                .apply(lambda d: np.average(d["e_adj"], weights=d["n"]),
                       include_groups=False)
                .rename("e_base").reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, key in zip(axes, ["occ_major", "ind_major"]):
        c = cur[(cur["stratifier"] == key) & (cur["n"] >= MIN_N)].copy()
        b = base_agg[base_agg["stratifier"] == key].set_index("level")
        c["e_base"] = c["level"].map(b["e_base"])
        c = c.sort_values("e_adj")
        y = np.arange(len(c))
        ax.axvline(0, color="#333", lw=0.8)
        for yi, (_, r) in zip(y, c.iterrows()):
            ax.plot([r["e_base"] * 100, r["e_adj"] * 100], [yi, yi],
                    color="#ccc", lw=1.4, zorder=1)
        ax.scatter(c["e_base"] * 100, y, s=42, color="#9ecae9", zorder=2,
                   label="2019→24 pooled")
        ax.scatter(c["e_adj"] * 100, y, s=52, color="#f58518", zorder=3,
                   label="2024→25")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{LBL.get(r.level, r.level)}  (n={r.n})"
                            for r in c.itertuples()], fontsize=8.5)
        ax.set_xlabel("Gross emigration rate (%), standardized")
        ax.set_title(TITLES[key], fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        if key == "occ_major":
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Foreign-born emigration signal by occupation and industry "
                 "(employed adults, CPS matching method)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
             "Occupation/industry measured in year t for employed persons; "
             "job characteristics are pre-departure. Standardized rates; "
             "subgroup CIs are several points wide — read orderings and "
             "changes vs baseline, not levels.",
             ha="center", fontsize=8.3, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("outputs/figure_occind.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_occind.png")


if __name__ == "__main__":
    main()
