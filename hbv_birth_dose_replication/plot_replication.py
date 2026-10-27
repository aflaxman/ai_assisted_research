"""Summary figure for the replication (``results/hbv_replication.png``).

Left: vaccine expenditure per 1000 births by delivery strategy for every
strategy set modelled for all GBD regions, with the resulting birth-dose
coverage written above each bar (the paper's Figure 2, aggregate panel).

Right: the cost-effectiveness plane of the cheapest route to 90% coverage in
each GBD region (the paper's Figure 3, "optimised 90% coverage" points):
additional lifetime cost against DALYs averted per 1000 births, with the
interquartile range over the uncertainty draws when ``run_uncertainty.py``
has been run.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import hbv_model as m

matplotlib.use("Agg")
RESULTS = m.ROOT / "results"

# Validated categorical palette, slots 1-5, light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

det = pd.read_csv(RESULTS / "scenarios_per_1000.tsv", sep="\t")
psa_path = RESULTS / "uncertainty_summary.tsv"
psa = pd.read_csv(psa_path, sep="\t") if psa_path.exists() else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.35})

# --- left: expenditure stack, all GBD regions ----------------------------
order = ["baseline", "optimised_baseline", "scenario1", "scenario2", "scenario3", "scenario4", "optimised_90"]
labels = ["Base-\nline", "Optimised\nbaseline", "Scen. 1", "Scen. 2", "Scen. 3", "Scen. 4", "90%\noptimised"]
g = det[det.region == "All GBD Regions"].set_index("scenario").loc[order]
cols = ["vc_fac_cc", "vc_fac_ctc", "vc_com_cc", "vc_com_ctc", "vc_com_cpad"]
names = ["Facility, cold chain", "Facility, CTC", "Community, cold chain", "Community, CTC", "Community, CPAD"]
bottom = pd.Series(0.0, index=g.index)
x = range(len(order))
for col, name, color in zip(cols, names, SERIES):
    ax1.bar(x, g[col], bottom=bottom, color=color, width=0.62, label=name, edgecolor=SURFACE, linewidth=1.5)
    bottom = bottom + g[col]
for i, s in enumerate(order):
    ax1.text(i, bottom[s] + 150, f"{g.loc[s, 'coverage']:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Vaccine expenditure, US$ per 1000 births")
ax1.set_ylim(0, bottom.max() * 1.15)
ax1.set_title("All GBD regions: expenditure by strategy\n(birth-dose coverage above each bar)", fontsize=10, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="upper left")
ax1.yaxis.grid(True, color=GRID, linewidth=0.8)
ax1.set_axisbelow(True)
for side in ["top", "right"]:
    ax1.spines[side].set_visible(False)

# --- right: cost-effectiveness plane of the 90% optimum --------------------
regions = [r for r in det.region.unique() if r != "All GBD Regions"]
short = {
    "Central and Eastern Europe and Central Asia": "C/E Europe & C Asia",
    "East Asia and Pacific": "SE Asia, E Asia\n& Oceania",
    "Latin America and Caribbean": "Latin America\n& Caribbean",
    "North Africa and Middle East": "N Africa &\nMiddle East",
    "South Asia": "South Asia",
    "Sub-Saharan Africa": "Sub-Saharan Africa",
}
# Label placement (offset in points, alignment), chosen by eye to avoid collisions.
LABEL_POS = {
    "Central and Eastern Europe and Central Asia": (9, 7, "left", "bottom"),
    "East Asia and Pacific": (-9, 0, "right", "center"),
    "Latin America and Caribbean": (9, 0, "left", "center"),
    "North Africa and Middle East": (-9, 0, "right", "center"),
    "South Asia": (9, 0, "left", "center"),
    "Sub-Saharan Africa": (9, 0, "left", "center"),
}
pt = det[det.scenario == "optimised_90"].set_index("region")
ax2.axvline(0, color=AXIS, linestyle="--", linewidth=1)
for r in regions:
    xv, yv = pt.loc[r, "additional_cost"], pt.loc[r, "dalys_averted_as_published"]
    if psa is not None:
        q = psa[(psa.region == r) & (psa.scenario == "optimised_90")].iloc[0]
        xv, yv = q["additional_cost_median"], q["dalys_averted_as_published_median"]
        ax2.plot([q["additional_cost_q25"], q["additional_cost_q75"]], [yv, yv], color=SERIES[0], linewidth=1.5, alpha=0.6)
        ax2.plot([xv, xv], [q["dalys_averted_as_published_q25"], q["dalys_averted_as_published_q75"]],
                 color=SERIES[0], linewidth=1.5, alpha=0.6)
    ax2.plot(xv, yv, "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    dx, dy, ha, va = LABEL_POS[r]
    ax2.annotate(short[r], (xv, yv), xytext=(dx, dy), textcoords="offset points",
                 fontsize=8, color=INK2, ha=ha, va=va)
ax2.set_xlabel("Additional lifetime cost, US$ per 1000 births (negative = cost saving)")
ax2.set_ylabel("DALYs averted per 1000 births (authors' YLD scaling)")
sub = "median and IQR of 1000 draws" if psa is not None else "point estimates"
ax2.set_title(f"Cheapest route to 90% coverage, by region\n({sub})", fontsize=10, loc="left")
ax2.yaxis.grid(True, color=GRID, linewidth=0.8)
ax2.set_axisbelow(True)
xmin, xmax = ax2.get_xlim()
ax2.set_xlim(xmin - 1800, xmax + 2500)
for side in ["top", "right"]:
    ax2.spines[side].set_visible(False)

fig.text(0.01, 0.01, "Seaman et al. (2020) replicated with camdl; data: authors' published inputs.",
         fontsize=7.5, color=MUTED)
fig.savefig(RESULTS / "hbv_replication.png", dpi=160, bbox_inches="tight")
print("wrote", RESULTS / "hbv_replication.png")
