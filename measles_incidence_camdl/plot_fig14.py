"""Replica of Bjornstad's Epidemics Fig 1.4: fortnightly measles incidence
in four pre-vaccination cities, with vertical bars at annual intervals.

Run after prepare_data.py:  uv run python plot_fig14.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
fig14 = pd.read_csv(HERE / "data" / "fig14_biweekly.tsv", sep="\t")

PANELS = [
    ("liverpool", "Liverpool"),
    ("newyork", "New York"),
    ("london", "London"),
    ("baltimore", "Baltimore"),
]

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

for ax, (key, label) in zip(axes.flat, PANELS):
    d = fig14[fig14["city"] == key]
    ax.plot(d["decyear"], d["cases"] / 1000.0, color="black", lw=0.8)
    for yr in range(int(d["decyear"].min()) + 1, int(d["decyear"].max()) + 1):
        ax.axvline(yr, color="0.75", lw=0.5, zorder=0)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("year")
    ax.set_ylim(bottom=0)
    # two-digit year ticks like the book
    lo, hi = int(d["decyear"].min()), int(d["decyear"].max()) + 1
    ticks = [y for y in range(lo, hi + 1) if y % 2 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{y % 100:02d}" for y in ticks])

for ax in axes[:, 0]:
    ax.set_ylabel(r"Cases ($\times 10^{-3}$)")

fig.suptitle(
    "Fortnightly measles incidence, pre-vaccination era\n"
    "(after Bjornstad, Epidemics, Fig. 1.4)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "results" / "fig14_replica.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"wrote {out}")
