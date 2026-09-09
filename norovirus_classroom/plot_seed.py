"""Two-panel comparison of the two ways to start the outbreak.

A: R0 posterior under the day-1 head-count model vs the one-seed model
B: posterior for the seed day (one-seed model), with the weekend marked

Inputs:
  results/posterior_draws.tsv        main model (fit.toml)
  results/posterior_draws_seed.tsv   one-seed model (fit_seed.toml)

Usage: uv run python plot_seed.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

main = pd.read_csv("results/posterior_draws.tsv", sep="\t")
seed = pd.read_csv("results/posterior_draws_seed.tsv", sep="\t")
for d in (main, seed):
    d["R0"] = d["beta"] * d["d_school"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
fig.patch.set_facecolor(SURFACE)
for ax in (axA, axB):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

# -- A: R0 under the two models ---------------------------------------------
bins = np.arange(0, 10.25, 0.25)
axA.hist(main["R0"], bins=bins, density=True, color=BLUE, alpha=0.75,
         label="head counts on day 1 free (main model)")
axA.hist(seed["R0"], bins=bins, density=True, color=ORANGE, alpha=0.65,
         label="one seed kid, day estimated")
axA.axvline(1.0, color=INK2, lw=1.0)
for d, color, y in ((main, BLUE, 0.92), (seed, ORANGE, 0.82)):
    lo, med, hi = d["R0"].quantile([0.025, 0.5, 0.975])
    axA.plot([0.36], [y - 0.02], transform=axA.transAxes, marker="s", color=color, ms=8,
             clip_on=False)
    axA.annotate(f"median {med:.1f}   (95%: {lo:.1f} to {hi:.1f})   P(R0>1) = {(d['R0'] > 1).mean():.2f}",
                 xy=(0.39, y), xycoords="axes fraction", ha="left", va="top", fontsize=9,
                 color=INK2)
axA.set_xlim(0, 10)
axA.set_xlabel("R0", color=INK2, fontsize=9.5)
axA.set_ylabel("Posterior density", color=INK2, fontsize=9.5)
axA.set_title("A  R0 depends on how you think it started", loc="left", color=INK, fontsize=11)
axA.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK2,
           bbox_to_anchor=(1.0, 0.55))

# -- B: seed day --------------------------------------------------------------
days = np.arange(-13, 1)
counts = seed["t_seed"].round().value_counts(normalize=True).reindex(days, fill_value=0)
# day 1 is a Monday, so day 0 is Sunday, -1 Saturday, -6 the Monday before, ...
labels = {-13: "Mon", -12: "Tue", -11: "Wed", -10: "Thu", -9: "Fri", -8: "Sat", -7: "Sun",
          -6: "Mon", -5: "Tue", -4: "Wed", -3: "Thu", -2: "Fri", -1: "Sat", 0: "Sun"}
for wk in (-8, -1):  # Sat + Sun
    axB.axvspan(wk - 0.5, wk + 1.5, color=GRID, alpha=0.5, lw=0)
axB.bar(days, counts.values, width=0.8, color=ORANGE, alpha=0.85)
axB.set_xticks(days)
axB.set_xticklabels([f"{labels[d]}\n{d}" for d in days], fontsize=8)
axB.set_xlabel("Day the first infectious kid came to school (day 1 = first count, Monday)",
               color=INK2, fontsize=9.5)
axB.set_ylabel("Posterior probability", color=INK2, fontsize=9.5)
axB.set_title("B  When did it start? (one-seed model; shaded = weekend)", loc="left",
              color=INK, fontsize=11)

fig.savefig("results/seed_comparison.png", dpi=150, facecolor=SURFACE)
print("wrote results/seed_comparison.png")
for name, d in (("main", main), ("seed", seed)):
    lo, med, hi = d["R0"].quantile([0.025, 0.5, 0.975])
    print(f"{name}: R0 median {med:.2f} ({lo:.2f}, {hi:.2f}), P(R0>1)={(d['R0'] > 1).mean():.2f}")
lo, med, hi = seed["t_seed"].quantile([0.025, 0.5, 0.975])
print(f"t_seed median {med:.1f} ({lo:.1f}, {hi:.1f})")
