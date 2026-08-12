"""Plot camdl replication of He et al. (2010) London measles.

Top panel: observed weekly case notifications vs one camdl forward
simulation at the published MLE. Bottom panel: particle-filter
log-likelihood replicates (camdl) against the pomp reference mean.

Usage: uv run python plot_replication.py
Inputs: data/he2010_london_cases.tsv, results/sim_obs.tsv,
        results/pfilter_logliks.tsv
Output: results/he2010_replication.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Validated categorical palette (light mode), slots 1-2
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#1a1a19"
MUTED = "#6f6e66"

POMP_REF_MEAN = -5827.354958556549  # pomp 2000 particles x 20 replicates
POMP_REF_SD = 12.327855805476533

obs = pd.read_csv("data/he2010_london_cases.tsv", sep="\t")
sim = pd.read_csv("results/sim_obs.tsv", sep="\t")
pf = pd.read_csv("results/pfilter_logliks.tsv", sep="\t")

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 6.5), height_ratios=[2, 1], constrained_layout=True
)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e5e4dd", linewidth=0.8)
    ax.set_axisbelow(True)

# -- Top: observed vs simulated weekly cases ---------------------------------
years_obs = 1944 + obs["time"] / 365.25
years_sim = 1944 + sim["time"] / 365.25
ax1.plot(years_obs, obs["weekly_cases"], color=BLUE, lw=1.2, label="Observed (London 1944–65)")
ax1.plot(years_sim, sim["weekly_cases"], color=ORANGE, lw=1.2, alpha=0.9,
         label="camdl simulation at He et al. MLE")
ax1.set_ylabel("Weekly measles cases", color=TEXT, fontsize=10)
ax1.set_title(
    "camdl replication of He, Ionides & King (2010): London measles SEIR",
    color=TEXT, fontsize=12, loc="left",
)
ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=TEXT)

# -- Bottom: pfilter log-likelihood vs pomp reference ------------------------
reps = range(1, len(pf) + 1)
ax2.axhspan(POMP_REF_MEAN - 2 * POMP_REF_SD, POMP_REF_MEAN + 2 * POMP_REF_SD,
            color=BLUE, alpha=0.12, lw=0)
ax2.axhline(POMP_REF_MEAN, color=BLUE, lw=2, label="pomp reference mean ±2 SD")
ax2.scatter(reps, pf["loglik"], color=ORANGE, s=36, zorder=3,
            label="camdl pfilter replicates (2000 particles)")
camdl_mean = pf["loglik"].mean()
ax2.axhline(camdl_mean, color=ORANGE, lw=1.2, ls="--")
ax2.annotate(f"camdl mean {camdl_mean:.1f}", xy=(len(pf), camdl_mean),
             xytext=(-2, -14), textcoords="offset points", ha="right",
             fontsize=9, color=TEXT)
ax2.annotate(f"pomp mean {POMP_REF_MEAN:.1f}", xy=(1, POMP_REF_MEAN),
             xytext=(2, 6), textcoords="offset points", fontsize=9, color=TEXT)
ax2.set_xticks(list(reps))
ax2.set_xlabel("Particle-filter replicate", color=TEXT, fontsize=10)
ax2.set_ylabel("Log-likelihood at MLE", color=TEXT, fontsize=10)
ax2.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT)

fig.savefig("results/he2010_replication.png", dpi=150, facecolor="white")
print("wrote results/he2010_replication.png")
print(f"camdl pfilter mean loglik: {camdl_mean:.2f}  (pomp: {POMP_REF_MEAN:.2f})")
