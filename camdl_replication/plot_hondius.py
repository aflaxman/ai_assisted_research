"""Four-panel replication figure for Cui (2026) Hondius hantavirus SEIRD.

A: posterior-predictive cumulative cases vs observed (paper Fig 2)
B: R0 posterior vs paper's reported CI (paper Fig 3A)
C: beta x D RMSE surface with the constant-R0 ridge (paper Fig 3C)
D: cumulative identified vs active exposed (paper Fig 4)

Usage: uv run python plot_hondius.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#1a1a19"
MUTED = "#6f6e66"

obs = pd.read_csv("data/hondius_cumulative.tsv", sep="\t")
pp = pd.read_csv("results/hondius_postpred_obs.tsv", sep="\t")
draws = pd.read_csv("results/hondius_posterior_draws.tsv", sep="\t")
grid = pd.read_csv("results/rmse_grid.tsv", sep="\t")
traj = pd.read_csv("results/hondius_postpred_traj.tsv", sep="\t", comment="#")

fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
(axA, axB), (axC, axD) = axes
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)

# -- A: posterior-predictive envelope vs observed ----------------------------
axA.grid(axis="y", color="#e5e4dd", linewidth=0.8)
cum = (pp.pivot_table(index="time", columns=["draw", "replicate"],
                      values="daily_cases").cumsum(axis=0))
qs = cum.quantile([0.025, 0.25, 0.5, 0.75, 0.975], axis=1).T
axA.fill_between(qs.index, qs[0.025], qs[0.975], color=BLUE, alpha=0.15,
                 lw=0, label="posterior predictive 95%")
axA.fill_between(qs.index, qs[0.25], qs[0.75], color=BLUE, alpha=0.3, lw=0,
                 label="posterior predictive IQR")
axA.plot(qs.index, qs[0.5], color=BLUE, lw=2, label="median")
axA.scatter(obs["time"], obs["cumulative_cases"], marker="x", color=ORANGE,
            s=40, zorder=3, label="observed (digitized Fig 2)")
axA.set_xlabel("Day (Apr 1 = 1)", color=TEXT, fontsize=10)
axA.set_ylabel("Cumulative reported cases", color=TEXT, fontsize=10)
axA.set_title("A  Fit to reported cases", loc="left", color=TEXT, fontsize=11)
axA.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)

# -- B: R0 posterior --------------------------------------------------------
axB.grid(axis="y", color="#e5e4dd", linewidth=0.8)
axB.hist(draws["R0"], bins=40, color=BLUE, alpha=0.85, density=True)
axB.axvspan(2.52, 2.99, color=ORANGE, alpha=0.25, lw=0)
axB.axvline(2.76, color=ORANGE, lw=2)
axB.annotate("paper: 2.76\n(2.52, 2.99)", xy=(2.76, axB.get_ylim()[1]*0.75),
             xytext=(8, 0), textcoords="offset points", fontsize=9, color=TEXT)
lo, hi, med = draws["R0"].quantile(.025), draws["R0"].quantile(.975), draws["R0"].median()
axB.annotate(f"camdl posterior:\n{med:.2f} ({lo:.2f}, {hi:.2f})",
             xy=(0.02, 0.85), xycoords="axes fraction", fontsize=9, color=TEXT)
axB.set_xlabel("R0 = beta × D", color=TEXT, fontsize=10)
axB.set_ylabel("Posterior density", color=TEXT, fontsize=10)
axB.set_title("B  R0: posterior vs paper CI", loc="left", color=TEXT, fontsize=11)

# -- C: RMSE surface --------------------------------------------------------
piv = grid.pivot(index="D", columns="beta", values="rmse")
im = axC.pcolormesh(piv.columns, piv.index, piv.values, cmap="viridis_r",
                    shading="nearest")
fig.colorbar(im, ax=axC, label="RMSE (median cum. cases)", shrink=0.9)
bb = np.linspace(0.1, 0.5, 100)
axC.plot(bb, 2.76 / bb, color="white", lw=2, ls="--")
axC.annotate("beta × D = 2.76", xy=(0.4, 2.76/0.4), color="white", fontsize=9,
             xytext=(0, 14), textcoords="offset points")
axC.scatter([0.24], [11.52], marker="x", color=ORANGE, s=70, zorder=3,
            label="paper optimum")
axC.set_ylim(7, 14)
axC.set_xlabel("Transmission rate beta", color=TEXT, fontsize=10)
axC.set_ylabel("Infectious period D (days)", color=TEXT, fontsize=10)
axC.set_title("C  RMSE surface: a ridge, not a basin", loc="left",
              color=TEXT, fontsize=11)
axC.legend(frameon=False, fontsize=9, loc="upper right", labelcolor="white")

# -- D: hidden exposed reservoir ---------------------------------------------
axD.grid(axis="y", color="#e5e4dd", linewidth=0.8)
tmean = traj.groupby("t")[["E", "flow_onset"]].mean()
cum_id = traj.groupby(["draw", "replicate"])["flow_onset"].cumsum()
traj2 = traj.assign(cum_id=cum_id)
mean_cum_id = traj2.groupby("t")["cum_id"].mean()
axD.bar(tmean.index, mean_cum_id, color=BLUE, width=0.85,
        label="cumulative identified (mean)")
axD.bar(tmean.index, tmean["E"], bottom=mean_cum_id, color=ORANGE,
        width=0.85, label="active exposed (mean)")
axD.set_xlabel("Day (Apr 1 = 1)", color=TEXT, fontsize=10)
axD.set_ylabel("Number of cases", color=TEXT, fontsize=10)
axD.set_title("D  Hidden exposed reservoir", loc="left", color=TEXT, fontsize=11)
axD.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TEXT)

fig.suptitle("camdl replication of Cui (2026): MV Hondius hantavirus SEIRD",
             color=TEXT, fontsize=13, x=0.02, ha="left")
fig.savefig("results/hondius_replication.png", dpi=150, facecolor="white")
print("wrote results/hondius_replication.png")
print(f"final active exposed (mean, day 37): {tmean['E'].iloc[-1]:.1f}")
print(f"final cum identified (mean, day 37): {mean_cum_id.iloc[-1]:.1f}")
