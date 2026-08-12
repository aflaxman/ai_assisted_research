"""Erlang-4 refinement of the Hondius model: shape choice and results.

Left: why the dwell shape matters — exponential vs Erlang-4 incubation
densities at mean 22 d, against the observed Andes-virus range.
Middle: R0 posteriors, exponential z22 fit vs Erlang fit.
Right: Erlang out-of-sample projection to May 27 vs WHO counts.

Usage: uv run python plot_hondius_erlang.py
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#1a1a19"
MUTED = "#6f6e66"

d_exp = pd.read_csv("results/hondius_z22_posterior_draws.tsv", sep="\t")
d_erl = pd.read_csv("results/hondius_erlang_posterior_draws.tsv", sep="\t")
traj = pd.read_csv("results/hondius_erlang_postpred_traj.tsv", sep="\t", comment="#")
obs = pd.read_csv("data/hondius_cumulative.tsv", sep="\t")

fig, (axS, axR, axP) = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
for ax in (axS, axR, axP):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e5e4dd", linewidth=0.8)
    ax.set_axisbelow(True)

# -- dwell-shape choice -------------------------------------------------------
x = np.linspace(0.01, 60, 500)
mean = 22.0
exp_pdf = np.exp(-x / mean) / mean
k = 4
theta = mean / k
erl_pdf = x ** (k - 1) * np.exp(-x / theta) / (math.gamma(k) * theta**k)
axS.axvspan(7, 39, color=MUTED, alpha=0.12, lw=0)
axS.annotate("observed Andes-virus\nrange 7–39 d", xy=(31, 0.030), fontsize=8,
             color=MUTED, ha="center")
axS.plot(x, exp_pdf, color=BLUE, lw=2, label="exponential (mean 22 d)")
axS.plot(x, erl_pdf, color=ORANGE, lw=2, label="Erlang-4 (mean 22 d)")
axS.set_xlabel("Incubation period (days)", color=TEXT, fontsize=10)
axS.set_ylabel("Density", color=TEXT, fontsize=10)
axS.set_title("Same mean, different biology", loc="left", color=TEXT, fontsize=11)
axS.legend(frameon=False, fontsize=8, labelcolor=TEXT)

# -- R0 posteriors ------------------------------------------------------------
bins = np.linspace(0.5, 7, 40)
axR.hist(d_exp["R0"], bins=bins, color=BLUE, alpha=0.75, density=True,
         label="exponential dwell")
axR.hist(d_erl["R0"], bins=bins, color=ORANGE, alpha=0.75, density=True,
         label="Erlang-4 dwell")
axR.axvspan(2.52, 2.99, color=MUTED, alpha=0.2, lw=0)
axR.annotate("paper CI", xy=(2.55, axR.get_ylim()[1] * 0.93), fontsize=8, color=TEXT)
for d, c, dy in ((d_exp, BLUE, 0.82), (d_erl, ORANGE, 0.72)):
    m = d["R0"].median()
    axR.annotate(f"median {m:.2f}", xy=(m, axR.get_ylim()[1] * dy), fontsize=8,
                 color=c, xytext=(4, 0), textcoords="offset points")
axR.set_xlabel("R0 = beta × D", color=TEXT, fontsize=10)
axR.set_title("R0 under the two dwell shapes", loc="left", color=TEXT, fontsize=11)
axR.legend(frameon=False, fontsize=8, labelcolor=TEXT)

# -- projection ----------------------------------------------------------------
cum = traj.groupby(["draw", "replicate"])["flow_onset"].cumsum()
traj = traj.assign(cum=cum)
at37 = traj[traj.t == 37].set_index(["draw", "replicate"])["cum"]
keep = at37[(at37 >= 4) & (at37 <= 9)].index
sub = traj.set_index(["draw", "replicate"]).loc[keep].reset_index()
qs = (sub.pivot_table(index="t", columns=["draw", "replicate"], values="cum")
      .quantile([0.025, 0.5, 0.975], axis=1).T)
axP.fill_between(qs.index, qs[0.025], qs[0.975], color=ORANGE, alpha=0.18, lw=0,
                 label="Erlang projection 95%")
axP.plot(qs.index, qs[0.5], color=ORANGE, lw=2, label="median")
axP.scatter(obs["time"], obs["cumulative_cases"], marker="x", color=BLUE, s=36,
            zorder=3, label="fitted window (paper Fig 2)")
axP.scatter([38, 43, 57], [8, 11, 13], marker="o", facecolor="none",
            edgecolor=TEXT, s=48, zorder=3, label="WHO DONs (incl. probable)")
axP.axvline(37, color=MUTED, lw=1, ls=":")
axP.set_xlabel("Day (Apr 1 = 1)", color=TEXT, fontsize=10)
axP.set_ylabel("Cumulative onsets", color=TEXT, fontsize=10)
axP.set_ylim(0, 40)
axP.set_title("Out-of-sample: median 13 vs WHO 13", loc="left", color=TEXT,
              fontsize=11)
axP.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)

fig.savefig("results/hondius_erlang_comparison.png", dpi=150, facecolor="white")
print("wrote results/hondius_erlang_comparison.png")
