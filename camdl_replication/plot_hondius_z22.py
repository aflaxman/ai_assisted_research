"""Effect of WHO's updated 22-day mean incubation on the Hondius fit.

Left: Z posterior under the paper's [6,12] range vs the Z~22d prior.
Middle: the induced R0 shift.
Right: out-of-sample projection to May 27 vs WHO's reported counts.

Usage: uv run python plot_hondius_z22.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#1a1a19"
MUTED = "#6f6e66"

d0 = pd.read_csv("results/hondius_posterior_draws.tsv", sep="\t")
d22 = pd.read_csv("results/hondius_z22_posterior_draws.tsv", sep="\t")
traj = pd.read_csv("results/hondius_z22_postpred_traj.tsv", sep="\t", comment="#")
obs = pd.read_csv("data/hondius_cumulative.tsv", sep="\t")

fig, (axZ, axR, axP) = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
for ax in (axZ, axR, axP):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e5e4dd", linewidth=0.8)
    ax.set_axisbelow(True)

# -- Z posteriors -------------------------------------------------------------
axZ.hist(d0["z_lat"], bins=30, color=BLUE, alpha=0.8, density=True,
         label="paper's Z range [6,12]")
axZ.hist(d22["z_lat"], bins=30, color=ORANGE, alpha=0.8, density=True,
         label="Z ~ lognormal(mean 22 d)")
axZ.axvline(22, color=TEXT, lw=1.2, ls=":")
axZ.annotate("WHO: 22 d", xy=(22, axZ.get_ylim()[1] * 0.9), fontsize=9,
             color=TEXT, xytext=(4, 0), textcoords="offset points")
axZ.set_xlabel("Latency period Z (days)", color=TEXT, fontsize=10)
axZ.set_ylabel("Posterior density", color=TEXT, fontsize=10)
axZ.set_title("Z posterior ≈ prior, either way", loc="left", color=TEXT, fontsize=11)
axZ.legend(frameon=False, fontsize=8, labelcolor=TEXT)

# -- R0 shift -----------------------------------------------------------------
axR.hist(d0["R0"], bins=35, color=BLUE, alpha=0.8, density=True,
         label="paper's Z range")
axR.hist(d22["R0"], bins=35, color=ORANGE, alpha=0.8, density=True,
         label="Z ~ 22 d")
axR.axvspan(2.52, 2.99, color=MUTED, alpha=0.2, lw=0)
axR.annotate("paper CI", xy=(2.55, axR.get_ylim()[1] * 0.92), fontsize=8, color=TEXT)
for d, c, dy in ((d0, BLUE, 0.80), (d22, ORANGE, 0.70)):
    m = d["R0"].median()
    axR.annotate(f"median {m:.2f}", xy=(m, axR.get_ylim()[1] * dy), fontsize=8,
                 color=c, xytext=(4, 0), textcoords="offset points")
axR.set_xlabel("R0 = beta × D", color=TEXT, fontsize=10)
axR.set_title("Longer incubation ⇒ higher R0", loc="left", color=TEXT, fontsize=11)
axR.legend(frameon=False, fontsize=8, labelcolor=TEXT)

# -- projection to May 27 ------------------------------------------------------
cum = traj.groupby(["draw", "replicate"])["flow_onset"].cumsum()
traj = traj.assign(cum=cum)
at37 = traj[traj.t == 37].set_index(["draw", "replicate"])["cum"]
keep = at37[(at37 >= 4) & (at37 <= 9)].index
sub = traj.set_index(["draw", "replicate"]).loc[keep].reset_index()
qs = (sub.pivot_table(index="t", columns=["draw", "replicate"], values="cum")
      .quantile([0.025, 0.5, 0.975], axis=1).T)
axP.fill_between(qs.index, qs[0.025], qs[0.975], color=ORANGE, alpha=0.18, lw=0,
                 label="Z~22 projection 95%")
axP.plot(qs.index, qs[0.5], color=ORANGE, lw=2, label="median")
axP.scatter(obs["time"], obs["cumulative_cases"], marker="x", color=BLUE, s=36,
            zorder=3, label="fitted window (paper Fig 2)")
axP.scatter([38, 43, 57], [8, 11, 13], marker="o", facecolor="none",
            edgecolor=TEXT, s=48, zorder=3, label="WHO DONs (incl. probable)")
axP.axvline(37, color=MUTED, lw=1, ls=":")
axP.annotate("end of fitted data\n(May 7)", xy=(37, 1), fontsize=8, color=MUTED,
             xytext=(-78, 4), textcoords="offset points")
axP.set_xlabel("Day (Apr 1 = 1)", color=TEXT, fontsize=10)
axP.set_ylabel("Cumulative onsets", color=TEXT, fontsize=10)
axP.set_ylim(0, 40)
axP.set_title("Out-of-sample: pipeline keeps delivering", loc="left",
              color=TEXT, fontsize=11)
axP.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)

fig.savefig("results/hondius_z22_comparison.png", dpi=150, facecolor="white")
print("wrote results/hondius_z22_comparison.png")
