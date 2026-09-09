"""Three-panel figure for the classroom norovirus fit.

A: observed absences vs the model's posterior predictive band
B: what the model thinks was happening inside the class (posterior mean of
   each box, stacked)
C: R0, prior vs posterior

Inputs (all written by the camdl commands in README.md):
  data/absent.tsv                 observed absences
  results/predictive_absent.tsv   fit predict band (free_forward rows)
  results/posterior_traj.tsv      simulate --draws posterior trajectories
  results/posterior_draws.tsv     PGAS draws.tsv

Usage: uv run python plot.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reference palette (dataviz skill): categorical slots 1-4, chrome, ink.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

obs = pd.read_csv("data/absent.tsv", sep="\t", na_values="NA")
pred = pd.read_csv("results/predictive_absent.tsv", sep="\t")
pred = pred[pred["horizon"] == "free_forward"].set_index("time")
traj = pd.read_csv("results/posterior_traj.tsv", sep="\t", comment="#")
draws = pd.read_csv("results/posterior_draws.tsv", sep="\t")
draws["R0"] = draws["beta"] * draws["d_school"]

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
fig.patch.set_facecolor(SURFACE)
for ax in (axA, axB, axC):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

weekday = {1: "M", 2: "T", 3: "W", 4: "T", 5: "F", 8: "M", 9: "T", 10: "W", 11: "T", 12: "F", 15: "M"}


def day_axis(ax, lo=0.5, hi=15.5, on_top=False):
    for wk in (6, 13):  # Sat + Sun
        if on_top:  # fade whatever is drawn underneath
            ax.axvspan(wk - 0.5, wk + 1.5, color=SURFACE, alpha=0.55, lw=0, zorder=5)
        else:
            ax.axvspan(wk - 0.5, wk + 1.5, color=GRID, alpha=0.5, lw=0)
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(weekday))
    ax.set_xticklabels(list(weekday.values()))
    ax.set_xlabel("School day (day 1 = Monday; shaded = weekend)", color=INK2, fontsize=9.5)


# -- A: fit ----------------------------------------------------------------
day_axis(axA)
axA.fill_between(pred.index, pred["q05"], pred["q95"], color=BLUE, alpha=0.18, lw=0,
                 label="model, 90% band")
axA.fill_between(pred.index, pred["q25"], pred["q75"], color=BLUE, alpha=0.35, lw=0,
                 label="model, 50% band")
axA.plot(pred.index, pred["q50"], color=BLUE, lw=2, label="model, median")
a0_med = draws["a0"].median()
axA.axhline(a0_med, color=INK2, lw=1.2, ls=(0, (4, 3)),
            label=f"background absences (median {a0_med:.1f})")
seen = obs.dropna()
axA.scatter(seen["time"], seen["absent"], color=INK, s=34, zorder=4, label="observed absences")
axA.set_ylim(0, 24)
axA.set_ylabel("Students absent (of 28)", color=INK2, fontsize=9.5)
axA.set_title("A  Absences: data and fitted model", loc="left", color=INK, fontsize=11)
axA.legend(fontsize=8.5, loc="upper right", labelcolor=INK2, framealpha=0.92,
           facecolor=SURFACE, edgecolor=GRID, markerscale=0.8)

# -- B: hidden boxes ---------------------------------------------------------
day_axis(axB, on_top=True)
mean_state = traj.groupby("t")[["H", "I", "S", "R"]].mean()
t = mean_state.index.values
order = [("H", BLUE, "H  home sick"), ("I", ORANGE, "I  infectious, at school"),
         ("S", AQUA, "S  susceptible"), ("R", YELLOW, "R  recovered")]
bottom = np.zeros(len(t))
for col, color, label in order:
    top = bottom + mean_state[col].values
    axB.fill_between(t, bottom, top, color=color, alpha=0.85, lw=0, label=label)
    axB.plot(t, top, color=SURFACE, lw=1.5)  # 2px surface gap between stacked fills
    # direct label at the right edge, only if the band is thick enough there
    thick = mean_state[col].values[-1]
    if thick >= 2.0:
        axB.annotate(col, xy=(t[-1], bottom[-1] + thick / 2), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK2)
    bottom = top
axB.set_ylim(0, 28)
axB.set_ylabel("Students (posterior mean per box)", color=INK2, fontsize=9.5)
axB.set_title("B  Inside the class, as the model sees it", loc="left", color=INK, fontsize=11)
leg = axB.legend(fontsize=8.5, loc="upper right", labelcolor=INK2, framealpha=0.92,
                 facecolor=SURFACE, edgecolor=GRID)
leg.set_zorder(6)

# -- C: R0 prior vs posterior ------------------------------------------------
axC.grid(axis="y", color=GRID, linewidth=0.8)
xs = np.linspace(0.05, 10, 400)
mu, sigma = 0.69, 1.0  # the log-normal prior in fit.toml (R0 = beta x 1 day)
prior = np.exp(-(np.log(xs) - mu) ** 2 / (2 * sigma**2)) / (xs * sigma * np.sqrt(2 * np.pi))
axC.plot(xs, prior, color=MUTED, lw=1.6, ls=(0, (4, 3)), label="prior")
axC.hist(draws["R0"], bins=np.arange(0, 10.25, 0.25), density=True, color=BLUE, alpha=0.8,
         label="posterior")
axC.axvline(1.0, color=INK2, lw=1.0)
axC.annotate("R0 = 1", xy=(1.0, axC.get_ylim()[1] * 0.99), xytext=(-4, 0),
             textcoords="offset points", fontsize=8.5, color=INK2, va="top", ha="right")
lo, med, hi = draws["R0"].quantile([0.025, 0.5, 0.975])
p_gt1 = (draws["R0"] > 1).mean()
axC.annotate(f"posterior median {med:.1f}\n95% interval {lo:.1f} to {hi:.1f}\n"
             f"P(R0 > 1) = {p_gt1:.2f}", xy=(0.97, 0.75), xycoords="axes fraction",
             ha="right", va="top", fontsize=9, color=INK2)
axC.set_xlim(0, 10)
axC.set_xlabel("R0 = infections per infectious kid, fully susceptible class", color=INK2,
               fontsize=9.5)
axC.set_ylabel("Density", color=INK2, fontsize=9.5)
axC.set_title("C  R0: what nine numbers can and cannot say", loc="left", color=INK, fontsize=11)
axC.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)

fig.savefig("results/classroom_fit.png", dpi=150, facecolor=SURFACE)
print("wrote results/classroom_fit.png")
q = draws[["R0", "d_home", "i0", "h0", "a0"]].quantile([0.025, 0.5, 0.975]).T
q.columns = ["q2.5", "median", "q97.5"]
print(q.round(2).to_string())
print(f"P(R0 > 1) = {p_gt1:.2f}")
