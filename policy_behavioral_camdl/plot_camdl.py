"""Four-panel figure for the camdl policy-behavioral SIR replication.

A: posterior-predictive daily incidence vs the observed synthetic series
B: parameter recovery -- posteriors vs the generating truth
C: counterfactual -- policy-informed behavior flattens the curve
D: the mechanism -- policy signal p(t) and the realized alarm a(t)

Usage: uv run python plot_camdl.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_inputs import TRUTH, POLICY_START

BLUE, ORANGE, GREEN, TEXT, MUTED = "#2a78d6", "#eb6834", "#2ca25f", "#1a1a19", "#6f6e66"


def _read(path):
    return pd.read_csv(path, sep="\t", comment="#")


def _time_col(df):
    for c in ("time", "t"):
        if c in df.columns:
            return c
    return df.columns[0]


def _ensemble_mean(df, value="daily_cases"):
    """Mean over replicates/draws of a per-time value column."""
    tc = _time_col(df)
    return df.groupby(tc)[value].mean()


fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
(axA, axB), (axC, axD) = axes
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e4dd", linewidth=0.8)

# -- A: posterior-predictive envelope vs observed ---------------------------
obs = _read("data/cases.tsv")
pp = _read("results/policy_postpred_obs.tsv")
tc = _time_col(pp)
piv = pp.pivot_table(index=tc, columns=[c for c in pp.columns if c not in (tc, "daily_cases")],
                     values="daily_cases")
qs = piv.quantile([0.025, 0.5, 0.975], axis=1).T
axA.fill_between(qs.index, qs[0.025], qs[0.975], color=BLUE, alpha=0.18, lw=0,
                 label="posterior predictive 95%")
axA.plot(qs.index, qs[0.5], color=BLUE, lw=2, label="median")
axA.scatter(obs[_time_col(obs)], obs["daily_cases"], s=14, color=ORANGE,
            zorder=3, label="observed (synthetic)")
axA.axvline(POLICY_START, color=MUTED, ls=":", lw=1)
axA.set_xlabel("day", color=TEXT); axA.set_ylabel("new infections / day", color=TEXT)
axA.set_title("A  Fit to the observed series", loc="left", color=TEXT, fontsize=11)
axA.legend(frameon=False, fontsize=8, labelcolor=TEXT)

# -- B: parameter recovery --------------------------------------------------
draws = _read("results/policy_posterior_draws.tsv")
names = ["beta", "gamma", "endog_delta", "policy_weight"]
present = [n for n in names if n in draws.columns]
# small inset-style overlay: 2x2 mini-hists inside panel B via twin axes grid
axB.axis("off")
axB.set_title("B  Parameter recovery (truth = black)", loc="left", color=TEXT, fontsize=11)
gs = axB.get_subplotspec().subgridspec(2, 2, wspace=0.45, hspace=0.55)
for i, n in enumerate(present):
    sub = fig.add_subplot(gs[i // 2, i % 2])
    sub.hist(draws[n], bins=30, color=BLUE, alpha=0.8, density=True)
    sub.axvline(TRUTH[n], color="black", lw=2)
    lo, med, hi = draws[n].quantile([0.025, 0.5, 0.975])
    sub.set_title(f"{n}\n{med:.3f} ({lo:.3f}, {hi:.3f})", fontsize=8, color=TEXT)
    sub.tick_params(labelsize=7, colors=MUTED)
    sub.spines[["top", "right"]].set_visible(False)

# -- C: counterfactual flattening -------------------------------------------
cf = {
    "no behavioral response": ("results/cf_nobehavior_obs.tsv", MUTED),
    "endogenous fear only": ("results/cf_endog_obs.tsv", GREEN),
    "policy-informed": ("results/cf_full_obs.tsv", BLUE),
}
for label, (path, color) in cf.items():
    m = _ensemble_mean(_read(path))
    axC.plot(m.index, m.values, color=color, lw=2, label=label)
axC.axvspan(POLICY_START, m.index.max(), color="grey", alpha=0.10, lw=0)
axC.set_xlabel("day", color=TEXT); axC.set_ylabel("new infections / day", color=TEXT)
axC.set_title("C  Policy-informed behavior flattens the curve", loc="left",
              color=TEXT, fontsize=11)
axC.legend(frameon=False, fontsize=8, labelcolor=TEXT)

# -- D: policy signal + realized alarm --------------------------------------
pol = _read("data/policy_signal.tsv")
traj = _read("results/traj_full.tsv")
ttc = _time_col(traj)
Icol = "I" if "I" in traj.columns else [c for c in traj.columns if c.endswith("I")][0]
I = traj.set_index(ttc)[Icol].astype(float)
p = pol.set_index("time")["policy"].reindex(I.index).ffill().fillna(0.0)
endog = TRUTH["endog_delta"] / (1.0 + (TRUTH["endog_x0"] / (I + 1.0)) ** TRUTH["endog_nu"])
alarm = 1.0 - (1.0 - endog) * (1.0 - TRUTH["policy_weight"] * p)
axD.plot(p.index, p.values, color=ORANGE, ls="--", lw=2, label="policy p(t)")
axD.plot(alarm.index, alarm.values, color=BLUE, lw=2, label="realized alarm a(t)")
axD.plot(endog.index, endog.values, color=GREEN, lw=1.5, alpha=0.8,
         label="endogenous component")
axD.set_xlabel("day", color=TEXT); axD.set_ylabel("intensity / alarm", color=TEXT)
axD.set_title("D  Policy signal and the alarm it drives", loc="left",
              color=TEXT, fontsize=11)
axD.legend(frameon=False, fontsize=8, labelcolor=TEXT)

fig.suptitle("camdl replication: policy-informed behavioral-change SIR",
             color=TEXT, fontsize=13, x=0.02, ha="left")
fig.savefig("results/policy_behavioral_camdl.png", dpi=150, facecolor="white")
print("wrote results/policy_behavioral_camdl.png")

# Console summary for the README.
print("\nposterior recovery (median [95% CrI] vs truth):")
for n in present:
    lo, med, hi = draws[n].quantile([0.025, 0.5, 0.975])
    print(f"  {n:14s} {med:7.3f} [{lo:6.3f}, {hi:6.3f}]   truth={TRUTH[n]}")
