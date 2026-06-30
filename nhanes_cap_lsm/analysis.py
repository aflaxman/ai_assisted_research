"""
NHANES 2017-2020 (pre-pandemic) liver stiffness (LSM) by CAP threshold.

Compares the liver stiffness measure (LSM, kPa; var LUXSMED) distribution
between participants with controlled attenuation parameter (CAP, dB/m;
var LUXCAPM) below vs. at/above 288 dB/m -- a common cutoff for hepatic
steatosis.

Fibrosis staging (per request, mutually-exclusive bins on LSM):
    F0  : LSM < 6        (no / minimal fibrosis)
    F1  : 6  <= LSM < 8
    F2  : 8  <= LSM < 10
    F3  : 10 <= LSM < 15
    F4  : LSM >= 15

Data: P_LUX (Liver Ultrasound Transient Elastography) + P_DEMO, merged on SEQN.
Inclusion: complete elastography exam (LUAXSTAT == 1) with non-missing LSM & CAP.
Note: estimates are UNWEIGHTED sample distributions (survey weights not applied).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------
# Fetch raw NHANES 2017-2020 pre-pandemic files if not already present
# --------------------------------------------------------------------------
CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
os.makedirs("data", exist_ok=True)
for fname in ("P_LUX.xpt", "P_DEMO.xpt"):
    dest = os.path.join("data", fname)
    if not os.path.exists(dest):
        import requests
        print(f"downloading {fname} ...")
        r = requests.get(f"{CDC}/{fname}", timeout=180)
        r.raise_for_status()
        open(dest, "wb").write(r.content)

# --------------------------------------------------------------------------
# Load & build analysis dataset
# --------------------------------------------------------------------------
lux = pd.read_sas("data/P_LUX.xpt")[["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"]]
demo = pd.read_sas("data/P_DEMO.xpt")[["SEQN", "RIDAGEYR"]]

df = lux.merge(demo, on="SEQN", how="left")
df = df[(df.LUAXSTAT == 1) & df.LUXSMED.notna() & df.LUXCAPM.notna()].copy()
df = df.rename(columns={"LUXSMED": "LSM", "LUXCAPM": "CAP", "RIDAGEYR": "age"})

CAP_CUT = 288
df["cap_hi"] = df.CAP >= CAP_CUT  # True = steatosis (CAP >= 288)

# Fibrosis stage thresholds & labels
THRESH = {"F1": 6, "F2": 8, "F3": 10, "F4": 15}
STAGE_RANGE = {"F1": "6-8", "F2": "8-10", "F3": "10-15", "F4": ">=15"}
STAGES = ["F1", "F2", "F3", "F4"]


def stage(lsm):
    if lsm < 6:
        return "F0"
    if lsm < 8:
        return "F1"
    if lsm < 10:
        return "F2"
    if lsm < 15:
        return "F3"
    return "F4"


df["stage"] = df.LSM.apply(stage)

CAP_GROUPS = [(False, f"CAP < {CAP_CUT} dB/m"), (True, f"CAP ≥ {CAP_CUT} dB/m")]
CAP_COLORS = {False: "#2c7fb8", True: "#d95f0e"}

# 4 age groups for the small multiples
AGE_BINS = [12, 30, 45, 60, 81]
AGE_LABELS = ["12-29", "30-44", "45-59", "60-80"]
df["agegrp4"] = pd.cut(df.age, bins=AGE_BINS, right=False, labels=AGE_LABELS)

print(f"Analysis sample: n={len(df)}")
print(df.groupby("cap_hi").LSM.describe()[["count", "mean", "50%", "75%", "max"]])
print("\nStage counts by CAP group:")
print(pd.crosstab(df.stage, df.cap_hi, margins=True))


# --------------------------------------------------------------------------
# Shared helper: draw one LSM panel (histogram + KDE + stage lines)
# --------------------------------------------------------------------------
XMAX = 25.0
HIST_BINS = np.arange(0, XMAX + 0.5, 0.5)


def draw_lsm_panel(ax, data, color, label_stages=False, ymax=None):
    """Histogram (density) + KDE for an LSM array, with fibrosis stage lines."""
    d = data[np.isfinite(data)]
    # histogram over the shown range only (rare LSM > XMAX kPa omitted, not
    # piled into the last bin); KDE below is still fit to the full data.
    ax.hist(d[d <= XMAX], bins=HIST_BINS, density=True,
            color=color, alpha=0.35, edgecolor="white", linewidth=0.3)
    if len(d) > 5 and d.std() > 0:
        kde = gaussian_kde(d)
        xs = np.linspace(0, XMAX, 400)
        ax.plot(xs, kde(xs), color=color, linewidth=2)

    # vertical stage threshold lines at 6, 8, 10, 15
    for t in THRESH.values():
        ax.axvline(t, color="0.35", linestyle="--", linewidth=0.8, zorder=1)

    if label_stages:
        # place F-stage labels in the middle of each region near the top
        y = ax.get_ylim()[1] if ymax is None else ymax
        centers = {"F0": 3, "F1": 7, "F2": 9, "F3": 12.5, "F4": 20}
        for s, xc in centers.items():
            ax.text(xc, y * 0.93, s, ha="center", va="top", fontsize=8,
                    color="0.25", fontweight="bold")
    ax.set_xlim(0, XMAX)


# --------------------------------------------------------------------------
# FIGURE 1: Overall stacked (CAP<288 above CAP>=288), all ages
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)
# pre-compute a common y-limit
ymax = 0
for hi, _ in CAP_GROUPS:
    d = df.loc[df.cap_hi == hi, "LSM"].values
    kde = gaussian_kde(d[np.isfinite(d)])
    ymax = max(ymax, kde(np.linspace(0, XMAX, 400)).max())
ymax *= 1.15

for ax, (hi, name) in zip(axes, CAP_GROUPS):
    d = df.loc[df.cap_hi == hi, "LSM"].values
    ax.set_ylim(0, ymax)
    draw_lsm_panel(ax, d, CAP_COLORS[hi], label_stages=True, ymax=ymax)
    ax.set_ylabel("density")
    ax.set_title(f"{name}   (n = {len(d):,})", loc="left", fontweight="bold")
axes[-1].set_xlabel("Liver stiffness measure, LSM (kPa)")
fig.suptitle("NHANES 2017-2020: liver stiffness distribution by CAP steatosis "
             "threshold\n(dashed lines: F1=6, F2=8, F3=10, F4=15 kPa; "
             "unweighted)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig1_lsm_overall.png", dpi=130)
plt.close(fig)
print("\nwrote fig1_lsm_overall.png")


# --------------------------------------------------------------------------
# FIGURE 2: Small multiples -- 2 CAP rows x 4 age columns
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True, sharey=True)
# common y across all panels
ymax = 0
for hi, _ in CAP_GROUPS:
    for ag in AGE_LABELS:
        d = df.loc[(df.cap_hi == hi) & (df.agegrp4 == ag), "LSM"].values
        d = d[np.isfinite(d)]
        if len(d) > 5 and d.std() > 0:
            kde = gaussian_kde(d)
            ymax = max(ymax, kde(np.linspace(0, XMAX, 300)).max())
ymax *= 1.15

for r, (hi, name) in enumerate(CAP_GROUPS):
    for c, ag in enumerate(AGE_LABELS):
        ax = axes[r, c]
        d = df.loc[(df.cap_hi == hi) & (df.agegrp4 == ag), "LSM"].values
        ax.set_ylim(0, ymax)
        draw_lsm_panel(ax, d, CAP_COLORS[hi], label_stages=(r == 0),
                       ymax=ymax)
        if r == 0:
            ax.set_title(f"Age {ag}", fontsize=12, fontweight="bold")
        ax.text(0.97, 0.97, f"n = {len(d):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="0.3")
        if c == 0:
            ax.set_ylabel(f"{name}\n\ndensity", fontweight="bold")
        if r == 1:
            ax.set_xlabel("LSM (kPa)")

fig.suptitle("NHANES 2017-2020 liver stiffness (LSM) by CAP threshold and age "
             "group\nTop: CAP < 288 dB/m   |   Bottom: CAP ≥ 288 dB/m   "
             "(dashed lines F1=6, F2=8, F3=10, F4=15; unweighted)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("fig2_lsm_smallmultiples.png", dpi=130)
plt.close(fig)
print("wrote fig2_lsm_smallmultiples.png")


# --------------------------------------------------------------------------
# FIGURE 3: Fibrosis-stage prevalence vs age (5-year groups), by CAP group
# --------------------------------------------------------------------------
AGE5_BINS = list(range(10, 86, 5))          # 10-15, 15-20, ..., 80-85
df["age5"] = pd.cut(df.age, bins=AGE5_BINS, right=False)
MIN_CELL = 25  # suppress prevalence estimates from cells smaller than this

# prevalence table: for each (cap_hi, age5), fraction in each stage
rows = []
for hi, _ in CAP_GROUPS:
    for interval, g in df[df.cap_hi == hi].groupby("age5", observed=True):
        n = len(g)
        mid = (interval.left + interval.right) / 2
        rec = {"cap_hi": hi, "age_mid": mid, "n": n}
        for s in STAGES:
            rec[s] = (g.stage == s).mean() * 100 if n >= MIN_CELL else np.nan
        rows.append(rec)
prev = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
for ax, s in zip(axes.ravel(), STAGES):
    for hi, name in CAP_GROUPS:
        sub = prev[prev.cap_hi == hi].sort_values("age_mid")
        ax.plot(sub.age_mid, sub[s], marker="o", markersize=4,
                color=CAP_COLORS[hi], label=name)
    ax.set_title(f"{s}  (LSM {STAGE_RANGE[s]} kPa)", fontweight="bold")
    ax.set_ylabel("prevalence (% of CAP group)")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
for ax in axes[1]:
    ax.set_xlabel("Age (5-year group midpoint, years)")
axes[0, 0].legend(frameon=False, fontsize=10)
fig.suptitle("NHANES 2017-2020: fibrosis-stage prevalence by age, above vs. "
             f"below CAP {CAP_CUT} dB/m\n(5-year age groups; cells with n < "
             f"{MIN_CELL} suppressed; unweighted)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig3_fibrosis_prevalence_by_age.png", dpi=130)
plt.close(fig)
print("wrote fig3_fibrosis_prevalence_by_age.png")

# --------------------------------------------------------------------------
# Summary tables to stdout + csv
# --------------------------------------------------------------------------
print("\n=== Stage prevalence (%) by CAP group, all ages ===")
tab = (pd.crosstab(df.stage, df.cap_hi, normalize="columns") * 100).round(1)
tab.columns = ["CAP<288", "CAP>=288"]
print(tab.reindex(["F0", "F1", "F2", "F3", "F4"]))
tab.to_csv("stage_prevalence_by_cap.csv")
prev.to_csv("stage_prevalence_by_age_cap.csv", index=False)
print("\nwrote stage_prevalence_by_cap.csv, stage_prevalence_by_age_cap.csv")
