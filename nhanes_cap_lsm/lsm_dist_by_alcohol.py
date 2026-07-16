"""
LSM distribution stratified by CAP threshold AND alcohol consumption.

Extends the original LSM-distribution figures (analysis.py) by adding alcohol as
a second stratifier. Rows show the CAP dimension both WITHOUT the threshold
(All) and WITH it (CAP < 288 / CAP >= 288); columns show alcohol (none /
moderate / heavy). Adults 18+ only (alcohol questionnaire P_ALQ is adult-only).

Alcohol categories (sex-specific, MASLD/ALD-aligned): none = no alcohol in past
12 mo; heavy = >=30 g/day men / >=20 g/day women; moderate = in between.
Average ethanol = (drinking days/yr from ALQ121) * (ALQ130 drinks/day) * 14 / 365.

Survey-weighted (WTMECPRP). Read XPTs with pyreadstat (pandas.read_sas miscodes
0 as ~5.4e-79, which breaks the ALQ121 == 0 "none" test).
"""

import os
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
os.makedirs("data", exist_ok=True)
for fname in ("P_LUX.xpt", "P_DEMO.xpt", "P_ALQ.xpt"):
    dest = os.path.join("data", fname)
    if not os.path.exists(dest):
        import requests
        r = requests.get(f"{CDC}/{fname}", timeout=180); r.raise_for_status()
        open(dest, "wb").write(r.content)


def rx(fname, cols):
    df, _ = pyreadstat.read_xport(os.path.join("data", fname))
    return df[cols]


lux = rx("P_LUX.xpt", ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"])
demo = rx("P_DEMO.xpt", ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMECPRP"])
alq = rx("P_ALQ.xpt", ["SEQN", "ALQ111", "ALQ121", "ALQ130"])

df = lux.merge(demo, on="SEQN").merge(alq, on="SEQN", how="left")
df = df.rename(columns={"LUXSMED": "LSM", "LUXCAPM": "CAP",
                        "RIDAGEYR": "age", "WTMECPRP": "w"})
df = df[(df.LUAXSTAT == 1) & df.LSM.notna() & df.CAP.notna()
        & df.age.notna() & (df.w > 0) & (df.age >= 18)].copy()

DAYS_PER_YEAR = {1: 365, 2: 313, 3: 182, 4: 104, 5: 52, 6: 30,
                 7: 12, 8: 9, 9: 4.5, 10: 1.5}


def ethanol_gpd(r):
    if r.ALQ111 == 2 or r.ALQ121 == 0:
        return 0.0
    if pd.isna(r.ALQ121) or r.ALQ121 in (77, 99):
        return np.nan
    if pd.isna(r.ALQ130) or r.ALQ130 >= 777:
        return np.nan
    return DAYS_PER_YEAR.get(int(r.ALQ121), np.nan) * r.ALQ130 * 14.0 / 365.0


df["gpd"] = df.apply(ethanol_gpd, axis=1)


def alc_cat(r):
    if pd.isna(r.gpd):
        return np.nan
    if r.gpd == 0:
        return "none"
    return "heavy" if r.gpd >= (30.0 if r.RIAGENDR == 1 else 20.0) else "moderate"


df["alc"] = df.apply(alc_cat, axis=1)
df = df[df.alc.notna()].copy()

# strata: CAP rows (incl. pooled "All"), alcohol columns
CAP_ROWS = [("All (no CAP threshold)", np.ones(len(df), bool)),
            ("CAP < 288 dB/m", (df.CAP < 288).values),
            ("CAP ≥ 288 dB/m", (df.CAP >= 288).values)]
ALC_COLS = [("none", "None\n(no alcohol)"),
            ("moderate", "Moderate\n(<30/20 g/d)"),
            ("heavy", "Heavy\n(≥30 g/d M, ≥20 g/d W)")]
ALC_COLORS = {"none": "#1a9850", "moderate": "#e08214", "heavy": "#d73027"}

THRESH = {"F1": 6, "F2": 8, "F3": 10, "F4": 15}
XMAX = 25.0
HIST_BINS = np.arange(0, XMAX + 0.5, 0.5)

print(f"adults 18+ with valid elastography + alcohol class: n={len(df):,}")
print("cell counts (CAP row x alcohol):")
for rlabel, rmask in CAP_ROWS:
    cnts = {a: int((rmask & (df.alc == a).values).sum()) for a, _ in ALC_COLS}
    print(f"  {rlabel:24s} {cnts}")


def wkde_max(lsm, wt):
    ok = np.isfinite(lsm) & np.isfinite(wt)
    lsm, wt = lsm[ok], wt[ok]
    if len(lsm) <= 5 or lsm.std() == 0:
        return 0.0
    return gaussian_kde(lsm, weights=wt)(np.linspace(0, XMAX, 300)).max()


def draw_panel(ax, lsm, wt, color, label_stages=False, ymax=None):
    ok = np.isfinite(lsm) & np.isfinite(wt)
    lsm, wt = lsm[ok], wt[ok]
    m = lsm <= XMAX
    ax.hist(lsm[m], bins=HIST_BINS, density=True, weights=wt[m],
            color=color, alpha=0.30, edgecolor="white", linewidth=0.3)
    if len(lsm) > 5 and lsm.std() > 0:
        kde = gaussian_kde(lsm, weights=wt)
        xs = np.linspace(0, XMAX, 400)
        ax.plot(xs, kde(xs), color=color, lw=2)
    for t in THRESH.values():
        ax.axvline(t, color="0.35", ls="--", lw=0.8, zorder=1)
    if label_stages:
        y = ax.get_ylim()[1] if ymax is None else ymax
        for s, xc in {"F0": 3, "F1": 7, "F2": 9, "F3": 12.5, "F4": 20}.items():
            ax.text(xc, y * 0.93, s, ha="center", va="top", fontsize=8,
                    color="0.25", fontweight="bold")
    ax.set_xlim(0, XMAX)


# --------------------------------------------------------------------------
# FIGURE 8: grid — CAP rows x alcohol columns, weighted hist + KDE
# --------------------------------------------------------------------------
ymax = 0
for _, rmask in CAP_ROWS:
    for akey, _ in ALC_COLS:
        m = rmask & (df.alc == akey).values
        ymax = max(ymax, wkde_max(df.LSM.values[m], df.w.values[m]))
ymax *= 1.15

fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
for r, (rlabel, rmask) in enumerate(CAP_ROWS):
    for c, (akey, alabel) in enumerate(ALC_COLS):
        ax = axes[r, c]
        m = rmask & (df.alc == akey).values
        ax.set_ylim(0, ymax)
        draw_panel(ax, df.LSM.values[m], df.w.values[m],
                   ALC_COLORS[akey], label_stages=(r == 0), ymax=ymax)
        if r == 0:
            ax.set_title(alabel, fontsize=11, fontweight="bold")
        ax.text(0.97, 0.97, f"n = {int(m.sum()):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="0.3")
        if c == 0:
            ax.set_ylabel(f"{rlabel}\n\nweighted density", fontweight="bold")
        if r == 2:
            ax.set_xlabel("LSM (kPa)")
fig.suptitle("LSM distribution by CAP threshold and alcohol consumption — "
             "NHANES 2017-2020 adults 18+\nrows: without CAP threshold (top) "
             "vs. with it (CAP < / ≥ 288);  dashed lines F1=6, F2=8, F3=10, "
             "F4=15 kPa;  survey-weighted", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig8_lsm_by_cap_alcohol.png", dpi=130)
plt.close(fig)
print("\nwrote fig8_lsm_by_cap_alcohol.png")

# --------------------------------------------------------------------------
# FIGURE 9: KDE overlay — one panel per CAP row, 3 alcohol curves overlaid
# --------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, sharey=True)
xs = np.linspace(0, XMAX, 400)
ymax2 = ymax
for ax, (rlabel, rmask) in zip(axes, CAP_ROWS):
    ax.set_ylim(0, ymax2)
    for akey, alabel in ALC_COLS:
        m = rmask & (df.alc == akey).values
        lsm, wt = df.LSM.values[m], df.w.values[m]
        ok = np.isfinite(lsm) & np.isfinite(wt)
        if ok.sum() > 5 and lsm[ok].std() > 0:
            kde = gaussian_kde(lsm[ok], weights=wt[ok])
            ax.plot(xs, kde(xs), color=ALC_COLORS[akey], lw=2.2,
                    label=f"{akey} (n={int(m.sum()):,})")
    for t in THRESH.values():
        ax.axvline(t, color="0.35", ls="--", lw=0.8, zorder=1)
    for s, xc in {"F0": 3, "F1": 7, "F2": 9, "F3": 12.5, "F4": 20}.items():
        ax.text(xc, ymax2 * 0.95, s, ha="center", va="top", fontsize=8,
                color="0.25", fontweight="bold")
    ax.set_xlim(0, XMAX)
    ax.set_ylabel("weighted density")
    ax.set_title(rlabel, loc="left", fontweight="bold")
    ax.legend(title="alcohol", fontsize=9, loc="upper right")
axes[-1].set_xlabel("Liver stiffness measure, LSM (kPa)")
fig.suptitle("LSM distribution by alcohol consumption, with vs. without CAP "
             "threshold\nNHANES 2017-2020 adults 18+ (survey-weighted KDE; "
             "F1=6, F2=8, F3=10, F4=15 kPa)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig9_lsm_alcohol_overlay.png", dpi=130)
plt.close(fig)
print("wrote fig9_lsm_alcohol_overlay.png")
