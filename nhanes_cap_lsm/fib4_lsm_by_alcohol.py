"""
FIB-4 vs LSM-defined fibrosis, STRATIFIED BY ALCOHOL CONSUMPTION.

Extends fib4_lsm_analysis.py: same survey-weighted diagnostic-accuracy machinery
(verified), but the stratifier is average alcohol consumption rather than CAP.

Alcohol categories (sex-specific, MASLD/ALD-aligned):
    none      : no alcohol in the past 12 months
    moderate  : >0 but below the heavy threshold
    heavy     : >= 30 g ethanol/day (men) or >= 20 g/day (women)
Average ethanol = (drinking days/year from ALQ121) * (ALQ130 drinks/day) * 14 g / 365.
Alcohol questions are asked of adults only -> analysis is restricted to ages 18+.

NOTE: read XPTs with pyreadstat, NOT pandas.read_sas: this pandas build decodes
the value 0 as a denormalized float (~5.4e-79), which silently breaks the
ALQ121 == 0 ("none in past 12 months") test. pyreadstat decodes 0 correctly.

All estimates survey-weighted (WTMECPRP); 95% CIs design-based (Taylor
linearization over SDMVSTRA/SDMVPSU, with-replacement, logit-transformed).
"""

import os
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
os.makedirs("data", exist_ok=True)
for fname in ("P_LUX.xpt", "P_DEMO.xpt", "P_BIOPRO.xpt", "P_CBC.xpt", "P_ALQ.xpt"):
    dest = os.path.join("data", fname)
    if not os.path.exists(dest):
        import requests
        print(f"downloading {fname} ...")
        r = requests.get(f"{CDC}/{fname}", timeout=180); r.raise_for_status()
        open(dest, "wb").write(r.content)


def rx(fname, cols):
    df, _ = pyreadstat.read_xport(os.path.join("data", fname))
    return df[cols]


lux = rx("P_LUX.xpt", ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"])
demo = rx("P_DEMO.xpt", ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMECPRP",
                         "SDMVPSU", "SDMVSTRA"])
bio = rx("P_BIOPRO.xpt", ["SEQN", "LBXSASSI", "LBXSATSI"])
cbc = rx("P_CBC.xpt", ["SEQN", "LBXPLTSI"])
alq = rx("P_ALQ.xpt", ["SEQN", "ALQ111", "ALQ121", "ALQ130"])

df = (lux.merge(demo, on="SEQN").merge(bio, on="SEQN", how="left")
         .merge(cbc, on="SEQN", how="left").merge(alq, on="SEQN", how="left"))
df = df.rename(columns={"LUXSMED": "LSM", "LUXCAPM": "CAP", "RIDAGEYR": "age",
                        "WTMECPRP": "w", "LBXSASSI": "AST", "LBXSATSI": "ALT",
                        "LBXPLTSI": "PLT"})
df = df[(df.LUAXSTAT == 1) & df.LSM.notna() & df.CAP.notna()
        & (df.AST > 0) & (df.ALT > 0) & (df.PLT > 0)
        & df.age.notna() & (df.w > 0) & (df.age >= 18)].copy()
df["FIB4"] = (df.age * df.AST) / (df.PLT * np.sqrt(df.ALT))

# --- derive average ethanol g/day and alcohol category --------------------
# ALQ121 (past-12-mo drinking frequency) -> drinking days/year (category midpoints)
DAYS_PER_YEAR = {1: 365, 2: 313, 3: 182, 4: 104, 5: 52, 6: 30,
                 7: 12, 8: 9, 9: 4.5, 10: 1.5}
GRAMS_PER_DRINK = 14.0


def ethanol_gpd(r):
    if r.ALQ111 == 2:                      # never drank in lifetime
        return 0.0
    if r.ALQ121 == 0:                      # none in the past 12 months
        return 0.0
    if pd.isna(r.ALQ121):
        return 0.0 if r.ALQ111 == 2 else np.nan
    if r.ALQ121 in (77, 99):               # refused / don't know
        return np.nan
    if pd.isna(r.ALQ130) or r.ALQ130 >= 777:
        return np.nan
    return DAYS_PER_YEAR.get(int(r.ALQ121), np.nan) * r.ALQ130 * GRAMS_PER_DRINK / 365.0


df["gpd"] = df.apply(ethanol_gpd, axis=1)


def alc_cat(r):
    if pd.isna(r.gpd):
        return np.nan
    if r.gpd == 0:
        return "none"
    thr = 30.0 if r.RIAGENDR == 1 else 20.0      # men 30 g/d, women 20 g/d
    return "heavy" if r.gpd >= thr else "moderate"


df["alc"] = df.apply(alc_cat, axis=1)
df = df[df.alc.notna()].copy()

ALC = [("none", "None (no alcohol, 12 mo)"),
       ("moderate", "Moderate (<30/20 g/d)"),
       ("heavy", "Heavy (≥30 g/d M, ≥20 g/d W)")]
ALC_COLORS = {"none": "#1a9850", "moderate": "#e08214", "heavy": "#d73027"}

TARGETS = [("≥F2 (LSM≥8)", 8.0), ("≥F3 (LSM≥10)", 10.0), ("F4 (LSM≥15)", 15.0)]
PRIMARY = "≥F3 (LSM≥10)"
CUTOFFS = [1.30, 2.67, 3.25]
MIN_DIS = 20        # suppress sens/spec when the relevant domain has < 20 obs

strata = df.SDMVSTRA.values
psu = df.SDMVPSU.values
w = df.w.values
fib4 = df.FIB4.values
_pairs = set(zip(strata, psu))
DESIGN_DF = len(_pairs) - len(set(strata))


def svy_prop(num_mask, den_mask):
    """Survey-weighted P(num|den) with design-based SE & logit 95% CI."""
    num = num_mask.astype(float); den = den_mask.astype(float)
    X = np.sum(w * den)
    if X == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    p = np.sum(w * num) / X
    a = w * (num - p * den)
    var_total = 0.0
    g = pd.DataFrame({"h": strata, "psu": psu, "a": a})
    for h, gh in g.groupby("h"):
        pt = gh.groupby("psu").a.sum().values
        n_h = len(pt)
        if n_h < 2:
            continue
        var_total += n_h / (n_h - 1) * np.sum((pt - pt.mean()) ** 2)
    se = np.sqrt(max(var_total / X ** 2, 0.0))
    tcrit = student_t.ppf(0.975, DESIGN_DF)
    if 0 < p < 1 and se > 0:
        gg = np.log(p / (1 - p)); se_g = se / (p * (1 - p))
        lo, hi = (1 / (1 + np.exp(-(gg - tcrit * se_g))),
                  1 / (1 + np.exp(-(gg + tcrit * se_g))))
    else:
        lo, hi = max(0.0, p - tcrit * se), min(1.0, p + tcrit * se)
    return (p, se, lo, hi, int(den_mask.sum()))


def weighted_auroc(pos_mask, neg_mask):
    fp, wp = fib4[pos_mask], w[pos_mask]
    fn, wn = fib4[neg_mask], w[neg_mask]
    if len(fp) == 0 or len(fn) == 0:
        return np.nan
    o = np.argsort(fn); fn_s, wn_s = fn[o], wn[o]
    cum = np.concatenate([[0.0], np.cumsum(wn_s)])
    lo = np.searchsorted(fn_s, fp, "left"); hi = np.searchsorted(fn_s, fp, "right")
    conc = np.sum(wp * (cum[lo] + 0.5 * (cum[hi] - cum[lo])))
    return conc / (np.sum(wp) * cum[-1])


def weighted_roc(dis, well):
    o = np.argsort(-fib4)
    dord, word, weo = dis[o], well[o], w[o]
    P = np.sum(w[dis]); N = np.sum(w[well])
    tpr = np.cumsum(weo * dord) / P; fpr = np.cumsum(weo * word) / N
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])


# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
print("=" * 74)
print("ALCOHOL-STRATIFIED FIB-4 vs LSM  (adults 18+, survey-weighted)")
print(f"  n = {len(df):,}   design df = {DESIGN_DF}")
print(f"{'category':10s} {'n':>5s} {'wt %':>6s} {'med g/d':>8s} "
      f"{'F3+ n':>6s} {'F3+ wt%':>8s}")
for key, _ in ALC:
    m = (df.alc == key).values
    medg = np.median(df.gpd[m])
    dis = m & (df.LSM.values >= 10)
    wtpct = w[m].sum() / w.sum() * 100
    prev = w[dis].sum() / w[m].sum() * 100
    print(f"{key:10s} {int(m.sum()):5d} {wtpct:6.1f} {medg:8.1f} "
          f"{int(dis.sum()):6d} {prev:8.1f}")
print("=" * 74)

# --------------------------------------------------------------------------
# Main table: sens/spec/PPV/NPV/AUROC by alcohol x target x cutoff
# --------------------------------------------------------------------------
rows = []
for akey, alabel in ALC:
    amask = (df.alc == akey).values
    for tname, cut in TARGETS:
        dis = amask & (df.LSM.values >= cut)
        well = amask & (df.LSM.values < cut)
        auroc = weighted_auroc(dis, well)
        for c in CUTOFFS:
            tp = df.FIB4.values >= c
            sens = svy_prop(dis & tp, dis)
            spec = svy_prop(well & ~tp, well)
            ppv = svy_prop(tp & dis & amask, tp & amask)
            npv = svy_prop(~tp & well & amask, ~tp & amask)
            sens_ok = dis.sum() >= MIN_DIS
            spec_ok = well.sum() >= MIN_DIS
            rows.append({
                "alcohol": akey, "target": tname, "cutoff": c,
                "n": int(amask.sum()), "n_disease": int(dis.sum()),
                "auroc": round(auroc, 3),
                "sens": round(sens[0] * 100, 1) if sens_ok else np.nan,
                "sens_lo": round(sens[2] * 100, 1) if sens_ok else np.nan,
                "sens_hi": round(sens[3] * 100, 1) if sens_ok else np.nan,
                "spec": round(spec[0] * 100, 1) if spec_ok else np.nan,
                "spec_lo": round(spec[2] * 100, 1) if spec_ok else np.nan,
                "spec_hi": round(spec[3] * 100, 1) if spec_ok else np.nan,
                "ppv": round(ppv[0] * 100, 1), "npv": round(npv[0] * 100, 1),
            })
res = pd.DataFrame(rows)
res.to_csv("fib4_sensspec_by_alcohol.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print(f"\nPRIMARY TARGET = {PRIMARY}. Sens/spec (weighted %, 95% design-based CI):")
print(res[res.target == PRIMARY][
    ["alcohol", "cutoff", "auroc", "sens", "sens_lo", "sens_hi",
     "spec", "spec_lo", "spec_hi", "ppv", "npv"]].to_string(index=False))
print("\n(full table for all targets -> fib4_sensspec_by_alcohol.csv;"
      f" cells with disease/non-disease n < {MIN_DIS} suppressed)")

# --------------------------------------------------------------------------
# FIGURE 6: ROC by alcohol stratum, one panel per target
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
for ax, (tname, cut) in zip(axes, TARGETS):
    for akey, alabel in ALC:
        amask = (df.alc == akey).values
        dis = amask & (df.LSM.values >= cut)
        well = amask & (df.LSM.values < cut)
        if dis.sum() < MIN_DIS:
            continue
        fpr, tpr = weighted_roc(dis, well)
        auc = weighted_auroc(dis, well)
        ax.plot(fpr, tpr, color=ALC_COLORS[akey], lw=2,
                label=f"{akey} (AUROC {auc:.2f}, n+={int(dis.sum())})")
    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=1)
    ax.set_title(f"FIB-4 ROC for {tname}", fontweight="bold")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Weighted ROC of FIB-4 vs LSM fibrosis by alcohol consumption — "
             "NHANES 2017-2020 adults 18+ (WTMECPRP)", fontsize=13,
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig6_fib4_roc_by_alcohol.png", dpi=130)
plt.close(fig)
print("\nwrote fig6_fib4_roc_by_alcohol.png")

# --------------------------------------------------------------------------
# FIGURE 7: sens & spec (95% CI) for primary target, by cutoff & alcohol
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
keys = [a[0] for a in ALC]
offs = {keys[0]: -0.22, keys[1]: 0.0, keys[2]: 0.22}
for ax, metric, mname in [(axes[0], "sens", "Sensitivity"),
                          (axes[1], "spec", "Specificity")]:
    for akey in keys:
        sub = res[(res.target == PRIMARY) & (res.alcohol == akey)
                  ].sort_values("cutoff")
        xs = np.arange(len(CUTOFFS)) + offs[akey]
        y = sub[metric].values
        lo = sub[f"{metric}_lo"].values; hi = sub[f"{metric}_hi"].values
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt="o", capsize=4,
                    color=ALC_COLORS[akey], label=akey, markersize=7, lw=1.5)
    ax.set_xticks(range(len(CUTOFFS)))
    ax.set_xticklabels([f"≥{c}" for c in CUTOFFS])
    ax.set_xlabel("FIB-4 cutoff (test positive)")
    ax.set_title(f"{mname} for {PRIMARY}", fontweight="bold")
    ax.set_ylim(0, 102); ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("weighted % (95% design-based CI)")
axes[0].legend(fontsize=10, loc="upper right", title="alcohol")
fig.suptitle("FIB-4 sensitivity & specificity for advanced fibrosis (LSM≥10) "
             "by alcohol consumption — NHANES 2017-2020 adults 18+",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig7_fib4_sensspec_by_alcohol.png", dpi=130)
plt.close(fig)
print("wrote fig7_fib4_sensspec_by_alcohol.png")
print("wrote fib4_sensspec_by_alcohol.csv")
