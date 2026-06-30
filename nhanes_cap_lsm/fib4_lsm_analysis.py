"""
Sensitivity & specificity of FIB-4 for LSM-defined liver fibrosis in
NHANES 2017-2020 (pre-pandemic), WITH and WITHOUT a CAP steatosis threshold.

Index test:      FIB-4 = (age * AST) / (platelets * sqrt(ALT))
                 cutoffs 1.30 (rule-out), 2.67 (rule-in), 3.25 (Sterling).
Reference std:   LSM (FibroScan) fibrosis stage, this project's bins:
                 >=F2 significant = LSM >= 8;  >=F3 advanced = LSM >= 10 (primary);
                 F4 cirrhosis = LSM >= 15 kPa.
Populations:     "without CAP threshold" = all; "with CAP threshold" = CAP >= 288
                 (hepatic steatosis, where FIB-4 is clinically applied); plus CAP < 288.

All estimates are SURVEY-WEIGHTED (WTMECPRP). 95% CIs are DESIGN-BASED, via
Taylor linearization of the domain ratio estimator using SDMVSTRA / SDMVPSU
(with-replacement PSU approximation; logit transform to keep CIs in [0,1]).

Variables: P_LUX (LUXSMED, LUXCAPM, LUAXSTAT), P_DEMO (RIDAGEYR, WTMECPRP,
SDMVPSU, SDMVSTRA), P_BIOPRO (LBXSASSI=AST, LBXSATSI=ALT), P_CBC (LBXPLTSI=plt).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

# --------------------------------------------------------------------------
# Load / build the complete-case analysis dataset
# --------------------------------------------------------------------------
CDC = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
os.makedirs("data", exist_ok=True)
for fname in ("P_LUX.xpt", "P_DEMO.xpt", "P_BIOPRO.xpt", "P_CBC.xpt"):
    dest = os.path.join("data", fname)
    if not os.path.exists(dest):
        import requests
        print(f"downloading {fname} ...")
        r = requests.get(f"{CDC}/{fname}", timeout=180); r.raise_for_status()
        open(dest, "wb").write(r.content)

lux = pd.read_sas("data/P_LUX.xpt")[["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"]]
demo = pd.read_sas("data/P_DEMO.xpt")[
    ["SEQN", "RIDAGEYR", "WTMECPRP", "SDMVPSU", "SDMVSTRA"]]
bio = pd.read_sas("data/P_BIOPRO.xpt")[["SEQN", "LBXSASSI", "LBXSATSI"]]
cbc = pd.read_sas("data/P_CBC.xpt")[["SEQN", "LBXPLTSI"]]

df = (lux.merge(demo, on="SEQN").merge(bio, on="SEQN", how="left")
         .merge(cbc, on="SEQN", how="left"))
df = df.rename(columns={"LUXSMED": "LSM", "LUXCAPM": "CAP", "RIDAGEYR": "age",
                        "WTMECPRP": "w", "LBXSASSI": "AST", "LBXSATSI": "ALT",
                        "LBXPLTSI": "PLT"})

# valid elastography + complete FIB-4 components + positive exam weight
df = df[(df.LUAXSTAT == 1) & df.LSM.notna() & df.CAP.notna()
        & (df.AST > 0) & (df.ALT > 0) & (df.PLT > 0)
        & df.age.notna() & (df.w > 0)].copy()

df["FIB4"] = (df.age * df.AST) / (df.PLT * np.sqrt(df.ALT))
df["cap_hi"] = df.CAP >= 288

# --------------------------------------------------------------------------
# Reference targets, FIB-4 cutoffs, populations
# --------------------------------------------------------------------------
TARGETS = [("≥F2 (LSM≥8)", 8.0), ("≥F3 (LSM≥10)", 10.0), ("F4 (LSM≥15)", 15.0)]
PRIMARY = "≥F3 (LSM≥10)"
CUTOFFS = [1.30, 2.67, 3.25]
POPS = [("All (no CAP threshold)", np.ones(len(df), bool)),
        ("CAP < 288", (~df.cap_hi).values),
        ("CAP ≥ 288 (steatosis)", df.cap_hi.values)]

strata = df.SDMVSTRA.values
psu = df.SDMVPSU.values
w = df.w.values
fib4 = df.FIB4.values

# design degrees of freedom = (#PSUs) - (#strata)  [full analysis design]
_pairs = set(zip(strata, psu))
N_PSU = len(_pairs)
N_STRATA = len(set(strata))
DESIGN_DF = N_PSU - N_STRATA


def svy_prop(num_mask, den_mask):
    """Survey-weighted proportion P(num | den) with design-based SE & 95% CI.

    Ratio estimator R = sum(w | num) / sum(w | den), num_mask subset of den_mask.
    Taylor linearization over strata/PSU (with-replacement); logit-transform CI.
    Returns (p, se, lo, hi, n_unweighted_denominator).
    """
    num = num_mask.astype(float)
    den = den_mask.astype(float)
    X = np.sum(w * den)
    if X == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    p = np.sum(w * num) / X
    # linearized residual: e_i = den_i*(num_i/den_i - p) handled as y - R*x
    e = num - p * den               # 0 outside denominator domain
    a = w * e                       # per-person score contribution
    # stratified, with-replacement variance of the total sum(a), then /X^2
    var_total = 0.0
    df_strat = pd.DataFrame({"h": strata, "psu": psu, "a": a})
    for h, gh in df_strat.groupby("h"):
        psu_tot = gh.groupby("psu").a.sum().values
        n_h = len(psu_tot)
        if n_h < 2:
            continue  # singleton stratum contributes ~0 (no PSU contrast)
        var_total += n_h / (n_h - 1) * np.sum((psu_tot - psu_tot.mean()) ** 2)
    var_R = var_total / X ** 2
    se = np.sqrt(max(var_R, 0.0))
    n_den = int(den_mask.sum())
    # logit-transformed CI (keeps within 0..1); fall back near boundaries
    tcrit = student_t.ppf(0.975, DESIGN_DF)
    if 0 < p < 1 and se > 0:
        g = np.log(p / (1 - p))
        se_g = se / (p * (1 - p))
        lo, hi = (1 / (1 + np.exp(-(g - tcrit * se_g))),
                  1 / (1 + np.exp(-(g + tcrit * se_g))))
    else:
        lo, hi = max(0.0, p - tcrit * se), min(1.0, p + tcrit * se)
    return (p, se, lo, hi, n_den)


def weighted_auroc(pos_mask, neg_mask):
    """Weighted AUROC = P(FIB4 of a random diseased > random non-diseased)."""
    fp = fib4[pos_mask]; wp = w[pos_mask]
    fn = fib4[neg_mask]; wn = w[neg_mask]
    if len(fp) == 0 or len(fn) == 0:
        return np.nan
    order = np.argsort(fn)
    fn_s, wn_s = fn[order], wn[order]
    cum = np.concatenate([[0.0], np.cumsum(wn_s)])
    total_n = cum[-1]
    # for each diseased: weight of non-diseased strictly below + half of ties
    lo = np.searchsorted(fn_s, fp, side="left")
    hi = np.searchsorted(fn_s, fp, side="right")
    below = cum[lo]
    ties = cum[hi] - cum[lo]
    conc = np.sum(wp * (below + 0.5 * ties))
    return conc / (np.sum(wp) * total_n)


# --------------------------------------------------------------------------
# Sanity checks (printed) -- catch the high-risk errors
# --------------------------------------------------------------------------
print("=" * 72)
print("SANITY CHECKS")
print(f"  n analysis sample = {len(df):,}   sum weights = {w.sum():,.0f}")
print(f"  design: {N_STRATA} strata, {N_PSU} PSUs, df = {DESIGN_DF}")
singletons = sum(1 for _, g in pd.DataFrame({'h': strata, 'psu': psu})
                 .groupby('h') if g.psu.nunique() < 2)
print(f"  singleton strata (dropped from variance): {singletons}")
print(f"  AST median={np.median(df.AST):.0f}  ALT median={np.median(df.ALT):.0f}"
      f"  PLT median={np.median(df.PLT):.0f}  -> FIB4 median={np.median(fib4):.2f}"
      f" (expect ~0.8; swap of AST/ALT would change this)")
for name, _ in TARGETS:
    pass
for label, mask in POPS:
    for tname, cut in [(PRIMARY, 10.0)]:
        dis = mask & (df.LSM.values >= cut)
        pw = np.sum(w[dis]) / np.sum(w[mask]) * 100
        pu = dis.sum() / mask.sum() * 100
        print(f"  {label:24s} {tname} prevalence: weighted {pw:4.1f}%  "
              f"unweighted {pu:4.1f}%  (n_dis={int(dis.sum())})")
print("=" * 72)

# --------------------------------------------------------------------------
# Main table: sens/spec/PPV/NPV at each cutoff x target x population
# --------------------------------------------------------------------------
rows = []
for plabel, pmask in POPS:
    for tname, cut in TARGETS:
        dis = pmask & (df.LSM.values >= cut)   # diseased domain
        well = pmask & (df.LSM.values < cut)    # non-diseased domain
        auroc = weighted_auroc(dis, well)
        for c in CUTOFFS:
            testpos = df.FIB4.values >= c
            sens = svy_prop(dis & testpos, dis)
            spec = svy_prop(well & ~testpos, well)
            ppv = svy_prop(testpos & dis & pmask, testpos & pmask)
            npv = svy_prop(~testpos & well & pmask, ~testpos & pmask)
            rows.append({
                "population": plabel, "target": tname, "cutoff": c,
                "n": int(pmask.sum()), "n_disease": int(dis.sum()),
                "auroc": round(auroc, 3),
                "sens": round(sens[0] * 100, 1),
                "sens_lo": round(sens[2] * 100, 1), "sens_hi": round(sens[3] * 100, 1),
                "spec": round(spec[0] * 100, 1),
                "spec_lo": round(spec[2] * 100, 1), "spec_hi": round(spec[3] * 100, 1),
                "ppv": round(ppv[0] * 100, 1), "npv": round(npv[0] * 100, 1),
            })
res = pd.DataFrame(rows)
res.to_csv("fib4_sensspec.csv", index=False)

# Indeterminate-zone share (1.30-2.67) per population
ind_rows = []
for plabel, pmask in POPS:
    lo = svy_prop(pmask & (df.FIB4.values < 1.30), pmask)
    ind = svy_prop(pmask & (df.FIB4.values >= 1.30) & (df.FIB4.values < 2.67), pmask)
    hi = svy_prop(pmask & (df.FIB4.values >= 2.67), pmask)
    ind_rows.append({"population": plabel, "low_<1.30_%": round(lo[0] * 100, 1),
                     "indeterminate_%": round(ind[0] * 100, 1),
                     "high_≥2.67_%": round(hi[0] * 100, 1)})
ind = pd.DataFrame(ind_rows)
ind.to_csv("fib4_zones.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\nFIB-4 risk-zone distribution (weighted % of each population):")
print(ind.to_string(index=False))
print(f"\nPRIMARY TARGET = {PRIMARY}.  Sens/spec (weighted %, 95% design-based CI):")
show = res[res.target == PRIMARY][
    ["population", "cutoff", "auroc", "sens", "sens_lo", "sens_hi",
     "spec", "spec_lo", "spec_hi", "ppv", "npv"]]
print(show.to_string(index=False))
print("\n(full table for all targets -> fib4_sensspec.csv)")


# --------------------------------------------------------------------------
# FIGURE A: weighted ROC curves, one panel per target, 3 populations overlaid
# --------------------------------------------------------------------------
def weighted_roc(dis, well):
    """Return (fpr, tpr) arrays sweeping the FIB-4 threshold, weighted."""
    f = fib4
    order = np.argsort(-f)             # high FIB4 first
    fo, dord, word, weo = f[order], dis[order], well[order], w[order]
    P = np.sum(w[dis]); N = np.sum(w[well])
    tp = np.cumsum(weo * dord) / P
    fp = np.cumsum(weo * word) / N
    # prepend origin
    return np.concatenate([[0], fp]), np.concatenate([[0], tp])


POP_COLORS = {"All (no CAP threshold)": "#444444", "CAP < 288": "#2c7fb8",
              "CAP ≥ 288 (steatosis)": "#d95f0e"}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
for ax, (tname, cut) in zip(axes, TARGETS):
    for plabel, pmask in POPS:
        dis = pmask & (df.LSM.values >= cut)
        well = pmask & (df.LSM.values < cut)
        fpr, tpr = weighted_roc(dis, well)
        auc = weighted_auroc(dis, well)
        ax.plot(fpr, tpr, color=POP_COLORS[plabel], lw=2,
                label=f"{plabel} (AUROC {auc:.2f}, n+={int(dis.sum())})")
    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=1)
    ax.set_title(f"FIB-4 ROC for {tname}", fontweight="bold")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
fig.suptitle("Weighted ROC of FIB-4 vs LSM-defined fibrosis, NHANES 2017-2020 "
             "(survey-weighted, WTMECPRP)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig4_fib4_roc.png", dpi=130)
plt.close(fig)
print("\nwrote fig4_fib4_roc.png")


# --------------------------------------------------------------------------
# FIGURE B: sens & spec (95% CI) for the PRIMARY target, by cutoff & population
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
pop_order = [p[0] for p in POPS]
offsets = {pop_order[0]: -0.22, pop_order[1]: 0.0, pop_order[2]: 0.22}
for ax, metric, mname in [(axes[0], "sens", "Sensitivity"),
                          (axes[1], "spec", "Specificity")]:
    for plabel in pop_order:
        sub = res[(res.target == PRIMARY) & (res.population == plabel)
                  ].sort_values("cutoff")
        xs = np.arange(len(CUTOFFS)) + offsets[plabel]
        y = sub[metric].values
        lo = sub[f"{metric}_lo"].values; hi = sub[f"{metric}_hi"].values
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt="o", capsize=4,
                    color=POP_COLORS[plabel], label=plabel, markersize=7, lw=1.5)
    ax.set_xticks(range(len(CUTOFFS)))
    ax.set_xticklabels([f"≥{c}" for c in CUTOFFS])
    ax.set_xlabel("FIB-4 cutoff (test positive)")
    ax.set_title(f"{mname} for {PRIMARY}", fontweight="bold")
    ax.set_ylim(0, 102); ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("weighted % (95% design-based CI)")
axes[0].legend(fontsize=9, loc="lower left", frameon=True)
fig.suptitle("FIB-4 sensitivity & specificity for advanced fibrosis "
             "(LSM≥10 kPa), with vs. without CAP threshold — NHANES 2017-2020",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig5_fib4_sensspec.png", dpi=130)
plt.close(fig)
print("wrote fig5_fib4_sensspec.png")
print("wrote fib4_sensspec.csv, fib4_zones.csv")
