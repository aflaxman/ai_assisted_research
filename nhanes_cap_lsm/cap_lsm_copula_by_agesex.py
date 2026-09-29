"""
Does the CAP-LSM *dependence* (the copula) differ by sex and age?

The marginals of CAP and LSM shift with age/sex, but "the relationship" is the
dependence structure. Probit-of-percentile space removes each subgroup's OWN
marginals (map each variable to Phi^-1 of its within-subgroup weighted
percentile), so any difference between panels is a difference in dependence, not
in levels. A Gaussian copula would be a bivariate normal with a single
correlation rho; we overlay that and the empirical (2-D KDE) density per cell.

Pooled NHANES 2017-2020 + 2021-2023, survey-weighted (see cap_lsm_copula.py for
the pooling/weight rationale).

Outputs:
  fig13_copula_probit_by_agesex.png  -- 2 sex rows x 4 age cols, probit space.
  fig14_copula_rho_by_agesex.png     -- Gaussian-copula rho vs age, by sex.
"""

import os
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, gaussian_kde, multivariate_normal

os.makedirs("data", exist_ok=True)
SOURCES = {
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles":
        ["P_LUX.xpt", "P_DEMO.xpt"],
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles":
        ["LUX_L.xpt", "DEMO_L.xpt"],
}
for base, files in SOURCES.items():
    for fname in files:
        dest = os.path.join("data", fname)
        if not os.path.exists(dest):
            import requests
            r = requests.get(f"{base}/{fname}", timeout=180); r.raise_for_status()
            open(dest, "wb").write(r.content)
TOTAL_YEARS = 5.2


def rx(fname, cols):
    try:
        df, _ = pyreadstat.read_xport(os.path.join("data", fname))
    except UnicodeDecodeError:
        df, _ = pyreadstat.read_xport(os.path.join("data", fname), encoding="latin1")
    return df[cols]


def build_cycle(lux_f, demo_f, wcol, years):
    d = rx(lux_f, ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"]).merge(
        rx(demo_f, ["SEQN", wcol, "RIDAGEYR", "RIAGENDR"]), on="SEQN")
    d = d[(d.LUAXSTAT == 1) & d.LUXSMED.notna() & d.LUXCAPM.notna()
          & (d[wcol] > 0) & d.RIDAGEYR.notna()].copy()
    d["w"] = d[wcol] * years / TOTAL_YEARS
    return d[["LUXSMED", "LUXCAPM", "w", "RIDAGEYR", "RIAGENDR"]]


df = pd.concat([
    build_cycle("P_LUX.xpt", "P_DEMO.xpt", "WTMECPRP", 3.2),
    build_cycle("LUX_L.xpt", "DEMO_L.xpt", "WTMEC2YR", 2.0),
], ignore_index=True).rename(columns={"LUXSMED": "lsm", "LUXCAPM": "cap",
                                      "RIDAGEYR": "age", "RIAGENDR": "sex"})

AGE_BINS = [12, 30, 45, 60, 81]
AGE_LABELS = ["12–29", "30–44", "45–59", "60–80"]
df["agegrp"] = pd.cut(df.age, bins=AGE_BINS, right=False, labels=AGE_LABELS)
SEXES = [(1.0, "Male"), (2.0, "Female")]
SEX_COLORS = {"Male": "#2166ac", "Female": "#c0392b"}
PROBS = [0.50, 0.90]
C_GAUSS, C_EMP, C_PTS = "#2166ac", "#b2182b", "#333333"
_rng = np.random.default_rng(7)


def weighted_ecdf(x, w):
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o]
    W = ws.sum()
    csum = np.cumsum(ws)
    below = csum - ws
    u_sorted = np.empty_like(xs, float)
    i, n = 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        u_sorted[i:j + 1] = (below[i] + 0.5 * (csum[j] - below[i])) / W
        i = j + 1
    u = np.empty_like(u_sorted); u[o] = u_sorted
    vals, idx = np.unique(xs, return_index=True)
    return u, vals, u_sorted[idx]


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return cov / np.sqrt(np.average((a - ma) ** 2, weights=w)
                         * np.average((b - mb) ** 2, weights=w))


def hdr_levels(dens, cell, probs):
    ds = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(ds) * cell; cum /= cum[-1]
    return sorted({ds[min(np.searchsorted(cum, p), len(ds) - 1)] for p in probs})


def prob_contours(ax, X, Y, dens, color, ls, lw=1.7):
    cell = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    ax.contour(X, Y, dens, levels=hdr_levels(dens, cell, PROBS),
               colors=color, linestyles=ls, linewidths=lw, zorder=4)


def subgroup(mask):
    """Return per-subgroup normal scores, jittered display scores, weights, rho."""
    cap, lsm, w = df.cap.values[mask], df.lsm.values[mask], df.w.values[mask]
    u_c, cv, cu = weighted_ecdf(cap, w)
    u_l, lv, lu = weighted_ecdf(lsm, w)
    zc, zl = norm.ppf(u_c), norm.ppf(u_l)
    rho = wcorr(zc, zl, w)
    # jitter within recording resolution, re-map through the subgroup ECDF
    cj = cap + _rng.uniform(-0.5, 0.5, len(cap))
    lj = lsm + _rng.uniform(-0.05, 0.05, len(lsm))
    zcj = norm.ppf(np.clip(np.interp(cj, cv, cu), 1e-4, 1 - 1e-4))
    zlj = norm.ppf(np.clip(np.interp(lj, lv, lu), 1e-4, 1 - 1e-4))
    n_eff = w.sum() ** 2 / np.sum(w ** 2)                      # Kish effective n
    z = np.arctanh(rho); se = 1 / np.sqrt(max(n_eff - 3, 1))   # Fisher z CI
    ci = (np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se))
    return dict(zc=zc, zl=zl, w=w, rho=rho, zcj=zcj, zlj=zlj, n=int(mask.sum()),
                n_eff=n_eff, ci=ci, spearman=wcorr(u_c, u_l, w),
                med_cap=np.interp(0.5, np.sort(u_c), cap[np.argsort(u_c)]),
                med_lsm=float(np.median(lsm)))


# grid for probit-space densities
lim = 3.6
gg = np.linspace(-lim, lim, 200)
GX, GY = np.meshgrid(gg, gg)
POS = np.dstack([GX, GY])

# ==========================================================================
# FIGURE 13: probit-space copula, small multiples (sex x age)
# ==========================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
rows = []
for r, (sxv, sxl) in enumerate(SEXES):
    for c, agl in enumerate(AGE_LABELS):
        ax = axes[r, c]
        m = (df.sex.values == sxv) & (df.agegrp.values == agl)
        s = subgroup(m)
        rows.append({"sex": sxl, "age": agl, "n": s["n"], "rho": s["rho"],
                     "rho_lo": s["ci"][0], "rho_hi": s["ci"][1],
                     "spearman": s["spearman"], "med_cap": s["med_cap"],
                     "med_lsm": s["med_lsm"]})
        ax.scatter(s["zcj"], s["zlj"], s=4, c=C_PTS, alpha=0.10,
                   edgecolors="none", zorder=1)
        gd = multivariate_normal([0, 0], [[1, s["rho"]], [s["rho"], 1]]).pdf(POS)
        ed = gaussian_kde(np.vstack([s["zc"], s["zl"]]), weights=s["w"])(
            np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        prob_contours(ax, GX, GY, gd, C_GAUSS, "solid")
        prob_contours(ax, GX, GY, ed, C_EMP, "dashed")
        ax.axhline(0, color="0.9", lw=0.7, zorder=0)
        ax.axvline(0, color="0.9", lw=0.7, zorder=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"{sxl}, {agl}   (ρ={s['rho']:.2f}, n={s['n']:,})",
                     fontsize=12, fontweight="bold")
        if c == 0:
            ax.set_ylabel(f"{sxl}\n\nprobit LSM pctile", fontweight="bold")
        if r == 1:
            ax.set_xlabel("probit CAP pctile")
res = pd.DataFrame(rows)
axes[0, 0].legend(handles=[
    Line2D([], [], color=C_GAUSS, lw=2, label="Gaussian copula (ρ per panel)"),
    Line2D([], [], color=C_EMP, lw=1.8, ls="--", label="empirical (2-D KDE)")],
    loc="upper left", fontsize=8.5, framealpha=0.9)
fig.suptitle("Does the CAP–LSM dependence differ by sex and age?  "
             "Probit-of-percentile space (marginals removed)\n"
             "contours = 50% & 90% regions;  NHANES 2017-2020 + 2021-2023, "
             "survey-weighted", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig13_copula_probit_by_agesex.png", dpi=130)
plt.close(fig)
print("wrote fig13_copula_probit_by_agesex.png")

# ==========================================================================
# FIGURE 14: Gaussian-copula rho vs age, by sex (the quantitative answer)
# ==========================================================================
rho_all = wcorr(norm.ppf(weighted_ecdf(df.cap.values, df.w.values)[0]),
                norm.ppf(weighted_ecdf(df.lsm.values, df.w.values)[0]),
                df.w.values)
fig, ax = plt.subplots(figsize=(9.5, 6.2))
xs = np.arange(len(AGE_LABELS))
for sxv, sxl in SEXES:
    sub = res[res.sex == sxl].set_index("age").loc[AGE_LABELS]
    off = -0.06 if sxl == "Male" else 0.06
    ax.errorbar(xs + off, sub.rho, yerr=[sub.rho - sub.rho_lo,
                sub.rho_hi - sub.rho], fmt="o-", capsize=4, lw=2, markersize=8,
                color=SEX_COLORS[sxl], label=sxl)
ax.axhline(rho_all, color="0.5", ls="--", lw=1.2,
           label=f"overall (ρ={rho_all:.2f})")
ax.set_xticks(xs); ax.set_xticklabels(AGE_LABELS)
ax.set_xlabel("age group (years)", fontsize=12)
ax.set_ylabel("Gaussian-copula ρ  (CAP–LSM dependence)", fontsize=12)
ax.set_ylim(0, 0.6); ax.grid(alpha=0.3)
ax.legend(fontsize=11, title="sex")
ax.set_title("CAP–LSM dependence (copula ρ) by age and sex\n"
             "approx 95% CI (Fisher z, Kish n_eff; ignores clustering)",
             fontweight="bold")
fig.tight_layout()
fig.savefig("fig14_copula_rho_by_agesex.png", dpi=130)
plt.close(fig)
print("wrote fig14_copula_rho_by_agesex.png")

pd.set_option("display.width", 200)
print(f"\nOverall Gaussian-copula ρ = {rho_all:.3f}")
print("\nBy subgroup (rho = CAP-LSM dependence; medians are marginal levels):")
print(res.round(3).to_string(index=False))
res.to_csv("copula_rho_by_agesex.csv", index=False)
print("\nwrote copula_rho_by_agesex.csv")
