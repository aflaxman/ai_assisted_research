"""
Presentation slide (fig12 design) for the CAP-LSM copula, restricted to
MALES AGED 60-80. Everything -- Gaussian ρ, KDE margins, empirical density, and
the region shares -- is recomputed WITHIN this subgroup, survey-weighted (pooled
NHANES 2017-2020 + 2021-2023). See cap_lsm_copula.py for the pooling rationale.
"""

import os
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, FixedLocator, NullLocator
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
        d, _ = pyreadstat.read_xport(os.path.join("data", fname))
    except UnicodeDecodeError:
        d, _ = pyreadstat.read_xport(os.path.join("data", fname), encoding="latin1")
    return d[cols]


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

# ---- restrict to MALES aged 60-80 --------------------------------------
SUBGROUP = "Males aged 60–80"
sub = df[(df.sex == 1.0) & (df.age >= 60) & (df.age <= 80)].copy()
cap = sub.cap.values.astype(float)
lsm = sub.lsm.values.astype(float)
w = sub.w.values.astype(float)
W = w.sum()
print(f"{SUBGROUP}: n = {len(sub):,}   sum weights = {W:,.0f}")


def weighted_ecdf(x, w):
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o]
    csum = np.cumsum(ws); below = csum - ws
    us = np.empty_like(xs, float)
    i, n = 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        us[i:j + 1] = (below[i] + 0.5 * (csum[j] - below[i])) / W
        i = j + 1
    u = np.empty_like(us); u[o] = us
    vals, idx = np.unique(xs, return_index=True)
    return u, vals, us[idx]


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return cov / np.sqrt(np.average((a - ma) ** 2, weights=w)
                         * np.average((b - mb) ** 2, weights=w))


u_cap, cap_vals, cap_u = weighted_ecdf(cap, w)
u_lsm, lsm_vals, lsm_u = weighted_ecdf(lsm, w)
rho = wcorr(norm.ppf(u_cap), norm.ppf(u_lsm), w)
print(f"Gaussian-copula ρ (subgroup) = {rho:.3f}")

_rng = np.random.default_rng(42)
cap_j = cap + _rng.uniform(-0.5, 0.5, len(cap))
lsm_j = lsm + _rng.uniform(-0.05, 0.05, len(lsm))

# ---- densities on a CAP x LSM grid -------------------------------------
CAP_LO, CAP_HI = 100.0, 400.0
gx = np.linspace(CAP_LO, CAP_HI, 220)
gy = np.linspace(2.0, 28.0, 220)
Xd, Yd = np.meshgrid(gx, gy)
fX = gaussian_kde(cap, weights=w)
fY = gaussian_kde(lsm, weights=w)
EPS = 1e-4
ug = np.clip(np.interp(gx, cap_vals, cap_u), EPS, 1 - EPS)
vg = np.clip(np.interp(gy, lsm_vals, lsm_u), EPS, 1 - EPS)
Z1, Z2 = np.meshgrid(norm.ppf(ug), norm.ppf(vg))
mv = multivariate_normal([0, 0], [[1, rho], [rho, 1]])
cop = mv.pdf(np.dstack([Z1, Z2])) / (norm.pdf(Z1) * norm.pdf(Z2))
gauss_data = cop * np.outer(fY(gy), np.ones_like(gx)) * np.outer(np.ones_like(gy), fX(gx))
emp_data = gaussian_kde(np.vstack([cap, lsm]), weights=w)(
    np.vstack([Xd.ravel(), Yd.ravel()])).reshape(Xd.shape)

PROBS = [0.25, 0.50, 0.75, 0.95]
C_GAUSS, C_EMP, C_PTS = "#2166ac", "#b2182b", "#111111"


def hdr_levels(dens, cell):
    ds = np.sort(dens.ravel())[::-1]
    cum = np.cumsum(ds) * cell; cum /= cum[-1]
    return {p: ds[min(np.searchsorted(cum, p), len(ds) - 1)] for p in PROBS}


def draw_slide_contours(ax, X, Y, dens, color, ls, lw):
    cell = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    lv = hdr_levels(dens, cell)
    levels = sorted(set(round(v, 12) for v in lv.values()))
    cs = ax.contour(X, Y, dens, levels=levels, colors=color,
                    linestyles=ls, linewidths=lw, zorder=4)
    lab = {v: f"{int(p*100)}%" for p, v in lv.items()}
    ax.clabel(cs, fmt=lambda v: lab.get(round(v, 12), ""), fontsize=12)


# ---- region shares (within subgroup) -----------------------------------
Y_MAX = 15.0
BANDS = [("F0", "< 6 kPa", 2.0, 6.0, "#dcedc8", "#33691e"),
         ("F1", "6–8 kPa", 6.0, 8.0, "#fff3c4", "#8d6e00"),
         ("F2", "8–10 kPa", 8.0, 10.0, "#ffe0b2", "#bf560a"),
         ("F3", "10–15 kPa", 10.0, Y_MAX, "#ffcdd2", "#b71c1c")]
edges = [0, 6, 8, 10, 15, np.inf]
names5 = ["F0", "F1", "F2", "F3", "F4"]
band_of = np.array(names5)[np.searchsorted(edges, lsm, side="right") - 1]
cap_hi = cap >= 288
region_pct = {(side, b): w[(msk) & (band_of == b)].sum() / W * 100
              for side, msk in (("lo", ~cap_hi), ("hi", cap_hi))
              for b in names5}
print("Region shares (CAP<288 / CAP≥288):")
for b in names5:
    print(f"  {b}: {region_pct[('lo', b)]:.1f}% / {region_pct[('hi', b)]:.1f}%")

# ==========================================================================
# SLIDE
# ==========================================================================
fig, ax = plt.subplots(figsize=(13.33, 7.5))
for name, rng, lo, hi, fill, txt in BANDS:
    yc = np.sqrt(lo * hi)
    ax.axhspan(lo, hi, color=fill, alpha=0.7, zorder=0)
    ax.text(103, yc, f"{name}\n{rng}", ha="left", va="center", fontsize=14,
            fontweight="bold", color=txt, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=txt, alpha=0.9))
    for side, xpos in (("lo", 212), ("hi", 344)):
        ax.text(xpos, yc, f"{region_pct[(side, name)]:.0f}%", ha="center",
                va="center", fontsize=13, fontweight="bold", color="#1a1a1a",
                zorder=6, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    edgecolor=txt, alpha=0.85))
for t in (6, 8, 10):
    ax.axhline(t, color="0.35", lw=1.2, zorder=1)
ax.scatter(cap_j, lsm_j, s=6, c=C_PTS, alpha=0.16, edgecolors="none", zorder=2)
ax.axvline(288, color="0.35", lw=1.4, ls=":", zorder=3)
ax.text(293, 2.15, "CAP ≥ 288 (steatosis) →", fontsize=12.5, color="0.25",
        style="italic", ha="left", va="bottom", zorder=6)
ax.text(283, 2.15, "← below 288", fontsize=12.5, color="0.25",
        style="italic", ha="right", va="bottom", zorder=6)
draw_slide_contours(ax, Xd, Yd, gauss_data, C_GAUSS, "solid", 3.0)
draw_slide_contours(ax, Xd, Yd, emp_data, C_EMP, "dashed", 2.8)

ax.set_yscale("log")
ax.set_xlim(CAP_LO, CAP_HI); ax.set_ylim(2.0, Y_MAX)
ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 8, 10, 15]))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(NullLocator())
ax.tick_params(axis="both", labelsize=15)
ax.set_xlabel("Median CAP  (dB/m)  — steatosis marker", fontsize=17)
ax.set_ylabel("Median LSM  (kPa, log scale)", fontsize=17)
ax.legend(handles=[
    Line2D([], [], color=C_GAUSS, lw=3, label="Gaussian copula"),
    Line2D([], [], color=C_EMP, lw=2.8, ls="--", label="Empirical density"),
    Line2D([], [], marker="o", color="none", markerfacecolor=C_PTS,
           markersize=8, alpha=0.5, label=f"Participants (n={len(sub):,})")],
    loc="upper center", ncol=3, fontsize=13, framealpha=0.95)
fig.suptitle("CAP and liver stiffness — Gaussian copula vs. empirical:  "
             f"{SUBGROUP}", fontsize=21, fontweight="bold")
ax.set_title(f"NHANES 2017–2020 + 2021–2023, survey-weighted    ·    "
             f"Gaussian ρ = {rho:.2f}, but empirical adds upper-tail dependence",
             fontsize=13.5, color="0.3")
fig.text(0.5, 0.032, "Boxed % = survey-weighted share of this subgroup in each "
         "region (CAP side × fibrosis band).", fontsize=10, color="0.45",
         ha="center")
fig.text(0.5, 0.008, f"F4 (≥15 kPa) is above the axis: "
         f"{region_pct[('lo','F4')]:.1f}% below / {region_pct[('hi','F4')]:.1f}% "
         "above CAP 288.    Curve labels 25/50/75/95% = probability contours.",
         fontsize=10, color="0.45", ha="center")
fig.tight_layout(rect=[0, 0.06, 1, 0.955])
fig.savefig("fig15_copula_slide_m60_80.png", dpi=150)
plt.close(fig)
print("wrote fig15_copula_slide_m60_80.png")
