"""
Gaussian-copula diagnostic for the joint distribution of median CAP and LSM
(NHANES 2017-2020 elastography, survey-weighted WTMECPRP).

Goal: help decide between a GAUSSIAN copula and an EMPIRICAL copula for
(CAP, LSM). A Gaussian copula says: after mapping each margin to its percentile
and applying the probit (inverse normal), the pair is bivariate-normal with a
single correlation rho. So:

  * PROBIT-OF-PERCENTILE space -- transform each variable to Phi^{-1}(weighted
    percentile). A Gaussian copula is then bivariate normal N(0, [[1,rho],[rho,1]]).
    We scatter the normal scores and overlay that model's probability-region
    contours. Misfit (tail clustering, asymmetry) is visible directly.
  * CAP / LSM units -- the same Gaussian-copula model mapped back to data units:
    f(x,y) = c_rho(F_X(x), F_Y(y)) * f_X(x) * f_Y(y), with empirical (KDE) margins.

For comparison, the *empirical* joint density (2-D weighted KDE = what an
empirical copula reproduces) is overlaid as dashed contours at the same
probability-mass levels.
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

# Pooled cycles: 2017-March 2020 pre-pandemic (P_, weight WTMECPRP, 3.2 yr) and
# August 2021-August 2023 (_L, weight WTMEC2YR, 2.0 yr). Per NHANES guidance for
# combining cycles of unequal length, the pooled weight is the cycle MEC weight
# times (cycle years / total years); total = 3.2 + 2.0 = 5.2.
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
    except UnicodeDecodeError:                      # some _L labels are latin1
        df, _ = pyreadstat.read_xport(os.path.join("data", fname),
                                      encoding="latin1")
    return df[cols]


def build_cycle(lux_f, demo_f, wcol, years, label):
    d = rx(lux_f, ["SEQN", "LUAXSTAT", "LUXSMED", "LUXCAPM"]).merge(
        rx(demo_f, ["SEQN", wcol]), on="SEQN")
    d = d[(d.LUAXSTAT == 1) & d.LUXSMED.notna() & d.LUXCAPM.notna()
          & (d[wcol] > 0)].copy()
    d["w"] = d[wcol] * years / TOTAL_YEARS          # pooled weight
    d["cycle"] = label
    return d[["LUXSMED", "LUXCAPM", "w", "cycle"]]


df = pd.concat([
    build_cycle("P_LUX.xpt", "P_DEMO.xpt", "WTMECPRP", 3.2, "2017-2020"),
    build_cycle("LUX_L.xpt", "DEMO_L.xpt", "WTMEC2YR", 2.0, "2021-2023"),
], ignore_index=True)
cap = df.LUXCAPM.values.astype(float)
lsm = df.LUXSMED.values.astype(float)
w = df.w.values.astype(float)
W = w.sum()
print(f"pooled n = {len(df):,}  ("
      + ", ".join(f"{c}: {int((df.cycle == c).sum()):,}"
                  for c in ("2017-2020", "2021-2023")) + ")")
print(f"sum pooled weights = {W:,.0f}")


# --------------------------------------------------------------------------
# Weighted mid-distribution ECDF  ->  percentile u in (0,1), with ties averaged
# --------------------------------------------------------------------------
def weighted_ecdf(x, w):
    """Return per-point weighted mid-rank percentile u_i, plus (vals,F) for interp."""
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o]
    csum = np.cumsum(ws)
    below = csum - ws                       # weight strictly before (within sorted)
    u_sorted = np.empty_like(xs, float)
    # average over ties: assign each tie group the same mid-rank
    i = 0
    n = len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        grp_below = below[i]                # weight strictly below the group value
        grp_wt = csum[j] - below[i]         # total weight at this value
        u_sorted[i:j + 1] = (grp_below + 0.5 * grp_wt) / W
        i = j + 1
    u = np.empty_like(u_sorted)
    u[o] = u_sorted
    # interpolation support: unique values -> their u
    uniq_vals, idx = np.unique(xs, return_index=True)
    uniq_u = u_sorted[idx]
    return u, uniq_vals, uniq_u


u_cap, cap_vals, cap_u = weighted_ecdf(cap, w)
u_lsm, lsm_vals, lsm_u = weighted_ecdf(lsm, w)
z_cap = norm.ppf(u_cap)
z_lsm = norm.ppf(u_lsm)


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return cov / np.sqrt(np.average((a - ma) ** 2, weights=w)
                         * np.average((b - mb) ** 2, weights=w))


rho = wcorr(z_cap, z_lsm, w)                       # Gaussian-copula parameter
spearman = wcorr(u_cap, u_lsm, w)                  # weighted Spearman
rho_implied_S = (6 / np.pi) * np.arcsin(rho / 2)   # Gaussian-copula-implied Spearman
print(f"Gaussian-copula rho (normal scores) = {rho:.3f}")
print(f"weighted Spearman (empirical)        = {spearman:.3f}")
print(f"Gaussian-copula-implied Spearman     = {rho_implied_S:.3f}")


# --------------------------------------------------------------------------
# Tail-concordance diagnostic (the key Gaussian weakness): empirical vs Gaussian
# --------------------------------------------------------------------------
def emp_tail(u1, u2, w, q, upper=True):
    if upper:
        joint = (u1 > q) & (u2 > q); marg = (u1 > q)
    else:
        joint = (u1 < 1 - q) & (u2 < 1 - q); marg = (u1 < 1 - q)
    return np.sum(w[joint]) / np.sum(w[marg]) if np.sum(w[marg]) > 0 else np.nan


def gauss_tail(rho, q, upper=True):
    zq = norm.ppf(q if upper else 1 - q)
    mv = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    if upper:
        joint = 1 - 2 * norm.cdf(zq) + mv.cdf([zq, zq])   # P(Z1>zq,Z2>zq)
        return joint / (1 - q)
    else:
        return mv.cdf([zq, zq]) / (1 - q)                 # P(Z1<zq,Z2<zq)


print("\nTail concordance  P(both extreme | one extreme):")
print(f"{'q':>6} {'upper emp':>10} {'upper Gauss':>12} {'lower emp':>10} {'lower Gauss':>12}")
for q in (0.90, 0.95):
    print(f"{q:>6} {emp_tail(u_cap,u_lsm,w,q,True):>10.3f} "
          f"{gauss_tail(rho,q,True):>12.3f} "
          f"{emp_tail(u_cap,u_lsm,w,q,False):>10.3f} "
          f"{gauss_tail(rho,q,False):>12.3f}")


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
PROBS = [0.25, 0.50, 0.75, 0.95]
C_GAUSS = "#2166ac"     # Gaussian-copula model  (solid)
C_EMP = "#b2182b"       # empirical density      (dashed)
C_PTS = "#111111"

# Jitter points for DISPLAY ONLY (the fit/contours use the real data): CAP is
# recorded to 1 dB/m and LSM to 0.1 kPa, so ties band the scatter. Spread each
# point uniformly within its recording resolution, then re-map to normal scores.
_rng = np.random.default_rng(42)
cap_j = cap + _rng.uniform(-0.5, 0.5, len(cap))
lsm_j = lsm + _rng.uniform(-0.05, 0.05, len(lsm))
z_cap_disp = norm.ppf(np.clip(np.interp(cap_j, cap_vals, cap_u), 1e-4, 1 - 1e-4))
z_lsm_disp = norm.ppf(np.clip(np.interp(lsm_j, lsm_vals, lsm_u), 1e-4, 1 - 1e-4))
PT_KW = dict(s=5, c=C_PTS, alpha=0.16, edgecolors="none", zorder=1)


def hdr_levels(dens, cell_area):
    """Density thresholds enclosing each probability mass in PROBS (HDR)."""
    d = dens.ravel()
    ds = np.sort(d)[::-1]
    cum = np.cumsum(ds) * cell_area
    cum /= cum[-1]
    lv = {}
    for p in PROBS:
        k = min(np.searchsorted(cum, p), len(ds) - 1)
        lv[p] = ds[k]
    # ensure strictly increasing levels for contour()
    return lv


def draw_prob_contours(ax, X, Y, dens, color, ls):
    cell = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    lv = hdr_levels(dens, cell)
    levels = sorted(set(round(v, 12) for v in lv.values()))
    cs = ax.contour(X, Y, dens, levels=levels, colors=color,
                    linestyles=ls, linewidths=1.8)
    # label each contour with the probability mass it encloses
    lab = {v: f"{int(p*100)}%" for p, v in lv.items()}
    ax.clabel(cs, fmt=lambda v: lab.get(round(v, 12), ""), fontsize=8)
    return cs


# ==========================================================================
# FIGURE 10: probit-of-percentile space
# ==========================================================================
fig, ax = plt.subplots(figsize=(8.2, 8))
lim = 3.6
gg = np.linspace(-lim, lim, 240)
Xz, Yz = np.meshgrid(gg, gg)
# Gaussian-copula model = bivariate normal N(0, [[1,rho],[rho,1]])
pos = np.dstack([Xz, Yz])
gauss_dens = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).pdf(pos)
# empirical density of the normal scores (what an empirical copula reproduces)
kde_z = gaussian_kde(np.vstack([z_cap, z_lsm]), weights=w)
emp_dens = kde_z(np.vstack([Xz.ravel(), Yz.ravel()])).reshape(Xz.shape)

ax.scatter(z_cap_disp, z_lsm_disp, **PT_KW)
draw_prob_contours(ax, Xz, Yz, gauss_dens, C_GAUSS, "solid")
draw_prob_contours(ax, Xz, Yz, emp_dens, C_EMP, "dashed")
ax.axhline(0, color="0.85", lw=0.8, zorder=0)
ax.axvline(0, color="0.85", lw=0.8, zorder=0)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xlabel("probit of CAP percentile,  Φ⁻¹(F̂_CAP)")
ax.set_ylabel("probit of LSM percentile,  Φ⁻¹(F̂_LSM)")
ax.set_title("Gaussian copula in probit-of-percentile space\n"
             f"points = normal scores (n={len(df):,});  ρ = {rho:.3f}  "
             "(survey-weighted)", fontweight="bold")
ax.legend(handles=[
    Line2D([], [], color=C_GAUSS, lw=2, label="Gaussian copula (bivariate normal)"),
    Line2D([], [], color=C_EMP, lw=1.8, ls="--", label="empirical density (2-D KDE)")],
    loc="upper left", fontsize=9, framealpha=0.9)
fig.text(0.01, 0.005, "points jittered within recording resolution "
         "(CAP ±0.5 dB/m, LSM ±0.05 kPa) for display; fit uses raw data",
         fontsize=7.5, color="0.45")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("fig10_copula_probit.png", dpi=130)
plt.close(fig)
print("\nwrote fig10_copula_probit.png")

# ==========================================================================
# FIGURE 11: CAP / LSM units
# ==========================================================================
CAP_LO, CAP_HI = 100.0, 400.0
LSM_LO, LSM_HI = 2.0, 28.0
gx = np.linspace(CAP_LO, CAP_HI, 220)
gy = np.linspace(LSM_LO, LSM_HI, 220)
Xd, Yd = np.meshgrid(gx, gy)

# Gaussian-copula model density in data units, with KDE margins
fX = gaussian_kde(cap, weights=w)
fY = gaussian_kde(lsm, weights=w)
EPS = 1e-4
ug = np.clip(np.interp(gx, cap_vals, cap_u), EPS, 1 - EPS)
vg = np.clip(np.interp(gy, lsm_vals, lsm_u), EPS, 1 - EPS)
z1 = norm.ppf(ug); z2 = norm.ppf(vg)
Z1, Z2 = np.meshgrid(z1, z2)
# copula density c_rho(u,v) = phi2(z1,z2;rho) / (phi(z1) phi(z2))
mv = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
cop = mv.pdf(np.dstack([Z1, Z2])) / (norm.pdf(Z1) * norm.pdf(Z2))
gauss_data = cop * np.outer(fY(gy), np.ones_like(gx)) * np.outer(np.ones_like(gy), fX(gx))
# empirical joint density (2-D weighted KDE)
kde_xy = gaussian_kde(np.vstack([cap, lsm]), weights=w)
emp_data = kde_xy(np.vstack([Xd.ravel(), Yd.ravel()])).reshape(Xd.shape)

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(cap_j, lsm_j, **PT_KW)
draw_prob_contours(ax, Xd, Yd, gauss_data, C_GAUSS, "solid")
draw_prob_contours(ax, Xd, Yd, emp_data, C_EMP, "dashed")
# reference lines linking to the fibrosis / steatosis analysis
for t in (6, 8, 10, 15):
    ax.axhline(t, color="0.85", lw=0.7, zorder=0)
ax.axvline(288, color="0.85", lw=0.7, zorder=0)
ax.text(392, 15.3, "F4", fontsize=7, color="0.5", ha="right")
ax.text(290, 27.3, "CAP 288", fontsize=7, color="0.5")
ax.set_yscale("log")   # spread the crowded low-LSM region for easier comparison
ax.set_xlim(CAP_LO, CAP_HI); ax.set_ylim(LSM_LO, LSM_HI)
ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 8, 10, 15, 20, 28]))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("median CAP (dB/m)")
ax.set_ylabel("median LSM (kPa, log scale)")
ax.set_title("Gaussian copula in CAP / LSM units\n"
             f"points = observed (n={len(df):,});  KDE margins × Gaussian "
             "dependence  (survey-weighted)", fontweight="bold")
ax.legend(handles=[
    Line2D([], [], color=C_GAUSS, lw=2, label="Gaussian copula (KDE margins)"),
    Line2D([], [], color=C_EMP, lw=1.8, ls="--", label="empirical joint density (2-D KDE)")],
    loc="upper right", fontsize=9, framealpha=0.9)
fig.text(0.01, 0.005, "points jittered within recording resolution "
         "(CAP ±0.5 dB/m, LSM ±0.05 kPa) for display; fit uses raw data",
         fontsize=7.5, color="0.45")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig("fig11_copula_capunits.png", dpi=130)
plt.close(fig)
print("wrote fig11_copula_capunits.png")

# ==========================================================================
# FIGURE 12: presentation slide version (16:9, big text, LSM capped at 15 to
#            show all of F3, bold fibrosis-stage bands, and the survey-weighted
#            share of participants in each CAP-side x fibrosis-band region).
# ==========================================================================
Y_MAX = 15.0
# fibrosis-stage bands on LSM (severity ramp): name, range label, lo, hi, fill, ink
BANDS = [("F0", "< 6 kPa", 2.0, 6.0, "#dcedc8", "#33691e"),
         ("F1", "6–8 kPa", 6.0, 8.0, "#fff3c4", "#8d6e00"),
         ("F2", "8–10 kPa", 8.0, 10.0, "#ffe0b2", "#bf560a"),
         ("F3", "10–15 kPa", 10.0, Y_MAX, "#ffcdd2", "#b71c1c")]

# survey-weighted share of participants in each region (CAP side x fibrosis band)
edges = [0, 6, 8, 10, 15, np.inf]
names5 = ["F0", "F1", "F2", "F3", "F4"]
band_of = np.array(names5)[np.searchsorted(edges, lsm, side="right") - 1]
cap_hi = cap >= 288
region_pct = {(side, b): w[(msk) & (band_of == b)].sum() / W * 100
              for side, msk in (("lo", ~cap_hi), ("hi", cap_hi))
              for b in names5}
print("\nWeighted % of participants by region (CAP side x fibrosis band):")
for b in names5:
    print(f"  {b}: CAP<288 {region_pct[('lo', b)]:4.1f}%   "
          f"CAP≥288 {region_pct[('hi', b)]:4.1f}%")


def draw_slide_contours(ax, X, Y, dens, color, ls, lw):
    cell = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])
    lv = hdr_levels(dens, cell)
    levels = sorted(set(round(v, 12) for v in lv.values()))
    cs = ax.contour(X, Y, dens, levels=levels, colors=color,
                    linestyles=ls, linewidths=lw, zorder=4)
    lab = {v: f"{int(p*100)}%" for p, v in lv.items()}
    ax.clabel(cs, fmt=lambda v: lab.get(round(v, 12), ""), fontsize=12)
    return cs


fig, ax = plt.subplots(figsize=(13.33, 7.5))
# fibrosis bands + bold LEFT-edge labels + region-share numbers
for name, rng, lo, hi, fill, txt in BANDS:
    yc = np.sqrt(lo * hi)                            # band center on log axis
    ax.axhspan(lo, hi, color=fill, alpha=0.7, zorder=0)
    ax.text(103, yc, f"{name}\n{rng}", ha="left", va="center",
            fontsize=14, fontweight="bold", color=txt, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=txt, alpha=0.9))
    for side, xpos in (("lo", 212), ("hi", 344)):    # CAP<288 vs CAP>=288
        ax.text(xpos, yc, f"{region_pct[(side, name)]:.0f}%", ha="center",
                va="center", fontsize=13, fontweight="bold", color="#1a1a1a",
                zorder=6, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    edgecolor=txt, alpha=0.85))
for t in (6, 8, 10):
    ax.axhline(t, color="0.35", lw=1.2, zorder=1)

ax.scatter(cap_j, lsm_j, s=5, c=C_PTS, alpha=0.13, edgecolors="none", zorder=2)
ax.axvline(288, color="0.35", lw=1.4, ls=":", zorder=3)
# CAP>=288 label moved to the BOTTOM (was overlapping the top boundary/text)
ax.text(293, 2.15, "CAP ≥ 288 (steatosis) →", fontsize=12.5, color="0.25",
        style="italic", ha="left", va="bottom", zorder=6)
ax.text(283, 2.15, "← below 288", fontsize=12.5, color="0.25",
        style="italic", ha="right", va="bottom", zorder=6)
draw_slide_contours(ax, Xd, Yd, gauss_data, C_GAUSS, "solid", 3.0)
draw_slide_contours(ax, Xd, Yd, emp_data, C_EMP, "dashed", 2.8)

ax.set_yscale("log")
ax.set_xlim(CAP_LO, CAP_HI); ax.set_ylim(LSM_LO, Y_MAX)
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
           markersize=8, alpha=0.5, label=f"Participants (n={len(df):,})")],
    loc="upper center", ncol=3, fontsize=13, framealpha=0.95)
fig.suptitle("Joint distribution of CAP and liver stiffness: "
             "Gaussian copula vs. empirical", fontsize=21, fontweight="bold")
ax.set_title(f"NHANES 2017–2020 + 2021–2023, survey-weighted    ·    "
             f"Gaussian ρ = {rho:.2f}, but empirical adds upper-tail dependence",
             fontsize=13.5, color="0.3")
fig.text(0.5, 0.032, "Boxed % = survey-weighted share of participants in each "
         "region (CAP side × fibrosis band).", fontsize=10, color="0.45",
         ha="center")
fig.text(0.5, 0.008, f"F4 (≥15 kPa) is above the axis: "
         f"{region_pct[('lo','F4')]:.1f}% below / {region_pct[('hi','F4')]:.1f}% "
         "above CAP 288.    Curve labels 25/50/75/95% = probability contours.",
         fontsize=10, color="0.45", ha="center")
fig.tight_layout(rect=[0, 0.06, 1, 0.955])
fig.savefig("fig12_copula_slide.png", dpi=150)
plt.close(fig)
print("wrote fig12_copula_slide.png")
