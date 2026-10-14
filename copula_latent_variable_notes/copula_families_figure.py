"""Explainer figure: Gaussian vs Gumbel vs Joe copulas at the same Kendall's tau.

Joe is sampled by inverting its conditional CDF h(v | u) with bisection
(Joe is not in copulas.py).  Run:  uv run python copula_families_figure.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats as sps

import copulas as cp

COLORS = {"gaussian": "#2a78d6", "gumbel": "#eb6834", "joe": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "legend.fontsize": 9, "savefig.dpi": 160, "savefig.bbox": "tight",
})


# --- Joe copula -------------------------------------------------------------

def joe_cdf(u, v, d):
    ub, vb = 1.0 - u, 1.0 - v
    return 1.0 - (ub ** d + vb ** d - ub ** d * vb ** d) ** (1.0 / d)


def joe_h(u, v, d):
    """dC/du = P(V <= v | U = u)."""
    ub, vb = 1.0 - u, 1.0 - v
    A = ub ** d + vb ** d - ub ** d * vb ** d
    return A ** (1.0 / d - 1.0) * ub ** (d - 1.0) * (1.0 - vb ** d)


def joe_ktau(d, terms=20000):
    k = np.arange(1, terms + 1, dtype=float)
    return 1.0 - 4.0 * np.sum(1.0 / (k * (d * k + 2.0) * (d * (k - 1.0) + 2.0)))


def joe_delta_from_ktau(t):
    return optimize.brentq(lambda d: joe_ktau(d) - t, 1.0001, 60.0)


def joe_sample(d, n, rng):
    u, w = rng.uniform(size=n), rng.uniform(size=n)
    lo, hi = np.zeros(n), np.ones(n)
    for _ in range(60):                        # bisection on the increasing function h(. | u)
        mid = 0.5 * (lo + hi)
        above = joe_h(u, mid, d) > w
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return u, 0.5 * (lo + hi)


# --- numbers ----------------------------------------------------------------

KTAU = 0.5
rho = float(cp.ktau_to_param("gaussian", KTAU))
d_gumbel = float(cp.ktau_to_param("gumbel", KTAU))
d_joe = joe_delta_from_ktau(KTAU)
lam_gumbel = 2 - 2 ** (1 / d_gumbel)
lam_joe = 2 - 2 ** (1 / d_joe)


def cond_exceed(q, fam):
    """P(V > q | U > q) = (1 - 2q + C(q, q)) / (1 - q)."""
    if fam == "gaussian":
        c = cp.cdf(q, q, "gaussian", KTAU)
    elif fam == "gumbel":
        c = cp.cdf(q, q, "gumbel", KTAU)
    else:
        c = joe_cdf(q, q, d_joe)
    return (1 - 2 * q + c) / (1 - q)


print(f"Kendall's tau = {KTAU}:  Gaussian rho = {rho:.3f};  Gumbel delta = {d_gumbel:.3f}, lambda_U = {lam_gumbel:.3f};"
      f"  Joe delta = {d_joe:.3f}, lambda_U = {lam_joe:.3f}")
for q in (0.5, 0.9, 0.95, 0.99, 0.999):
    print(f"P(V > {q} | U > {q}):  " + "  ".join(f"{fam} {cond_exceed(q, fam):.3f}" for fam in ("gaussian", "gumbel", "joe")))
# also the lower corner: P(V < q | U < q) for small q = symmetry for Gaussian, independence-like for Gumbel/Joe
for q in (0.05, 0.01):
    vals = []
    for fam in ("gaussian", "gumbel", "joe"):
        c = cp.cdf(q, q, fam, KTAU) if fam != "joe" else joe_cdf(q, q, d_joe)
        vals.append(f"{fam} {c / q:.3f}")
    print(f"P(V < {q} | U < {q}):  " + "  ".join(vals))

# --- figure -----------------------------------------------------------------

rng = np.random.default_rng(0)
n = 2500
samples = {
    "gaussian": cp.sample("gaussian", KTAU, n, rng),
    "gumbel": cp.sample("gumbel", KTAU, n, rng),
    "joe": joe_sample(d_joe, n, rng),
}
titles = {
    "gaussian": f"Gaussian, rho = {rho:.2f}\nsymmetric, no tail dependence",
    "gumbel": f"Gumbel, delta = {d_gumbel:.1f}\nupper-tail dependence {lam_gumbel:.2f}",
    "joe": f"Joe, delta = {d_joe:.2f}\nupper-tail dependence {lam_joe:.2f}",
}
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
fig.subplots_adjust(wspace=0.3)
for ax, fam in zip(axes[:3], ("gaussian", "gumbel", "joe")):
    u, v = samples[fam]
    ax.scatter(sps.norm.ppf(u), sps.norm.ppf(v), s=7, lw=0, color=COLORS[fam], alpha=0.5)
    ax.set_title(titles[fam])
    ax.set_xlabel("latent variable 1 (normal scores)")
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-3.7, 3.7)
axes[0].set_ylabel("latent variable 2 (normal scores)")
ax = axes[3]
qs = np.concatenate([np.linspace(0.5, 0.99, 120), np.linspace(0.99, 0.9995, 60)])
for fam, lab in (("gaussian", "Gaussian"), ("gumbel", "Gumbel"), ("joe", "Joe")):
    ax.plot(qs, [cond_exceed(q, fam) for q in qs], color=COLORS[fam], lw=2, label=lab)
ax.axhline(lam_joe, color=COLORS["joe"], lw=0.8, ls=":")
ax.axhline(lam_gumbel, color=COLORS["gumbel"], lw=0.8, ls=":")
ax.text(0.98, lam_joe + 0.02, f"limit {lam_joe:.2f}", fontsize=8, color=COLORS["joe"], ha="right")
ax.text(0.98, lam_gumbel - 0.05, f"limit {lam_gumbel:.2f}", fontsize=8, color=COLORS["gumbel"], ha="right")
ax.text(0.98, 0.07, "limit 0", fontsize=8, color=COLORS["gaussian"], ha="right")
ax.set_xlabel("quantile q")
ax.set_ylabel("P(var. 2 > its q-quantile | var. 1 > its q-quantile)")
ax.set_title("Tail dependence\nconditional exceedance as q -> 1")
ax.set_ylim(0, 1)
ax.legend(loc="lower left")
fig.suptitle(f"Three copulas with the same Kendall's tau = {KTAU} and the same N(0, 1) margins",
             fontsize=11.5, color=INK, fontweight="bold", y=1.04)
fig.savefig("figures/copula_families.png")
print("wrote figures/copula_families.png")
