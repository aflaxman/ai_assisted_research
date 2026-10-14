"""Figures for both studies, read from results/study1 and results/study2.

Run after the two study scripts:  uv run python make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

import copulas as cp

matplotlib.use("Agg")

# Palette (dataviz skill reference instance): categorical slots in fixed order
COLORS = {
    "gaussian": "#2a78d6",   # slot 1 blue
    "gumbel": "#eb6834",     # slot 2 orange
    "clayton": "#1baf7a",    # slot 3 aqua
    "joe": "#eda100",        # slot 4 yellow
    "2pl_only": "#898781",   # muted ink
    "truth": "#0b0b0b",      # primary ink
}
LABELS = {"gaussian": "Gaussian copula (van der Linden)", "gumbel": "Gumbel copula (Kwon & Zhang)",
          "joe": "Joe copula (upper tail)", "clayton": "Clayton copula (wrong tail)",
          "2pl_only": "2PL only (no response times)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
    "legend.fontsize": 9, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def load(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Study 1
# --------------------------------------------------------------------------

def fig_study1_dgp(out_dir, fig_dir, n_show=1000, seed=0):
    """Why a copula: the same Kendall's tau, different tail shape."""
    data = np.load(os.path.join(out_dir, "data.npz"))
    summ = load(os.path.join(out_dir, "summary.json"))
    ktau, sigma_tau = summ["truth"]["ktau"], summ["truth"]["sigma_tau"]
    rng = np.random.default_rng(seed)
    z1g, z2g = cp.sample_z("gaussian", ktau, n_show, rng)
    panels = [("gumbel", data["theta"][:n_show], data["tau"][:n_show]),
              ("gaussian", z1g, sigma_tau * z2g)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.subplots_adjust(wspace=0.28)
    for ax, (fam, th, ta) in zip(axes[:2], panels):
        ax.scatter(th, ta, s=9, lw=0, color=COLORS[fam], alpha=0.55)
        lam_u = cp.tail_dependence(fam, ktau)[1]
        ax.set_title(f"{'Gumbel' if fam == 'gumbel' else 'Gaussian'} copula\n"
                     f"tau {ktau}, upper-tail dep. {lam_u:.2f}")
        ax.set_xlabel("latent ability theta")
        ax.set_xlim(-3.6, 4.4)
        ax.set_ylim(-1.8, 1.5)
    axes[0].set_ylabel("latent speed tau")
    axes[1].set_ylabel("latent speed tau")
    # (c) conditional spread of speed given ability, from a large sample in equal-width bins
    ax = axes[2]
    edges = np.linspace(-2.5, 2.5, 21)
    mids = 0.5 * (edges[:-1] + edges[1:])
    for fam in ("gaussian", "gumbel"):
        z1, z2 = cp.sample_z(fam, ktau, 400_000, rng)
        th, ta = z1, sigma_tau * z2
        sds = [ta[(th >= lo) & (th < hi)].std() for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(mids, sds, color=COLORS[fam], lw=2, label=f"{'Gumbel' if fam == 'gumbel' else 'Gaussian'} copula")
    ax.set_xlabel("latent ability theta (bin centre)")
    ax.set_ylabel("SD of latent speed given ability")
    ax.set_title("Speed is tightly tied to ability at the top,\nloosely at the bottom, under Gumbel")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    fig.suptitle("Simulated persons with the same margins and the same Kendall's tau",
                 fontsize=11, color=INK, fontweight="bold", y=1.03)
    fig.savefig(os.path.join(fig_dir, "study1_dgp.png"))
    plt.close(fig)


def fig_study1_results(out_dir, fig_dir):
    summ = load(os.path.join(out_dir, "summary.json"))
    models = [m for m in ("2pl_only", "gaussian", "gumbel", "clayton", "joe") if m in summ["models"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) posterior predictive check: tail concentration of (score, speed proxy)
    ax = axes[0]
    obs = summ["truth"]["observed_tail_concentration"]
    labels = ["lower 10%", "lower 20%", "upper 20%", "upper 10%"]
    x = np.arange(len(labels))
    fit_models = [m for m in models if m != "2pl_only"]
    width = 0.8 / len(fit_models)
    for i, m in enumerate(fit_models):
        ppc = summ["models"][m]["ppc_tail_concentration"]
        idx = [ppc["labels"].index(lab) for lab in labels]
        pos = x + (i - (len(fit_models) - 1) / 2) * width
        lo, hi = np.array(ppc["q05"])[idx], np.array(ppc["q95"])[idx]
        mean = np.array(ppc["mean"])[idx]
        ax.vlines(pos, lo, hi, color=COLORS[m], lw=2.2, alpha=0.9)
        ax.plot(pos, mean, "_", color=COLORS[m], ms=9, mew=2, label=LABELS[m].split(" (")[0] + " fit (90% PPC)")
    ax.plot(x, [obs[lab] for lab in labels], "o", color=INK, ms=7, label="observed (simulated data)", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("tail of the observed score distribution")
    ax.set_ylabel("share of those persons also in that tail of speed")
    ax.set_title("Posterior predictive check: tail concentration")
    ax.legend(loc="lower right", fontsize=8)

    # (b) ability recovery by tercile
    ax = axes[1]
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        r = summ["models"][m]["theta"]["rmse_by_tercile"]
        pos = np.arange(3) + (i - (len(models) - 1) / 2) * width
        ax.bar(pos, r, width * 0.9, color=COLORS[m], label=LABELS[m].split(" (")[0], lw=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["lowest third", "middle third", "highest third"])
    ax.set_xlabel("true ability tercile")
    ax.set_ylabel("RMSE of posterior-mean ability")
    ax.set_ylim(0, max(summ["models"][m]["theta"]["rmse_by_tercile"][k] for m in models for k in range(3)) * 1.25)
    ax.set_title("Ability recovery")
    ax.legend(loc="upper right", ncol=1)

    # (c) posterior of Kendall's tau
    ax = axes[2]
    for i, m in enumerate([m for m in models if m != "2pl_only"]):
        post = np.load(os.path.join(out_dir, f"posterior_{m}.npz"))["ktau"]
        kde = sps.gaussian_kde(post)
        grid = np.linspace(max(0, post.min() - 0.05), min(1, post.max() + 0.05), 300)
        ax.plot(grid, kde(grid), color=COLORS[m], lw=2, label=LABELS[m].split(" (")[0])
        ax.fill_between(grid, 0, kde(grid), color=COLORS[m], alpha=0.12, lw=0)
    ax.axvline(summ["truth"]["ktau"], color=INK, lw=1.2, ls="--")
    ax.text(summ["truth"]["ktau"] + 0.004, ax.get_ylim()[1] * 0.95, "true Kendall's tau", fontsize=8, color=INK2, va="top")
    ax.set_xlabel("Kendall's tau between ability and speed")
    ax.set_ylabel("posterior density")
    ax.set_title("Dependence strength")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(fig_dir, "study1_results.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Study 2
# --------------------------------------------------------------------------

def fig_study2(out_dir, fig_dir):
    summ = load(os.path.join(out_dir, "summary.json"))
    truth = np.load(os.path.join(out_dir, "curves_truth.npz"))
    data = np.load(os.path.join(out_dir, "data.npz"))
    y_test = data["test_Y"]
    fams = [f for f in ("gaussian", "gumbel", "clayton", "joe") if f in summ["models"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) P(Y = 1 | eta): truth vs fitted
    ax = axes[0]
    grid = truth["eta_grid"]
    for f in fams:
        curves = np.load(os.path.join(out_dir, f"posterior_{f}.npz"))["curves"]
        ax.fill_between(grid, np.quantile(curves, 0.05, 0), np.quantile(curves, 0.95, 0),
                        color=COLORS[f], alpha=0.18, lw=0)
        ax.plot(grid, curves.mean(0), color=COLORS[f], lw=2, label=LABELS[f].split(" (")[0] + " fit")
    ax.plot(grid, truth["truth"], color=INK, lw=1.6, ls="--", label="truth (Gumbel)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.05)
    ax.set_xlabel("latent risk factor eta (SD units)")
    ax.set_ylabel("P(outcome | eta)")
    ax.set_title("Outcome probability along the risk factor")
    ax.legend(loc="upper left")
    for q, lab in ((0.95, "95th pct"), (0.99, "99th pct")):
        ax.axvline(sps.norm.ppf(q), color=GRID, lw=1)
        ax.text(sps.norm.ppf(q), 1.3e-4, lab, rotation=90, fontsize=7, color=MUTED, ha="right", va="bottom")

    # (b) calibration of out-of-sample predictions by predicted-risk bins
    ax = axes[1]
    pct = np.array([0, 50, 80, 90, 95, 98, 99, 100])
    for f in fams:
        p = np.load(os.path.join(out_dir, f"posterior_{f}.npz"))["p_test"]
        edges = np.percentile(p, pct)
        edges[-1] = np.inf
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            xs.append(p[m].mean())
            ys.append(y_test[m].mean())
        ax.plot(xs, ys, "o-", color=COLORS[f], lw=1.8, ms=5, label=LABELS[f].split(" (")[0] + " fit")
    lim = (2e-4, 1.0)
    ax.plot(lim, lim, color=AXIS, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("mean predicted P(outcome | indicators), by predicted-risk bin")
    ax.set_ylabel("observed outcome rate in bin")
    ax.set_title("Calibration on 20,000 held-out persons")
    ax.text(0.02, 3.5e-4, "bins: bottom 50%, 50-80, 80-90,\n90-95, 95-98, 98-99, top 1%", fontsize=7.5, color=INK2)
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(fig_dir, "study2_results.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    if os.path.exists("results/study1/summary.json"):
        fig_study1_dgp("results/study1", "figures")
        fig_study1_results("results/study1", "figures")
        print("study 1 figures written")
    if os.path.exists("results/study2/summary.json"):
        fig_study2("results/study2", "figures")
        print("study 2 figures written")
