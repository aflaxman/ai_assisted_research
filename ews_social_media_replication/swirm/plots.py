"""Plot helpers so the notebook stays about the science, not matplotlib.

Each function takes data and an optional axis and returns the axis, so cells
read as one line per figure.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .model import Params
from .evaluate import roc_curve

VAR_COLOR = {"I": "#c1272d", "W": "#0000a7", "M": "#008176"}


def plot_realization(times, traj, p: Params, var_index=2, ax=None, n_show=1):
    """A sample trajectory with the bifurcation time marked."""
    ax = ax or plt.gca()
    for j in range(min(n_show, traj.shape[0])):
        ax.plot(times, traj[j, :, var_index], lw=0.8)
    ax.axvline(p.bifurcation_time(), color="k", ls="--", lw=1,
               label=f"bifurcation (t={p.bifurcation_time():.1f}d)")
    ax.set_xlabel("time (days)")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_R0(p: Params, t_max=40, ax=None):
    """R0(t) rising through the critical value 1."""
    ax = ax or plt.gca()
    t = np.linspace(0, t_max, 400)
    ax.plot(t, p.R0(t), color="#333")
    ax.axhline(1, color="#c1272d", ls="--", lw=1, label="R0 = 1")
    ax.axvline(p.bifurcation_time(), color="k", ls=":", lw=1)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("R0(t)")
    ax.legend(frameon=False, fontsize=8)
    return ax


def raincloud(values_by_group: dict, ax=None, title=""):
    """A minimal raincloud: half-violin + jittered points + boxplot."""
    ax = ax or plt.gca()
    rng = np.random.default_rng(0)
    for i, (name, vals) in enumerate(values_by_group.items()):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        parts = ax.violinplot(vals, positions=[i], showextrema=False, widths=0.7)
        for b in parts["bodies"]:
            b.set_alpha(0.35)
            # keep only the right half of the violin
            b.get_paths()[0].vertices[:, 0] = np.clip(
                b.get_paths()[0].vertices[:, 0], i, np.inf)
        ax.scatter(np.full(vals.size, i) - 0.18 + rng.normal(0, 0.03, vals.size),
                   vals, s=6, alpha=0.4)
        ax.boxplot(vals, positions=[i], widths=0.12, showfliers=False,
                   medianprops=dict(color="k"))
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xticks(range(len(values_by_group)))
    ax.set_xticklabels(list(values_by_group.keys()))
    ax.set_ylabel("Kendall tau")
    ax.set_title(title, fontsize=10)
    return ax


def plot_roc(df, var, ax=None):
    """ROC curves comparing variance and autocorrelation for one variable.

    ``df`` is a per-realization table for a single (scenario, variable), as
    returned by ``run_scenario`` filtered to one variable.
    """
    from .evaluate import _auc
    ax = ax or plt.gca()
    for name, color in [("variance", "#0000a7"), ("autocorr", "#e08b00")]:
        pos_col = "tau_var_cp" if name == "variance" else "tau_ac_cp"
        neg_col = "tau_var_bp" if name == "variance" else "tau_ac_bp"
        pos, neg = df[pos_col].dropna(), df[neg_col].dropna()
        fpr, tpr = roc_curve(pos, neg)
        auc = _auc(pos, neg)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(var, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return ax
