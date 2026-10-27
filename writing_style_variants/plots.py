"""Matplotlib charts for the notebook, following the dataviz skill's reference palette.

Series colors come from the validated reference palette: slot 1 blue for PR #304
variants, slot 2 orange for Karpathy variants, text-secondary gray for the control.
Marks are thin, gridlines are solid hairlines, and text uses ink tokens, never series color.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyze import PRETTY, group_of
from variants import ORDER as VORDER
from variants import VARIANTS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
GROUP_COLOR = {"PR #304": "#2a78d6", "Karpathy": "#eb6834", "control": "#8a8985"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.titlecolor": INK,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "legend.frameon": False,
})


def _labels_in_order(summary: pd.DataFrame) -> list[str]:
    present = set(summary["variant"].astype(str))
    return [v for v in VORDER if v in present]


def _legend(fig, groups):
    """Legend in the title row, top right of the figure, clear of the panel titles."""
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=GROUP_COLOR[g], label=g)
        for g in ["control", "PR #304", "Karpathy"]
        if g in groups
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.045), ncol=3, fontsize=9, handletextpad=0.3)


def dot_plot(summary: pd.DataFrame, metrics: list[str], title: str, xlim=None):
    """One panel per metric: mean (dot, >= 8px) and 95% bootstrap CI (2px line) per variant."""
    variants = _labels_in_order(summary)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics) + 2.2, 0.42 * len(variants) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(variants))[::-1]
    for ax, m in zip(axes, metrics):
        s = summary[summary["metric"] == m].set_index(summary[summary["metric"] == m]["variant"].astype(str))
        for yi, v in zip(y, variants):
            if v not in s.index:
                continue
            r = s.loc[v]
            c = GROUP_COLOR[group_of(v)]
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(r["mean"], yi, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(f"{r['mean']:.2f}", (r["mean"], yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK2)
        ax.set_title(PRETTY.get(m, m), loc="left")
        ax.grid(axis="x", linewidth=1, color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        if xlim and m in xlim:
            ax.set_xlim(*xlim[m])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([VARIANTS[v]["label"] for v in variants], color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    _legend(fig, set(summary["group"]))
    return fig


def bar_panels(summary: pd.DataFrame, metrics: list[str], title: str):
    """Horizontal thin bars with the value at the tip, one panel per metric."""
    variants = _labels_in_order(summary)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics) + 2.2, 0.42 * len(variants) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(variants))[::-1]
    for ax, m in zip(axes, metrics):
        s = summary[summary["metric"] == m].set_index(summary[summary["metric"] == m]["variant"].astype(str))
        vals = [s.loc[v, "mean"] if v in s.index else np.nan for v in variants]
        colors = [GROUP_COLOR[group_of(v)] for v in variants]
        ax.barh(y, vals, height=0.5, color=colors, zorder=2)
        for yi, v in zip(y, variants):
            if v in s.index:
                r = s.loc[v]
                ax.plot([r["lo"], r["hi"]], [yi, yi], color=INK2, linewidth=1, zorder=3)
                ax.annotate(f"{r['mean']:.1f}", (max(r["hi"], r["mean"]), yi), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8, color=INK2)
        ax.set_title(PRETTY.get(m, m), loc="left")
        ax.grid(axis="x", linewidth=1, color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.margins(x=0.18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([VARIANTS[v]["label"] for v in variants], color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    _legend(fig, {group_of(v) for v in variants})
    return fig


def heatmap(pivot: pd.DataFrame, title: str, fmt: str = "{:.0%}", vmin=0.0, vmax=1.0):
    """Grid of means, one hue light->dark, value printed in each cell in ink chosen by luminance."""
    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 3.2, 0.5 * pivot.shape[0] + 1.2))
    data = pivot.to_numpy(dtype=float)
    ax.imshow(data, cmap=CMAP_BLUE, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=0, color=INK)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=9, color="white" if frac > 0.55 else INK)
    # 2px surface gaps between cells
    for k in range(1, pivot.shape[1]):
        ax.axvline(k - 0.5, color=SURFACE, linewidth=2)
    for k in range(1, pivot.shape[0]):
        ax.axhline(k - 0.5, color=SURFACE, linewidth=2)
    ax.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    return fig
