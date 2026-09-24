"""Figures for the write-up. Run after microbench.py and run_experiment.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"

# palette: three categorical slots (validated), plus chart chrome
COLORS = {"none": "#2a78d6", "abbreviate": "#eb6834", "expand": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlecolor": INK,
    }
)

LABELS = {"none": "no cleaning", "abbreviate": "abbreviate (Street→St)", "expand": "expand (St→Street)"}


def survival(vals, grid):
    return np.array([(vals >= t).mean() for t in grid])


def fig_mechanism(noise="elevated", regime="split"):
    """True-pair and non-match survival curves of JW similarity."""
    sims = np.load(RESULTS / "microbench_sims.npz")
    grid = np.linspace(0.55, 1.0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for treatment, color in COLORS.items():
        pos = sims[f"{noise}|{regime}|{treatment}|pos"]
        neg = sims[f"{noise}|{regime}|{treatment}|neg"]
        axes[0].plot(grid, survival(pos, grid), color=color, lw=2, label=LABELS[treatment])
        axes[1].semilogy(
            grid, np.maximum(survival(neg, grid), 1e-6), color=color, lw=2
        )
    axes[0].set_title("True pairs: share scoring ≥ threshold", fontsize=11, loc="left")
    axes[0].set_ylim(0.8, 1.005)
    axes[1].set_title("Non-matching pairs: share scoring ≥ threshold", fontsize=11, loc="left")
    axes[1].set_ylim(1e-4, 1)
    for ax in axes:
        ax.set_xlabel("Jaro–Winkler similarity threshold")
        ax.axvline(0.92, color=BASELINE, lw=1, ls="--")
        ax.axvline(0.7, color=BASELINE, lw=1, ls="--")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    sims_none = sims[f"{noise}|{regime}|none|pos"]
    sims_std = sims[f"{noise}|{regime}|abbreviate|pos"]
    axes[0].annotate(
        f"exact-match share: {(sims_std == 1).mean():.0%} standardized,\n"
        f"only {(sims_none == 1).mean():.0%} with no cleaning (off axis)",
        xy=(0.993, 0.857), xytext=(0.56, 0.885), fontsize=8.5, color=MUTED,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
    )
    fig.suptitle(
        f"Street-name similarity under three standardization treatments\n"
        f"(datasets with split conventions, {noise} noise; dashes mark the splink model's fuzzy levels)",
        fontsize=10,
        color=MUTED,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIGS / f"jw_survival_{regime}_{noise}.png", dpi=150)
    plt.close(fig)


MODEL_TITLES = {"full": "full model", "address_heavy": "address-heavy model"}


def fig_outcome():
    """Best-F1 dot plot: treatments x (regime, noise) cells, replicate dots,
    one row per linkage model."""
    frames = []
    for suffix in ["", "_address"]:
        path = RESULTS / f"linkage_results{suffix}.csv"
        if path.exists():
            d = pd.read_csv(path)
            if "model" not in d.columns:
                d.insert(0, "model", "full")
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    models = [m for m in MODEL_TITLES if m in set(df.model)]
    cells = [
        ("consistent", "default"),
        ("consistent", "elevated"),
        ("split", "default"),
        ("split", "elevated"),
    ]
    order = list(COLORS)
    fig, axes = plt.subplots(
        len(models), 4, figsize=(11, 3.2 * len(models)), sharex=True, squeeze=False
    )
    for r, model in enumerate(models):
        for ax, (regime, noise) in zip(axes[r], cells):
            sub = df[(df.model == model) & (df.regime == regime) & (df.noise == noise)]
            piv = sub.pivot_table(index="rep", columns="treatment", values="best_f1")
            for _, row in piv.iterrows():
                ax.plot(
                    range(len(order)), [row[t] for t in order],
                    color=BASELINE, lw=0.9, zorder=1,
                )
            for i, treatment in enumerate(order):
                vals = sub[sub.treatment == treatment]["best_f1"]
                ax.scatter(
                    [i] * len(vals), vals, color=COLORS[treatment], s=42, zorder=3,
                    edgecolors=SURFACE, linewidths=1.2,
                )
                ax.hlines(
                    vals.mean(), i - 0.28, i + 0.28,
                    color=COLORS[treatment], lw=2, zorder=2,
                )
            if r == 0:
                ax.set_title(f"{regime} conventions\n{noise} noise", fontsize=10, loc="left")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(["none", "abbrev.", "expand"], fontsize=9)
            ax.set_xlim(-0.6, len(order) - 0.4)
        axes[r, 0].set_ylabel(MODEL_TITLES[model], fontsize=9)
        # y-limits per model row
        row_vals = df[df.model == model]["best_f1"]
        pad = (row_vals.max() - row_vals.min()) * 0.15 + 1e-4
        for ax in axes[r]:
            ax.set_ylim(row_vals.min() - pad, row_vals.max() + pad)
    fig.suptitle(
        "Splink linkage quality (best F1) by street-name treatment\n"
        "dots: 3 replicates, bar: mean; gray lines connect the same replicate "
        "(top: names + DOB + address; bottom: first name + address only)",
        fontsize=10.5,
        x=0.02,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=[0.01, 0, 1, 0.91])
    fig.savefig(FIGS / "linkage_best_f1.png", dpi=150)
    plt.close(fig)


PHASE2_TITLES = {
    ("jw_no_tf", "elevated"): "JW, no TF adjustment\nelevated noise",
    ("lev_abs", "elevated"): "Levenshtein ≤ 1, 2 edits\nelevated noise",
    ("jw_tf", "severe"): "JW + TF (baseline)\nsevere noise (~39%)",
    ("jw_tf", "garbled"): "JW + TF (baseline)\ngarbled noise",
}


def fig_phase2():
    """Best F1 by treatment across the tire-kicking conditions
    (split conventions, address-heavy model)."""
    df = pd.read_csv(RESULTS / "linkage_results_phase2.csv")
    df = df[df.model == "address_heavy"]
    order = list(COLORS)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharex=True)
    for ax, ((comparator, noise), title) in zip(axes, PHASE2_TITLES.items()):
        sub = df[(df.comparator == comparator) & (df.noise == noise)]
        piv = sub.pivot_table(index="rep", columns="treatment", values="best_f1")
        for _, row in piv.iterrows():
            ax.plot(
                range(len(order)), [row[t] for t in order],
                color=BASELINE, lw=0.9, zorder=1,
            )
        for i, treatment in enumerate(order):
            vals = sub[sub.treatment == treatment]["best_f1"]
            ax.scatter(
                [i] * len(vals), vals, color=COLORS[treatment], s=42, zorder=3,
                edgecolors=SURFACE, linewidths=1.2,
            )
            ax.hlines(
                vals.mean(), i - 0.28, i + 0.28,
                color=COLORS[treatment], lw=2, zorder=2,
            )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["none", "abbrev.", "expand"], fontsize=9)
        ax.set_xlim(-0.6, len(order) - 0.4)
    axes[0].set_ylabel("best F1")
    fig.suptitle(
        "Tire-kicking: best F1 under other comparators and heavier noise\n"
        "(split conventions, address-heavy model; dots: 3 replicates, bar: mean, "
        "gray lines connect the same replicate)",
        fontsize=10.5,
        x=0.02,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIGS / "phase2_best_f1.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_mechanism()
    if (RESULTS / "linkage_results.csv").exists():
        fig_outcome()
    if (RESULTS / "linkage_results_phase2.csv").exists():
        fig_phase2()
    print("figures written to", FIGS)
