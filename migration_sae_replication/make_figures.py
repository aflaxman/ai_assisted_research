"""Figures mirroring Dupuis (2026) Figures 1-8, 15 and 16 for the numpyro replication.

Colour conventions follow the repository's data-viz guidance: a blue <-> orange
diverging scale with a neutral grey midpoint for method differences (orange =
migration-adjusted method better, blue = comparator better), a single-hue blue
ramp for magnitudes on maps, and method identity carried by both colour and
marker shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap
import numpy as np
import pandas as pd

from geography import load_geography
from synthetic import BASES, BASE_LABELS, PATTERNS, PATTERN_LABELS, LEVELS, base_pattern
from summarize import scenario_metrics, differences

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#f0efec"
BLUE_DARK, ORANGE_DARK = "#104281", "#b8431a"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("blue_orange", [BLUE_DARK, BLUE, NEUTRAL, ORANGE, ORANGE_DARK])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
LEVEL_LABELS = {"low": "Low migration", "medium": "Medium migration", "high": "High migration"}
OUT = Path("results/figures")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
})


def _style_ax(ax):
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.tick_params(length=2, width=0.6)


# ----------------------------------------------------------------------------- heatmaps
def heatmap_grid(scen, method, metrics, title, path, reference="naive", bases=None, patterns=None, note=None):
    """Rows: base pattern; columns: migration pattern; one panel per migration level per metric.
    Cell value = favourability of `method` vs the reference (orange favours `method`).
    A metric may be given as (column, label, sign) or (column, label, sign, category) for
    multinomial output, where `category` selects rows of the long table."""
    bases = bases or BASES
    patterns = patterns or PATTERNS
    diff = differences(scen, reference)
    diff = diff[diff["method"] == method]
    nrow, ncol = len(metrics), len(LEVELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 1.55 * nrow + 0.8), squeeze=False)
    for i, spec in enumerate(metrics):
        metric, label, sign = spec[:3]
        d_i = diff if len(spec) == 3 else diff[diff["category"] == spec[3]]
        vals = []
        grids = {}
        for level in LEVELS:
            g = np.full((len(bases), len(patterns)), np.nan)
            sub = d_i[d_i["level"] == level]
            for _, row in sub.iterrows():
                if row["base"] in bases and row["pattern"] in patterns:
                    g[bases.index(row["base"]), patterns.index(row["pattern"])] = sign * row[f"d_{metric}"]
            grids[level] = g
            vals.append(g)
        vmax = np.nanmax(np.abs(np.concatenate([v.ravel() for v in vals]))) or 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for j, level in enumerate(LEVELS):
            ax = axes[i, j]
            im = ax.imshow(grids[level], cmap=DIVERGING, norm=norm, aspect="auto")
            ax.set_xticks(range(len(patterns)))
            ax.set_xticklabels([PATTERN_LABELS[p] for p in patterns] if i == nrow - 1 else [], rotation=45,
                               ha="right", fontsize=7.5)
            ax.set_yticks(range(len(bases)))
            ax.set_yticklabels([BASE_LABELS[b] for b in bases] if j == 0 else [], fontsize=8)
            if i == 0:
                ax.set_title(LEVEL_LABELS[level], fontsize=9.5, pad=6)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            # 2px surface gaps between cells
            ax.set_xticks(np.arange(-0.5, len(patterns)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(bases)), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.015, pad=0.03)
        cb.set_label(label, fontsize=8, color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, length=2, color=MUTED)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.962, note or "orange: migration-adjusted method better; blue: migration-naive method better",
             fontsize=8, color=INK2)
    fig.subplots_adjust(top=0.88 if nrow <= 4 else 0.9, bottom=0.2 if nrow <= 3 else (0.14 if nrow <= 4 else 0.1),
                        left=0.09, right=0.86, hspace=0.25, wspace=0.08)
    fig.savefig(path, dpi=160)
    plt.close(fig)


METRICS_BINOMIAL = [("bias", "Bias reduction\n(|naive| − |adjusted|)", +1), ("bias2", "Bias² reduction\n(naive − adjusted)", -1),
                    ("rmse", "RMSE reduction\n(naive − adjusted)", -1), ("spearman", "Spearman gain\n(adjusted − naive)", +1)]


def heatmap_grid_abs_bias(scen, method, path, title, reference="naive"):
    """Variant of the heatmap where the bias row compares absolute biases (as in the dissertation,
    whose bias panel shows the difference in average bias)."""
    s = scen.copy()
    s["bias"] = s["bias"].abs()
    heatmap_grid(s, method, [("bias", "|bias| reduction\n(naive − adjusted)", -1)] + METRICS_BINOMIAL[1:], title, path, reference)


# ----------------------------------------------------------------------------- coverage dots
def coverage_dots(scen, methods, labels, path, title, ylim=(0, 102), yticks=(25, 50, 75, 95)):
    scen = scen.copy()
    scen["scenario"] = scen["level"].map({"low": 0, "medium": 1, "high": 2}) * len(PATTERNS) + scen["pattern"].map(PATTERNS.index)
    fig, axes = plt.subplots(len(BASES), 1, figsize=(11, 7.5), sharex=True, sharey=True)
    markers = {methods[0]: "o", methods[1]: "^"}
    colors = {methods[0]: BLUE, methods[1]: ORANGE}
    for ax, base in zip(axes, BASES):
        sub = scen[scen["base"] == base]
        ax.axhline(95, color=MUTED, linewidth=0.8)
        for x in (len(PATTERNS) - 0.5, 2 * len(PATTERNS) - 0.5):
            ax.axvline(x, color=GRID, linewidth=0.6)
        for m in methods:
            d = sub[sub["method"] == m].sort_values("scenario")
            ax.scatter(d["scenario"], 100 * d["coverage"], marker=markers[m], s=22, color=colors[m],
                       edgecolor="white", linewidth=0.6, label=labels[m], zorder=3)
        ax.set_ylabel("Coverage (%)")
        ax.set_title(BASE_LABELS[base], loc="left", fontsize=9, color=INK2, pad=3)
        ax.set_ylim(*ylim)
        ax.set_yticks(list(yticks))
        _style_ax(ax)
    axes[-1].set_xticks(range(3 * len(PATTERNS)))
    axes[-1].set_xticklabels([PATTERN_LABELS[p] for _ in LEVELS for p in PATTERNS], rotation=45, ha="right", fontsize=7.5)
    for k, level in enumerate(LEVELS):
        axes[0].text(k * len(PATTERNS) + len(PATTERNS) / 2 - 0.5, ylim[1] + 0.16 * (ylim[1] - ylim[0]), LEVEL_LABELS[level],
                     ha="center", fontsize=9, color=INK)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.0, -0.02))
    fig.suptitle(title, fontsize=11, x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(top=0.9, bottom=0.14, left=0.07, right=0.98, hspace=0.35)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- forest plot
def bias_forest(raw, method, path, title, reference="naive"):
    """Difference in average bias (method - reference) per scenario with 95% CI over replications."""
    raw = raw.copy()
    raw["err"] = raw["est"] - raw["truth"]
    per = raw.groupby(["base", "pattern", "level", "method", "rep"])["err"].mean().unstack("method")
    d = (per[method] - per[reference]).rename("d").reset_index()
    g = d.groupby(["base", "pattern", "level"])["d"].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 6.5), sharey=True, sharex=True)
    ylabels = [f"{BASE_LABELS[b]} · {PATTERN_LABELS[p]}" for b in BASES for p in PATTERNS]
    for ax, level in zip(axes, LEVELS):
        sub = g[g["level"] == level].set_index(["base", "pattern"])
        y = np.arange(len(ylabels))
        means = np.array([sub.loc[(b, p), "mean"] if (b, p) in sub.index else np.nan for b in BASES for p in PATTERNS])
        cis = np.array([sub.loc[(b, p), "ci"] if (b, p) in sub.index else np.nan for b in BASES for p in PATTERNS])
        ax.axvline(0, color="#e34948", linewidth=0.8)
        ax.errorbar(means, y, xerr=cis, fmt="o", ms=3.5, color=BLUE, ecolor=BLUE, elinewidth=0.9, capsize=0)
        ax.set_title(LEVEL_LABELS[level], fontsize=9.5)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"bias({method}) − bias({reference})", fontsize=8)
        ax.grid(axis="x")
        _style_ax(ax)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.955, "left of the red line: migration-adjusted method less biased; bars are 95% CIs over replications",
             fontsize=8, color=INK2)
    fig.subplots_adjust(top=0.9, bottom=0.08, left=0.2, right=0.98, wspace=0.08)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- maps
def base_pattern_maps(path):
    geo = load_geography()
    all_gdf = load_geography(included_only=False).gdf
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    norm = plt.Normalize(0.1, 0.4)
    for ax, base in zip(axes.ravel(), BASES):
        vals = base_pattern(base, geo)
        all_gdf[~all_gdf["included"]].plot(ax=ax, color="#eeeeea", edgecolor="white", linewidth=0.25)
        geo.gdf.assign(v=vals).plot(column="v", ax=ax, cmap=SEQUENTIAL, norm=norm, edgecolor="white", linewidth=0.25)
        ax.set_title(BASE_LABELS[base] + (" (stand-in for the registry pattern)" if base == "gradient" else ""), fontsize=9)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL)
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    cb.set_label("RR-proportion at t0", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Base distributions of the rifampicin-resistant proportion across 128 Ukrainian districts",
                 fontsize=11, x=0.02, ha="left")
    fig.text(0.02, 0.93, "grey: Crimea and Sevastopol, excluded as in the dissertation", fontsize=8, color=INK2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def rmse_maps(per_district, study, base, level, patterns, methods, labels, path, title):
    geo = load_geography()
    sub = per_district[(per_district["study"] == study) & (per_district["base"] == base) & (per_district["level"] == level)]
    vmax = sub[sub["pattern"].isin(patterns)]["rmse"].quantile(0.99)
    norm = plt.Normalize(0, vmax)
    fig, axes = plt.subplots(len(methods), len(patterns), figsize=(2.9 * len(patterns), 1.9 * len(methods) + 0.9), squeeze=False)
    for j, pat in enumerate(patterns):
        for i, m in enumerate(methods):
            d = sub[(sub["pattern"] == pat) & (sub["method"] == m)].set_index("district")["rmse"]
            vals = d.reindex(range(geo.r)).to_numpy()
            geo.gdf.assign(v=vals).plot(column="v", ax=axes[i, j], cmap=SEQUENTIAL, norm=norm, edgecolor="white",
                                        linewidth=0.2, missing_kwds={"color": "#eeeeea"})
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(PATTERN_LABELS[pat], fontsize=9)
    for i, m in enumerate(methods):
        axes[i, 0].text(-0.05, 0.5, labels[m], transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9, color=INK2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL)
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("district RMSE", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    if "base" in which:
        base_pattern_maps(OUT / "fig1_base_patterns.png")
    if "prediction" in which:
        raw = pd.read_parquet("results/raw/prediction.parquet")
        scen, per_district = scenario_metrics(raw)
        heatmap_grid_abs_bias(scen, "pp", OUT / "fig2_prediction_heatmap.png",
                              "Binomial power prior vs migration-naive prediction (Dupuis Fig. 2 analogue)")
        heatmap_grid_abs_bias(scen, "pp_sample", OUT / "fig2b_prediction_heatmap_pp_sample.png",
                              "Power prior with sample counts in the likelihood vs migration-naive (variant)")
        rmse_maps(per_district, "prediction", "gradient", "medium", ["neighbors", "distance", "crisis_idp", "into_urban"],
                  ["naive", "pp"], {"naive": "Migration-naive", "pp": "Power prior"}, OUT / "fig3_prediction_rmse_maps.png",
                  "District RMSE, SE-gradient base, medium migration (Dupuis Fig. 3 analogue)")
        coverage_dots(scen, ["naive", "pp"], {"naive": "Migration-naive", "pp": "Power prior (migration-adjusted)"},
                      OUT / "fig4_prediction_coverage.png", "95% interval coverage: power prior study (Dupuis Fig. 4 analogue)")
    if "estimation" in which:
        raw = pd.read_parquet("results/raw/estimation.parquet")
        scen, per_district = scenario_metrics(raw)
        heatmap_grid_abs_bias(scen, "amatrix", OUT / "fig5_estimation_heatmap.png",
                              "Binomial A-matrix model vs migration-naive estimation (Dupuis Fig. 5 analogue)")
        bias_forest(raw, "amatrix", OUT / "fig6_estimation_bias_forest.png",
                    "Difference in bias, A-matrix minus migration-naive (Dupuis Fig. 6 analogue)")
        rmse_maps(per_district, "estimation", "gradient", "medium", ["neighbors", "distance", "crisis_idp", "into_urban"],
                  ["naive", "amatrix"], {"naive": "Migration-naive", "amatrix": "A-matrix"}, OUT / "fig7_estimation_rmse_maps.png",
                  "District RMSE, SE-gradient base, medium migration (Dupuis Fig. 7 analogue)")
        coverage_dots(scen, ["naive", "amatrix"], {"naive": "Migration-naive", "amatrix": "A-matrix (migration-adjusted)"},
                      OUT / "fig8_estimation_coverage.png", "95% interval coverage: A-matrix study (Dupuis Fig. 8 analogue)",
                      ylim=(84, 100.5), yticks=(85, 90, 95, 100))
    if "headtohead" in which:
        raw = pd.read_parquet("results/raw/headtohead.parquet")
        scen, per_district = scenario_metrics(raw)
        s = scen.copy()
        s["bias"] = s["bias"].abs()
        heatmap_grid(s, "pp", [("bias", "|bias| reduction\n(A-matrix − power prior)", -1), ("rmse", "RMSE reduction\n(A-matrix − power prior)", -1),
                              ("coverage", "Coverage gain\n(power prior − A-matrix)", +1)],
                     "Power prior vs A-matrix under the method-neutral DGP (Dupuis Fig. 15/16 analogue)",
                     OUT / "fig15_headtohead_heatmap.png", reference="amatrix",
                     note="orange: power prior better; blue: A-matrix better")
        coverage_dots(scen, ["amatrix", "pp"], {"amatrix": "A-matrix", "pp": "Power prior"},
                      OUT / "fig16_headtohead_coverage.png", "95% interval coverage under the method-neutral DGP (Dupuis Fig. 16 analogue)")
    if "multinomial" in which:
        from multinomial import CATS
        sub_b, sub_p = ["block", "hotcold"], ["neighbors", "distance", "crisis_idp", "into_urban"]
        cat_rows = [("rmse", f"RMSE reduction\n{c}", -1, c) for c in CATS]
        for study, method, ref, fname, title, note in [
            ("prediction", "pp", "naive", "fig9_multinomial_prediction_heatmap.png",
             "Multinomial power prior vs migration-naive (Dupuis Fig. 9 analogue)", None),
            ("estimation", "amatrix", "naive", "fig12_multinomial_estimation_heatmap.png",
             "Multinomial A-matrix vs migration-naive (Dupuis Fig. 12 analogue)", None),
            ("headtohead", "pp", "amatrix", "fig15m_multinomial_headtohead_heatmap.png",
             "Multinomial power prior vs A-matrix, method-neutral DGP (Dupuis Fig. 15 analogue)",
             "orange: power prior better; blue: A-matrix better")]:
            p = Path(f"results/raw/multinomial_{study}.parquet")
            if not p.exists():
                continue
            raw = pd.read_parquet(p)
            scen, _ = scenario_metrics(raw)
            metrics = [("bias", "TVD reduction", -1, "TVD")] + cat_rows
            heatmap_grid(scen, method, metrics, title, OUT / fname, reference=ref, bases=sub_b, patterns=sub_p, note=note)
            if study == "headtohead":
                cov = scen[scen["category"] != "TVD"].groupby(["method", "category"])["coverage"].mean().unstack("category")
                cov.to_csv(OUT.parent / "multinomial_headtohead_coverage_by_category.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "prediction", "estimation", "headtohead", "multinomial"])
