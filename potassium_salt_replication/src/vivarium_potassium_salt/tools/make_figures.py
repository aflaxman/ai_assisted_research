"""Draw the figures in the README from the comparison tables.

Usage
-----
    python -m vivarium_potassium_salt.tools.compare_with_paper
    python -m vivarium_potassium_salt.tools.make_figures

Each figure has a table behind it in ``results/comparison/``, named in the
function that draws it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vivarium_potassium_salt.constants import paths  # noqa: E402

# Colours: three categorical slots that stay apart under colour-vision
# deficiency, a grey for context, and quiet chart furniture.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CONTEXT = "#a8a69f"
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

SCENARIO_LABELS = {
    "all_salt": "1. All salt",
    "discretionary": "2. Discretionary salt",
    "non_discretionary": "3. Salt in processed\nand restaurant food",
    "discretionary_diagnosed_hypertension": "4. Discretionary salt,\ndiagnosed hypertension",
    "discretionary_treated_hypertension": "5. Discretionary salt,\ntreated hypertension",
}
EMPHASISED_REGION = "North Africa and Middle East"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def _title(ax, title: str, subtitle: str) -> None:
    lines = subtitle.count("\n") + 1
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK,
                 pad=10 + 13.5 * lines)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=SECONDARY, linespacing=1.3)


def _compact(value: float, _position=None) -> str:
    """1, 10, 100, 1,000, 10K, 100K, 1M: axis labels for counts on a log scale."""
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e4:
        return f"{value / 1e3:g}K"
    return f"{value:,.0f}" if value >= 1 else f"{value:g}"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(path)


def uncertainty_intervals(comparison_dir: Path, output: Path) -> None:
    """How wide the interval is, by how the parameters are drawn (``global.csv``)."""
    table = pd.read_csv(comparison_dir / "global.csv")
    table = table[table["quantity"] == "Deaths averted, CVD and CKD (net)"].set_index("scenario")
    series = [
        ("paper", "paper", "Huang et al.", BLUE),
        ("independent", "independent_median",
         "This replication, a fresh coefficient for every\ncountry, sex, and age band", AQUA),
        ("shared", "shared_median",
         "This replication, one coefficient per draw", ORANGE),
    ]
    scenarios = [name for name in SCENARIO_LABELS if name in table.index]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    offsets = {"paper": 0.24, "independent": 0.0, "shared": -0.24}
    for row, scenario in enumerate(scenarios):
        y0 = len(scenarios) - 1 - row
        reference = table.loc[scenario, "paper"]
        for key, centre, label, colour in series:
            y = y0 + offsets[key]
            low, high = (table.loc[scenario, f"{key}_{bound}"] / reference
                         for bound in ("lower", "upper"))
            ax.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle="round",
                    label=label if row == 0 else None, zorder=3)
            ax.plot(table.loc[scenario, centre] / reference, y, "o", color=colour, markersize=7.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.axvline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([
        f"{SCENARIO_LABELS[name]}\n{table.loc[name, 'paper']:.2f} million"
        for name in reversed(scenarios)], fontsize=9)
    ax.set_ylim(-0.6, len(scenarios) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Deaths averted per year, as a multiple of the paper's estimate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=1, fontsize=9,
              handlelength=1.6, labelcolor=SECONDARY)
    _title(ax, "The estimates replicate, and the width of their intervals rests on one choice",
           "Median and 95% uncertainty interval of deaths averted per year from cardiovascular "
           "and kidney disease,\nnet of hyperkalaemia, in five scenarios.  Under each "
           "scenario's name is the paper's estimate.")
    _save(fig, output)


def country_scatter(comparison_dir: Path, output: Path, scenario: str = "all_salt") -> None:
    """Share of cardiovascular deaths averted by country, mine against the paper's (``country.csv``)."""
    table = pd.read_csv(comparison_dir / "country.csv")
    table = table[table["scenario"] == scenario]
    both = table.groupby(["location", "super_region"], as_index=False)[
        ["paper_baseline", "paper_averted", "baseline", "net_averted_median"]].sum()
    both["paper_percent"] = 100 * both["paper_averted"] / both["paper_baseline"]
    both["percent"] = 100 * both["net_averted_median"] / both["baseline"]
    emphasised = both["super_region"] == EMPHASISED_REGION

    fig, ax = plt.subplots(figsize=(6.8, 6.6))
    top = float(np.ceil(max(both["paper_percent"].max(), both["percent"].max()) / 5) * 5)
    ax.plot([0, top], [0, top], color=AXIS, linewidth=0.8, zorder=1)
    for mask, colour, label in ((~emphasised, CONTEXT, "Other countries"),
                                (emphasised, ORANGE, EMPHASISED_REGION)):
        ax.scatter(both.loc[mask, "paper_percent"], both.loc[mask, "percent"], s=44, color=colour,
                   edgecolor=SURFACE, linewidth=1.2, label=label, zorder=3 if colour == ORANGE else 2)
    # A few names, set in the empty space above their points, in the points'
    # left-to-right order so that the leader lines do not cross.
    names = {"Türkiye": "Türkiye", "Iran (Islamic Republic of)": "Iran", "Egypt": "Egypt"}
    labelled = both[both["location"].isin(names)].sort_values("paper_percent")
    for step, (_, row) in enumerate(labelled.iterrows()):
        ax.annotate(names[row["location"]], (row["paper_percent"], row["percent"]),
                    xytext=(0.05 * top + 0.085 * top * step, 0.47 * top + 0.035 * top * step),
                    textcoords="data", ha="center", va="bottom", fontsize=8.5, color=SECONDARY,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7,
                                    shrinkA=1, shrinkB=4))
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Cardiovascular deaths averted in the paper (GBD 2021 inputs), %")
    ax.set_ylabel("In this replication (GBD 2023 inputs), %")
    ax.legend(loc="lower right", fontsize=9, handletextpad=0.3, labelcolor=SECONDARY)
    correlation = np.corrcoef(both["paper_percent"], both["percent"])[0, 1]
    rest = np.corrcoef(both.loc[~emphasised, "paper_percent"], both.loc[~emphasised, "percent"])[0, 1]
    _title(ax, "Country by country, with one region that moved",
           f"Share of cardiovascular deaths averted by replacing all salt, {len(both)} "
           f"countries.\nThe correlation is {correlation:.2f}, and {rest:.2f} without the "
           "highlighted region, where GBD 2023\nputs older adults well above the 2 g/day "
           "sodium floor.")
    _save(fig, output)


def simulation_agreement(comparison_dir: Path, output: Path) -> None:
    """Microsimulation over closed form, by country (``country.csv``)."""
    table = pd.read_csv(comparison_dir / "country.csv")
    if "averted_simulation" not in table:
        print("no simulation results; skipping", output)
        return
    series = [("all_salt", "All salt, everyone", BLUE),
              ("discretionary_diagnosed_hypertension",
               "Discretionary salt, 80% of people with diagnosed hypertension", ORANGE)]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for scenario, label, colour in series:
        both = table[table["scenario"] == scenario].groupby("location")[
            ["averted_simulation", "averted_mean_inputs"]].sum()
        both = both[both["averted_mean_inputs"] > 0]
        ax.scatter(both["averted_mean_inputs"], both["averted_simulation"] / both["averted_mean_inputs"],
                   s=36, color=colour, edgecolor=SURFACE, linewidth=1.0, label=label, zorder=3)
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_compact))
    ax.minorticks_off()
    ax.set_xlabel("Cardiovascular deaths averted per year in the country, closed form")
    ax.set_ylabel("Simulation / closed form")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9, handletextpad=0.3,
              labelcolor=SECONDARY)
    _title(ax, "The microsimulation matches the closed form, to within its sampling error",
           "One simulation of 15,000 simulants per country and scenario, at the mean of every "
           "input.  In the targeted\nscenario who takes up the salt is random, and the scatter "
           "is that sampling.")
    _save(fig, output)


SENSITIVITY_LABELS = {
    10: "Reported discretionary share, discretionary salt",
    11: "Reported discretionary share, other salt",
    12: "70% uptake in poorer countries, diagnosed hypertension",
    13: "70% uptake in poorer countries, treated hypertension",
}


def sensitivity(comparison_dir: Path, output: Path) -> None:
    """Each sensitivity analysis as a ratio to its primary analysis (``sensitivity.csv``)."""
    table = pd.read_csv(comparison_dir / "sensitivity.csv").sort_values("analysis")
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    positions = np.arange(len(table))[::-1].astype(float)
    # The two estimates often coincide, so each has its own half of the row.
    ax.plot(table["paper_ratio_to_primary"], positions + 0.17, "o", color=BLUE, markersize=7.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Huang et al.", zorder=3)
    ax.plot(table["mine_ratio_to_primary"], positions - 0.17, "o", color=ORANGE, markersize=7.5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="This replication", zorder=3)
    ax.axvline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([
        f"{int(row['analysis'])}. {SENSITIVITY_LABELS.get(int(row['analysis']), row['description'])}"
        for _, row in table.iterrows()], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.set_xlim(0, max(1.3, float(table[["paper_ratio_to_primary", "mine_ratio_to_primary"]]
                                  .max().max()) + 0.08))
    ax.set_xlabel("Deaths averted, as a multiple of the primary analysis of the same scenario")
    ax.legend(loc="lower left", fontsize=9, handletextpad=0.3, labelcolor=SECONDARY)
    _title(ax, "The thirteen sensitivity analyses move the answer by nearly the same amounts",
           "Each analysis changes one assumption.  Analyses 1 to 9 are for replacing all salt, "
           "and the rest name their scenario.")
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison-dir", type=Path, default=paths.RESULTS_DIR / "comparison")
    parser.add_argument("--output-dir", type=Path, default=paths.PROJECT_ROOT / "figures")
    arguments = parser.parse_args()
    _style()
    directory, out = arguments.comparison_dir, arguments.output_dir
    uncertainty_intervals(directory, out / "uncertainty_intervals.png")
    country_scatter(directory, out / "country_scatter.png")
    simulation_agreement(directory, out / "simulation_vs_closed_form.png")
    sensitivity(directory, out / "sensitivity_analyses.png")


if __name__ == "__main__":
    main()
