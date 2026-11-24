"""Print the README's result tables, in Markdown, from ``results/comparison``.

The tables in the README are pasted from this script's output, so that the
prose cannot drift from the numbers.

Usage
-----
    python -m vivarium_potassium_salt.tools.readme_tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vivarium_potassium_salt.constants import paths

SCENARIO_LABELS = {
    "all_salt": "1. All salt",
    "discretionary": "2. Discretionary salt",
    "non_discretionary": "3. Salt in processed and restaurant food",
    "discretionary_diagnosed_hypertension": "4. Discretionary salt, 80% of diagnosed hypertension",
    "discretionary_treated_hypertension": "5. Discretionary salt, 80% of treated hypertension",
}


def _interval(value, lower, upper, digits=2) -> str:
    if pd.isna(value):
        return ""
    if pd.isna(lower) or pd.isna(upper):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({lower:.{digits}f} to {upper:.{digits}f})"


def _markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.to_numpy()]
    return "\n".join([header, rule] + rows)


def global_table(directory: Path, quantity: str = "Deaths averted, CVD and CKD (net)") -> str:
    table = pd.read_csv(directory / "global.csv")
    table = table[table["quantity"] == quantity]
    digits = 0 if quantity.startswith("People") else 2
    rows = []
    for _, row in table.iterrows():
        rows.append({
            "Scenario": SCENARIO_LABELS[row["scenario"]],
            "Paper": _interval(row["paper"], row["paper_lower"], row["paper_upper"], digits),
            "Simulation, mean inputs": f"{row['simulation_mean_inputs']:.{digits}f}"
            if not pd.isna(row["simulation_mean_inputs"]) else "",
            "Closed form, mean inputs": f"{row['closed_form_mean_inputs']:.{digits}f}",
            "Median (95% UI), one coefficient per draw": _interval(
                row["shared_median"], row["shared_lower"], row["shared_upper"], digits),
            "Median (95% UI), one per stratum": _interval(
                row["independent_median"], row["independent_lower"], row["independent_upper"],
                digits),
            "Ratio to paper": f"{row['ratio_to_paper']:.2f}" if not pd.isna(
                row["ratio_to_paper"]) else "",
        })
    return _markdown(pd.DataFrame(rows))


def baseline_table(directory: Path) -> str:
    table = pd.read_csv(directory / "baseline.csv")
    rows = [{
        "Quantity (millions)": row["quantity"],
        "GBD 2021, paper": f"{row['gbd_2021_paper']:,.2f}",
        "GBD 2023, here": f"{row['gbd_2023_here']:,.2f}",
        "Ratio": f"{row['ratio']:.2f}",
    } for _, row in table.iterrows()]
    return _markdown(pd.DataFrame(rows))


def sensitivity_table(directory: Path) -> str:
    table = pd.read_csv(directory / "sensitivity.csv")
    rows = [{
        "": int(row["analysis"]),
        "Analysis": row["description"],
        "Paper": _interval(row["paper"], row["paper_lower"], row["paper_upper"]),
        "Here": _interval(row["mine"], row["mine_lower"], row["mine_upper"]),
        "Paper, relative to primary": f"{row['paper_ratio_to_primary']:.2f}",
        "Here, relative to primary": f"{row['mine_ratio_to_primary']:.2f}",
    } for _, row in table.iterrows()]
    return _markdown(pd.DataFrame(rows))


def region_table(directory: Path, scenario: str = "all_salt") -> str:
    table = pd.read_csv(directory / "region.csv")
    table = table[table["scenario"] == scenario]
    rows = [{
        "GBD super-region": row["super_region"],
        "CVD deaths, GBD 2023 / GBD 2021": f"{row['baseline_ratio']:.2f}",
        "Averted in the paper, %": f"{row['paper_percent']:.1f}",
        "Averted here, %": f"{row['percent']:.1f}",
        "Ratio": f"{row['percent_ratio']:.2f}",
    } for _, row in table.iterrows()]
    return _markdown(pd.DataFrame(rows))


def width_table(directory: Path) -> str:
    table = pd.read_csv(directory / "interval_widths.csv")
    rows = [{
        "Width of the 95% interval / median": row["level"],
        "Paper": f"{100 * row['paper']:.0f}%",
        "Here, one coefficient per stratum": f"{100 * row['independent']:.0f}%",
        "Here, one coefficient per draw": f"{100 * row['shared']:.0f}%",
    } for _, row in table.iterrows()]
    return _markdown(pd.DataFrame(rows))


def agreement_summary(directory: Path) -> str:
    """Simulation over closed form by country, and correlation with the paper."""
    table = pd.read_csv(directory / "country.csv")
    lines = []
    both = table.groupby(["scenario", "location", "super_region"], as_index=False)[
        [c for c in ("paper_baseline", "paper_averted", "baseline", "net_averted_median",
                     "averted_mean_inputs", "averted_simulation") if c in table]].sum()
    for scenario, group in both.groupby("scenario", sort=False):
        text = [SCENARIO_LABELS[scenario]]
        paper = group["paper_averted"] / group["paper_baseline"]
        mine = group["net_averted_median"] / group["baseline"]
        text.append(f"correlation of the share averted with the paper's, {len(group)} countries: "
                    f"{np.corrcoef(paper, mine)[0, 1]:.3f}")
        if "averted_simulation" in group:
            ratio = (group["averted_simulation"] / group["averted_mean_inputs"]).replace(
                [np.inf, -np.inf], np.nan).dropna()
            total = group["averted_simulation"].sum() / group["averted_mean_inputs"].sum()
            text.append(f"simulation / closed form: total {total:.4f}; by country, median "
                        f"{ratio.median():.4f}, 2.5th to 97.5th percentile "
                        f"{ratio.quantile(0.025):.3f} to {ratio.quantile(0.975):.3f}")
        lines.append("- " + "; ".join(text))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison-dir", type=Path, default=paths.RESULTS_DIR / "comparison")
    directory = parser.parse_args().comparison_dir
    sections = [
        ("Deaths averted per year, millions (cardiovascular and kidney disease, net of "
         "hyperkalaemia)", global_table(directory)),
        ("Deaths averted, CVD (before hyperkalaemia)", global_table(
            directory, "Deaths averted, CVD (before hyperkalaemia)")),
        ("Deaths averted, CKD", global_table(directory, "Deaths averted, CKD")),
        ("People using the substitute, millions", global_table(
            directory, "People using the substitute")),
        ("CVD deaths prevented in people with CKD stages 3-5, millions", global_table(
            directory, "CVD deaths prevented in people with CKD stages 3-5")),
        ("CVD deaths caused by hyperkalaemia, millions", global_table(
            directory, "CVD deaths caused by hyperkalaemia")),
        ("Baseline", baseline_table(directory)),
        ("Sensitivity analyses", sensitivity_table(directory)),
        ("By super-region, all salt", region_table(directory)),
        ("Interval widths", width_table(directory)),
        ("Agreement", agreement_summary(directory)),
    ]
    for title, body in sections:
        print(f"\n### {title}\n\n{body}")


if __name__ == "__main__":
    main()
