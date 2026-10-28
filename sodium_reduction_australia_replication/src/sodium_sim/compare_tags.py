"""Combine the comparison tables of several run tags into one table and figure.

Usage: python -m sodium_sim.compare_tags <tag>=<label> [<tag>=<label> ...]

Each tag must already have ``results/comparison_<tag>.csv`` from ``analyze``.
Writes ``results/comparison_all.csv`` and ``results/halys_comparison_all.png``.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sodium_sim.build_artifact import ROOT
from sodium_sim.scenarios import PAPER_HALYS, SCENARIOS

COLORS = ["#fdae6b", "#a1d99b", "#bcbddc", "#fc9272", "#9ecae1"]


def main(specs: list[str]) -> pd.DataFrame:
    tags = [s.split("=", 1) for s in specs]
    frames = []
    for tag, label in tags:
        t = pd.read_csv(ROOT / "results" / f"comparison_{tag}.csv", index_col=0)
        t = t[["halys_gained", "halys_gained_sd", "deaths_averted", "acmr_pct_change", "n_seeds", "ratio_to_paper"]]
        t.columns = pd.MultiIndex.from_product([[label], t.columns])
        frames.append(t)
    table = pd.concat(frames, axis=1)
    order = [s for s in PAPER_HALYS if s in table.index]
    table = table.loc[order]
    table.to_csv(ROOT / "results" / "comparison_all.csv")

    n = len(tags)
    height = 0.8 / (n + 1)
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    paper = pd.DataFrame.from_dict(PAPER_HALYS, orient="index", columns=["med", "lo", "hi"]).loc[order]
    ax.barh(y + 0.4 - height / 2, paper["med"], height=height, color="#9ecae1", label="Dhungel et al. (2025), median and 95% UI")
    ax.errorbar(paper["med"], y + 0.4 - height / 2, xerr=[paper["med"] - paper["lo"], paper["hi"] - paper["med"]],
                fmt="none", ecolor="#3182bd", capsize=2)
    for i, (tag, label) in enumerate(tags):
        vals = table[(label, "halys_gained")]
        ax.barh(y + 0.4 - height * (i + 1.5), vals, height=height, color=COLORS[i % len(COLORS)], label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIOS[s].label for s in order])
    ax.set_xscale("log")
    ax.set_xlabel("HALYs gained, 2024-2043, 3% discount")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "halys_comparison_all.png", dpi=150)
    with pd.option_context("display.width", 250, "display.float_format", "{:,.2f}".format):
        print(table.xs("ratio_to_paper", axis=1, level=1).round(2).to_string())
    return table


if __name__ == "__main__":
    main(sys.argv[1:])
