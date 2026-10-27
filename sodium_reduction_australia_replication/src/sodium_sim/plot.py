"""Plot model HALY gains against Figure 1 of the paper."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sodium_sim.build_artifact import ROOT


def main() -> None:
    table = pd.read_csv(ROOT / "results" / "comparison.csv", index_col=0)
    gains = pd.read_csv(ROOT / "results" / "gains_by_seed.csv")
    y = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + 0.2, table["paper_halys"], height=0.38, color="#9ecae1", label="Dhungel et al. (2025), median and 95% UI")
    ax.errorbar(table["paper_halys"], y + 0.2,
                xerr=[table["paper_halys"] - table["paper_lower"], table["paper_upper"] - table["paper_halys"]],
                fmt="none", ecolor="#3182bd", capsize=3)
    ax.barh(y - 0.2, table["halys_gained"], height=0.38, color="#fdae6b", label="vivarium replication, mean over seeds")
    for scenario, yy in zip(table.index, y):
        pts = gains.loc[gains["scenario"] == scenario, "halys_gained"]
        ax.plot(pts, np.full(len(pts), yy - 0.2), "k.", ms=4)
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"])
    ax.set_xlabel("HALYs gained, 2024-2043, 3% discount")
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "halys_comparison.png", dpi=150)
    print("wrote results/halys_comparison.png")


if __name__ == "__main__":
    main()
