"""Figures comparing the replication with the paper (Figures 1-3 and a parity plot).

    uv run python plot_results.py [--tag _n100k]
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from myriad import parameters as P

RESULTS = pathlib.Path(__file__).parent / "results"
COLORS = {"overdose": "#1f77b4", "hcv": "#9467bd", "sbi": "#ff7f0e", "hiv": "#2ca02c", "other": "#7f7f7f"}
LABELS = {"overdose": "Drug overdose", "hcv": "HCV", "sbi": "Severe injection-related infections",
          "hiv": "HIV", "other": "Other causes"}
SCEN_LABELS = {"base": "Base case", "sustained_decline": "Sustained Decline",
               "precipitous_increase": "Precipitous Increase"}


def base_mean(annual: pd.DataFrame, scenario: str) -> pd.DataFrame:
    a = annual[annual.scenario == scenario].drop(columns=["scenario"])
    return a.groupby("year").mean(numeric_only=True).loc[2025:2034]


def figure_deaths(annual: pd.DataFrame, out: pathlib.Path) -> None:
    """Paper Figure 1: cumulative deaths by cause, 2025-2034."""
    m = base_mean(annual, "base")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in ["overdose", "hcv", "sbi", "hiv", "other"]:
        ax.plot(m.index, m[f"deaths_{c}"].cumsum() / 1e3, color=COLORS[c], lw=2, label=f"{LABELS[c]} (replication)")
        ax.plot(P.PAPER_ANNUAL.index, P.PAPER_ANNUAL[f"deaths_{c}"].cumsum() / 1e3, color=COLORS[c],
                lw=1.2, ls="--", marker="o", ms=3, label=f"{LABELS[c]} (paper)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative deaths (thousands)")
    ax.set_title("Cumulative deaths among US PWID, 2025-2034\nsolid = vivarium replication, dashed = Chiosi et al. Table C2")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_infections(annual: pd.DataFrame, out: pathlib.Path) -> None:
    """Paper Figure 2: annual new injection-related infections, stacked."""
    m = base_mean(annual, "base")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (title, src) in zip(axes, [("Replication", m), ("Paper (Table C1)", P.PAPER_ANNUAL)]):
        bottom = 0
        for c in ["sbi", "hcv", "hiv"]:
            ax.bar(src.index, src[f"new_{c}"] / 1e3, bottom=bottom, color=COLORS[c], label=LABELS[c])
            bottom = bottom + src[f"new_{c}"] / 1e3
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("New infections per year (thousands)")
    axes[1].legend(fontsize=8)
    fig.suptitle("New injection-related infections among US PWID, 2025-2034")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_scenarios(annual: pd.DataFrame, out: pathlib.Path) -> None:
    """Paper Figure 3: cumulative IDU-related deaths under the overdose scenarios."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"base": "-", "sustained_decline": ":", "precipitous_increase": "--"}
    for sc, ls in styles.items():
        if sc not in set(annual.scenario):
            continue
        m = base_mean(annual, sc)
        idu = m[[f"deaths_{c}" for c in ["overdose", "sbi", "hiv", "hcv"]]].sum(axis=1).cumsum()
        ax.plot(m.index, idu / 1e3, color="#d62728", ls=ls, lw=2, label=f"{SCEN_LABELS[sc]} (replication)")
    # paper: base case is the sum of the IDU-related columns in Table C2; the
    # scenario deltas are reported only as 10-year totals
    paper_idu = P.PAPER_ANNUAL[[f"deaths_{c}" for c in ["overdose", "sbi", "hiv", "hcv"]]].sum(axis=1).cumsum()
    ax.plot(paper_idu.index, paper_idu / 1e3, color="k", lw=1, marker="o", ms=3, label="Base case (paper)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative IDU-related deaths (thousands)")
    ax.set_title("Scenario analyses of overdose-death trends, 2025-2034")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure_parity(summary: pd.DataFrame, out: pathlib.Path) -> None:
    """All Table 2 quantities: paper vs replication on log axes."""
    s = summary[summary.scenario == "base"].drop(columns=["scenario", "seed", "n_simulants"]).mean()
    t2 = P.PAPER_TABLE2
    pairs = {
        "New SBI": (t2["new_infections"]["sbi"], s["new_sbi"]),
        "New HIV": (t2["new_infections"]["hiv"], s["new_hiv"]),
        "New HCV": (t2["new_infections"]["hcv"], s["new_hcv"]),
        "Overdose deaths": (t2["deaths"]["overdose"], s["deaths_overdose"]),
        "SBI deaths": (t2["deaths"]["sbi"], s["deaths_sbi"]),
        "HIV deaths": (t2["deaths"]["hiv"], s["deaths_hiv"]),
        "HCV deaths": (t2["deaths"]["hcv"], s["deaths_hcv"]),
        "Other deaths": (t2["deaths"]["other"], s["deaths_other"]),
        "Drug-use cost ($M)": (t2["costs_billion"]["drug_use"] * 1e3, s["cost_drug_use_billion"] * 1e3),
        "SBI cost ($M)": (t2["costs_billion"]["sbi"] * 1e3, s["cost_sbi_billion"] * 1e3),
        "HIV cost ($M)": (t2["costs_billion"]["hiv"] * 1e3, s["cost_hiv_billion"] * 1e3),
        "HCV cost ($M)": (t2["costs_billion"]["hcv"] * 1e3, s["cost_hcv_billion"] * 1e3),
    }
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    xs = [v[0] for v in pairs.values()]
    ys = [v[1] for v in pairs.values()]
    lo, hi = min(xs + ys) / 1.5, max(xs + ys) * 1.5
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1)
    ax.fill_between([lo, hi], [lo * 0.8, hi * 0.8], [lo * 1.25, hi * 1.25], color="grey", alpha=0.15, label="±20-25%")
    ax.scatter(xs, ys, color="#d62728", zorder=3)
    for name, (x, y) in pairs.items():
        ax.annotate(name, (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Paper (10-year total, 2025-2034)")
    ax.set_ylabel("vivarium replication")
    ax.set_title("Paper vs replication, all Table 2 quantities")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    annual = pd.read_csv(RESULTS / f"annual{args.tag}.csv")
    summary = pd.read_csv(RESULTS / f"summary{args.tag}.csv")
    figure_deaths(annual, RESULTS / f"fig1_cumulative_deaths{args.tag}.png")
    figure_infections(annual, RESULTS / f"fig2_new_infections{args.tag}.png")
    figure_parity(summary, RESULTS / f"fig_parity{args.tag}.png")
    if annual.scenario.nunique() > 1:
        figure_scenarios(annual, RESULTS / f"fig3_scenarios{args.tag}.png")
    print("figures written to", RESULTS)


if __name__ == "__main__":
    main()
