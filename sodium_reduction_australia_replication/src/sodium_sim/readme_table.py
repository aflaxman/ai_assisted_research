"""Print the README's headline table from the per-tag comparison CSVs.

Usage: python -m sodium_sim.readme_table <tag>=<column label> [...]
"""

from __future__ import annotations

import sys

import pandas as pd

from sodium_sim.build_artifact import ROOT
from sodium_sim.scenarios import PAPER_HALYS, SCENARIOS


def fmt(x: float) -> str:
    return f"{x:,.0f}" if x >= 1000 else f"{x:,.0f}"


def main(specs: list[str]) -> None:
    tags = [s.split("=", 1) for s in specs]
    tables = {label: pd.read_csv(ROOT / "results" / f"comparison_{tag}.csv", index_col=0) for tag, label in tags}
    head = "| Intervention | Paper, median (95% UI) | " + " | ".join(label for _, label in tags) + " |"
    print(head)
    print("|" + "---|" * (len(tags) + 2))
    for s, (med, lo, hi) in PAPER_HALYS.items():
        cells = []
        for _, label in tags:
            t = tables[label]
            cells.append(f"{fmt(t.loc[s, 'halys_gained'])} ({t.loc[s, 'ratio_to_paper']:.2f})")
        print(f"| {SCENARIOS[s].label} | {fmt(med)} ({fmt(lo)} to {fmt(hi)}) | " + " | ".join(cells) + " |")
    print()
    print("| Intervention | Paper ACMR change 2044, SEIFA 1 / SEIFA 5 | " + " | ".join(label for _, label in tags) + " |")
    print("|" + "---|" * (len(tags) + 2))
    paper_acmr = {"mandate_australia": (-0.083, -0.051), "mandate_uk": (-0.179, -0.111), "mandate_who": (-0.292, -0.179),
                  "mandate_australia_then_uk": (-0.173, -0.111), "kcl_30pct_all_immediate": (-0.661, -0.358),
                  "kcl_10pct_all_10y": (-0.190, -0.119), "kcl_30pct_discretionary_3y": (-0.107, -0.068),
                  "uk_mass_media": (0.0, -0.009), "uk_salt_program": (-0.048, -0.034)}
    for s in PAPER_HALYS:
        p1, p5 = paper_acmr[s]
        cells = [f"{tables[label].loc[s, 'acmr_pct_change']:+.2f}%" for _, label in tags]
        print(f"| {SCENARIOS[s].label} | {p1:+.2f}% / {p5:+.2f}% | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
