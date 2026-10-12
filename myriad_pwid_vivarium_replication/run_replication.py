"""Run the MYRIAD replication and compare with the published results.

Usage (defaults reproduce the figures in the README):

    uv run python run_replication.py --n 369450 --seeds 0 1 2 \\
        --scenarios base sustained_decline precipitous_increase

Each replicate simulates ``--n`` people standing in for the 3,694,500 PWID of
2018; outputs are scaled by 3,694,500 / n. Results are written to ``results/``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pandas as pd

from myriad import parameters as P
from myriad import simulation as S

RESULTS = pathlib.Path(__file__).parent / "results"


def fmt(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}"


def comparison_table(summary: dict[str, float], sd: dict[str, float] | None) -> str:
    """Markdown table in the layout of the paper's Table 2."""
    t2 = P.PAPER_TABLE2
    rows = [
        ("PWID alive at start of 2025", t2["pwid_2025"], summary["pwid_2025"], 0),
        ("New severe injection-related infections", t2["new_infections"]["sbi"], summary["new_sbi"], 0),
        ("New HIV infections", t2["new_infections"]["hiv"], summary["new_hiv"], 0),
        ("New HCV infections or reinfections", t2["new_infections"]["hcv"], summary["new_hcv"], 0),
        ("Deaths: drug overdose", t2["deaths"]["overdose"], summary["deaths_overdose"], 0),
        ("Deaths: severe injection-related infections", t2["deaths"]["sbi"], summary["deaths_sbi"], 0),
        ("Deaths: HIV", t2["deaths"]["hiv"], summary["deaths_hiv"], 0),
        ("Deaths: HCV", t2["deaths"]["hcv"], summary["deaths_hcv"], 0),
        ("Deaths: other causes", t2["deaths"]["other"], summary["deaths_other"], 0),
        ("Deaths: total", t2["deaths"]["total"], summary["deaths_total"], 0),
        ("Costs, drug use ($bn, disc.)", t2["costs_billion"]["drug_use"], summary["cost_drug_use_billion"], 1),
        ("Costs, severe infections ($bn)", t2["costs_billion"]["sbi"], summary["cost_sbi_billion"], 1),
        ("Costs, HIV ($bn)", t2["costs_billion"]["hiv"], summary["cost_hiv_billion"], 1),
        ("Costs, HCV ($bn)", t2["costs_billion"]["hcv"], summary["cost_hcv_billion"], 1),
        ("Costs, total ($bn)", t2["costs_billion"]["total"], summary["cost_total_billion"], 1),
        ("Average annual cost per person-year ($)", t2["per_person_annual_cost"], summary["per_person_annual_cost"], 0),
        ("Life-years per person (per 2025 PWID)", t2["life_years_per_person"], summary["life_years_per_pwid_2025"], 2),
    ]
    lines = ["| Outcome, 2025-2034 | Paper | This replication | Ratio |", "|---|---:|---:|---:|"]
    for name, paper, mine, d in rows:
        lines.append(f"| {name} | {fmt(paper, d)} | {fmt(mine, d)} | {mine / paper:.2f} |")
    return "\n".join(lines)


def annual_comparison(annual: pd.DataFrame) -> pd.DataFrame:
    """Paper (S-C1..C3) vs replication, by year, for the base case."""
    a = annual.loc[2025:2034]
    paper = P.PAPER_ANNUAL
    out = pd.DataFrame(index=a.index)
    for col in ["new_sbi", "new_hiv", "new_hcv", "deaths_overdose", "deaths_sbi", "deaths_hiv",
                "deaths_hcv", "deaths_other"]:
        out[f"{col}_paper"] = paper[col]
        out[f"{col}_model"] = a[col].round(0)
    for col in ["cost_drug_use", "cost_sbi", "cost_hiv", "cost_hcv"]:
        out[f"{col}_paper_bn"] = (paper[col] / 1e9).round(2)
        out[f"{col}_model_bn"] = (a[f"{col}_disc"] / 1e9).round(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="simulants per replicate")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scenarios", nargs="+", default=["base"], choices=list(P.SCENARIOS))
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--aggregate", nargs="*", default=None, metavar="TAG",
                    help="skip simulation; combine results/annual<TAG>.csv and summary<TAG>.csv "
                         "for the listed tags and write the comparison tables under --tag")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.aggregate is not None:
        annual_all = pd.concat([pd.read_csv(RESULTS / f"annual{t}.csv") for t in args.aggregate])
        summ = pd.concat([pd.read_csv(RESULTS / f"summary{t}.csv") for t in args.aggregate])
        args.scenarios = sorted(set(summ.scenario), key=list(P.SCENARIOS).index)
    else:
        annuals = []
        summaries = []
        t0 = time.time()
        for scenario in args.scenarios:
            for seed in args.seeds:
                print(f"== {scenario}, seed {seed}, n={args.n}", flush=True)
                monthly = S.run(args.n, seed, scenario)
                monthly.to_csv(RESULTS / f"monthly_{scenario}_seed{seed}{args.tag}.csv", index=False)
                annual = S.annualise(monthly)
                annuals.append(annual.reset_index())
                summary = {"scenario": scenario, "seed": seed, "n_simulants": args.n,
                           **S.projection_summary(annual)}
                summaries.append(summary)
                print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in summary.items()},
                                 indent=1), flush=True)
        print(f"total wall time {time.time() - t0:.0f}s")
        annual_all = pd.concat(annuals)
        summ = pd.DataFrame(summaries)
    annual_all.to_csv(RESULTS / f"annual{args.tag}.csv", index=False)
    summ.to_csv(RESULTS / f"summary{args.tag}.csv", index=False)

    # --- base case comparison ------------------------------------------------
    if "base" in args.scenarios:
        base = summ[summ.scenario == "base"]
        mean = base.drop(columns=["scenario", "seed", "n_simulants"]).mean().to_dict()
        sd = base.drop(columns=["scenario", "seed", "n_simulants"]).std().to_dict() if len(base) > 1 else None
        table = comparison_table(mean, sd)
        (RESULTS / f"comparison_table2{args.tag}.md").write_text(table + "\n")
        print(table)
        base_annual = (annual_all[annual_all.scenario == "base"]
                       .drop(columns=["scenario"]).groupby("year").mean(numeric_only=True))
        ac = annual_comparison(base_annual)
        ac.to_csv(RESULTS / f"comparison_annual{args.tag}.csv")
        base_annual.to_csv(RESULTS / f"annual_base_mean{args.tag}.csv")

    # --- scenarios vs base ---------------------------------------------------
    if len(args.scenarios) > 1:
        means = summ.drop(columns=["seed", "n_simulants"]).groupby("scenario").mean()
        lines = ["| Scenario | Δ deaths (paper) | Δ deaths (model) | Δ new infections (paper) | Δ new infections (model) | Δ overdose deaths (model) |",
                 "|---|---:|---:|---:|---:|---:|"]
        for sc, deltas in P.PAPER_SCENARIO_DELTAS.items():
            if sc in means.index:
                dd = means.loc[sc, "deaths_total"] - means.loc["base", "deaths_total"]
                di = means.loc[sc, "new_infections_total"] - means.loc["base", "new_infections_total"]
                do = means.loc[sc, "deaths_overdose"] - means.loc["base", "deaths_overdose"]
                lines.append(f"| {sc} | {deltas['deaths']:+,} | {dd:+,.0f} | {deltas['new_infections']:+,} | {di:+,.0f} | {do:+,.0f} |")
        text = "\n".join(lines)
        (RESULTS / f"comparison_scenarios{args.tag}.md").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
