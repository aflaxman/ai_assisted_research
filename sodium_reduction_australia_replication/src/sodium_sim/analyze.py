"""Turn raw observer output into HALYs gained and compare with the paper.

Usage::

    python -m sodium_sim.analyze

HALYs follow the paper's definition: person-years lived, weighted by one
minus the prevalent disability (modeled causes from the simulation, plus the
background all-cause disability rate from GBD for everything not modeled),
discounted at 3% per year to 2024 and summed over 2024-2043. Gains are the
difference from business as usual under the same random seed, scaled from
the simulated population to Australia's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from vivarium.artifact import Artifact

from sodium_sim.build_artifact import ARTIFACT_PATH, ROOT, age_bins
from sodium_sim.run import RAW_RESULTS
from sodium_sim.scenarios import PAPER_HALYS, SCENARIOS

DISCOUNT = 0.03
BASE_YEAR = 2024
HORIZON = range(2024, 2044)


def discount_factor(year: pd.Series) -> pd.Series:
    return (1 + DISCOUNT) ** -(year.astype(int) - BASE_YEAR)


def load_run(scenario: str, seed: int, raw: Path = RAW_RESULTS) -> dict[str, pd.DataFrame]:
    d = raw / scenario / f"seed_{seed}"
    out = {p.stem: pd.read_csv(p) for p in d.glob("*.csv")}
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


def background_dw(artifact: Path = ARTIFACT_PATH) -> pd.DataFrame:
    art = Artifact(str(artifact))
    dw = art.load("cause.background.disability_weight").reset_index()
    bins = age_bins()
    dw = dw.merge(bins, on=["age_start", "age_end"])
    # The results stratifier writes age group names with underscores ("10_to_14").
    dw["age_group"] = dw["age_group_name"].str.replace(" ", "_").str.lower()
    return dw[["sex", "age_group", "value"]].rename(columns={"value": "background_dw"})


def halys_by_year(run: dict[str, pd.DataFrame], bg: pd.DataFrame) -> pd.Series:
    """Undiscounted HALYs per event year for one run."""
    pt = run["person_time"].merge(bg, on=["sex", "age_group"], how="left")
    if pt["background_dw"].isna().any():
        raise ValueError("Background disability weights did not match every age group and sex")
    healthy = (pt["value"] * (1 - pt["background_dw"])).groupby(pt["event_year"]).sum()
    ylds = run["ylds"]
    ylds = ylds[ylds["entity"] != "all_causes"]
    modeled = ylds.groupby("event_year")["value"].sum()
    return healthy.sub(modeled, fill_value=0.0)


def summarize(scenario: str, seed: int, bg: pd.DataFrame, raw: Path = RAW_RESULTS) -> dict:
    run = load_run(scenario, seed, raw)
    scale = run["meta"]["scale"]
    halys = halys_by_year(run, bg)
    halys.index = halys.index.astype(int)
    years = halys.index.to_series()
    disc = discount_factor(years)
    in_horizon = years.isin(HORIZON)
    deaths = run["deaths"].groupby("event_year")["value"].sum()
    deaths.index = deaths.index.astype(int)
    pt = run["person_time"].groupby("event_year")["value"].sum()
    pt.index = pt.index.astype(int)
    return {
        "scenario": scenario,
        "seed": seed,
        "halys_discounted": float((halys * disc)[in_horizon].sum() * scale),
        "halys_undiscounted": float(halys[in_horizon].sum() * scale),
        "deaths_2024_2043": float(deaths[deaths.index.isin(HORIZON)].sum() * scale),
        "deaths_2040_2044": float(deaths[deaths.index >= 2040].sum() * scale),
        "person_years_2040_2044": float(pt[pt.index >= 2040].sum() * scale),
    }


def main(tag: str = "") -> pd.DataFrame:
    raw = RAW_RESULTS if not tag else ROOT / "results" / f"raw_{tag}"
    suffix = f"_{tag}" if tag else ""
    bg = background_dw()
    rows = []
    for scenario_dir in sorted(raw.iterdir()):
        for seed_dir in sorted(scenario_dir.glob("seed_*")):
            rows.append(summarize(scenario_dir.name, int(seed_dir.name.split("_")[1]), bg, raw))
    runs = pd.DataFrame(rows)
    bau = runs[runs["scenario"] == "bau"].set_index("seed")
    gains = runs[runs["scenario"] != "bau"].copy()
    gains["acmr_2040_2044"] = gains["deaths_2040_2044"] / gains["person_years_2040_2044"]
    bau["acmr_2040_2044"] = bau["deaths_2040_2044"] / bau["person_years_2040_2044"]
    for col in ["halys_discounted", "halys_undiscounted", "deaths_2024_2043", "acmr_2040_2044"]:
        gains[col] = gains[col].to_numpy() - bau.loc[gains["seed"], col].to_numpy()
    gains["halys_gained"] = gains["halys_discounted"]
    gains["deaths_averted"] = -gains["deaths_2024_2043"]
    # Crude all-cause mortality rate change over 2040-2044 (the paper reports 2044 alone).
    gains["acmr_pct_change"] = 100 * gains["acmr_2040_2044"] / bau.loc[gains["seed"], "acmr_2040_2044"].to_numpy()
    gains.to_csv(ROOT / "results" / f"gains_by_seed{suffix}.csv", index=False)

    table = gains.groupby("scenario").agg(
        halys_gained=("halys_gained", "mean"),
        halys_gained_sd=("halys_gained", "std"),
        halys_gained_undiscounted=("halys_undiscounted", "mean"),
        deaths_averted=("deaths_averted", "mean"),
        acmr_pct_change=("acmr_pct_change", "mean"),
        acmr_pct_change_sd=("acmr_pct_change", "std"),
        n_seeds=("seed", "count"),
    )
    paper = pd.DataFrame.from_dict(PAPER_HALYS, orient="index",
                                   columns=["paper_halys", "paper_lower", "paper_upper"])
    table = table.join(paper)
    table["ratio_to_paper"] = table["halys_gained"] / table["paper_halys"]
    table["label"] = [SCENARIOS[s].label for s in table.index]
    order = [s for s in PAPER_HALYS if s in table.index]
    table = table.loc[order]
    table.to_csv(ROOT / "results" / f"comparison{suffix}.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.0f}".format):
        print(table[["label", "halys_gained", "halys_gained_sd", "paper_halys", "paper_lower",
                     "paper_upper", "deaths_averted", "n_seeds"]].to_string())
    print(table[["ratio_to_paper", "acmr_pct_change", "acmr_pct_change_sd"]].round(3).to_string())
    return table


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "")
