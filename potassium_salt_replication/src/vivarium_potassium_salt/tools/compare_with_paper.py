"""Lay this replication's results beside the numbers in the paper.

Reads the closed-form summaries (``tools/run_closed_form.py``), the
microsimulation results if they are there (``tools/run_simulation.py``), and
the published tables transcribed under ``data/paper``, and writes tidy
comparison tables to ``results/comparison/``:

``global.csv``
    Global totals for each scenario: the paper, the simulation and the closed
    form at mean inputs, and the closed form's median and 95% interval under
    both ways of drawing the dose-response parameters.
``sensitivity.csv``
    The paper's thirteen sensitivity analyses, each as deaths averted and as a
    ratio to the primary analysis of the same scenario.
``baseline.csv``
    The burden before any intervention, GBD 2023 here and GBD 2021 in the paper.
``interval_widths.csv``
    How wide the 95% intervals are, for a country and sex and for the world.
``region.csv``
    Share of cardiovascular deaths averted by GBD super-region, mine and the
    paper's.
``country.csv``
    The same by country and sex.
``simulation_sampled_draws.csv``
    Simulation against closed form at sampled input draws, for the countries
    where the simulation was run at draws other than 0.

Usage
-----
    python -m vivarium_potassium_salt.tools.compare_with_paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vivarium_potassium_salt.constants import metadata, paths
from vivarium_potassium_salt.data import gbd_extract
from vivarium_potassium_salt.tools import run_closed_form

SCENARIOS = [scenario.name for scenario in metadata.SCENARIOS]
PRIMARY, INDEPENDENT = run_closed_form.PRIMARY.name, run_closed_form.INDEPENDENT.name
BOTH = run_closed_form.BOTH_SEXES
MILLION = 1e6
#: Cardiovascular deaths averted by lower blood pressure, and the same net of
#: the cardiovascular deaths that hyperkalaemia adds (the paper's basis).
GROSS, NET = "deaths_averted__cvd", "deaths_averted__cvd_net"

#: Rows of the global comparison: (label, this model's quantity, where the paper has it).
#: The paper's deaths are net of hyperkalaemia deaths (footnote to its Table 1),
#: so each is set against this model's net figure.
GLOBAL_ROWS = [
    ("Deaths averted, CVD and CKD (net)", "deaths_averted__total_net", ("table1", "cvd_and_ckd")),
    ("Deaths averted, CVD (net)", "deaths_averted__cvd_net", ("table1", "cvd")),
    ("Deaths averted, CVD (before hyperkalaemia)", "deaths_averted__cvd", None),
    ("Deaths averted, CKD", "deaths_averted__ckd", ("table1", "ckd")),
    ("People using the substitute", "users", ("people", None)),
    ("CVD deaths prevented in people with CKD stages 3-5", "prevented_cvd_deaths_in_ckd",
     ("table2", "prevented_cvd_deaths_in_ckd")),
    ("CVD deaths caused by hyperkalaemia", "hyperkalaemia_deaths",
     ("table2", "hyperkalaemia_deaths")),
]


def read_paper() -> dict[str, pd.DataFrame]:
    """The published tables, as transcribed under ``data/paper``."""
    directory = paths.PAPER_DIR
    return {
        "table1": pd.read_csv(directory / "table1_global_results.csv"),
        "people": pd.read_csv(directory / "table1_people_covered.csv"),
        "table2": pd.read_csv(directory / "table2_ckd_results.csv"),
        "sensitivity": pd.read_csv(directory / "extended_data_table1_sensitivity.csv"),
        "country": pd.read_csv(directory / "supp_table3_country_results.csv"),
        "country_ckd": pd.read_csv(directory / "supp_table4_ckd_results.csv"),
    }


def _paper_value(paper: dict[str, pd.DataFrame], source, scenario: str):
    """(value, lower, upper) in millions from one of the paper's global tables, or NaNs."""
    missing = (np.nan, np.nan, np.nan)
    if source is None:
        return missing
    table_name, key = source
    table = paper[table_name]
    table = table[table["scenario"] == scenario]
    if table_name == "table1":
        table = table[(table["outcome"] == key) & (table["measure"] == "deaths")]
    elif table_name == "table2":
        table = table[table["quantity"] == key]
    if table.empty:
        return missing
    row = table.iloc[0]
    return row["millions"], row["lower"], row["upper"]


def simulation_totals(directory: Path, stem: str = "draw_0") -> pd.DataFrame | None:
    """Microsimulation totals by location, scenario, quantity, and sex, if the run exists."""
    from vivarium_potassium_salt.tools import analysis

    burden, policy = (directory / f"{stem}.{name}.parquet" for name in ("burden", "salt_policy"))
    if not (burden.exists() and policy.exists()):
        return None
    cache = directory / f"{stem}.totals.parquet"
    if cache.exists() and cache.stat().st_mtime > burden.stat().st_mtime:
        return pd.read_parquet(cache)
    _, totals = analysis.summarise_simulation(pd.read_parquet(burden), pd.read_parquet(policy))
    totals = totals[totals["draw"] == 0].drop(columns="draw")
    totals.to_parquet(cache, index=False)
    return totals


def sampled_draw_agreement(directory: Path, stem: str = "sampled_draws") -> pd.DataFrame | None:
    """Simulation against closed form at sampled input draws, where such a run exists.

    A row per location, draw, and scenario, with cardiovascular deaths averted
    both ways.  The run is ``tools/run_simulation.py`` with ``--draws 1 2 3``
    for a handful of large countries.
    """
    from vivarium_potassium_salt import analytic
    from vivarium_potassium_salt.tools import analysis

    burden, policy = (directory / f"{stem}.{name}.parquet" for name in ("burden", "salt_policy"))
    if not (burden.exists() and policy.exists()):
        return None
    _, totals = analysis.summarise_simulation(pd.read_parquet(burden), pd.read_parquet(policy))
    quantity = GROSS
    simulated = totals[totals["quantity"] == quantity].groupby(
        ["location", "draw", "scenario"])["value"].sum().rename("simulation")
    rows = []
    for location, group in simulated.groupby("location"):
        draws = sorted(group.index.get_level_values("draw").unique())
        exact = analytic.run_location(location, n_sampled=max(draws)).xs(
            quantity, level="quantity").groupby("scenario").sum()
        for draw in draws:
            for scenario in SCENARIOS:
                if (location, draw, scenario) in simulated.index:
                    rows.append({
                        "location": location, "draw": draw, "scenario": scenario,
                        "simulation": simulated[(location, draw, scenario)],
                        "closed_form": exact.loc[scenario, f"draw_{draw}"],
                    })
    table = pd.DataFrame(rows)
    table["ratio"] = table["simulation"] / table["closed_form"]
    return table


def _with_net(totals: pd.DataFrame) -> pd.DataFrame:
    """Add the simulation's both-sexes rows and net quantities, like the closed form's."""
    draws = totals.set_index(["location", "scenario", "quantity", "sex"])["value"].rename(
        "draw_0").to_frame()
    return run_closed_form.with_derived_quantities(draws)["draw_0"].rename("value").reset_index()


def global_comparison(summary: pd.DataFrame, paper: dict[str, pd.DataFrame],
                      simulation: pd.DataFrame | None) -> pd.DataFrame:
    summary = summary[summary["sex"] == BOTH].set_index(["analysis", "scenario", "quantity"])
    simulated = None
    if simulation is not None:
        simulated = simulation[simulation["sex"] == BOTH].groupby(
            ["scenario", "quantity"])["value"].sum()
    rows = []
    for scenario in SCENARIOS:
        for label, quantity, source in GLOBAL_ROWS:
            if (PRIMARY, scenario, quantity) not in summary.index:
                continue
            value, lower, upper = _paper_value(paper, source, scenario)
            shared = summary.loc[(PRIMARY, scenario, quantity)]
            row = {
                "scenario": scenario, "quantity": label,
                "paper": value, "paper_lower": lower, "paper_upper": upper,
                "simulation_mean_inputs": (
                    np.nan if simulated is None else simulated.get((scenario, quantity), np.nan)
                    / MILLION),
                "closed_form_mean_inputs": shared["mean_inputs"] / MILLION,
            }
            for name, analysis in (("shared", PRIMARY), ("independent", INDEPENDENT)):
                if (analysis, scenario, quantity) not in summary.index:
                    continue
                result = summary.loc[(analysis, scenario, quantity)]
                for column in ("median", "lower", "upper"):
                    row[f"{name}_{column}"] = result.get(column, np.nan) / MILLION
            row["ratio_to_paper"] = row.get("shared_median", np.nan) / value
            rows.append(row)
    return pd.DataFrame(rows)


def baseline_comparison(summary: pd.DataFrame, paper: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Burden before any intervention: GBD 2023 here, GBD 2021 in the paper."""
    mine = summary[(summary["analysis"] == PRIMARY) & (summary["scenario"] == "baseline")
                   & (summary["sex"] == BOTH)].set_index("quantity")["mean_inputs"]
    country = paper["country"]
    world = country[country["country"] == "Global"].groupby("outcome")["baseline"].sum()
    rows = [
        ("CVD deaths, ages 25+", mine["deaths__cvd"], world["cvd_deaths"]),
        ("CKD deaths, ages 25+", mine["deaths__ckd"], world["ckd_deaths"]),
        ("Adults aged 25+", mine["population"], 4.63e9),
        ("People with CKD stages 3-5", mine["people_with_ckd_stage_3_to_5"], 239e6),
    ]
    table = pd.DataFrame(rows, columns=["quantity", "gbd_2023_here", "gbd_2021_paper"])
    table[["gbd_2023_here", "gbd_2021_paper"]] /= MILLION
    table["ratio"] = table["gbd_2023_here"] / table["gbd_2021_paper"]
    return table


def sensitivity_comparison(summary: pd.DataFrame, paper: dict[str, pd.DataFrame]) -> pd.DataFrame:
    quantity = "deaths_averted__total_net"
    summary = summary[(summary["sex"] == BOTH) & (summary["quantity"] == quantity)].set_index(
        ["analysis", "scenario"])
    primary_paper = paper["table1"]
    primary_paper = primary_paper[(primary_paper["outcome"] == "cvd_and_ckd")
                                  & (primary_paper["measure"] == "deaths")].set_index("scenario")
    rows = []
    for analysis in run_closed_form.ANALYSES:
        for number in analysis.paper_analyses:
            published = paper["sensitivity"].set_index("analysis").loc[number]
            scenario = published["scenario"]
            if (analysis.name, scenario) not in summary.index:
                continue
            mine, base = summary.loc[(analysis.name, scenario)], summary.loc[(PRIMARY, scenario)]
            rows.append({
                "analysis": number, "scenario": scenario, "description": analysis.label,
                "paper": published["deaths"], "paper_lower": published["deaths_lower"],
                "paper_upper": published["deaths_upper"],
                "paper_primary": primary_paper.loc[scenario, "millions"],
                "paper_ratio_to_primary":
                    published["deaths"] / primary_paper.loc[scenario, "millions"],
                "mine": mine["median"] / MILLION, "mine_lower": mine["lower"] / MILLION,
                "mine_upper": mine["upper"] / MILLION,
                "mine_primary": base["median"] / MILLION,
                "mine_ratio_to_primary": mine["median"] / base["median"],
            })
    return pd.DataFrame(rows).sort_values("analysis").reset_index(drop=True)


def _paper_country(paper: dict[str, pd.DataFrame], outcome: str = "cvd_deaths") -> pd.DataFrame:
    """The paper's country results in long form: location, sex, scenario, baseline, averted."""
    names = gbd_extract.locations()
    names = names[names["in_paper"]].set_index("paper_name")["gbd_name"]
    table = paper["country"]
    table = table[(table["outcome"] == outcome) & (table["country"] != "Global")].copy()
    table["location"] = table["country"].map(names)
    assert table["location"].notna().all(), table.loc[table["location"].isna(), "country"].unique()
    parts = []
    for scenario in SCENARIOS:
        parts.append(pd.DataFrame({
            "location": table["location"], "sex": table["sex"], "scenario": scenario,
            "paper_baseline": table["baseline"],
            "paper_averted": table[f"{scenario}__averted"],
            "paper_averted_lower": table[f"{scenario}__averted_lower"],
            "paper_averted_upper": table[f"{scenario}__averted_upper"],
        }))
    return pd.concat(parts, ignore_index=True)


def country_comparison(country: pd.DataFrame, paper: dict[str, pd.DataFrame],
                       simulation: pd.DataFrame | None) -> pd.DataFrame:
    """Cardiovascular deaths averted by country and sex: the paper, closed form, simulation.

    ``net_averted_*`` are net of hyperkalaemia deaths, like the paper's figures,
    and ``percent_median`` is their share of baseline deaths.  ``averted_*`` are
    the deaths averted by lower blood pressure alone, which is the quantity to
    compare between the simulation and the closed form.
    """
    mine = country[country["analysis"] == PRIMARY]
    keys = ["location", "sex"]
    baseline = mine[(mine["scenario"] == "baseline") & (mine["quantity"] == "deaths__cvd")]
    baseline = baseline.set_index(keys)["mean_inputs"].rename("baseline")
    table = _paper_country(paper).set_index(keys + ["scenario"])
    for quantity, prefix in ((GROSS, "averted"), (NET, "net_averted")):
        averted = mine[mine["quantity"] == quantity].set_index(keys + ["scenario"])
        table = table.join(averted[["mean_inputs", "median", "lower", "upper"]].rename(
            columns=lambda name: f"{prefix}_{name}"))
    table = table.join(baseline, on=keys)
    if simulation is not None:
        simulated = simulation[simulation["quantity"] == GROSS].set_index(
            keys + ["scenario"])["value"].rename("averted_simulation")
        table = table.join(simulated)
    table["paper_percent"] = 100 * table["paper_averted"] / table["paper_baseline"]
    table["percent_median"] = 100 * table["net_averted_median"] / table["baseline"]
    regions = gbd_extract.locations().set_index("gbd_name")["super_region"]
    table = table.reset_index()
    table.insert(1, "super_region", table["location"].map(regions))
    return table


def region_comparison(country: pd.DataFrame, region_summary: pd.DataFrame,
                      summary: pd.DataFrame) -> pd.DataFrame:
    """Share of cardiovascular deaths averted by super-region, both sexes, each scenario.

    Mine is the median over draws of the region's total, net of hyperkalaemia
    deaths.  The paper prints no regional totals, so its column is the sum of
    its country medians, which is close to, but not the same as, the median of
    the sum.
    """
    published = country.groupby(["super_region", "scenario"])[
        ["paper_baseline", "paper_averted"]].sum()
    world = country.groupby("scenario")[["paper_baseline", "paper_averted"]].sum()
    world.index = pd.MultiIndex.from_product([["All 193 countries"], world.index],
                                             names=published.index.names)
    published = pd.concat([published, world])

    def mine(table: pd.DataFrame) -> pd.DataFrame:
        """Baseline and averted cardiovascular deaths, indexed by super_region and scenario."""
        table = table[(table["analysis"] == PRIMARY) & (table["sex"] == BOTH)]
        baseline = table[(table["scenario"] == "baseline") & (table["quantity"] == "deaths__cvd")]
        baseline = baseline.set_index("super_region")["mean_inputs"].rename("baseline")
        averted = table[table["quantity"] == NET].set_index(
            ["super_region", "scenario"])[["median"]].rename(
                columns={"median": "net_averted_median"})
        return averted.join(baseline, on="super_region")

    table = published.join(pd.concat([
        mine(region_summary), mine(summary.assign(super_region="All 193 countries"))]))
    table["paper_percent"] = 100 * table["paper_averted"] / table["paper_baseline"]
    table["percent"] = 100 * table["net_averted_median"] / table["baseline"]
    table["percent_ratio"] = table["percent"] / table["paper_percent"]
    table["baseline_ratio"] = table["baseline"] / table["paper_baseline"]
    return table.reset_index()


def interval_widths(country: pd.DataFrame, summary: pd.DataFrame,
                    paper: dict[str, pd.DataFrame], scenario: str = "all_salt") -> pd.DataFrame:
    """Width of the 95% interval relative to the median, at three levels of aggregation.

    The levels are the world, a country and sex (the 40 with the most deaths
    averted in the paper), and, in between, both sexes of those countries.  The
    paper's code is not public, and how fast its intervals narrow as strata are
    added up is the evidence for how it drew its parameters.
    """
    quantity = NET
    mine = country[(country["scenario"] == scenario) & (country["quantity"] == quantity)]
    mine = mine.assign(width=(mine["upper"] - mine["lower"]) / mine["median"])
    mine = mine.pivot_table(index=["location", "sex"], columns="analysis", values="width")
    published = _paper_country(paper)
    published = published[published["scenario"] == scenario].set_index(["location", "sex"])
    published = published.assign(paper=(published["paper_averted_upper"]
                                        - published["paper_averted_lower"])
                                 / published["paper_averted"].where(published["paper_averted"] > 0))
    table = published[["paper_averted", "paper"]].join(mine).reset_index()
    largest = table.nlargest(40, "paper_averted")
    rows = [{
        "level": "A country and sex (median of the 40 largest)",
        "paper": largest["paper"].median(),
        "independent": largest[INDEPENDENT].median(), "shared": largest[PRIMARY].median(),
    }]
    world = summary[(summary["scenario"] == scenario) & (summary["sex"] == BOTH)
                    & (summary["quantity"] == quantity)].set_index("analysis")
    world = (world["upper"] - world["lower"]) / world["median"]
    paper_world = paper["table1"]
    paper_world = paper_world[(paper_world["scenario"] == scenario)
                              & (paper_world["outcome"] == "cvd")
                              & (paper_world["measure"] == "deaths")].iloc[0]
    rows.append({
        "level": "The world",
        "paper": (paper_world["upper"] - paper_world["lower"]) / paper_world["millions"],
        "independent": world[INDEPENDENT], "shared": world[PRIMARY],
    })
    return pd.DataFrame(rows)


def compare(closed_form_dir: Path, simulation_dir: Path) -> dict[str, pd.DataFrame]:
    paper = read_paper()
    summary = pd.read_csv(closed_form_dir / "global_summary.csv")
    country_summary = pd.read_csv(closed_form_dir / "country_summary.csv.gz")
    region_summary = pd.read_csv(closed_form_dir / "region_summary.csv")
    simulation = simulation_totals(simulation_dir)
    if simulation is not None:
        simulation = _with_net(simulation)
    country = country_comparison(country_summary, paper, simulation)
    sampled = sampled_draw_agreement(simulation_dir)
    return ({} if sampled is None else {"simulation_sampled_draws": sampled}) | {
        "global": global_comparison(summary, paper, simulation),
        "baseline": baseline_comparison(summary, paper),
        "sensitivity": sensitivity_comparison(summary, paper),
        "interval_widths": interval_widths(country_summary, summary, paper),
        "region": region_comparison(country, region_summary, summary),
        "country": country,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closed-form-dir", type=Path, default=paths.RESULTS_DIR / "closed_form")
    parser.add_argument("--simulation-dir", type=Path, default=paths.RESULTS_DIR / "simulation")
    parser.add_argument("--output-dir", type=Path, default=paths.RESULTS_DIR / "comparison")
    arguments = parser.parse_args()
    tables = compare(arguments.closed_form_dir, arguments.simulation_dir)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = arguments.output_dir / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.5g")
        print(f"{path}: {len(table):,} rows")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tables["global"].round(3).to_string(index=False))
        print(tables["baseline"].round(3).to_string(index=False))
        print(tables["sensitivity"].round(3).to_string(index=False))
        print(tables["interval_widths"].round(3).to_string(index=False))
        print(tables["region"][tables["region"]["scenario"] == "all_salt"].round(3).to_string(
            index=False))


if __name__ == "__main__":
    main()
