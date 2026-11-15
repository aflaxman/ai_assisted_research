"""Run the closed-form model for every country: uncertainty and sensitivity analyses.

The microsimulation answers the question at one draw of the inputs.  This
script answers it at a thousand, for the primary analysis, for the paper's
thirteen sensitivity analyses, and for three variants of my own, by evaluating
``analytic.evaluate`` for each country and adding the countries up draw by draw.

Usage
-----
    python -m vivarium_potassium_salt.tools.run_closed_form --draws 1000 --jobs 4

It writes, under ``results/closed_form/``:

``global_summary.csv``
    A row per analysis, scenario, quantity, and sex: the value at mean inputs
    and the mean, median, and 95% interval over draws.
``region_summary.csv``, ``country_summary.csv.gz``
    The same by GBD super-region and by country, for the primary analysis under
    both ways of drawing the dose-response parameters.
``global_draws.parquet``
    Every draw of every global total, for anyone who wants another summary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from vivarium_potassium_salt import analytic
from vivarium_potassium_salt.constants import metadata, paths
from vivarium_potassium_salt.data import gbd_extract

ALL_SALT, DISCRETIONARY, NON_DISCRETIONARY, DIAGNOSED, TREATED = (
    scenario.name for scenario in metadata.SCENARIOS)
EVERY_SCENARIO = (ALL_SALT, DISCRETIONARY, NON_DISCRETIONARY, DIAGNOSED, TREATED)
POPULATION_WIDE = (ALL_SALT, DISCRETIONARY, NON_DISCRETIONARY)
BOTH_SEXES = "Both"


@dataclass(frozen=True)
class Analysis:
    """One run of the model: a name, what it changes, and the scenarios it is about."""

    name: str
    label: str
    changes: dict = field(default_factory=dict)
    scenarios: tuple[str, ...] = EVERY_SCENARIO
    #: Row or rows of the paper's Extended Data Table 1, if it is one of those.
    paper_analyses: tuple[int, ...] = ()

    @property
    def settings(self) -> analytic.Settings:
        return analytic.Settings(**self.changes)


PRIMARY = Analysis("primary", "Primary analysis")
#: The primary analysis again, with every dose-response parameter drawn afresh
#: for each country, sex, and age band instead of once per draw.
INDEPENDENT = Analysis("independent_parameters", "Parameters drawn separately for every stratum",
                       dict(parameter_draws="independent"))
ANALYSES = (
    PRIMARY,
    INDEPENDENT,
    # The paper's sensitivity analyses, each for the scenarios the paper reports.
    Analysis("sa01_replace_half", "Replace 50% of regular salt",
             dict(replacement_proportion=0.5), (ALL_SALT,), (1,)),
    Analysis("sa02_replace_fifth", "Replace 20% of regular salt",
             dict(replacement_proportion=0.2), (ALL_SALT,), (2,)),
    Analysis("sa03_filippini_2021", "Sodium dose-response from Filippini et al. (2021)",
             dict(sodium_dose_response="filippini_2021"), (ALL_SALT,), (3,)),
    Analysis("sa04_combined_effect_80", "80% of the combined sodium and potassium effect",
             dict(combined_effect_scale=0.8), (ALL_SALT,), (4,)),
    Analysis("sa05_potassium_effect_half", "50% of the potassium effect",
             dict(potassium_effect_scale=0.5), (ALL_SALT,), (5,)),
    Analysis("sa06_potassium_in_hypertension_only",
             "Potassium effect only in people with hypertension",
             dict(potassium_effect_in_normotensives=False), (ALL_SALT,), (6,)),
    Analysis("sa07_kcl_15", "Substitute is 85% NaCl and 15% KCl",
             dict(kcl_fraction=0.15), (ALL_SALT,), (7,)),
    Analysis("sa08_floor_2_5", "No effect below 2.5 g/day of sodium",
             dict(sodium_floor=2.5), (ALL_SALT,), (8,)),
    Analysis("sa09_floor_3", "No effect below 3 g/day of sodium",
             dict(sodium_floor=3.0), (ALL_SALT,), (9,)),
    Analysis("sa10_11_reported_share", "Reported discretionary share in 24 countries",
             dict(discretionary_share_source="reported_where_available"),
             (DISCRETIONARY, NON_DISCRETIONARY), (10, 11)),
    Analysis("sa12_13_lmic_uptake_70", "70% uptake in low- and middle-income countries",
             dict(targeted_uptake_lmic=0.7), (DIAGNOSED, TREATED), (12, 13)),
    # Choices the paper does not spell out, to see how much rides on my reading.
    Analysis("ranked_hypertension", "People with hypertension are those with the highest pressure",
             dict(association_with_blood_pressure="ranked")),
    Analysis("stratum_mean_response",
             "Each stratum's blood pressure distribution moves by the stratum's mean fall",
             dict(blood_pressure_response="stratum_mean")),
    Analysis("stage_deaths_all_adults",
             "Deaths divided among kidney disease stages by the parent model's equation 4",
             dict(stage_deaths="all_adults"), POPULATION_WIDE),
)
ANALYSES_BY_NAME = {analysis.name: analysis for analysis in ANALYSES}
#: Analyses kept by country and by region as well as globally.
DETAILED = (PRIMARY.name, INDEPENDENT.name)


def run_location(location: str, n_sampled: int,
                 analyses: tuple[Analysis, ...] = ANALYSES) -> dict[str, pd.DataFrame]:
    """Every analysis for one location.

    Returns, for each analysis name, draws indexed by scenario, quantity, and
    sex (``analytic.summarise`` without its location level).
    """
    scenarios = {scenario.name: scenario for scenario in metadata.SCENARIOS}
    inputs: dict[str, analytic.LocationInputs] = {}
    results = {}
    for analysis in analyses:
        regime = analysis.settings.parameter_draws
        if regime not in inputs:
            inputs[regime] = analytic.load_inputs(location, n_sampled, regime)
        evaluated = analytic.evaluate(
            inputs[regime], analysis.settings, [scenarios[name] for name in analysis.scenarios])
        summary = analytic.summarise(inputs[regime], evaluated)
        results[analysis.name] = summary.droplevel("location")
    return results


################
# Summaries    #
################


def with_derived_quantities(draws: pd.DataFrame) -> pd.DataFrame:
    """Add both sexes together, and the totals and net figures the paper reports.

    ``draws`` is indexed by scenario, quantity, and sex, possibly after other
    levels.  The additions are made draw by draw, so that an interval for a
    difference is the interval of the differences.
    """
    names = list(draws.index.names)
    outer = [name for name in names if name not in ("quantity", "sex")]
    both = draws.groupby(outer + ["quantity"]).sum()
    both[["sex"]] = BOTH_SEXES
    draws = pd.concat([draws, both.set_index("sex", append=True).reorder_levels(names)])

    wide = draws.stack().unstack("quantity")        # rows: ..., sex, draw; columns: quantity

    def column(name):
        return wide[name].fillna(0.0) if name in wide else 0.0

    scenario = wide.index.get_level_values("scenario")
    is_scenario = scenario != metadata.BASELINE.name
    derived = pd.DataFrame(index=wide.index)
    for measure in metadata.MEASURES:
        if f"{measure}_averted__cvd" not in wide:
            continue
        derived[f"{measure}_averted__total"] = (
            column(f"{measure}_averted__cvd") + column(f"{measure}_averted__ckd"))
    derived["deaths_averted__cvd_net"] = (
        column("deaths_averted__cvd") - column("hyperkalaemia_deaths"))
    derived["deaths_averted__total_net"] = (
        derived["deaths_averted__total"] - column("hyperkalaemia_deaths"))
    derived["net_deaths_averted_in_ckd"] = (
        column("prevented_cvd_deaths_in_ckd") - column("hyperkalaemia_deaths"))
    derived = derived[is_scenario]
    wide = pd.concat([wide, derived], axis=1)
    long = wide.stack().unstack(-2)                 # back to a column per draw
    long.index = long.index.set_names("quantity", level=-1)
    long = long.dropna(how="all")                   # quantities a scenario does not have
    return long.reorder_levels(names).sort_index()[list(draws.columns)]


def summarise_draws(draws: pd.DataFrame) -> pd.DataFrame:
    """Value at mean inputs, and mean, median, and 95% interval over the sampled draws."""
    summary = pd.DataFrame({"mean_inputs": draws["draw_0"]})
    sampled = draws.drop(columns="draw_0")
    if sampled.shape[1]:
        values = sampled.to_numpy()
        summary["mean"] = values.mean(axis=1)
        summary["median"] = np.median(values, axis=1)
        summary["lower"], summary["upper"] = np.percentile(values, [2.5, 97.5], axis=1)
    return summary


def run(locations: list[str], n_sampled: int, jobs: int = 1,
        analyses: tuple[Analysis, ...] = ANALYSES) -> dict[str, pd.DataFrame]:
    """Every analysis for every location, added up.

    Returns ``global_draws`` (indexed by analysis, scenario, quantity, sex),
    and ``global``, ``region``, and ``country`` summaries.
    """
    regions = gbd_extract.locations().set_index("gbd_name")["super_region"]
    worker = partial(run_location, n_sampled=n_sampled, analyses=analyses)
    global_sum: dict[str, pd.DataFrame] = {}
    region_sum: dict[tuple[str, str], pd.DataFrame] = {}
    country_summaries = []

    def add(total: dict, key, table: pd.DataFrame) -> None:
        total[key] = table if key not in total else total[key].add(table, fill_value=0.0)

    def collect(location: str, results: dict[str, pd.DataFrame]) -> None:
        for name, draws in results.items():
            add(global_sum, name, draws)
            if name in DETAILED:
                add(region_sum, (name, regions[location]), draws)
                summary = summarise_draws(with_derived_quantities(draws))
                country_summaries.append(summary.assign(analysis=name, location=location))

    if jobs == 1:
        for location in locations:
            collect(location, worker(location))
    else:
        with ProcessPoolExecutor(jobs) as pool:
            for location, results in zip(locations, pool.map(worker, locations)):
                collect(location, results)

    global_draws = with_derived_quantities(
        pd.concat(global_sum, names=["analysis"]))
    output = {"global_draws": global_draws,
              "global": summarise_draws(global_draws).reset_index()}
    if region_sum:
        region_draws = with_derived_quantities(
            pd.concat(region_sum, names=["analysis", "super_region"]))
        output["region"] = summarise_draws(region_draws).reset_index()
    if country_summaries:
        country = pd.concat(country_summaries).reset_index()
        leading = ["analysis", "location", "scenario", "quantity", "sex"]
        output["country"] = country[leading + [c for c in country.columns if c not in leading]]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("locations", nargs="*", help="GBD location names (default: all 193)")
    parser.add_argument("--draws", type=int, default=1000, help="sampled draws beside the mean")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--analyses", nargs="+", default=None, choices=list(ANALYSES_BY_NAME),
                        help="a subset of the analyses")
    parser.add_argument("--output-dir", type=Path, default=paths.RESULTS_DIR / "closed_form")
    parser.add_argument("--add", action="store_true",
                        help="keep the other analyses already in the output directory")
    arguments = parser.parse_args()

    locations = arguments.locations or gbd_extract.paper_locations()
    analyses = ANALYSES if arguments.analyses is None else tuple(
        ANALYSES_BY_NAME[name] for name in arguments.analyses)
    output = run(locations, arguments.draws, arguments.jobs, analyses)
    output["global_draws"] = output["global_draws"].reset_index()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "global_draws": "global_draws.parquet", "global": "global_summary.csv",
        "region": "region_summary.csv",
        # The country table has 57,000 rows, so it is kept compressed.
        "country": "country_summary.csv.gz",
    }
    rerun = {analysis.name for analysis in analyses}
    for name, file in files.items():
        path = arguments.output_dir / file
        table = output.get(name)
        if arguments.add and path.exists():
            kept = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            kept = kept[~kept["analysis"].isin(rerun)]
            table = kept if table is None else pd.concat([kept, table], ignore_index=True)
        if table is None:
            continue
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False, float_format="%.6g")
        print(f"{path}: {len(table):,} rows")


if __name__ == "__main__":
    main()
