"""Made-up inputs for two made-up countries, in the layout of the real extract.

The model's real inputs are GBD 2023 estimates, which are not kept in the
repository.  The mechanics do not need them: any plausible numbers will do to
run the model end to end, and the tests and the tutorial notebook use these.

``write_inputs(root)`` fills ``root/data`` with every file the model reads, for
"Testland" (a middle-income country with a salt survey) and "Richland" (a
high-income one without).  The files have exactly the layout that
``gbd_pull/tidy.py`` and ``inputs/`` produce from real sources, so anything
that runs on them runs on the real extract.  The published literature tables
are real and are copied as they are, except the table of surveyed
discretionary salt, which names real countries and is replaced by one row.

To use them, point the package at ``root`` *before* importing anything else
from it:

    import os
    os.environ["POTASSIUM_SALT_ROOT"] = "/some/empty/directory"
    from vivarium_potassium_salt.data import synthetic
    synthetic.write_inputs("/some/empty/directory")

None of the numbers here is an estimate of anything.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: The repository's own data directory, wherever ``POTASSIUM_SALT_ROOT`` points.
REPOSITORY_DATA = Path(__file__).resolve().parents[3] / "data"

# (name, location id, ISO3, GDP per capita in 2021 USD, sodium scale, deaths scale)
COUNTRIES = [("Testland", 9001, "TSL", 4_000.0, 1.0, 1.0),
             ("Richland", 9002, "RCH", 60_000.0, 0.8, 0.4)]
REPORTED_SHARE_IN_TESTLAND = 0.76
AGE_BANDS = [f"{a} to {a + 4}" for a in range(25, 95, 5)] + ["95 plus"]
SEXES = ["Female", "Male"]
# connector cause name -> (deaths per 100,000 at ages 60-64, how it is supplied)
CAUSES = {
    "Ischemic heart disease": (160.0, "by_age"),
    "Ischemic stroke": (60.0, "by_age"),
    "Intracerebral hemorrhage": (55.0, "by_age"),
    "Subarachnoid hemorrhage": (6.0, "by_age"),
    "Hypertensive heart disease": (25.0, "by_age"),
    "Rheumatic heart disease": (6.0, "total_with_own_pattern"),
    "Non-rheumatic calcific aortic valve disease": (2.5, "by_age"),
    "Other cardiomyopathy": (5.0, "by_age"),
    "Atrial fibrillation and flutter": (6.0, "by_age"),
    "Aortic aneurysm": (3.0, "total_with_borrowed_pattern"),
    "Lower extremity peripheral arterial disease": (1.2, "by_age"),
    "Endocarditis": (1.5, "by_age"),
    "Chronic kidney disease": (25.0, "by_age"),
}


def _strata() -> pd.DataFrame:
    rows = [(age, i, sex) for i, age in enumerate(AGE_BANDS) for sex in SEXES]
    return pd.DataFrame(rows, columns=["age_group", "age_index", "sex"])


def write_inputs(root: str | Path) -> Path:
    """Write the synthetic inputs under ``root/data`` and return ``root``."""
    root = Path(root)

    def write(table: pd.DataFrame, *parts: str) -> None:
        path = root.joinpath("data", *parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    literature = root / "data" / "literature"
    if not literature.exists():
        shutil.copytree(REPOSITORY_DATA / "literature", literature)
    # The one literature table that names real countries: give Testland a survey.
    write(pd.DataFrame([("Testland", 2020, "Adults", 100 * REPORTED_SHARE_IN_TESTLAND, "yes")],
                       columns=["country", "year_of_data_collection", "population",
                                "discretionary_percent", "in_figure_3_regression"]),
          "literature", "bhat_2020_discretionary_salt_by_country.csv")
    strata = _strata()
    male = (strata["sex"] == "Male").to_numpy()
    age = strata["age_index"].to_numpy()

    crosswalk, regions, gdp, exposure, population, by_age, totals, hypertension = (
        [], [], [], [], [], [], [], [])
    for name, location_id, iso3, income, sodium_scale, death_scale in COUNTRIES:
        ids = dict(location_id=location_id, location_name=name, year=2021)
        crosswalk.append((location_id, name, iso3, name, name, True))
        regions.append((location_id, name, "Testregion", "synthetic"))
        gdp.append((location_id, name, iso3, True, income, "synthetic", 2021))

        people = 2.0e6 * np.exp(-0.25 * age) * np.where(male, 0.95, 1.05)
        population.append(strata[["age_group", "sex"]].assign(**ids, population=people))

        # Sodium falls with age and is lower in women; a few strata sit near the
        # 2 g/day floor on purpose.
        sodium = sodium_scale * (4.6 - 0.16 * age) * np.where(male, 1.0, 0.8)
        pressure = 116 + 2.1 * age - 0.04 * age**2 + np.where(male, 3.0, 0.0) * (age < 6)
        for key, mean, spread in (("sodium", sodium, 0.12), ("sbp", pressure, 0.012)):
            exposure.append(strata[["age_group", "sex"]].assign(
                **ids, exposure=key, mean=mean, lower=mean * (1 - spread),
                upper=mean * (1 + spread)))

        for cause, (rate_at_60, how) in CAUSES.items():
            rate = death_scale * rate_at_60 * np.exp(0.42 * (age - 7)) * np.where(male, 1.3, 1.0)
            count = rate / 1e5 * people
            table = strata[["age_group", "sex"]].assign(
                **ids, cause=cause, measure="deaths", count=count,
                count_lower=count * 0.85, count_upper=count * 1.15, rate=rate)
            if how == "by_age":
                by_age.append(table)
            else:
                total = table.groupby("sex", as_index=False)[
                    ["count", "count_lower", "count_upper"]].sum()
                totals.append(total.assign(**ids, cause=cause, measure="deaths",
                                           age_group="25 plus", rate=np.nan))

        prevalence = np.clip(0.10 + 0.045 * age + np.where(male, 0.03, 0.0), 0, 0.75)
        diagnosed = np.clip(0.35 + 0.03 * age, 0, 0.85) * (0.8 if income < 13_205 else 1.0)
        treated = diagnosed * 0.8
        table = strata.assign(country=name, iso3=iso3, year=2019,
                              age=strata["age_group"].str.replace(" to ", "-"))
        for column, values in (("htn_prev", prevalence), ("diagnosed", diagnosed),
                               ("treated", treated)):
            table[column] = values
            table[f"{column}_lower"] = values * 0.9
            table[f"{column}_upper"] = np.minimum(values * 1.1, 1.0)
        hypertension.append(table[table["age_index"].between(1, 10)].drop(
            columns=["age_group", "age_index"]))

    # Super-region age patterns: rheumatic heart disease has its own, and
    # aortic aneurysm borrows ischemic heart disease's (see gbd_extract).
    region = []
    for cause, slope in (("Rheumatic heart disease", 0.25), ("Ischemic heart disease", 0.42)):
        rate = 10 * np.exp(slope * (age - 7))
        region.append(strata[["age_group", "sex"]].assign(
            location_id=1, location_name="Testregion", year=2021, cause=cause,
            measure="deaths", count=rate, count_lower=rate, count_upper=rate, rate=rate))

    stage3 = np.minimum(0.006 * np.exp(0.33 * age), 0.4)
    kidney = strata[["age_group", "sex"]].assign(
        stage3=stage3, stage4=stage3 * 0.08, stage5=stage3 * 0.04)

    write(pd.DataFrame(crosswalk, columns=["location_id", "gbd_name", "iso3", "paper_name",
                                           "ncdrisc_name", "in_paper"]),
          "locations", "location_crosswalk.csv")
    write(pd.DataFrame(regions, columns=["location_id", "location_name", "super_region",
                                         "source"]), "locations", "gbd_super_regions.csv")
    write(pd.DataFrame(gdp, columns=["location_id", "gbd_name", "iso3", "in_paper",
                                     "gdp_per_capita_current_usd", "source", "source_year"]),
          "worldbank", "gdp_per_capita_2021.csv")
    write(pd.concat(exposure), "gbd_2023", "exposure.csv")
    write(pd.concat(population), "gbd_2023", "population.csv")
    write(pd.concat(by_age), "gbd_2023", "burden_by_age.csv")
    write(pd.concat(totals), "gbd_2023", "burden_25_plus.csv")
    write(pd.concat(region), "gbd_2023", "burden_region_by_age.csv")
    write(kidney, "gbd_2023", "ckd_stage_global_by_age.csv")
    write(pd.concat(hypertension), "ncdrisc", "ncdrisc_hypertension_2019_age_specific.csv")
    return root


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory to fill (its data/ subdirectory)")
    arguments = parser.parse_args()
    print(write_inputs(arguments.root) / "data")


if __name__ == "__main__":
    main()
