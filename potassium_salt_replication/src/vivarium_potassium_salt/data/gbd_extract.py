"""Read the cached GBD 2023 extract and fill in the strata that were not pulled.

The IHME GBD connector caps a user at 50,000 rows a day (and at 10,000 rows of
full-precision CSV), which is not enough for deaths, new cases, and DALYs by
country, age, sex, and cause for thirteen causes.  The pull plan
(``gbd_pull/plan.py``) therefore spends its rows where they matter:

* deaths for the three largest causes come by country, age, and sex;
* everything else comes as a country total for ages 25 and over, and is spread
  over ages with the age pattern of the country's GBD super-region.

This module hides that difference.  ``burden_counts`` returns one table with
the same shape for every cause and measure, and says in a ``source`` column
which rows are GBD estimates and which are my disaggregation of one.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from vivarium_potassium_salt.constants import metadata, paths

STRATUM = ["location_id", "age_group", "sex"]
# The pull stopped before the super-region age patterns of these four causes
# (about 3% of the deaths in the model) arrived.  Each borrows the age pattern
# of a cause that kills at similar ages; its country totals are its own.
BORROWED_AGE_PATTERN = {
    "atrial_fibrillation_and_flutter": "ischemic_stroke",
    "lower_extremity_peripheral_arterial_disease": "ischemic_stroke",
    "aortic_aneurysm": "ischemic_heart_disease",
    "endocarditis": "ischemic_heart_disease",
}
# How this model names measures -> how the connector names them.
CONNECTOR_MEASURE = {"deaths": "deaths", "incident_cases": "incidence", "dalys": "dalys"}
MODEL_CAUSE = {gbd: model for model, gbd in metadata.GBD_CAUSE_NAMES.items()}


@lru_cache
def locations() -> pd.DataFrame:
    """GBD locations with ISO3 codes, the paper's names, and super-regions."""
    crosswalk = pd.read_csv(paths.LOCATION_DIR / "location_crosswalk.csv")
    super_regions = paths.LOCATION_DIR / "gbd_super_regions.csv"
    if super_regions.exists():
        crosswalk = crosswalk.merge(
            pd.read_csv(super_regions)[["location_id", "super_region"]],
            on="location_id", how="left", validate="one_to_one",
        )
    else:  # fall back to the global age pattern
        crosswalk["super_region"] = "Global"
    crosswalk["super_region"] = crosswalk["super_region"].fillna("Global")
    return crosswalk


def location_id(location: str) -> int:
    table = locations()
    match = table.loc[table["gbd_name"] == location, "location_id"]
    if match.empty:
        raise KeyError(f"{location!r} is not a GBD location name in the cached extract")
    return int(match.iloc[0])


def paper_locations() -> list[str]:
    """GBD names of the 193 countries and territories the paper models."""
    table = locations()
    return sorted(table.loc[table["in_paper"], "gbd_name"])


@lru_cache
def exposure(key: str) -> pd.DataFrame:
    """Mean and 95% interval of one exposure by location, age band, and sex.

    ``key`` is ``sodium`` (g/day of urinary sodium), ``sbp`` (mmHg), or
    ``ckd_stage3`` (prevalence of stage 3 chronic kidney disease).
    """
    table = pd.read_csv(paths.GBD_EXTRACT_DIR / "exposure.csv")
    table = table[table["exposure"] == key]
    if table.empty:
        raise KeyError(f"no exposure named {key!r} in the cached extract")
    return table[STRATUM + ["mean", "lower", "upper"]].reset_index(drop=True)


def has_exposure(key: str) -> bool:
    try:
        exposure(key)
    except KeyError:
        return False
    return True


@lru_cache
def population() -> pd.DataFrame:
    """Persons by location, age band, and sex in the reference year."""
    return pd.read_csv(paths.GBD_EXTRACT_DIR / "population.csv")[STRATUM + ["population"]]


@lru_cache
def global_ckd_stage_prevalence() -> pd.DataFrame:
    """Global prevalence of CKD stages 3, 4, and 5 by age band and sex."""
    table = pd.read_csv(paths.GBD_EXTRACT_DIR / "ckd_stage_global_by_age.csv")
    table["stage4_per_stage3"] = table["stage4"] / table["stage3"]
    table["stage5_per_stage3"] = table["stage5"] / table["stage3"]
    return table


def _read_burden(name: str) -> pd.DataFrame:
    table = pd.read_csv(paths.GBD_EXTRACT_DIR / name)
    table["cause"] = table["cause"].map(MODEL_CAUSE)
    return table


@lru_cache
def _region_rates() -> pd.DataFrame:
    """Age-specific rates per person for Global and the seven super-regions."""
    table = _read_burden("burden_region_by_age.csv")
    table["rate"] = table["rate"] / 1e5
    return table.rename(columns={"location_name": "super_region"})[
        ["cause", "measure", "super_region", "age_group", "sex", "rate"]
    ]


def _spread_over_ages(totals: pd.DataFrame, shape: pd.DataFrame) -> pd.DataFrame:
    """Split country totals for ages 25+ over age bands in proportion to ``shape``.

    ``totals`` has one row per location and sex with ``count``, ``count_lower``,
    ``count_upper``.  ``shape`` has one row per location, age band, and sex with
    a non-negative ``weight``.  The interval is scaled with the mean, so within
    a country and sex the age bands move together.
    """
    shape = shape.copy()
    denominator = shape.groupby(["location_id", "sex"])["weight"].transform("sum")
    shape["share"] = np.where(denominator > 0, shape["weight"] / denominator.replace(0, np.nan), 0.0)
    merged = shape.merge(totals, on=["location_id", "sex"], how="left", validate="many_to_one")
    for column in ("count", "count_lower", "count_upper"):
        merged[column] = merged[column].fillna(0.0) * merged["share"]
    return merged[STRATUM + ["count", "count_lower", "count_upper"]]


def _region_shape(cause: str, connector_measure: str) -> pd.DataFrame:
    """The super-region's age-specific rate for every country stratum."""
    rates = _region_rates()
    rates = rates[rates["measure"] == connector_measure]
    own = rates[rates["cause"] == cause]
    if own.empty and cause in BORROWED_AGE_PATTERN:
        own = rates[rates["cause"] == BORROWED_AGE_PATTERN[cause]]
    if own.empty:
        raise KeyError(f"no {connector_measure} age pattern for {cause} in the cached extract")
    people = population().merge(
        locations()[["location_id", "super_region"]], on="location_id", how="left"
    )
    return people.merge(
        own[["super_region", "age_group", "sex", "rate"]],
        on=["super_region", "age_group", "sex"], how="left",
    )


def available_measures() -> list[str]:
    """The measures the cached extract can supply for every cause that needs them."""
    measures = []
    for measure in metadata.MEASURES:
        try:
            burden_counts(measure)
        except KeyError:
            continue
        measures.append(measure)
    return measures


def _deaths(cause: str) -> pd.DataFrame:
    """Deaths by country, age band, and sex for one cause.

    Strata pulled directly are used as they are.  Where a cause has a country
    total for ages 25+ instead (or as well, for hypertensive heart disease,
    whose younger bands were pulled before the CSV cap), the total not yet
    accounted for is spread over the missing age bands in proportion to the
    super-region's age-specific death rate times the country's population.
    """
    by_age = _read_burden("burden_by_age.csv")
    direct = by_age[(by_age["cause"] == cause) & (by_age["measure"] == "deaths")]
    direct = direct[STRATUM + ["count", "count_lower", "count_upper"]].assign(source="gbd")
    if direct["age_group"].nunique() == len(metadata.AGE_BAND_NAMES):
        return direct

    totals = _read_burden("burden_25_plus.csv")
    total = totals[(totals["cause"] == cause) & (totals["measure"] == "deaths")]
    if total.empty:
        raise KeyError(f"no deaths for {cause} in the cached extract")
    total = total[["location_id", "sex", "count", "count_lower", "count_upper"]].copy()
    if not direct.empty:
        covered = direct.groupby(["location_id", "sex"], as_index=False)["count"].sum()
        total = total.merge(covered, on=["location_id", "sex"], how="left", suffixes=("", "_covered"))
        remaining = (1 - total["count_covered"].fillna(0) / total["count"].where(total["count"] > 0))
        remaining = remaining.clip(lower=0).fillna(0)
        for column in ("count", "count_lower", "count_upper"):
            total[column] = total[column] * remaining
        total = total.drop(columns="count_covered")
    shape = _region_shape(cause, "deaths")
    shape = shape[~shape["age_group"].isin(set(direct["age_group"]))]
    shape = shape.assign(weight=(shape["rate"] * shape["population"]).fillna(0.0))
    spread = _spread_over_ages(total, shape[STRATUM + ["weight"]]).assign(source="disaggregated")
    return pd.concat([direct, spread], ignore_index=True)


@lru_cache
def burden_counts(measure: str) -> pd.DataFrame:
    """Annual counts of one measure by location, age band, sex, and cause.

    Parameters
    ----------
    measure
        ``deaths``, ``incident_cases``, or ``dalys``.

    Returns
    -------
        Columns ``location_id, age_group, sex, cause, count, count_lower,
        count_upper, source``.  ``source`` is ``gbd`` for strata pulled as they
        are, and ``disaggregated`` for strata split from a country total.
    """
    connector_measure = CONNECTOR_MEASURE[measure]
    totals = _read_burden("burden_25_plus.csv")

    pieces = []
    for cause in metadata.CAUSES_BY_MEASURE[measure]:
        deaths = _deaths(cause)
        if measure == "deaths":
            piece = deaths
        else:
            total = totals[(totals["cause"] == cause) & (totals["measure"] == connector_measure)]
            if total.empty:
                raise KeyError(f"no {measure} for {cause} in the cached extract")
            total = total[["location_id", "sex", "count", "count_lower", "count_upper"]]
            # Age pattern: this country's deaths, times the super-region's ratio of
            # the measure to deaths at each age and sex.  Where the super-region
            # records no deaths, fall back on its rate of the measure itself.
            shape = _region_shape(cause, connector_measure).merge(
                _region_shape(cause, "deaths")[STRATUM + ["rate"]].rename(
                    columns={"rate": "death_rate"}
                ),
                on=STRATUM,
            ).merge(deaths[STRATUM + ["count"]].rename(columns={"count": "deaths"}),
                    on=STRATUM, how="left")
            ratio = shape["rate"] / shape["death_rate"].where(shape["death_rate"] > 0)
            by_deaths = shape["deaths"] * ratio
            by_rate = shape["rate"] * shape["population"]
            usable = by_deaths.groupby([shape["location_id"], shape["sex"]]).transform(
                lambda weights: weights.notna().all() and weights.sum() > 0
            )
            shape["weight"] = np.where(usable, by_deaths, by_rate)
            shape["weight"] = shape["weight"].fillna(0.0)
            piece = _spread_over_ages(total, shape[STRATUM + ["weight"]]).assign(
                source="disaggregated"
            )
        pieces.append(piece.assign(cause=cause))

    table = pd.concat(pieces, ignore_index=True)
    table = table[table["age_group"].isin(metadata.AGE_BAND_NAMES)]
    return table[STRATUM + ["cause", "count", "count_lower", "count_upper", "source"]]
