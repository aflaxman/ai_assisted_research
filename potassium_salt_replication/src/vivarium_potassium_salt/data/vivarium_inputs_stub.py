"""A stand-in for ``vivarium_inputs.interface``, backed by a cached GBD 2023 extract.

On the IHME cluster a vivarium model gets its GBD inputs by handing a
``gbd_mapping`` entity to ``vivarium_inputs.get_measure``.  ``vivarium_inputs``
cannot be installed next to the current vivarium suite off the cluster (its
PyPI release predates the ``vivarium.*`` namespace, and its data access needs
the IHME-internal ``vivarium_gbd_access``), so this module offers functions
with the same names, arguments, and return shapes, and answers them from the
extract in ``data/gbd_2023``.  The loader in ``data/loader.py`` is written
against this interface, which makes the swap to the real package a matter of
changing one import.

Where the stub departs from the real interface, it says so:

* It knows one year, 2021, and only the entities and measures this model uses.
* The connector serves a mean and a 95% uncertainty interval, not draws.
  ``data_type="draws"`` returns pseudo-draws sampled independently for every
  stratum from a normal (or lognormal, for relative risks) distribution that
  matches the mean and interval.  Real GBD draws are correlated across strata;
  these are not, which is also how the paper sampled its inputs.
* ``incidence_rate`` is new cases per person-year in the whole population.
  The real interface divides by the susceptible population.
* The standard deviation of systolic blood pressure is rebuilt from the
  regression in the GBD 2023 methods appendix, and its distribution is the
  paper's single gamma rather than the GBD ensemble.
* The relative risks are the GBD 2019 log-linear ones the paper used.  GBD 2021
  and later use non-log-linear risk curves that the connector does not serve.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from vivarium_potassium_salt.constants import data_values, metadata
from vivarium_potassium_salt.data import gbd_extract, literature

#: Number of pseudo-draws returned for ``data_type="draws"``.
NUM_DRAWS = 1000
DRAW_COLUMNS = [f"draw_{i}" for i in range(NUM_DRAWS)]
MEAN_COLUMNS = ["value"]

INDEX_COLUMNS = ["location", "sex", "age_start", "age_end", "year_start", "year_end"]
YEAR_START, YEAR_END = metadata.REFERENCE_YEAR, metadata.REFERENCE_YEAR + 1
_AGE_BOUNDS = pd.DataFrame(
    [(band.name, band.start, 125.0 if band.name == "95 plus" else band.end)
     for band in metadata.AGE_BANDS],
    columns=["age_group", "age_start", "age_end"],
)


class DataNotInExtractError(KeyError):
    """The stub was asked for something the cached extract does not hold."""


####################
# Public interface #
####################


def get_measure(entity, measure: str, location: str, years=None, data_type: str = "draws") -> pd.DataFrame:
    """Return ``measure`` for a ``gbd_mapping`` entity in the vivarium input format.

    Mirrors ``vivarium_inputs.get_measure``.  The frame is indexed by location,
    sex, age and year intervals, and any measure-specific columns, with a
    ``value`` column (``data_type="means"``) or ``draw_*`` columns
    (``data_type="draws"``).
    """
    _check_years(years)
    try:
        producer = _PRODUCERS[(entity.kind, measure)]
    except KeyError:
        raise DataNotInExtractError(
            f"The stub has no {measure!r} for entities of kind {entity.kind!r}. "
            "On the IHME cluster, vivarium_inputs.get_measure answers this."
        ) from None
    return producer(entity, location, _value_columns(data_type))


def get_dalys(entity, location: str, years=None, data_type: str = "draws") -> pd.DataFrame:
    """DALYs per person-year for a cause.

    This is not a ``vivarium_inputs`` measure: a dynamic vivarium model counts
    its own DALYs.  The paper scales GBD's DALYs directly, so the static model
    needs them as an input.  On the cluster they come from the DALYnator draws
    (``get_draws(..., source="dalynator", measure_id=2)``) divided by
    population.
    """
    _check_years(years)
    return _burden_rate(entity, location, _value_columns(data_type), "dalys")


def get_population_structure(location: str, years=None) -> pd.DataFrame:
    """Persons by sex and age band.  Mirrors ``vivarium_inputs.get_population_structure``."""
    _check_years(years)
    table = _one_location(gbd_extract.population(), location)
    return _format(table.rename(columns={"population": "value"}), location, ["value"])


def get_age_bins() -> pd.DataFrame:
    """The age bands this extract covers.  Mirrors ``vivarium_inputs.get_age_bins``."""
    bins = _AGE_BOUNDS.rename(columns={"age_group": "age_group_name"})
    return bins[["age_start", "age_end", "age_group_name"]].copy()


def get_demographic_dimensions(location: str, years=None) -> pd.DataFrame:
    """Every sex and age band for a location, as an empty-columned index frame."""
    _check_years(years)
    rows = [
        (location, sex, row.age_start, row.age_end, YEAR_START, YEAR_END)
        for sex in metadata.SEXES for row in _AGE_BOUNDS.itertuples()
    ]
    return pd.DataFrame(index=pd.MultiIndex.from_tuples(rows, names=INDEX_COLUMNS))


#############
# Producers #
#############


def _cause_specific_mortality_rate(entity, location, columns):
    return _burden_rate(entity, location, columns, "deaths")


def _incidence_rate(entity, location, columns):
    return _burden_rate(entity, location, columns, "incident_cases")


def _burden_rate(entity, location, columns, measure):
    if entity.name not in metadata.CAUSES_BY_MEASURE[measure]:
        raise DataNotInExtractError(f"no {measure} for {entity.name} in the cached extract")
    counts = gbd_extract.burden_counts(measure)
    counts = _one_location(counts[counts["cause"] == entity.name], location)
    people = _one_location(gbd_extract.population(), location)
    table = counts.merge(people, on=["age_group", "sex"], validate="one_to_one")
    for column in ("count", "count_lower", "count_upper"):
        table[column] = table[column] / table["population"]
    table = table.rename(columns={"count": "mean", "count_lower": "lower", "count_upper": "upper"})
    key = f"{entity.name}|{measure}|{location}"
    return _format(_with_values(table, columns, key, lower_bound=0.0), location, columns)


def _exposure(entity, location, columns):
    if entity.name == "kidney_dysfunction":
        return _kidney_dysfunction_exposure(location, columns)
    key = {"diet_high_in_sodium": "sodium", "high_systolic_blood_pressure": "sbp"}.get(entity.name)
    if key is None:
        raise DataNotInExtractError(f"no exposure for {entity.name} in the cached extract")
    table = _one_location(gbd_extract.exposure(key), location)
    table = _with_values(table, columns, f"{entity.name}|exposure|{location}", lower_bound=1e-6)
    table["parameter"] = "continuous"
    return _format(table, location, columns, extra_index=["parameter"])


def _kidney_dysfunction_exposure(location, columns):
    """Prevalence of chronic kidney disease stages 5, 4, and 3 (cat1, cat2, cat3).

    The pull was cut short before the country estimates of stage 3 arrived, so
    as the extract stands every country gets the *global* prevalence of each
    stage at each age and sex, with no uncertainty.  If country estimates of
    stage 3 are added to the extract, they are used, and stages 4 and 5 become
    the country's stage 3 times the global ratio of each stage to stage 3.
    Albuminuria with preserved filtration (cat4) is not used by this model and
    is folded into the unexposed category (cat5).
    """
    world = gbd_extract.global_ckd_stage_prevalence().sort_values(
        ["sex", "age_group"]).reset_index(drop=True)
    if gbd_extract.has_exposure("ckd_stage3"):
        country = _one_location(gbd_extract.exposure("ckd_stage3"), location)
        country = _with_values(country, columns, f"kidney_dysfunction|exposure|{location}",
                               lower_bound=0.0, upper_bound=1.0)
        assert country[["sex", "age_group"]].equals(world[["sex", "age_group"]])
        stage3 = country[list(columns)].to_numpy()
    else:
        stage3 = np.repeat(world[["stage3"]].to_numpy(), len(columns), axis=1)
    prevalence = {
        "cat1": stage3 * world[["stage5_per_stage3"]].to_numpy(),
        "cat2": stage3 * world[["stage4_per_stage3"]].to_numpy(),
        "cat3": stage3,
    }
    exposed = sum(prevalence.values())
    # Keep the three stages a valid partition where the scaled sum exceeds one.
    scale = np.where(exposed > 0.999, 0.999 / np.maximum(exposed, 1e-12), 1.0)
    prevalence = {category: values * scale for category, values in prevalence.items()}
    prevalence["cat5"] = 1.0 - np.minimum(exposed, 0.999)
    table = pd.concat([
        pd.concat([
            world[["age_group", "sex"]].assign(parameter=category),
            pd.DataFrame(values, columns=list(columns)),
        ], axis=1)
        for category, values in prevalence.items()
    ], ignore_index=True)
    return _format(table, location, columns, extra_index=["parameter"])


def _exposure_standard_deviation(entity, location, columns):
    if entity.name != "high_systolic_blood_pressure":
        raise DataNotInExtractError(f"no exposure SD for {entity.name} in the cached extract")
    mean = _exposure(entity, location, columns).droplevel("parameter")
    shape = mean.shape
    sd = data_values.sbp_standard_deviation(
        mean_sbp=mean.to_numpy(),
        age_start=np.broadcast_to(mean.index.get_level_values("age_start").to_numpy()[:, None], shape),
        sex=np.broadcast_to(mean.index.get_level_values("sex").to_numpy()[:, None], shape),
    )
    return pd.DataFrame(sd, index=mean.index, columns=mean.columns)


def _exposure_distribution_weights(entity, location, columns):
    """Ensemble weights with everything on the gamma, the paper's distribution."""
    if entity.name != "high_systolic_blood_pressure":
        raise DataNotInExtractError(f"no ensemble weights for {entity.name}")
    base = get_demographic_dimensions(location).reset_index()
    parts = []
    for name in data_values.ENSEMBLE_DISTRIBUTIONS:
        part = base.copy()
        part["parameter"] = name
        part["value"] = 1.0 if name == "gamma" else 0.0
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    return table.set_index(INDEX_COLUMNS + ["parameter"]).sort_index()


def _relative_risk(entity, location, columns):
    """GBD 2019 relative risks per 10 mmHg, by age band and affected cause."""
    if entity.name != "high_systolic_blood_pressure":
        raise DataNotInExtractError(f"no relative risks for {entity.name}")
    risks = literature.sbp_relative_risks()
    log_mean = np.log(risks["mean"].to_numpy())
    log_sd = (np.log(risks["upper"].to_numpy()) - np.log(risks["lower"].to_numpy())) / (2 * 1.96)
    if columns == MEAN_COLUMNS:
        risks["value"] = risks["mean"]
    else:
        # Relative risks are not location-specific, so their draws must not be:
        # the seed leaves the location out.
        noise = _standard_normals("high_systolic_blood_pressure|relative_risk", len(risks))
        values = np.exp(log_mean[:, None] + log_sd[:, None] * noise)
        risks = pd.concat([risks, pd.DataFrame(values, columns=columns, index=risks.index)], axis=1)
    risks = risks.rename(columns={"cause": "affected_entity"})
    risks["parameter"] = "per unit"
    parts = []
    for sex in metadata.SEXES:
        part = risks.copy()
        part["sex"] = sex
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    return _format(table, location, columns, extra_index=["affected_entity", "parameter"])


def _covariate_estimate(entity, location, columns):
    if entity.name != "gdp_per_capita_base_2010":
        raise DataNotInExtractError(f"no covariate {entity.name} in the cached extract")
    value = literature.gdp_per_capita_2010_usd(location)
    return pd.DataFrame(
        {"mean_value": [value], "upper_value": [value], "lower_value": [value]},
        index=pd.MultiIndex.from_tuples(
            [(location, YEAR_START, YEAR_END)], names=["location", "year_start", "year_end"]
        ),
    )


_PRODUCERS = {
    ("cause", "cause_specific_mortality_rate"): _cause_specific_mortality_rate,
    ("cause", "incidence_rate"): _incidence_rate,
    ("risk_factor", "exposure"): _exposure,
    ("risk_factor", "exposure_standard_deviation"): _exposure_standard_deviation,
    ("risk_factor", "exposure_distribution_weights"): _exposure_distribution_weights,
    ("risk_factor", "relative_risk"): _relative_risk,
    ("covariate", "estimate"): _covariate_estimate,
}


###########
# Helpers #
###########


def _check_years(years) -> None:
    if years not in (None, metadata.REFERENCE_YEAR, [metadata.REFERENCE_YEAR]):
        raise DataNotInExtractError(
            f"The cached extract holds {metadata.REFERENCE_YEAR} only; asked for {years!r}."
        )


def _value_columns(data_type: str) -> list[str]:
    if data_type == "means":
        return MEAN_COLUMNS
    if data_type == "draws":
        return DRAW_COLUMNS
    raise ValueError(f"data_type must be 'means' or 'draws', not {data_type!r}")


def _one_location(table: pd.DataFrame, location: str) -> pd.DataFrame:
    selected = table[table["location_id"] == gbd_extract.location_id(location)]
    if selected.empty:
        raise DataNotInExtractError(f"no rows for {location!r} in the cached extract")
    return selected.drop(columns="location_id").reset_index(drop=True)


def _standard_normals(key: str, n_rows: int) -> np.ndarray:
    """Reproducible standard normal noise, one row per stratum and one column per draw."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return rng.standard_normal((NUM_DRAWS, n_rows)).T


def _with_values(table, columns, key, lower_bound=None, upper_bound=None) -> pd.DataFrame:
    """Add a ``value`` column or pseudo-draw columns built from mean, lower, upper."""
    table = table.sort_values(["sex", "age_group"]).reset_index(drop=True)
    mean = table["mean"].to_numpy(dtype=float)
    if columns == MEAN_COLUMNS:
        values = mean[:, None]
    else:
        sd = (table["upper"].to_numpy(dtype=float) - table["lower"].to_numpy(dtype=float)) / (2 * 1.96)
        values = mean[:, None] + sd[:, None] * _standard_normals(key, len(table))
        if lower_bound is not None or upper_bound is not None:
            values = np.clip(values, lower_bound, upper_bound)
    return pd.concat([table, pd.DataFrame(values, columns=columns, index=table.index)], axis=1)


def _format(table, location, columns, extra_index=()) -> pd.DataFrame:
    """Index a tidy table the way vivarium_inputs does."""
    keys = table[["age_group", "sex"] + list(extra_index)].merge(
        _AGE_BOUNDS, on="age_group", validate="many_to_one"
    )
    keys = keys.assign(location=location, year_start=YEAR_START, year_end=YEAR_END)
    index = pd.MultiIndex.from_frame(keys[INDEX_COLUMNS + list(extra_index)])
    return pd.DataFrame(table[list(columns)].to_numpy(), index=index, columns=list(columns)).sort_index()
