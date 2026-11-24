"""Load the data behind each artifact key.

This is the one place where the model asks for GBD data.  Every request is a
``gbd_mapping`` entity handed to a function with the signature of
``vivarium_inputs.get_measure``.  Off the IHME cluster that function is the
stub in ``vivarium_inputs_stub.py``; on the cluster, change the import below
and the same calls reach the GBD databases:

    from vivarium_inputs import interface

I have not been able to run that path, because ``vivarium_inputs`` and its data
access are not installable here, so read it as a marked seam and not as tested
code.  Two requests go beyond what ``vivarium_inputs`` offers and would need a
few lines of ``get_draws`` on the cluster: DALYs by cause (``interface.get_dalys``),
and incidence per person in the whole population.

Artifact convention: every table carries ``draw_0, draw_1, ...`` columns, and
**draw 0 is the mean**, so that ``input_draw_number: 0`` runs the model at
central values.  Draws 1 and up are samples.
"""
from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from vivarium.gbd_mapping import causes, covariates, risk_factors

from vivarium_potassium_salt.constants import data_keys, data_values, metadata
from vivarium_potassium_salt.data import literature
from vivarium_potassium_salt.data import vivarium_inputs_stub as interface

#: Number of sampled draws written to an artifact, beside the mean in draw 0.
N_SAMPLED_DRAWS = 20

SBP = risk_factors.high_systolic_blood_pressure
SODIUM = risk_factors.diet_high_in_sodium
KIDNEY = risk_factors.kidney_dysfunction
GDP = covariates.gdp_per_capita_base_2010

_DEMOGRAPHY = ["sex", "age_start", "age_end", "year_start", "year_end"]


def get_data(key: str, location: str) -> Any:
    """Return the data for one artifact key and location."""
    if key in _LOADERS:
        return _LOADERS[key](location)
    kind, name, measure = key.split(".")
    if kind == "cause" and measure in data_keys.RATE_MEASURE.values():
        return _burden_rate(name, measure, location)
    raise KeyError(f"no loader for artifact key {key!r}")


def draw_columns(n_sampled: int = N_SAMPLED_DRAWS) -> list[str]:
    return [f"draw_{i}" for i in range(n_sampled + 1)]


###############
# GBD inputs  #
###############


def _gbd(request: Callable[..., pd.DataFrame], *args) -> pd.DataFrame:
    """Combine a request's means (as draw 0) with its first sampled draws."""
    means = request(*args, data_type="means")
    draws = request(*args, data_type="draws")
    sampled = draws[[f"draw_{i}" for i in range(N_SAMPLED_DRAWS)]]
    sampled.columns = draw_columns()[1:]
    table = pd.concat([means.rename(columns={"value": "draw_0"}), sampled], axis=1)
    return table.droplevel("location") if "location" in table.index.names else table


def _burden_rate(cause: str, artifact_measure: str, location: str) -> pd.DataFrame:
    entity = causes[cause]
    if artifact_measure == "daly_rate":
        return _gbd(interface.get_dalys, entity, location)
    return _gbd(interface.get_measure, entity, artifact_measure, location)


def _sbp_exposure(location: str) -> pd.DataFrame:
    return _gbd(interface.get_measure, SBP, "exposure", location).droplevel("parameter")


def _sbp_exposure_sd(location: str) -> pd.DataFrame:
    return _gbd(interface.get_measure, SBP, "exposure_standard_deviation", location)


def _sbp_exposure_weights(location: str) -> pd.DataFrame:
    weights = interface.get_measure(SBP, "exposure_distribution_weights", location)
    return weights.droplevel("location")


def _sbp_relative_risk(location: str) -> pd.DataFrame:
    risks = _gbd(interface.get_measure, SBP, "relative_risk", location)
    return risks.droplevel("parameter")


def _sbp_paf(location: str) -> pd.DataFrame:
    """Population attributable fraction of each cause to blood pressure above the TMREL.

    GBD publishes these, but the model must use fractions that are consistent
    with its own exposure distribution, relative risks, and TMREL, or its
    baseline would not reproduce the burden it is given.  So they are computed
    here: ``1 - 1 / E[RR(X)]`` with ``X`` gamma distributed, and with each
    draw's own TMREL (``components.SampledTmrelRiskEffect`` uses the same one).
    """
    mean = _sbp_exposure(location)
    sd = _sbp_exposure_sd(location).reindex(mean.index)
    risks = _sbp_relative_risk(location)
    quantiles = (np.arange(2000) + 0.5) / 2000
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    excess = {
        column: np.maximum(stats.gamma.ppf(
            quantiles[None, :], shape[column].to_numpy()[:, None],
            scale=scale[column].to_numpy()[:, None],
        ) - data_values.ModelParameters.for_draw(draw).tmrel, 0
        ) / data_values.RELATIVE_RISK_SCALAR_MMHG
        for draw, column in enumerate(mean.columns)
    }
    pieces = []
    for cause, cause_risks in risks.groupby("affected_entity"):
        log_rr = np.log(cause_risks.droplevel("affected_entity").reindex(mean.index))
        paf = pd.DataFrame({
            column: 1 - 1 / np.exp(log_rr[column].to_numpy()[:, None] * excess[column]).mean(axis=1)
            for column in mean.columns
        }, index=mean.index)
        pieces.append(paf.assign(affected_entity=cause).set_index("affected_entity", append=True))
    return pd.concat(pieces).sort_index()


def _sodium_exposure(location: str) -> pd.DataFrame:
    return _gbd(interface.get_measure, SODIUM, "exposure", location).droplevel("parameter")


def _kidney_exposure(location: str) -> pd.DataFrame:
    return _gbd(interface.get_measure, KIDNEY, "exposure", location)


def _gdp_per_capita(location: str) -> float:
    estimate = interface.get_measure(GDP, "estimate", location)
    return float(estimate["mean_value"].iloc[0])


###################
# Other inputs    #
###################


def _scalar_draws(values: list[float]) -> pd.DataFrame:
    """A one-row table of draws, for inputs that do not vary by age or sex."""
    return pd.DataFrame([values], columns=draw_columns(), index=pd.Index(["all"], name="group"))


def _discretionary_share(location: str) -> pd.DataFrame:
    gdp = _gdp_per_capita(location)
    return _scalar_draws([
        float(data_values.discretionary_share(gdp, data_values.ModelParameters.for_draw(draw)))
        for draw in range(N_SAMPLED_DRAWS + 1)
    ])


def _reported_discretionary_share(location: str) -> float:
    """The surveyed share where Bhat et al. report one, else NaN (sensitivity analyses 10, 11)."""
    return float(literature.reported_discretionary_shares().get(location, np.nan))


def hypertension_draws(location: str, column: str, n_sampled: int) -> pd.DataFrame:
    """NCD-RisC ``prevalence``, ``diagnosed``, or ``treated``: mean as draw 0, then samples.

    Sampled from a normal distribution matched to the published interval and
    kept inside [0, 1].  Draw ``k`` is the same whatever ``n_sampled`` is.

    The diagnosed and treated shares are drawn independently, so in a sampled
    draw the treated share can exceed the diagnosed one, which cannot happen
    in life.  The model then counts everyone treated as diagnosed, and that
    raises the number of people reached in scenario 4 by about 1% in the
    sampled draws (not at the mean).  Drawing the two with common noise would
    remove it.
    """
    table = literature.hypertension(location).merge(
        interface.get_age_bins().rename(columns={"age_group_name": "age_group"}), on="age_group"
    )
    table["year_start"], table["year_end"] = interface.YEAR_START, interface.YEAR_END
    table = table.sort_values(["sex", "age_start"]).reset_index(drop=True)
    mean = table[column].to_numpy()
    sd = (table[f"{column}_upper"] - table[f"{column}_lower"]).to_numpy() / (2 * 1.96)
    rng = np.random.default_rng(zlib.crc32(f"hypertension|{column}|{location}".encode()))
    sampled = mean[:, None] + sd[:, None] * rng.standard_normal((n_sampled, len(table))).T
    values = np.clip(np.column_stack([mean, sampled]), 0.0, 1.0)
    draws = pd.DataFrame(values, columns=draw_columns(n_sampled))
    return pd.concat([table[_DEMOGRAPHY], draws], axis=1).set_index(_DEMOGRAPHY)


def _hypertension(column: str) -> Callable[[str], pd.DataFrame]:
    return lambda location: hypertension_draws(location, column, N_SAMPLED_DRAWS)


def _ckd_cvd_hazard_ratio(location: str) -> pd.DataFrame:
    table = literature.ckd_cvd_hazard_ratio_by_age().merge(
        interface.get_age_bins().rename(columns={"age_group_name": "age_group"}), on="age_group"
    )
    parts = []
    for sex in metadata.SEXES:
        part = table.assign(sex=sex, year_start=interface.YEAR_START, year_end=interface.YEAR_END)
        for draw, column in enumerate(draw_columns()):
            z = data_values.ModelParameters.for_draw(draw).ckd_age_hazard_z
            part[column] = np.maximum(part["hazard_ratio"] * np.exp(z * part["log_sd"]), 1.0)
        parts.append(part)
    return pd.concat(parts)[_DEMOGRAPHY + draw_columns()].set_index(_DEMOGRAPHY).sort_index()


_LOADERS: dict[str, Callable[[str], Any]] = {
    data_keys.Population.LOCATION: lambda location: location,
    data_keys.Population.STRUCTURE: lambda location: interface.get_population_structure(
        location).droplevel("location"),
    data_keys.Population.AGE_BINS: lambda location: interface.get_age_bins(),
    data_keys.Population.DEMOGRAPHIC_DIMENSIONS: lambda location: (
        interface.get_demographic_dimensions(location).droplevel("location")),
    data_keys.SystolicBloodPressure.EXPOSURE: _sbp_exposure,
    data_keys.SystolicBloodPressure.EXPOSURE_SD: _sbp_exposure_sd,
    data_keys.SystolicBloodPressure.EXPOSURE_WEIGHTS: _sbp_exposure_weights,
    # GBD describes blood pressure with an ensemble of distributions.  The paper
    # uses a gamma, which is the ensemble with all its weight on one member.
    data_keys.SystolicBloodPressure.DISTRIBUTION: lambda location: "ensemble",
    # gbd_mapping's SBP.tmred is the current 105-115 mmHg; the paper's GBD 2019
    # relative risks go with 110-115.
    data_keys.SystolicBloodPressure.TMRED: lambda location: {
        "distribution": "uniform", "inverted": False,
        "min": data_values.TMREL_RANGE_MMHG[0], "max": data_values.TMREL_RANGE_MMHG[1],
    },
    data_keys.SystolicBloodPressure.RELATIVE_RISK_SCALAR: lambda location: float(
        SBP.relative_risk_scalar),
    data_keys.SystolicBloodPressure.RELATIVE_RISK: _sbp_relative_risk,
    data_keys.SystolicBloodPressure.PAF: _sbp_paf,
    data_keys.Sodium.EXPOSURE: _sodium_exposure,
    data_keys.Sodium.DISCRETIONARY_SHARE: _discretionary_share,
    data_keys.Sodium.REPORTED_DISCRETIONARY_SHARE: _reported_discretionary_share,
    data_keys.Hypertension.PREVALENCE: _hypertension("prevalence"),
    data_keys.Hypertension.DIAGNOSED: _hypertension("diagnosed"),
    data_keys.Hypertension.TREATED: _hypertension("treated"),
    data_keys.KidneyDysfunction.EXPOSURE: _kidney_exposure,
    data_keys.KidneyDysfunction.CVD_HAZARD_RATIO: _ckd_cvd_hazard_ratio,
    data_keys.Covariates.GDP_PER_CAPITA: _gdp_per_capita,
    data_keys.Covariates.HIGH_INCOME: lambda location: bool(literature.is_high_income(location)),
}
