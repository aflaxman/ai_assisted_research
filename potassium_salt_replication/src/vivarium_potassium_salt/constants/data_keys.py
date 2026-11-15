"""Artifact keys: the names under which each input is stored for the simulation."""
from __future__ import annotations

from vivarium_potassium_salt.constants import metadata


class Population:
    LOCATION = "population.location"
    STRUCTURE = "population.structure"
    AGE_BINS = "population.age_bins"
    DEMOGRAPHIC_DIMENSIONS = "population.demographic_dimensions"


class SystolicBloodPressure:
    ENTITY = "risk_factor.high_systolic_blood_pressure"
    EXPOSURE = f"{ENTITY}.exposure"
    EXPOSURE_SD = f"{ENTITY}.exposure_standard_deviation"
    EXPOSURE_WEIGHTS = f"{ENTITY}.exposure_distribution_weights"
    DISTRIBUTION = f"{ENTITY}.distribution"
    TMRED = f"{ENTITY}.tmred"
    RELATIVE_RISK_SCALAR = f"{ENTITY}.relative_risk_scalar"
    RELATIVE_RISK = f"{ENTITY}.relative_risk"
    PAF = f"{ENTITY}.population_attributable_fraction"


class Sodium:
    ENTITY = "risk_factor.diet_high_in_sodium"
    EXPOSURE = f"{ENTITY}.exposure"
    # Not GBD measures: the split of sodium between salt added at home and salt
    # already in food, from GDP per capita.
    DISCRETIONARY_SHARE = f"{ENTITY}.discretionary_share"
    REPORTED_DISCRETIONARY_SHARE = f"{ENTITY}.reported_discretionary_share"


class Hypertension:
    # NCD-RisC estimates, which are not GBD entities.
    PREVALENCE = "ncd_risc.hypertension.prevalence"
    DIAGNOSED = "ncd_risc.hypertension.proportion_diagnosed"
    TREATED = "ncd_risc.hypertension.proportion_treated"


class KidneyDysfunction:
    ENTITY = "risk_factor.kidney_dysfunction"
    EXPOSURE = f"{ENTITY}.exposure"
    # Hazard ratio of cardiovascular death with stage 3-5 disease, by age.
    CVD_HAZARD_RATIO = f"{ENTITY}.cvd_mortality_hazard_ratio"


class Covariates:
    GDP_PER_CAPITA = "covariate.gdp_per_capita_base_2010.estimate"
    HIGH_INCOME = "covariate.gdp_per_capita_base_2010.is_high_income"


# The artifact measure that holds each burden measure, as an annual rate per person.
RATE_MEASURE = {
    "deaths": "cause_specific_mortality_rate",
    "incident_cases": "incidence_rate",
    "dalys": "daly_rate",
}


def burden_rate(cause: str, measure: str) -> str:
    """Artifact key of a cause's annual rate of ``deaths``, ``incident_cases``, or ``dalys``."""
    return f"cause.{cause}.{RATE_MEASURE[measure]}"


BURDEN_RATES = [
    burden_rate(cause, measure)
    for measure in metadata.MEASURES
    for cause in metadata.CAUSES_BY_MEASURE[measure]
]
