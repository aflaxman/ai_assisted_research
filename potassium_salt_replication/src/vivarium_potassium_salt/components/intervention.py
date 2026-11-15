"""Replacing regular salt with potassium-enriched salt."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.population import SimulantData

from vivarium_potassium_salt.components import hypertension, sodium
from vivarium_potassium_salt.constants import data_keys, data_values, metadata

USES_SUBSTITUTE = "salt_substitute.uses_substitute"
SODIUM_REDUCTION = "salt_substitute.sodium_reduction"
POTASSIUM_INCREASE = "salt_substitute.potassium_increase"
SBP_REDUCTION = "salt_substitute.sbp_reduction"
SBP_EXPOSURE = "high_systolic_blood_pressure.exposure"


class PotassiumEnrichedSalt(Component):
    """Swap sodium chloride for a blend with potassium chloride, and lower blood pressure.

    The chain, for one simulant who uses the substitute:

    1. Sodium falls by the share of their intake that comes from the replaced
       sources, times the share of that salt replaced, times the share of the
       substitute that is potassium chloride (a quarter).  Intake is not pushed
       below the floor of 2 g/day, and nothing happens to people already below
       it.
    2. Potassium rises by 1.33 g for each gram of sodium removed, because the
       two chlorides replace each other weight for weight.
    3. Systolic blood pressure falls by the sum of a sodium effect, which is
       larger with age and with hypertension, and a potassium effect, which is
       larger where sodium intake is high and has a ceiling.

    The component then subtracts that fall from the simulant's blood pressure
    exposure, and the stock risk effects carry it through to the outcomes.

    Configuration (under ``potassium_enriched_salt``)
    -------------------------------------------------
    scenario
        ``baseline`` or one of the paper's five scenarios (``metadata.SCENARIOS``).
    kcl_fraction, replacement_proportion, sodium_floor
        The blend, the share of the targeted salt that is replaced, and the
        intake below which no benefit is modelled.  The paper's sensitivity
        analyses 1, 2, 7, 8, and 9 change these.
    targeted_uptake, targeted_uptake_lmic
        Share of eligible people who use the salt in the targeted scenarios; the
        second, if set, applies outside high-income countries (analyses 12, 13).
    potassium_effect_scale, combined_effect_scale, potassium_effect_in_normotensives
        Analyses 5, 4, and 6.
    sodium_dose_response
        ``mozaffarian_2014`` or ``filippini_2021`` (analysis 3).
    """

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "scenario": metadata.BASELINE.name,
                "kcl_fraction": data_values.KCL_FRACTION_OF_SALT_SUBSTITUTE,
                "replacement_proportion": 1.0,
                "sodium_floor": data_values.SODIUM_FLOOR_G_PER_DAY,
                "targeted_uptake": data_values.TARGETED_UPTAKE,
                "targeted_uptake_lmic": None,
                "potassium_effect_scale": 1.0,
                "combined_effect_scale": 1.0,
                "potassium_effect_in_normotensives": True,
                "sodium_dose_response": "mozaffarian_2014",
            }
        }

    def setup(self, builder: Builder) -> None:
        self.config = self.configuration
        self.scenario = metadata.SCENARIOS_BY_NAME[self.config.scenario]
        if self.config.sodium_dose_response not in ("mozaffarian_2014", "filippini_2021"):
            raise ValueError(f"unknown sodium_dose_response {self.config.sodium_dose_response!r}")
        self.parameters = data_values.ModelParameters.for_draw(
            builder.configuration.input_data.input_draw_number
        )
        self.uptake = self.scenario.uptake
        if self.scenario.eligible != "everyone":
            self.uptake = self.config.targeted_uptake
            high_income = builder.data.load(data_keys.Covariates.HIGH_INCOME)
            if self.config.targeted_uptake_lmic is not None and not high_income:
                self.uptake = self.config.targeted_uptake_lmic

        # The population is a fixed cross-section and the policy does not change
        # during the year, so each simulant's response is worked out once, when
        # the simulant is created, and stored.  The framework exposes each
        # stored column as an attribute of the same name.
        self.randomness = builder.randomness.get_stream("salt_substitute_uptake")
        builder.population.register_initializer(
            initializer=self.initialize_response,
            columns=list(self.COLUMNS),
            required_resources=[
                self.randomness, hypertension.STATUS, sodium.EXPOSURE,
                sodium.DISCRETIONARY_SHARE, "age",
            ],
        )
        builder.value.register_attribute_modifier(
            SBP_EXPOSURE,
            modifier=self.lower_blood_pressure,
            required_resources=[SBP_REDUCTION],
            description="Lower blood pressure by the effect of the salt substitute",
        )

    COLUMNS = {
        USES_SUBSTITUTE: "Whether the simulant's salt is replaced under this scenario",
        SODIUM_REDUCTION: "Fall in sodium intake from the substitute, g/day",
        POTASSIUM_INCREASE: "Rise in potassium intake from the substitute, g/day",
        SBP_REDUCTION: "Fall in systolic blood pressure from the substitute, mmHg",
    }

    ########################
    # Event-driven methods #
    ########################

    def initialize_response(self, pop_data: SimulantData) -> None:
        index = pop_data.index
        pop = self.population_view.get(
            index, [hypertension.STATUS, sodium.EXPOSURE, sodium.DISCRETIONARY_SHARE, "age"]
        )
        status = pop[hypertension.STATUS]
        uses = self.uses_substitute(status, self.randomness.get_draw(index))
        intake = pop[sodium.EXPOSURE].to_numpy()
        less_sodium = uses.to_numpy() * sodium_reduction(
            intake=intake,
            discretionary_share=pop[sodium.DISCRETIONARY_SHARE].to_numpy(),
            sources=self.scenario.sources,
            replacement_proportion=self.config.replacement_proportion,
            kcl_fraction=self.config.kcl_fraction,
            floor=self.config.sodium_floor,
        )
        more_potassium = less_sodium * data_values.POTASSIUM_GAINED_PER_SODIUM_LOST
        lower_pressure = sbp_reduction(
            sodium_reduction=less_sodium,
            potassium_increase=more_potassium,
            baseline_sodium=intake,
            age=pop["age"].to_numpy(),
            hypertensive=(status != metadata.NORMOTENSIVE).to_numpy(),
            parameters=self.parameters,
            sodium_dose_response=self.config.sodium_dose_response,
            potassium_effect_scale=self.config.potassium_effect_scale,
            combined_effect_scale=self.config.combined_effect_scale,
            potassium_effect_in_normotensives=self.config.potassium_effect_in_normotensives,
        )
        self.population_view.initialize(pd.DataFrame({
            USES_SUBSTITUTE: uses,
            SODIUM_REDUCTION: less_sodium,
            POTASSIUM_INCREASE: more_potassium,
            SBP_REDUCTION: lower_pressure,
        }, index=index))

    def uses_substitute(self, status: pd.Series, propensity: pd.Series) -> pd.Series:
        """Who uses the substitute: everyone, or a share of those diagnosed or treated."""
        if self.scenario.sources == "none":
            return pd.Series(False, index=status.index)
        eligible_states = {
            "everyone": list(metadata.HYPERTENSION_STATES),
            "diagnosed": [metadata.DIAGNOSED_UNTREATED, metadata.TREATED],
            "treated": [metadata.TREATED],
        }[self.scenario.eligible]
        return status.isin(eligible_states) & (propensity < self.uptake)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def lower_blood_pressure(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        return exposure - self.population_view.get(index, SBP_REDUCTION)


########################################################################
# The dose-response arithmetic, kept free of the framework so that the #
# closed-form check in ``analytic.py`` runs exactly the same code.      #
########################################################################


def sodium_reduction(intake, discretionary_share, sources, replacement_proportion=1.0,
                     kcl_fraction=data_values.KCL_FRACTION_OF_SALT_SUBSTITUTE,
                     floor=data_values.SODIUM_FLOOR_G_PER_DAY):
    """Fall in sodium intake, g/day, when ``sources`` of salt are replaced.

    Intake is not taken below ``floor``; where it starts at or below the floor,
    nothing changes.
    """
    intake = np.asarray(intake, dtype=float)
    share = {
        "none": 0.0,
        "all": 1.0,
        "discretionary": np.asarray(discretionary_share, dtype=float),
        "non_discretionary": 1.0 - np.asarray(discretionary_share, dtype=float),
    }[sources]
    reduction = intake * share * replacement_proportion * kcl_fraction
    return np.where(intake > floor, np.minimum(reduction, intake - floor), 0.0)


def sodium_effect_per_100_mmol(age, hypertensive, parameters, dose_response="mozaffarian_2014"):
    """Fall in systolic blood pressure, mmHg, per 100 mmol/day less sodium."""
    hypertensive = np.asarray(hypertensive, dtype=bool)
    if dose_response == "filippini_2021":
        return np.where(hypertensive, parameters.sodium_effect_filippini_hypertensive,
                        parameters.sodium_effect_filippini_normotensive)
    age = np.maximum(np.asarray(age, dtype=float), data_values.SODIUM_EFFECT_AGE_FLOOR)
    effect = (
        parameters.sodium_effect_at_50
        + parameters.sodium_effect_per_year * (age - data_values.SODIUM_EFFECT_REFERENCE_AGE)
        + parameters.sodium_effect_hypertension * hypertensive
    )
    return np.maximum(effect, 0.0)


def potassium_effect(potassium_increase, baseline_sodium, parameters):
    """Fall in systolic blood pressure, mmHg, from more potassium, with its ceiling.

    The slope and the ceiling depend on the band of baseline sodium intake
    (under 3, 3 to 4, and 4 g/day or more).
    """
    baseline_sodium = np.asarray(baseline_sodium, dtype=float)
    mmol = np.asarray(potassium_increase, dtype=float) * data_values.MMOL_PER_GRAM_POTASSIUM
    slopes = np.asarray(parameters.potassium_slopes, dtype=float)
    effect = np.zeros(np.broadcast(mmol, baseline_sodium).shape)
    lower = -np.inf
    for band, (upper, _, ceiling) in enumerate(data_values.POTASSIUM_EFFECT_BY_BASELINE_SODIUM):
        in_band = (baseline_sodium >= lower) & (baseline_sodium < upper)
        effect = np.where(in_band, np.minimum(slopes[band] * mmol, ceiling), effect)
        lower = upper
    return effect


def sbp_reduction(sodium_reduction, potassium_increase, baseline_sodium, age, hypertensive,
                  parameters, sodium_dose_response="mozaffarian_2014",
                  potassium_effect_scale=1.0, combined_effect_scale=1.0,
                  potassium_effect_in_normotensives=True):
    """Fall in systolic blood pressure, mmHg: the sodium and potassium effects added."""
    hypertensive = np.asarray(hypertensive, dtype=bool)
    from_sodium = (
        sodium_effect_per_100_mmol(age, hypertensive, parameters, sodium_dose_response)
        * np.asarray(sodium_reduction, dtype=float) * data_values.MMOL_PER_GRAM_SODIUM / 100
    )
    from_potassium = potassium_effect(potassium_increase, baseline_sodium, parameters)
    from_potassium = from_potassium * potassium_effect_scale
    if not potassium_effect_in_normotensives:
        from_potassium = np.where(hypertensive, from_potassium, 0.0)
    return combined_effect_scale * (from_sodium + from_potassium)
