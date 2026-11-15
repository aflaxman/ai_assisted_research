"""Advanced chronic kidney disease and serum potassium."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.population import SimulantData

from vivarium_potassium_salt.components import intervention
from vivarium_potassium_salt.constants import data_keys, data_values, metadata
from vivarium_potassium_salt.data import literature

STAGE = "ckd_stage"
SERUM_POTASSIUM_MEAN = "serum_potassium.stage_mean"
HYPERKALAEMIA_RELATIVE_RISK = "hyperkalaemia.relative_risk"
REFERENCE_SERUM_POTASSIUM = 4.2


class ChronicKidneyDisease(Component):
    """Who has stage 3 to 5 kidney disease, and what extra potassium does to them.

    Healthy kidneys clear extra dietary potassium; failing ones do not, and a
    high serum potassium can stop the heart.  That is the harm the paper sets
    against the benefit of lower blood pressure.

    Each simulant is given a stage (none, G3a, G3b, G4, or G5) from the GBD
    prevalence of kidney dysfunction for their age and sex.  Serum potassium at
    each stage is normally distributed.  Extra dietary potassium from the salt
    substitute raises its mean, by more in later stages, and the distribution
    stretches with its mean, as in the model this one descends from.  The
    relative risk of cardiovascular death is one below 5.5 mmol/L and follows
    the CKD Prognosis Consortium's curve above it.

    A simulant's relative risk is the *average* over the potassium distribution
    of their stage (``literature.expected_hyperkalaemia_relative_risk``), not
    the risk at one sampled level.  This is the same choice the observers make
    when they add up rates instead of counting deaths: few people at any one
    time are above 5.5 mmol/L, and sampling them would spend the simulation's
    precision on noise.

    People without a stage keep a reference potassium and a relative risk of
    one: the paper assumes their kidneys cope.
    """

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {self.name: {"data_sources": {"exposure": data_keys.KidneyDysfunction.EXPOSURE}}}

    def setup(self, builder: Builder) -> None:
        self.parameters = data_values.ModelParameters.for_draw(
            builder.configuration.input_data.input_draw_number
        )
        self.prevalence = self.build_lookup_table(
            builder, "exposure", data_source=self.stage_prevalence(builder)
        )
        self.randomness = builder.randomness.get_stream("ckd_stage")
        builder.population.register_initializer(
            initializer=self.initialize_stage,
            columns=STAGE,
            required_resources=[self.prevalence, self.randomness],
        )
        builder.value.register_attribute_producer(
            SERUM_POTASSIUM_MEAN,
            source=self.serum_potassium_mean,
            required_resources=[STAGE, intervention.POTASSIUM_INCREASE],
            description="Mean serum potassium of the simulant's CKD stage, mmol/L",
        )
        builder.value.register_attribute_producer(
            HYPERKALAEMIA_RELATIVE_RISK,
            source=self.hyperkalaemia_relative_risk,
            required_resources=[STAGE, intervention.POTASSIUM_INCREASE],
            description="Relative risk of cardiovascular death from serum potassium above 5.5",
        )
        builder.results.register_stratification(
            STAGE, list(metadata.CKD_STATES), requires_attributes=[STAGE]
        )

    def stage_prevalence(self, builder: Builder) -> pd.DataFrame:
        """Prevalence of stages 5, 4, and 3, one column each, by sex and age."""
        data = self.get_data(builder, self.configuration.data_sources.exposure)
        names = {"cat1": "stage5", "cat2": "stage4", "cat3": "stage3"}
        data = data[data["parameter"].isin(names)]
        index = [c for c in data.columns if c not in ("parameter", "value")]
        wide = data.pivot_table(index=index, columns="parameter", values="value")
        wide.columns.name = None
        return wide.rename(columns=names)[["stage5", "stage4", "stage3"]]

    ########################
    # Event-driven methods #
    ########################

    def initialize_stage(self, pop_data: SimulantData) -> None:
        index = pop_data.index
        prevalence = self.prevalence(index)
        draw = self.randomness.get_draw(index)
        # Stack the stages along the unit interval from most to least severe.
        g5 = prevalence["stage5"]
        g4 = g5 + prevalence["stage4"]
        g3b = g4 + prevalence["stage3"] * (1 - data_values.STAGE_3A_SHARE_OF_STAGE_3)
        g3a = g4 + prevalence["stage3"]
        stage = pd.Series(metadata.NO_ADVANCED_CKD, index=index, name=STAGE)
        stage[draw < g3a] = "G3a"
        stage[draw < g3b] = "G3b"
        stage[draw < g4] = "G4"
        stage[draw < g5] = "G5"
        self.population_view.initialize(stage)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def serum_potassium_mean(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index, [STAGE, intervention.POTASSIUM_INCREASE])
        mean, _ = serum_potassium_distribution(
            pop[STAGE].to_numpy(), pop[intervention.POTASSIUM_INCREASE].to_numpy(),
            self.parameters)
        return pd.Series(mean, index=index)

    def hyperkalaemia_relative_risk(self, index: pd.Index) -> pd.Series:
        pop = self.population_view.get(index, [STAGE, intervention.POTASSIUM_INCREASE])
        return pd.Series(
            mean_hyperkalaemia_relative_risk(
                pop[STAGE].to_numpy(), pop[intervention.POTASSIUM_INCREASE].to_numpy(),
                self.parameters),
            index=index,
        )


def serum_potassium_distribution(stage, dietary_increase, parameters):
    """Mean and standard deviation of serum potassium, mmol/L, for each simulant.

    Before any dietary change, potassium at a stage is normal with that stage's
    mean and standard deviation.  ``dietary_increase`` g/day more potassium
    raises the mean by the stage's dose-response slope, and the standard
    deviation in proportion to the mean.  People without a stage get the
    reference level and no spread.
    """
    stage = np.asarray(stage)
    dietary_increase = np.asarray(dietary_increase, dtype=float)
    mean = np.full(len(stage), REFERENCE_SERUM_POTASSIUM)
    sd = np.zeros(len(stage))
    for name in metadata.CKD_STAGES:
        here = stage == name
        before = parameters.serum_potassium_mean[name]
        rise = (
            parameters.serum_potassium_per_g
            * data_values.SERUM_POTASSIUM_RESPONSE_MULTIPLIER[name]
            * dietary_increase[here]
        )
        mean[here] = before + rise
        sd[here] = data_values.SERUM_POTASSIUM_MMOL_PER_L[name][1] * (1 + rise / before)
    return mean, sd


def mean_hyperkalaemia_relative_risk(stage, dietary_increase, parameters):
    """Each simulant's relative risk, averaged over their stage's potassium distribution."""
    mean, sd = serum_potassium_distribution(stage, dietary_increase, parameters)
    risk = np.ones(len(mean))
    affected = sd > 0
    if affected.any():
        risk[affected] = literature.expected_hyperkalaemia_relative_risk(
            mean[affected], sd[affected], parameters.hyperkalaemia_hazard_z)
    return risk
