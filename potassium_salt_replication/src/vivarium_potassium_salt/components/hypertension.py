"""Hypertension status: who has it, who knows, and who is treated."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.population import SimulantData

from vivarium_potassium_salt.constants import data_keys, metadata

STATUS = "hypertension_status"
SBP_PROPENSITY = "high_systolic_blood_pressure.propensity"


class HypertensionStatus(Component):
    """Assign each simulant a place on the hypertension care cascade.

    NCD-RisC gives, by sex and age, the prevalence of hypertension and the
    shares of people with hypertension who are diagnosed and who are treated.
    The model needs them twice: sodium lowers blood pressure more in people
    with hypertension, and two scenarios give the salt only to people who are
    diagnosed or treated.

    What NCD-RisC does not give is how hypertension lines up with the GBD blood
    pressure distribution, and the paper does not say what it assumed.  The
    ``association_with_blood_pressure`` setting covers the two ends:

    ``independent``
        Status is drawn without regard to the simulant's blood pressure.  This
        is the default.  It is the individual-level counterpart of what I take
        the paper to do, which is to move a stratum's one blood pressure
        distribution by the prevalence-weighted average effect; the closed
        form can do exactly that (``blood_pressure_response="stratum_mean"``),
        and averts about 5% more in the targeted scenarios when it does.
    ``ranked``
        The people with the highest blood pressure are the ones with
        hypertension.  This is closer to the definition of the condition, but
        it ignores that treatment lowers the pressure of the treated.
    """

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    "prevalence": data_keys.Hypertension.PREVALENCE,
                    "diagnosed": data_keys.Hypertension.DIAGNOSED,
                    "treated": data_keys.Hypertension.TREATED,
                },
                "association_with_blood_pressure": "independent",
            }
        }

    def setup(self, builder: Builder) -> None:
        self.association = self.configuration.association_with_blood_pressure
        if self.association not in ("independent", "ranked"):
            raise ValueError(f"unknown association_with_blood_pressure {self.association!r}")
        self.prevalence = self.build_lookup_table(builder, "prevalence")
        self.diagnosed = self.build_lookup_table(builder, "diagnosed")
        self.treated = self.build_lookup_table(builder, "treated")
        self.status_randomness = builder.randomness.get_stream("hypertension_status")
        self.cascade_randomness = builder.randomness.get_stream("hypertension_care_cascade")
        required = [self.prevalence, self.diagnosed, self.treated,
                    self.status_randomness, self.cascade_randomness]
        if self.association == "ranked":
            required.append(SBP_PROPENSITY)
        builder.population.register_initializer(
            initializer=self.initialize_status, columns=STATUS, required_resources=required
        )
        builder.results.register_stratification(
            STATUS, list(metadata.HYPERTENSION_STATES), requires_attributes=[STATUS]
        )

    def initialize_status(self, pop_data: SimulantData) -> None:
        index = pop_data.index
        if self.association == "ranked":
            rank = 1 - self.population_view.get(index, SBP_PROPENSITY)
        else:
            rank = self.status_randomness.get_draw(index)
        hypertensive = rank < self.prevalence(index)
        # One draw places a person on the cascade, so that the treated are a
        # subset of the diagnosed.
        cascade = self.cascade_randomness.get_draw(index)
        treated = cascade < self.treated(index)
        diagnosed = cascade < np.maximum(self.diagnosed(index), self.treated(index))
        status = pd.Series(metadata.NORMOTENSIVE, index=index, name=STATUS)
        status[hypertensive] = metadata.UNDIAGNOSED
        status[hypertensive & diagnosed] = metadata.DIAGNOSED_UNTREATED
        status[hypertensive & treated] = metadata.TREATED
        self.population_view.initialize(status)
