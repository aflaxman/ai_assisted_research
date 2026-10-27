"""Custom vivarium components for the sodium reduction model.

Everything else in the model (population, fertility, mortality, disease
state machines, risk exposures, risk effects, observers) comes from
``vivarium.public_health``. These three components add the pieces specific
to this paper: the intervention schedule, the sodium-to-SBP link, and an
observer for person-time so that HALYs can be computed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.public_health.results.observer import PublicHealthObserver
from vivarium.public_health.utilities import to_years

from sodium_sim.sbp_link import sbp_reduction
from sodium_sim.scenarios import get_scenario

SODIUM_EXPOSURE = "diet_high_in_sodium.exposure"
SBP_EXPOSURE = "high_systolic_blood_pressure.exposure"


class SodiumReduction(Component):
    """Scale the sodium exposure pipeline down by the scenario's lagged reduction.

    The modified sodium exposure feeds the stomach cancer risk effect, so it
    carries the paper's 2-to-20-year lag for cancer incidence.
    """

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {"sodium_intervention": {"scenario": "bau"}}

    @property
    def columns_required(self) -> list[str]:
        return ["age", "sex"]

    def setup(self, builder: Builder) -> None:
        self.scenario = get_scenario(builder.configuration.sodium_intervention.scenario)
        self.clock = builder.time.clock()
        builder.value.register_attribute_modifier(
            SODIUM_EXPOSURE,
            modifier=self.reduce_sodium,
            required_resources=["age", "sex"],
            description="Proportional sodium reduction from the intervention",
        )

    def reduce_sodium(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        pop = self.population_view.get(index, ["age", "sex"])
        year = self.clock().year
        reduction = self.scenario.reduction(pop["age"], pop["sex"], year, lag="cancer")
        return exposure * (1 - reduction)


class SodiumToSBP(Component):
    """Lower SBP by the sodium removed times the age- and SBP-specific slope.

    The absolute sodium reduction is the scenario's proportional reduction
    (lagged 0 to 3 years, the paper's CVD lag) times the simulant's
    pre-intervention sodium intake, which is read from the exposure pipeline
    in ``source`` mode so the intervention modifier is not applied twice.
    The scenario name is read from the ``sodium_intervention`` block that
    :class:`SodiumReduction` declares.
    """

    @property
    def columns_required(self) -> list[str]:
        return ["age", "sex", SODIUM_EXPOSURE]

    def setup(self, builder: Builder) -> None:
        self.scenario = get_scenario(builder.configuration.sodium_intervention.scenario)
        self.clock = builder.time.clock()
        builder.value.register_attribute_modifier(
            SBP_EXPOSURE,
            modifier=self.lower_sbp,
            required_resources=["age", "sex", SODIUM_EXPOSURE],
            description="SBP reduction from the sodium intervention",
        )

    def lower_sbp(self, index: pd.Index, sbp: pd.Series) -> pd.Series:
        pop = self.population_view.get(index, ["age", "sex"])
        year = self.clock().year
        reduction = self.scenario.reduction(pop["age"], pop["sex"], year, lag="cvd")
        sodium0 = self.population_view.get(index, SODIUM_EXPOSURE, mode="source")
        delta = sbp_reduction(reduction * sodium0, pop["age"], sbp)
        return sbp - delta


class PersonTimeObserver(PublicHealthObserver):
    """Record person-years lived, stratified like the other observers."""

    def setup(self, builder: Builder) -> None:
        self.step_size = pd.Timedelta(days=builder.configuration.time.step_size)

    def register_observations(self, builder: Builder) -> None:
        self.register_adding_observation(
            builder=builder,
            name="person_time",
            pop_filter="is_alive == True",
            when="time_step",
            priority=1,
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.person_years,
        )

    def person_years(self, pop: pd.DataFrame) -> float:
        return len(pop) * to_years(self.step_size)
