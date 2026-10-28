"""Custom vivarium components for the sodium reduction model.

Everything else in the model (population, fertility, mortality, disease
state machines, risk exposures, risk effects, observers) comes from
``vivarium.public_health``. These three components add the pieces specific
to this paper: the intervention schedule, the sodium-to-SBP link, and an
observer for person-time so that HALYs can be computed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.public_health.disease import DiseaseModel
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
    Only the part of the reduction above the 2 g/day floor moves SBP.
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
        delta = sbp_reduction(sodium0, sodium0 * (1 - reduction), pop["age"], sbp)
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


class RateTrends(Component):
    """Apply the paper's business-as-usual time trends to disease rates.

    Dhungel et al. forecast incidence, case fatality and all-cause mortality
    from their 1990 to 2019 trends out to 2034, then hold them constant. This
    component multiplies each cause's incidence and excess mortality pipelines,
    the unmodeled-cause mortality pipeline and the all-cause mortality hazard
    by ``exp(apc * (min(year, end_year) - base_year))``, with annual percentage
    changes (as log slopes) read from a CSV with columns ``cause``, ``measure``
    (``incidence``, ``case_fatality`` or ``deaths``) and ``apc``.
    """

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {"rate_trends": {"file": "data/gbd2019/trends_apc.csv", "base_year": 2019, "end_year": 2034}}

    def setup(self, builder: Builder) -> None:
        from sodium_sim.build_artifact import ROOT

        config = builder.configuration.rate_trends
        self.base_year, self.end_year = int(config.base_year), int(config.end_year)
        self.clock = builder.time.clock()
        apc = pd.read_csv(ROOT / config.file)
        self.apc = {(r.cause, r.measure): float(r.apc) for r in apc.itertuples()}
        causes = [c.name.split(".")[-1] for c in builder.components.get_components_by_type(DiseaseModel)]
        # Risk-affected pipelines (incidence, excess mortality, unmodeled CSMR) combine their
        # modifiers multiplicatively: a modifier takes the index and returns a factor. The
        # all-cause hazard uses the replace combiner: a modifier takes (index, value).
        for cause in causes:
            if (cause, "incidence") in self.apc:
                builder.value.register_attribute_modifier(
                    f"{cause}.incidence_rate", modifier=self._factor_modifier(cause, "incidence"),
                    description="BAU incidence trend")
            if (cause, "case_fatality") in self.apc:
                builder.value.register_attribute_modifier(
                    f"{cause}.excess_mortality_rate", modifier=self._factor_modifier(cause, "case_fatality"),
                    description="BAU case fatality trend")
        if ("aortic_aneurysm", "deaths") in self.apc:
            builder.value.register_attribute_modifier(
                "affected_unmodeled.cause_specific_mortality_rate",
                modifier=self._factor_modifier("aortic_aneurysm", "deaths"), description="BAU aortic aneurysm trend")
        if ("all_causes", "deaths") in self.apc:
            builder.value.register_attribute_modifier(
                "mortality_rate", modifier=self._value_modifier("all_causes", "deaths"), description="BAU all-cause trend")

    def factor(self, cause: str, measure: str) -> float:
        years = min(self.clock().year, self.end_year) - self.base_year
        return float(np.exp(self.apc[(cause, measure)] * years))

    def _factor_modifier(self, cause: str, measure: str):
        def factor(index: pd.Index) -> pd.Series:
            return pd.Series(self.factor(cause, measure), index=index)

        return factor

    def _value_modifier(self, cause: str, measure: str):
        def scale(index: pd.Index, value):
            return value * self.factor(cause, measure)

        return scale
