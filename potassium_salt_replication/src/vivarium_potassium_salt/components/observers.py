"""Observers: add up expected events instead of counting simulated ones."""
from __future__ import annotations

import pandas as pd
from vivarium.engine import Observer
from vivarium.engine.framework.engine import Builder

from vivarium_potassium_salt.components import burden, hypertension, intervention, kidney, sodium
from vivarium_potassium_salt.constants import metadata

PERSON_COUNT = "person_count"
BURDEN_RESULTS = "burden"
POLICY_RESULTS = "salt_policy"


def _keep_columns(measure: str, results: pd.DataFrame) -> pd.DataFrame:
    """Results formatter: one row per stratum, one column per summed quantity."""
    return results.reset_index()


class BurdenObserver(Observer):
    """Sum each outcome's rate over the simulants in every stratum.

    A vivarium model usually rolls dice for each simulant and counts the deaths
    that happen.  For a one-year, static estimate that adds noise and nothing
    else, so this observer adds up the *rates* instead: the sum of the annual
    death rates of a group of simulants is the number of deaths expected among
    them this year.  Dividing a scenario's sum by the baseline's gives the
    stratum's potential impact fraction, with the simulants and their blood
    pressures held in common by the framework's common random numbers.

    The sums are stratified by age band, sex, and kidney disease stage, the
    last so that deaths averted among people with kidney disease, and the
    change in their hyperkalaemia risk, can be read off the same table.

    Parameters
    ----------
    measures
        Comma-separated subset of ``deaths``, ``incident_cases``, ``dalys``.
    """

    def __init__(self, measures: str = "deaths"):
        super().__init__()
        self.measures = [m.strip() for m in measures.split(",")]

    def register_observations(self, builder: Builder) -> None:
        self.rate_pipelines = [
            burden.rate_pipeline(cause, measure)
            for measure in self.measures
            for cause in metadata.CAUSES_BY_MEASURE[measure]
        ]
        sources = self.rate_pipelines + [kidney.HYPERKALAEMIA_RELATIVE_RISK]
        builder.results.register_adding_observation(
            name=BURDEN_RESULTS,
            requires_attributes=sources,
            aggregator_sources=sources,
            aggregator=self.sum_rates,
            additional_stratifications=[kidney.STAGE],
            results_formatter=_keep_columns,
        )

    @staticmethod
    def sum_rates(group: pd.DataFrame) -> pd.Series:
        totals = group.sum()
        totals[PERSON_COUNT] = len(group)
        return totals


class SaltPolicyObserver(Observer):
    """Record who uses the substitute and what it does to their intake and pressure."""

    SOURCES = [
        intervention.USES_SUBSTITUTE,
        intervention.SODIUM_REDUCTION,
        intervention.POTASSIUM_INCREASE,
        intervention.SBP_REDUCTION,
        intervention.SBP_EXPOSURE,
        sodium.EXPOSURE,
    ]

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name=POLICY_RESULTS,
            requires_attributes=self.SOURCES,
            aggregator_sources=self.SOURCES,
            aggregator=self.sum_quantities,
            additional_stratifications=[hypertension.STATUS],
            results_formatter=_keep_columns,
        )

    @staticmethod
    def sum_quantities(group: pd.DataFrame) -> pd.Series:
        totals = group.astype(float).sum()
        totals[PERSON_COUNT] = len(group)
        return totals
