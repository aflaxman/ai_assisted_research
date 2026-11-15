"""The burden of each cause, as annual rates that risk effects can act on."""
from __future__ import annotations

from typing import Any

from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.public_health.causal_factor import register_risk_affected_rate_producer

from vivarium_potassium_salt.constants import data_keys, metadata


def rate_pipeline(cause: str, measure: str) -> str:
    """Name of the pipeline holding a cause's rate of deaths, new cases, or DALYs."""
    return f"{cause}.{data_keys.RATE_MEASURE[measure]}"


class CauseBurden(Component):
    """Annual rates of deaths, new cases, and DALYs from one cause.

    A dynamic vivarium model would let blood pressure change incidence and let
    deaths and disability follow from the course of disease.  The paper does
    something simpler and static: it scales this year's deaths, new cases, and
    DALYs by one potential impact fraction.  This component is the static
    version.  Each measure is a rate per person-year, read from GBD, that a
    ``RiskEffect`` can multiply by a simulant's relative risk.  Nobody is made
    ill and nobody dies; an observer adds up the rates.

    Parameters
    ----------
    cause
        Name of the cause, as in ``vivarium.gbd_mapping``.
    measures
        Comma-separated subset of ``deaths``, ``incident_cases``, ``dalys``.
    """

    def __init__(self, cause: str, measures: str = "deaths"):
        super().__init__()
        self.cause = cause
        self.measures = [m.strip() for m in measures.split(",")]
        for measure in self.measures:
            if cause not in metadata.CAUSES_BY_MEASURE[measure]:
                raise ValueError(f"the model has no {measure} for {cause}")

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    data_keys.RATE_MEASURE[measure]: data_keys.burden_rate(self.cause, measure)
                    for measure in self.measures
                }
            }
        }

    def setup(self, builder: Builder) -> None:
        for measure in self.measures:
            table = self.build_lookup_table(builder, data_keys.RATE_MEASURE[measure])
            register_risk_affected_rate_producer(
                builder,
                rate_pipeline(self.cause, measure),
                source=table,
                description=f"Annual rate of {measure.replace('_', ' ')} from {self.cause}",
            )
