"""Dietary sodium: how much, and how much of it is added at home."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder

from vivarium_potassium_salt.constants import data_keys

EXPOSURE = "diet_high_in_sodium.exposure"
DISCRETIONARY_SHARE = "diet_high_in_sodium.discretionary_share"


class DietarySodium(Component):
    """Sodium intake in g/day and the share of it from discretionary salt.

    As in the paper, every simulant has the mean intake of their sex and age
    band, and the discretionary share is one number for the whole country.
    GBD describes intake as a distribution; the paper uses its mean, and so
    does this component.
    """

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {"exposure": data_keys.Sodium.EXPOSURE},
                # 'regression' predicts the share from GDP per capita for every
                # country.  'reported_where_available' prefers a survey estimate
                # where one exists (the paper's sensitivity analyses 10 and 11).
                "discretionary_share_source": "regression",
            }
        }

    def setup(self, builder: Builder) -> None:
        config = self.configuration
        self.exposure_table = self.build_lookup_table(builder, "exposure")
        share = builder.data.load(data_keys.Sodium.DISCRETIONARY_SHARE)
        self.discretionary_share = float(share["value"].iloc[0])
        if config.discretionary_share_source == "reported_where_available":
            reported = builder.data.load(data_keys.Sodium.REPORTED_DISCRETIONARY_SHARE)
            if not np.isnan(reported):
                self.discretionary_share = float(reported)
        elif config.discretionary_share_source != "regression":
            raise ValueError(
                f"unknown discretionary_share_source {config.discretionary_share_source!r}"
            )
        builder.value.register_attribute_producer(
            EXPOSURE,
            source=self.exposure_table,
            description="Usual sodium intake, g/day (24-hour urinary excretion)",
        )
        builder.value.register_attribute_producer(
            DISCRETIONARY_SHARE,
            source=lambda index: pd.Series(self.discretionary_share, index=index),
            description="Share of sodium intake from salt added at the table or in cooking",
        )
