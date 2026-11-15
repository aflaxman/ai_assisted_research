"""A cross-section of adults for one location and one year."""
from __future__ import annotations

import numpy as np
import pandas as pd
from vivarium.engine import Component
from vivarium.engine.framework.engine import Builder
from vivarium.engine.framework.population import SimulantData

from vivarium_potassium_salt.constants import metadata


class AdultPopulation(Component):
    """Simulants aged 25 and over, the same number in every sex and age band.

    The paper's estimate is for a single year, so nobody here ages, is born, or
    dies.  The simulants are a sample from which to estimate, for each sex and
    five-year age band, how much a salt policy changes the average rate of each
    outcome.  Those stratum averages are later multiplied by the stratum's GBD
    burden, so the sample does not need to mirror the age structure, and equal
    numbers per stratum keep the oldest bands, where most deaths occur and few
    people live, as well estimated as the youngest.

    This also registers the ``age_group`` and ``sex`` stratifications that the
    observers use.
    """

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("age_within_band")
        builder.population.register_initializer(
            initializer=self.initialize_age_and_sex,
            columns=["age", "sex"],
            required_resources=[self.randomness],
        )
        self.register_stratifications(builder)

    def register_stratifications(self, builder: Builder) -> None:
        edges = [band.start for band in metadata.AGE_BANDS] + [np.inf]
        builder.results.register_binned_stratification(
            target="age",
            binned_column="age_group",
            bin_edges=edges,
            labels=[band.label for band in metadata.AGE_BANDS],
        )
        builder.results.register_stratification(
            "sex", list(metadata.SEXES), requires_attributes=["sex"]
        )

    def initialize_age_and_sex(self, pop_data: SimulantData) -> None:
        index = pop_data.index
        strata = [(sex, band) for sex in metadata.SEXES for band in metadata.AGE_BANDS]
        stratum = np.arange(len(index)) % len(strata)
        start = np.array([band.start for _, band in strata])[stratum]
        width = np.array([band.end - band.start for _, band in strata])[stratum]
        age = start + width * self.randomness.get_draw(index).to_numpy()
        sex = np.array([sex for sex, _ in strata])[stratum]
        self.population_view.initialize(pd.DataFrame({"age": age, "sex": sex}, index=index))
