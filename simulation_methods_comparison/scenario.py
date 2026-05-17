"""Shared scenario parameters for the four-lens comparison.

A respiratory virus enters a small town of 1,000 people. Every model in this
project uses the same disease, the same town, and the same intervention choice
(a one-shot mass vaccination campaign). Differences between model outputs come
from what each method *can see*, not from different inputs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    # Population
    N: int = 1_000
    initial_infected: int = 3

    # Disease (loosely calibrated to a moderately transmissible respiratory virus)
    R0: float = 2.0
    infectious_days: float = 5.0  # mean duration of infectiousness

    # Care-seeking & clinic. A small-town clinic with one doctor; visits average
    # 90 minutes door-to-door. Capacity (~16 patients/day) was chosen so the
    # outbreak peak meaningfully overruns the clinic - that is the DES story.
    care_seeking_prob: float = 0.7     # fraction of cases that seek clinic care
    service_minutes: float = 90.0      # mean clinic visit length
    clinic_servers: int = 1            # one doctor on duty

    # Outcomes & costs (US dollars; illustrative, not calibrated)
    hospitalization_rate: float = 0.05
    cost_mild_case: float = 500.0
    cost_hospitalization: float = 8_000.0
    cost_vaccine_dose: float = 50.0

    # Vaccination campaign
    vaccine_effectiveness: float = 0.90
    achievable_coverage: float = 0.70   # fraction of population reached

    # Simulation horizon
    horizon_days: int = 120

    @property
    def gamma(self) -> float:
        """Recovery rate per day."""
        return 1.0 / self.infectious_days

    @property
    def beta(self) -> float:
        """Transmission rate per day."""
        return self.R0 * self.gamma

    def effective_susceptible_fraction(self, vaccinated: bool) -> float:
        """Fraction of the population that remains susceptible after the campaign."""
        if not vaccinated:
            return 1.0
        return 1.0 - self.achievable_coverage * self.vaccine_effectiveness


DEFAULT = Scenario()
