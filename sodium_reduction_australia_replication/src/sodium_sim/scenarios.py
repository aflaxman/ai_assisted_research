"""Intervention scenarios: proportional sodium reduction by age, sex and year.

The schedules transcribe Table 1 of Dhungel et al. (2025). Each scenario
maps a calendar year to a proportional reduction in sodium intake (0 to 1)
for each sex and five-year age band from 25 upward; ages under 25 get zero
for the reformulation scenarios because the source effect sizes (Trieu et
al.) are only given for adults.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

AGE_BAND_STARTS = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
SEXES = ("Male", "Female")

# Mean proportionate reduction in sodium at full effect, Table 1, rows (Male, Female).
_REFORMULATION_TABLES: dict[str, list[tuple[float, float]]] = {
    "australia": [
        (0.026, 0.030), (0.031, 0.033), (0.033, 0.028), (0.030, 0.029),
        (0.030, 0.034), (0.026, 0.026), (0.036, 0.027), (0.038, 0.033),
        (0.034, 0.034), (0.036, 0.034), (0.041, 0.030), (0.037, 0.034),
    ],
    "uk": [
        (0.048, 0.057), (0.059, 0.061), (0.060, 0.057), (0.059, 0.057),
        (0.064, 0.060), (0.059, 0.063), (0.088, 0.056), (0.078, 0.072),
        (0.076, 0.066), (0.085, 0.081), (0.080, 0.077), (0.086, 0.088),
    ],
    "who": [
        (0.099, 0.109), (0.121, 0.111), (0.117, 0.125), (0.110, 0.113),
        (0.117, 0.119), (0.112, 0.113), (0.130, 0.100), (0.135, 0.136),
        (0.126, 0.126), (0.127, 0.135), (0.143, 0.119), (0.133, 0.124),
    ],
}


def _table(name: str) -> pd.DataFrame:
    rows = _REFORMULATION_TABLES[name]
    return pd.DataFrame(rows, index=AGE_BAND_STARTS, columns=list(SEXES))


def _uniform(value: float) -> pd.DataFrame:
    return pd.DataFrame(value, index=AGE_BAND_STARTS, columns=list(SEXES))


def _ramp(year: int) -> float:
    """Phase-in of the reformulation targets: 25% in 2025, 50% in 2026, full from 2027."""
    return {2024: 0.0, 2025: 0.25, 2026: 0.5}.get(year, 1.0 if year > 2026 else 0.0)


def _reformulation(name: str) -> Callable[[int], pd.DataFrame]:
    table = _table(name)
    return lambda year: table * _ramp(year)


def _australia_then_uk(year: int) -> pd.DataFrame:
    aus, uk = _table("australia"), _table("uk")
    if year <= 2027:
        return aus * _ramp(year)
    if year == 2028:
        return 0.75 * aus + 0.25 * uk
    if year == 2029:
        return 0.5 * aus + 0.5 * uk
    return uk


def _kcl_discretionary(year: int) -> pd.DataFrame:
    return _uniform(min(max(year - 2024, 0), 3) / 3 * 0.045)


def _kcl_10pct_10y(year: int) -> pd.DataFrame:
    return _uniform(min(max(year - 2026, 0), 10) / 10 * 0.095)


def _uk_mass_media(year: int) -> pd.DataFrame:
    if year <= 2030:
        value = max(year - 2023, 0) / 7 * 0.006
    else:
        value = max(2037 - year, 0) / 7 * 0.006
    return _uniform(value)


def _uk_salt_program(year: int) -> pd.DataFrame:
    return _uniform(min(max(year - 2023, 0), 7) * 0.003)


@dataclass
class Scenario:
    """A sodium reduction scenario with the paper's time lags to outcomes."""

    name: str
    label: str
    raw: Callable[[int], pd.DataFrame]
    _cache: dict = field(default_factory=dict, repr=False)

    def schedule(self, year: int, lag: str = "none") -> pd.DataFrame:
        """Return the effective reduction table for a year.

        ``lag="cvd"`` averages the reduction over the current and previous two
        years (the paper's "mean PIF in the last 3 years"); ``lag="cancer"``
        averages over 2 to 20 years back (stomach cancer); ``"none"`` is the
        unlagged schedule.
        """
        key = (year, lag)
        if key not in self._cache:
            if lag == "none":
                years = [year]
            elif lag == "cvd":
                years = [year - 2, year - 1, year]
            elif lag == "cancer":
                years = list(range(year - 20, year - 1))
            else:
                raise ValueError(f"Unknown lag {lag!r}")
            self._cache[key] = sum(self.raw(y) for y in years) / len(years)
        return self._cache[key]

    def reduction(
        self, age: pd.Series, sex: pd.Series, year: int, lag: str = "none"
    ) -> pd.Series:
        """Return the proportional sodium reduction for each simulant."""
        table = self.schedule(year, lag)
        band = np.searchsorted(AGE_BAND_STARTS, age.to_numpy(), side="right") - 1
        out = np.zeros(len(age))
        for i, s in enumerate(SEXES):
            mask = (sex.to_numpy() == s) & (band >= 0)
            out[mask] = table[s].to_numpy()[band[mask]]
        return pd.Series(out, index=age.index)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("bau", "Business as usual", lambda year: _uniform(0.0)),
        Scenario("mandate_australia", "Mandating - Australia", _reformulation("australia")),
        Scenario("mandate_uk", "Mandating - UK", _reformulation("uk")),
        Scenario("mandate_who", "Mandating - WHO", _reformulation("who")),
        Scenario("mandate_australia_then_uk", "Mandating - Aus then UK", _australia_then_uk),
        Scenario("kcl_30pct_all_immediate", "30% KCl substitution, all foods, immediate",
                 lambda year: _uniform(0.285 if year >= 2024 else 0.0)),
        Scenario("kcl_10pct_all_10y", "10% KCl substitution, all foods, over 10 years",
                 _kcl_10pct_10y),
        Scenario("kcl_30pct_discretionary_3y", "30% KCl substitution, discretionary, over 3 years",
                 _kcl_discretionary),
        Scenario("uk_mass_media", "UK mass media campaign", _uk_mass_media),
        Scenario("uk_salt_program", "UK salt reduction program", _uk_salt_program),
    ]
}

# Figure 1 of the paper: median HALYs gained over 2024-2043, 3% discount, with 95% UI.
PAPER_HALYS: dict[str, tuple[float, float, float]] = {
    "mandate_australia": (11100, 7530, 15800),
    "mandate_uk": (25300, 16800, 37100),
    "mandate_who": (40100, 27200, 56700),
    "mandate_australia_then_uk": (21000, 14200, 30500),
    "kcl_30pct_all_immediate": (103000, 71100, 145000),
    "kcl_10pct_all_10y": (15600, 10500, 22900),
    "kcl_30pct_discretionary_3y": (13100, 8970, 19100),
    "uk_mass_media": (1130, 560, 2100),
    "uk_salt_program": (6880, 4640, 9820),
}


def get_scenario(name: str) -> Scenario:
    """Return a scenario by name."""
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"Unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
