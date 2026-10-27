import numpy as np
import pandas as pd
import pytest

from sodium_sim import sbp_link
from sodium_sim.scenarios import SCENARIOS, get_scenario


def test_calibration_point():
    assert sbp_link.sbp_reduction(1.0, 50, 140) == pytest.approx(2.026)


def test_sbp_reduction_grows_with_age_and_sbp_and_is_nonnegative():
    assert sbp_link.sbp_reduction(1.0, 70, 150) > sbp_link.sbp_reduction(1.0, 50, 140)
    assert sbp_link.sbp_reduction(1.0, 25, 100) >= 0.0


def test_reformulation_phase_in():
    aus = get_scenario("mandate_australia")
    male_50 = lambda y: aus.schedule(y)["Male"].loc[50]
    assert male_50(2024) == 0.0
    assert male_50(2025) == pytest.approx(0.25 * 0.026)
    assert male_50(2026) == pytest.approx(0.5 * 0.026)
    assert male_50(2027) == pytest.approx(0.026)
    assert male_50(2043) == pytest.approx(0.026)


def test_australia_then_uk_matches_endpoints():
    s = get_scenario("mandate_australia_then_uk")
    assert s.schedule(2027).equals(get_scenario("mandate_australia").schedule(2027))
    assert s.schedule(2030).equals(get_scenario("mandate_uk").schedule(2030))


def test_uniform_scenarios():
    assert get_scenario("kcl_30pct_all_immediate").schedule(2024)["Female"].loc[80] == 0.285
    assert get_scenario("kcl_10pct_all_10y").schedule(2036)["Male"].loc[25] == pytest.approx(0.095)
    assert get_scenario("uk_mass_media").schedule(2030)["Male"].loc[25] == pytest.approx(0.006)
    assert get_scenario("uk_mass_media").schedule(2037)["Male"].loc[25] == 0.0
    assert get_scenario("uk_salt_program").schedule(2030)["Male"].loc[25] == pytest.approx(0.021)
    assert get_scenario("uk_salt_program").schedule(2040)["Male"].loc[25] == pytest.approx(0.021)


def test_reduction_vector_respects_age_and_sex():
    s = get_scenario("mandate_who")
    age = pd.Series([10.0, 27.0, 62.0, 90.0])
    sex = pd.Series(["Male", "Female", "Male", "Female"])
    r = s.reduction(age, sex, 2030)
    assert list(r) == pytest.approx([0.0, 0.109, 0.135, 0.124])


def test_lags_are_bounded_by_unlagged_schedule():
    for s in SCENARIOS.values():
        for year in (2027, 2035, 2043):
            raw = s.schedule(year)["Male"].loc[50]
            assert s.schedule(year, "cvd")["Male"].loc[50] <= max(raw, s.schedule(year - 2)["Male"].loc[50]) + 1e-12
            assert s.schedule(year, "cancer")["Male"].loc[50] <= max(s.schedule(y)["Male"].loc[50] for y in range(year - 20, year - 1)) + 1e-12
