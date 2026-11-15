"""The arithmetic that turns replaced salt into a fall in blood pressure."""
import numpy as np
import pytest

from vivarium_potassium_salt.components import intervention
from vivarium_potassium_salt.constants import data_values

PARAMETERS = data_values.ModelParameters()


def test_potassium_gained_per_gram_of_sodium_lost():
    # 1 g of salt: 0.393 g sodium out, 0.524 g potassium in.
    assert data_values.SODIUM_FRACTION_OF_NACL == pytest.approx(0.3934, abs=1e-4)
    assert data_values.POTASSIUM_FRACTION_OF_KCL == pytest.approx(0.5244, abs=1e-4)
    assert data_values.POTASSIUM_GAINED_PER_SODIUM_LOST == pytest.approx(1.333, abs=1e-3)


def test_replacing_all_salt_removes_a_quarter_of_the_sodium():
    assert intervention.sodium_reduction(4.0, 0.6, "all") == pytest.approx(1.0)


def test_the_two_sources_of_salt_add_up_to_all_of_it():
    intake, share = 4.4, 0.35
    parts = (intervention.sodium_reduction(intake, share, "discretionary")
             + intervention.sodium_reduction(intake, share, "non_discretionary"))
    assert parts == pytest.approx(intervention.sodium_reduction(intake, share, "all"))


@pytest.mark.parametrize("intake, expected", [
    (1.8, 0.0),     # already below the floor: nothing changes
    (2.0, 0.0),     # at the floor
    (2.4, 0.4),     # a quarter would take it to 1.8, so it stops at 2.0
    (2.8, 0.7),     # 2.8 -> 2.1, floor not reached
])
def test_sodium_is_not_taken_below_the_floor(intake, expected):
    assert intervention.sodium_reduction(intake, 0.5, "all") == pytest.approx(expected)


def test_sensitivity_switches_scale_the_sodium_reduction():
    full = intervention.sodium_reduction(5.0, 0.5, "all")
    assert intervention.sodium_reduction(5.0, 0.5, "all", replacement_proportion=0.5) == \
        pytest.approx(full / 2)
    assert intervention.sodium_reduction(5.0, 0.5, "all", kcl_fraction=0.15) == \
        pytest.approx(full * 0.15 / 0.25)
    # a higher floor bites sooner: 5.0 -> 3.75 without it, held at 4.5 with it
    assert intervention.sodium_reduction(5.0, 0.5, "all", floor=4.5) == pytest.approx(0.5)


def test_sodium_effect_follows_the_published_meta_regression():
    effect = intervention.sodium_effect_per_100_mmol
    assert effect(50, False, PARAMETERS) == pytest.approx(3.735)
    assert effect(50, True, PARAMETERS) == pytest.approx(3.735 + 1.874)
    assert effect(70, False, PARAMETERS) == pytest.approx(3.735 + 20 * 0.105)
    # no smaller below age 40
    assert effect(27.5, False, PARAMETERS) == pytest.approx(effect(40, False, PARAMETERS))
    assert effect(40, False, PARAMETERS) == pytest.approx(3.735 - 10 * 0.105)


def test_alternative_sodium_dose_response_ignores_age():
    effect = intervention.sodium_effect_per_100_mmol
    assert effect(30, False, PARAMETERS, "filippini_2021") == pytest.approx(2.30)
    assert effect(80, True, PARAMETERS, "filippini_2021") == pytest.approx(6.50)


@pytest.mark.parametrize("baseline_sodium, slope, ceiling", [
    (2.5, 0.07, 2.2), (3.0, 0.11, 3.3), (3.9, 0.11, 3.3), (4.0, 0.11, 6.5), (7.0, 0.11, 6.5),
])
def test_potassium_effect_has_a_slope_and_a_ceiling_by_sodium_band(baseline_sodium, slope, ceiling):
    one_mmol = 1 / data_values.MMOL_PER_GRAM_POTASSIUM
    assert intervention.potassium_effect(10 * one_mmol, baseline_sodium, PARAMETERS) == \
        pytest.approx(10 * slope)
    assert intervention.potassium_effect(1000 * one_mmol, baseline_sodium, PARAMETERS) == \
        pytest.approx(ceiling)


def test_blood_pressure_fall_is_the_sum_of_the_two_effects():
    less_sodium, more_potassium = 1.0, 1.333
    fall = intervention.sbp_reduction(less_sodium, more_potassium, 4.0, 60, True, PARAMETERS)
    from_sodium = (3.735 + 10 * 0.105 + 1.874) * data_values.MMOL_PER_GRAM_SODIUM / 100
    from_potassium = 0.11 * more_potassium * data_values.MMOL_PER_GRAM_POTASSIUM
    assert fall == pytest.approx(from_sodium + from_potassium)

    def variant(**settings):
        return intervention.sbp_reduction(less_sodium, more_potassium, 4.0, 60, True,
                                          PARAMETERS, **settings)

    assert variant(combined_effect_scale=0.8) == pytest.approx(0.8 * fall)
    assert variant(potassium_effect_scale=0.5) == pytest.approx(from_sodium + from_potassium / 2)
    # with hypertension the potassium effect stays; without it, it goes
    assert variant(potassium_effect_in_normotensives=False) == pytest.approx(fall)
    normotensive = intervention.sbp_reduction(
        less_sodium, more_potassium, 4.0, 60, False, PARAMETERS,
        potassium_effect_in_normotensives=False)
    assert normotensive == pytest.approx((3.735 + 10 * 0.105) * data_values.MMOL_PER_GRAM_SODIUM / 100)


def test_draw_zero_is_central_and_other_draws_are_reproducible():
    central = data_values.ModelParameters.for_draw(0)
    assert central == data_values.ModelParameters()
    assert central.tmrel == pytest.approx(112.5)
    first, again = data_values.ModelParameters.for_draw(7), data_values.ModelParameters.for_draw(7)
    assert first == again
    assert first != data_values.ModelParameters.for_draw(8)
    assert 110 <= first.tmrel <= 115
    assert all(slope >= 0 for slope in first.potassium_slopes)


def test_discretionary_share_falls_with_income_and_stays_in_bounds():
    share = data_values.discretionary_share
    assert share(0) == pytest.approx(0.633)
    assert share(10_000) == pytest.approx(0.633 - 0.087)
    assert share(200_000) == pytest.approx(data_values.DISCRETIONARY_SHARE_BOUNDS[0])
    incomes = np.linspace(0, 150_000, 40)
    assert np.all(np.diff(share(incomes)) <= 0)
