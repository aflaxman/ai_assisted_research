"""The closed-form version of the model, against numerical integration and common sense."""
import dataclasses

import numpy as np
import pytest
from scipy import integrate, stats

from vivarium_potassium_salt import analytic
from vivarium_potassium_salt.constants import data_values, metadata

LOCATION = "Testland"
ALL_SALT, DISCRETIONARY, NON_DISCRETIONARY, DIAGNOSED, TREATED = (
    scenario.name for scenario in metadata.SCENARIOS)


@pytest.fixture(scope="module")
def inputs():
    return analytic.load_inputs(LOCATION, n_sampled=0)


@pytest.fixture(scope="module")
def primary(inputs):
    return analytic.evaluate(inputs)


def averted(inputs, results, scenario, draw=0):
    """Deaths averted, all causes of the model together."""
    return float((results[scenario]["pif"] * inputs.counts["deaths"])[:, :, draw].sum())


####################################
# The integral behind everything   #
####################################


def integrated_relative_risk(shift, shape, scale, log_rr, tmrel, lower=0.0, upper=np.inf):
    """The same integral by quadrature, split at the kink in the relative risk."""
    density = stats.gamma(shape, scale=scale).pdf
    risk = lambda x: np.exp(log_rr * max(x - shift - tmrel, 0.0)) * density(x)  # noqa: E731
    top = min(upper, 400.0)
    kink = min(max(tmrel + shift, lower), top)
    breaks = sorted({lower, kink, top})
    total = sum(integrate.quad(risk, a, b, epsabs=1e-12)[0] for a, b in zip(breaks, breaks[1:]))
    if upper > 400:
        total += integrate.quad(risk, 400, 1500, epsabs=1e-12)[0]
    return total


@pytest.mark.parametrize("mean, sd, rr_per_10, shift", [
    (128.0, 17.0, 1.50, 0.0),
    (128.0, 17.0, 1.50, 4.2),
    (112.0, 13.0, 2.10, 1.0),     # most people at or below the TMREL
    (150.0, 24.0, 1.12, 9.0),
])
def test_mean_relative_risk_matches_numerical_integration(mean, sd, rr_per_10, shift):
    shape, scale, log_rr, tmrel = (mean / sd) ** 2, sd**2 / mean, np.log(rr_per_10) / 10, 112.5
    closed = analytic.mean_relative_risk(shift, shape, scale, log_rr, tmrel)
    assert closed == pytest.approx(
        integrated_relative_risk(shift, shape, scale, log_rr, tmrel), rel=1e-7)


@pytest.mark.parametrize("cut", [105.0, 118.0, 140.0])
def test_slices_of_the_distribution_add_up_to_the_whole(cut):
    mean, sd, shift, tmrel = 131.0, 19.0, 3.0, 112.5
    shape, scale, log_rr = (mean / sd) ** 2, sd**2 / mean, np.log(1.4) / 10
    below = analytic.mean_relative_risk(shift, shape, scale, log_rr, tmrel, upper=cut)
    above = analytic.mean_relative_risk(shift, shape, scale, log_rr, tmrel, lower=cut)
    whole = analytic.mean_relative_risk(shift, shape, scale, log_rr, tmrel)
    assert below + above == pytest.approx(whole, rel=1e-10)
    assert below == pytest.approx(
        integrated_relative_risk(shift, shape, scale, log_rr, tmrel, upper=cut), rel=1e-7)
    assert above == pytest.approx(
        integrated_relative_risk(shift, shape, scale, log_rr, tmrel, lower=cut), rel=1e-7)


def test_no_fall_in_pressure_means_no_change_in_risk():
    args = ((128 / 17) ** 2, 17**2 / 128, np.log(1.5) / 10, 112.5)
    assert analytic.mean_relative_risk(5.0, *args) < analytic.mean_relative_risk(0.0, *args)
    assert analytic.mean_relative_risk(500.0, *args) == pytest.approx(1.0)


###########################
# One synthetic country   #
###########################


def test_impact_fractions_are_proportions_and_scenarios_are_ordered(inputs, primary):
    for result in primary.values():
        assert (result["pif"] >= 0).all() and (result["pif"] < 1).all()
    everyone = primary[ALL_SALT]["pif"]
    for part in (DISCRETIONARY, NON_DISCRETIONARY):
        assert (primary[part]["pif"] <= everyone + 1e-12).all()
        assert averted(inputs, primary, part) < averted(inputs, primary, ALL_SALT)
    assert (primary[TREATED]["pif"] <= primary[DIAGNOSED]["pif"] + 1e-12).all()
    assert (primary[DIAGNOSED]["pif"] <= primary[DISCRETIONARY]["pif"] + 1e-12).all()
    # Sodium removed by the two halves adds up exactly; deaths averted add up
    # only roughly, because of the floor, the ceilings, and a convex risk curve.
    halves = averted(inputs, primary, DISCRETIONARY) + averted(inputs, primary, NON_DISCRETIONARY)
    assert halves == pytest.approx(averted(inputs, primary, ALL_SALT), rel=0.25)


def test_nothing_happens_where_sodium_is_already_at_the_floor(inputs, primary):
    at_floor = inputs.sodium[:, 0] <= data_values.SODIUM_FLOOR_G_PER_DAY
    assert at_floor.any() and not at_floor.all()        # the synthetic data has both kinds
    assert (primary[ALL_SALT]["pif"][:, at_floor, 0] == 0).all()
    assert (primary[ALL_SALT]["sbp_reduction"][at_floor, 0] == 0).all()
    assert (primary[ALL_SALT]["pif"][:, ~at_floor, 0] > 0).all()


def test_everyone_uses_the_substitute_unless_the_scenario_targets_hypertension(inputs, primary):
    people = inputs.strata["population"].to_numpy()
    for scenario in (ALL_SALT, DISCRETIONARY, NON_DISCRETIONARY):
        assert primary[scenario]["users"][:, 0] == pytest.approx(people)
    hypertensive = people * inputs.hypertension[:, 0]
    assert primary[DIAGNOSED]["users"][:, 0] == pytest.approx(
        0.8 * hypertensive * inputs.diagnosed[:, 0])
    assert primary[TREATED]["users"][:, 0] == pytest.approx(
        0.8 * hypertensive * inputs.treated[:, 0])


def test_a_bigger_relative_risk_means_a_bigger_impact_fraction(inputs, primary):
    # Same fall in pressure for every cause, so the ordering is the relative risk's.
    pif, risk = primary[ALL_SALT]["pif"][:, :, 0], inputs.relative_risk[:, :, 0]
    stratum = int(np.argmax(inputs.sodium[:, 0]))
    order = np.argsort(risk[:, stratum])
    assert (np.diff(pif[order, stratum]) >= -1e-12).all()


@pytest.mark.parametrize("change, quantity", [
    (dict(replacement_proportion=0.5), ALL_SALT),
    (dict(replacement_proportion=0.2), ALL_SALT),
    (dict(combined_effect_scale=0.8), ALL_SALT),
    (dict(potassium_effect_scale=0.5), ALL_SALT),
    (dict(potassium_effect_in_normotensives=False), ALL_SALT),
    (dict(kcl_fraction=0.15), ALL_SALT),
    (dict(sodium_floor=2.5), ALL_SALT),
    (dict(sodium_floor=3.0), ALL_SALT),
    (dict(targeted_uptake_lmic=0.7), DIAGNOSED),
    (dict(targeted_uptake_lmic=0.7), TREATED),
])
def test_each_conservative_sensitivity_analysis_averts_fewer_deaths(inputs, primary, change, quantity):
    variant = analytic.evaluate(inputs, analytic.Settings(**change))
    assert 0 < averted(inputs, variant, quantity) < averted(inputs, primary, quantity)


def test_halving_replacement_roughly_halves_the_fall_in_pressure(inputs, primary):
    half = analytic.evaluate(inputs, analytic.Settings(replacement_proportion=0.5))
    full_fall, half_fall = (r[ALL_SALT]["sbp_reduction"][:, 0] for r in (primary, half))
    moved = full_fall > 0
    ratio = half_fall[moved] / full_fall[moved]
    # Exactly half where neither the sodium floor nor a potassium ceiling binds;
    # more than half where one of them was holding the full effect back.
    assert ratio.min() == pytest.approx(0.5, abs=1e-9)
    assert (ratio <= 1 + 1e-12).all()


def test_income_based_uptake_only_changes_poorer_countries():
    settings = analytic.Settings(targeted_uptake_lmic=0.7)
    for location, changes in (("Testland", True), ("Richland", False)):
        inputs = analytic.load_inputs(location)
        assert inputs.high_income is not changes
        base = analytic.evaluate(inputs)[DIAGNOSED]["users"].sum()
        variant = analytic.evaluate(inputs, settings)[DIAGNOSED]["users"].sum()
        assert variant == pytest.approx(base * (0.7 / 0.8 if changes else 1.0))


def test_discretionary_share_comes_from_income_or_from_the_survey(inputs):
    by_income = analytic.discretionary_share(inputs, analytic.Settings())
    assert by_income[0] == pytest.approx((63.3 - 0.87 * 4_000 / 1.24 / 1000) / 100)
    surveyed = analytic.discretionary_share(
        inputs, analytic.Settings(discretionary_share_source="reported_where_available"))
    assert surveyed[0] == pytest.approx(0.76)
    rich = analytic.load_inputs("Richland")
    assert np.isnan(rich.reported_discretionary_share)       # no survey: falls back on income
    settings = analytic.Settings(discretionary_share_source="reported_where_available")
    assert analytic.discretionary_share(rich, settings)[0] == pytest.approx(
        analytic.discretionary_share(rich, analytic.Settings())[0])
    assert analytic.discretionary_share(rich, analytic.Settings())[0] == pytest.approx(
        (63.3 - 0.87 * 60_000 / 1.24 / 1000) / 100)


def test_ranking_hypertension_by_pressure_raises_the_targeted_scenarios(inputs, primary):
    ranked = analytic.evaluate(
        inputs, analytic.Settings(association_with_blood_pressure="ranked"))
    for scenario in (DIAGNOSED, TREATED):
        assert averted(inputs, ranked, scenario) > averted(inputs, primary, scenario)
        assert ranked[scenario]["users"] == pytest.approx(primary[scenario]["users"])
    # With everyone covered, only who gets the larger sodium effect changes.
    assert averted(inputs, ranked, ALL_SALT) == pytest.approx(
        averted(inputs, primary, ALL_SALT), rel=0.15)
    assert averted(inputs, ranked, ALL_SALT) > averted(inputs, primary, ALL_SALT)


def test_moving_a_whole_stratum_by_its_mean_fall_overstates_the_targeted_scenarios(inputs, primary):
    """Jensen's inequality: the relative risk is convex in the fall in pressure."""
    lumped = analytic.evaluate(inputs, analytic.Settings(blood_pressure_response="stratum_mean"))
    for scenario in (ALL_SALT, DIAGNOSED, TREATED):
        assert (lumped[scenario]["pif"] >= primary[scenario]["pif"] - 1e-12).all()
        assert lumped[scenario]["sbp_reduction"] == pytest.approx(primary[scenario]["sbp_reduction"])
    # When everyone switches the two falls are close, and so are the answers.
    assert averted(inputs, lumped, ALL_SALT) == pytest.approx(
        averted(inputs, primary, ALL_SALT), rel=0.005)
    assert averted(inputs, lumped, DIAGNOSED) > 1.02 * averted(inputs, primary, DIAGNOSED)
    with pytest.raises(ValueError):
        analytic.evaluate(inputs, analytic.Settings(blood_pressure_response="nonsense"))


#########################
# Kidney disease        #
#########################


def test_hyperkalaemia_is_counted_only_where_the_whole_population_switches(inputs, primary):
    for scenario in metadata.SCENARIOS:
        extra = primary[scenario.name]["hyperkalaemia_deaths"]
        if scenario.hyperkalaemia_modelled:
            assert (extra >= 0).all() and extra.sum() > 0
        else:
            assert (extra == 0).all()
    # More potassium, more hyperkalaemia.
    assert (primary[ALL_SALT]["hyperkalaemia_deaths"].sum()
            > primary[DISCRETIONARY]["hyperkalaemia_deaths"].sum())


def test_deaths_prevented_in_kidney_disease_are_a_share_of_all_deaths_prevented(inputs, primary):
    cvd = [cause in metadata.CVD_CAUSES for cause in metadata.CAUSES]
    all_prevented = (primary[ALL_SALT]["pif"] * inputs.counts["deaths"])[cvd].sum(axis=0)[:, 0]
    in_ckd = primary[ALL_SALT]["prevented_cvd_deaths_in_ckd"][:, 0]
    moved = all_prevented > 0
    share = in_ckd[moved] / all_prevented[moved]
    prevalence = sum(inputs.ckd_prevalence.values())[moved, 0]
    assert (share > prevalence).all()      # hazard ratio above one
    assert (share < 1).all()


def test_dividing_only_kidney_disease_deaths_among_stages_gives_fewer_hyperkalaemia_deaths(
        inputs, primary):
    assert analytic.Settings().stage_deaths == "among_ckd"
    literal = analytic.evaluate(inputs, analytic.Settings(stage_deaths="all_adults"))
    assert 0 < primary[ALL_SALT]["hyperkalaemia_deaths"].sum() < \
        literal[ALL_SALT]["hyperkalaemia_deaths"].sum()
    assert literal[ALL_SALT]["pif"] == pytest.approx(primary[ALL_SALT]["pif"])


def test_stage_shares_under_both_readings():
    prevalence = {"G3a": 0.08, "G3b": 0.02, "G4": 0.004, "G5": 0.001}
    hazard = {stage: value[0] for stage, value in data_values.CKD_STAGE_HAZARD_RATIO.items()}
    weighted = sum(prevalence[s] * hazard[s] for s in prevalence)
    literal = analytic.stage_shares_of_cvd_deaths(prevalence, hazard, None, "all_adults")
    assert sum(literal.values()) == pytest.approx(weighted / (1 - 0.105 + weighted))
    within = analytic.stage_shares_of_cvd_deaths(prevalence, hazard, 0.2, "among_ckd")
    assert sum(within.values()) == pytest.approx(0.2)
    assert within["G5"] / within["G3a"] == pytest.approx(literal["G5"] / literal["G3a"])


#########################
# Draws                 #
#########################


def test_draw_zero_is_the_mean_whatever_the_number_of_draws(inputs, primary):
    many = analytic.load_inputs(LOCATION, n_sampled=6)
    assert many.n_draws == 7
    results = analytic.evaluate(many)
    assert results[ALL_SALT]["pif"][:, :, 0] == pytest.approx(primary[ALL_SALT]["pif"][:, :, 0])
    assert results[ALL_SALT]["pif"][:, :, 1:].std() > 0
    fewer = analytic.evaluate(analytic.load_inputs(LOCATION, n_sampled=3))
    assert fewer[ALL_SALT]["pif"] == pytest.approx(results[ALL_SALT]["pif"][:, :, :4])


def test_parameters_drawn_once_per_draw_give_a_wider_interval_than_one_per_stratum():
    """The finding in the README, in miniature.

    A dose-response coefficient is one unknown number.  Drawing a fresh value
    for every country, sex, and age band lets the errors average out across
    strata, and the total looks more certain than it is.
    """
    spread = {}
    for regime in ("shared", "independent"):
        totals = sum(
            analytic.run_location(location, analytic.Settings(parameter_draws=regime), 150)
            .xs((ALL_SALT, "deaths_averted__cvd"), level=("scenario", "quantity")).sum()
            for location in ("Testland", "Richland"))
        sampled = totals.iloc[1:]
        spread[regime] = sampled.std() / sampled.mean()
        assert totals.iloc[0] == pytest.approx(sampled.mean(), rel=0.15)
    assert spread["shared"] > 2 * spread["independent"]


def test_summary_has_a_row_per_scenario_quantity_and_sex(inputs, primary):
    summary = analytic.summarise(inputs, primary)
    assert list(summary.index.names) == ["location", "scenario", "quantity", "sex"]
    assert list(summary.columns) == ["draw_0"]
    assert set(summary.index.get_level_values("scenario")) == {"baseline"} | set(primary)
    deaths = summary.xs(("baseline", "deaths__cvd"), level=("scenario", "quantity"))["draw_0"]
    cvd = [cause in metadata.CVD_CAUSES for cause in metadata.CAUSES]
    assert deaths.sum() == pytest.approx(inputs.counts["deaths"][cvd].sum())
    total = summary.xs((ALL_SALT, "deaths_averted__cvd"), level=("scenario", "quantity"))
    kidney = summary.xs((ALL_SALT, "deaths_averted__ckd"), level=("scenario", "quantity"))
    assert total["draw_0"].sum() + kidney["draw_0"].sum() == pytest.approx(
        averted(inputs, primary, ALL_SALT))


def test_stratum_table_is_tidy(inputs, primary):
    table = analytic.stratum_table(inputs, primary)
    assert len(table) == len(metadata.SCENARIOS) * len(metadata.CAUSES) * len(inputs.strata)
    assert (table["averted"] == table["baseline"] * table["pif"]).all()


def test_settings_are_immutable_and_named_like_the_component_configuration():
    settings = analytic.Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.kcl_fraction = 0.5
    from vivarium_potassium_salt.components import PotassiumEnrichedSalt
    configured = PotassiumEnrichedSalt().configuration_defaults["potassium_enriched_salt"]
    shared = set(configured) & {f.name for f in dataclasses.fields(settings)}
    assert {"kcl_fraction", "replacement_proportion", "sodium_floor", "targeted_uptake",
            "potassium_effect_scale", "combined_effect_scale", "sodium_dose_response"} <= shared
    for name in shared:
        assert configured[name] == getattr(settings, name), name
