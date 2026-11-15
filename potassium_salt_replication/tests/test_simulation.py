"""The microsimulation, against its own inputs and against the closed form.

The closed form and the simulation share the dose-response arithmetic and
nothing else.  The simulation gets its blood pressure from the stock ``Risk``
component, its risks from stock ``RiskEffect`` components, and its answer from
observers; the closed form gets all three from a formula.  If they agree, the
components are wired as intended.
"""
import numpy as np
import pandas as pd
import pytest
from vivarium.artifact import Artifact
from vivarium.fuzzy_checker import FuzzyChecker

from vivarium_potassium_salt import analytic
from vivarium_potassium_salt.components import burden, hypertension, intervention, kidney, observers
from vivarium_potassium_salt.constants import metadata
from vivarium_potassium_salt.data import builder
from vivarium_potassium_salt.tools import analysis, run_simulation

LOCATION = "Testland"
ALL_SALT, DIAGNOSED = "all_salt", "discretionary_diagnosed_hypertension"
STRATUM = ["sex", "age_group"]
PER_STRATUM = 500          # simulants per sex and age band at the default population size


@pytest.fixture(scope="session")
def fuzzy_checker():
    return FuzzyChecker()


@pytest.fixture(scope="module")
def artifact():
    return builder.build_artifact(LOCATION)


@pytest.fixture(scope="module")
def inputs():
    return analytic.load_inputs(LOCATION)


@pytest.fixture(scope="module")
def simulated(artifact):
    """The baseline and two scenarios with the same seed, as the observers recorded them."""
    return run_simulation.run_location(LOCATION, scenarios=["baseline", ALL_SALT, DIAGNOSED])


@pytest.fixture(scope="module")
def totals(simulated):
    _, totals = analysis.summarise_simulation(
        simulated[observers.BURDEN_RESULTS], simulated[observers.POLICY_RESULTS])
    return totals.groupby(["scenario", "quantity"])["value"].sum()


@pytest.fixture(scope="module")
def closed_form(inputs):
    summary = analytic.summarise(inputs, analytic.evaluate(inputs))
    return summary["draw_0"].groupby(["scenario", "quantity"]).sum()


def by_stratum(table, scenario, extra=()):
    table = table[table["scenario"] == scenario]
    table = table.assign(age_group=table["age_group"].astype(str).str.replace("_", " "),
                         sex=table["sex"].astype(str))
    return table.groupby(STRATUM + list(extra), observed=True).sum(numeric_only=True)


def test_artifact_holds_every_key_the_model_reads(artifact):
    keys = set(Artifact(artifact).keys)
    assert set(builder.STANDARD_KEYS) <= keys
    assert set(builder.burden_keys()) <= keys
    assert len(builder.burden_keys()) == len(metadata.CAUSES)


def test_simulants_are_spread_evenly_over_sex_and_age(simulated):
    people = by_stratum(simulated[observers.POLICY_RESULTS], "baseline")[observers.PERSON_COUNT]
    assert len(people) == len(metadata.SEXES) * len(metadata.AGE_BANDS)
    assert (people == PER_STRATUM).all()


def test_scenarios_share_their_simulants(simulated):
    """Common random numbers: only the intervention differs between scenarios."""
    policy = simulated[observers.POLICY_RESULTS]
    keys = STRATUM + [hypertension.STATUS]
    baseline = by_stratum(policy, "baseline", [hypertension.STATUS])
    for scenario in (ALL_SALT, DIAGNOSED):
        other = by_stratum(policy, scenario, [hypertension.STATUS])
        pd.testing.assert_series_equal(other[observers.PERSON_COUNT],
                                       baseline[observers.PERSON_COUNT])
        # Sodium intake is the baseline's; blood pressure is lower by the recorded fall.
        pd.testing.assert_series_equal(other["diet_high_in_sodium.exposure"],
                                       baseline["diet_high_in_sodium.exposure"])
        fall = baseline[intervention.SBP_EXPOSURE] - other[intervention.SBP_EXPOSURE]
        assert fall.to_numpy() == pytest.approx(other[intervention.SBP_REDUCTION].to_numpy(),
                                                abs=1e-6)
    assert (baseline[intervention.SBP_REDUCTION] == 0).all()
    assert (baseline[intervention.USES_SUBSTITUTE] == 0).all()
    assert keys == list(baseline.index.names)


def test_baseline_reproduces_the_death_rates_it_was_given(simulated, inputs):
    """Rate = GBD rate x (1 - PAF) x RR, so its mean over simulants is the GBD rate again."""
    sums = by_stratum(simulated[observers.BURDEN_RESULTS], "baseline")
    order = pd.MultiIndex.from_frame(inputs.strata[STRATUM])
    people = inputs.strata["population"].to_numpy()
    for c, cause in enumerate(metadata.CAUSES):
        column = burden.rate_pipeline(cause, "deaths")
        rate = (sums[column] / sums[observers.PERSON_COUNT]).reindex(order).to_numpy()
        expected = inputs.counts["deaths"][c, :, 0]
        assert (rate * people).sum() == pytest.approx(expected.sum(), rel=0.03), cause


def test_everyone_switches_in_the_population_wide_scenario(simulated):
    policy = by_stratum(simulated[observers.POLICY_RESULTS], ALL_SALT)
    assert (policy[intervention.USES_SUBSTITUTE] == policy[observers.PERSON_COUNT]).all()
    ratio = policy[intervention.POTASSIUM_INCREASE] / policy[intervention.SODIUM_REDUCTION]
    assert ratio.dropna().to_numpy() == pytest.approx(1.333, abs=1e-3)


def test_only_people_with_diagnosed_hypertension_switch_in_the_targeted_scenario(simulated):
    policy = by_stratum(simulated[observers.POLICY_RESULTS], DIAGNOSED, [hypertension.STATUS])
    users = policy[intervention.USES_SUBSTITUTE].groupby(hypertension.STATUS).sum()
    assert users[metadata.NORMOTENSIVE] == 0
    assert users[metadata.UNDIAGNOSED] == 0
    assert users[metadata.DIAGNOSED_UNTREATED] > 0
    assert users[metadata.TREATED] > 0


@pytest.mark.parametrize("sex", metadata.SEXES)
def test_proportions_of_simulants_match_their_targets(simulated, inputs, fuzzy_checker, sex):
    """Hypertension, uptake, and kidney disease, each as a fuzzy check on a proportion.

    Simulants are allocated equally to age bands, so the target for a sex is
    the plain average of its age bands' targets.
    """
    rows = (inputs.strata["sex"] == sex).to_numpy()
    policy = simulated[observers.POLICY_RESULTS]
    statuses = by_stratum(policy, "baseline", [hypertension.STATUS]).xs(sex, level="sex")
    people = int(statuses[observers.PERSON_COUNT].sum())
    hypertensive = int(statuses[observers.PERSON_COUNT].drop(
        metadata.NORMOTENSIVE, level=hypertension.STATUS).sum())
    fuzzy_checker.assert_proportion(
        hypertensive, people, float(inputs.hypertension[rows, 0].mean()),
        name="hypertension_prevalence", name_additional=sex)

    users = int(by_stratum(policy, DIAGNOSED).xs(sex, level="sex")[
        intervention.USES_SUBSTITUTE].sum())
    target = 0.8 * (inputs.hypertension[rows, 0] * inputs.diagnosed[rows, 0]).mean()
    fuzzy_checker.assert_proportion(users, people, float(target),
                                    name="targeted_uptake", name_additional=sex)

    stages = by_stratum(simulated[observers.BURDEN_RESULTS], "baseline", [kidney.STAGE]).xs(
        sex, level="sex")[observers.PERSON_COUNT].groupby(kidney.STAGE).sum()
    for stage in metadata.CKD_STAGES:
        fuzzy_checker.assert_proportion(
            int(stages[stage]), people, float(inputs.ckd_prevalence[stage][rows, 0].mean()),
            name=f"ckd_stage_{stage}_prevalence", name_additional=sex)


@pytest.mark.parametrize("scenario, quantity, tolerance", [
    (ALL_SALT, "deaths_averted__cvd", 0.01),
    (ALL_SALT, "deaths_averted__ckd", 0.01),
    # These two rest on the few hundred simulants with advanced kidney disease;
    # the slow test below repeats them with ten times the simulants.
    (ALL_SALT, "prevented_cvd_deaths_in_ckd", 0.06),
    (ALL_SALT, "hyperkalaemia_deaths", 0.15),
    (ALL_SALT, "users", 1e-9),
    # Who takes up the salt is random here, and 500 simulants a stratum is few.
    (DIAGNOSED, "users", 0.05),
    (DIAGNOSED, "deaths_averted__cvd", 0.08),
    (DIAGNOSED, "deaths_averted__ckd", 0.08),
])
def test_simulation_matches_the_closed_form(totals, closed_form, scenario, quantity, tolerance):
    assert totals[(scenario, quantity)] == pytest.approx(
        closed_form[(scenario, quantity)], rel=tolerance)


def test_impact_fraction_matches_the_closed_form_stratum_by_stratum(simulated, inputs):
    results = analysis.simulation_impact(
        simulated[observers.BURDEN_RESULTS], simulated[observers.POLICY_RESULTS], inputs)
    exact = analytic.evaluate(inputs)
    heart = metadata.CAUSES.index("ischemic_heart_disease")
    assert results[ALL_SALT]["pif"][heart, :, 0] == pytest.approx(
        exact[ALL_SALT]["pif"][heart, :, 0], abs=0.01)
    assert results[ALL_SALT]["sbp_reduction"][:, 0] == pytest.approx(
        exact[ALL_SALT]["sbp_reduction"][:, 0], abs=0.15)


def test_a_stratum_with_no_deaths_averts_none(simulated, inputs):
    """Sampled death rates can be zero, and zero over zero must not become a missing value."""
    burden_results = simulated[observers.BURDEN_RESULTS].copy()
    column = burden.rate_pipeline("endocarditis", "deaths")
    youngest = burden_results["age_group"].astype(str) == "25_to_29"
    burden_results.loc[youngest, column] = 0.0
    results = analysis.simulation_impact(
        burden_results, simulated[observers.POLICY_RESULTS], inputs)
    cause = metadata.CAUSES.index("endocarditis")
    young = (inputs.strata["age_group"] == "25 to 29").to_numpy()
    assert (results[ALL_SALT]["pif"][cause, young, 0] == 0).all()
    for value in results[ALL_SALT].values():
        assert np.isfinite(value).all()


def test_blood_pressure_pipeline_falls_by_each_simulants_own_response(artifact):
    """Look inside two simulations that share a seed, simulant by simulant."""
    columns = [intervention.SBP_EXPOSURE, intervention.SBP_REDUCTION,
               intervention.SODIUM_REDUCTION, "diet_high_in_sodium.exposure", "age"]
    before = run_simulation.build_simulation(LOCATION, "baseline", population_size=3_000)
    after = run_simulation.build_simulation(LOCATION, ALL_SALT, population_size=3_000)
    baseline, scenario = before.get_population(columns), after.get_population(columns)
    assert (baseline["age"] == scenario["age"]).all()
    assert (baseline[intervention.SBP_REDUCTION] == 0).all()
    fall = baseline[intervention.SBP_EXPOSURE] - scenario[intervention.SBP_EXPOSURE]
    assert fall.to_numpy() == pytest.approx(scenario[intervention.SBP_REDUCTION].to_numpy())
    # Nobody is taken below 2 g/day of sodium, and nobody below it is touched.
    intake = scenario["diet_high_in_sodium.exposure"]
    assert ((intake - scenario[intervention.SODIUM_REDUCTION] >= 2.0 - 1e-9) | (intake <= 2.0)).all()
    assert (scenario.loc[intake <= 2.0, intervention.SBP_REDUCTION] == 0).all()
    assert (scenario.loc[intake > 2.0, intervention.SBP_REDUCTION] > 0).all()


@pytest.mark.slow
def test_kidney_disease_quantities_converge_with_more_simulants(artifact, closed_form):
    results = run_simulation.run_location(LOCATION, scenarios=["baseline", ALL_SALT],
                                          population_size=150_000)
    _, totals = analysis.summarise_simulation(
        results[observers.BURDEN_RESULTS], results[observers.POLICY_RESULTS])
    simulated = totals.groupby(["scenario", "quantity"])["value"].sum()
    for quantity in ("hyperkalaemia_deaths", "prevented_cvd_deaths_in_ckd"):
        assert simulated[(ALL_SALT, quantity)] == pytest.approx(
            closed_form[(ALL_SALT, quantity)], rel=0.04)


@pytest.mark.slow
def test_configuration_overrides_reach_the_components(artifact, inputs):
    """Sensitivity analysis 7 (a 15% potassium chloride blend) through the simulation."""
    configuration = {"potassium_enriched_salt": {"kcl_fraction": 0.15}}
    results = run_simulation.run_location(LOCATION, scenarios=["baseline", ALL_SALT],
                                          configuration=configuration)
    _, totals = analysis.summarise_simulation(
        results[observers.BURDEN_RESULTS], results[observers.POLICY_RESULTS])
    simulated = totals.groupby(["scenario", "quantity"])["value"].sum()
    exact = analytic.summarise(
        inputs, analytic.evaluate(inputs, analytic.Settings(kcl_fraction=0.15))
    )["draw_0"].groupby(["scenario", "quantity"]).sum()
    primary = analytic.summarise(inputs, analytic.evaluate(inputs))["draw_0"].groupby(
        ["scenario", "quantity"]).sum()
    key = (ALL_SALT, "deaths_averted__cvd")
    assert simulated[key] == pytest.approx(exact[key], rel=0.01)
    assert simulated[key] < 0.8 * primary[key]


@pytest.mark.slow
def test_a_sampled_draw_and_another_seed_also_match(artifact):
    """Draw 3 of the inputs, with different simulants."""
    results = run_simulation.run_location(LOCATION, draws=[3], seed=11,
                                          scenarios=["baseline", ALL_SALT])
    _, totals = analysis.summarise_simulation(
        results[observers.BURDEN_RESULTS], results[observers.POLICY_RESULTS])
    simulated = totals.groupby(["scenario", "quantity"])["value"].sum()
    inputs = analytic.load_inputs(LOCATION, n_sampled=3)
    exact = analytic.summarise(inputs, analytic.evaluate(inputs))["draw_3"].groupby(
        ["scenario", "quantity"]).sum()
    for quantity in ("deaths_averted__cvd", "deaths_averted__ckd"):
        assert simulated[(ALL_SALT, quantity)] == pytest.approx(
            exact[(ALL_SALT, quantity)], rel=0.015)
    mean = analytic.summarise(analytic.load_inputs(LOCATION), analytic.evaluate(
        analytic.load_inputs(LOCATION)))["draw_0"].groupby(["scenario", "quantity"]).sum()
    assert exact[(ALL_SALT, "deaths_averted__cvd")] != pytest.approx(
        mean[(ALL_SALT, "deaths_averted__cvd")], rel=1e-3)

    # This draw's TMREL is 115.0 mmHg, not the midpoint of 112.5.  The risk
    # effect and the artifact's attributable fractions must both use it, or
    # the baseline would not return the death rates it was given.
    sums = by_stratum(results[observers.BURDEN_RESULTS], "baseline")
    order = pd.MultiIndex.from_frame(inputs.strata[STRATUM])
    people = inputs.strata["population"].to_numpy()
    heart = metadata.CAUSES.index("ischemic_heart_disease")
    column = burden.rate_pipeline("ischemic_heart_disease", "deaths")
    rate = (sums[column] / sums[observers.PERSON_COUNT]).reindex(order).to_numpy()
    assert (rate * people).sum() == pytest.approx(
        inputs.counts["deaths"][heart, :, 3].sum(), rel=0.03)
