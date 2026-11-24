"""The stand-in for vivarium_inputs, and the cached extract behind it."""
import numpy as np
import pandas as pd
import pytest
from vivarium.gbd_mapping import causes, covariates, risk_factors

from vivarium_potassium_salt.constants import data_keys, data_values, metadata
from vivarium_potassium_salt.data import gbd_extract, literature, loader
from vivarium_potassium_salt.data import vivarium_inputs_stub as interface

LOCATION = "Testland"
DEMOGRAPHY = ["location", "sex", "age_start", "age_end", "year_start", "year_end"]
N_STRATA = len(metadata.SEXES) * len(metadata.AGE_BANDS)
SBP = risk_factors.high_systolic_blood_pressure


@pytest.mark.parametrize("entity, measure, extra_levels, rows", [
    (causes.ischemic_heart_disease, "cause_specific_mortality_rate", [], N_STRATA),
    (causes.chronic_kidney_disease, "cause_specific_mortality_rate", [], N_STRATA),
    (SBP, "exposure", ["parameter"], N_STRATA),
    (SBP, "exposure_standard_deviation", [], N_STRATA),
    (SBP, "relative_risk", ["affected_entity", "parameter"], N_STRATA * len(metadata.CAUSES)),
    (risk_factors.diet_high_in_sodium, "exposure", ["parameter"], N_STRATA),
    (risk_factors.kidney_dysfunction, "exposure", ["parameter"], N_STRATA * 4),
])
def test_get_measure_returns_the_vivarium_inputs_layout(entity, measure, extra_levels, rows):
    means = interface.get_measure(entity, measure, LOCATION, data_type="means")
    draws = interface.get_measure(entity, measure, LOCATION)
    assert list(means.index.names) == DEMOGRAPHY + extra_levels
    assert list(means.columns) == ["value"]
    assert list(draws.columns) == [f"draw_{i}" for i in range(interface.NUM_DRAWS)]
    assert len(means) == len(draws) == rows
    assert means.index.equals(draws.index)
    assert means.index.is_unique
    assert np.isfinite(draws.to_numpy()).all()


def test_pseudo_draws_centre_on_the_mean_and_fill_the_interval():
    entity = causes.ischemic_heart_disease
    means = interface.get_measure(entity, "cause_specific_mortality_rate", LOCATION,
                                  data_type="means")["value"]
    draws = interface.get_measure(entity, "cause_specific_mortality_rate", LOCATION)
    # The synthetic interval is mean * (0.85, 1.15), so sd = 0.15 / 1.96 of the mean.
    relative = draws.div(means, axis=0)
    assert relative.mean(axis=1).to_numpy() == pytest.approx(1.0, abs=0.012)
    assert relative.std(axis=1).to_numpy() == pytest.approx(0.15 / 1.96, rel=0.12)


def test_pseudo_draws_are_reproducible_and_differ_between_strata():
    first = interface.get_measure(SBP, "exposure", LOCATION)
    again = interface.get_measure(SBP, "exposure", LOCATION)
    pd.testing.assert_frame_equal(first, again)
    correlation = np.corrcoef(first.to_numpy())
    off_diagonal = correlation[~np.eye(len(correlation), dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.15     # independent across strata


def test_relative_risk_draws_do_not_depend_on_the_location():
    here = interface.get_measure(SBP, "relative_risk", "Testland").droplevel("location")
    there = interface.get_measure(SBP, "relative_risk", "Richland").droplevel("location")
    pd.testing.assert_frame_equal(here, there)
    assert (here.to_numpy() > 0).all()


def test_blood_pressure_is_an_ensemble_with_all_its_weight_on_the_gamma():
    weights = interface.get_measure(SBP, "exposure_distribution_weights", LOCATION)["value"]
    by_member = weights.groupby("parameter").mean()
    assert by_member["gamma"] == 1.0
    assert by_member.drop("gamma").eq(0.0).all()
    assert set(by_member.index) == set(data_values.ENSEMBLE_DISTRIBUTIONS)


def test_kidney_dysfunction_categories_partition_the_population():
    kidney = interface.get_measure(risk_factors.kidney_dysfunction, "exposure", LOCATION,
                                   data_type="means")["value"]
    total = kidney.groupby(["sex", "age_start"]).sum()
    assert total.to_numpy() == pytest.approx(1.0)
    assert (kidney >= 0).all()
    assert set(kidney.index.get_level_values("parameter")) == {"cat1", "cat2", "cat3", "cat5"}


def test_requests_outside_the_extract_fail_loudly():
    with pytest.raises(interface.DataNotInExtractError):
        interface.get_measure(causes.ischemic_heart_disease, "prevalence", LOCATION)
    with pytest.raises(interface.DataNotInExtractError):
        interface.get_measure(causes.ischemic_heart_disease, "cause_specific_mortality_rate",
                              LOCATION, years=1990)
    with pytest.raises(interface.DataNotInExtractError):
        interface.get_measure(risk_factors.high_body_mass_index_in_adults, "exposure", LOCATION)
    with pytest.raises(KeyError, match="not a GBD location name"):
        interface.get_measure(SBP, "exposure", "Atlantis")


def test_population_and_age_bins():
    people = interface.get_population_structure(LOCATION)
    assert list(people.index.names) == DEMOGRAPHY
    assert len(people) == N_STRATA and (people["value"] > 0).all()
    bins = interface.get_age_bins()
    assert list(bins["age_start"]) == [band.start for band in metadata.AGE_BANDS]
    assert list(bins["age_group_name"]) == list(metadata.AGE_BAND_NAMES)
    covariate = interface.get_measure(covariates.gdp_per_capita_base_2010, "estimate", LOCATION)
    assert covariate["mean_value"].iloc[0] == pytest.approx(4_000 / 1.24)


#########################################
# Disaggregating totals for ages 25+    #
#########################################


def test_every_cause_has_deaths_for_every_stratum():
    deaths = gbd_extract.burden_counts("deaths")
    assert set(deaths["cause"]) == set(metadata.CAUSES)
    assert (deaths.groupby(["location_id", "cause"]).size() == N_STRATA).all()
    assert (deaths["count"] >= 0).all()


def test_disaggregated_deaths_keep_the_country_total_and_take_the_regional_age_pattern():
    deaths = gbd_extract.burden_counts("deaths")
    split = deaths[deaths["source"] == "disaggregated"]
    assert set(split["cause"]) == {"rheumatic_heart_disease", "aortic_aneurysm"}
    totals = pd.read_csv(gbd_extract.paths.GBD_EXTRACT_DIR / "burden_25_plus.csv")
    totals["cause"] = totals["cause"].map(gbd_extract.MODEL_CAUSE)
    expected = totals.set_index(["location_id", "cause", "sex"])["count"]
    observed = split.groupby(["location_id", "cause", "sex"])["count"].sum()
    pd.testing.assert_series_equal(observed, expected.reindex(observed.index), rtol=1e-9)

    # Shares within a country and sex follow regional rate times population.
    people = gbd_extract.population().set_index(["location_id", "age_group", "sex"])["population"]
    one = split[(split["cause"] == "rheumatic_heart_disease") & (split["sex"] == "Female")
                & (split["location_id"] == 9001)].set_index("age_group")["count"]
    age = np.arange(len(metadata.AGE_BAND_NAMES))
    weight = np.exp(0.25 * (age - 7)) * people.xs((9001, "Female"), level=(0, 2)).reindex(
        metadata.AGE_BAND_NAMES).to_numpy()
    assert (one / one.sum()).reindex(metadata.AGE_BAND_NAMES).to_numpy() == pytest.approx(
        weight / weight.sum())


def test_measures_that_were_not_pulled_are_reported_as_unavailable():
    assert gbd_extract.available_measures() == ["deaths"]
    with pytest.raises(KeyError):
        gbd_extract.burden_counts("dalys")


#########################
# Non-GBD inputs        #
#########################


def test_relative_risks_cover_every_cause_and_age_and_fall_with_age():
    risks = literature.sbp_relative_risks()
    assert len(risks) == len(metadata.CAUSES) * len(metadata.AGE_BANDS)
    assert (risks["lower"] <= risks["mean"]).all() and (risks["mean"] <= risks["upper"]).all()
    heart = risks[risks["cause"] == "ischemic_heart_disease"].set_index("age_group")["mean"]
    assert heart["25 to 29"] > heart["55 to 59"] > heart["80 to 84"] > 1
    assert heart["80 to 84"] == heart["95 plus"]     # last published value carried forward


def test_hypertension_borrows_the_nearest_age_band_at_both_ends():
    table = literature.hypertension(LOCATION).set_index(["sex", "age_group"])
    assert len(table) == N_STRATA
    for column in ("prevalence", "diagnosed", "treated"):
        assert table.loc[("Male", "25 to 29"), column] == table.loc[("Male", "30 to 34"), column]
        assert table.loc[("Male", "95 plus"), column] == table.loc[("Male", "75 to 79"), column]
    assert (table["treated"] <= table["diagnosed"]).all()


def test_hyperkalaemia_risk_is_one_below_the_threshold_and_matches_the_printed_value():
    risk = literature.hyperkalaemia_relative_risk
    assert risk(5.49) == 1.0
    assert risk(4.2) == 1.0
    # The one value in print: 1.22 (1.04 to 1.42) at 5.5 mmol/L.
    assert float(risk(5.5)) == pytest.approx(1.22, abs=0.02)
    assert float(risk(5.5, -1.96)) == pytest.approx(1.04, abs=0.03)
    assert float(risk(5.5, 1.96)) == pytest.approx(1.42, abs=0.03)
    assert float(risk(9.0)) == float(risk(6.5))       # held at its last value


def test_reported_discretionary_share_is_read_where_there_is_a_survey():
    shares = literature.reported_discretionary_shares()
    assert shares == {"Testland": pytest.approx(0.76)}


#########################
# Artifact loader       #
#########################


def test_loader_puts_the_mean_in_draw_zero():
    key = data_keys.burden_rate("ischemic_stroke", "deaths")
    table = loader.get_data(key, LOCATION)
    assert list(table.columns) == [f"draw_{i}" for i in range(loader.N_SAMPLED_DRAWS + 1)]
    means = interface.get_measure(causes.ischemic_stroke, "cause_specific_mortality_rate",
                                  LOCATION, data_type="means").droplevel("location")
    assert table["draw_0"].to_numpy() == pytest.approx(means["value"].to_numpy())
    draws = interface.get_measure(causes.ischemic_stroke, "cause_specific_mortality_rate",
                                  LOCATION).droplevel("location")
    assert table["draw_3"].to_numpy() == pytest.approx(draws["draw_2"].to_numpy())


def test_attributable_fraction_is_consistent_with_exposure_and_relative_risk():
    """PAF = 1 - 1/E[RR]: the quantity that makes the baseline reproduce its inputs."""
    from scipy import stats

    paf = loader.get_data(data_keys.SystolicBloodPressure.PAF, LOCATION)["draw_0"]
    mean = loader.get_data(data_keys.SystolicBloodPressure.EXPOSURE, LOCATION)["draw_0"]
    sd = loader.get_data(data_keys.SystolicBloodPressure.EXPOSURE_SD, LOCATION)["draw_0"]
    risks = loader.get_data(data_keys.SystolicBloodPressure.RELATIVE_RISK, LOCATION)["draw_0"]
    stratum = ("Male", 60.0, 65.0, 2021, 2022)
    m, s = mean.loc[stratum], sd.loc[stratum]
    rr = risks.loc[stratum + ("ischemic_heart_disease",)]
    rng = np.random.default_rng(1)
    pressure = stats.gamma.rvs((m / s) ** 2, scale=s**2 / m, size=2_000_000, random_state=rng)
    mean_rr = np.mean(rr ** (np.maximum(pressure - 112.5, 0) / 10))
    assert paf.loc[stratum + ("ischemic_heart_disease",)] == pytest.approx(
        1 - 1 / mean_rr, abs=2e-3)
    assert ((paf >= 0) & (paf < 1)).all()


@pytest.mark.parametrize("mean, sd, hazard_z", [
    (4.34, 0.45, 0.0),       # stage G3a before any change
    (4.80, 0.50, 0.0),
    (5.50, 0.70, 1.3),
    (4.63, 0.67, -2.6),      # a draw low enough for the curve to dip below one
    (6.80, 0.40, 0.0),       # mostly beyond the last tabulated level
])
def test_expected_hyperkalaemia_risk_is_the_integral_of_the_pointwise_risk(mean, sd, hazard_z):
    from scipy import integrate, stats

    density = stats.norm(mean, sd).pdf
    risk = lambda k: float(literature.hyperkalaemia_relative_risk(k, hazard_z)) * density(k)  # noqa: E731
    breaks = [mean - 10 * sd] + list(np.arange(5.5, 6.5001, 0.05)) + [mean + 12 * sd]
    breaks = sorted(b for b in breaks if mean - 10 * sd <= b <= mean + 12 * sd)
    numerical = sum(integrate.quad(risk, a, b, epsabs=1e-13, limit=200)[0]
                    for a, b in zip(breaks, breaks[1:]))
    exact = float(literature.expected_hyperkalaemia_relative_risk(mean, sd, hazard_z))
    assert exact == pytest.approx(numerical, abs=2e-6)


def test_expected_hyperkalaemia_risk_broadcasts_and_handles_no_spread():
    risk = literature.expected_hyperkalaemia_relative_risk
    grid = risk(np.array([[4.3], [5.0]]), np.array([0.4, 0.6, 0.8]), 0.0)
    assert grid.shape == (2, 3)
    assert (grid >= 1).all() and (np.diff(grid, axis=1) > 0).all()     # more spread, more tail
    assert float(risk(4.2, 0.0)) == 1.0
    assert float(risk(6.0, 0.0)) == pytest.approx(float(literature.hyperkalaemia_relative_risk(6.0)))
    # At G3a about one person in two hundred is above 5.5 mmol/L.
    assert float(risk(4.34, 0.45)) - 1 == pytest.approx(0.215 * 0.005, rel=0.05)
