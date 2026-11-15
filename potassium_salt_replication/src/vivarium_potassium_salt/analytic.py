"""The same model in closed form, to check the simulation and to carry uncertainty.

The simulation estimates each stratum's potential impact fraction by averaging
over simulants.  For the paper's model that average has an exact answer.  With
blood pressure gamma distributed and a log-linear relative risk above the
TMREL ``T``, a group whose pressure falls by ``d`` has mean relative risk

    E[RR(X - d)] = F(T + d; k, s) + exp(-b (T + d)) (1 - b s)^-k (1 - F(T + d; k, s')),

where ``F`` is the gamma distribution function with shape ``k`` and scale ``s``,
``b`` is the log relative risk per mmHg, and ``s' = s / (1 - b s)``.  The first
term is the people left at or below the TMREL, whose relative risk is one.

This module evaluates that expression for every stratum, cause, and draw at
once.  It has two jobs:

* It checks the simulation.  The two share the dose-response arithmetic in
  ``components/intervention.py`` and nothing else, so agreement stratum by
  stratum (``tests/test_simulation.py``) says the components, the risk
  effects, and the observers are wired as intended.
* It propagates uncertainty.  A thousand draws for 193 countries and five
  scenarios take minutes here.  Through the simulation, on one four-core
  machine, they would take about two weeks; on a cluster they are what
  ``psimulate`` is for.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import special, stats
from vivarium.gbd_mapping import causes, risk_factors

from vivarium_potassium_salt.components import intervention
from vivarium_potassium_salt.constants import data_values, metadata
from vivarium_potassium_salt.data import gbd_extract, literature, loader
from vivarium_potassium_salt.data import vivarium_inputs_stub as interface


@dataclass(frozen=True)
class Settings:
    """Model switches; the defaults are the paper's primary analysis.

    The first group has the same names and meanings as the configuration of the
    ``PotassiumEnrichedSalt``, ``DietarySodium``, and ``HypertensionStatus``
    components.
    """

    kcl_fraction: float = data_values.KCL_FRACTION_OF_SALT_SUBSTITUTE
    replacement_proportion: float = 1.0
    sodium_floor: float = data_values.SODIUM_FLOOR_G_PER_DAY
    targeted_uptake: float = data_values.TARGETED_UPTAKE
    targeted_uptake_lmic: float | None = None
    potassium_effect_scale: float = 1.0
    combined_effect_scale: float = 1.0
    potassium_effect_in_normotensives: bool = True
    sodium_dose_response: str = "mozaffarian_2014"
    discretionary_share_source: str = "regression"
    association_with_blood_pressure: str = "independent"
    # Only the closed form has these.
    #: ``individual``: each person's blood pressure falls by their own response,
    #: which depends on whether they have hypertension and whether they use the
    #: substitute; this is what the simulation does.  ``stratum_mean``: the
    #: whole stratum's distribution moves by the average of those responses,
    #: which is how a comparative risk assessment with one distribution per
    #: stratum (the paper, as I read it) has to work.  The relative risk is
    #: convex, so the second always averts at least as much as the first.
    blood_pressure_response: str = "individual"
    #: ``shared``: a dose-response coefficient is one unknown number, so each
    #: draw uses one value of it everywhere.  ``independent``: every country,
    #: sex, and age band gets its own value in each draw.
    parameter_draws: str = "shared"
    sample_tmrel: bool = True
    #: How cardiovascular deaths are divided among kidney disease stages before
    #: the hyperkalaemia risk is applied.  ``among_ckd`` takes the deaths among
    #: people with kidney disease implied by the age-specific hazard ratio (the
    #: figure the blood pressure half of the model uses) and divides those among
    #: the stages in proportion to prevalence times the stage's hazard ratio.
    #: ``all_adults`` is equation 4 of the parent model taken literally within
    #: each stratum: a stage's share is its prevalence times its crude hazard
    #: ratio, against a hazard ratio of one for everyone else.  The paper does
    #: not say which it used; the first reproduces its hyperkalaemia deaths and
    #: the second gives about 60% more, so the first is the default.
    stage_deaths: str = "among_ckd"
    sbp_sd_scale: float = 1.0
    discretionary_share_floor: float = data_values.DISCRETIONARY_SHARE_BOUNDS[0]


#######################
# Inputs for a place  #
#######################


@dataclass
class LocationInputs:
    """Everything the model needs for one location.

    Arrays have a row per stratum (sex, then age band) and a column per draw,
    with the mean in column 0.  ``relative_risk`` and ``counts`` carry a
    leading axis over causes.
    """

    location: str
    strata: pd.DataFrame                  # sex, age_group, age_midpoint, population
    sodium: np.ndarray
    sbp_mean: np.ndarray
    hypertension: np.ndarray
    diagnosed: np.ndarray                 # share of people with hypertension
    treated: np.ndarray
    relative_risk: np.ndarray             # [cause, stratum, draw]
    counts: dict[str, np.ndarray]         # measure -> [cause, stratum, draw]
    ckd_prevalence: dict[str, np.ndarray]  # stage -> [stratum, 1]
    ckd_hazard_ratio: np.ndarray          # by age; [stratum, draw]
    gdp_per_capita: float
    high_income: bool
    reported_discretionary_share: float
    parameters: "ParameterArrays" = field(repr=False, default=None)

    @property
    def n_draws(self) -> int:
        return self.sodium.shape[1]


@dataclass
class ParameterArrays:
    """``ModelParameters`` for many draws at once: each field is an array over draws."""

    tmrel: np.ndarray
    discretionary_intercept: np.ndarray
    discretionary_slope: np.ndarray
    sodium_effect_at_50: np.ndarray
    sodium_effect_per_year: np.ndarray
    sodium_effect_hypertension: np.ndarray
    sodium_effect_filippini_normotensive: np.ndarray
    sodium_effect_filippini_hypertensive: np.ndarray
    potassium_slopes: np.ndarray          # [band, draw]
    serum_potassium_per_g: np.ndarray
    serum_potassium_mean: dict[str, np.ndarray]
    ckd_stage_hazard_ratio: dict[str, np.ndarray]
    hyperkalaemia_hazard_z: np.ndarray
    ckd_age_hazard_z: np.ndarray

    @classmethod
    def independent(cls, location: str, n_strata: int, n_sampled: int) -> "ParameterArrays":
        """A fresh value of every parameter for each stratum of ``location`` in each draw.

        Arrays are [stratum, draw] (the discretionary-share regression, which
        is one line per country, stays [draw]).  Column 0 holds the central
        values.
        """
        central = data_values.ModelParameters()
        rng = np.random.default_rng(zlib.crc32(f"independent_parameters|{location}".encode()))
        size = (n_strata, n_sampled)

        def with_mean(mean, sampled):
            first = np.full(sampled.shape[:-1] + (1,), mean, dtype=float)
            return np.concatenate([first, sampled], axis=-1)

        def normal(mean_sd, lower=None, shape=size):
            sampled = rng.normal(*mean_sd, size=shape)
            return with_mean(mean_sd[0], sampled if lower is None else np.maximum(sampled, lower))

        def lognormal(mean_lower_upper):
            mean, lower, upper = mean_lower_upper
            sampled = np.exp(rng.normal(np.log(mean), np.log(upper / lower) / (2 * 1.96), size))
            return with_mean(mean, sampled)

        filippini = data_values.SODIUM_EFFECT_FILIPPINI_2021
        return cls(
            tmrel=with_mean(central.tmrel, rng.uniform(*data_values.TMREL_RANGE_MMHG, size)),
            discretionary_intercept=normal(
                data_values.DISCRETIONARY_INTERCEPT_PERCENT, shape=(n_sampled,)),
            discretionary_slope=normal(
                data_values.DISCRETIONARY_SLOPE_PERCENT_PER_1000_USD, shape=(n_sampled,)),
            sodium_effect_at_50=normal(data_values.SODIUM_EFFECT_AT_AGE_50, 0.0),
            sodium_effect_per_year=normal(data_values.SODIUM_EFFECT_PER_YEAR_OF_AGE),
            sodium_effect_hypertension=normal(data_values.SODIUM_EFFECT_HYPERTENSION),
            sodium_effect_filippini_normotensive=normal(filippini["normotensive"], 0.0),
            sodium_effect_filippini_hypertensive=normal(filippini["hypertensive"], 0.0),
            potassium_slopes=np.stack([
                normal(slope, 0.0)
                for _, slope, _ in data_values.POTASSIUM_EFFECT_BY_BASELINE_SODIUM
            ]),
            serum_potassium_per_g=normal(data_values.SERUM_POTASSIUM_PER_G_DIETARY, 0.0),
            serum_potassium_mean={
                stage: np.full((n_strata, n_sampled + 1), value[0])
                for stage, value in data_values.SERUM_POTASSIUM_MMOL_PER_L.items()
            },
            ckd_stage_hazard_ratio={
                stage: lognormal(value) for stage, value in data_values.CKD_STAGE_HAZARD_RATIO.items()
            },
            hyperkalaemia_hazard_z=with_mean(0.0, rng.standard_normal(size)),
            ckd_age_hazard_z=with_mean(0.0, rng.standard_normal(size)),
        )

    @classmethod
    @lru_cache
    def for_draws(cls, n_sampled: int) -> "ParameterArrays":
        draws = [data_values.ModelParameters.for_draw(d) for d in range(n_sampled + 1)]

        def stack(name):
            return np.array([getattr(d, name) for d in draws])

        return cls(
            **{name: stack(name) for name in (
                "tmrel", "discretionary_intercept", "discretionary_slope",
                "sodium_effect_at_50", "sodium_effect_per_year", "sodium_effect_hypertension",
                "sodium_effect_filippini_normotensive", "sodium_effect_filippini_hypertensive",
                "serum_potassium_per_g", "hyperkalaemia_hazard_z", "ckd_age_hazard_z",
            )},
            potassium_slopes=np.array([d.potassium_slopes for d in draws]).T,
            serum_potassium_mean={
                stage: np.array([d.serum_potassium_mean[stage] for d in draws])
                for stage in metadata.CKD_STAGES
            },
            ckd_stage_hazard_ratio={
                stage: np.array([d.ckd_stage_hazard_ratio[stage] for d in draws])
                for stage in metadata.CKD_STAGES
            },
        )


def _mean_and_draws(request, *args, n_sampled: int, drop=("location", "year_start", "year_end")):
    """A stub request as one frame: the mean in column 0, then ``n_sampled`` draws."""
    means = request(*args, data_type="means")
    if n_sampled:
        draws = request(*args, data_type="draws").iloc[:, :n_sampled]
        means = pd.concat([means, draws], axis=1)
    return means.droplevel([level for level in drop if level in means.index.names])


def load_inputs(location: str, n_sampled: int = 0,
                parameter_draws: str = "shared") -> LocationInputs:
    """Read one location's inputs through the same stubbed interface the artifact uses."""
    sbp, sodium = risk_factors.high_systolic_blood_pressure, risk_factors.diet_high_in_sodium

    def exposure(entity):
        frame = _mean_and_draws(interface.get_measure, entity, "exposure", location,
                                n_sampled=n_sampled)
        return frame.droplevel("parameter").sort_index()

    sodium_frame, sbp_frame = exposure(sodium), exposure(sbp)
    index = sodium_frame.index                      # sex, age_start, age_end
    bands = {band.start: band for band in metadata.AGE_BANDS}
    people = interface.get_population_structure(location).droplevel(
        ["location", "year_start", "year_end"]).reindex(index)["value"]
    strata = pd.DataFrame({
        "sex": index.get_level_values("sex"),
        "age_group": [bands[a].name for a in index.get_level_values("age_start")],
        "age_midpoint": [bands[a].midpoint for a in index.get_level_values("age_start")],
        "population": people.to_numpy(),
    })

    risks = _mean_and_draws(interface.get_measure, sbp, "relative_risk", location,
                            n_sampled=n_sampled).droplevel("parameter")
    relative_risk = np.stack([
        risks.xs(cause, level="affected_entity").reindex(index).to_numpy()
        for cause in metadata.CAUSES
    ])

    people_array = people.to_numpy()[:, None]
    counts = {}
    for measure in gbd_extract.available_measures():
        request = interface.get_dalys if measure == "dalys" else (
            lambda entity, place, data_type, m=measure: interface.get_measure(
                entity, {"deaths": "cause_specific_mortality_rate",
                         "incident_cases": "incidence_rate"}[m], place, data_type=data_type))
        tables = []
        for cause in metadata.CAUSES:
            if cause in metadata.CAUSES_BY_MEASURE[measure]:
                rate = _mean_and_draws(request, causes[cause], location, n_sampled=n_sampled)
                tables.append(rate.reindex(index).to_numpy() * people_array)
            else:
                tables.append(np.zeros_like(sodium_frame.to_numpy()))
        counts[measure] = np.stack(tables)

    def hypertension(column):
        frame = loader.hypertension_draws(location, column, n_sampled)
        return frame.droplevel(["year_start", "year_end"]).reindex(index).to_numpy()

    kidney = interface.get_measure(
        risk_factors.kidney_dysfunction, "exposure", location, data_type="means"
    ).droplevel(["location", "year_start", "year_end"])
    stage3 = kidney.xs("cat3", level="parameter").reindex(index).to_numpy()
    ckd_prevalence = {
        "G3a": stage3 * data_values.STAGE_3A_SHARE_OF_STAGE_3,
        "G3b": stage3 * (1 - data_values.STAGE_3A_SHARE_OF_STAGE_3),
        "G4": kidney.xs("cat2", level="parameter").reindex(index).to_numpy(),
        "G5": kidney.xs("cat1", level="parameter").reindex(index).to_numpy(),
    }

    if parameter_draws == "shared":
        parameters = ParameterArrays.for_draws(n_sampled)
    elif parameter_draws == "independent":
        parameters = ParameterArrays.independent(location, len(strata), n_sampled)
        # Relative risks too: a fresh value for this country in every draw.
        risk_table = literature.sbp_relative_risks().set_index(["cause", "age_group"])
        rng = np.random.default_rng(zlib.crc32(f"independent_relative_risk|{location}".encode()))
        for c, cause in enumerate(metadata.CAUSES):
            rows = risk_table.loc[cause].reindex(strata["age_group"])
            log_sd = np.log(rows["upper"] / rows["lower"]).to_numpy() / (2 * 1.96)
            noise = rng.standard_normal((len(strata), n_sampled))
            relative_risk[c, :, 1:] = np.exp(
                np.log(rows["mean"].to_numpy())[:, None] + log_sd[:, None] * noise)
    else:
        raise ValueError(f"unknown parameter_draws {parameter_draws!r}")
    hazard = literature.ckd_cvd_hazard_ratio_by_age().set_index("age_group").reindex(
        strata["age_group"])
    ckd_hazard_ratio = np.maximum(
        hazard["hazard_ratio"].to_numpy()[:, None]
        * np.exp(_by_draw(parameters.ckd_age_hazard_z) * hazard["log_sd"].to_numpy()[:, None]),
        1.0,
    )
    return LocationInputs(
        location=location,
        strata=strata,
        sodium=sodium_frame.to_numpy(),
        sbp_mean=sbp_frame.to_numpy(),
        hypertension=hypertension("prevalence"),
        diagnosed=hypertension("diagnosed"),
        treated=hypertension("treated"),
        relative_risk=relative_risk,
        counts=counts,
        ckd_prevalence=ckd_prevalence,
        ckd_hazard_ratio=ckd_hazard_ratio,
        gdp_per_capita=literature.gdp_per_capita_2010_usd(location),
        high_income=literature.is_high_income(location),
        reported_discretionary_share=float(
            literature.reported_discretionary_shares().get(location, np.nan)),
        parameters=parameters,
    )


#################
# The mechanics #
#################


def _by_draw(values: np.ndarray) -> np.ndarray:
    """Shape a parameter to broadcast against [stratum, draw] arrays."""
    values = np.asarray(values)
    return values[None, :] if values.ndim == 1 else values


def mean_relative_risk(shift, shape, scale, log_rr_per_mmhg, tmrel, lower=None, upper=None):
    """Integral of RR(x - shift) over a slice of a gamma distribution of blood pressure.

    ``RR(x) = exp(log_rr_per_mmhg * max(x - tmrel, 0))``.  ``lower`` and
    ``upper`` are blood pressures bounding the slice (default: all of it).  The
    result is not divided by the slice's probability, so slices add.
    """
    threshold = tmrel + shift
    if lower is not None or upper is not None:
        lower = 0.0 if lower is None else lower
        upper = np.inf if upper is None else upper
        threshold = np.clip(threshold, lower, upper)
    tilt = 1.0 / (1.0 - log_rr_per_mmhg * scale)
    cdf = lambda x, s: special.gammainc(shape, np.maximum(x, 0.0) / s)  # noqa: E731
    low_p = 0.0 if lower is None else cdf(lower, scale)
    high_tilted = 1.0 if upper is None or np.all(np.isinf(upper)) else cdf(upper, scale * tilt)
    at_reference = cdf(threshold, scale) - low_p
    above = (
        np.exp(-log_rr_per_mmhg * (tmrel + shift) + shape * np.log(tilt))
        * (high_tilted - cdf(threshold, scale * tilt))
    )
    return at_reference + above


def discretionary_share(inputs: LocationInputs, settings: Settings) -> np.ndarray:
    """Discretionary share of sodium for each draw."""
    if settings.discretionary_share_source == "reported_where_available" and not np.isnan(
            inputs.reported_discretionary_share):
        return np.full(inputs.n_draws, inputs.reported_discretionary_share)
    parameters = inputs.parameters
    percent = (parameters.discretionary_intercept
               + parameters.discretionary_slope * inputs.gdp_per_capita / 1000)
    return np.clip(percent / 100, settings.discretionary_share_floor,
                   data_values.DISCRETIONARY_SHARE_BOUNDS[1])


def evaluate(inputs: LocationInputs, settings: Settings = Settings(),
             scenarios=metadata.SCENARIOS) -> dict[str, dict]:
    """Potential impact fractions and kidney-disease quantities for each scenario.

    Returns, for each scenario name, a dict with ``pif`` ([cause, stratum,
    draw]), ``users`` ([stratum, draw], people using the substitute),
    ``sbp_reduction`` ([stratum, draw], mean fall in mmHg),
    ``prevented_cvd_deaths_in_ckd`` and ``hyperkalaemia_deaths`` ([stratum,
    draw]).
    """
    if settings.blood_pressure_response not in ("individual", "stratum_mean"):
        raise ValueError(
            f"unknown blood_pressure_response {settings.blood_pressure_response!r}")
    parameters = inputs.parameters
    age = inputs.strata["age_midpoint"].to_numpy()[:, None]
    people = inputs.strata["population"].to_numpy()[:, None]
    sd = data_values.sbp_standard_deviation(
        inputs.sbp_mean, np.broadcast_to(age - 2.5, inputs.sbp_mean.shape),
        np.broadcast_to(inputs.strata["sex"].to_numpy()[:, None], inputs.sbp_mean.shape),
    ) * settings.sbp_sd_scale
    shape, scale = (inputs.sbp_mean / sd) ** 2, sd**2 / inputs.sbp_mean
    log_rr = np.log(inputs.relative_risk) / data_values.RELATIVE_RISK_SCALAR_MMHG
    tmrel = _by_draw(parameters.tmrel) if settings.sample_tmrel else np.full(
        (1, inputs.n_draws), 0.5 * sum(data_values.TMREL_RANGE_MMHG))
    share = discretionary_share(inputs, settings)[None, :]
    prevalence = inputs.hypertension

    ranked = settings.association_with_blood_pressure == "ranked"
    if ranked:
        # People with hypertension are those above this blood pressure.
        cut = stats.gamma.ppf(1 - prevalence, shape, scale=scale)
        slices = {False: dict(lower=None, upper=cut), True: dict(lower=cut, upper=None)}
        mass = {False: 1.0, True: 1.0}
    else:
        slices = {False: {}, True: {}}
        mass = {False: 1 - prevalence, True: prevalence}

    def risk(shift, hypertensive):
        return mass[hypertensive] * mean_relative_risk(
            shift, shape, scale, log_rr, tmrel, **slices[hypertensive])

    # The incomplete gamma function is where the time goes, so nothing is
    # integrated twice: the baseline once, and a scenario only for the people
    # whose blood pressure it moves.
    if ranked:
        unchanged = {hypertensive: risk(0.0, hypertensive) for hypertensive in (False, True)}
    else:
        whole = mean_relative_risk(0.0, shape, scale, log_rr, tmrel)
        unchanged = {hypertensive: mass[hypertensive] * whole for hypertensive in (False, True)}
    baseline_risk = unchanged[False] + unchanged[True]

    cvd = np.array([cause in metadata.CVD_CAUSES for cause in metadata.CAUSES])
    cvd_deaths = inputs.counts["deaths"][cvd].sum(axis=0)
    ckd_total = sum(inputs.ckd_prevalence.values())
    ckd_share_of_cvd_deaths = ckd_total * inputs.ckd_hazard_ratio / (
        ckd_total * inputs.ckd_hazard_ratio + 1 - ckd_total)

    results = {}
    for scenario in scenarios:
        less_sodium = intervention.sodium_reduction(
            inputs.sodium, share, scenario.sources, settings.replacement_proportion,
            settings.kcl_fraction, settings.sodium_floor)
        more_potassium = less_sodium * data_values.POTASSIUM_GAINED_PER_SODIUM_LOST
        fall = {
            hypertensive: intervention.sbp_reduction(
                less_sodium, more_potassium, inputs.sodium, age, hypertensive, parameters,
                settings.sodium_dose_response, settings.potassium_effect_scale,
                settings.combined_effect_scale, settings.potassium_effect_in_normotensives)
            for hypertensive in (False, True)
        }
        if scenario.eligible == "everyone":
            uptake = {False: 1.0, True: 1.0}
        else:
            reached = settings.targeted_uptake
            if settings.targeted_uptake_lmic is not None and not inputs.high_income:
                reached = settings.targeted_uptake_lmic
            cascade = {"diagnosed": np.maximum(inputs.diagnosed, inputs.treated),
                       "treated": inputs.treated}[scenario.eligible]
            uptake = {False: 0.0, True: cascade * reached}

        user_share = (1 - prevalence) * uptake[False] + prevalence * uptake[True]
        mean_fall = ((1 - prevalence) * uptake[False] * fall[False]
                     + prevalence * uptake[True] * fall[True])
        if settings.blood_pressure_response == "stratum_mean":
            scenario_risk = mean_relative_risk(mean_fall, shape, scale, log_rr, tmrel)
        else:
            scenario_risk = sum(
                (uptake[h] * risk(fall[h], h) if np.any(uptake[h]) else 0.0)
                + (1 - uptake[h]) * unchanged[h]
                for h in (False, True)
            )
        pif = 1 - scenario_risk / baseline_risk
        averted_cvd_deaths = (pif * inputs.counts["deaths"])[cvd].sum(axis=0)
        results[scenario.name] = {
            "pif": pif,
            "users": user_share * people,
            "sbp_reduction": mean_fall,
            "prevented_cvd_deaths_in_ckd": averted_cvd_deaths * ckd_share_of_cvd_deaths,
            "hyperkalaemia_deaths": (
                hyperkalaemia_deaths(inputs, more_potassium, cvd_deaths,
                                     ckd_share_of_cvd_deaths, settings.stage_deaths)
                if scenario.hyperkalaemia_modelled else np.zeros_like(cvd_deaths)),
        }
    return results


def stage_shares_of_cvd_deaths(ckd_prevalence, stage_hazard_ratio, ckd_share, stage_deaths):
    """Share of a stratum's cardiovascular deaths that occur at each kidney disease stage.

    ``ckd_prevalence`` and ``stage_hazard_ratio`` map stage to array;
    ``ckd_share`` is the share among all people with stage 3 to 5 disease
    implied by the age-specific hazard ratio.  See ``Settings.stage_deaths``.
    """
    weights = {stage: ckd_prevalence[stage] * stage_hazard_ratio[stage]
               for stage in metadata.CKD_STAGES}
    total = sum(weights.values())
    if stage_deaths == "all_adults":
        denominator = 1 - sum(ckd_prevalence.values()) + total
        return {stage: weights[stage] / denominator for stage in metadata.CKD_STAGES}
    if stage_deaths == "among_ckd":
        return {stage: ckd_share * weights[stage] / total for stage in metadata.CKD_STAGES}
    raise ValueError(f"unknown stage_deaths {stage_deaths!r}")


def hyperkalaemia_deaths(inputs: LocationInputs, potassium_increase, cvd_deaths,
                         ckd_share, stage_deaths="among_ckd") -> np.ndarray:
    """Extra cardiovascular deaths from serum potassium above 5.5 mmol/L, [stratum, draw].

    For each stage, the stratum's cardiovascular deaths among people at that
    stage are multiplied by the proportional rise in their mean relative risk.
    """
    parameters = inputs.parameters
    shares = stage_shares_of_cvd_deaths(
        inputs.ckd_prevalence,
        {stage: _by_draw(parameters.ckd_stage_hazard_ratio[stage]) for stage in metadata.CKD_STAGES},
        ckd_share, stage_deaths,
    )
    extra = np.zeros_like(cvd_deaths)
    hazard_z = _by_draw(parameters.hyperkalaemia_hazard_z)
    for stage in metadata.CKD_STAGES:
        mean = _by_draw(parameters.serum_potassium_mean[stage])
        sd = data_values.SERUM_POTASSIUM_MMOL_PER_L[stage][1]
        rise = (_by_draw(parameters.serum_potassium_per_g)
                * data_values.SERUM_POTASSIUM_RESPONSE_MULTIPLIER[stage] * potassium_increase)
        # More potassium moves the stage's distribution up and stretches it.
        risk_before = literature.expected_hyperkalaemia_relative_risk(mean, sd, hazard_z)
        risk_after = literature.expected_hyperkalaemia_relative_risk(
            mean + rise, sd * (1 + rise / mean), hazard_z)
        extra += cvd_deaths * shares[stage] * (risk_after / risk_before - 1)
    return extra


####################
# Tidy summaries   #
####################


def stratum_table(inputs: LocationInputs, results: dict[str, dict], draw: int = 0) -> pd.DataFrame:
    """One draw in long form: a row per scenario, measure, cause, sex, and age band."""
    rows = []
    for scenario, result in results.items():
        for measure, counts in inputs.counts.items():
            for c, cause in enumerate(metadata.CAUSES):
                if cause not in metadata.CAUSES_BY_MEASURE[measure]:
                    continue
                rows.append(inputs.strata[["sex", "age_group"]].assign(
                    location=inputs.location, scenario=scenario, measure=measure, cause=cause,
                    baseline=counts[c, :, draw],
                    pif=result["pif"][c, :, draw],
                    averted=counts[c, :, draw] * result["pif"][c, :, draw],
                ))
    return pd.concat(rows, ignore_index=True)


def summarise(inputs: LocationInputs, results: dict[str, dict]) -> pd.DataFrame:
    """Every draw, summed over age bands: a row per scenario, quantity, and sex.

    Quantities are ``<measure>_averted__cvd``, ``<measure>_averted__ckd``,
    ``prevented_cvd_deaths_in_ckd``, ``hyperkalaemia_deaths``, and ``users``.
    Draw columns are ``draw_0`` (the mean inputs) onward.
    """
    cvd = np.array([cause in metadata.CVD_CAUSES for cause in metadata.CAUSES])
    sexes = inputs.strata["sex"].to_numpy()
    columns = [f"draw_{i}" for i in range(inputs.n_draws)]
    records, values = [], []

    def add(scenario, quantity, by_stratum):
        for sex in metadata.SEXES:
            records.append((inputs.location, scenario, quantity, sex))
            values.append(by_stratum[sexes == sex].sum(axis=0))

    for measure, counts in inputs.counts.items():
        add("baseline", f"{measure}__cvd", counts[cvd].sum(axis=0))
        add("baseline", f"{measure}__ckd", counts[~cvd].sum(axis=0))
    add("baseline", "population", np.broadcast_to(
        inputs.strata["population"].to_numpy()[:, None], inputs.sodium.shape))
    ckd = sum(inputs.ckd_prevalence.values()) * inputs.strata["population"].to_numpy()[:, None]
    add("baseline", "people_with_ckd_stage_3_to_5", np.broadcast_to(ckd, inputs.sodium.shape))
    for scenario, result in results.items():
        for measure, counts in inputs.counts.items():
            averted = result["pif"] * counts
            add(scenario, f"{measure}_averted__cvd", averted[cvd].sum(axis=0))
            add(scenario, f"{measure}_averted__ckd", averted[~cvd].sum(axis=0))
        for quantity in ("prevented_cvd_deaths_in_ckd", "hyperkalaemia_deaths", "users"):
            add(scenario, quantity, result[quantity])
    index = pd.MultiIndex.from_tuples(records, names=["location", "scenario", "quantity", "sex"])
    return pd.DataFrame(np.array(values), index=index, columns=columns)


def run_location(location: str, settings: Settings = Settings(), n_sampled: int = 0,
                 scenarios=metadata.SCENARIOS) -> pd.DataFrame:
    """Load a location, evaluate every scenario, and return the summary over draws."""
    inputs = load_inputs(location, n_sampled, settings.parameter_draws)
    return summarise(inputs, evaluate(inputs, settings, scenarios))
