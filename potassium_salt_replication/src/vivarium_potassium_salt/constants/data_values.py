"""The paper's fixed inputs, with their sources, and draws from their uncertainty.

Everything that is not location-specific lives here: the chemistry of the salt,
the dose-response relationships, the hazard ratios for kidney disease, and the
handful of numbers I had to reconstruct because the paper does not print them.
Each reconstructed number says so where it is defined.

``ModelParameters.for_draw(0)`` returns the central values.  Any other draw
number returns one joint sample, seeded by the draw number, so that the
simulation and the closed-form check use identical parameters for a draw.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace

import numpy as np
import patsy

##################
# Salt chemistry #
##################

SODIUM_MOLAR_MASS = 22.990
POTASSIUM_MOLAR_MASS = 39.098
CHLORINE_MOLAR_MASS = 35.453
#: grams of sodium per gram of sodium chloride (0.393)
SODIUM_FRACTION_OF_NACL = SODIUM_MOLAR_MASS / (SODIUM_MOLAR_MASS + CHLORINE_MOLAR_MASS)
#: grams of potassium per gram of potassium chloride (0.524)
POTASSIUM_FRACTION_OF_KCL = POTASSIUM_MOLAR_MASS / (POTASSIUM_MOLAR_MASS + CHLORINE_MOLAR_MASS)
#: grams of potassium gained per gram of sodium lost, when potassium chloride
#: replaces sodium chloride weight for weight (1.33)
POTASSIUM_GAINED_PER_SODIUM_LOST = POTASSIUM_FRACTION_OF_KCL / SODIUM_FRACTION_OF_NACL
MMOL_PER_GRAM_SODIUM = 1000 / SODIUM_MOLAR_MASS
MMOL_PER_GRAM_POTASSIUM = 1000 / POTASSIUM_MOLAR_MASS

################################
# Systolic blood pressure, GBD #
################################

#: Theoretical minimum risk exposure level used with the GBD 2019 relative
#: risks.  (GBD 2021 and 2023 use 105 to 115 mmHg.)
TMREL_RANGE_MMHG = (110.0, 115.0)
RELATIVE_RISK_SCALAR_MMHG = 10.0

#: The distributions vivarium's ensemble knows about.  The paper uses a gamma.
ENSEMBLE_DISTRIBUTIONS = (
    "betasr", "exp", "gamma", "glnorm", "gumbel", "invgamma", "invweibull",
    "llogis", "lnorm", "mgamma", "mgumbel", "norm", "weibull",
)

# GBD 2023 risk-factor methods appendix, high systolic blood pressure, Table 4:
#   log(SD) = b0 + b1 log(mean SBP) + sum_k b_k s_k(age) + b8 sex + b9 year,
# where s_k is a cubic spline on the starting age of the five-year age group
# with knots at 35, 50, and 65.  The appendix does not say which spline basis,
# how sex is coded, or how year is scaled.  I take a B-spline basis with
# boundary knots at 25 and 95, female = 1, and the calendar year.
SBP_SD_INTERCEPT = -9.845
SBP_SD_LOG_MEAN = 1.703
SBP_SD_AGE_SPLINE = (0.015, 0.121, 0.326, 0.320, 0.359, 0.305)
SBP_SD_AGE_KNOTS = (35.0, 50.0, 65.0)
SBP_SD_AGE_BOUNDS = (25.0, 95.0)
SBP_SD_FEMALE = 0.099
SBP_SD_YEAR = 0.002
# Table 6 of the same appendix: the factor that shrinks the standard deviation
# of one measurement to that of "usual" blood pressure, by starting age.
SBP_USUAL_LEVEL_CORRECTION = {
    25: 0.763, 30: 0.812, 35: 0.825, 40: 0.801, 45: 0.793, 50: 0.779,
    55: 0.761, 60: 0.757, 65: 0.738, 70: 0.719, 75: 0.745, 80: 0.700,
}


def sbp_standard_deviation(mean_sbp, age_start, sex, year: int = 2021):
    """Standard deviation of usual systolic blood pressure within a stratum.

    The IHME connector serves the mean but not the standard deviation, so this
    rebuilds it from the regression GBD 2023 publishes.  It is my reading of a
    two-line description, not GBD's own number; on the cluster the real one is
    ``get_measure(risk_factors.high_systolic_blood_pressure,
    "exposure_standard_deviation", location)``.

    The three arguments broadcast against each other.
    """
    mean_sbp, age_start, sex = np.broadcast_arrays(
        np.asarray(mean_sbp, dtype=float), np.asarray(age_start, dtype=float), np.asarray(sex)
    )
    ages, position = np.unique(np.clip(age_start, *SBP_SD_AGE_BOUNDS), return_inverse=True)
    basis = np.asarray(patsy.dmatrix(
        "bs(age, knots=knots, degree=3, include_intercept=False, "
        "lower_bound=lower, upper_bound=upper) - 1",
        {"age": ages, "knots": SBP_SD_AGE_KNOTS,
         "lower": SBP_SD_AGE_BOUNDS[0], "upper": SBP_SD_AGE_BOUNDS[1]},
    ))
    age_term = (basis @ np.asarray(SBP_SD_AGE_SPLINE))[position].reshape(mean_sbp.shape)
    starts = np.minimum(5 * np.floor(ages / 5), 80).astype(int)
    correction = np.array([SBP_USUAL_LEVEL_CORRECTION[int(a)] for a in starts])
    correction = correction[position].reshape(mean_sbp.shape)
    log_sd = (
        SBP_SD_INTERCEPT + SBP_SD_LOG_MEAN * np.log(mean_sbp) + age_term
        + SBP_SD_FEMALE * (sex == "Female") + SBP_SD_YEAR * year
    )
    return np.exp(log_sd) * correction


##############################
# Sources of dietary sodium  #
##############################

# Bhat et al. (2020), Figure 3: percent of salt from discretionary sources
# against GDP per capita in constant 2010 US dollars.  The slope and its
# interval are printed (-0.00087 per dollar; 95% CI -0.0012 to -0.00051).  The
# intercept is not.  Refitting the line to the 23 points digitised from the
# figure reproduces the printed slope, F statistic, and R-squared, and gives an
# intercept of 63.3 (standard error 5.4), which is what I use.
DISCRETIONARY_INTERCEPT_PERCENT = (63.3, 5.4)
DISCRETIONARY_SLOPE_PERCENT_PER_1000_USD = (-0.87, (1.2 - 0.51) / (2 * 1.96))
#: The paper divides 2021 dollars by this to reach 2010 dollars.
GDP_DEFLATOR_2021_TO_2010 = 1.24
#: The regression line goes below zero above about $72,000.  The paper does not
#: say what it did there.  Its results for Luxembourg, Ireland, Switzerland,
#: and Norway behave as if a small positive share had been used, and 5% is the
#: floor that reproduces them (see the README).
DISCRETIONARY_SHARE_BOUNDS = (0.05, 0.95)

##############################################
# Sodium, potassium, and blood pressure      #
##############################################

#: World Health Organization limit, below which the paper models no benefit.
SODIUM_FLOOR_G_PER_DAY = 2.0
KCL_FRACTION_OF_SALT_SUBSTITUTE = 0.25

# Mozaffarian et al. (2014), as quoted in the paper's Supplementary Text 1:
# fall in systolic blood pressure, mmHg, per 100 mmol/day less sodium.
SODIUM_EFFECT_AT_AGE_50 = (3.735, 0.73)          # normotensive, age 50
SODIUM_EFFECT_PER_YEAR_OF_AGE = (0.105, 0.029)   # more per year above 50, less below
SODIUM_EFFECT_AGE_FLOOR = 40.0                   # "does not go any less for age below 40"
SODIUM_EFFECT_REFERENCE_AGE = 50.0
SODIUM_EFFECT_HYPERTENSION = (1.874, 0.884)      # additional, with hypertension

# Filippini et al. (2021), the paper's third sensitivity analysis: the same
# quantity by hypertension status, with no age gradient.
SODIUM_EFFECT_FILIPPINI_2021 = {
    "normotensive": (2.30, (3.27 - 1.33) / (2 * 1.96)),
    "hypertensive": (6.50, (7.79 - 5.22) / (2 * 1.96)),
}

# Filippini et al. (2020), as quoted in Supplementary Text 1: fall in systolic
# blood pressure per mmol/day more potassium, and the ceiling on that fall, by
# baseline sodium intake.  Rows are (upper bound of sodium band in g/day,
# (slope, sd), ceiling in mmHg).
POTASSIUM_EFFECT_BY_BASELINE_SODIUM = (
    (3.0, (0.07, 0.05), 2.2),
    (4.0, (0.11, 0.04), 3.3),
    (np.inf, (0.11, 0.05), 6.5),
)

###########################################
# Chronic kidney disease and hyperkalaemia #
###########################################

#: Share of stage 3 that is stage 3a.  From the general-population cohorts of
#: Matsushita et al. (2012), Figure 2, CKD-EPI equation: 46,601 participants
#: with eGFR 45-59 and 9,993 with eGFR 30-44.
STAGE_3A_SHARE_OF_STAGE_3 = 46601 / (46601 + 9993)

# Marklund et al. (2020), Table 1.  Crude hazard ratio of cardiovascular death
# against normal kidney function (from Matsushita et al., 2012), and mean and
# standard deviation of serum potassium (from Kovesdy et al., 2018), by stage.
CKD_STAGE_HAZARD_RATIO = {
    "G3a": (4.03, 2.76, 5.88),
    "G3b": (6.95, 4.35, 11.10),
    "G4": (9.93, 6.09, 16.17),
    "G5": (17.51, 9.72, 31.55),
}
SERUM_POTASSIUM_MMOL_PER_L = {
    "G3a": (4.34, 0.45),
    "G3b": (4.42, 0.51),
    "G4": (4.52, 0.59),
    "G5": (4.63, 0.67),
}
# Rise in serum potassium, mmol/L, per g/day more dietary potassium in stage 3
# (Turban et al., 2021), and how much larger the rise is in later stages
# (Ogata et al., 2023), as given in the paper's Supplementary Text 1.
SERUM_POTASSIUM_PER_G_DIETARY = (0.23, (0.38 - 0.08) / (2 * 1.96))
SERUM_POTASSIUM_RESPONSE_MULTIPLIER = {"G3a": 1.0, "G3b": 1.0, "G4": 1.5, "G5": 2.0}
#: The paper counts as hyperkalaemia deaths only those above this level.
HYPERKALAEMIA_THRESHOLD_MMOL_PER_L = 5.5

# Uncertainty in the age-specific hazard ratio of cardiovascular death with
# chronic kidney disease (Hallan et al., 2012, via Marklund et al., 2020,
# eTable 1).  The interpolated point estimates carry no interval, so I borrow
# the width of Hallan's printed intervals for all-cause mortality.
CKD_HAZARD_RATIO_LOG_SD = {"under_55": 0.16, "55_plus": 0.05}

#########################
# Targeted scenarios    #
#########################

#: In scenarios 4 and 5 a fifth of people with hypertension are assumed to be
#: advised against the salt.  Sensitivity analyses 12 and 13 raise that to
#: three in ten in low- and middle-income countries.
TARGETED_UPTAKE = 0.8
TARGETED_UPTAKE_LMIC_SENSITIVITY = 0.7
#: World Bank high-income threshold for fiscal year 2023 (2021 GNI per capita),
#: applied here to GDP per capita because that is the series in hand.
HIGH_INCOME_THRESHOLD_USD = 13_205


####################
# Parameter draws  #
####################


@dataclass(frozen=True)
class ModelParameters:
    """One joint value of every non-location-specific parameter."""

    draw: int = 0
    tmrel: float = 0.5 * sum(TMREL_RANGE_MMHG)
    discretionary_intercept: float = DISCRETIONARY_INTERCEPT_PERCENT[0]
    discretionary_slope: float = DISCRETIONARY_SLOPE_PERCENT_PER_1000_USD[0]
    sodium_effect_at_50: float = SODIUM_EFFECT_AT_AGE_50[0]
    sodium_effect_per_year: float = SODIUM_EFFECT_PER_YEAR_OF_AGE[0]
    sodium_effect_hypertension: float = SODIUM_EFFECT_HYPERTENSION[0]
    sodium_effect_filippini_normotensive: float = SODIUM_EFFECT_FILIPPINI_2021["normotensive"][0]
    sodium_effect_filippini_hypertensive: float = SODIUM_EFFECT_FILIPPINI_2021["hypertensive"][0]
    potassium_slopes: tuple[float, ...] = tuple(
        slope for _, (slope, _), _ in POTASSIUM_EFFECT_BY_BASELINE_SODIUM
    )
    serum_potassium_per_g: float = SERUM_POTASSIUM_PER_G_DIETARY[0]
    serum_potassium_mean: dict[str, float] = field(
        default_factory=lambda: {k: v[0] for k, v in SERUM_POTASSIUM_MMOL_PER_L.items()}
    )
    ckd_stage_hazard_ratio: dict[str, float] = field(
        default_factory=lambda: {k: v[0] for k, v in CKD_STAGE_HAZARD_RATIO.items()}
    )
    #: standard normal deviates reused for quantities drawn elsewhere
    hyperkalaemia_hazard_z: float = 0.0
    ckd_age_hazard_z: float = 0.0

    @classmethod
    def for_draw(cls, draw: int | None) -> "ModelParameters":
        """Central values for draw 0 (or None); a seeded joint sample otherwise."""
        if not draw:
            return cls()
        rng = np.random.default_rng(zlib.crc32(f"model_parameters|{draw}".encode()))

        def normal(mean_sd, lower=None):
            value = rng.normal(*mean_sd)
            return value if lower is None else max(value, lower)

        def lognormal(mean_lower_upper):
            mean, lower, upper = mean_lower_upper
            return float(np.exp(rng.normal(np.log(mean), np.log(upper / lower) / (2 * 1.96))))

        return cls(
            draw=draw,
            # The paper draws the level uniformly from the range each time.
            tmrel=float(rng.uniform(*TMREL_RANGE_MMHG)),
            discretionary_intercept=normal(DISCRETIONARY_INTERCEPT_PERCENT),
            discretionary_slope=normal(DISCRETIONARY_SLOPE_PERCENT_PER_1000_USD),
            # Effects that would raise blood pressure are set to zero, as in
            # the model this one descends from.
            sodium_effect_at_50=normal(SODIUM_EFFECT_AT_AGE_50, 0.0),
            sodium_effect_per_year=normal(SODIUM_EFFECT_PER_YEAR_OF_AGE),
            sodium_effect_hypertension=normal(SODIUM_EFFECT_HYPERTENSION),
            sodium_effect_filippini_normotensive=normal(
                SODIUM_EFFECT_FILIPPINI_2021["normotensive"], 0.0),
            sodium_effect_filippini_hypertensive=normal(
                SODIUM_EFFECT_FILIPPINI_2021["hypertensive"], 0.0),
            potassium_slopes=tuple(
                normal(slope, 0.0) for _, slope, _ in POTASSIUM_EFFECT_BY_BASELINE_SODIUM
            ),
            serum_potassium_per_g=normal(SERUM_POTASSIUM_PER_G_DIETARY, 0.0),
            serum_potassium_mean={k: v[0] for k, v in SERUM_POTASSIUM_MMOL_PER_L.items()},
            ckd_stage_hazard_ratio={k: lognormal(v) for k, v in CKD_STAGE_HAZARD_RATIO.items()},
            hyperkalaemia_hazard_z=float(rng.standard_normal()),
            ckd_age_hazard_z=float(rng.standard_normal()),
        )

    def with_changes(self, **changes) -> "ModelParameters":
        return replace(self, **changes)


def discretionary_share(gdp_per_capita_2010_usd, parameters: ModelParameters | None = None):
    """Proportion of dietary sodium that is discretionary, from GDP per capita."""
    parameters = parameters or ModelParameters()
    percent = (
        parameters.discretionary_intercept
        + parameters.discretionary_slope * np.asarray(gdp_per_capita_2010_usd, dtype=float) / 1000
    )
    return np.clip(percent / 100, *DISCRETIONARY_SHARE_BOUNDS)
