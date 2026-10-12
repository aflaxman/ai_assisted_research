"""Model inputs for the MYRIAD replication.

Every value is transcribed from Chiosi et al. (2026), Open Forum Infect Dis,
doi:10.1093/ofid/ofag675 -- main-text Table 1 ("T1") or the Supplement
("S-B2" = Supplement Part B, Table B2) -- unless marked ASSUMPTION, in
which case the reason is given. The 2023 model-description paper
(Chiosi et al., BMC Health Serv Res 2023;23:760, "C23") is cited where the
2026 supplement is silent and the 2023 paper is explicit.

Conventions
-----------
* All probabilities are stored as *monthly* probabilities (fractions, not %).
* Annual probabilities from the paper are converted with ``annual_to_monthly``.
* Calendar schedules are dicts keyed by year; ``schedule_lookup`` extends the
  last value forward (and the first value backward).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def annual_to_monthly(p_annual: float) -> float:
    """Constant-hazard conversion of an annual probability to a monthly one."""
    return 1.0 - (1.0 - p_annual) ** (1.0 / 12.0)


def schedule_lookup(schedule: dict[int, float], year: int) -> float:
    """Value for ``year``; holds the first/last value flat outside the range."""
    years = sorted(schedule)
    if year <= years[0]:
        return schedule[years[0]]
    if year >= years[-1]:
        return schedule[years[-1]]
    return schedule[year]


def pct(x: float) -> float:
    return x / 100.0


# --------------------------------------------------------------------------
# Simulation framing
# --------------------------------------------------------------------------

COHORT_START_YEAR = 2018  # cohort is 3,694,500 PWID "in 2018" (T1); the paper ages
PROJECTION_START_YEAR = 2025  # it to 2025 before reporting (Methods, "Model inputs")
PROJECTION_END_YEAR = 2034
DISCOUNT_RATE = 0.03  # per year, costs discounted to 2025 (Methods)
DISCOUNT_BASE_YEAR = 2025

# --------------------------------------------------------------------------
# Population (T1; S-B part 1)
# --------------------------------------------------------------------------

N_PWID_2018 = 3_694_500  # Bradley et al. 2023 (T1)
PROPORTION_FEMALE = pct(29.7)  # T1
AGE_MEAN, AGE_SD, AGE_MIN, AGE_MAX = 40.0, 10.0, 18.0, 65.0  # T1 footnote b; S-B1

# --------------------------------------------------------------------------
# Drug use behaviour (T1; S-B1, S-B2)
# --------------------------------------------------------------------------

DRUG_STATES = ["low", "medium", "high", "remission"]
# T1 / S-B1 part A: 5-year NSDUH 2018-2022 average; nobody starts in remission.
INITIAL_DRUG_STATE = {"low": pct(2.0), "medium": pct(53.0), "high": pct(45.1)}
# S-B2: calibrated monthly transition probabilities (best LHS parameter set).
DRUG_TRANSITIONS = {
    ("low", "medium"): pct(1.34),
    ("low", "high"): pct(3.49),
    ("medium", "high"): pct(5.10),
    ("high", "remission"): pct(1.88),
    ("remission", "high"): pct(9.77),
}
# Drug-use states with active injection: these pay the monthly syringe cost.
ACTIVE_IDU_STATES = {"low", "medium", "high"}
# Whether people in remission keep the PWID (IDU) infection inputs. The
# supplement defines remission as "not engaging in active IDU" and gives
# separate non-IDU incidence/testing/ART/discharge inputs, but the paper's own
# 2025 counts (S-C1: 159,916 SBI, 3,303 HIV, 23,844 HCV) are reproduced only
# when the whole cohort keeps the IDU infection inputs; with remission on the
# non-IDU inputs SBI falls 13% and HCV 20% short. Remission still carries its
# own overdose rate and cost. Set False for the alternative reading.
REMISSION_KEEPS_IDU_INFECTION_RISK = os.environ.get("MYRIAD_REMISSION_NON_IDU", "0") != "1"

# --------------------------------------------------------------------------
# Overdose mortality: annual probability by drug-use state and calendar year
# (S-B3 "Estimated" 2018-2024 and "Projected" 2025-2034 base case; S-B5
# Sustained Decline; S-B6 Precipitous Increase). Columns: high, medium, low,
# remission.
# --------------------------------------------------------------------------

_OD_ESTIMATED_2018_2024 = {
    2018: (0.98, 0.66, 1.29, 0.20),
    2019: (1.05, 0.68, 1.12, 0.21),
    2020: (1.44, 0.83, 1.08, 0.29),
    2021: (1.68, 0.97, 1.01, 0.34),
    2022: (1.71, 0.97, 0.92, 0.35),
    2023: (1.66, 0.96, 0.80, 0.34),
    2024: (1.13, 0.82, 0.78, 0.23),
}
_OD_BASE_2025_2034 = {
    2025: (1.17, 0.81, 2.11, 0.24),
    2026: (1.21, 0.83, 2.14, 0.25),
    2027: (1.26, 0.85, 2.18, 0.26),
    2028: (1.30, 0.88, 2.21, 0.27),
    2029: (1.34, 0.90, 2.25, 0.27),
    2030: (1.38, 0.92, 2.28, 0.28),
    2031: (1.42, 0.95, 2.32, 0.29),
    2032: (1.47, 0.97, 2.35, 0.30),
    2033: (1.51, 0.99, 2.38, 0.31),
    2034: (1.55, 1.02, 2.42, 0.32),
}
_OD_SUSTAINED_DECLINE = {
    2025: (1.12, 0.82, 0.71, 0.23),
    2026: (0.99, 0.77, 0.65, 0.20),
    2027: (0.88, 0.73, 0.60, 0.18),
    2028: (0.78, 0.68, 0.55, 0.16),
    2029: (0.69, 0.64, 0.50, 0.14),
    2030: (0.61, 0.60, 0.46, 0.12),
    2031: (0.54, 0.56, 0.42, 0.11),
    2032: (0.48, 0.53, 0.38, 0.10),
    2033: (0.42, 0.49, 0.34, 0.09),
    2034: (0.37, 0.46, 0.31, 0.08),
}
_OD_PRECIPITOUS_INCREASE = {
    2025: (1.49, 0.92, 1.36, 0.30),
    2026: (1.55, 0.95, 1.36, 0.32),
    2027: (1.60, 0.98, 1.35, 0.33),
    2028: (1.66, 1.01, 1.35, 0.34),
    2029: (1.72, 1.04, 1.34, 0.35),
    2030: (1.78, 1.07, 1.34, 0.36),
    2031: (1.84, 1.10, 1.33, 0.38),
    2032: (1.90, 1.13, 1.32, 0.39),
    2033: (1.96, 1.16, 1.32, 0.40),
    2034: (2.02, 1.18, 1.31, 0.41),
}

SCENARIOS = {
    "base": _OD_BASE_2025_2034,
    "sustained_decline": _OD_SUSTAINED_DECLINE,
    "precipitous_increase": _OD_PRECIPITOUS_INCREASE,
}


def overdose_schedule(scenario: str) -> dict[int, dict[str, float]]:
    """Monthly overdose-death probability by year and drug state for a scenario."""
    annual = {**_OD_ESTIMATED_2018_2024, **SCENARIOS[scenario]}
    return {
        year: {
            "high": annual_to_monthly(pct(h)),
            "medium": annual_to_monthly(pct(m)),
            "low": annual_to_monthly(pct(lo)),
            "remission": annual_to_monthly(pct(r)),
        }
        for year, (h, m, lo, r) in annual.items()
    }


# --------------------------------------------------------------------------
# Severe bacterial (injection-related) infections (T1; S-B part 3)
# --------------------------------------------------------------------------

SBI_TYPES = ["ssti", "endocarditis", "osteomyelitis", "cns"]
# T1: distribution of infection type (2017 NIS distribution). Normalised below.
_SBI_TYPE_SHARE = {"ssti": 87.3, "endocarditis": 4.8, "osteomyelitis": 5.9, "cns": 2.1}
SBI_TYPE_P = np.array([_SBI_TYPE_SHARE[t] for t in SBI_TYPES]) / sum(_SBI_TYPE_SHARE.values())


def sbi_annual_probability_idu(year: int) -> float:
    """Annual probability of SBI hospitalisation among active PWID.

    S-B Eq. 8 gives IDU-related hospitalisations = 5,532*(year-2017) + intercept,
    divided by the 3,694,500 PWID. The printed intercept (59,664) is
    inconsistent with the 2017 observed count (94,068, S-B7: 2.55%) and with
    T1's 2018 value of 2.7%; using the 2017 observed count as the intercept
    reproduces T1 exactly (99,600 / 3,694,500 = 2.70%).  ASSUMPTION: the linear
    trend continues through 2034 (S-B says "2018-2023"); the paper's own SBI
    counts (S-C1) rise 11% over 2025-2034 while the cohort shrinks, which
    requires a rising per-capita rate.
    """
    hospitalisations = 94_068 + 5_532 * (year - 2017)
    return hospitalisations / N_PWID_2018


SBI_ANNUAL_PROBABILITY_NON_IDU = pct(0.33)  # S-B7 non-SUD 2017; Eq. 9 is ~flat

# S-B19: odds ratios for invasive bacterial infection by CD4, applied as rate
# multipliers to people with HIV.
SBI_HIV_MULTIPLIER = {"high": 3.52, "medium": 5.52, "low": 10.39}
# S-B8: probability of in-hospital death, by infection type
SBI_DEATH_P = {"endocarditis": pct(5.8), "osteomyelitis": pct(1.5), "cns": pct(3.6), "ssti": pct(1.5)}
# S-B9: survivors are cured unless self-directed discharge. OUD = "high" state.
SBI_SELF_DISCHARGE_P = {"oud": pct(19.2), "non_oud": pct(2.7)}
# C23: people with an uncured ("active") infection are re-evaluated each month
# with the same cure probability; an active infection blocks a new one.

# --------------------------------------------------------------------------
# HIV (T1; S-B part 4)
# --------------------------------------------------------------------------

CD4_STATES = ["high", "medium", "low"]  # >=500, 200-499, <200 cells/uL
HIV_PREVALENCE_2018 = pct(5.12)  # S-B10 (IDU)
HIV_INITIAL_DIAGNOSED = pct(91.8)  # S-B12 (IDU)
HIV_INITIAL_ON_ART_GIVEN_DIAGNOSED = pct(47.7)  # S-B13: viral suppression as proxy
# S-B15 initial CD4 distribution (high, medium, low)
HIV_INITIAL_CD4 = {
    "on_art": (pct(45.4), pct(40.6), pct(14.0)),
    "off_art": (pct(34.3), pct(40.8), pct(24.9)),
    "new_infection": (pct(41.0), pct(16.0), pct(43.0)),
}
# S-B11 annual incidence among HIV-negative, by calendar year. ASSUMPTION: held
# at the 2024 value after 2024 ("extrapolated thereafter", T1 footnote d).
HIV_INCIDENCE_ANNUAL_IDU = {
    2018: pct(0.1198), 2019: pct(0.1198), 2020: pct(0.0980), 2021: pct(0.1167),
    2022: pct(0.1182), 2023: pct(0.1120), 2024: pct(0.1049),
}
HIV_INCIDENCE_ANNUAL_NON_IDU = {
    2018: pct(0.0128), 2019: pct(0.0124), 2020: pct(0.0108), 2021: pct(0.0127),
    2022: pct(0.0134), 2023: pct(0.0140), 2024: pct(0.0139),
}
# S-B17: proportion tested in the past 12 months -> monthly diagnosis probability
HIV_TESTING_ANNUAL = {"idu": pct(45.0), "non_idu": pct(39.9)}
# S-B14/S-B18: ART start (within 30 d of diagnosis; monthly thereafter) and stop
HIV_ART = {
    "idu": {
        "start_at_diagnosis": {"low": pct(52.0), "medium": pct(30.0), "high": pct(22.2)},
        "start_monthly": {"low": pct(3.2), "medium": pct(1.0), "high": pct(0.3)},
        "stop_monthly": {"low": pct(2.0), "medium": pct(1.3), "high": pct(0.9)},
    },
    "non_idu": {
        "start_at_diagnosis": {"low": pct(53.4), "medium": pct(36.1), "high": pct(25.8)},
        "start_monthly": {"low": pct(2.2), "medium": pct(1.4), "high": pct(0.7)},
        "stop_monthly": {"low": pct(1.5), "medium": pct(1.0), "high": pct(0.8)},
    },
}
# S-B16 monthly CD4 progression: {(from, to): p}
CD4_TRANSITIONS = {
    "off_art": {("high", "medium"): pct(1.24), ("medium", "low"): pct(7.00)},
    "on_art_idu": {
        ("high", "medium"): pct(1.38), ("high", "low"): pct(0.01),
        ("medium", "low"): pct(0.75), ("medium", "high"): pct(1.62),
        ("low", "medium"): pct(0.03), ("low", "high"): pct(2.73),
    },
    "on_art_non_idu": {
        ("high", "medium"): pct(1.13), ("high", "low"): pct(0.01),
        ("medium", "low"): pct(0.63), ("medium", "high"): pct(1.63),
        ("low", "medium"): pct(0.03), ("low", "high"): pct(2.69),
    },
}
# S-B21 monthly probability of death from HIV
HIV_DEATH_P = {
    "on_art": {"high": pct(0.0013), "medium": pct(0.0024), "low": pct(0.0102)},
    "off_art": {"high": pct(0.0033), "medium": pct(0.0155), "low": pct(0.1936)},
}
# S-B20 relative risk of liver-fibrosis progression with HIV
HCV_FIBROSIS_RR_HIV = {"on_art": 1.72, "off_art": 2.49}

# --------------------------------------------------------------------------
# HCV (T1; S-B part 5)
# --------------------------------------------------------------------------

FIBROSIS_STATES = ["F0", "F1", "F2", "F3", "F4", "DC", "HCC"]
HCV_PREVALENCE_2018 = pct(43.5)  # S-B22 (IDU)
# S-B23 annual incidence among susceptible. ASSUMPTION: held at 2023 value after.
HCV_INCIDENCE_ANNUAL_IDU = {
    2018: pct(1.23), 2019: pct(1.35), 2020: pct(1.43),
    2021: pct(1.35), 2022: pct(1.32), 2023: pct(1.28),
}
HCV_INCIDENCE_ANNUAL_NON_IDU = pct(0.01)  # S-B23
# S-B24 initial fibrosis distribution among people with HCV
HCV_INITIAL_FIBROSIS = {
    "F0": pct(39.8), "F1": pct(39.8), "F2": pct(7.6), "F3": pct(7.6),
    "F4": pct(3.0), "DC": pct(1.5), "HCC": pct(0.7),
}
# S-B25 annual fibrosis progression; converted to monthly at use
HCV_PROGRESSION_ANNUAL = {
    ("F0", "F1"): pct(11.7), ("F1", "F2"): pct(8.5), ("F2", "F3"): pct(12.0),
    ("F3", "F4"): pct(11.6), ("DC", "HCC"): pct(6.8),
}
HCV_F4_PROGRESSION_ANNUAL = {
    "with_hcv": {("F4", "DC"): pct(3.9), ("F4", "HCC"): pct(1.4)},
    "without_hcv": {("F4", "DC"): pct(0.8), ("F4", "HCC"): pct(0.5)},
}
# S-B26 annual mortality from HCV-related liver disease
HCV_DEATH_ANNUAL = {"DC_first_year": pct(18.2), "DC_after": pct(11.2), "HCC": pct(42.7)}
# C23 Table 1: monthly probability of spontaneous clearance of acute HCV (2.4%,
# Grebely et al.). ASSUMPTION: applied during the first 12 months of infection,
# which yields the ~25% one-year clearance proportion in Grebely et al.
HCV_SPONTANEOUS_CLEARANCE_MONTHLY = pct(2.4)
HCV_ACUTE_MONTHS = 12
# No HCV treatment in the model (no DAA inputs or costs in T1/Supplement).

# --------------------------------------------------------------------------
# Background mortality (S-B27): monthly probability by single-year age and sex
# --------------------------------------------------------------------------

BACKGROUND_MORTALITY = pd.read_csv(DATA_DIR / "background_mortality_monthly.csv").set_index("age")

# --------------------------------------------------------------------------
# Costs, 2025 USD (T1; S-B28 - S-B31)
# --------------------------------------------------------------------------

COST_DRUG_USE_MONTHLY = {"high": 1791.10, "medium": 1343.33, "low": 447.78, "remission": 764.53}
COST_IDU_MONTHLY = 115.30  # S-B28, average cost per syringe; active IDU states only
COST_OVERDOSE_DEATH = 7173.71
COST_SBI_HOSPITALISATION = {"endocarditis": 47_085.0, "osteomyelitis": 30_255.0, "cns": 44_719.0, "ssti": 12_839.0}
COST_HIV_MONTHLY_BY_CD4 = {"high": 708.39, "medium": 1367.61, "low": 5497.29}
COST_ART_MONTHLY = 4806.69
COST_HCV_MONTHLY = {"F0": 85.92, "F1": 85.92, "F2": 85.92, "F3": 176.52, "F4": 219.39, "DC": 2290.21, "HCC": 4207.66}

# --------------------------------------------------------------------------
# Published results to compare against (Table 2; S-C1..C3), base case
# --------------------------------------------------------------------------

PAPER_TABLE2 = {
    "new_infections": {"sbi": 1_714_773, "hiv": 29_715, "hcv": 218_186},
    "deaths": {"overdose": 348_650, "sbi": 37_658, "hiv": 10_455, "hcv": 90_486, "other": 179_966, "total": 667_216},
    "costs_billion": {"drug_use": 430.5, "sbi": 24.0, "hiv": 77.2, "hcv": 33.8, "total": 565.4},
    "pwid_2025": 3_300_000,
    "per_person_annual_cost": 17_375,
    "life_years_per_person": 6.4,
}

PAPER_ANNUAL = pd.DataFrame(
    {
        "year": range(2025, 2035),
        "new_sbi": [159916, 164195, 167394, 169038, 172190, 175710, 175607, 176394, 177332, 176996],
        "new_hiv": [3303, 3314, 3203, 3281, 2945, 2900, 2730, 2797, 2579, 2664],
        "new_hcv": [23844, 23767, 23309, 22692, 22352, 21506, 21206, 20353, 20161, 18997],
        "deaths_overdose": [32988, 33428, 33720, 34230, 35223, 35308, 35526, 36154, 36044, 36029],
        "deaths_sbi": [3414, 3606, 3488, 3887, 3835, 3813, 3927, 3905, 4020, 3765],
        "deaths_hiv": [1123, 1134, 1164, 1156, 953, 1123, 1023, 887, 957, 935],
        "deaths_hcv": [6240, 6698, 7415, 8265, 8819, 9288, 9994, 10781, 11180, 11808],
        "deaths_other": [14095, 15040, 15639, 16263, 16943, 18376, 19019, 20431, 21768, 22392],
        "cost_drug_use": [53477828900, 50989022607, 48564716610, 46206389002, 43913530931,
                          41680455350, 39513126728, 37409579154, 35368690872, 33402477676],
        "cost_sbi": [2555767848, 2520841478, 2500371847, 2476858814, 2434129936,
                     2409823197, 2332393471, 2288574456, 2228078365, 2152986856],
        "cost_hiv": [9510983033, 9123269960, 8731310162, 8330872705, 7927960361,
                     7499365848, 7097492379, 6728510488, 6328739133, 5961643568],
        "cost_hcv": [3698240033, 3623451885, 3549679884, 3476447705, 3409835525,
                     3346335792, 3277183291, 3207404435, 3131014161, 3042577930],
    }
).set_index("year")

PAPER_SCENARIO_DELTAS = {  # vs base case over 2025-2034 (Results text)
    "sustained_decline": {"deaths": -158_169, "new_infections": +36_679},
    "precipitous_increase": {"deaths": +84_744, "new_infections": -33_443},
}
