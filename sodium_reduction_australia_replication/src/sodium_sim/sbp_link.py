"""Sodium to systolic blood pressure link.

Dhungel et al. refit the Huang et al. (2020) meta-regression on 48 trials as
an interaction model with no intercept, equation (1) of the paper, and report
one calibration point: a 1 g/day sodium reduction for a 50-year-old white
person with SBP 140 mmHg lowers SBP by 2.026 mmHg. The supplement with the
fitted coefficients is not available to me, so I reconstruct the age and
baseline-SBP slopes from the published multivariable meta-regression of
Huang et al. (BMJ 2020;368:m315, Table 1): 0.111 mmHg more reduction per
year of age and 0.074 per mmHg of baseline SBP, at the mean sodium reduction
of 130 mmol/day in those trials. Dividing by 130 mmol and multiplying by
43.5 mmol per gram gives per-gram interaction slopes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MMOL_PER_GRAM_SODIUM = 1000 / 22.99
HUANG_MEAN_REDUCTION_MMOL = 130.0

SBP_PER_GRAM_AT_REFERENCE = 2.026
REFERENCE_AGE = 50.0
REFERENCE_SBP = 140.0
AGE_SLOPE = 0.111 / HUANG_MEAN_REDUCTION_MMOL * MMOL_PER_GRAM_SODIUM
SBP_SLOPE = 0.074 / HUANG_MEAN_REDUCTION_MMOL * MMOL_PER_GRAM_SODIUM


def sbp_reduction_per_gram(age: pd.Series | np.ndarray, sbp: pd.Series | np.ndarray):
    """Return the SBP reduction (mmHg) per gram/day of sodium removed."""
    value = (
        SBP_PER_GRAM_AT_REFERENCE
        + AGE_SLOPE * (np.asarray(age, dtype=float) - REFERENCE_AGE)
        + SBP_SLOPE * (np.asarray(sbp, dtype=float) - REFERENCE_SBP)
    )
    return np.clip(value, 0.0, None)


def sbp_reduction(sodium_reduction_grams, age, sbp):
    """Return the SBP reduction (mmHg) for an absolute sodium reduction in g/day."""
    return np.asarray(sodium_reduction_grams, dtype=float) * sbp_reduction_per_gram(age, sbp)
