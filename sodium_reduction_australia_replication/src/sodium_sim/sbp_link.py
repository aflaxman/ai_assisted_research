"""Sodium to systolic blood pressure link, from Table S3 of the paper's appendix.

Dhungel et al. refit the Huang et al. (2020) meta-regression on the 48 trials
with a sodium change of at most 100 mmol/day lasting more than 14 days, as an
interaction model with no intercept. Per gram of sodium removed, the SBP
reduction in mmHg is

    2.026 + 0.502 (age - 50)/10 + 0.659 (SBP - 140)/10 + 2.353 p_nonwhite,

with p_nonwhite set to zero for Australia and the result floored at zero
(Table S5 shows 0.00 for young people with low SBP). Following Trieu et al.,
the appendix sets a theoretical minimum of 2 g/day for this link: only sodium
above 2 g/day moves SBP, so a reduction that takes intake below 2 g/day counts
only down to that floor.

Before the appendix was available this module used slopes rescaled from the
published Huang multivariable regression (0.371 per decade of age and 0.248
per 10 mmHg); those were 1.35 and 2.7 times too small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SBP_PER_GRAM_AT_REFERENCE = 2.026
REFERENCE_AGE = 50.0
REFERENCE_SBP = 140.0
AGE_SLOPE = 0.502 / 10
SBP_SLOPE = 0.659 / 10
SODIUM_FLOOR_G = 2.0


def sbp_reduction_per_gram(age: pd.Series | np.ndarray, sbp: pd.Series | np.ndarray):
    """Return the SBP reduction (mmHg) per gram/day of sodium removed."""
    value = (
        SBP_PER_GRAM_AT_REFERENCE
        + AGE_SLOPE * (np.asarray(age, dtype=float) - REFERENCE_AGE)
        + SBP_SLOPE * (np.asarray(sbp, dtype=float) - REFERENCE_SBP)
    )
    return np.clip(value, 0.0, None)


def effective_sodium_reduction(sodium_before, sodium_after):
    """Return the part of a sodium reduction that lies above the 2 g/day floor."""
    before = np.clip(np.asarray(sodium_before, dtype=float) - SODIUM_FLOOR_G, 0.0, None)
    after = np.clip(np.asarray(sodium_after, dtype=float) - SODIUM_FLOOR_G, 0.0, None)
    return before - after


def sbp_reduction(sodium_before, sodium_after, age, sbp):
    """Return the SBP reduction (mmHg) for a change in sodium intake in g/day."""
    return effective_sodium_reduction(sodium_before, sodium_after) * sbp_reduction_per_gram(age, sbp)
