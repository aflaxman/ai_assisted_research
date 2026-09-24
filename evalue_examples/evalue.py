"""E-values for unmeasured confounding (VanderWeele & Ding 2017).

The E-value is the minimum strength of association, on the risk ratio
scale, that an unmeasured confounder would need to have with *both* the
exposure and the outcome to fully explain away an observed
exposure-outcome association.

E = RR + sqrt(RR * (RR - 1))

Hazard ratios, and odds ratios when the outcome is uncommon (<15%), can
be treated as risk ratios in this formula.

References:
- VanderWeele TJ, Ding P. Sensitivity analysis in observational research:
  introducing the E-value. Ann Intern Med. 2017;167(4):268-274.
- Ding P, VanderWeele TJ. Sensitivity analysis without assumptions.
  Epidemiology. 2016;27(3):368-377.
- Online calculator: https://www.evalue-calculator.com/
"""

import math


def evalue(rr):
    """E-value for a risk ratio point estimate.

    Protective estimates (RR < 1) are inverted before applying the
    formula, so evalue(0.5) == evalue(2.0).
    """
    if rr <= 0:
        raise ValueError("risk ratio must be positive")
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))


def evalue_ci(estimate, lo, hi):
    """E-value for the confidence-interval limit closer to the null.

    Returns 1.0 when the interval includes the null: no unmeasured
    confounding at all is needed to shift the interval across 1.
    """
    if lo > hi:
        raise ValueError("lower CI bound exceeds upper")
    if lo <= 1 <= hi:
        return 1.0
    return evalue(lo if estimate >= 1 else hi)


def bias_factor(rr_eu, rr_ud):
    """Ding & VanderWeele bounding factor B for a confounder of known strength.

    rr_eu: how much more prevalent the confounder is among the exposed
    than the unexposed (risk ratio scale).
    rr_ud: risk ratio for the outcome across confounder levels.

    Confounding by a variable of this strength can inflate an observed
    risk ratio by at most B; equivalently, the true causal RR is at
    least RR_observed / B. A confounder can fully explain away an
    observed RR exactly when B >= RR_observed, and the E-value is the
    smallest E with bias_factor(E, E) == RR_observed.
    """
    if rr_eu < 1 or rr_ud < 1:
        raise ValueError("express confounder associations as ratios >= 1")
    return rr_eu * rr_ud / (rr_eu + rr_ud - 1)
