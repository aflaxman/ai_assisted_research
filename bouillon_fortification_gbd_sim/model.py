"""Tier 1 PAF-shift model for a national MMN-fortified bouillon program in Ghana.

Every function takes plain floats from data/gbd_anchors.csv so that provisional
anchors can be swapped for draw-level GBD 2023 pulls without touching the code.
"""

import numpy as np
from scipy import stats


def paf(prevalence, rr):
    """Population attributable fraction for a dichotomous exposure."""
    return prevalence * (rr - 1) / (prevalence * (rr - 1) + 1)


def dalys_averted_dichotomous(prev0, coverage, efficacy, rr, cause_dalys):
    """DALYs averted when fortification resolves deficiency in a share of the
    covered deficient population (vitamin A, zinc pathways)."""
    prev1 = prev0 * (1 - coverage * efficacy)
    return cause_dalys * (paf(prev0, rr) - paf(prev1, rr))


def cases_averted(prev0, population, coverage, efficacy):
    """Deficiency cases resolved (used for VA, zinc, and the B12 endpoint)."""
    return population * prev0 * coverage * efficacy


def anemia_reduction(hb_mean, hb_sd, dhb, coverage, iron_responsive_share,
                     threshold=120.0):
    """Relative reduction in anemia prevalence from a hemoglobin shift.

    Models WRA hemoglobin as Normal(hb_mean, hb_sd); covered iron-responsive
    women shift up by dhb g/L. Returns the fraction of baseline anemia
    prevalence averted.
    """
    p0 = stats.norm.cdf(threshold, hb_mean, hb_sd)
    p_shifted = stats.norm.cdf(threshold, hb_mean + dhb, hb_sd)
    affected = coverage * iron_responsive_share
    p1 = affected * p_shifted + (1 - affected) * p0
    return (p0 - p1) / p0


def maternal_dalys_averted(daly_maternal, dhb, coverage, rr_per_10gL):
    """Maternal-disorder DALYs averted from a pregnancy hemoglobin shift,
    treating the GBD hemoglobin RR as log-linear in Hb."""
    rr_shift = rr_per_10gL ** (dhb / 10.0)
    return daly_maternal * coverage * (1 - 1 / rr_shift)


def neonatal_dalys_averted(neonatal_deaths, lbw_attrib_share, rr_lbw_supp,
                           dose_ratio, coverage, yll_per_death):
    """Extended iron pathway (c): maternal iron -> LBW -> neonatal deaths.

    Scales the supplementation LBW risk ratio down to the bouillon dose on the
    log scale, then applies it to the LBW-attributable share of neonatal
    deaths among covered pregnancies.
    """
    rr_scaled = np.exp(np.log(rr_lbw_supp) * dose_ratio)
    deaths_averted = (neonatal_deaths * lbw_attrib_share
                      * coverage * (1 - rr_scaled))
    return deaths_averted * yll_per_death, deaths_averted


def ntd_dalys_averted(births, ntd_prev, preventable_frac, coverage,
                      daly_per_birth):
    """Folic acid pathway: neural tube defect births averted."""
    births_averted = births * ntd_prev * preventable_frac * coverage
    return births_averted * daly_per_birth, births_averted
