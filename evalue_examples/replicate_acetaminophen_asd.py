"""Replicate the acetaminophen/ASD E-value example from Andrade (2026).

Andrade's tutorial (J Clin Psychiatry 2026;87(1):26f16324) computes
E-values for the association between gestational acetaminophen exposure
and autism spectrum disorder reported by Ahlqvist et al. (JAMA
2024;331(14):1205-1214). The individual-level Swedish register data are
not public, but the JAMA paper is open access (PMC11004836) and its
published summary statistics are all this calculation needs. They live
in data/ahlqvist2024_jama_*.csv.

Run: python replicate_acetaminophen_asd.py
"""

import csv
from pathlib import Path

from evalue import evalue, evalue_ci

DATA = Path(__file__).parent / "data"

# Values as published in the tutorial, for comparison.
ANDRADE_POINT = 1.28
ANDRADE_CI = 1.16


def load_rows(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def main():
    estimates = {r["analysis"]: r for r in load_rows("ahlqvist2024_jama_estimates.csv")}
    cohort = {r["group"]: r for r in load_rows("ahlqvist2024_jama_cohort.csv")}

    print("Replication of Andrade (2026) acetaminophen/ASD E-value example")
    print("Source data: Ahlqvist et al. JAMA 2024 (open access, PMC11004836)")
    print()

    full = estimates["full_cohort_adjusted"]
    hr, lo, hi = (float(full[k]) for k in ("estimate", "ci_lower", "ci_upper"))
    e_point = evalue(hr)
    e_ci = evalue_ci(hr, lo, hi)
    print(f"Full cohort, adjusted: HR {hr} (95% CI {lo}-{hi})")
    print(f"  E-value (point):    {e_point:.2f}  [tutorial reports {ANDRADE_POINT}]")
    print(f"  E-value (CI bound): {e_ci:.2f}  [tutorial reports {ANDRADE_CI}]")
    assert round(e_point, 2) == ANDRADE_POINT
    assert round(e_ci, 2) == ANDRADE_CI
    print("  MATCH: both values reproduce the tutorial exactly.")
    print()

    # Crude association, reconstructed from the published 10-year
    # cumulative incidences and group sizes.
    risk = {g: float(r["autism_cum_incidence_10yr_pct"]) / 100 for g, r in cohort.items()}
    n = {g: int(r["n_children"]) for g, r in cohort.items()}
    rr_crude = risk["exposed"] / risk["unexposed"]
    print(f"Crude 10-year cumulative incidence: {risk['exposed']:.2%} exposed "
          f"(n={n['exposed']:,}) vs {risk['unexposed']:.2%} unexposed (n={n['unexposed']:,})")
    print(f"  Crude RR: {rr_crude:.2f}   E-value: {evalue(rr_crude):.2f}")
    print()

    # The sibling-control analysis, which Andrade's article does not
    # cover, shows why the small E-value mattered: familial confounding
    # of modest strength was enough to explain the association away.
    sib = estimates["sibling_control"]
    hr_s, lo_s, hi_s = (float(sib[k]) for k in ("estimate", "ci_lower", "ci_upper"))
    print(f"Sibling-control analysis: HR {hr_s} (95% CI {lo_s}-{hi_s})")
    print(f"  E-value (point):    {evalue(hr_s):.2f}")
    print(f"  E-value (CI bound): {evalue_ci(hr_s, lo_s, hi_s):.2f} "
          "(CI includes 1: nothing to explain away)")
    print()
    print("Interpretation: an unmeasured confounder associated with both")
    print("exposure and outcome by a risk ratio of just 1.28 could nullify")
    print("the full-cohort association. Sibling controls, which absorb")
    print("shared family factors, did exactly that (HR 0.98).")


if __name__ == "__main__":
    main()
