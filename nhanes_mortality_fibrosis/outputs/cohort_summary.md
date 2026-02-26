# NHANES Mortality by Liver Fibrosis Status

### Cohort selection rationale

| Cohort | Max FU (months) | % ≥36m | % ≥24m | UCOD detail | Elastography |
|--------|----------------|--------|--------|-------------|-------------|
| 2007-2008 | ~159 | ~96% | ~98% | Full (10 groups) | No |
| 2011-2012 | ~113 | ~97% | ~98% | Full (10 groups) | No |
| 2017-2018 | ~37  | ~2%  | ~51% | Coarsened (3 groups) | Yes (VCTE) |

### Key findings

**Unmatched analysis:**
- FIB-4-defined fibrosis consistently associated with 13–25× higher crude mortality
- After age/sex adjustment (Poisson), the association persists (IRR 1.8–7.3×)
- LSM-based fibrosis (2017-2018) shows age/sex-adjusted IRR ~2.4×

**PS-matched analysis (age, sex, BMI, SBP, LDL-C, FPG, smoking):**
- Matching restricted to the fasting subsample (~30–40% of cohort) with complete covariates
- After matching, fibrosis still associated with elevated mortality, though smaller sample sizes
  widen confidence intervals
- The Kaplan-Meier curves show clear separation in the unmatched analysis;
  matched curves attenuate but fibrosis+ survival remains lower

### Limitations
1. **Fibrosis is proxy-defined** — FIB-4 and LSM are not histological diagnoses
2. **FIB-4 includes age** — crude RR overstates the association; matched/adjusted estimates more reliable
3. **Public-use COD coarsening** — Only 3 UCOD groups available for 2017-2018
4. **Short follow-up** for 2017-2018 (~51% have ≥24 months)
5. **Fasting subsample** — LDL-C and FPG restrict matching to ~1/3 of the cohort
6. **Small matched samples** — Many cells have <10 events
7. **No survey weights** — Unweighted estimates; not nationally representative
