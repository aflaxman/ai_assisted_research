# NHANES Mortality-Fibrosis Analysis: Cohort Summary

## Cohort Selection

Three NHANES cohorts were selected to illustrate tradeoffs in follow-up time
and cause-of-death detail available in the public-use Linked Mortality Files:

| Cohort | Follow-up (max months) | % with >=36m | % with >=24m | UCOD detail | Elastography |
|--------|----------------------|-------------|-------------|-------------|-------------|
| 2007-2008 | ~159 | ~96% | ~98% | Full (10 groups) | No |
| 2011-2012 | ~113 | ~97% | ~98% | Full (10 groups) | No |
| 2017-2018 | ~37 | ~2% | ~51% | Coarsened (3 groups) | Yes (VCTE) |

**Rationale:**
- 2007-2008 provides the longest follow-up and full cause-of-death detail.
- 2011-2012 also has full detail with slightly shorter follow-up.
- 2017-2018 uniquely includes transient elastography (liver stiffness) but has
  limited follow-up (censored Dec 31, 2019) and coarsened UCOD to only 3 groups
  (heart disease, malignant neoplasms, all other).

## Analysis Windows

- **36-month window**: Used for 2007-2008 and 2011-2012 (>96% have >=36m).
- **24-month harmonized window (H=24)**: Used for cross-cohort comparison.
  For 2017-2018, only ~51% have >=24 months follow-up, so results reflect
  the subset with sufficient observation time.

## Fibrosis Definitions

### A) FIB-4 (all cohorts)
- FIB-4 = (age x AST) / (platelets x sqrt(ALT))
- Fibrosis+ (advanced fibrosis proxy): FIB-4 >= 2.67
- Fibrosis- (low fibrosis): FIB-4 < 1.30
- Indeterminate (1.30-2.67): excluded from primary binary comparison

### B) Liver Stiffness (2017-2018 only)
- Two cutpoint sets applied to liver stiffness median (kPa):
  - Castera/EASL: F0-F1 <7.1, F2 7.1-9.5, F3 9.5-12.5, F4 >=12.5
  - Eddowes/NAFLD: F0-F1 <8.2, F2 8.2-9.7, F3 9.7-13.6, F4 >=13.6
- Fibrosis+ = F3+F4, Fibrosis- = F0-F1 (F2 excluded as indeterminate)

## UCOD Cause-of-Death Detail

| UCOD Code | Label | 2007-2008 | 2011-2012 | 2017-2018 |
|-----------|-------|-----------|-----------|-----------|
| 1 | Heart disease | Yes | Yes | Yes |
| 2 | Malignant neoplasms | Yes | Yes | Yes |
| 3 | Chronic lower resp. | Yes | Yes | Suppressed |
| 4 | Accidents | Yes | Yes | Suppressed |
| 5 | Cerebrovascular | Yes | Yes | Suppressed |
| 6 | Alzheimer's | Yes | Yes | Suppressed |
| 7 | Diabetes | Yes | Yes | Suppressed |
| 8 | Influenza/pneumonia | Yes | Yes | Suppressed |
| 9 | Nephritis | Yes | Yes | Suppressed |
| 10 | All other causes | Yes | Yes | Yes |

Note: For 2015-2016 and 2017-2018, NCHS suppressed cause-group detail
in the public-use LMF, reporting only heart disease (1), malignant
neoplasms (2), and all other causes (10). This limits cause-specific
mortality analysis for recent cohorts.

## Main Findings

### All-Cause Mortality by Fibrosis Status

| Cohort | Window | Fibrosis Status | N | Deaths | Rate/1000 PY |
|--------|--------|-----------------|---|--------|-------------|
| 2007-2008 | 24m | fibrosis_yes | 172 | 22 | 68.3 |
| 2007-2008 | 24m | fibrosis_no | 3982 | 39 | 4.9 |
| 2007-2008 | 36m | fibrosis_yes | 172 | 31 | 66.1 |
| 2007-2008 | 36m | fibrosis_no | 3982 | 56 | 4.7 |
| 2011-2012 | 24m | fibrosis_yes | 215 | 23 | 56.5 |
| 2011-2012 | 24m | fibrosis_no | 3523 | 19 | 2.7 |
| 2011-2012 | 36m | fibrosis_yes | 215 | 36 | 60.7 |
| 2011-2012 | 36m | fibrosis_no | 3523 | 38 | 3.6 |
| 2017-2018 | 24m | fibrosis_yes | 189 | 26 | 86.6 |
| 2017-2018 | 24m | fibrosis_no | 3573 | 20 | 3.2 |

### Effect Estimates (FIB-4 based)

| Cohort | Window | Unadj RR (95% CI) | Poisson IRR unadj | Poisson IRR adj (age+sex) |
|--------|--------|-------------------|-------------------|--------------------------|
| 2007-2008 | 24m | 13.06 (7.92-21.53) | 13.25 (7.79-22.52) | 2.07 (1.06-4.01) |
| 2007-2008 | 36m | 12.82 (8.49-19.34) | 13.55 (8.7-21.11) | 1.83 (1.05-3.18) |
| 2011-2012 | 24m | 19.84 (10.98-35.85) | 20.0 (10.83-36.96) | 1.87 (0.8-4.38) |
| 2011-2012 | 36m | 15.52 (10.05-23.97) | 16.35 (10.33-25.87) | 1.84 (0.98-3.46) |
| 2017-2018 | 24m | 24.58 (13.98-43.21) | 28.24 (15.63-51.02) | 7.28 (3.33-15.89) |

### Cause-Group Mortality (where available)

See `outputs/tables/mortality_causegroup_by_cohort_window_fibrosisdef.csv` for full detail.
Note: Many cause-group cells have <10 events and are flagged as UNSTABLE.
For 2017-2018, only heart disease, cancer, and 'all other' are available.

## Limitations

1. **Fibrosis is proxy-defined.** FIB-4 is a non-invasive biomarker index,
   not a histological diagnosis. It has moderate sensitivity and specificity
   for advanced fibrosis. LSM from VCTE is also a proxy with known overlap
   between fibrosis stages.
2. **Public-use COD coarsening.** UCOD_LEADING is suppressed to 3 groups for
   2015-2016 and 2017-2018, preventing liver-specific cause-of-death analysis
   in the cohort with elastography.
3. **Short follow-up for 2017-2018.** With max ~37 months and only ~51% having
   >=24 months, the 2017-2018 cohort has few deaths and limited power.
4. **Small event counts.** Many subgroup analyses have <10 events, making
   rate estimates unstable. Results should be interpreted with caution.
5. **No survey weights in primary analysis.** Unweighted estimates are presented
   for simplicity; survey-weighted results would better represent the US population.
6. **Perturbed variables.** The public-use LMF includes perturbed death dates
   for disclosure avoidance, which may slightly affect person-time calculations.
7. **Confounding.** FIB-4 increases with age by construction (age is in the
   numerator), so crude comparisons partly reflect age differences. The age-adjusted
   Poisson models partially address this.
