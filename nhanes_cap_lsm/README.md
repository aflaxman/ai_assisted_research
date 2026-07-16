# NHANES 2017–2020: liver stiffness by CAP steatosis threshold

Compares the **liver stiffness measure (LSM, kPa)** distribution between
participants with **CAP < 288 dB/m** and **CAP ≥ 288 dB/m** in the NHANES
2017–March 2020 pre-pandemic Liver Ultrasound Transient Elastography
(FibroScan) component. CAP ≥ 288 dB/m is a common cutoff for hepatic
steatosis, so the comparison contrasts liver stiffness in those with vs.
without significant steatosis.

## Data

| Variable | NHANES var | File |
|----------|-----------|------|
| LSM (median stiffness, kPa) | `LUXSMED` | `P_LUX` |
| CAP (median, dB/m) | `LUXCAPM` | `P_LUX` |
| Exam status | `LUAXSTAT` | `P_LUX` |
| Age (years) | `RIDAGEYR` | `P_DEMO` |

**Inclusion:** complete elastography exam (`LUAXSTAT == 1`) with non-missing
LSM and CAP → **n = 9,021** (ages 12–80). Files merged on `SEQN`.

**Survey weights:** all densities, KDEs, and prevalences are weighted with the
2017–2020 pre-pandemic MEC exam weight **`WTMECPRP`** — the correct weight for
the MEC-based elastography component. Point estimates therefore represent the
US population aged 12+ (the weights sum to ≈241 million). Weighting is not
cosmetic here: e.g., F4 below the CAP threshold is 0.5% weighted vs. 0.9%
unweighted, and F0 below threshold is 81.8% vs. 79.8%.

**Fibrosis staging** (mutually-exclusive bins on LSM, per request):

| Stage | LSM (kPa) |
|-------|-----------|
| F0 | < 6 |
| F1 | 6 – 8 |
| F2 | 8 – 10 |
| F3 | 10 – 15 |
| F4 | ≥ 15 |

## Run

```bash
uv run python analysis.py   # auto-downloads P_LUX.xpt / P_DEMO.xpt on first run
```

## Outputs

- `fig1_lsm_overall.png` — LSM histogram + KDE, CAP < 288 stacked above CAP ≥ 288 (all ages), with F1–F4 threshold lines.
- `fig2_lsm_smallmultiples.png` — the same stacked comparison faceted into four age groups (12–29, 30–44, 45–59, 60–80).
- `fig3_fibrosis_prevalence_by_age.png` — F1–F4 prevalence vs. age (5-year groups), one panel per stage, comparing above vs. below the CAP 288 threshold.
- `stage_prevalence_by_cap.csv`, `stage_prevalence_by_age_cap.csv` — underlying tables.

## Key findings

Stage prevalence by CAP group, all ages (**survey-weighted**, %; unweighted in
parentheses):

| Stage | CAP < 288 | CAP ≥ 288 |
|-------|-----------|-----------|
| F0 | 81.8 (79.8) | 55.9 (55.1) |
| F1 | 14.2 (15.3) | 25.6 (26.3) |
| F2 | 2.3 (2.5) | 7.5 (8.1) |
| F3 | 1.2 (1.4) | 6.8 (6.8) |
| F4 | 0.5 (0.9) | 4.2 (3.7) |

- The CAP ≥ 288 group's LSM distribution is shifted right at every age group; weighted median LSM 5.6 vs. 4.6 kPa (mean 7.0 vs. 5.0).
- Advanced-fibrosis prevalence (F3, F4) is roughly **5–8× higher** above the CAP threshold and rises steeply with age.

## Caveats

- **Point estimates only.** Estimates are survey-weighted (`WTMECPRP`) and so
  are population-representative, but design-based **standard errors / CIs are
  not yet computed**. Proper variance needs the design variables (`SDMVPSU`,
  `SDMVSTRA`, loaded but unused) via Taylor linearization — a natural next
  step, especially for the noisy small cells in `fig3`.
- Histograms truncate the x-axis at 25 kPa for resolution; the 0.6% of
  participants with LSM > 25 kPa are omitted from the bars (the KDE still
  uses all data).
- In `fig3`, 5-year age × CAP cells with n < 25 *observations* are suppressed
  (suppression uses the unweighted count, not the weighted total); the
  youngest CAP ≥ 288 points still rest on small cells and are noisy.

---

# FIB-4 vs LSM-defined fibrosis (with / without CAP threshold)

`fib4_lsm_analysis.py` evaluates the diagnostic accuracy of **FIB-4** (a
lab-based fibrosis index) against the **LSM/FibroScan** fibrosis reference
standard, for the whole sample and restricted to the CAP ≥ 288 (steatosis)
population where FIB-4 is clinically applied.

## Method

- **Index test:** FIB-4 = (age × AST) / (platelets × √ALT). AST=`LBXSASSI`,
  ALT=`LBXSATSI` (P_BIOPRO), platelets=`LBXPLTSI` (10⁹/L, P_CBC). Cutoffs
  **1.30** (rule-out), **2.67** (rule-in), **3.25** (Sterling).
- **Reference standard (LSM):** ≥F2 = LSM≥8; **≥F3 = LSM≥10 (primary)**; F4 = LSM≥15.
- **Populations:** *without* CAP threshold (all, n=8,282); *with* CAP threshold
  (CAP≥288, n=2,623); plus CAP<288 (n=5,659) for contrast.
- **Survey-weighted** (`WTMECPRP`). 95% CIs are **design-based** — Taylor
  linearization of the domain ratio estimator over `SDMVSTRA`/`SDMVPSU`
  (with-replacement PSU approximation, logit-transformed; design df = 25).

## Headline results (primary target ≥F3, LSM ≥ 10 kPa; weighted)

| Population | AUROC | Sens @1.30 | Spec @1.30 | Sens @2.67 | Spec @2.67 |
|---|---|---|---|---|---|
| All (no CAP threshold) | 0.65 | 40% | 80% | 12% | 99% |
| CAP < 288 | 0.72 | 55% | 80% | 26% | 99% |
| CAP ≥ 288 (steatosis) | 0.59 | 35% | 77% | 7% | 99% |

- At the rule-in cutoff (≥2.67) FIB-4 is highly **specific (~99%)** but very
  **insensitive** — it misses ~88% of LSM-defined advanced fibrosis overall and
  ~93% within the steatosis group.
- Even the rule-out cutoff (1.30) leaves sensitivity at 35–55%, so a low FIB-4
  does not confidently exclude advanced fibrosis at the population level.
- FIB-4 discriminates **worse in CAP ≥ 288 steatosis** (AUROC 0.59) than below
  the threshold (0.72) — consistent with FIB-4's known weakness in MASLD, where
  platelets stay normal and ALT (in the denominator) is often elevated.
- These population-based, weighted operating characteristics are weaker than the
  clinic-based studies that defined the cutoffs (spectrum + prevalence effects).

## Outputs

- `fig4_fib4_roc.png` — weighted ROC, one panel per target, three populations overlaid.
- `fig5_fib4_sensspec.png` — sens & spec (95% design-based CI) for ≥F3 by cutoff & population.
- `fib4_sensspec.csv` — full table (population × target × cutoff: sens/spec/PPV/NPV/AUROC + CIs).
- `fib4_zones.csv` — FIB-4 risk-zone distribution (low / indeterminate / high) per population.

## Caveats

- **Complete-case:** 8,282 of 9,021 valid-elastography participants had all FIB-4
  labs (~8% dropped for missing AST/ALT/platelets). The dropped group skews
  slightly younger (mean age 40 vs 45) and lower-CAP, but the fibrosis outcome is
  balanced across kept vs. dropped (LSM≥10 in 4.9% vs 4.7%), so selection bias on
  the advanced-fibrosis endpoint is low.
- **Cohort includes minors (age 12+)**, matching the rest of this project, but
  FIB-4 is an adult tool; adolescents add low-FIB-4 / low-LSM true negatives.
- FIB-4's age term overlaps with age-related fibrosis risk; age-specific cutoffs
  (≥65 → 2.0) are not separately applied. LSM is itself an imperfect (non-biopsy)
  reference standard.

---

# FIB-4 vs LSM-fibrosis, stratified by alcohol consumption

`fib4_lsm_by_alcohol.py` re-runs the diagnostic-accuracy analysis with **alcohol
consumption** as the stratifier (adults 18+ only, since the alcohol questionnaire
`P_ALQ` is adult-only). Average ethanol is derived as
`(drinking days/yr from ALQ121) × (ALQ130 drinks/day) × 14 g ÷ 365`, then:

| Category | Definition | n | wt % | median g/day |
|---|---|---|---|---|
| None | no alcohol in past 12 months | 1,996 | 22.7% | 0 |
| Moderate | >0, below heavy | 4,453 | 70.2% | 1.4 |
| Heavy | ≥30 g/day (men) / ≥20 g/day (women) | 413 | 7.1% | 42 |

(348 adults with refused/missing intake were dropped. Read XPTs with `pyreadstat`,
not `pandas.read_sas` — this pandas build decodes `0` as ~5.4e-79, which would
break the `ALQ121 == 0` "none" test.)

## Headline results (advanced fibrosis ≥F3, LSM ≥ 10 kPa; weighted)

| Alcohol | AUROC | Sens @1.30 | Spec @1.30 | Sens @2.67 | Spec @2.67 |
|---|---|---|---|---|---|
| None | 0.57 | 37% | 65% | 15% | 98% |
| Moderate | 0.61 | 36% | 82% | 7% | 99% |
| **Heavy** | **0.92** | **90%** | 74% | **43%** | 97% |

**FIB-4 works far better in heavy drinkers.** AUROC jumps to ~0.9 for ≥F2/≥F3
versus ~0.6 in none/moderate. Mechanism (confirmed in the data): heavy drinkers
*with* advanced fibrosis show the alcoholic-injury pattern — median AST 62 (vs 22
in non-fibrotic heavy drinkers), platelets 176 (vs 242), median FIB-4 **2.78**,
already above the rule-in cutoff. Moderate-drinker fibrosis sits at FIB-4 ~1.2
with a metabolic AST/ALT ratio, so FIB-4 barely separates it. This is the
flip-side of the CAP finding: FIB-4 keys on exactly the derangement that
alcoholic (not metabolic) fibrosis produces.

## Outputs
- `fig6_fib4_roc_by_alcohol.png` — weighted ROC per target, three alcohol strata.
- `fig7_fib4_sensspec_by_alcohol.png` — sens & spec (95% CI) for ≥F3 by cutoff & alcohol.
- `fib4_sensspec_by_alcohol.csv` — full table (alcohol × target × cutoff + CIs).

### LSM distribution by CAP × alcohol (`lsm_dist_by_alcohol.py`)
Adds alcohol to the original LSM-distribution figures (adults 18+, weighted):
- `fig8_lsm_by_cap_alcohol.png` — 3×3 grid of histogram+KDE with F-stage lines:
  rows = without CAP threshold (All) and with it (CAP < 288 / ≥ 288);
  columns = none / moderate / heavy alcohol.
- `fig9_lsm_alcohol_overlay.png` — the three alcohol KDEs overlaid within each
  CAP stratum, for direct comparison.

Within each CAP stratum, alcohol shifts the LSM distribution only modestly
(heavy drinkers carry a slightly heavier right/F3–F4 tail) — the CAP threshold
moves the distribution far more than alcohol category does.

## Caveats
- **The heavy stratum is small** (n=413; only 20 with LSM≥10, 12 with LSM≥15), so
  its AUROC/sensitivity are imprecise — CIs are wide (e.g. heavy sens @1.30
  [60–98%]) and F4 sensitivity is suppressed (n<20). The *direction* is robust
  (two independent AUROC methods agree; mechanism is clear) but the magnitudes are
  uncertain.
- Alcohol intake is self-reported (recall/social-desirability bias, typically
  under-reporting → some true heavy drinkers misclassified as moderate).
- Alcohol stratum is **not** crossed with CAP here: heavy × CAP × fibrosis cells
  (~10 each) are too small for stable estimates.

---

# Gaussian vs. empirical copula for (CAP, LSM)

`cap_lsm_copula.py` is a diagnostic to help choose between a **Gaussian copula**
and an **empirical copula** for the joint distribution of median CAP and LSM.
It **pools two cycles** — NHANES 2017–2020 pre-pandemic + August 2021–August
2023 (`_L`) — for **n = 15,300** valid elastography exams, survey-weighted. It
scatters the data and overlays the density a Gaussian copula would give, in two
coordinate systems:

- `fig10_copula_probit.png` — **probit-of-percentile space**: each variable
  mapped to `Φ⁻¹(weighted percentile)`. A Gaussian copula is then bivariate
  normal `N(0, [[1,ρ],[ρ,1]])`, so its probability-region contours are ellipses.
- `fig11_copula_capunits.png` — **CAP / LSM units** (LSM on a **log** axis to
  spread the crowded low-LSM mode): the same model mapped back,
  `f(x,y) = c_ρ(F_X(x),F_Y(y))·f_X(x)·f_Y(y)` with empirical (KDE) margins.
- `fig12_copula_slide.png` — **presentation version** of fig11: 16:9, large
  fonts, LSM capped at **15 kPa** (all of F3 visible), shaded/labeled
  fibrosis-stage bands (F0–F3), and the **survey-weighted share of participants
  in each CAP-side × fibrosis-band region** shown as boxed %.

Both overlay the empirical joint density (2-D weighted KDE — what an empirical
copula reproduces) as dashed contours at matching probability-mass levels.

**Pooling & weights.** The two cycles have different weight bases (`WTMECPRP`,
3.2 yr; `WTMEC2YR`, 2.0 yr). Per NHANES guidance for combining unequal-length
cycles, the pooled weight is `cycle MEC weight × (cycle years / 5.2)`
(`WTMECPRP×3.2/5.2` and `WTMEC2YR×2/5.2`); the pooled weights sum to ~243 M (one
population, not double-counted). (The other analyses in this project still use
2017–2020 only — this pooling is applied to the copula analysis.)

Weighted region shares (fig12): F0 55.9 / 17.0, F1 9.6 / 8.0, F2 1.7 / 2.5,
F3 1.0 / 2.3, F4 0.5 / 1.6 (% CAP<288 / CAP≥288). CAP≥288 (steatosis) ≈ 31%.

## What the diagnostic says (pooled)

| Quantity | Value |
|---|---|
| Gaussian-copula ρ (normal scores) | 0.343 |
| Weighted Spearman (empirical) | 0.331 |
| Gaussian-copula-implied Spearman | 0.330 |

Overall association is captured well (implied vs. empirical Spearman 0.330 vs.
0.331). But the **tails are asymmetric**, which a Gaussian copula (radially
symmetric, asymptotically tail-independent) cannot represent:

| P(both extreme \| one extreme) | Empirical | Gaussian |
|---|---|---|
| Upper, q=0.95 (both high) | **0.28** | 0.16 |
| Lower, q=0.95 (both low) | **0.11** | 0.16 |

High CAP and high LSM co-occur **more** than Gaussian predicts (advanced MASLD
clusters in the upper-right corner), while the lower tail co-occurs **less**.
The Gaussian model's upper- and lower-tail concordance are identical by
construction (0.16 = 0.16); the data are not. Robust to weighting (unweighted
upper q=0.95 ≈ 0.21 vs. Gaussian 0.16).

## Recommendation
- If you only need the **overall rank correlation** (e.g., a rough joint
  simulation), the Gaussian copula is adequate and simple (one parameter).
- If **co-extremes matter** — jointly identifying people with both severe
  steatosis and advanced fibrosis, or anything tail-sensitive — prefer the
  **empirical copula** (or an asymmetric parametric one, e.g. a survival/rotated
  Clayton or Gumbel with upper-tail dependence). The Gaussian will
  under-represent that upper-right cluster.

**Caveat:** CAP (integer dB/m) and LSM (0.1 kPa) are recorded coarsely, so the
margins have many ties. The plotted points are jittered within recording
resolution (CAP ±0.5, LSM ±0.05) for display only — the fit and contours use the
raw data, and the mid-rank transform handles ties for the point estimates. An
empirical copula *fit* should likewise jitter/break ties (or model the
discreteness) to avoid artifacts.

## Does the relationship vary by age and sex?

`cap_lsm_copula_by_agesex.py` re-does the diagnostic in probit-of-percentile
space **per subgroup** (each subgroup's own marginals removed), so panels differ
only in *dependence*, not in CAP/LSM levels.

- `fig13_copula_probit_by_agesex.png` — 2 sex rows × 4 age columns; ellipses
  (Gaussian ρ per panel) vs empirical KDE, with per-panel ρ and n.
- `fig14_copula_rho_by_agesex.png` — Gaussian-copula ρ vs age, by sex, with
  approximate 95% CIs (Fisher z, Kish n_eff; ignores clustering).

**Age matters; sex barely does.** The copula ρ traces an inverted-U with age —
weakest in the young and strongest in middle age — identically for both sexes:

| Age | Male ρ | Female ρ |
|---|---|---|
| 12–29 | 0.22 | 0.25 |
| 30–44 | 0.36 | 0.33 |
| 45–59 | **0.40** | **0.35** |
| 60–80 | 0.30 | 0.30 |

The 12–29 vs 45–59 CIs don't overlap (≈0.16–0.28 vs 0.33–0.46), so the age
gradient is real, not noise; the male/female tracks overlap at every age.
Interpretation: in the young, high CAP (steatosis) rarely coincides with high
LSM (fibrosis takes years to develop), so CAP and LSM are only weakly linked;
by middle age steatosis has had time to drive fibrosis, tightening the coupling;
in the elderly other fibrosis causes and survivorship loosen it again. (CAP/LSM
*levels* also rise with age — a marginal effect — but that is removed here.) The
empirical-vs-Gaussian upper-tail gap persists across subgroups, most visibly in
the middle-aged and older panels.
