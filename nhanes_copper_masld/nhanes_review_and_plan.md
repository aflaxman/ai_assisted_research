# Copper & MASLD

*NHANES · Dietary copper · Hepatic steatosis*

**Review of the August 2026 Stata analyses and a proposed analysis plan**

Reviewed 14 Aug 2026. Files: `Cu_CAP_Aug2026.log`, `CuNHANES_DexaFat_Aug2026.log`,
`CuNHANES_CAP_Notes.docx`.

> Transcribed on 2026-09-17 from the private Claude Artifact
> [Copper & MASLD](https://claude.ai/artifact/9SL27yht5LmW1kbRoo1Epv).
> Content unchanged apart from formatting.

## What the project is

The hypothesis, from the draft abstract: copper homeostasis is disturbed in
steatotic liver disease (SLD), and *lower dietary copper intake* may contribute
to hepatic steatosis. Two NHANES-based analyses test this:

- **Analysis 1 — CAP cohort.** Adults 18+ with transient elastography (single
  cycle, 15 strata / 30 PSUs; N = 4,137). Outcome: controlled attenuation
  parameter (`cap_score`); exposure: energy-adjusted dietary copper
  (`copper_per_1000kcal`) from two 24-hour recalls.
- **Analysis 2 — DXA cohort.** Combined earlier cycles (44 strata / 91 PSUs;
  ~2012–16 per notes). Outcomes: DXA visceral fat (`dxxvfatv`/`dxxvfata`), HSI,
  FIB-4; exposures: dietary copper (`cukcal`) and serum copper (`lbxscu`).

## What the logs show

### Analysis 1 — dietary copper and CAP

| Model | N | β (copper/1000 kcal) | SE | P |
|---|---:|---:|---:|---:|
| CAP ~ copper (crude) | 4,137 | **−15.8** | 5.29 | 0.009 |
| CAP ~ copper, restricted to CAP ≥ 248 | 2,317 | **−8.6** | 2.77 | 0.007 |
| CAP ~ copper, restricted to CAP ≥ 268 | 1,858 | **−8.3** | 3.19 | 0.021 |
| CAP ~ copper + age + sex | 4,137 | **−16.2** | 5.44 | 0.010 |
| CAP ~ alcohol g/day (crude) | 4,137 | +0.01 | 0.08 | 0.86 |

Copper intake is consistently negatively associated with CAP, and the
association survives age/sex adjustment. Alcohol (continuous g/day) shows
nothing — but see the issues below on how alcohol is modeled.

### Analysis 2 — dietary/serum copper, visceral fat, FIB-4

| Model | N | β (exposure) | P |
|---|---:|---:|---:|
| HSI ~ dietary Cu (crude) | 4,253 | −0.73 | 0.098 |
| VAT volume ~ dietary Cu (crude) | 2,810 | −27.0 | 0.18 |
| VAT volume ~ dietary Cu + age + sex | 2,810 | **−78.7** | 0.006 |
| VAT volume ~ dietary Cu + age + sex + BMI | 2,803 | −25.1 | 0.056 |
| FIB-4 ~ dietary Cu (crude) | 4,288 | **+0.183** | 0.001 |
| FIB-4 ~ dietary Cu + age + sex | 4,288 | +0.0001 | 0.998 |
| VAT volume ~ serum Cu (crude) | 3,278 | **+1.30** | < 0.001 |
| VAT volume ~ serum Cu + age + sex | 3,278 | **+1.66** | < 0.001 |
| VAT volume ~ serum Cu + age + sex + BMI | 3,267 | −0.11 | 0.47 |

Bold marks the coefficients highlighted as significant in the original.

## Answers to the open questions in the notes

**Why does copper → visceral fat only become significant after adding age and
sex? Don't confounders usually weaken associations?**

Confounders weaken an association only when they bias it *away* from the null.
Here age acts as a **negative confounder (suppressor)**: age is strongly
positively associated with visceral fat (β ≈ +11.4 cm³/yr, R² = 0.22), and
older adults also tend to have higher copper *density* (more nutrient-dense
diets, fewer total calories). The two effects run in opposite directions and
cancel in the crude model; removing age's influence unmasks the negative copper
association. Easy to verify: regress `cukcal` on `ridageyr` — expect a positive
coefficient.

**Why does adding BMI or waist wipe out the association?**

This is **over-adjustment, not confounding control**. BMI and waist
circumference are essentially co-measurements of the outcome (BMI alone
explains 46% of VAT variance, waist 59%). If copper affects adiposity, BMI sits
on the causal pathway (a mediator), so conditioning on it removes most of the
effect being estimated. Keep the BMI model as a sensitivity analysis ("is the
copper association VAT-specific beyond overall adiposity?"), not as the primary
model.

**Why is dietary copper positively associated with FIB-4, opposite to the
elastography stiffness result?**

**FIB-4 contains age in its formula** (age × AST / (platelets × √ALT)). The
crude positive association is the age–copper correlation passing straight
through the index: after adjusting for age the coefficient is exactly zero
(β = 0.0001, P = 0.998). There is no contradiction with elastography — adjusted
FIB-4 shows nothing, and stiffness (a direct measurement) shows a negative
association. FIB-4 also performs poorly as a fibrosis measure in
general-population samples; prefer LSM.

**Why is serum copper unrelated to dietary copper, and positively related to
visceral fat?**

Expected. Serum copper is homeostatically regulated and ~90%
ceruloplasmin-bound; ceruloplasmin is an **acute-phase reactant**, elevated
with adiposity-related inflammation and with estrogen (higher in women and OC
users). The positive serum-Cu–VAT association vanishing after BMI adjustment
fits an inflammation story, not an intake story. Treat serum and dietary copper
as measuring different constructs; consider adjusting serum-Cu models for CRP
and reporting them separately.

## Methodological issues to fix

> **1 · No sampling weights — affects every estimate.** Both `svyset` calls
> omit `pweight` (the logs print `Sampling weights: <none>` and Population size
> = N of obs). Variances are design-adjusted but the estimates are
> *unweighted*, so nothing is nationally representative — a problem for a paper
> framed as a population-based NHANES analysis. Use the day-2 dietary weights
> (`wtdr2d`), rescaled when combining cycles, e.g.
> `svyset sdmvpsu [pweight=wtdr2d], strata(sdmvstra)`. Every number in the
> abstract will change and must be regenerated.

- **Subsetting with `if` instead of `subpop()`.** With survey data, restrict
  with `svy, subpop(adult): regress …` so variance estimation uses the full
  design. This also fixes the inconsistent restriction (`>=248` in one model,
  `>248` in another).
- **Conditioning on the outcome.** The CAP ≥ 248 / ≥ 268 restricted regressions
  select on the dependent variable; the attenuated coefficients (−16 → −8.6)
  are largely a mechanical range-restriction artifact, not evidence of a weaker
  effect in steatosis. Replace with quantile regression or logistic models for
  steatosis categories.
- **Extreme exposure skew.** `copper_per_1000kcal` has skewness 10.6 and
  kurtosis 258 (max 10.3 mg/1000 kcal — likely recall errors or unusual items
  like organ meats/oysters). A linear term lets a handful of points drive the
  fit. Model log₂(copper) or quartiles; winsorize as sensitivity.
- **Alcohol is zero-inflated.** Median intake is 0 g/day, so a continuous
  linear term is a weak test — the abstract's claim that alcohol is unrelated to
  steatosis is not yet supported. Use categories aligned with MASLD/MetALD
  thresholds and supplement recall data with the ALQ questionnaire.
- **DXA visceral fat exists only for ages 8–59.** The "adults" DXA cohort is
  really 18–59; state this and don't generalize to older adults.
- **Analytic N varies across models** (2,803–5,365 in the DXA file), so
  coefficients are not comparable across nested models. Fix one complete-case
  sample per outcome before comparing.
- **HSI is BMI-derived** (BMI, ALT/AST, sex, diabetes), so HSI models adjusted
  for BMI are circular. Prefer CAP and DXA as steatosis outcomes; drop or
  clearly demote HSI.
- **Missing confounders.** No adjustment yet for race/ethnicity, income (PIR),
  education, smoking, physical activity, diabetes/HOMA-IR, or total energy; no
  supplement copper (dietary files are food-only).
- **Results cited but not in these logs.** The sex interaction (P = 0.005),
  stiffness (`luxsmed`) models, and sex-stratified estimates in the abstract
  aren't in either log; the abstract's "age-adjusted" β = −17.0 doesn't match
  the logged age+sex model (−16.2, P = 0.010). Everything should be regenerated
  from version-controlled .do files rather than interactive sessions.
- **Minor:** `riagendr` (1/2) is used as continuous — fine for a binary, but
  recode to a 0/1 female indicator or use `i.riagendr` for interpretability;
  CAP is machine-bounded at 100–400 (mild ceiling effects — another reason
  quantile regression is a useful check); one syntax error
  (`svyset sdmvpsu, sdmvstra`) was harmless.

## Proposed analysis plan

1. **Cohort construction and survey design.** Confirm cycles: elastography
   (2017–Mar 2020 and/or Aug 2021–Aug 2023) for the CAP cohort; 2011–2018 for
   DXA VAT (ages 18–59). Apply day-2 dietary weights rescaled for combined
   cycles; `svyset` with `pweight`; use `subpop()` for all restrictions.
   Exclusions: age < 18, pregnancy, unreliable recalls (`DR1DRSTZ`/`DR2DRSTZ`
   ≠ 1 or single-day only), implausible energy (< 500 or > 3,500 kcal for
   women; < 800 or > 4,200 for men), incomplete/unreliable elastography exams
   (fasting < 3 h, < 10 valid measures, IQR/median > 30%). Build a participant
   flow diagram.
2. **Exposure definition.** Primary: mean copper across both recall days per
   1,000 kcal, modeled as quartiles and as log₂(copper). Sensitivities: Willett
   residual method instead of density; winsorized continuous copper; total
   copper including supplements (DSQ files); usual intake via the NCI method if
   feasible.
3. **Outcomes.** Primary: CAP (continuous) and steatosis as binary CAP ≥ 248
   (sensitivity: ≥ 263, ≥ 274, ≥ 285). Secondary: LSM (`luxsmed`,
   log-transformed; fibrosis ≥ 8 kPa), DXA VAT volume/area/mass and
   android:gynoid ratio. Classify MASLD vs MetALD vs other-SLD using
   cardiometabolic criteria + alcohol thresholds — this makes the paper current
   with the 2023 nomenclature and matches the project title. Demote HSI; report
   FIB-4 only with the age caveat, if at all.
4. **Model sequence.** Model 1: crude. Model 2: age, sex, race/ethnicity.
   Model 3 (primary): + PIR, education, smoking, physical activity, total
   energy intake, alcohol category. Model 4 (sensitivity, labeled
   over-adjusted): + BMI or waist. `svy: regress` for continuous outcomes,
   `svy: logistic` for steatosis/MASLD, quantile regression as a robustness
   check on the upper CAP tail (replacing the outcome-restricted models).
5. **Effect modification.** Formal interaction terms for copper × sex and
   copper × age group in the primary model of each cohort; report stratified
   estimates with the interaction P-value. This substantiates the abstract's
   women-vs-men claim under weighting.
6. **Dose–response.** Quartile trend tests plus restricted cubic splines (3–4
   knots) to test linearity — copper adequacy may matter mainly at the low end
   (RDA 0.9 mg/day), which a linear term would blur.
7. **Serum copper as a separate aim.** Report dietary–serum copper correlation
   (expected ≈ 0, homeostasis); model serum Cu with adjustment for sex, age,
   and CRP, framing it as an inflammation/acute-phase marker rather than an
   intake proxy. Check whether serum Cu requires the biochemistry subsample
   weights for those cycles.
8. **Reconciliation and reproducibility.** A short supplement decomposing the
   suppression finding (age–copper and age–VAT correlations; stepwise
   coefficient changes) preempts reviewer confusion. Convert the interactive
   sessions into numbered .do files (data build → derivations → models →
   tables) with logs regenerated end-to-end, so every abstract number traces to
   a script.

## Bottom line

The core signal is promising and internally consistent: energy-adjusted dietary
copper is inversely associated with steatosis by two independent measurement
modalities (elastography CAP and DXA visceral fat), the apparent contradictions
dissolve on inspection (age suppression, FIB-4's age term, BMI
over-adjustment), and the serum-copper null is expected biology. But no current
estimate is usable as-is: the analyses are unweighted, several models condition
on the outcome, and the exposure's extreme skew is unhandled. The plan above is
mostly re-plumbing, not new science — the substantive additions are the
MASLD/MetALD classification, proper alcohol handling, spline dose–response, and
the formal sex interaction.

---

Prepared from `Cu_CAP_Aug2026.log` (10 Aug 2026), `CuNHANES_DexaFat_Aug2026.log`
(11 Aug 2026), and `CuNHANES_CAP_Notes.docx`. Source files untouched on the
shared drive.
