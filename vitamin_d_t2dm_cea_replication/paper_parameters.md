# Briody et al. 2026 — extracted model inputs and headline results

Source: Briody J, Pittas AG, Zhang P, Shao Y, Gregg E. "The cost-effectiveness of
vitamin D supplementation for the prevention of type 2 diabetes in the United
States: a microsimulation modelling study." *Lancet Regional Health – Americas*
2026;61:101543. DOI 10.1016/j.lana.2026.101543. Open access (CC BY).

This file transcribes the numbers needed to *replicate* the study. It is the
"input bundle" the authors say they will deposit on Mendeley Data — reconstructed
here from the article and its tables so that a replication can proceed before that
deposit is available.

## Model

- **Engine:** CDC/RTI Prediabetes Microsimulation Model (Hoerger et al. 2023,
  *Value Health* 26:1372–1380). Discrete-time, annual-step, individual-level.
  Prediabetes module (tracks people pre-diabetes, incl. CVD events + mortality)
  and diabetes module (17 diabetes-related complications + mortality).
- **Perspective:** US health-care system (direct medical costs).
- **Currency/year:** 2024 USD (inflated with the BEA GDP price index).
- **Discount rate:** 3% per year for both costs and QALYs (varied 0–5%).
- **WTP threshold:** $100,000/QALY (scenario: $50,000).
- **Horizons:** lifetime (primary) and 10 years.
- **Runs:** mean of 100 runs × 10,000 simulated individuals; 95% UI = 2.5th–97.5th
  percentile across runs. PSA = 100 iterations × 10,000.

## Population

- NHANES **2013–2018** adults ≥18 y with prediabetes; analytic N = **4176**;
  simulated cohort = 10,000.
- **Prediabetes definition:** HbA1c 5.7–6.4%, OR FPG 100–125 mg/dL, OR 2-h OGTT
  140–199 mg/dL. Exclude diagnosed diabetes / HbA1c ≥6.5 / FPG ≥126.
- Table 1 (mean [SD] or n(%)): age 53.3 (17.0); female 2024 (49%); NH White 2603
  (62%), NH Black 488 (12%), Hispanic 668 (16%), other 417 (10%); postsecondary
  education 61%; current smoker 18%; HbA1c 5.7 (0.5)%; BMI 30.3 (7.3); SBP 128
  (19); HDL 1.35 (0.39) mmol/L; LDL 3.00 (0.93); triglycerides 1.12 (0.67);
  creatinine 80 (35) µmol/L; FPG 5.9 (0.8) mmol/L; 25(OH)D <30 nmol/L 7.6%,
  30–50 nmol/L 21.9%, ≥50 nmol/L 70.5%; microalbuminuria 8.2%; macroalbuminuria
  0.8%; CKD 3–5 (eGFR<60) 9.7%; CKD 4–5 (eGFR<30) 0.5%; dialysis <0.1%;
  neuropathy 8.8%; MI 2.5%; stroke 3.1%; CHF 1.3%; angina 2.8%; family history
  of diabetes 27%.

## Intervention

- Vitamin D3 (cholecalciferol) **4000 IU/day** (~3500 IU trial weighted avg).
- Eligibility: prediabetes **and BMI ≥25** (BMI<25 included in denominator but
  only starts treatment if/when BMI crosses 25 during follow-up).
- **Adherence 85%.**
- **Effect: 15% relative risk reduction** in prediabetes→diabetes onset
  (SE 7.65%; from Pittas et al. 2023 IPD meta-analysis). Applied uniformly (no
  effect modification by sex/race/BMI).
- Vitamin D stopped at diabetes onset; no post-diagnosis benefit modelled.
- No adverse events modelled.

## Table 2 — key cost / disutility / probability inputs

Costs are first-year / following-year in USD (SE). Disutilities are first-year /
following-year (SE).

| Parameter | First-yr cost (SE) | Following-yr cost (SE) | First-yr disutility | Following-yr disutility |
|---|---|---|---|---|
| Base cost normoglycemia | 2636 (+12/yr age) | — | baseline QALY 0.935 | duration −0.008/yr |
| Base cost prediabetes | 3244 (+12/yr age) | — | — | — |
| Type 2 diabetes | 11,322 (541) (−78/yr age) | — | baseline QALY 0.935 | — |
| Macroalbuminuria | 13,995 (383) | 3717 (354) | — | — |
| Foot ulcer | 13,431 (856) | 2611 (832) | −0.017 (0.009) | −0.020 (0.009) |
| Amputation | 30,410 (4581) | 0 | −0.092 (0.028) | −0.150 (0.034) |
| Blindness | 15,802 (2107) | 2892 (2205) | −0.045 (0.010) | −0.023 (0.010) |
| CKD 4–5 (eGFR<30) | 13,995 (383) | 3717 (354) | −0.043 (0.010) | −0.025 (0.010) |
| CKD 3–5 (eGFR<60) | 13,995 (383) | 3717 (354) | −0.014 (0.003) | −0.015 (0.003) |
| Dialysis | 114,585 (5273) | 120,361 (6160) | −0.038 (0.015) | −0.015 (0.013) |
| Neuropathy | 5257 (195) | 2447 (226) | −0.007 (0.004) | −0.007 (0.004) |
| Retinopathy (laser) | 5342 (1210) | 2678 (336) | −0.011 (0.007) | −0.014 (0.006) |
| Myocardial infarction | 55,025 (1463) | 10,424 (1346) | −0.027 (0.009) | −0.007 (0.008) |
| Stroke | 28,916 (1034) | 5750 (1065) | −0.107 (0.015) | −0.051 (0.014) |
| Congestive heart failure | 37,942 (1119) | 8587 (1074) | −0.050 (0.014) | −0.043 (0.014) |
| CHD revascularization | 24,719 (681) | 0 | −0.005 (0.006) | 0 (0.007) |
| Angina | 10,831 (402) | 0 | −0.015 (0.009) | −0.028 (0.008) |
| Hypoglycemia (medical) | 9310 (360) | — | 0 | — |
| Vitamin D supplementation | 60 (6) | 60 (6) | — | — |
| Smoking | — | — | −0.006 (0.005) | — |
| BMI (per +1 unit) | — | — | −0.003 (0) | — |

- Cost sources: Yang et al. 2020 (<65 y complication costs); Wang et al. 2022
  (Medicare, ≥65 y). Utilities: Neuwahl et al. 2021.
- Intervention cost $60/yr (sensitivity $72/yr for counselling + materials).

## Table 3 — headline results (per person)

| Outcome | 10-year incremental | Lifetime incremental |
|---|---|---|
| Remaining life-years (undiscounted) | +0.001 (−0.018 to 0.018) | +0.27 (0.06–0.50) |
| Costs (discounted) | −$401 (−$908 to $192) | −$3208 (−$5974 to $171) |
| QALYs (discounted) | +0.01 (−0.01 to 0.02) | +0.12 (0.05–0.19) |
| ICER ($/QALY) | −$61,122 | −$26,134 (−$72,486 to $3947) |
| NMB ($) | $1058 (−$172 to $1964) | $15,483 ($8108–$21,907) |
| Diabetes cumulative incidence | −9.89% (−12.75 to −7.14) | −8.04% (−10.57 to −6.37) |

Diabetes cumulative incidence per 100: 10-yr 20.14→18.15; lifetime 32.35→29.75.
Lifetime complication relative reductions ranged ~1.8% (angina) to ~9.2%
(retinopathy), 8.96% (hypoglycemia), 8.79% (foot ulcer).

## Table 4 — subgroups & scenarios (lifetime NMB, $100k WTP)

Core −$26,134 / $15,483. BMI<25 $9447; BMI 25–30 $14,018; BMI≥30 $20,618;
BMI≥40 $25,260. Age<65 $19,289; ≥65 $15,350. FPG<106 $9113; FPG>106 $19,270.
No family history $14,466; 100% family history $18,590. Scenarios: 3-yr only
$3774; declining efficacy over 5 yr $3438; higher implementation cost $15,872;
WTP −50% $9346; different seed $16,515. Every scenario remained cost-saving.

## Data-sharing statement (verbatim gist)

- Engine source code = property of CDC/RTI, **NOT publicly distributable**;
  technical manual + appendices "available from RTI on reasonable request".
- Will share: deidentified NHANES-derived analytic dataset + data dictionary;
  all input tables (risks, costs, utilities, PSA distributions, scenarios);
  R + Python post-processing/figure scripts; deidentified aggregate outputs.
- Config files that generated results: "available from the corresponding author
  on reasonable request."
- Repository: **Mendeley Data**, DOI at publication; pre-publication access on
  request to jonathanbriody@rcsi.ie.
