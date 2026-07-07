I have verified all the key variables and the critical gotchas against CDC documentation. Here is the briefing.

---

# NHANES 2013–2018 Reconstruction Scoping: Briody 2026 Prediabetes Cohort

## 0. Headline findings (read these first)

- **The single biggest gotcha: OGTT was discontinued after 2015–2016.** There is `OGTT_H` (2013–14) and `OGTT_I` (2015–16) but **no `OGTT_J`** — the 2-hr OGTT criterion can only be applied to two of the three cycles. Confirmed via CDC 404 on the 2017–2018 OGTT page and the search result "OGTT dropped in 2018."
- **Two different subsample weights are in play, not one.** Fasting glucose / triglycerides / LDL use the **fasting** subsample weight `WTSAF2YR`. The 2-hr OGTT uses a *separate* **OGTT** subsample weight `WTSOG2YR`. They have different denominators and different availability, which complicates a "HbA1c OR FPG OR OGTT" cohort definition.
- **2017–2018 is a standalone 2-year cycle here** (suffix `_J`, weights `WTMEC2YR`/`WTINT2YR`/`WTSAF2YR`). Do **not** substitute the 2017–March-2020 pre-pandemic combined file (`P_` prefix, `WTMECPRP`/`WTSAFPRP`) — that pools an extra ~1.5 years and is a different sample. Your CLAUDE.md note about `WTMECPRP` applies to the *combined* file only and is the wrong choice for this study.
- **File URLs moved.** The old `wwwn.cdc.gov/Nchs/Nhanes/<cycle>/FILE.htm` paths now 404. Current pattern: `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<startyear>/DataFiles/FILE_<suffix>.htm` (e.g. `.../Public/2017/DataFiles/GLU_J.htm`).
- **Neuropathy has no clean NHANES source in 2013–2018** — the peripheral-neuropathy monofilament exam was discontinued after 2004. This is the least reproducible Table 1 row (see §4).
- **BP is auscultatory (`BPXSY1–4`) in all three cycles** — the oscillometric `BPXO` file only appears with the 2017–2020 combined release, so you are consistent across 2013–2018.

File suffixes: **`_H` = 2013–2014, `_I` = 2015–2016, `_J` = 2017–2018.**

---

## 1. Variable-to-file mapping table

All variable names verified against CDC documentation except a handful marked "(std.)" that are standard, stable NHANES names carried unchanged across these cycles. Files exist in all three cycles unless noted.

| Table 1 quantity | Variable(s) | File (all 3 cycles: `_H`/`_I`/`_J`) | Weight domain | Notes |
|---|---|---|---|---|
| Age | `RIDAGEYR` | `DEMO` | — | Restrict ≥18 |
| Sex | `RIAGENDR` | `DEMO` | — | 1=Male, 2=Female |
| Race/ethnicity | `RIDRETH3` | `DEMO` | — | Recode: 3=NH White, 4=NH Black, 1+2=Hispanic, 6+7=Other (incl. NH Asian). Use `RIDRETH3`, not `RIDRETH1`, for the NH-Asian split. |
| Education | `DMDEDUC2` | `DEMO` | — | **Gotcha:** `DMDEDUC2` is asked of adults **20+**. The 18–19 subset is captured by youth `DMDEDUC3` — imperfect mapping for a ≥18 cohort. |
| Survey weights | `WTMEC2YR`, `WTINT2YR` | `DEMO` | — | Plus fasting/OGTT subsample weights below |
| Design variables | `SDMVPSU`, `SDMVSTRA` | `DEMO` | — | Required for SEs/CIs (Taylor linearization) |
| HbA1c | `LBXGH` | `GHB` | Full MEC (`WTMEC2YR`) | Prediabetes 5.7–6.4%; exclude ≥6.5% |
| Fasting plasma glucose | `LBXGLU` (`LBDGLUSI` = mmol/L) | `GLU` | **Fasting** (`WTSAF2YR`) | 100–125 mg/dL; exclude ≥126. `WTSAF2YR` lives in this file. |
| 2-hr OGTT glucose | `LBXGLT` | `OGTT` (`_H`, `_I` only — **no `_J`**) | **OGTT** (`WTSOG2YR`) | 140–199 mg/dL. Not collected 2017–18. Separate weight from fasting. |
| BMI | `BMXBMI` (`BMXWT`,`BMXHT`) | `BMX` | Full MEC | |
| Systolic BP | `BPXSY1`–`BPXSY4` | `BPX` | Full MEC | Auscultatory in all 3 cycles. Convention: mean of available readings, often dropping reading 1. |
| Total cholesterol | `LBXTC` | `TCHOL` | Full MEC | (needed for Friedewald LDL if recomputing) |
| HDL | `LBDHDD` (`LBDHDDSI`) | `HDL` | Full MEC | Confirmed "Direct HDL-Cholesterol (mg/dL)" |
| Triglycerides | `LBXTR` (`LBDTRSI`) | `TRIGLY` | **Fasting** (`WTSAF2YR`) | Fasting subsample |
| LDL | `LBDLDL` (Friedewald) | `TRIGLY` | **Fasting** (`WTSAF2YR`) | **Gotcha:** `TRIGLY_J` (2017–18) also carries `LBDLDLM` (Martin-Hopkins) and `LBDLDLN` (NIH eq-2); the older cycles have only Friedewald `LBDLDL`. Use `LBDLDL` for cross-cycle consistency. |
| Serum creatinine | `LBXSCR` (`LBDSCRSI`) | `BIOPRO` | Full MEC | IDMS-standardized in these cycles → no recalibration needed. Feeds eGFR. |
| Fasting glucose (Table 1 FPG row) | `LBXGLU` | `GLU` | Fasting | Same as prediabetes FPG |
| Serum 25(OH)D | `LBXVIDMS` (nmol/L); comment `LBDVIDLC` | `VID` | Full MEC | Total = D2+D3, excl. epi. Categories (e.g. <30 deficient / 30–<50 insufficient / ≥50 sufficient) computed by you. |
| Urine albumin | `URXUMA` (µg/mL) | `ALB_CR` | Full MEC | |
| Urine creatinine | `URXUCR` (mg/dL) | `ALB_CR` | Full MEC | |
| Albumin-creatinine ratio | `URDACT` (mg/g) | `ALB_CR` | Full MEC | Micro = 30–300; macro = >300 mg/g. `URDACT = URXUMA/URXUCR × 100`. |
| CKD stage 3–5 / 4–5 | derived from `LBXSCR`+`RIDAGEYR`+`RIAGENDR` | — | Full MEC | eGFR via CKD-EPI (see §3). Stage 3–5 = eGFR<60; 4–5 = <30. |
| Dialysis | `KIQ025` | `KIQ_U` | Interview | "Received dialysis in past 12 months." Also `KIQ022` = weak/failing kidneys. |
| Neuropathy | *(no clean source)* — closest `DIQ175M` | `DIQ` | Interview | **See §4.** `DIQ175M` ("tingling/numbness in hands or feet") is a *conditional* diabetes-risk follow-up, not a clinical neuropathy diagnosis. |
| MI / heart attack | `MCQ160E` | `MCQ` | Interview | "Ever told you had heart attack" |
| Stroke | `MCQ160F` | `MCQ` | Interview | |
| CHF | `MCQ160B` | `MCQ` | Interview | Congestive heart failure |
| Angina | `MCQ160D` | `MCQ` | Interview | Angina/angina pectoris (`MCQ160C` = CHD if needed) |
| Family history of diabetes | `MCQ300C` | `MCQ` | Interview | "Close relative had diabetes?" — the robust, universally-asked item. **Prefer this over `DIQ175A`**, which is only asked of a conditional subset. |
| Smoking | `SMQ020`, `SMQ040` | `SMQ` | Interview | `SMQ020`=≥100 cigs lifetime; `SMQ040`=now smoke. Derive never/former/current. |
| Diagnosed diabetes (exclusion) | `DIQ010` | `DIQ` | Interview | Plus HbA1c≥6.5 / FPG≥126 exclusions. `DIQ160`=told prediabetes. |

To pull a specific cycle's page, swap the year folder and suffix, e.g. 2013–14 fasting glucose = `.../Public/2013/DataFiles/GLU_H.htm`.

---

## 2. Survey weighting when combining fasting-subsample variables across three cycles

**Rule (NHANES Analytic Guidelines, Series 2 No. 161 / Weighting Tutorial):** to combine k two-year cycles, build a multi-year weight by dividing each person's 2-year weight by the number of cycles. The same rule applies to *subsample* weights, not just the MEC weight.

- **Fasting-based rows (FPG, triglycerides, LDL):** construct `WTSAF6YR = WTSAF2YR / 3` and use it with `SDMVPSU`/`SDMVSTRA`.
- **OGTT row:** the OGTT criterion uses `WTSOG2YR`, and OGTT exists only for 2013–14 + 2015–16, so any OGTT-weighted estimate spans **2 cycles → divide by 2** (`WTSOG4YR = WTSOG2YR / 2`) and cannot include 2017–18.
- **Full-MEC rows (HbA1c, BMI, BP, HDL, TC, creatinine, vitamin D, ACR):** `WTMEC6YR = WTMEC2YR / 3`.
- **Interview-only rows (MCQ, DIQ, SMQ, KIQ):** `WTINT6YR = WTINT2YR / 3` (though these items are also present on MEC examinees, so if the analytic sample is the fasting subsample, the fasting weight governs the sample-definition domain).

**The core weighting tension for this cohort.** Prediabetes is "HbA1c OR FPG OR OGTT." HbA1c is a full-MEC variable (everyone), but FPG and OGTT exist only in the ~50% morning **fasting** subsample. So the cohort's *definable domain* is the fasting subsample, and the defensible choice is to **restrict the analytic sample to fasting-subsample examinees and weight with `WTSAF2YR/3`**. This is almost certainly what yields N≈4176. Note that this down-weights/discards HbA1c-only prediabetics who were in afternoon/evening sessions — a deliberate trade to keep the three criteria applicable to everyone. Document whichever choice you make; it materially changes N and the means.

---

## 3. eGFR / derived variables

- **Inputs:** `LBXSCR` (serum creatinine, mg/dL, IDMS-standardized in 2013–2018 so no ×0.95 recalibration), `RIDAGEYR`, `RIAGENDR`. Race is *not* an input if you use the current equation.
- **Equation choice matters:**
  - **CKD-EPI 2021 creatinine (race-free; Inker et al., NEJM 2021)** — now the NKF/ASN-recommended standard; use this unless the paper says otherwise.
  - **CKD-EPI 2009 (Levey et al., Ann Intern Med 2009)** — includes a Black-race coefficient; older diabetes-cohort papers often used it. A 2026 paper *could* have used either; the eGFR-derived CKD prevalences shift noticeably between them, so match the paper's stated equation for validation.
- **Thresholds:** eGFR (mL/min/1.73 m²) <60 → CKD stage 3–5; <30 → stage 4–5. Albuminuria from `URDACT`: 30–300 mg/g micro-, >300 macro-albuminuria (a single spot sample — NHANES lacks the confirmatory second sample KDIGO wants, a minor caveat).

---

## 4. How hard is reproducing Table 1 (N=4176; mean age 53.3; 49% female)? 

**Difficulty: moderate and mostly mechanical for ~90% of rows; three genuine hazards.** I could not independently locate the Briody 2026 paper in indexed sources, so the following is scoped from the stated definition and NHANES structure.

**Straightforward:** demographics, HbA1c, FPG, BMI, BP, lipids, creatinine, vitamin D, ACR, and the questionnaire comorbidities (MI/stroke/CHF/angina/family history/smoking/dialysis) are all direct variable pulls. The prediabetes definition, exclusions, ≥18 filter, and `WTSAF`-based weighting reproduce the design cleanly.

**Gotchas that will make N and the means drift if mishandled:**

1. **OGTT only in 2 of 3 cycles.** If the paper counts anyone meeting *any* of the three criteria, the OGTT-only prediabetics can be identified in 2013–2016 but are invisible in 2017–2018. Whether they included the OGTT arm at all, and how they weighted it, directly moves N=4176. This is the first thing to pin down against the paper's methods.
2. **Fasting subsample restriction and weight.** N=4176 is consistent with restricting to the adult fasting subsample across 3 cycles (roughly half of ~8–9k fasting adults meet a prediabetes criterion after excluding diagnosed/overt diabetes). Using full-MEC HbA1c-only inclusion would give a *larger* N and *different* means — a common reproduction failure. Match the subsample and the `WTSAF2YR/3` weight.
3. **Neuropathy.** No monofilament/nerve-conduction exam exists in 2013–2018 (discontinued after 2004). If Table 1 reports a neuropathy prevalence, it must come from a self-report proxy — most likely `DIQ175M` (tingling/numbness), which is only asked as a *conditional follow-up* to the diabetes-risk battery, so its denominator is not the full cohort. Reproducing this row exactly is the hardest; expect to reverse-engineer their definition.

**Secondary drift sources:** LDL variable choice (`LBDLDL` Friedewald for cross-cycle vs the new `LBDLDLM/N` only in `_J`); education for 18–19 y (`DMDEDUC2` vs `DMDEDUC3`); BP averaging convention (which of `BPXSY1–4` to include); vitamin-D category cutoffs; and standalone `_J` vs the pre-pandemic `P_` files. A weighted mean age of 53.3 and 49% female are good, sensitive checkpoints — hit those two first before trusting the biomarker rows.

**Recommended validation order:** (1) unweighted N per cycle and total → 4176; (2) weighted mean age 53.3 / %female 49; (3) weighted HbA1c/FPG/BMI means; (4) the eGFR-CKD and albuminuria rows (equation-sensitive); (5) neuropathy last.

---

## 5. International analogues: what plays NHANES's role elsewhere

No other country matches NHANES's *full* combination (fasting + OGTT + spot-urine ACR + serum creatinine + 25(OH)D + questionnaire comorbidities) in one nationally representative exam survey. The realistic options, tiered by how much of the biomarker set they deliver:

**Tier 1 — near-complete NHANES analogues (venous blood + exam, nationally representative):**
- **KNHANES (Korea)** — the closest twin: fasting glucose, HbA1c, full lipids, creatinine, urine ACR, BP, anthropometry, plus questionnaire. Vitamin D and OGTT vary by year.
- **CHMS (Canadian Health Measures Survey)** — direct NHANES design; fasting subsample, labs, exam.
- **ENSANUT (Mexico)** — fasting glucose, HbA1c, lipids, creatinine; large and nationally representative.
- **Health Survey for England (HSE)** / **ELSA** — nurse visit with venous blood: HbA1c, total/HDL chol, creatinine, fibrinogen; fasting glucose and OGTT are rare/absent, and biomarker collection has thinned in recent HSE waves.

**Tier 2 — aging cohorts with venous blood (50/45+, so wrong age range for a ≥18 prediabetes cohort but rich biomarkers):**
- **SHARE (Europe)** — dried blood spots: HbA1c, total/HDL chol, CRP, etc. DBS cannot do fasting glucose reliably and no urine ACR.
- **CHARLS (China)**, **LASI (India)**, **HRS (US)**, **ELSA (UK)** — HbA1c/lipids/creatinine to varying degrees; HRS uses DBS (HbA1c, chol) only.
- **ELSA-Brasil / PNS (Brazil)** — PNS has a lab subsample (HbA1c, glucose, lipids, creatinine).

**Tier 3 — LMIC surveillance, standardized but biomarker-thin:**
- **WHO STEPS** — STEP 3 biochemical module gives fasting glucose and total cholesterol (HbA1c and full lipid panel only in some countries); no OGTT, no ACR, no vitamin D, limited comorbidity history. Broadest global coverage, thinnest panel.
- **DHS (Demographic & Health Surveys)** — biomarker menu is mostly anemia/HIV/malaria; a minority of recent surveys add random (non-fasting) blood glucose; essentially no fasting glucose, OGTT, lipids, creatinine, or ACR. Weakest fit for a metabolic cohort.
- **China Chronic Disease & Risk Factor Surveillance / CHNS** — fasting glucose, HbA1c, lipids in recent rounds.

**The general challenge (what breaks portability):**
- **Fasting is logistically expensive**, so most surveys outside NHANES/KNHANES/CHMS use non-fasting blood or DBS → **fasting glucose and OGTT largely disappear**; you fall back to HbA1c-defined prediabetes, changing the case definition.
- **DBS cannot reliably measure creatinine, fasting glucose, or spot-urine ACR** → **eGFR-based CKD staging and albuminuria are usually unavailable** in DBS-based cohorts (SHARE, HRS, IFLS).
- **Vitamin D (25(OH)D)** is rarely measured outside NHANES/KNHANES.
- **Urine ACR** is almost NHANES-specific among general-population surveys.
- **OGTT** is disappearing even within NHANES (post-2016) and is essentially absent elsewhere at national scale.
- **Comorbidity history** (MI/stroke/CHF/angina/neuropathy/family history) is generally *more* portable — most surveys carry self-reported physician-diagnosis questions — but wording and reference periods differ, so harmonization is required.

Practical implication for adapting the microsimulation abroad: expect to redefine prediabetes on **HbA1c alone** in most settings, drop or impute **OGTT, ACR, vitamin D, and eGFR-CKD**, and harmonize comorbidities from heterogeneous self-report items. KNHANES, CHMS, and ENSANUT are the only settings where the Table 1 panel transfers with minor loss.

---

## Sources / URLs

Data documentation (2017–18 shown; swap year folder + suffix `_H`/`_I`/`_J` for other cycles):
- Fasting glucose (`WTSAF2YR`, `LBXGLU`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm
- Glycohemoglobin (`LBXGH`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GHB_J.htm
- OGTT (`LBXGLT`, `WTSOG2YR`; 2015–16, **no 2017–18**): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/OGTT_I.htm and /Public/2013/DataFiles/OGTT_H.htm
- Triglycerides & LDL (`LBXTR`, `LBDLDL`/`LBDLDLM`/`LBDLDLN`, `WTSAF2YR`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TRIGLY_J.htm
- HDL (`LBDHDD`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HDL_J.htm
- Vitamin D (`LBXVIDMS`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/VID_J.htm
- Blood pressure (`BPXSY1–4`, auscultatory): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm
- Albumin-creatinine (`URXUMA`,`URXUCR`,`URDACT`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALB_CR_J.htm
- Medical conditions (`MCQ160E/F/B/D/C`, `MCQ300C`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.htm
- Kidney conditions (`KIQ025`, `KIQ022`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/KIQ_U_J.htm
- Diabetes questionnaire (`DIQ010`, `DIQ160`, `DIQ175M`): https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.htm
- Component/data page index by cycle: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2017

Guidelines / methods:
- NHANES Analytic Guidelines (Series 2, No. 161) — multi-cycle weighting, divide 2-yr weights by number of cycles: https://www.cdc.gov/nchs/data/series/sr_02/sr02_161.pdf
- NHANES Weighting Tutorial (subsample weights, `WTSAF`): https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx
- NHANES subsample weights notes: https://wwwn.cdc.gov/nchs/nhanes/search/subsample_weights.aspx
- CKD-EPI 2021 (race-free): Inker et al., N Engl J Med 2021;385:1737. CKD-EPI 2009: Levey et al., Ann Intern Med 2009;150:604.

Note: I could not independently locate the Briody 2026 paper in indexed sources, so N=4176 / age 53.3 / 49% female should be treated as targets to confirm against the paper's own methods (especially its OGTT handling, subsample/weight choice, and neuropathy definition).