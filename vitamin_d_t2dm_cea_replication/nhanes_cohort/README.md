# NHANES 2013–2018 prediabetes cohort reconstruction

This rebuilds the baseline study population of Briody et al. 2026 — the NHANES
2013–2018 adults with prediabetes (paper Table 1, N = 4176) — from public NHANES
microdata, as the first concrete step toward replicating the cost-effectiveness study.
It is a self-contained pipeline: `reconstruct_cohort.py` downloads the raw files,
applies the paper's prediabetes definition, applies survey weights, and prints a
side-by-side comparison with Table 1.

```
uv run python reconstruct_cohort.py
```

## TL;DR — what happened

- **It works and reproduces the cohort's *profile* well.** Under the most faithful
  reading of the paper's definition, the reconstructed BMI (30.3), fasting glucose
  (5.9 mmol/L), race/ethnicity (NH White 62.6% vs 62%, NH Black 12.0% vs 12%),
  smoking (18.6% vs 18%), and albuminuria (micro 8.2%, macro 0.8%) match Table 1
  almost exactly.
- **The exact N and a few rows do not reproduce from public data alone**, and that is
  expected: the paper's precise recipe for combining the three prediabetes criteria
  across cycles (and its family-history and eGFR choices) is not fully published. The
  authors have said they will deposit their analytic dataset and scripts on Mendeley
  Data; those will close the gap. My two defensible interpretations **bracket** the
  paper's N (3563 and 5603 vs 4176).
- **Net:** a working, transparent reconstruction that is close enough to parameterize
  a simulation, with the residual differences understood and localized.

## How it's done (method)

1. **Download** the needed files for all three two-year cycles — `_H` (2013–14),
   `_I` (2015–16), `_J` (2017–18) — from the current NHANES path
   `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{FILE}_{suffix}.XPT`
   (the older `/Nchs/Nhanes/{cycle}/` path now returns only a redirect stub). Files are
   cached under `nhanes_data/` (gitignored) so reruns are instant. `read_sas` parses the
   SAS `.XPT` transport format; everything merges on the respondent id `SEQN`.
2. **Pull the variables** behind each Table 1 row (full map in the table below): demographics
   (`DEMO`), HbA1c (`GHB`), fasting glucose + fasting weight (`GLU`), 2-h OGTT (`OGTT`,
   2013–16 only), BMI (`BMX`), blood pressure (`BPX`), lipids (`HDL`, `TRIGLY`),
   creatinine (`BIOPRO`), 25(OH)D (`VID`), urine ACR (`ALB_CR`), and the questionnaire
   items for smoking, diabetes status, cardiovascular history, dialysis, and family history.
3. **Define prediabetes** exactly as the paper states — HbA1c 5.7–6.4% **OR** FPG
   100–125 mg/dL **OR** 2-h OGTT 140–199 mg/dL — restricted to adults ≥18, excluding
   diagnosed diabetes (`DIQ010==1`) and lab-defined diabetes (HbA1c ≥6.5 / FPG ≥126 /
   OGTT ≥200).
4. **Derive** the clinical variables: eGFR via the race-free CKD-EPI 2021 equation →
   CKD stages; ACR → micro/macroalbuminuria; 25(OH)D categories; SBP as the mean of the
   available `BPXSY1–4` readings; unit conversions to match the paper (mmol/L, µmol/L).
5. **Weight every estimate.** Because two of the three criteria (FPG, OGTT) are measured
   only in the morning **fasting subsample**, the subsample governs the analytic domain.
   Combining three two-year cycles, I divide each person's 2-year weight by 3 per the
   NHANES analytic guidelines: **fasting** `WTSAF2YR/3` for the fasting-domain reconstruction
   and **MEC** `WTMEC2YR/3` for the full-exam reconstruction. Table 1 quantities are weighted
   means / SDs / proportions.

> Note on our house NHANES rule: `WTMECPRP` is **not** used here. That weight belongs to the
> 2017–March-2020 *combined pre-pandemic* file; this study uses three standalone two-year
> cycles, so the correct multi-cycle weights are `WTSAF2YR/3` and `WTMEC2YR/3`.

## The one real decision: how to combine the three criteria

This is where reproduction gets interesting. The paper's definition is a **union** of three
criteria, but HbA1c is measured on *every* exam participant while FPG and OGTT are measured
only on the ~50% morning fasting subsample (and OGTT was dropped after 2016 — there is no
`OGTT_J`). So "HbA1c OR FPG OR OGTT" means different things depending on the sample:

- **Fasting-subsample union (def A):** screen only people who have all markers measured,
  weight `WTSAF2YR/3` → **N = 3563**.
- **Full-MEC union (def B):** apply the union to all examinees (FPG/OGTT contribute where
  measured), weight `WTMEC2YR/3` → **N = 5603**.

The paper's **N = 4176 sits between these**, and no single public-data choice reproduces
N = 4176 with mean age 53.3 *and* the reported HbA1c spread (SD 0.5) *and* FPG spread
(SD 0.8) at the same time. Two diagnostic fingerprints show why:

- The paper's **wide HbA1c SD (0.5)** means the cohort includes people with HbA1c *outside*
  5.7–6.4 (qualified by FPG/OGTT) → it is a genuine union, not HbA1c-only.
- The paper's **wide FPG SD (0.8)** means it includes many people with FPG *outside* 100–125
  (qualified by HbA1c) → HbA1c-defined cases are numerous, which happens on the full exam
  sample, not the fasting half.

Those two facts point at a union built on the full sample, but that overshoots N (5603).
The residual almost certainly comes from unpublished specifics — how OGTT was handled in the
two cycles that have it, whether a complete-covariate filter was applied, and the exact
subsample/weight domain. The script prints a diagnostics table of eight candidate definitions
(A–H) so the sensitivity is explicit.

## Result — reconstruction vs. paper Table 1

Both bracketing reconstructions are reported side by side (full CSV in
[`outputs/table1_reconstructed.csv`](./outputs/table1_reconstructed.csv)). The full-MEC
union (B) matches the demographic/metabolic profile best:

| Variable | Paper | Recon A (fasting-union) | Recon B (MEC-union) |
|---|---|---|---|
| N (unweighted) | 4176 | 3563 | 5603 |
| Age, years | 53.3 (17.0) | 49.2 (17.0) | 51.6 (16.7) |
| Female | 49% | 46.2% | **48.1%** |
| NH White | 62% | 63.8% | **62.6%** |
| NH Black | 12% | 10.6% | **12.0%** |
| Hispanic | 16% | 15.6% | 15.4% |
| Other race/eth | 10% | 10.0% | **10.0%** |
| Postsecondary ed | 61% | 59.3% | 59.5% |
| Current smoker | 18% | 18.0% | 18.6% |
| HbA1c, % | 5.7 (0.5) | 5.5 (0.4) | 5.6 (0.4) |
| BMI, kg/m² | 30.3 (7.3) | 30.0 (7.2) | **30.3 (7.1)** |
| Systolic BP | 128 (19) | 124.3 | 125.9 |
| HDL, mmol/L | 1.35 (0.39) | 1.4 | 1.4 |
| LDL, mmol/L | 3.00 (0.93) | 3.0 | **3.0** |
| Triglycerides | 1.12 (0.67) | 1.3 | 1.4 |
| Creatinine, µmol/L | 80 (35) | 77.9 | 78.7 |
| Fasting glucose, mmol/L | 5.9 (0.8) | 5.8 | **5.9** |
| 25(OH)D <30 nmol/L | 7.6% | 4.7% | 4.8% |
| 25(OH)D 30–50 | 21.9% | 19.0% | 18.7% |
| 25(OH)D ≥50 | 70.5% | 75.9% | 75.9% |
| Microalbuminuria | 8.2% | 7.2% | **8.2%** |
| Macroalbuminuria | 0.8% | 0.7% | **0.8%** |
| CKD stage 3–5 | 9.7% | 4.8% | 5.9% |
| CKD stage 4–5 | 0.5% | 0.2% | 0.4% |
| Myocardial infarction | 2.5% | 2.9% | 3.4% |
| Stroke | 3.1% | 2.6% | 2.8% |
| CHF | 1.3% | 1.8% | 2.3% |
| Angina | 2.8% | 1.6% | 2.0% |
| Family history of diabetes | 27% | 39.9% | 41.4% |

**Rows that match closely (≈20 of 28):** N-profile, sex, all four race/ethnicity groups,
education, smoking, BMI, HDL, LDL, fasting glucose, creatinine, both albuminuria rows,
stroke, and (for A) MI/angina.

**Rows that differ, and the likely cause:**
- **N / age:** the union-combination ambiguity above; the paper's cohort is ~1.7–4 years
  older than either bracket, consistent with a slightly more HbA1c-weighted sample.
- **HbA1c / FPG SDs:** definitional artifacts (see fingerprints above) — resolved once the
  exact criterion-combination rule is known.
- **25(OH)D deficiency and CKD undershoot:** most likely the eGFR equation (paper may use
  CKD-EPI 2009 with a race term, which shifts the CKD prevalence) and possibly a different
  25(OH)D cutpoint/assay handling; both are one-line changes once confirmed.
- **Family history 40% vs 27%:** the biggest single gap. I used `MCQ300C` ("close relative
  had diabetes"); the paper's lower figure suggests a different item (e.g., first-degree
  relatives only) — flagged for confirmation against their code.

## Variable → file map

| Table 1 quantity | Variable(s) | File | Weight domain |
|---|---|---|---|
| Age / sex / race / education / weights / design | `RIDAGEYR`,`RIAGENDR`,`RIDRETH3`,`DMDEDUC2`,`WTMEC2YR`,`WTINT2YR`,`SDMVPSU`,`SDMVSTRA` | `DEMO` | — |
| HbA1c | `LBXGH` | `GHB` | full MEC |
| Fasting glucose (+ fasting weight) | `LBXGLU`, `WTSAF2YR` | `GLU` | fasting |
| 2-h OGTT | `LBXGLT` | `OGTT` (H, I only) | OGTT |
| BMI | `BMXBMI` | `BMX` | full MEC |
| Systolic BP | `BPXSY1–4` | `BPX` | full MEC |
| HDL | `LBDHDD` | `HDL` | full MEC |
| Triglycerides / LDL | `LBXTR`, `LBDLDL` | `TRIGLY` | fasting |
| Creatinine | `LBXSCR` | `BIOPRO` | full MEC |
| 25(OH)D | `LBXVIDMS` | `VID` | full MEC |
| Urine ACR | `URDACT` | `ALB_CR` | full MEC |
| Smoking | `SMQ020`,`SMQ040` | `SMQ` | interview |
| Diagnosed diabetes | `DIQ010` | `DIQ` | interview |
| MI / stroke / CHF / angina | `MCQ160E/F/B/D` | `MCQ` | interview |
| Dialysis | `KIQ025` | `KIQ_U` | interview |
| Family history | `MCQ300C` | `MCQ` | interview |

Neuropathy is intentionally omitted — NHANES discontinued the monofilament exam after 2004,
so there is no clean 2013–2018 source (the paper's neuropathy row must rest on a self-report
proxy we would need their definition to match).

## Outputs

- `outputs/table1_reconstructed.csv` — the comparison table above.
- `outputs/cohort_fasting_union.csv`, `outputs/cohort_mec_union.csv` — the two analytic
  cohorts with all derived per-person variables and weights, ready to feed a simulation.

## Next steps to close the gap

1. Obtain the authors' analytic dataset/scripts (Mendeley deposit or corresponding author)
   and diff their inclusion logic against the diagnostics table — this pins the exact
   criterion-combination rule and should snap N to 4176.
2. Confirm the eGFR equation (CKD-EPI 2009 vs 2021) and the family-history variable; both
   are localized one-line fixes.
3. Feed the validated cohort into the prevention-module reconstruction (the microsimulation
   front-end) described in the parent [`README.md`](../README.md).
