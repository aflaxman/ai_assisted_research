# Replicating Amboko et al. (2026): adolescent vs adult maternal care in the MNH eCohort

Paper: Amboko B, Mugenya I, Odipo E, et al. *Maternal healthcare quality, continuity, and
delivery experiences among pregnant adolescents and adult women in Ethiopia, Kenya, and
South Africa: a cohort study.* PLOS Global Public Health 6(10): e0004729.
<https://doi.org/10.1371/journal.pgph.0004729>

## TL;DR

- The paper's own analytic dataset is **restricted** (Harvard Dataverse doi:10.7910/DVN/AUQ8C8,
  access by request to the KEMRI-Wellcome Data Governance Committee). Its two Stata do-files
  and README **are** public and are copied into `data/`.
- Three sibling deposits from the same MNH eCohort are fully **open (CC0)** on Harvard Dataverse.
  One of them, `eCOANC1.dta` (Arsenault et al., doi:10.7910/DVN/SSVXKY), is essentially the same
  baseline sample: 1000 / 1002 / 1044 women in Ethiopia / Kenya / South Africa versus
  1000 / 1009 / 1042 in the paper.
- From that open file I reproduce **Table 1** (participant characteristics by country and age group)
  almost cell-for-cell, plus the Ethiopia first-ANC counselling index from Table 2.
  See `table1_comparison.md`.
- **Not** replicable from open data: everything downstream of the first ANC visit. Follow-up ANC
  visit counts, delivery experience (mistreatment, vaginal exam consent, pain relief, privacy),
  obstetric complications, postnatal check-up, the continuity-of-care index, and the two
  adolescent-only logistic regressions (Tables 3 to 5, Fig 1) all need Module 2 and Module 3
  variables that exist only in the restricted file.

## What data is available

| Deposit | DOI | Access | Contents |
|---|---|---|---|
| Replication data for Amboko et al. | 10.7910/DVN/AUQ8C8 | data **restricted**; code open | `Adolescent paper analysis data.tab` (restricted), two `.do` files, README |
| eCohort paper 1: ANC standards (Arsenault) | 10.7910/DVN/SSVXKY | CC0 | `eCOANC1.dta` (4068 women, 4 countries, 148 vars, first ANC visit only) + codebook |
| ANC quality and perinatal outcome (Yang) | 10.7910/DVN/Q0YKOT | CC0 | `ecohort_perinatal_outcome*.dta` (3600 women; baseline covariates + birth outcome, LBW, fetal loss) |
| Ethiopia malnutrition and anaemia (Clarke-Deelder) | 10.7910/DVN/JGAXNG | CC0 | `econut_analyticdata.dta` (811 Ethiopian women, Modules 1 and 2) + codebook |

`python download_data.py` fetches all of the above into `data/` (the `.dta`/`.xlsx` files are
git-ignored; the do-files and README are committed).

## Replication results

Adolescent = enrolment age 15 to 19, adult = 20+, as in the paper. Open-data cells use the
same derivations as the paper's `Amboko_analysis_2025.do` wherever the input variable exists.

Rows that match the paper exactly (every country and the pooled total): N, mean age, married or
partnered, completed secondary or higher, rural residence, very good health literacy, trimester at
first ANC, pregnancy intended, at least one danger sign, antenatal depression, self-rated health
poor/fair. Kenya cells differ by at most a few women because the open file has 1002 Kenyan women
versus the paper's 1009.

Rows that differ, and why:

- **Wealth "tertiles"**: South Africa matches exactly. Ethiopia and Kenya do not, because the
  paper's do-file builds those groups from wealth *quintiles* (Q1 / Q2 to Q4 / Q5), not tertiles,
  even though Table 1 labels them tertiles. The open file carries only tertiles. This is also why
  the paper's Ethiopian "middle" group is 56.5% of adolescents but 18.5% of adults.
- **Underweight, Ethiopia**: the paper uses `m1_low_BMI | m1_malnutrition`; the open file has a
  single `maln_underw` flag with a narrower definition (218 vs 272 women flagged).
- **First-ANC screening completeness index**: the paper averages 13 items including separate HIV,
  syphilis, blood sugar and haemoglobin tests, ultrasound, calcium and tetanus toxoid. The open file
  collapses blood tests into one item and ultrasound/calcium/TT are mostly missing, so my 7-item
  version is systematically higher. Not a replication.
- **First-ANC counselling index**: Ethiopia matches (31.4 vs 31.3 for adolescents; 33.7 vs 33.7
  for adults). The item-level counselling variables are absent for Kenya and South Africa.
- **Depression**: matching required following the do-file's country-specific choice,
  `depress` (PHQ-9 mild to severe) for Ethiopia and Kenya but `anc1depression` for South Africa.

Rows of Table 1 with no counterpart in the open data: employment, social support, gravidity
mean, risky health behaviour, intimate partner violence.

The full side-by-side table is in `table1_comparison.md` / `table1_comparison.csv`.

## Possible extensions with the open data

- The perinatal-outcome file (Q0YKOT) has stillbirth, neonatal death, late miscarriage and low
  birth weight by maternal age for the same cohort, which the paper does not report.
  Adolescents versus adults in the three African countries can be compared directly.
- The Ethiopia nutrition file (JGAXNG) has Module 2 follow-up visits (ANC visit counts, IFA
  adherence, blood tests), so the Ethiopia column of the follow-up ANC rows in Table 2 could be
  approximated for 811 of the 1000 women.

## Files

- `download_data.py` - fetches the Dataverse files listed above
- `replicate_table1.py` - the replication; writes `table1_comparison.{md,csv}`
- `data/Amboko_*.do`, `data/Amboko_Data_Readme.txt` - the paper's public code and README
- `requirements.txt` - Python dependencies (run with `uv run --with-requirements requirements.txt python replicate_table1.py`)
