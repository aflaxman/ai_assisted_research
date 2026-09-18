# UKB Copper Replication

*Copper & MASLD · external replication*

**Plan and punch list for replicating the NHANES dietary copper – liver fat
findings in UK Biobank**

Drafted 17 Aug 2026. Companion to `analysis/` notebooks 01–07. Repo copy:
`ukb_replication_plan.md`.

> Transcribed on 2026-09-17 from the private Claude Artifact
> [UKB Copper Replication](https://claude.ai/artifact/54aBXLtw8WFFgMhn3V9sf8).
> The original "repo copy" of this file lives in the not-yet-recovered local
> repo (see [README](README.md)). If the two differ, prefer the repo copy.

## Why this replication, in three sentences

Every current result comes from one survey system (NHANES), one dietary
instrument (the 24-h recall), and one food-composition database (FNDDS) — UK
Biobank breaks all three dependencies at once, with a better liver-fat measure
(MRI-PDFF) and a different country's food patterning. It adds the one design
element NHANES cannot: **diet measured years before imaging**, so reverse
causation is directly testable. And it has the power for the one question
NHANES left open — whether the copper-specific component *beyond diet quality*
(≈ −4 dB/m-equivalent, n.s.) clears zero.

## 1 · Pre-specified targets — register before modeling

| # | Hypothesis | NHANES anchor | Counts against if |
|---|---|---|---|
| H1 | Usual copper density inversely associated with PDFF (fully adjusted) | ≈ −22 dB/m CAP per doubling (corrected); ≈ 15–30% relative PDFF | CI excludes even a 5% relative reduction |
| H2 | Threshold dose–response, plateau ≈ 0.8 mg/1000 kcal | non-linearity p < 0.001 | monotone-linear or increasing |
| H3 | **Copper-specific component survives diet-quality adjustment** | ≈ −4 to −5, n.s. at NHANES power | tight null CI after adjustment — the decisive test |
| H4 | Steatosis (PDFF ≥ 5.5%) OR < 1 per doubling | OR 0.70 observed / 0.49 corrected | OR CI at/above 1 |
| H5 | Liver-specific: PDFF association exceeds VAT/ASAT (per SD) | VAT was pattern-confounded | all depots move identically |
| H6 | Heavy drinkers: attenuation at most, not reversal | −5 to −8 vs −12; interaction p ≈ 0.4 | strong qualitative interaction |
| H7 | **Prospective:** baseline copper predicts later PDFF | untestable in NHANES | prospective null + cross-sectional signal → reverse causation |

## 2 · Data elements

Field IDs marked *verify* must be confirmed in the
[Showcase](https://biobank.ndph.ox.ac.uk/showcase/) before the basket is
finalized (punch-list step 4).

| Purpose | Source / fields | Status |
|---|---|---|
| Liver fat (primary) | MRI PDFF, imaging instances — field `22436` | verified vs literature |
| Fibro-inflammation (secondary) | Liver cT1 — `22437?` | verify |
| VAT / ASAT (specificity) | AMRA volumes — `22407`, `22408` | verify |
| Copper + co-nutrients | Oxford WebQ *updated* nutrients (Perez-Cornago 2021; UK Nutrient Databank): copper, zinc, energy (kJ → ÷4.184), fibre, sat fat, MUFA, PUFA, sugars, sodium, potassium, folate, vitamin C, alcohol — `260xx`, instances 0–4 | copper confirmed via publications; IDs verify |
| WebQ quality | completion timestamps, "typical diet" flag — `20080`+ | verify |
| Demographics / SES | age `21003`, sex `31`, ethnicity `21000`, Townsend `22189?`, education `6138` | Townsend verify |
| Lifestyle / anthropometry | smoking `20116`; activity `22032`/`22037–9`; BMI `21001`, waist `48` | standard |
| MASLD criteria | HbA1c `30750`, HDL `30760`, **triglycerides `30870`** (available here, unlike NHANES full sample), glucose `30740`, BP `4080`/`4079`, meds `6153`/`6177`, diabetes `2443` | standard |
| Liver-disease exclusions | HES ICD-10 `41270`/`41280` (B15–19, C22, K70–77), self-report `20002` | standard |
| Supplements | `6155`, `20084` — zinc listed, **no copper doses**: the supplement quasi-test does not replicate; food-only exposure | limitation |

## 3 · Cohort and analysis specification

- **Cohorts:** ≥ 2 completed WebQs with plausible energy (mirror NHANES
  bounds; UKB-conventional bounds as sensitivity) and non-missing copper.
  Cross-sectional: any WebQ + valid PDFF. Prospective (H7): pre-imaging WebQs
  only. Exclude prior liver disease before imaging.
- **Exposure:** copper mg/1000 kcal, log₂ (per doubling); food-energy
  (non-alcohol) denominator sensitivity; usual-intake calibration from repeated
  WebQs (`measurement.py` ports directly; per-person reliability
  λ_k = σ²_b / (σ²_b + σ²_w / k)).
- **Models:** M1 crude → M2 + age, sex, ethnicity, imaging center → M3 +
  Townsend, education, smoking, alcohol category, energy → + diet quality →
  M5 + physical activity. No survey weights (volunteer cohort — internal
  validity is the target; state this); robust SEs.
- **Outcomes:** log-PDFF primary (median ≈ 2.1%, heavily skewed); PDFF ≥ 5.5%
  logistic; VAT/ASAT per-SD comparison; MASLD with full cardiometabolic
  criteria.
- **Diet quality:** HEI-2020 is US-specific — primary adjustment is the
  portable nutrient-based DQS (from `causal.py`, enables NHANES comparison);
  secondary, a WebQ food-group score. Carry the double-measurement-error caveat
  (notebooks 06/07).
- **Battery:** splines (H2), negative-control nutrients (vitamin C, folate),
  zinc mutual adjustment, sex/alcohol interactions, E-values, exclusion
  sensitivity, corrected estimates. CEM optional at this n.

## 4 · Power

With diet × PDFF overlap of ~15,000–25,000 (verify at step 4 *before paying
fees*), SD(log₂ copper density) ≈ 0.45 and SD(log PDFF) ≈ 0.9 give
SE(β) ≈ 0.016 — an 80%-power MDE of ≈ **5% relative PDFF change per
doubling**, several-fold below the NHANES-implied effect. Power is not the
constraint; H3 finally gets a decisive confidence interval.

## 5 · Punch list

0. **Check for an existing UKB application at IHME/UW** *(days · do this
   first)*. Ask the research office and colleagues with UKB papers. Joining an
   approved application as a collaborator (if its scope and basket cover these
   fields) is weeks faster and cheaper than a new application.
1. **Register as an approved researcher** *(~1–2 weeks elapsed)*. AMS at
   `ams.ukbiobank.ac.uk`, institutional email.
2. **Submit the application (if step 0 fails)** *(~4–8 weeks review)*. Scope:
   "diet quality, micronutrient intake (copper) and hepatic/visceral fat by
   MRI." New applications run on the Research Analysis Platform (RAP /
   DNAnexus). Budget the access-tier fee plus modest RAP compute (check the
   current fee schedule).
3. **Set up the RAP project and billing** *(days)*.
4. **Verify field IDs and overlap counts in Showcase** *(half a day)*. Copper
   and co-nutrients in the updated WebQ release (`260xx`); PDFF `22436`
   instances; cT1; VAT/ASAT; Townsend; activity fields. Record the WebQ × PDFF
   overlap n — **the go/no-go number**.
5. **Build the field basket; dispense to RAP** *(days)*.
6. **Extraction + QC notebook** *(1–2 days)*. Reproduce published anchors
   before any modeling: mean PDFF ≈ 3.9% / median ≈ 2.1%; WebQ copper mean
   ≈ 1.2–1.5 mg/d; WebQ count distribution.
7. **Pre-register the §1 targets (OSF)** *(half a day · before unblinding any
   exposure–outcome model)*.
8. **Port the pipeline** *(~1 week)*. Copy the cohort-builder pattern (full
   frame + `analysis` mask + flow table); `measurement.py` and the DQS port
   as-is; `survey.py` not needed. Git repo, commit as you go.
9. **Run cross-sectional (H1–H6), then prospective (H7)** *(2–3 days)*.
   Corrected estimates and E-values throughout.
10. **Write up against the §1 table — whatever the answer** *(2–3 days)*. A
    clean null is publishable alongside the NHANES paper.

> **Main risks.** Imaging × diet overlap smaller than assumed (check at step 4,
> before fees); access fees and authorization lead time; copper field
> granularity in the updated WebQ release. Healthy-volunteer selection limits
> prevalence estimates, not effect estimates — say so in the write-up.

## Key references

- PDFF characterisation and field `22436`: Wilman et al., *PLOS ONE* 2017;
  Gnatiuc Friedrichs et al., *Obesity* 2023.
- Updated WebQ nutrient calculation (UK Nutrient Databank): Perez-Cornago et
  al., *Eur J Nutr* 2021.
- Precedent for WebQ-derived dietary copper: UK Biobank copper/zinc cohort
  analyses (e.g., 2024 copper–zinc–IBS study).

---

Companion artifact: *Copper & MASLD* (the NHANES review and analysis plan). The
repo copy of this plan lives at `ukb_replication_plan.md`; NHANES anchors cite
executed notebooks 01–07 at commit `1ef987c`.
