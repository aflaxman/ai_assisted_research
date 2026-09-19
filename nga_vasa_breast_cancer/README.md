# Nigeria 2024 Verbal and Social Autopsy: breast cancer and maternal mortality

Exploration of IHME's copy of the 2024 Nigeria Verbal and Social Autopsy (NVASA)
microdata and the matching 2024 Nigeria DHS recodes. Two notebooks:

- `breast_cancer_exploration.ipynb`: what the VASA says about breast cancer
  diagnosis, care-seeking, treatment, and death among women aged 12 to 49.
- `maternal_mortality.ipynb`: a DHS-style direct calculation of the maternal
  mortality ratio (MMR) from the VASA deaths and NDHS exposure, for both VASA
  sample frames, with causes of maternal death, a sibling-survival estimate from
  the same NDHS, and a comparison against DHS, GBD, and MMEIG series.

## Quickstart

```bash
cd nga_vasa_breast_cancer
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace breast_cancer_exploration.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace maternal_mortality.ipynb
```

Both notebooks read the data directly from the J: drive (read-only) and write only
figures into this directory.

## The data

| Location on the cluster | What it is |
|---|---|
| `/home/j/DATA/DHS_PROG_VASA/NGA/2024/NGA_VASA8_2024_VA_NGVA8JFL_Y2026M09D14.DTA` | VASA microdata: 4,879 deaths, 1,416 variables, Latin-1 encoded |
| same folder, `.DO` `.DCT` `.FRQ` `.FRW` | Stata labels, dictionary, unweighted and weighted frequency tables |
| same folder, `NGA_VASA8_2024_REP_FINAL_QUEST_Y2026M08D04.PDF` | The 409-page NVASA main report (June 2026) with questionnaire, physician coding criteria, and ICD-11 cause list |
| same folder, policy briefs and `DENOMINATOR_DEFINITIONS` | Under-5 and maternal briefs; official analytic denominators with Stata code |
| `/home/j/DATA/DHS_PROG_DHS/NGA/2023_2024/` | 2024 NDHS recodes. The household member file gives exposure; the women's file gives birth histories and the sibling survival module |
| `/home/j/DATA/DHS_PROG_DHS/NGA/CRUDE/NGA_DHS8_2023_2024_REP_FINAL_SUMMARY.zip` | NDHS 2024 final report (FR395). It has no adult or maternal mortality chapter |

Treat every J: location as read-only. Set `NVASA_DIR` or `NDHS_DIR` if your copies
are elsewhere.

The VASA has three modules: 947 stillbirth/neonatal, 1,591 child, and 2,269 adult
deaths. The adult module is women aged 12 to 49 only. Cause of death is
physician-certified verbal autopsy coded to ICD-11 in `vicd`. Deaths come from two
frames: NDHS-sampled households, which carry the NDHS household weight and have
exposure denominators, and extra screened households, which have neither.

## Files here

| File | Purpose |
|---|---|
| `nga_vasa.py` | Loader, label helpers, official denominator, breast cancer flag, duration and care-pathway helpers, and the disclosure-control functions |
| `mmr.py` | DHS-style mortality methods: woman-years from the household roster, GFR from birth histories, standardised rates, jackknife CIs, and the sibling-survival method |
| `breast_cancer_exploration.ipynb` | Breast cancer notebook |
| `maternal_mortality.ipynb` | Maternal mortality notebook |
| `fig1_durations.png`, `fig2_care_pathway.png` | Breast cancer figures |
| `fig3_mmr_comparison.png`, `fig4_household_vs_sibling.png` | Maternal mortality figures |
| `pyproject.toml`, `uv.lock` | Environment |

## Maternal mortality: headline findings

**The report's method reproduces.** Rebuilding woman-years from the NDHS household
roster matches the report's exposure table to within 0.1%, and the standardised
maternal mortality rate is 0.91 per 1,000 in both.

| Estimate, women 15–49, 5 years before the 2024 NDHS | Ratio per 100,000 live births | 95% CI |
|---|---|---|
| NVASA report MMR (household deaths + verbal autopsy) | 572 | 475–688 |
| This notebook, same method, NDHS frame | 576 | 469–683 |
| This notebook, pregnancy-related (timing only) | 580 | 472–688 |
| Indirect, screened frame's maternal share applied to NDHS-frame rates | 625 | |
| Indirect, both frames combined | 609 | |
| Sibling histories, same survey, 5-year window, pregnancy-related | 376 | 302–450 |
| Sibling histories, 7-year window, pregnancy-related | 420 | 355–486 |
| Sibling histories, 7-year window, NDHS 2018 MMR definition (42 days, excluding accidents and violence) | 369 | 308–430 |
| NDHS 2018 published MMR, sibling histories, 7 years | 512 | 447–578 |
| MMEIG 2023 (published 2025) | 993 | |
| GBD 2023, years 2021 to 2023, read from IHME slides | 385–405 | |

**Both frames tell the same cause story.** The maternal share of women's deaths is
27% in the NDHS frame and 30% in the screened frame. In both, obstetric haemorrhage
is the leading cause (about 37%), followed by unspecified maternal causes, indirect
causes, hypertensive disorders, abortion-related complications, and infection.
Direct causes are about 70% of maternal deaths.

**The sibling method disagrees with the household method inside the same survey.**
The 2024 NDHS collected sibling histories but published no estimate from them. Run
here, they give an all-cause female mortality rate of 2.4 per 1,000 against 3.3 from
the household roster, with the shortfall concentrated at ages 40 to 49, and a
pregnancy-related ratio around 400 rather than 580. The 2018 sibling estimate was
3.18 per 1,000, so the 2024 sibling data imply an implausibly fast decline and most
likely under-report sister deaths.

**The sibling code is validated.** Run on the 2018 NDHS women's file with a 7-year
window, it reproduces every published 2018 figure: female mortality 3.18 per 1,000
(1.59 at ages 15 to 19, 5.86 at 45 to 49), PRMR 556, and MMR 512 with CI 448–577
against the published 447–578. The published 2018 MMR counts deaths within 42 days
and excludes accidents and violence; that definition is what the 369 above uses.

**Where the survey lands between GBD and MMEIG.** The household-roster all-cause rate
for women 15 to 49 agrees with GBD's envelope and is about half of MMEIG's. The
household-death MMR sits well above GBD and well below MMEIG. The maternal fraction
observed in the VASA, applied to GBD's own envelope, gives an MMR near 570. Neither
survey method supports an MMR near 1,000.

## Breast cancer: findings

The main report only says "Neoplasms 6%" for women's causes of death, so this is
new tabulation.

| Measure | Value |
|---|---|
| ICD-11 breast cancer deaths in the adult module | 58 of 2,269 |
| Share of all neoplasm deaths | 58 of 131 |
| Breast cancer deaths in the official 15–49 denominator | 18 of 607 |
| Weighted share of all deaths, women 15–49 | 2.7% |
| Women reporting a breast lump or ulcer | 64, of whom 56 were coded to breast cancer |
| Ages | 22 to 49, mostly 30 to 45 |

A health professional had told the family it was cancer in 46 of the 51 cases where
the question was asked. The health worker's free-text cause names breast cancer in
25 of 58. Only 9 had a medical certificate of cause of death and 51 had no death
registration. Median lump duration before death was 6 months; 15 women reported a
year or more.

Almost all sought care and 42 reached a hospital at some point, but 28 went
somewhere else first. Sixteen used a traditional or non-formal provider at some
stage. The spouse was the strongest voice in the care decision in 24 cases, the
woman herself in 13. Problems accessing care were reported for 26 women, dominated
by cost of health care (24), cost of transport (18), and distance (14).

The treatment module has no oncology items. Among 44 treated women: IV fluids 40,
injectable antibiotics 37, transfusion 14, operation 15. Chemotherapy appears in
fewer than five narratives and radiotherapy in none. Several narratives describe
fear or refusal of mastectomy, or delay while using traditional medicine.

## Disclosure control

DHS Program microdata may not be redistributed, and the repo is public. Rules for
anything committed here:

- Every count table passes through `nga_vasa.redact` or `redact_crosstab`, which
  pools or masks unweighted cells below `nga_vasa.MIN_CELL` (5) and applies
  secondary suppression to two-way tables. Histogram bins are merged with
  `redact_bins` until each holds at least five deaths.
- Free-text narratives are never displayed in committed output. Set
  `SHOW_NARRATIVES = True` locally to read them.
- `.gitignore` blocks microdata extracts.
- No pull requests for this subdirectory; local commits only.

## Limitations

- Breast cancer: 58 deaths in total, 18 in the representative denominator; women
  aged 50 and older and men are absent; narratives are short; recall reaches back
  up to six years.
- Maternal mortality: verbal autopsy misclassification; household rosters miss
  women who died away from home and households that dissolved; sibling histories
  miss unknown or forgotten sisters; jackknife CIs cover sampling error only.
- GBD values in the comparison are read from the March 2026 IHME slides, not pulled
  from the GBD database.
