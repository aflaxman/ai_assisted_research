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
