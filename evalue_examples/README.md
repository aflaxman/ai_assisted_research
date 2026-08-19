# E-value examples: replicating Andrade's tutorial with open data

Andrade's tutorial on E-values ([J Clin Psychiatry
2026;87(1):26f16324](https://www.psychiatrist.com/jcp/e-value-regression-useful-easily-understood-easily-applied-statistic/))
works through an example relating gestational **acetaminophen** exposure to
autism spectrum disorder. (The drug is acetaminophen, not aspirin — an easy
mix-up.) This directory replicates that example from openly available data
and adds a companion example, built on NHANES, where the confounder (age) is
measured, so the E-value logic can be watched in action.

## TL;DR

- The tutorial's E-values (1.28 for HR 1.05; 1.16 for CI bound 1.02)
  **reproduce exactly** from the open-access source paper — no restricted
  data needed, because E-values are computed from published estimates.
- The underlying cohort is Swedish national register data, which is *not*
  public. The openly available replication data is the open-access JAMA
  paper itself (summary estimates, extracted to [`data/`](data/)).
- A fully open individual-level companion example: in NHANES, crude RR of
  hypertensive-range SBP for heart-disease death is **4.2**, but age alone
  (a confounder with RR_EU 3.1, RR_UD 13.6) can account for a bias factor
  of 2.7, and age-standardization collapses the RR to **1.5** — matching
  the bound's prediction almost exactly.

## The E-value in one line

For a risk ratio RR (or HR, or OR with an uncommon outcome),

    E = RR + sqrt(RR × (RR − 1))

is the minimum strength of association (risk-ratio scale) an unmeasured
confounder must have with *both* exposure and outcome to fully explain away
the observed association (VanderWeele & Ding 2017). Implementation:
[`evalue.py`](evalue.py); tests pinned to the tutorial's published values:
[`test_evalue.py`](test_evalue.py).

## Example 1: acetaminophen and ASD (Ahlqvist et al. 2024)

The tutorial's source study is open access: Ahlqvist et al., *Acetaminophen
Use During Pregnancy and Children's Risk of Autism, ADHD, and Intellectual
Disability*, [JAMA 2024;331(14):1205–1214](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004836/),
a cohort of 2,480,797 Swedish children (7.49% exposed). The individual-level
register data cannot be shared publicly under Swedish law, but every number
the E-value calculation needs is in the paper:

| Analysis | Estimate (95% CI) | E-value (point) | E-value (CI) |
|---|---|---|---|
| Full cohort, adjusted | HR 1.05 (1.02–1.08) | **1.28** | **1.16** |
| Crude (from 10-yr cumulative incidence 1.53% vs 1.33%) | RR 1.15 | 1.57 | — |
| Sibling control | HR 0.98 (0.93–1.04) | 1.16 | 1.00 |

The first row reproduces the tutorial's 1.28 and 1.16 exactly
([`replicate_acetaminophen_asd.py`](replicate_acetaminophen_asd.py)). The
sibling-control row, which the tutorial does not discuss, completes the
story: an unmeasured confounder of strength just 1.28 could nullify the
full-cohort association, and sibling comparison — which absorbs shared
familial confounding — did exactly that.

## Example 2: age confounds SBP and heart-disease death (NHANES)

![Paired dot plot: within age strata the SBP groups' risks sit close together, but the crude all-ages comparison splays wide](outputs/sbp_heart_death_by_age.png)

Same logic, but with the confounder measured, using genuinely open
individual-level data: NHANES 2003–2010 adults (n = 20,109 with measured
SBP) linked to the NCHS public-use mortality files through 2019. Exposure
is mean measured SBP ≥ 140 mm Hg; outcome is death from diseases of heart
(UCOD_LEADING = 1, the closest public proxy for IHD incidence) within 105
months. All estimates are MEC-weighted
([`sbp_heart_death_age_confounding.py`](sbp_heart_death_age_confounding.py)).

Pretend age were unmeasured:

- Crude RR = **4.20**, E-value = **7.88**: a confounder would need
  associations of 7.9 with both exposure and outcome to explain this away.
- Age's actual strength: RR_EU = 3.14 (age 60+ is 56% prevalent among
  exposed vs 18% among unexposed), RR_UD = 13.59 (heart-death risk, 60+ vs
  younger). Strong — but the joint E-value condition is **not met**.
- The Ding–VanderWeele bounding factor B = RR_EU·RR_UD/(RR_EU+RR_UD−1) =
  2.72, so age alone could shrink the crude RR to at most 4.20/2.72 = 1.55.
- Age-standardizing across four age strata gives RR = **1.51** (E-value
  2.38) — right at the bound's prediction.

So age accounts for most, but not all, of the crude association — unlike
the acetaminophen example, the adjusted estimate stays above the null,
consistent with SBP truly causing heart disease. The contrast is the
pedagogical point: a small E-value (1.28) fell to a modest familial
confounder; explaining away RR 4.2 would need a confounder far stronger
than even age, one of the strongest confounders in epidemiology.

Caveats: point estimates only (design-based CIs would need SDMVPSU/SDMVSTRA
Taylor linearization); antihypertensive treatment is ignored, so the
exposed group mixes treated and untreated hypertension; two young strata
have <30 unweighted deaths and are flagged.

## Quickstart

```bash
cd evalue_examples
uv venv && uv pip install -r requirements.txt
uv run pytest                                      # 10 tests vs published values
uv run python replicate_acetaminophen_asd.py       # example 1
uv run python sbp_heart_death_age_confounding.py   # example 2 (data already in repo)
```

Example 2 reads NHANES and mortality files from
`../nhanes_mortality_fibrosis/data/raw/`, already committed to this
repository; fresh copies come from the CDC URLs in that project's
`01_data_download.ipynb`.

## Files

- `evalue.py` — E-value, CI E-value, and bounding factor (stdlib only)
- `test_evalue.py` — tests pinned to values published in the tutorial and
  in VanderWeele & Ding (2017)
- `data/ahlqvist2024_jama_*.csv` — summary estimates extracted from the
  open-access JAMA paper, with row-level source citations
- `replicate_acetaminophen_asd.py` — example 1
- `sbp_heart_death_age_confounding.py` — example 2
- `outputs/` — results table and figure

## References

- Andrade C. The E-value in regressions: a useful, easily understood, and
  easily applied statistic. J Clin Psychiatry. 2026;87(1):26f16324.
- Ahlqvist VH, Sjöqvist H, Dalman C, et al. Acetaminophen use during
  pregnancy and children's risk of autism, ADHD, and intellectual
  disability. JAMA. 2024;331(14):1205–1214. doi:10.1001/jama.2024.3172
  ([open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC11004836/))
- VanderWeele TJ, Ding P. Sensitivity analysis in observational research:
  introducing the E-value. Ann Intern Med. 2017;167(4):268–274.
- Ding P, VanderWeele TJ. Sensitivity analysis without assumptions.
  Epidemiology. 2016;27(3):368–377.
- Online calculator: <https://www.evalue-calculator.com/>
- NHANES: <https://wwwn.cdc.gov/nchs/nhanes/> · Linked mortality files:
  <https://www.cdc.gov/nchs/data-linkage/mortality-public.htm>
