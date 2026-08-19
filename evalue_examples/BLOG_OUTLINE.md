# Blog post outline: E-values (healthyalgorithms.com)

Working title candidates:

1. "How strong would a confounder have to be? E-values, Tylenol, and blood pressure"
2. "The E-value: putting a number on 'but what about confounding?'"
3. "Explaining it away: two stories about E-values"

Target length: ~1,200–1,500 words plus code snippets and one figure.
Draft will replace/absorb the current `evalue_examples/README.md` per the
blog-post conventions (draft lives in the subdirectory's README).

---

## 1. Hook (visual + minimal code)

- Open with a four-line code snippet: the E-value formula applied to the
  acetaminophen/autism hazard ratio.

  ```python
  >>> from evalue import evalue
  >>> evalue(1.05)   # adjusted HR, acetaminophen in pregnancy -> autism
  1.279...
  ```

- One-sentence framing: this number answers the reviewer question every
  observational study gets — "couldn't this just be confounding?" —
  with a threshold instead of a shrug.

## 2. TL;DR

- The E-value is the minimum strength of association (risk-ratio scale) an
  unmeasured confounder would need with *both* exposure and outcome to
  fully explain away an observed association. One formula:
  E = RR + √(RR(RR−1)).
- Story 1 (from Andrade's tutorial): acetaminophen in pregnancy and ASD,
  HR 1.05, E-value 1.28 — and the source study's sibling-control analysis
  shows a confounder of roughly that modest strength really existed.
- Story 2 (my exercise): SBP and heart-disease mortality in NHANES, where
  the "unmeasured" confounder (age) is actually measured — so we can watch
  the E-value bound get cashed in, and see a real effect survive.
- All code and data links in the repo; E-value implementation is ~20 lines
  of stdlib Python with tests pinned to published values.

## 3. The Problem: "couldn't this just be confounding?"

- Every observational association meets this objection; usually both sides
  argue by hand-waving ("we adjusted for a lot" vs "you can't adjust for
  everything").
- The right question is quantitative: *how strong* would the leftover
  confounding have to be? Some associations are fragile, some are not.

## 4. An Answer: the E-value

- Credit the entry point: Andrade's tutorial in J Clin Psychiatry (2026),
  which I found unusually clear; VanderWeele & Ding (2017) for the method.
- The formula, its interpretation, and the two E-values worth reporting
  (point estimate and the CI bound closer to the null).
- Applies to RRs directly; HRs and ORs pass as RRs when the outcome is
  uncommon; protective estimates get inverted first.
- Mention https://www.evalue-calculator.com/ for the no-code path.

## 5. Story 1: Tylenol in pregnancy and autism — a small E-value, cashed in

- Ahlqvist et al. (JAMA 2024, open access): 2.48M Swedish children,
  adjusted HR 1.05 (1.02–1.08) for gestational acetaminophen and ASD.
- E-values: 1.28 (point), 1.16 (CI bound) — replicated exactly from the
  published estimates; note that E-values need only the paper, not the
  (non-public) register data.
- The part the tutorial doesn't tell you (my addition): the same paper's
  sibling-control analysis gives HR 0.98 (0.93–1.04). Sibling comparison
  absorbs shared familial confounding — an "unmeasured confounder" of
  exactly the modest strength the E-value flagged as sufficient. The
  E-value said "easy to explain away"; the sibling design explained it away.
- Table: full-cohort / crude / sibling-control rows with E-values.

## 6. Story 2: SBP and heart disease — watching the E-value logic work

- Motivation (first person): to test my understanding I wanted an example
  with real confounding *and* (I believe) a real effect — systolic blood
  pressure and ischemic heart disease, confounded by age.
- Openly available individual-level data: NHANES 2003–2010 + NCHS
  public-use linked mortality through 2019 (heart-disease death as the
  public proxy for IHD incidence). MEC survey weights applied throughout.
- The game: pretend age is unmeasured.
  - Crude RR 4.20 for SBP ≥140 vs <140; E-value 7.88.
  - Age's actual strength: RR 3.14 with exposure, 13.59 with outcome —
    strong, but far short of 7.88, so age *cannot* fully explain the
    association away.
  - The Ding–VanderWeele bounding factor B = 2.72 predicts age could
    shrink the RR to at most 4.20/2.72 ≈ 1.55.
  - Age-standardization actually gives 1.51. The bound nearly touches.
- FIGURE here: the paired dot plot (heart-death risk by age stratum ×
  SBP group, log scale) — within strata the pairs sit close; the crude
  all-ages pair splays wide.
- The contrast that makes the pair of stories pedagogical: E-value 1.28
  fell to a modest familial confounder; explaining away RR 4.2 would take
  something stronger than age — one of the strongest confounders there is.

## 7. The code (brief, file-by-file)

- `evalue.py` — the formula, the CI variant, the bounding factor (stdlib).
- `test_evalue.py` — tests pinned to every number published in the
  tutorial and in VanderWeele & Ding (a habit worth blogging about in
  itself: pin your implementation to published values).
- `sbp_heart_death_age_confounding.py` — NHANES pipeline; note the survey
  weighting and the fully-observed 105-month window.
- GitHub links (permalinks once merged); Colab/Binder badge decision below.

## 8. How to use this in your work

- Report E-values for your headline observational estimates, point and CI.
- Read them against *named candidate confounders* with plausible
  strengths, as in Story 2 — an E-value alone is neither alarm nor alibi.
- Cautions: E-values address unmeasured confounding only (not selection
  bias or measurement error); small E-values are common near the null;
  the HR≈RR shortcut needs an uncommon outcome.

## 9. Challenges (reader exercises)

1. Compute E-values for the last observational paper you refereed.
2. Rerun the NHANES example with smoking or BMI as the "pretend
   unmeasured" confounder — how close does the bound come then?
3. Verify algebraically that E solves bias_factor(E, E) = RR.
4. Find another exposure-outcome pair with both a cohort and a
   sibling-control estimate; does the E-value predict which survive?

## 10. Further reading

- Andrade C, J Clin Psychiatry 2026 (the tutorial).
- VanderWeele & Ding, Ann Intern Med 2017 (the E-value).
- Ding & VanderWeele, Epidemiology 2016 (the bounding factor).
- Ahlqvist et al., JAMA 2024 (open access source study).
- A critique/caution piece for balance (verify citation at prose stage —
  candidates: Poole's Epidemiology commentary; VanderWeele's replies).

---

## Decisions for review before prose

1. **Hook**: code snippet first (as outlined) or lead with the NHANES
   figure and move code to section 4?
2. **Title**: pick from the three candidates (or suggest another).
3. **Placement**: draft replaces `evalue_examples/README.md` (repo
   convention: blog draft = subdirectory README), with current README
   content absorbed into it — OK?
4. **Runnable-code badge**: add a `tutorial.ipynb` with a Colab badge that
   downloads NHANES/LMF directly from CDC (self-contained, ~5 MB), or
   skip the notebook and link the scripts only?
5. **Length/depth**: is ~1,200–1,500 words right, or should Story 2 carry
   more of the derivation detail (bounding factor algebra)?
