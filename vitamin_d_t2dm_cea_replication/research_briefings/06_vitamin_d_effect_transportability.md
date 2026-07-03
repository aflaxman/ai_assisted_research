I have all the primary-source data verified. Here is the briefing.

---

# Intervention-Effect Parameterization — Vitamin D for Prediabetes → T2D Progression

**Scope:** relative-effect parameter for the prediabetes→T2D transition in a vitamin-D CEA (Briody 2026 assumes HR 0.85 / 15% RRR, D3 4000 IU/day, 85% adherence, no effect modification), and whether/how to make it location-specific for a GBD re-base and other-location transport.

*Source attribution: the trial and review findings below are drawn from PubMed-indexed articles; DOIs are linked inline. Primary numbers for the pooled meta-analysis were extracted from the full text of the Pittas 2023 paper.*

---

## 1. The Pittas 2023 IPD meta-analysis — the source of the 15% RRR

Pittas AG, Kawahara T, Jorde R, et al. *Ann Intern Med* 2023;176(3):355-363. [DOI: 10.7326/M22-3018](https://doi.org/10.7326/M22-3018) · PMID 36745886 · [full-text PDF](https://d2dstudy.org/wp-content/uploads/2024/02/2023-Pittas-AnnInternMed-vitamin-D-and-t2DM-IPD.pdf) · [ACP page](https://www.acpjournals.org/doi/10.7326/M22-3018)

**Design.** IPD pool of the 3 trials designed specifically for prediabetes: **D2d** (USA, cholecalciferol 4000 IU/day), **Tromsø** (Norway, cholecalciferol 20,000 IU/week), **DPVD** (Japan, eldecalcitol 0.75 µg/day — an *active* analog). n = 4190 (2097 vitamin D / 2093 placebo). Median follow-up 3.0 y. Baseline: mean age 61, 44% women, 51% White/European, 33% Asian, 15% Black, mean BMI 30, **mean baseline 25(OH)D 63 nmol/L (25 ng/mL)** — i.e., the pooled cohort was on average vitamin-D *replete*.

**Primary effect (this is the 15% RRR).**
- Unadjusted ITT HR **0.88 (95% CI 0.77–0.99)**.
- Adjusted (age, sex, BMI, race, HbA1c) ITT HR **0.85 (0.75–0.96)** → **15% RRR**.
- 3-year absolute risk reduction **3.3% (0.6–6.0%)**; **NNT = 30**.
- The three trials individually were underpowered and non-significant: risk reductions of **10% (Tromsø), 12% (D2d), 13% (DPVD)** — nearly identical, which is why pooling was needed.

**Regression to normoglycemia (secondary, beneficial second channel).** Vitamin D increased reversion to normal glucose regulation by 30%: **rate ratio 1.30 (1.16–1.46)**. If your model has a prediabetes→normoglycemia transition, vitamin D should raise it too — Briody may under-count benefit if only the progression arrow is modified.

**Achieved / intratrial 25(OH)D dose-response (the striking finding).** In the two *cholecalciferol* trials there was a strong, significant interaction (p<0.001) between the on-treatment 25(OH)D level maintained and diabetes risk. Among those *assigned to cholecalciferol*, vs. maintaining 50–74 nmol/L (20–29 ng/mL):
- 100–124 nmol/L (40–50 ng/mL): HR **0.38 (0.27–0.55)**, 3-yr ARR 11.4%.
- ≥125 nmol/L (≥50 ng/mL): HR **0.24 (0.16–0.36)**, 3-yr ARR **18.1% (11.7–24.6%)** → 76% reduction.
- No such gradient in the placebo arm — argues against pure reverse-causation/healthy-user confounding, though achieved level is *not* randomized, so this is suggestive not causal.

**D2d ancillary (Chatterjee 2023) — the ≥40 ng/mL subgroup.** Chatterjee R, et al. *Am J Clin Nutr* 2023;118(1):59-67. [DOI: 10.1016/j.ajcnut.2023.03.021](https://doi.org/10.1016/j.ajcnut.2023.03.021) · PMID 37001590 · PMC10447481. Achieving intratrial mean 25(OH)D **≥40 ng/mL** vs. lower cut diabetes risk in every race group: **Black HR 0.51 (0.29–0.92), White HR 0.42 (0.30–0.60), Asian HR 0.39 (0.14–1.11)**; benefit concentrated in BMI <40. No significant interaction by race or BMI → the *achieved level*, not race/weight per se, drove benefit; authors propose **≥40 ng/mL as the target** for prevention.

**Safety** (all non-significant): kidney stones RR 1.17, hypercalcemia 2.34, hypercalciuria 1.65, death 0.85.

---

## 2. Does the effect vary by BASELINE vitamin D status? (the transportability crux)

This is the key uncertainty for moving the effect to more-deficient populations, and the evidence is genuinely mixed.

**Formal interaction tests were NOT significant.** The IPD meta-analysis states plainly: *"The effect of vitamin D did not differ in prespecified subgroups"* (age, sex, BMI, race, glycemic risk, calcium intake). So by the pre-registered test, there is **no proven effect modification by baseline status** — the honest default is a constant effect.

**But the point estimates lean toward larger benefit in the deficient / lean:**
- **Baseline 25(OH)D <30 nmol/L (<12 ng/mL), n=224: HR 0.58 (0.35–0.97)** vs. 0.85 overall. Directionally larger, but small n and the interaction was not significant.
- In the two cholecalciferol trials there *was* a significant **BMI** interaction (p=0.023): BMI <31.3 HR **0.74 (0.60–0.90)** vs. BMI ≥31.3 HR **1.01 (0.84–1.22)**. No BMI interaction for the eldecalcitol (active-analog) trial (p=0.82) — consistent with the mechanism that obese people convert a fixed cholecalciferol dose to less circulating 25(OH)D.
- The 2026 Pittas mini-review summarizes the current read as *"greater benefit among those with low baseline 25(OH)D levels or a BMI less than 30 kg/m²."* Amrein K, Kim SH, …, Pittas AG. *Metabolism* 2026;178:156566. [DOI: 10.1016/j.metabol.2026.156566](https://doi.org/10.1016/j.metabol.2026.156566) · PMID 41707752.

**Why the trials still showed benefit despite enrolling replete people:** D2d (NEJM 2019, Pittas et al., HR 0.88 [0.75–1.04], [DOI: 10.1056/NEJMoa1900906](https://doi.org/10.1056/NEJMoa1900906)) enrolled ~78% vitamin-D-*sufficient* participants; Tromsø and FIND enrolled frankly replete cohorts (means 60–75 nmol/L). The benefit therefore does **not require baseline deficiency** — it appears to operate by pushing 25(OH)D into a high therapeutic range (≥100–125 nmol/L), a range most placebo participants never reach.

**Implication for more-deficient (many LMIC) populations — plausible but unproven larger effect.** Two mechanisms point to *larger* benefit where baseline deficiency is common:
1. A fixed 4000 IU/day raises 25(OH)D *more* from a low start, so more people cross the ≥40–50 ng/mL therapeutic threshold that carries the big HRs (0.24–0.38).
2. The deficient subgroup point estimate (HR 0.58) and the observational literature both trend that way.

But three caveats keep this uncertain: (a) the randomized interaction-by-baseline-status was **not** statistically significant; (b) the deficient stratum was small; (c) BMI cuts the other way — where obesity is high, the fixed-dose 25(OH)D increment is blunted (D2d BMI≥31.3 HR ≈ 1.0; FIND BMI≥30 HR 1.00). So high-deficiency-but-high-obesity settings may not gain much. Net: a **defensible directional hypothesis (larger effect in deficient, lean populations), not an established, quantifiable modifier.**

---

## 3. Population 25(OH)D distributions by country — for a location-specific effect

**Global pooled distribution** (Cui et al., *Front Nutr* 2023, 7.9M participants, 81 countries; [DOI: 10.3389/fnut.2023.1070808](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2023.1070808/full), PMC10064807):

| Threshold | Global | E. Mediterranean | Europe | SE Asia | Africa | Americas | W. Pacific |
|---|---|---|---|---|---|---|---|
| <30 nmol/L (severe) | 15.7% | **35.2%** | 18.0% | 22.0% | 8.0% | **5.5%** | 10.0% |
| <50 nmol/L (deficient) | 47.9% | 71.8% | 53.0% | — | 18.9% | 18.9% | — |
| <75 nmol/L | 76.6% | 85.1% | — | — | 55.3% | — | — |

By World Bank income, deficiency is worst in **lower-middle-income** countries (<30: 26.7%, <50: 56.0%). A concordant 2025 systematic review (102 countries; PMC12670000) reports pooled mean 53.9 nmol/L, 18% <30, 47% <50, 75% <75.

**Standardized-reference sources** for cleaner country inputs (assay-harmonized, avoiding the assay heterogeneity that plagues raw meta-analyses):
- **Cashman KD, et al.** "Vitamin D deficiency in Europe: pandemic?" *Am J Clin Nutr* 2016;103(4):1033-44 ([DOI: 10.3945/ajcn.115.120873](https://doi.org/10.3945/ajcn.115.120873)) — standardized VDSP data, ~13% <30 nmol/L and ~40% <50 nmol/L across Europe, with strong ethnicity/season gradients.
- **Roth DE, et al.** "Global prevalence and disease burden of vitamin D deficiency: a roadmap for action in LMICs." *Ann N Y Acad Sci* 2018 (PMC7309365) — the reference for LMIC deficiency burden.

**GBD status — vitamin D is NOT a GBD risk factor.** In the GBD 2021 comparative risk assessment (**88 risk factors**, 631 risk–outcome pairs; GBD 2021 Risk Factors Collaborators, *Lancet* 2024, [DOI: 10.1016/S0140-6736(24)00933-4](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)00933-4/fulltext)), **vitamin D deficiency / low 25(OH)D is not included** as an exposure. GBD quantifies other micronutrient deficiencies (vitamin A, zinc, iron/anemia) and captures bone via **low bone mineral density**, and captures diet-calcium within aggregate "dietary risks" — but there is **no vitamin-D-deficiency risk–outcome pair, and no vitamin-D→diabetes pathway, in GBD.** Practical consequence: you cannot pull a ready-made GBD 25(OH)D exposure surface; country 25(OH)D distributions must come from the external reviews above.

**Two ways to make the effect location-specific:**
- *(a) Hold RRR constant, let absolute benefit float* (recommended base — see §5): the location signal enters through GBD's **location-specific prediabetes prevalence and progression incidence**. Same 15% RRR × higher baseline incidence = more DALYs averted, which is well-grounded and needs no contested effect-modifier.
- *(b) Modulate RRR by baseline-deficiency prevalence* (scenario only): map each country's %<30 or %<50 nmol/L to a shifted HR, using the deficient-stratum HR (~0.58–0.74) as an optimistic bound for high-deficiency settings and the replete-population estimates (VITAL 0.91, FIND 0.86, D2d BMI-high ≈1.0) as the attenuated bound. Because the randomized interaction was non-significant, treat (b) as sensitivity, not base case.

---

## 4. General-population (non-prediabetes) effect — relevance if extending beyond prediabetes

Two large trials in **unselected, largely replete general populations** show weaker, non-significant effects — a caution against applying the 15% RRR to normoglycemic incidence:

- **VITAL-T2D (USA):** HR **0.91 (0.76–1.09)**, n=22,220 without diabetes, median 5.3 y, general older US adults (mean baseline 25(OH)D replete). By baseline 25(OH)D: <20 ng/mL HR 1.17 (0.72–1.89), ≥20 HR 0.93 (0.71–1.21), p-interaction 0.44. (VITAL ancillary, PMC11978796; the original 2019 VITAL diabetes secondary endpoint was HR ≈0.96.)
- **FIND (Finland):** Virtanen JK, et al. *Diabetologia* 2024;68(4):715-726. [DOI: 10.1007/s00125-024-06336-9](https://doi.org/10.1007/s00125-024-06336-9) · PMID 39621103. Healthy older adults not at high diabetes risk, baseline 25(OH)D 74.5 nmol/L (replete). Combined vitamin D arms vs placebo HR **0.86 (0.58–1.29)** (1600 IU 0.81; 3200 IU 0.92). Strong BMI interaction (p<0.001): benefit only in BMI <25 (HR 0.43), none at BMI ≥25 (HR ≈1.0).

**Read-across:** the point estimates (0.86–0.91) are consistent with a *modest* real effect even in general populations, but CIs cross 1 and benefit concentrates in the lean/deficient. **Do not transport the 15% RRR to the normoglycemia→prediabetes transition or to general incident diabetes in the base case** — the effect is best-supported specifically for the **prediabetes→T2D** arrow. VITAL/FIND are useful as (i) a lower-bound scenario for the effect and (ii) evidence that BMI, not just baseline 25(OH)D, gates response.

---

## 5. Recommendation for the GBD re-base and other-locations goals

**Base case: keep a constant 15% RRR (HR 0.85), applied only to the prediabetes→T2D transition.** This is the correct transportability default because: it is the pooled *randomized* estimate; the pre-registered subgroup/interaction tests found **no** significant effect modification; and it matches Briody 2026, aiding comparability. Let location-specificity enter through **GBD's location-specific prediabetes prevalence and progression incidence** (drives absolute benefit and cost-effectiveness) rather than through a contested RRR modifier. Also add the beneficial **regression-to-normoglycemia** channel (RR 1.30) if the model structure allows — omitting it understates benefit.

**Uncertainty (PSA):** represent the effect as **HR 0.85, 95% CI 0.75–0.96** (lognormal: log-HR mean −0.163, SE ≈ 0.063 from the CI). This CI is the primary uncertainty; do not additionally widen it arbitrarily.

**Adherence caution:** the trial HRs are **ITT** and already embed real-world in-trial adherence. Multiplying an ITT HR by an extra 85% adherence factor risks *double-counting* the adherence penalty. Prefer either (i) the ITT HR as-is for a "programmatic effectiveness" estimate, or (ii) explicitly convert to a per-protocol/CACE-style efficacy and then re-apply modeled adherence — but not both. State which.

**Location scenarios (sensitivity, not base):**
- *Optimistic / high-deficiency locations:* shift toward the deficient-stratum and low-BMI estimates (HR ~0.58–0.74), motivated by the fixed-dose→higher-25(OH)D-increment mechanism and the ≥40–50 ng/mL achieved-level HRs (0.24–0.38). Use only where country data show high %<30 nmol/L **and** low obesity.
- *Conservative / replete or high-obesity locations:* shift toward HR ~0.90–1.00 (VITAL 0.91, FIND 0.86, D2d BMI-high ≈1.0).
- Drive the modifier off a country's 25(OH)D distribution (Cui 2023 / Cashman 2016 / Roth 2018) and obesity prevalence, since **both** baseline 25(OH)D and BMI gate the cholecalciferol response.

**What I would NOT do:** (i) transport the effect to non-prediabetes states in the base case; (ii) hard-code a deficiency-based RRR modifier as base case (interaction not significant); (iii) rely on GBD for a 25(OH)D exposure surface (vitamin D is not a GBD risk factor).

**Bottom line:** constant 15% RRR (HR 0.85, CI 0.75–0.96) on the prediabetes→T2D arrow as base case; make cost-effectiveness location-specific primarily through baseline incidence/prevalence; offer a two-sided RRR scenario modulated by each location's 25(OH)D **and** BMI distributions to bound the transportability uncertainty; and clarify the ITT-vs-adherence handling to avoid double-counting.

---

### Key sources
- Pittas 2023 IPD meta-analysis — [DOI: 10.7326/M22-3018](https://doi.org/10.7326/M22-3018) (PMID 36745886) · [PDF](https://d2dstudy.org/wp-content/uploads/2024/02/2023-Pittas-AnnInternMed-vitamin-D-and-t2DM-IPD.pdf)
- Chatterjee 2023 D2d ≥40 ng/mL ancillary — [DOI: 10.1016/j.ajcnut.2023.03.021](https://doi.org/10.1016/j.ajcnut.2023.03.021) (PMID 37001590)
- D2d main trial (NEJM 2019) — [DOI: 10.1056/NEJMoa1900906](https://doi.org/10.1056/NEJMoa1900906) (PMID 31173679)
- VITAL-T2D ancillary — [PMC11978796](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978796/)
- FIND trial (Diabetologia 2024) — [DOI: 10.1007/s00125-024-06336-9](https://doi.org/10.1007/s00125-024-06336-9) (PMID 39621103)
- Pittas mini-review (Metabolism 2026) — [DOI: 10.1016/j.metabol.2026.156566](https://doi.org/10.1016/j.metabol.2026.156566) (PMID 41707752)
- Global 25(OH)D prevalence (Cui, Front Nutr 2023) — [DOI: 10.3389/fnut.2023.1070808](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2023.1070808/full)
- Cashman Europe (AJCN 2016) — [DOI: 10.3945/ajcn.115.120873](https://doi.org/10.3945/ajcn.115.120873); Roth LMIC roadmap (2018) — [PMC7309365](https://pmc.ncbi.nlm.nih.gov/articles/PMC7309365/)
- GBD 2021 Risk Factors (88 risk factors; vitamin D not included) — [DOI: 10.1016/S0140-6736(24)00933-4](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)00933-4/fulltext)

*(Extracted IPD full-text working copy saved at `/tmp/claude-0/-home-user-ai-assisted-research/0204593c-8a54-599e-a074-6e5ddef73666/scratchpad/ipd.txt`.)*