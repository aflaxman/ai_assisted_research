I have enough authoritative material. Compiling the briefing now.

---

# Re-basing the CDC/RTI Prediabetes Microsimulation on GBD Evidence — Scoping Briefing

**Scope:** GBD 2021 (latest fully published round; GBD 2023 data now released but 2021 remains the citable public round). Framing assumes you (IHME) have `db_queries`/central-comp access in addition to the public tools.

**Bottom line up front:**
- GBD directly models the **diabetes lower-extremity and vision sequelae** (neuropathy, foot ulcer, amputation, retinopathy/blindness) and **diabetic nephropathy** — but the latter lives under the **Chronic kidney disease** cause, not under diabetes.
- GBD does **not** model the **cardiovascular complications as diabetes causes**. MI/stroke/angina exist as CVD causes/sequelae; the diabetes link is only through the **risk factor "High fasting plasma glucose" (PAFs)**. **Congestive heart failure, coronary revascularization, and hypoglycemia are not separately obtainable** from GBD.
- GBD has **no "prediabetes" entity** — only the continuous high-FPG exposure. Prediabetes prevalence/entry must come from NCD-RisC / IDF / Rooney 2023 (or by integrating GBD's own FPG exposure distribution between thresholds).
- **Disability weights ≠ utilities.** Keep the study on one currency (QALYs with Neuwahl utilities *or* DALYs with GBD DWs); do not blend.

---

## 1. Complication-by-complication mapping to GBD entities

Measure/definition notes: GBD defines diabetes as **FPG ≥ 7 mmol/L (126 mg/dL), or on insulin/glucose-lowering medication** (HbA1c/OGTT/post-prandial also accepted) ([GBD 2021 diabetes, Lancet 2023, PMC10364581](https://pmc.ncbi.nlm.nih.gov/articles/PMC10364581/)).

| # | Model complication | Modeled in GBD? | GBD cause / sequela entity | Combined disability weight | Notes / access route |
|---|---|---|---|---|---|
| a | **T2DM incidence given prediabetes** | Partial | Cause **Diabetes mellitus type 2** (incidence of the disease) | uncomplicated diabetes DW **0.049** | GBD gives *population* T2DM incidence, **not** conditional-on-prediabetes transition. You must apply GBD incidence to your prediabetic cohort (or keep the model's hazard and use GBD only for calibration). |
| 1 | Diabetic neuropathy | **Yes** | Sequela **"Diabetic neuropathy"** | **0.133** (0.089–0.187) | Peripheral neuropathy without ulcer/amputation. |
| 2 | Diabetic foot ulcer | **Yes** | Sequela **"Diabetic foot due to neuropathy"** | **≈0.150** (neuropathy 0.133 ⊕ foot-ulcer state 0.02) | Combined multiplicatively: 1−(1−0.133)(1−0.02). |
| 3 | Amputation | **Yes** | Sequelae **"Diabetic neuropathy and amputation with treatment"** and **"…without treatment"** | with prosthesis **≈0.167** (⊕0.039); without **≈0.283** (⊕0.173) | Major leg amputation. Split by prosthesis. |
| 4 | Diabetic retinopathy / blindness | **Yes** (as vision-loss endpoints) | Sequelae **"Moderate vision loss due to diabetes mellitus"**, **"Severe vision loss due to diabetes mellitus"**, **"Blindness due to diabetes mellitus"** | moderate ≈0.031; severe ≈0.184; **blindness 0.187** | GBD models *vision-acuity outcomes*, not clinical retinopathy grades (background/proliferative/macular edema). "Laser photocoagulation" is a treatment, not a GBD state. |
| 5 | CKD / dialysis (diabetic nephropathy) | **Yes — but under CKD, not diabetes** | Causes **"Chronic kidney disease due to diabetes mellitus type 2"** and **"…type 1"** (sub-causes of **Chronic kidney disease**), with stage sequelae (albuminuria, stage III/IV/V, **ESRD on dialysis**, **ESRD with kidney transplant**) + associated anemia | dialysis **≈0.571**; transplant **≈0.024**; earlier stages lower | Key gotcha: pull nephropathy from the **CKD cause tree by etiology**, not from the diabetes cause. |
| 6 | Myocardial infarction | **Not diabetes-specific** | Sequela **"Acute myocardial infarction"** under cause **Ischemic heart disease** | — | Diabetes-attributable MI = IHD burden × **PAF(high FPG)**. |
| 7 | Stroke | **Not diabetes-specific** | Cause **Stroke** (subtypes: ischemic / intracerebral hemorrhage / subarachnoid hemorrhage) | — | Diabetes link via high-FPG PAF only. |
| 8 | Angina | **Not diabetes-specific** | IHD sequelae **"…angina"** (asymptomatic→severe health states) | — | Via IHD × high-FPG PAF. |
| 9 | Congestive heart failure | **No usable entity** | Heart failure is a GBD **impairment**, attributed to underlying causes (IHD, hypertensive heart disease, cardiomyopathies…), **not** a standalone or diabetes-specific cause; **diabetic cardiomyopathy is not separately modeled** | — | **FLAG.** No clean diabetes-attributable HF from GBD. |
| 10 | Coronary revascularization | **No** | procedure — not a GBD disease/sequela | — | **FLAG — keep model's own rates.** |
| 11 | Hypoglycemia | **No** | not a GBD cause or sequela | — | **FLAG — keep model's own rates.** (Also non-significant in the study's own utility data.) |
| c | Cause-specific & all-cause mortality | **Yes** | see §5 | — | Diabetes deaths (cause **Diabetes mellitus** T1/T2), CKD-due-to-diabetes deaths, + attributable IHD/stroke deaths; all-cause via GBD life tables. |
| d | HRQoL decrements | **Yes (as DWs)** | disability weights above | — | See §4 — do not mix with the study's utilities. |

Confirmed sequela naming and DWs from the GBD lower-extremity-complications analysis ([Pacella et al., *Diabetes Care* 2020, PMC/Greenwich PDF](https://gala.gre.ac.uk/id/eprint/26858/1/26858%20PACELLA_Global_Disability_Burden_of%20Diabetes-related_Lower_Extremity_Complications_2020.pdf); journal version [diabetesjournals.org 43/5/964](https://diabetesjournals.org/care/article/43/5/964/35731/)), which uses "uncomplicated diabetes DW = 0.049" and "blindness due to diabetes DW = 0.187". CKD-due-to-diabetes as a distinct CKD sub-cause confirmed in [GBD 2021 CKD, PMC11919670](https://pmc.ncbi.nlm.nih.gov/articles/PMC11919670/) and [PMC11872908](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11872908/). Heart-failure-as-impairment confirmed in [GBD 2021 heart failure, PMC11755835](https://pmc.ncbi.nlm.nih.gov/articles/PMC11755835/).

**Practical implication for the microsim:** GBD cleanly supplies the *microvascular* arm (neuropathy → foot → amputation; retinopathy → blindness; nephropathy → dialysis). The *macrovascular* arm (MI/stroke/angina/CHF/revascularization) and hypoglycemia must stay on the model's own equations, optionally *calibrated* to GBD IHD/stroke levels and to diabetes-attributable fractions via high-FPG PAFs.

---

## 2. Diabetes incidence & prevalence by age/sex/location — access paths

**Entities (verify IDs with `get_ids`/`get_cause_metadata`; these are stable across recent rounds):**
- Diabetes mellitus (parent), **Diabetes mellitus type 1**, **Diabetes mellitus type 2** — cause tree.
- Chronic kidney disease → CKD due to DM type 1 / type 2 — separate cause tree.
- Risk: **High fasting plasma glucose** (metabolic `rei`).

**Public:**
- **GBD Results Tool** — https://vizhub.healthdata.org/gbd-results/ (select measure = Prevalence/Incidence, cause = Diabetes mellitus type 2, by age/sex/location/year; downloads mean + 95% UI).
- **GHDx** — https://ghdx.healthdata.org/gbd-2021 (codebooks, definitions, disability-weights file).
- Data-tools guide: [IHME GBD 2021 Data & Tools Guide (PDF)](https://www.healthdata.org/sites/default/files/2024-05/IHME_GBD_2021_DATA_TOOLS_GUIDE_Y2024M05D28_0.PDF).

**Internal `db_queries` (central comp) — the route that gives a microsim what it needs:**
- `get_outputs("cause", cause_id=<T2DM>, measure_id=[5,6], metric_id=[1,3], age_group_id=..., sex_id=..., location_id=..., year_id=..., release_id=<GBD2021>)` → final **prevalence (measure 5)** and **incidence (measure 6)**, number (metric 1) / rate (metric 3), mean + UI. **No draws.**
- `get_draws(gbd_id_type="cause_id", gbd_id=<T2DM>, source="como", measure_id=[5,6], ...)` → **1000 draws** of prevalence/incidence for uncertainty propagation into the microsim.
- `get_model_results("epi", gbd_id=<modelable_entity_id>, ...)` **or** `get_draws(..., source="epi")` → the **DisMod-MR 2.1** modelable-entity outputs: **incidence, prevalence, remission, EMR (excess mortality), CSMR**. This is the richest source for a state-transition microsim because it gives incidence *and* transition/mortality parameters on the same age/sex/location grid.
- Helpers: `get_population`, `get_age_metadata`, `get_cause_metadata`, `get_rei_metadata`, `get_ids`, `get_demographics`.

Always pull at **draw level** (1000 draws) for the sequelae you propagate, so the microsim's PSA inherits GBD uncertainty rather than only the summarized UI.

---

## 3. Prediabetes in GBD (there isn't a disease entity)

GBD has **no "prediabetes," "IFG," or "IGT" cause or sequela.** It models the metabolic risk factor **"High fasting plasma glucose"** as a **continuous exposure** (modeled mean FPG + SD, ensemble distribution) with:
- **TMREL ≈ 4.8–5.4 mmol/L** (theoretical-minimum-risk exposure level), and
- **RR functions** (dose-response from cohort meta-analyses) linking FPG to diabetes, IHD, stroke, CKD, TB, etc. — the machinery behind the PAFs.

Confirmed: TMREL 4.8–5.4 mmol/L and the attributable-burden framing in [GBD 2021 high-FPG analysis, *Nutrition & Diabetes* 2025](https://www.nature.com/articles/s41387-025-00405-7) and [Frontiers Endocrinol 2025 (ischemic stroke, HFPG)](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1490428/full).

**Consequences for the model:**
- The continuous FPG exposure **spans the prediabetic range but is never thresholded** into IFG/IGT categories, and the *disease* endpoint is diabetes (FPG ≥ 7 or Rx). So **your prediabetes cohort definition and prediabetes→diabetes progression cannot be read off GBD outputs directly.**
- **Two options:**
  1. **External prevalence** — NCD-RisC (glucose/diabetes), IDF Diabetes Atlas, or **Rooney et al. 2023**. According to PubMed, Rooney et al. estimated 2021 global **IGT 9.1% (464M)** and **IFG 5.8% (298M)** among adults 20–79 ([*Diabetes Care* 2023;46:1388–1394, DOI 10.2337/dc22-2376](https://doi.org/10.2337/dc22-2376)). **Watch the definition mismatch:** Rooney uses WHO IFG (FPG 6.1–6.9) and IGT (2-h 7.8–11.0); the CDC/NDPP world typically uses **ADA prediabetes** (FPG 100–125 mg/dL or HbA1c 5.7–6.4). IGT and IFG identify overlapping-but-different populations, so pick the definition your incidence-to-diabetes hazard was estimated under.
  2. **Derive from GBD internally** — pull the high-FPG **exposure distribution** (`get_draws` for the FPG exposure ME) and **integrate the modeled FPG density between your chosen prediabetes thresholds** by age/sex/location. This keeps prediabetes prevalence on the same platform as everything else and is an option only IHME-internal users have.

---

## 4. Disability weights vs. utilities — and why not to mix them

**Different scales, different construction:**

| | GBD disability weight (DW) | Utility / QALY weight (study uses Neuwahl) |
|---|---|---|
| Anchors | **0 = full health, 1 = death** | **1 = full health, 0 = death** (can be **< 0**, worse than death) |
| Elicitation | Lay descriptions, **paired comparisons + population-health-equivalence** surveys (Salomon 2012/2015); condition described **in isolation** | Preference-based (**TTO/SG**, or off **HUI3/EQ-5D**); reflects the **whole person**, comorbidities included |
| Comorbidity | Combined **multiplicatively** across sequelae, assuming independence | Empirically captured in the instrument; Neuwahl fits **additive fixed-effects decrements** |
| Purpose | YLDs → **DALYs** | QALYs |

**Rough relation:** `utility ≈ 1 − DW` **only as a crude approximation.** They diverge because (a) elicitation differs, (b) DWs describe an *isolated sequela* while utilities reflect the *whole patient*, (c) DW health states don't map 1:1 onto the model's clinical states, and (d) utilities can be negative.

**The study's decrements (keep these for a QALY CEA).** According to PubMed, Neuwahl et al. (HUI3 in ACCORD + Look AHEAD, n≈15,252) report utility decrements: **stroke** event −0.109 / history −0.051; **amputation** event −0.092 / history −0.150; **CHF** event −0.051 / history −0.041; **dialysis** event −0.039; **eGFR <30** event −0.043 / history −0.025; **angina** history −0.028; **MI** event −0.028; smaller for laser photocoagulation and eGFR <60; **non-significant for hypoglycemia, revascularization, dialysis history, angina event, MI history** ([*Diabetes Care* 2021;44(2):381–389, DOI 10.2337/dc20-1207](https://doi.org/10.2337/dc20-1207); [RTI summary](https://www.rti.org/publication/patient-health-utility-equations-type-2-diabetes-model)).

**Caveat / recommendation:** **Do not add a GBD DW-derived decrement on top of a Neuwahl utility decrement** — you would double-count and mix two anchoring systems. **Decide the outcome currency first:**
- **QALY CEA (recommended for consistency with Briody 2026 / CDC-RTI lineage):** keep **Neuwahl utilities**; use GBD only for epidemiology (incidence/prevalence/mortality), *not* HRQoL.
- **DALY analysis:** switch wholesale to **GBD disability weights** applied to GBD (or model) prevalence, consistently.
Also note GBD DWs are for *acuity/severity* health states, not the model's event-vs-history structure, so any DW substitution loses the acute-vs-chronic distinction Neuwahl provides.

---

## 5. Mortality — replacing the model's mortality equation

GBD gives you three layers; a microsim typically needs all three:

1. **Cause-specific mortality (direct diabetes/nephropathy deaths):**
   - Causes: **Diabetes mellitus** (type 1 / type 2) and **Chronic kidney disease due to diabetes mellitus type 1/2**.
   - Access: `get_outputs("cause", measure_id=1, metric_id=[1,3], ...)` for deaths; **draws** via `get_draws(source="codcorrect", measure_id=1, ...)`. Produced by **CODEm → CoDCorrect** (rescaled to the all-cause envelope).

2. **Diabetes-attributable CVD deaths (the macrovascular arm):**
   - Not a cause — obtain as **attributable burden of high FPG** on IHD/stroke: `get_outputs`/`get_draws(source="burdenator")` with the **high-FPG `rei`**, or pull **PAFs** and multiply IHD/stroke deaths by PAF(high FPG). See [GBD high-FPG attributable analysis](https://www.nature.com/articles/s41387-025-00405-7).

3. **All-cause / background mortality:**
   - GBD demographics: **life tables and the mortality envelope** (`get_life_table`, `get_envelope`, `get_population`).
   - To avoid double counting, use **background all-cause mortality minus the causes you model explicitly**, then add back your modeled cause-specific hazards.
   - For excess/relative mortality of prevalent diabetes, the **DisMod-MR EMR / CSMR** outputs (via `get_model_results("epi", ...)` / `get_draws(source="epi")`) give **excess mortality rate by age/sex/location** — a direct drop-in for a state-transition death equation.

**How this replaces the model's equation:** substitute the model's mortality function with GBD **cause-specific mortality rates (deaths ÷ population)** for diabetes and each *modeled* complication cause, layered on GBD **background all-cause mortality** (net of those causes), with diabetes-attributable IHD/stroke mortality applied via **high-FPG PAFs**. Draw-level pulls let mortality uncertainty flow into the CEA's PSA.

---

## Key entity/ID cheat-sheet (confirm with `get_ids` / `get_cause_metadata` / `get_rei_metadata`)

- Causes: `Diabetes mellitus`, `Diabetes mellitus type 1`, `Diabetes mellitus type 2`; `Chronic kidney disease` → `…due to diabetes mellitus type 1`, `…due to diabetes mellitus type 2`; `Ischemic heart disease`; `Stroke` (ischemic / ICH / SAH).
- Diabetes sequelae: `Uncomplicated diabetes` (DW 0.049); `Diabetic neuropathy` (0.133); `Diabetic foot due to neuropathy` (≈0.150); `Diabetic neuropathy and amputation with/without treatment` (≈0.167 / ≈0.283); `Moderate/Severe vision loss due to diabetes mellitus`; `Blindness due to diabetes mellitus` (0.187).
- Risk: `High fasting plasma glucose` (TMREL 4.8–5.4 mmol/L).
- **Not available from GBD:** congestive heart failure as a diabetes-attributable cause; coronary revascularization; hypoglycemia; clinical retinopathy grades; prediabetes/IFG/IGT.

## Sources
- GBD 2021 diabetes (definitions, four sequelae): [PMC10364581](https://pmc.ncbi.nlm.nih.gov/articles/PMC10364581/) · [ScienceDirect/Lancet](https://www.sciencedirect.com/science/article/pii/S0140673623013016)
- GBD sequela names + disability weights (0.049, 0.133, 0.187, foot/amputation composites): [Pacella et al. 2020, PDF](https://gala.gre.ac.uk/id/eprint/26858/1/26858%20PACELLA_Global_Disability_Burden_of%20Diabetes-related_Lower_Extremity_Complications_2020.pdf)
- CKD due to diabetes as CKD sub-cause: [GBD 2021 CKD, PMC11919670](https://pmc.ncbi.nlm.nih.gov/articles/PMC11919670/) · [PMC11872908](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11872908/)
- Heart failure as impairment: [GBD 2021 HF, PMC11755835](https://pmc.ncbi.nlm.nih.gov/articles/PMC11755835/)
- High-FPG risk factor / TMREL / attributable burden: [Nature *Nutr & Diabetes* 2025](https://www.nature.com/articles/s41387-025-00405-7) · [Frontiers Endocrinol 2025](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2025.1490428/full)
- Access tools: [GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) · [GHDx GBD 2021](https://ghdx.healthdata.org/gbd-2021) · [GBD 2021 Data & Tools Guide PDF](https://www.healthdata.org/sites/default/files/2024-05/IHME_GBD_2021_DATA_TOOLS_GUIDE_Y2024M05D28_0.PDF)
- Prediabetes (per PubMed): Rooney et al., [*Diabetes Care* 2023, DOI 10.2337/dc22-2376](https://doi.org/10.2337/dc22-2376)
- Utility decrements (per PubMed): Neuwahl et al., [*Diabetes Care* 2021, DOI 10.2337/dc20-1207](https://doi.org/10.2337/dc20-1207) · [RTI](https://www.rti.org/publication/patient-health-utility-equations-type-2-diabetes-model)

*Attribution: Neuwahl et al. and Rooney et al. bibliographic details and abstracts retrieved via PubMed. DW composites computed as 1−∏(1−DWᵢ). GBD numeric IDs should be confirmed against your release with `get_ids`/`get_cause_metadata` before coding.*