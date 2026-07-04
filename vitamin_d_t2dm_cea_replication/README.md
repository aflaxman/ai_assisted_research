# Replicating Briody et al. 2026 (vitamin D for T2D prevention), re-basing it on GBD, and extending it to other locations — a feasibility assessment

*Feasibility memo. The question was: can we replicate this paper, see how its results
change under GBD evidence, and reproduce it for other locations? Short answer: yes,
with one important caveat about the model engine and a real dependency on the authors'
data deposit. This memo lays out what exists, what is missing, and the concrete steps.*

## TL;DR

- **The paper** ([Briody et al. 2026, *Lancet Reg Health – Americas* 61:101543](https://doi.org/10.1016/j.lana.2026.101543))
  runs the **CDC/RTI diabetes microsimulation** on a NHANES 2013–2018 prediabetes cohort and finds
  vitamin D3 (4000 IU/day, 15% relative risk reduction on progression to diabetes) is **cost-saving**:
  lifetime −$3208/person, +0.12 QALYs, NMB $15,483 at $100k/QALY.
- **The engine is more available than the paper admits.** The paper says the code is "not publicly
  distributable," but the established-diabetes **complications engine is public and runnable today**
  ([`RTIInternational/diabetes-simbackend-only`](https://github.com/RTIInternational/diabetes-simbackend-only))
  — I cloned and ran it. It ships every complication/mortality/cost/QALY equation *with standard errors*,
  plus NHANES/ACCORD/Look AHEAD starting populations.
- **The genuinely missing piece is small but decisive.** The **prediabetes-prevention front-end** —
  the `pre/` equation module holding the *onset* equations (`complication_prediabetes`,
  `complication_diabetes`) that the vitamin D effect actually acts on — is **referenced by the code but
  not shipped**. That, plus the authors' vitamin-D config and post-processing scripts, is exactly what
  Briody promises to deposit on Mendeley Data "at publication." As of this writing that deposit is **not
  yet discoverable**. So the complications engine is reproducible now; *their specific vitamin-D result*
  is blocked until the deposit lands or RTI/the authors share the prevention module.
- **Goal 1 (replicate): feasible**, on a spectrum from "wait for the deposit and re-run" (lowest effort,
  highest fidelity) to "rebuild the prevention front-end from the paper's methods on top of the public
  engine" (~6–12 weeks).
- **Goal 2 (re-base on GBD): feasible and the scientifically interesting part.** GBD cleanly supplies the
  *microvascular* complication arm, diabetes incidence/prevalence, and mortality by location. It does
  **not** supply prediabetes as an entity, the *macrovascular* complications as diabetes-specific causes,
  costs, or vitamin-D-as-a-risk-factor. Those gaps are workable but must be handled explicitly.
- **Goal 3 (other locations): feasible for epidemiology, hard for costs.** GBD gives location-specific
  epidemiology out of the box. Location-specific *costs* have no GBD analogue and are the binding
  constraint; use WHO-CHOICE unit-cost build-ups and opportunity-cost WTP thresholds, not PPP-converted
  US dollars.
- **Recommended path:** replicate first on the public/authors' engine to establish a baseline; then
  rebuild the model as **`vivarium` components** (IHME's own microsimulation framework) for the GBD
  re-base and multi-location work, which is what it is designed for.

---

## 1. What the paper does

A discrete-time, annual-step **individual microsimulation** (the CDC/RTI model of
[Hoerger et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11017333/)) follows a simulated cohort of
10,000 US adults with prediabetes, drawn to match NHANES 2013–2018 (analytic N = 4176). Each person-year
the model draws 17 diabetes complications and mortality from parametric hazard equations, accrues
discounted costs and QALYs, and runs to death. Vitamin D reduces the prediabetes→diabetes transition
hazard by **15%** (HR 0.85, from the [Pittas 2023 IPD meta-analysis](https://doi.org/10.7326/M22-3018)),
applied to everyone with prediabetes and BMI ≥25 at 85% adherence, stopping at diabetes onset.

Headline lifetime result: **cost-saving** (−$3208/person, +0.12 QALYs, +0.27 life-years, NMB $15,483 at
$100k/QALY, ICER −$26,134). Robust across subgroups, one-way, scenario, and probabilistic sensitivity
analyses. The exact inputs (Table 2) and results (Tables 3–4) are transcribed in
[`paper_parameters.md`](./paper_parameters.md).

## 2. The critical resource finding: what's public, what's missing

This determines everything downstream, so it comes first. I verified it by cloning and running the code,
not by reading the paper's data statement.

### Public and runnable today — the complications engine
[`github.com/RTIInternational/diabetes-simbackend-only`](https://github.com/RTIInternational/diabetes-simbackend-only)
(PI Tom Hoerger; code by Rainer Hilscher — the same person the paper thanks in its acknowledgments).
A fresh anonymous `git clone` gave me:

- `src/simulation/equations/t2d/` — **all 17 complication equations + 3 mortality equations + 9
  risk-factor progression equations + 6 cost/QALY modules, with coefficients *and their standard errors*
  hard-coded** (the SEs are what the PSA draws from). E.g. the MI Weibull carries
  `lambda_val:[-10.829,0.588], age_entry:[0.033,0.004], twd_hba1c:[0.156,0.029], ...`.
- `data/default-populations/` — ready-to-run cohorts: `t2d_nhanes.csv`, `t2d_accord.csv`,
  `t2d_lookahead.csv`.
- `scenarios/*.json` — 14 scenario configs (costs, disutilities, multipliers, interventions).
- `run-model.py` + `requirements.txt` (numpy/pandas/numba/pyarrow, Python 3.8–3.10).

The equations derive from **US trial data (ACCORD + Look AHEAD)**, not UKPDS — a deliberate design choice
that makes this model the natural US baseline. Coefficients are *also* published in the Hoerger 2023
appendices, so the engine is reproducible even independent of the repo.

**I ran it to confirm this is not just readable but executable.** With the pinned dependencies
(numpy 1.23.5 / pandas 1.5.2 / numba 0.56.4 / pyarrow), `python run-model.py
general_population-no_intervention-single` ran a 10,000-person cohort to end-of-life in ~4 minutes
(exit 0) and produced a complete cost-effectiveness table — total discounted cost $189,109, 8.83 QALYs,
20.07 remaining life-years, plus per-person, per-year cumulative cost/QALY trajectories. So the
complications-and-costs machinery works out of the box on a fresh clone.

**Licensing caveat:** the repo has **no LICENSE file**, so by default it is all-rights-reserved
(copyright RTI/CDC). You may read, run, and learn from it; formal reuse/redistribution or releasing a
derivative needs written permission from RTI. This is the sense in which the paper's "not publicly
distributable" is technically defensible even though the code is publicly visible.

### Referenced but NOT shipped — the prevention front-end (the actual blocker)
The orchestration code (`src/simulation/model.py`) loads equation sets `{pre, t2d, t1d, screen}` by
importing `src.simulation.equations.<set>`. **Only `t2d/` and `models/` exist in the public repo.** The
code explicitly branches on `eq == 'complication_prediabetes'` and `eq == 'complication_diabetes'` ("only
applies to pre-diabetes"), and `custom_values.py` reads an `annual_prob_prediabetes` — but the **`pre/`
module and its onset equations are absent**, and **no public scenario references prediabetes**. That
missing `pre/` module is precisely where the vitamin-D effect lives. This is why the paper describes its
model as "a development of the CDC/RTI **Prediabetes** Microsimulation Model" and offers config files
"on reasonable request" — the prevention layer is the part they built on and did not open-source.

### Promised but not yet available — the Briody-specific bundle
The data statement promises, via **Mendeley Data at publication**: a deidentified NHANES-derived analytic
dataset + dictionary, all input tables, R/Python post-processing scripts, and aggregate outputs (not the
engine). As of this writing the deposit is **not discoverable** (Mendeley/DOI/PubMed searches empty; the
article is not yet in PubMed/PMC). Pre-publication access is offered on request to the corresponding
author (jonathanbriody@rcsi.ie).

### Reproducibility tiers

| Component | Status | Can you run/rebuild it? |
|---|---|---|
| Complications engine (Hoerger 2023) | **Public + all coefficients + populations + scenarios** | **Yes — clone and run today** (I did) |
| Right to *redistribute* it | No license → all-rights-reserved | Read/run yes; reuse needs RTI permission |
| **Prediabetes/vitamin-D front-end** | **Referenced in code but not shipped; not in Mendeley yet** | **Not yet** — the real current blocker |
| Precursor CDC-RTI Markov model | Described (RTI Press MR-0013-0909); coefficients scattered across UKPDS/Framingham/DCCT/WESDR | Partial; harder |

## 3. Goal 1 — Replicate the published US results

**Verdict: feasible.** Three routes, in increasing effort and decreasing dependence on the authors:

**Route A — Wait for / request the deposit, then re-run (lowest effort, highest fidelity).**
Email the corresponding author for pre-publication access, or watch Mendeley for the DOI. With their
prevention config + scripts on top of the (public or author-provided) engine, you re-run and reproduce
Tables 3–4 directly. This is the honest definition of "replication."

**Route B — Rebuild the prevention front-end on the public engine (~6–12 weeks).**
Take the public complications engine as-is and add the missing prevention layer from the paper's methods:
1. **Reconstruct the NHANES 2013–2018 prediabetes cohort.** ✅ **Done — see
   [`nhanes_cohort/`](./nhanes_cohort/).** A working pipeline downloads the public microdata, applies the
   paper's prediabetes definition, weights every estimate, and reproduces most of Table 1 (BMI 30.3,
   fasting glucose 5.9, race/ethnicity and albuminuria near-exact). It also uncovered the one real
   definitional fork — how the three prediabetes criteria are combined across cycles — whose two
   defensible readings *bracket* the paper's N (3563 ↔ 5603 vs 4176); the exact recipe (and the
   family-history/eGFR choices) awaits the authors' deposit. Weighting subtlety confirmed in practice:
   this is **three standalone two-year cycles**, so use **`WTSAF2YR/3`** (fasting) or **`WTMEC2YR/3`**
   (MEC) — *not* `WTMECPRP`, which is for the 2017–2020 combined file (a correction to our house NHANES
   rule, which applies to that combined file only). OGTT existed only in 2013–16 (no `OGTT_J`).
2. **Implement the onset layer.** ✅ **Done — see [`simulation/`](./simulation/).** A transparent
   prediabetes→diabetes→death microsimulation, calibrated to the paper's control-arm Table 3, reproduces
   the headline as an *emergent* output. Two variants bracket the paper: **v1** (susceptible fraction)
   matches the cost/QALY increments (lifetime −$3,318 vs −$3,208; +0.149 vs +0.120 QALY) but
   under-predicts the incidence reduction and survival; **v2** (risk-factor-dependent onset + a
   normoglycemia/reversion state) matches the incidence reduction (−7.9% vs −8.0%) and improves
   life-years, but overshoots the magnitudes ~2.4× because strong onset heterogeneity concentrates the
   benefit on young, long-duration converters. **The ICER (≈−$25k vs −$26k) and the cost-saving/dominant
   conclusion are robust across both** — the public data under-identify the onset structure; the exact
   split needs the RTI onset equations. Diagnosis and the fix (age-dependent onset) are in the sub-README.
3. **Wire in the published inputs** from [`paper_parameters.md`](./paper_parameters.md): $60/yr
   supplement, 3% discount, $100k WTP, Yang/Wang complication costs, Neuwahl utilities.
4. **Run the Monte Carlo harness** (100 × 10,000) and compute ICER + NMB with 2.5/97.5 empirical UIs.
5. **Validate** against Briody's numbers and against Mount Hood Diabetes Challenge reference scenarios.

**Route C — Full clean-room rebuild from published equations (~4–6 months).** Only if you cannot use the
repo at all (licensing). Higher risk on the underspecified synthetic-population correlation structure and
risk-factor trajectories.

**Recommendation:** pursue A and B in parallel — request the deposit while standing up the reconstruction,
so you are not blocked and you get an independent cross-check of their numbers.

**Two methodological flags worth carrying into the replication (and fixing if you extend):**
- **ITT × adherence may double-count.** The 15% RRR is an **intention-to-treat** estimate that already
  embeds trial adherence. Multiplying it by a further 85% adherence factor risks penalizing adherence
  twice. Decide: use the ITT HR as a programmatic-effectiveness estimate, *or* convert to per-protocol and
  then apply adherence — not both.
- **The regression-to-normoglycemia channel is omitted.** Vitamin D also raised reversion to normal
  glucose (rate ratio 1.30). If the model has a prediabetes→normoglycemia arrow, modifying only the
  progression arrow understates benefit.

## 4. Goal 2 — Re-base the evidence on GBD

**Verdict: feasible, and this is where IHME has the comparative advantage.** The move is to replace the
model's *epidemiologic* layers (incidence, complication rates, mortality) with GBD estimates while keeping
the economic layers, then see how the cost-effectiveness shifts. What matters is that GBD changes the
**absolute** benefit (baseline risk × constant 15% RRR = cases averted), which is what drives the ICER.

**What GBD supplies cleanly:**
- **Diabetes incidence & prevalence** by age/sex/location — `get_outputs`/`get_draws` on *Diabetes
  mellitus type 2* (measures 5/6); pull at draw level so GBD uncertainty flows into the PSA.
- **The microvascular complication arm** — GBD models *Diabetic neuropathy*, *Diabetic foot due to
  neuropathy*, amputation (with/without treatment), and vision loss/*Blindness due to diabetes mellitus*
  as diabetes sequelae; nephropathy lives under **CKD due to diabetes mellitus type 2** (pull it from the
  CKD cause tree, not the diabetes cause — easy to miss).
- **Mortality** — cause-specific diabetes and CKD-due-to-diabetes deaths (CoDCorrect), diabetes-attributable
  CVD deaths via the *High fasting plasma glucose* PAFs, and all-cause background from GBD life tables;
  or excess-mortality-rate (EMR) directly from the DisMod-MR epi outputs as a drop-in death equation.

**What GBD does NOT give — handle explicitly:**
- **No "prediabetes" entity.** GBD models high FPG only as a continuous risk exposure (TMREL ~4.8–5.4
  mmol/L), never thresholded into IFG/IGT. Prediabetes prevalence/entry must come from **NCD-RisC / IDF /
  [Rooney 2023](https://doi.org/10.2337/dc22-2376)** (mind the ADA-vs-WHO definition mismatch), or by
  integrating GBD's own FPG exposure distribution between your chosen thresholds (an IHME-internal option).
- **Macrovascular complications are not diabetes-specific causes.** MI/stroke/angina exist as CVD burden;
  the diabetes link is only through the high-FPG PAF. **CHF, coronary revascularization, and hypoglycemia
  have no usable GBD entity** — keep the model's own equations for those, optionally calibrated.
- **Disability weights ≠ QALY utilities.** GBD DWs (0=health, 1=death, condition-in-isolation,
  multiplicative) and the Neuwahl utilities the paper uses (1=health, TTO/HUI3-based, whole-person,
  additive, can be negative) are different currencies. **Pick one:** keep Neuwahl utilities for a
  QALY CEA comparable to Briody (recommended), *or* go fully to DALYs with GBD DWs. Do not blend.

**Steps:** (1) pull GBD draws for the entities above by location/age/sex; (2) map each model complication
to its GBD source (or flag "keep model equation"); (3) swap the baseline hazards/rates while retaining the
economic layer; (4) re-run and compare the ICER/NMB against the US base case; (5) attribute the change to
each swapped layer (incidence vs. complications vs. mortality) so the "how do results change under GBD"
question gets a decomposed answer, not just a new number.

## 5. Goal 3 — Reproduce for other locations

**Verdict: epidemiology is easy, costs are the binding constraint.** GBD gives you location-specific
incidence/prevalence/complications/mortality for ~200 countries and many subnationals — the same pulls as
Goal 2, re-parameterized by `location_id`. The hard parts are costs, WTP, and the input cohort.

**Costs (the real work):**
- GBD produces **no costs**. IHME's **DEX** is US-only. **Financing Global Health** is by payer, not by
  disease-complication.
- **Best practice by setting:**
  - *Another high-income country:* re-price each complication state from that country's DRG/tariff
    schedule (NHS Reference Costs, G-DRG, etc.); or transfer the US cost vector via **PPP** with the
    [CCEMG–EPPI cost converter](https://eppi.ioe.ac.uk/costconversion/) (defensible between HICs).
  - *An LMIC:* **do not PPP-convert US dollar totals** — it overstates costs and makes a $60 intervention
    look cost-saving for the wrong reason. Build complication costs **bottom-up**: WHO-CHOICE local
    unit costs (bed-day, outpatient visit) × a per-complication utilization vector; cross-check against
    IDF Atlas per-patient totals and country cost-of-illness studies.
- Follow the **ISPOR transferability** checklist (Drummond 2009); state everything in one named
  currency-year; localize the discount rate.

**WTP threshold:** don't reuse $100k/QALY abroad, and avoid the WHO 1–3× GDP rule. Use empirical
opportunity-cost thresholds — **[Pichon-Rivière 2023](https://www.thelancet.com/article/S2214-109X(23)00162-6/fulltext)**
(174 countries, free [IECS tool](https://iecs.org.ar/en/thresholds/)) for the headline, with
[Woods 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5193154/) and
[Ochalek 2018](https://pubmed.ncbi.nlm.nih.gov/30483412/) as sensitivity bounds. Note the honest
implication: the US opportunity-cost threshold (~$25–40k/QALY) is far below Briody's $100k, so for a fair
cross-country comparison you should also re-run the **US** base case at an opportunity-cost threshold.

**Input cohort abroad:** no other survey matches NHANES's full biomarker panel. **KNHANES (Korea), CHMS
(Canada), ENSANUT (Mexico)** transfer with minor loss; most settings force a **HbA1c-only** prediabetes
definition (fasting glucose/OGTT rarely available) and drop ACR/eGFR/25(OH)D — document the definition
change, because it moves both N and the baseline risk.

**Vitamin-D effect abroad:** keep the **constant 15% RRR (HR 0.85, 95% CI 0.75–0.96)** as base case —
the pre-registered trial subgroup tests found no significant effect modification, and vitamin D is **not**
a GBD risk factor so there is no exposure surface to pull. Location-specificity should enter through
baseline incidence, not a contested RRR modifier. Offer a **two-sided scenario** modulating the HR by each
country's 25(OH)D distribution ([Cui 2023](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2023.1070808/full),
Cashman 2016) *and* obesity prevalence (both gate the cholecalciferol response) — optimistic HR ~0.58–0.74
in deficient/lean populations, conservative ~0.90–1.00 in replete/obese ones.

## 6. Recommended architecture and phased plan

Borrow to replicate; rebuild in vivarium to extend.

1. **Phase 0 — secure inputs (now):** request the Briody deposit / prevention config from the authors;
   keep the public engine clone as the reference.
2. **Phase 1 — replicate US (weeks):** stand up the NHANES cohort reconstruction and the prevention
   front-end on the public engine; hit Briody's Tables 3–4 and Mount Hood reference scenarios. Deliverable:
   "we reproduced the US result (and here's where ITT×adherence and the normoglycemia channel matter)."
3. **Phase 2 — GBD re-base (weeks–months):** re-implement the equations as
   **[vivarium](https://github.com/ihmeuw/vivarium) / `vivarium.public_health`** components (IHME's own
   discrete-time microsimulation framework; BSD-3, ingests GBD exposure/RR natively, multi-location by
   design). Swap epidemiologic layers to GBD; decompose the change in ICER/NMB.
4. **Phase 3 — other locations (months):** re-parameterize by `location_id`; build costs per §5; report
   against opportunity-cost thresholds. Keep a lightweight **Markov cohort** version (structure borrowable
   from the open [PROSIT](https://pubmed.ncbi.nlm.nih.gov/27350481/) model) as a fast cross-check.

**Why vivarium over the RTI engine for Phases 2–3:** the RTI engine is US-trial-equation-bound and
licensing-encumbered; vivarium is built for exactly this (GBD-native inputs, correlated risk-factor
exposures, intervention components, geographic reuse) and is unambiguously ours to extend. **LASER is the
wrong tool** — it's for spatial infectious-disease ABMs; there's no transmission here.

## 7. Bottom line

All three goals are feasible. Replication is gated only by the authors' not-yet-live deposit / the
unshipped prevention module, both obtainable, and otherwise rebuildable on the public engine in weeks.
The GBD re-base is the scientifically valuable step and squarely in IHME's wheelhouse, with clear,
enumerable gaps (no prediabetes entity, macrovascular complications non-specific, DW≠utility) that are
handled rather than blocking. Extension to other locations is straightforward for epidemiology and
genuinely hard only for costs, where the discipline is bottom-up WHO-CHOICE build-ups and opportunity-cost
thresholds rather than PPP-converted US dollars. The durable home for goals 2–3 is a vivarium
reimplementation.

## 8. Appendix: source briefings

Detailed, fully-cited research briefings are archived in [`research_briefings/`](./research_briefings/):
[01 engine availability](./research_briefings/01_model_engine_availability.md) ·
[02 GBD evidence mapping](./research_briefings/02_gbd_evidence_mapping.md) ·
[03 cross-location costs](./research_briefings/03_cross_location_costs.md) ·
[04 NHANES cohort reconstruction](./research_briefings/04_nhanes_cohort_reconstruction.md) ·
[05 build-vs-borrow frameworks](./research_briefings/05_build_vs_borrow_frameworks.md) ·
[06 vitamin-D effect transportability](./research_briefings/06_vitamin_d_effect_transportability.md).
Key primary sources:

- CDC/RTI engine: [repo](https://github.com/RTIInternational/diabetes-simbackend-only) ·
  [Hoerger 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC11017333/) ·
  precursor validation [RTI Press MR-0013-0909](https://www.rti.org/rti-press-publication/validation-cdc-rti-diabetes-cost-effectiveness-model)
- Vitamin-D effect: [Pittas 2023 IPD meta-analysis](https://doi.org/10.7326/M22-3018) ·
  [Chatterjee 2023 (≥40 ng/mL)](https://doi.org/10.1016/j.ajcnut.2023.03.021) ·
  [D2d NEJM 2019](https://doi.org/10.1056/NEJMoa1900906)
- GBD access: [GBD Results Tool](https://vizhub.healthdata.org/gbd-results/) ·
  [GHDx 2021](https://ghdx.healthdata.org/gbd-2021) · prediabetes prevalence [Rooney 2023](https://doi.org/10.2337/dc22-2376)
- Utilities/DWs: [Neuwahl 2021](https://doi.org/10.2337/dc20-1207)
- Costs & thresholds: [WHO-CHOICE (Stenberg 2018)](https://resource-allocation.biomedcentral.com/articles/10.1186/s12962-018-0095-x) ·
  [IDF Atlas expenditure](https://diabetesatlas.org/data-by-indicator/diabetes-related-health-expenditure/diabetes-related-health-expenditure-per-person-usd/) ·
  [Pichon-Rivière 2023 + IECS tool](https://iecs.org.ar/en/thresholds/) ·
  [ISPOR transferability (Drummond 2009)](https://www.valueinhealthjournal.com/article/S1098-3015(10)60782-6/fulltext)
- Frameworks: [vivarium](https://github.com/ihmeuw/vivarium) · [PROSIT (open Markov)](https://pubmed.ncbi.nlm.nih.gov/27350481/) ·
  [Mount Hood Diabetes Challenge](https://www.mthooddiabeteschallenge.com/)

*Caveat: the Briody 2026 full text could not be retrieved independently (publisher returned 403) and the
article is not yet indexed in PubMed/PMC; parameter values here come from the supplied PDF. GBD numeric IDs
should be confirmed with `get_ids`/`get_cause_metadata` against your release before coding.*
