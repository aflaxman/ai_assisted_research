# Simulating a national MMN-fortified bouillon program with GBD data

A plan for translating the CoMIT bouillon-fortification trial into a
population-level simulation of health impact, using Global Burden of Disease
(GBD) inputs.

## The paper

Haskell M, Kumordzie S, Becher E, et al. **Effect of Multiple
Micronutrient-fortified Bouillon Cubes on Milk Vitamin B12 and Vitamin A
Concentrations Among Lactating Women in Northern Ghana: a Randomized,
Controlled Trial.** VeriXiv 2026, 3:264 (version 1).
https://verixiv.org/articles/3-264 — DOI: 10.12688/verixiv.3565.1
(Condiment Micronutrient Innovation Trial, CoMIT; NCT05178407.)

What the trial did:

- **Setting**: Kumbungu and Tolon districts, northern Ghana, Jan 2023–Sep 2024.
- **Design**: household-randomized, doubly-masked, 12-week trial; n=645
  lactating women 15–49 y with infants 4–18 mo.
- **Intervention**: bouillon cubes fortified with vitamin B12 (~2.8 µg/d),
  vitamin A (~500 µg RE/d intended), folic acid, iron, zinc, and iodine,
  versus control cubes with iodine only. Assumed intake ~2–2.5 g/d for women
  of reproductive age (WRA).
- **Key results**: milk vitamin B12 rose in the intervention group (median 215
  vs 189 pmol/L, P<0.001), though it stayed below the reference median. Milk
  retinol did not change (P=0.48) — but a premix manufacturing error meant
  only ~50–73% of the intended vitamin A dose (~249–293 µg RE/d) was
  delivered, and effect estimates pointed in a positive direction.
- **Reach and adherence**: household adherence exceeded 95% throughout; study
  cubes were ~89% of household bouillon use. In this region, 99% of households
  have cooked with bouillon and 77% cook with it at least twice daily — this
  reach is the entire case for bouillon as a fortification vehicle.

The trial establishes biological efficacy for one intermediate biomarker (milk
B12) over 12 weeks. It does not — and cannot — tell us what a *national
fortification program* would do to deficiency prevalence, anemia, child
mortality, or DALYs over a decade. That is a modeling question, and GBD
supplies most of the machinery.

## The program to simulate

National-scale mandatory (or dominant-market-share voluntary) fortification of
bouillon cubes in Ghana with the CoMIT micronutrient premix: vitamin A, vitamin
B12, folic acid, iron, zinc, iodine. The population reached is everyone who
eats bouillon-seasoned food — children, WRA, pregnant and lactating women —
not just the trial's lactating-women stratum.

## Why GBD

GBD provides, for Ghana (and subnationally-relevant proxies), all of:

1. **Exposure prevalence** for the nutrient risk factors it models: vitamin A
   deficiency, zinc deficiency, and iron deficiency (as the population
   hemoglobin distribution behind the anemia envelope).
2. **Relative risks** for the risk–outcome pairs: vitamin A deficiency →
   diarrhea, measles (children under 5); zinc deficiency → diarrhea, lower
   respiratory infections; low hemoglobin → maternal disorders.
3. **Cause-level burden** to apply averted-fraction estimates against:
   diarrheal diseases, measles, LRI, dietary iron deficiency, maternal
   disorders, neural tube defects, iodine deficiency, other nutritional
   deficiencies.
4. **Demography**: population counts, births, and all-cause mortality by
   age/sex/year for the projection horizon.

The comparative-risk-assessment identity does the work: shift the exposure
distribution, recompute the population attributable fraction (PAF), and the
difference in attributable burden is the program's simulated effect.

## Modeling approach: two tiers

### Tier 1 — PAF-shift comparative risk assessment (build this first)

A transparent, notebook-scale model. For each nutrient–outcome pathway:

```
DALYs averted = DALYs(cause) × [PAF(baseline exposure) − PAF(counterfactual exposure)]
```

where the counterfactual exposure comes from the coverage cascade below. This
tier runs in an afternoon once the GBD pulls are in hand, gives order-of-
magnitude answers, and identifies which pathways dominate (almost certainly
vitamin A in under-5s and iron in WRA).

### Tier 2 — Vivarium microsimulation (if Tier 1 justifies it)

IHME's Vivarium framework has directly relevant precedent: the
`vivarium_conic_lsff` large-scale food fortification models built for the
Gates Foundation simulated iron/folic acid/vitamin A fortification of staples.
A bouillon model is the same architecture with a different vehicle:

- Individual-level simulants with age/sex/location drawn from GBD demography.
- Heterogeneous bouillon intake (the trial's 24-hr recall and adult-male-
  equivalent distributions give the intake model; the NCI-method usual-intake
  distributions in the CoMIT protocol are the right inputs).
- Hemoglobin as a continuous exposure shifted by iron dose; anemia and
  maternal-disorder risk follow GBD's hemoglobin risk curves.
- Vitamin A deficiency as a dichotomous state with GBD prevalence and RRs.
- Folic acid → neural tube defect incidence at birth (as in the LSFF model).
- Time-varying coverage ramp-up as fortified cubes displace unfortified stock.

Tier 2 earns its cost when you need uncertainty propagation, interacting
pathways (e.g., anemia from both iron and B12), or subgroup equity results.

## Mapping trial nutrients to GBD entities

| Premix nutrient | GBD handle | Outcomes affected | Notes |
|---|---|---|---|
| Vitamin A | Risk: vitamin A deficiency | Diarrhea, measles (<5y); VAD sequelae | Trial was null at ~50–73% dose — simulate both delivered and intended dose |
| Iron | Risk: iron deficiency (hemoglobin shift); Cause: dietary iron deficiency | Anemia YLDs, maternal disorders, birth outcomes (extended) | Dose→Hb response from published fortification meta-analyses, not the trial; see iron pathway detail below |
| Zinc | Risk: zinc deficiency | Diarrhea, LRI (<5y) | GBD exposure data are weak; flag wide uncertainty |
| Folic acid | Cause: neural tube defects | NTD births averted | Requires reaching WRA periconceptionally — bouillon's reach is the argument |
| Iodine | Cause: iodine deficiency | Goiter, cognitive sequelae | Control cubes already had iodine; marginal effect ~0 where salt iodization works |
| Vitamin B12 | **No GBD risk factor** | Partial: B12-deficiency anemia inside "other nutritional deficiencies" | The trial's strongest result is the hardest to burden-ify; model B12 status (not DALYs) as a separate endpoint using trial effect sizes |

The B12 row is the honest limitation: GBD has no B12 risk–outcome pairs, so
the trial's headline finding translates into "cases of low milk/serum B12
averted," not DALYs. Report it as its own outcome rather than forcing it into
the DALY frame.

### Iron pathway detail

Note first that this paper reports neither the iron dose nor hemoglobin
outcomes — those are in a forthcoming companion CoMIT paper (a statistical
analysis plan, Arnold et al. 2025, is already published). Until it appears,
the iron pathway runs on the protocol dose plus external dose–response
evidence.

Expected magnitude: a modest hemoglobin shift, but likely the first- or
second-largest DALY contributor of the premix anyway. Three opposing forces:

1. **Small, poorly absorbed dose.** At 2–2.5 g of cube/day the iron dose is a
   few mg — a fraction of the RDA — and condiment fortification typically uses
   ferric pyrophosphate, with roughly a quarter to half the bioavailability of
   ferrous sulfate. Expect population Hb shifts of ~1–3 g/L in regular
   consumers, not the 5–10 g/L of supplementation trials.
2. **Much of northern Ghana's anemia is not iron-responsive.** The national
   survey the paper cites (Wegmüller 2020) found anemia heavily driven by
   malaria, inflammation, and hemoglobinopathies. GBD's anemia causal
   attribution gives the iron-responsive share directly — a key anchor pull,
   because it caps what any iron dose can avert.
3. **Enormous reached population with complete GBD machinery.** Anemia in WRA
   is a large prevalent YLD burden, and in IHME's vivarium LSFF modeling iron
   was consistently a dominant DALY contributor: small per-person effect ×
   very large population.

Model iron as **three stacked components**, reported separately so a reviewer
can accept or discount each:

- **(a) Anemia YLDs** from the hemoglobin shift — standard GBD, low
  controversy.
- **(b) Maternal disorders** via GBD's hemoglobin → maternal
  hemorrhage/sepsis relative risks — standard GBD CRA; small in absolute
  deaths but defensible.
- **(c) Birth outcomes (extended):** maternal hemoglobin → birthweight and
  gestational age → neonatal mortality via GBD's LBWSG risk curves. This is
  the mortality multiplier — (a) and (b) are mostly YLDs, while (c) converts
  maternal iron status into neonatal deaths (YLLs), which in the
  `vivarium_gates_iv_iron` and nutrition-optimization work was a major share
  of iron's total impact. It is **not** a standard GBD risk–outcome pair, so
  it needs literature effect sizes — e.g., Haider et al. 2013 (BMJ) prenatal
  iron meta-analysis (birth weight ≈ +40 g; LBW RR ≈ 0.81 for
  supplementation) — mapped onto GBD's LBWSG exposure and risk curves, cribbing
  the implementation from `vivarium_gates_iv_iron`.

Two caveats for component (c): the meta-analytic evidence comes from
supplementation doses several times a bouillon dose, so scale by dose–response
rather than applying pooled RRs wholesale; and the effect requires exposure
during pregnancy — which is bouillon's selling point, since a market vehicle
reaches women continuously rather than only after the first antenatal visit
(the same argument the paper makes for folic acid and B12).

Leave iron → child cognitive development as a qualitative mention only; the
causal evidence is contested and does not fit the DALY frame cleanly.

## The intervention effect model (coverage cascade)

Effective dose per person = product of:

1. **Vehicle coverage** — fraction of households using bouillon (trial region:
   99% ever, 77% ≥2×/day; use Ghana Living Standards Survey / the CoMIT
   market survey for national numbers).
2. **Fortified market share** — scenario variable: what fraction of cubes sold
   are fortified (mandatory ≈ 90–100%; voluntary single-brand ≈ brand share).
3. **Intake distribution** — g/day by age/sex from the trial's AME
   disappearance method (~2–2.5 g/d for WRA; children proportionally less).
4. **Nutrient content as consumed** — label dose × retention through
   manufacturing, storage, and cooking. The trial's premix error (vitamin A at
   50–73% of target) is not a footnote; it is a *scenario*: real programs have
   exactly this failure mode, so model "intended dose" and "as-delivered dose"
   as separate arms.
5. **Intake → status** — convert added nutrient intake to deficiency
   probability shift: trial effect sizes for B12; published dose–response
   models for vitamin A (e.g., the null-at-half-dose, positive-direction trial
   result brackets the response) and iron (fortification meta-analyses).

## Scenarios

| Scenario | Coverage | Dose | Purpose |
|---|---|---|---|
| S0 baseline | 0% fortified | — | Counterfactual |
| S1 program as intended | 77% regular use × 95% market share | Full premix | Headline estimate |
| S2 program as delivered | same | VA at 50–73% | Manufacturing-quality reality check |
| S3 low coverage | 50% | Full | Voluntary-fortification pessimism |
| S4 high coverage | 95% | Full | Mandatory + enforcement optimism |

Horizon: 10 years (2026–2035), reporting annual and cumulative DALYs, deaths,
and deficiency cases averted, by age group and sex.

## GBD data pulls needed

All for Ghana (location_id 207), GBD 2023 (or latest available round), with
uncertainty draws where possible:

- Population and births, by age/sex/year, 2026–2035 (forecasts or hold-flat).
- Prevalence of vitamin A deficiency and zinc deficiency exposure (REI
  exposure models), children <5 and WRA.
- Hemoglobin distribution / anemia prevalence by severity, WRA and children.
- Relative risks for: VAD→{diarrhea, measles}; zinc→{diarrhea, LRI};
  hemoglobin→maternal disorders.
- DALYs, deaths, YLDs for: diarrheal diseases, measles, LRI, dietary iron
  deficiency, maternal disorders, neural tube defects, iodine deficiency,
  other nutritional deficiencies.
- Current PAFs for the risk–outcome pairs above (to validate the recomputed
  baseline PAFs against GBD's own).

Access paths, in order of preference: the GBD 2023 MCP skills available in
Claude sessions (`disease-profile`, `risk-outcome-profile` for quick anchors),
`get_draws`/central comp tools on the IHME cluster for draw-level inputs, and
the public GBD Results Tool for anything shareable.

## Validation targets

- Simulated baseline deficiency prevalence vs the 2020 CoMIT pilot survey:
  19% of children 2–5y and 12% of WRA vitamin B12 deficient; 13.4% low milk
  retinol (<28 nmol/g fat) among lactating women.
- Ghana DHS / Ghana Micronutrient Survey (2017) anemia and VAD prevalence.
- GBD's own attributable-burden estimates for Ghana as the zero-coverage arm.

## Limitations to state up front

1. Milk biomarkers are intermediate outcomes; the chain from milk B12 to
   infant health passes through pathways GBD does not quantify.
2. GBD's vitamin A deficiency exposure model and the trial's biomarker frame
   differ; reconcile definitions before shifting prevalence.
3. A 12-week trial anchors short-run biomarker response, not steady-state
   status under years of exposure — the paper itself argues cumulative
   pre-pregnancy exposure may matter more.
4. Northern Ghana's bouillon intake may exceed the national average;
   national extrapolation needs national intake data.
5. Zinc deficiency exposure estimates carry wide uncertainty in GBD.

## Work plan

1. **Anchor pulls** (½ day) — get Ghana burden and exposure numbers for the
   table above via GBD MCP skills; record in `data/gbd_anchors.csv`.
2. **Tier 1 notebook** (1–2 days) — `tier1_paf_shift.ipynb`: coverage cascade
   → exposure shift → PAF recompute → DALYs averted, with the S0–S4 scenario
   table as output. Pure pandas/numpy, `uv`-managed environment.
3. **Sensitivity + writeup** (1 day) — tornado plot over coverage, dose,
   dose–response slope; draft results into this README.
4. **Decision point** — if Tier 1 shows material burden aversion with
   uncertainty spanning policy-relevant thresholds, scope the Tier 2 Vivarium
   model (reusing `vivarium_conic_lsff` components).

Related modeling work worth reading first: the MINIMOD project (Vosti, a CoMIT
coauthor, leads it) already does cost-effectiveness modeling of micronutrient
intervention portfolios in Ghana and Cameroon, including bouillon scenarios —
Tier 1 results should be sanity-checked against MINIMOD's published Ghana
estimates, and a cost layer (premix cost per cube, ~program overhead) can be
borrowed from it to report cost per DALY averted.
