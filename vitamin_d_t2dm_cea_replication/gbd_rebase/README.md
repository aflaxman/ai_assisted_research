# GBD re-base + other locations (goals 2 and 3)

This re-bases the vitamin-D / T2D prevention model on **GBD-style, location-specific
epidemiology** — driving diabetes onset (age pattern + level) and mortality from a
location's curve instead of the CDC/RTI-calibrated parametric hazard — and shows how the
cost-effectiveness result moves across locations.

```
uv run python gbd_rebase_model.py     # needs ../nhanes_cohort/outputs/cohort_canonical_n4176.csv
```

## Honest status: real GBD draws are not reachable from this sandbox

I tried the route you suggested (`ihmeuw/vivarium_nih_us_cvd`, `update_vivarium` branch) and hit
exactly the wall you anticipated:

- The per-state artifacts (`src/vivarium_nih_us_cvd/artifacts/*.hdf`, 51 of them) are **Git-LFS
  pointer stubs** (133 bytes each) pointing to GitHub's LFS server; the proxy gates github.com LFS
  access for this session, `git-lfs` isn't installed, and `add_repo` for the repo failed.
- **Building** an artifact (`build_usa_artifact.py` / `make_artifacts`) needs IHME cluster access
  (`vivarium_inputs` → `get_draws`), which isn't available here.
- The **GBD Results Tool** is an interactive SPA (not fetchable), and the GBD 2021 diabetes paper
  reports prevalence/DALYs, not age-specific incidence.

So the numbers in [`gbd_inputs.csv`](./gbd_inputs.csv) are **illustrative, GBD/US-surveillance-anchored
magnitudes**, not real draws. What is real and reusable is the **pipeline and the seam** that consume
age/sex/location epidemiology in the exact shape GBD provides — see "Drop-in seam" below. `vivarium`
and `vivarium_public_health` (v6.3.2) *do* pip-install here, so the framework is available; only the
data artifact is missing.

## What the pipeline does

It reuses the risk-factor + reversion + age structure of the onset model, but replaces the parametric
onset and mortality with a location's curve:

- **Onset** hazard(age) = `PREDIAB_RR · gbd_incidence(age, location) · REL_i`, where
  `gbd_incidence` is the general-population type-2-diabetes incidence for that location, `PREDIAB_RR`
  is how much faster prediabetics progress than the general population, and `REL_i` is the per-person
  HbA1c/FPG/BMI heterogeneity. **The age pattern and level of onset now come from GBD.**
- **Mortality** is the location's all-cause rate, with diabetics carrying an excess-mortality HR.
- **Calibration** touches only the US onset (PREDIAB_RR → 10-yr incidence, heterogeneity → lifetime
  incidence). The diabetes complication cost/disutility-per-year are **held fixed** at the
  paper-consistent values from the age-dependent model — deliberately *not* re-solved to the paper's
  cost/QALY totals, because those totals embed the paper's (inflated) life expectancy; under GBD
  mortality the totals must be free to differ.
- Swapping the location swaps **only** the GBD curve (incidence + mortality); the intervention effect,
  progression RR, heterogeneity, and (for now) US costs are held fixed, so cross-location differences
  come purely from GBD epidemiology.

## Result — how the conclusion moves with location

| | Incidence C→V /100 | Reduction | Rem. LY | Δcost | ΔQALY | ICER $/QALY | NMB @ $100k |
|---|---|---|---|---|---|---|---|
| **USA** (calibrated to paper) | 32.3 → 30.1 | −7.0% | 29.0 | −$6,427 | +0.250 | **−$25,713** | $31,423 |
| **HighBurden** (illustrative) | 35.9 → 33.2 | −7.3% | 26.3 | −$7,699 | +0.304 | **−$25,355** | $38,066 |

Two findings, both robust to the illustrative numbers because they're about the *mechanism*:

1. **The ICER is stable (~−$25.7k) and the intervention stays cost-saving/dominant** under the GBD
   re-base — and the GBD-re-based US ICER (−$25,713) lands right next to the paper's (−$26,134), even
   though this run uses **realistic GBD mortality** (US remaining life-years 29, vs the paper's 36.45).
   The cost-effectiveness *ratio* is structurally stable, echoing the identifiability finding from the
   onset model.
2. **Higher diabetes burden → larger absolute benefit.** The higher-incidence location has more
   diabetes to prevent, so vitamin D averts more cases → bigger cost saving, bigger QALY gain, and a
   **higher net monetary benefit ($38k vs $31k)** — even at a similar ICER. This is the core "how do
   results change by location" answer: the *conclusion* (cost-saving) transports, but the *magnitude of
   value* scales with local disease burden.

**Costs are held at US values here** — a real other-location analysis also needs location-specific
complication costs (IHME DEX is US-only; use WHO-CHOICE unit-cost build-ups) and an opportunity-cost
WTP threshold (Pichon-Rivière/IECS), not $100k. That is the separate, harder half of goal 3, scoped in
the parent [`README.md`](../README.md) §5.

## Drop-in seam — plugging in real GBD draws

`gbd_inputs.csv` is the only thing to replace. Two real sources, both grounded in the repo I explored:

**A. From a vivarium artifact** (`ihmeuw/vivarium_nih_us_cvd` once LFS/data access is available, or any
GBD artifact you build). The keys are in that repo's `constants/data_keys.py`:
```python
import pandas as pd
art = "west_virginia.hdf"                    # a real (LFS-resolved) artifact
pop  = pd.read_hdf(art, "population.structure")                       # age/sex population
acmr = pd.read_hdf(art, "cause.all_causes.cause_specific_mortality_rate")
# diabetes onset: add diabetes to the artifact's causes and read
inc  = pd.read_hdf(art, "cause.diabetes_mellitus_type_2.incidence_rate")
# collapse draws to a mean rate per age band -> gbd_inputs.csv columns
```
**B. From `get_draws` on the IHME cluster** (the source the artifact is built from):
```python
from get_draws.api import get_draws
inc = get_draws("cause_id", <T2D_id>, source="como",       measure_id=6, location_id=<loc>,
                age_group_id=..., sex_id=..., release_id=<GBD2021>)   # incidence, 1000 draws
mrt = get_draws("cause_id", 294,      source="codcorrect", measure_id=1, location_id=<loc>, ...)  # all-cause deaths
```
Either path yields per-age (and per-draw, for PSA) incidence + mortality by `location_id` — write them
into `gbd_inputs.csv` (one block per location) and rerun. For any real country/subnational unit, just
change `location_id`; the pipeline already loops over locations.

## Making it real: run `export_gbd_inputs.py` on the IHME cluster

`export_gbd_inputs.py` turns this into a genuine GBD re-base with **no model changes** — it writes
`gbd_inputs.csv` in the exact schema above, pulling real draws:

```
# on a cluster node with a central-comp env active (get_draws / db_queries):
#   1. confirm RELEASE_ID (GBD 2021), CAUSE_T2D (verification block prints the name), and locations
#   2. python export_gbd_inputs.py        # writes gbd_inputs.csv (both-sex, posterior mean)
#   3. git commit gbd_inputs.csv
# then, anywhere (no cluster needed):
#   uv run python gbd_rebase_model.py
```

It exports general-population **T2D incidence** (COMO, measure 6) and **all-cause mortality**
(CoDCorrect deaths ÷ population) by GBD 5-year adult age band, both-sex, for any `location_id`s you
list — add real countries/subnationals by editing the `LOCATIONS` dict. Three things to confirm for
your release are flagged at the top of the script (release id, the T2D cause id, and `metric_id`
support). To extend to PSA later, keep the draws instead of collapsing to the mean.

## Files
- `gbd_inputs.csv` — location × age epidemiology inputs (illustrative now; overwritten with real draws by the exporter).
- `export_gbd_inputs.py` — **run on the IHME cluster** to pull real GBD draws into `gbd_inputs.csv`.
- `gbd_rebase_model.py` — the re-base model, US calibration, and location comparison.
- `outputs/gbd_rebase_results.csv` — per-location incidence, life-years, Δcost, ΔQALY, ICER, NMB.
