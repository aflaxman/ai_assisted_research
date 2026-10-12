# Replicating the MYRIAD projections for US people who inject drugs (Chiosi et al. 2026) in vivarium

Target paper: Chiosi JJ, Giardina J, Wakeman SE, Freedberg KA, Chhatwal J,
Ciaranello AL. *Projected health and economic outcomes in people who inject
drugs in the United States, 2025-2034: A modeling study.* Open Forum Infect
Dis, 9 Oct 2026, [doi:10.1093/ofid/ofag675](https://doi.org/10.1093/ofid/ofag675)
(CC BY-NC-ND). The model, MYRIAD, was first described and validated in
[Chiosi et al. 2023, BMC Health Serv Res 23:760](https://doi.org/10.1186/s12913-023-09773-1).

RESULTS_PLACEHOLDER

## Why vivarium rather than camdl

MYRIAD is an individual-level, monthly-cycle microsimulation: every simulated
person carries an age, sex, drug-use state, HIV state (infection, diagnosis,
ART, CD4 stratum), HCV state (infection, acute/chronic, METAVIR stage), an
active-bacterial-infection flag, and per-event and per-state costs that are
discounted by calendar year. Cause of death is drawn in proportion to the
person's cause-specific hazards. There is no transmission: incidence is an
exogenous calendar schedule. That is exactly the shape of a
[vivarium](https://github.com/ihmeuw/vivarium) model: one component per
condition owning its own state columns, a value pipeline that pools the
cause-specific death probabilities, and an observer that tallies events and
costs by year. Each of the paper's input tables becomes a few lines of a
component.

[camdl](https://github.com/vsbuffalo/camdl) is built for stochastic
*compartmental* models with likelihood-based inference against time series
(see `../camdl_replication/`). MYRIAD could in principle be flattened into
compartments, but the product state space (4 drug states x 13 HIV states x
15 HCV states x 2 infection flags x age x sex) runs to tens of thousands of
strata, and camdl has no notion of per-person costs, discounting, or
proportional cause-of-death attribution. There is also nothing to infer: the
paper's only calibrated inputs (five drug-use transition probabilities) are
published as point values. vivarium gives the simpler and more faithful
replication, so that is what this directory uses.

## What the model does

One time step is one month. The cohort is closed: 3,694,500 PWID in January
2018 (the paper's input year), aged through 2024 so that about 3.3 million
remain in January 2025, then followed to December 2034. Outcomes are reported
for 2025-2034 and scaled from `n` simulants to the national cohort.

| Component (`myriad/components.py`) | State | Transitions each month |
|---|---|---|
| `Population` | age (N(40, 10) truncated 18-65), sex (29.7% female) | ageing |
| `DrugUse` | low / medium / high risk, remission | S-B2 transition probabilities; overdose death by state and calendar year (S-B3; S-B5/S-B6 for the scenarios) |
| `HIV` | infected, diagnosed, on ART, CD4 (>=500, 200-499, <200) | incidence by year and IDU status (S-B11), testing (S-B17), ART start at diagnosis and monthly start/stop by CD4 (S-B14/18), CD4 progression by regimen (S-B16), HIV death (S-B21) |
| `HCV` | infected, months since infection, METAVIR F0-F4 / decompensated cirrhosis / HCC | incidence by year (S-B23), spontaneous clearance in the acute phase, fibrosis progression with HIV relative risks (S-B20, S-B25), liver death (S-B26) |
| `SevereBacterialInfection` | uncured infection flag; this month's hospitalisation type | hospitalisation hazard rising linearly (S-B7, Eq. 8) times an HIV/CD4 odds ratio (S-B19); type split (T1); in-hospital death (S-B8); cure unless self-directed discharge (S-B9) |
| `Mortality` | alive, cause of death | background mortality by single-year age and sex (S-B27) plus the four cause-specific probabilities; rates summed, cause drawn in proportion |
| `OutcomeObserver` | - | events, deaths by cause, person-months, costs by category (S-B28-B31) |

"S-Bn" refers to the supplement tables and "T1" to the main-text Table 1; every
value in `myriad/parameters.py` carries that citation.

## Assumptions where the paper is silent or inconsistent

1. **SBI hospitalisation trend.** Supplement Eq. 8 prints an intercept
   (59,664) that contradicts both the 2017 observed count (94,068) and Table
   1's 2.7% for 2018. Using the 2017 count as the intercept reproduces Table 1
   exactly, so the model uses 94,068 + 5,532 x (year - 2017) hospitalisations
   per 3,694,500 PWID, and lets the line continue through 2034 (the supplement
   says the projection covers "2018-2023"). The paper's own SBI counts rise 11%
   over 2025-2034 while the cohort shrinks by a fifth, which requires exactly
   such a rising per-capita rate.
2. **HIV and HCV incidence after the last surveillance year** (2024 and 2023)
   are held flat. Table 1 says incidence was "extrapolated thereafter" without
   saying how; flat extrapolation reproduces the paper's per-capita HIV
   incidence in 2025 to within 2%.
3. **Spontaneous HCV clearance** is not in the 2026 supplement. The 2023
   model paper lists 2.4% per month (Grebely et al.); applying it during the
   first 12 months of infection gives the roughly 25% one-year clearance that
   source reports. Cleared people keep their fibrosis stage and progress from
   F4 at the "without HCV" rates.
4. **Remission is treated as non-injecting** ("not engaging in active IDU"),
   so people in remission use the non-IDU incidence, testing, ART and
   self-discharge inputs, pay no syringe cost, and carry the remission overdose
   rate.
5. **Initial HCV-infected people are chronically infected** (past the acute
   window); prevalent decompensated cirrhosis is treated as beyond its first
   year.
6. **Costs accrue for everyone alive at the start of a month**, including
   those who die during it; the overdose-death and hospitalisation costs are
   one-off. Costs are discounted at 3% per year to 2025 by calendar year, as
   in the paper's annual tables.
7. **Sequential within-month updating** in the order ageing, drug use, HIV,
   HCV, bacterial infection, death. The paper does not state its order.

## Running it

```bash
cd myriad_pwid_vivarium_replication
uv sync
uv run pytest -q                      # ~10 s
uv run python run_replication.py --n 100000 --seeds 0 --scenarios base   # ~4 min
uv run python plot_results.py
```

`run_replication.py` writes monthly and annual CSVs, `summary.csv`, and the
comparison tables under `results/`. The full runs reported above used
`--n 369450` (one simulant per ten PWID) and three seeds for the base case.

## Files

- `myriad/parameters.py` - every input with its source table
- `myriad/components.py` - the vivarium components
- `myriad/simulation.py` - builds the simulation, scales and discounts results
- `myriad/data/background_mortality_monthly.csv` - supplement Table B27
- `run_replication.py`, `plot_results.py`, `test_myriad.py`
- `results/` - CSV outputs, comparison tables and figures
