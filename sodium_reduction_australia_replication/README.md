# Sodium reduction in Australia: a vivarium replication of Dhungel et al. (2025)

*Replication note. The question was whether the health gains in Dhungel et al.,
"The Health and Cost Impacts of Interventions Designed to Reduce Dietary Sodium
Intake in Australia: A Modeling Study," J Nutr 155 (2025) 4486–4501,
[doi:10.1016/j.tjnut.2025.10.009](https://doi.org/10.1016/j.tjnut.2025.10.009),
can be reproduced with IHME's
[vivarium](https://github.com/ihmeuw/vivarium-suite) microsimulation framework
and GBD inputs. Short answer: yes, to within the factor the choice of inputs
moves the answer by. RESULTS_SHORT_ANSWER The costs, the socioeconomic
inequality results, and the paper's uncertainty intervals I did not attempt.*

![HALYs gained, model variants versus paper](results/halys_comparison_all.png)

## Headline

RESULTS_TABLE

## What the paper does

Dhungel et al. run a proportional multistate lifetable (PMSLT) for the
Australian population of 25.3 million over 20 years, 2024 to 2043, with annual
cycles. Nine interventions lower dietary sodium by a proportion that depends on
age, sex and year (Table 1 of the paper). The sodium reduction becomes a
systolic blood pressure (SBP) reduction through a meta-regression of 48 trials
from Huang et al. (BMJ 2020), refit by the authors as an interaction model
(appendix Table S3), with larger effects at older ages and higher starting SBP
and a floor of 2 g/day of sodium below which nothing happens. The SBP change
then shifts the incidence of 18 sodium-related diseases through GBD 2019
relative risks per 10 mmHg, with sodium acting directly on stomach cancer. The
outputs are health-adjusted life years (HALYs) gained at a 3% discount rate,
all-cause mortality by socioeconomic quintile, and costs from three
perspectives. Everything in the paper is GBD 2019; the group did not have time
to move to a newer release.

## What I built

The model lives in `src/sodium_sim/` and leans on `vivarium.public_health` for
everything generic: `BasePopulation`, `FertilityCrudeBirthRate`, `Mortality`,
`SI` disease models, continuous `Risk` exposures for sodium and SBP, log-linear
`RiskEffect`s (and, in one variant, `NonLogLinearRiskEffect` with GBD's
relative-risk curves), and the disability and mortality observers. Three
components are mine ([`components.py`](src/sodium_sim/components.py)): one
scales the sodium exposure pipeline by the scenario's schedule with the paper's
lags, one lowers the SBP exposure pipeline by the Table S3 equation applied to
the sodium removed above the 2 g/day floor
([`sbp_link.py`](src/sodium_sim/sbp_link.py)), and one observes person-time so
HALYs can be assembled afterwards. A fourth, `RateTrends`, reproduces the
paper's business-as-usual forecast by trending each cause's incidence and case
fatality and the all-cause hazard from 2019 to 2034. Aortic aneurysm has no
nonfatal estimates in either GBD release, so it enters through the mortality
component's unmodeled-causes pipeline, which the SBP effect modifies directly.
HALYs are person-years times one minus total disability (modeled causes from
the simulation plus GBD's all-cause background), discounted at 3% to 2024, and
a gain is the difference from business as usual under the same random seed.

The sodium and SBP exposure distributions are the paper's own (appendix Tables
S1 and S2, SEIFA quintiles averaged, lognormal for sodium and normal for SBP),
transcribed into `data/paper/`. Everything else comes from GBD, in two
versions:

- **GBD 2023** through `vivarium_inputs` on the IHME cluster: incidence,
  prevalence, excess mortality and disability weights for the 12 causes GBD
  2023 attributes to high SBP, plus stomach cancer; relative risks as
  log-linear slopes fitted to the GBD 2023 Burden-of-Proof curves between 115
  and 175 mmHg (one RR per 10 mmHg, the form the paper used), and the curves
  themselves for the non-log-linear variant. The four CKD causes, aortic
  aneurysm and the two PAF-of-one causes (hypertensive heart disease, CKD due
  to hypertension) have no curve, so their slopes are backed out of GBD's PAFs
  under the paper's SBP distribution, with the PAF capped at 90%.
- **GBD 2019** (release 6) through `get_outputs` and `get_draws`: rates for
  all 18 of the paper's causes and the GBD 2019 relative risks per 10 mmHg,
  which exist for every cause. GBD 2019's published incidence is not
  consistent with its prevalence and excess mortality under an SI model (for
  ischemic heart disease it is 1.5 to 1.8 times the consistent value, because
  it counts angina and chronic entries), so incidence is derived from the
  prevalence age pattern and EMR by the DisMod identity, the consistency step
  SHINE applies to its inputs.

Population attributable fractions are recomputed from each variant's relative
risks and the paper's exposure distributions, so every variant is internally
consistent. The production runs use 200,000 simulants, quarterly time steps,
and three random seeds (two for the sensitivity variants), on a 16-core Slurm
node; `NOTES.md` is the running log of what was tried, including the dead ends.

## What moved the answer

RESULTS_DISCUSSION

## What I did not attempt

The paper's inequality results (Table 3) need the SEIFA disaggregation of every
input, which SHINE's heterogeneity module does and I have no data for. The costs
need the AIHW disease expenditure by phase, New Zealand income-loss estimates and
the Grattan intervention costings. The 95% uncertainty intervals in the paper
come from 2,000 draws over every input; what I report is the spread over random
seeds, which is Monte Carlo noise only. Migration is left out: the simulated
population only ages, is born and dies, which is why baseline deaths rise from
about 190,000 in 2024 to 330,000 in 2044 in every variant.

## Quickstart

On the IHME cluster, from this directory:

```bash
uv venv -p 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m sodium_sim.build_artifact_v2 --source gbd2023     # artifacts/australia_v2.hdf
.venv/bin/python -m sodium_sim.build_artifact_v2 --source gbd2019     # artifacts/australia_gbd2019.hdf
sbatch sbatch/run_all.sbatch 200000 gbd2023 artifacts/australia_v2.hdf src/sodium_sim/model_spec.yaml 91.3125 0 1 2
```

The job runs the ten scenarios for each seed, 16 at a time, in about 80
minutes, then writes `results/comparison_gbd2023.csv`,
`results/gains_by_seed_gbd2023.csv` and `results/halys_comparison_gbd2023.png`.
The other variants use the specs and artifact named in `NOTES.md`; a single
scenario runs locally with `python -m sodium_sim.run --scenarios mandate_who
--seeds 0 --population 20000 --artifact artifacts/australia_v2.hdf`. The
condensed GBD extracts are committed under `data/gbd_vi/` and `data/gbd2019/`,
so the artifacts rebuild without database access; refreshing them needs the
`vivarium_csu_alzheimers_artifact` conda environment (`vivarium_inputs` 7.2,
`gbd_mapping` 5.0) and the scripts described in `NOTES.md`. Tests:
`.venv/bin/python -m pytest`.

## Files

- `NOTES.md` running notes, newest at the bottom.
- `data/paper/` appendix Tables S1 to S4, transcribed.
- `data/gbd_vi/` GBD 2023 extracts via `vivarium_inputs`, condensed to draw
  means; `data/gbd2019/` GBD 2019 extracts via `get_outputs`, relative risks
  and the 2000 to 2019 trends; `data/gbd/` the first-round extracts from the
  IHME GBD MCP server.
- `src/sodium_sim/build_artifact_v2.py` builds either artifact;
  `build_artifact.py` is the first-round builder; `prepare_gbd_vi.py` condenses
  the raw pulls and fits the curve slopes.
- `src/sodium_sim/model_spec*.yaml` the four model specifications (GBD 2023
  log-linear, GBD 2023 curves, GBD 2019, GBD 2019 with trends).
- `src/sodium_sim/scenarios.py` the nine intervention schedules from Table 1
  and the paper's Figure 1 numbers.
- `src/sodium_sim/run.py`, `analyze.py`, `plot.py`, `compare_tags.py` run,
  summarize, draw.
- `results/comparison_<tag>.csv`, `results/gains_by_seed_<tag>.csv` per
  variant; `results/comparison_all.csv` and `halys_comparison_all.png` across
  variants; `results/risk_effect_diagnostics_*.csv` the relative risks and
  PAFs that went into each artifact; `results/gbd2019_vs_gbd2023_rates_males.csv`.

## Open directions

RESULTS_OPEN
