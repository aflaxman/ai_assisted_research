# Sodium reduction in Australia: a vivarium replication of Dhungel et al. (2025)

*Replication note. The question was whether the health gains in Dhungel et al.,
"The Health and Cost Impacts of Interventions Designed to Reduce Dietary Sodium
Intake in Australia: A Modeling Study," J Nutr 155 (2025) 4486–4501,
[doi:10.1016/j.tjnut.2025.10.009](https://doi.org/10.1016/j.tjnut.2025.10.009),
can be reproduced with IHME's
[vivarium](https://github.com/ihmeuw/vivarium-suite) microsimulation framework
and GBD 2023 inputs, using only what is in the published paper. Short answer:
the ranking replicates, the absolute HALY gains come out at 65% to 88% of the
paper's medians for the seven reformulation and substitution scenarios (within
the paper's 95% uncertainty intervals for six of them), and the two smallest
programs come out at half to 60%. The mortality reductions are about half to
two-thirds of the paper's. The costs, the socioeconomic inequality results, and
the uncertainty intervals I did not attempt.*

![HALYs gained, model versus paper](results/halys_comparison.png)

## Headline

HALYs gained over 2024 to 2043 at a 3% discount rate, for Australia's
population, from `results/comparison.csv`. The model column is the mean over
three random seeds of 200,000 simulants each, with the seed-to-seed standard
deviation; the paper column is the median and 95% uncertainty interval of
Figure 1 in Dhungel et al.

| Intervention | This model | Paper | Ratio |
|---|---|---|---|
| Mandating Australia's reformulation targets | 7,200 (sd 1,300) | 11,100 (7,530 to 15,800) | 0.65 |
| Mandating the UK's targets | 18,700 (sd 1,200) | 25,300 (16,800 to 37,100) | 0.74 |
| Mandating the WHO's targets | 29,200 (sd 2,800) | 40,100 (27,200 to 56,700) | 0.73 |
| Australia then UK targets | 15,800 (sd 100) | 21,000 (14,200 to 30,500) | 0.75 |
| 30% KCl substitution, all foods, immediate | 90,100 (sd 3,000) | 103,000 (71,100 to 145,000) | 0.88 |
| 10% KCl substitution, all foods, over 10 years | 11,100 (sd 400) | 15,600 (10,500 to 22,900) | 0.71 |
| 30% KCl substitution, discretionary salt, over 3 years | 10,200 (sd 900) | 13,100 (8,970 to 19,100) | 0.78 |
| UK mass media campaign | 550 (sd 560) | 1,130 (560 to 2,100) | 0.49 |
| UK salt reduction program | 4,100 (sd 480) | 6,880 (4,640 to 9,820) | 0.59 |

The paper's Table 3 gives the age-standardised all-cause mortality rate in
2044 by socioeconomic quintile. My crude rate over 2040 to 2044 falls by 0.14%
under the WHO targets and 0.35% under immediate 30% substitution, against the
paper's 0.29% and 0.66% in the most deprived quintile and 0.18% and 0.36% in
the least deprived. For the four small programs the mortality change is within
seed noise of zero in my runs, as it nearly is in the paper's.

## What the paper does

Dhungel et al. run a proportional multistate lifetable (PMSLT) for the
Australian population of 25.3 million over 20 years, 2024 to 2043, with annual
cycles. Nine interventions lower dietary sodium by a proportion that depends on
age, sex, and year (Table 1 of the paper). The sodium reduction becomes a
systolic blood pressure (SBP) reduction through a meta-regression of 48 trials
from Huang et al. (BMJ 2020), with larger effects at older ages and higher
starting SBP. The SBP change then shifts the incidence of 18 SBP-related
diseases through GBD 2019 relative risks, with sodium acting directly on stomach
cancer. The outputs are health-adjusted life years (HALYs) gained at a 3%
discount rate, all-cause mortality by socioeconomic quintile, and costs from
three perspectives. The supplementary material with the fitted SBP equation,
the disease list, and the lifetable inputs was not available to me, so
everything below is reconstructed from the main text.

## What I built

The model lives in `src/sodium_sim/` and leans on `vivarium.public_health` for
everything generic. Each simulant has an age, sex, a sodium intake, and an SBP
drawn from normal distributions whose means come from GBD 2023 exposure
estimates, and progresses through twelve susceptible-infected disease models
(the eleven GBD 2023 outcomes of high SBP that have nonfatal estimates, fewer
than GBD 2019's, plus stomach cancer), with an all-cause mortality model that deletes the modeled
causes and adds back each disease's excess mortality. The components from the
framework are `BasePopulation`, `FertilityCrudeBirthRate`, `Mortality`, `SI`
disease models, `Risk` exposures, `RiskEffect` log-linear risk effects,
`DisabilityObserver`, and `MortalityObserver`. Three components are mine
([`components.py`](src/sodium_sim/components.py)): one scales the sodium exposure
pipeline by the scenario's schedule, one lowers the SBP exposure pipeline by the
sodium removed times the age- and SBP-specific slope, and one observes
person-time so HALYs can be assembled afterwards. Aortic aneurysm has no
nonfatal estimates in GBD 2023, so it enters through the mortality component's
"unmodeled causes" pipeline, which the SBP risk effect modifies directly.

Four reconstructions deserve a note, because each is a place where my model
and the paper's can differ.

1. **The sodium-to-SBP equation.** The paper reports one calibration point, a
   2.026 mmHg reduction per gram of sodium for a 50-year-old white person with
   SBP 140, and says the slope varies with age and starting SBP. I took the
   age and SBP slopes from the published multivariable meta-regression of Huang
   et al. (0.111 mmHg per year of age and 0.074 per mmHg, at their mean sodium
   reduction of 130 mmol/day) and rescaled them per gram
   ([`sbp_link.py`](src/sodium_sim/sbp_link.py)). A 70-year-old with SBP 150 gets
   about 3.0 mmHg per gram and a 30-year-old with SBP 120 about 0.8, which
   matches the paper's description of "very little impact on younger people and
   those with lower initial SBP," but the exact coefficients are the authors'.
2. **Relative risks per 10 mmHg.** GBD's SBP relative risks are not in the
   supplementary tables I can reach, so I backed them out of the GBD 2023
   population attributable fractions for Australia, by age and sex: the
   log-linear RR per 10 mmHg above a TMREL of 112.5 mmHg is the one whose
   population average over the assumed SBP distribution reproduces the GBD PAF
   ([`build_artifact.py`](src/sodium_sim/build_artifact.py), `solve_rr_from_paf`).
   This gives 1.35 per 10 mmHg for ischemic heart disease at age 40 falling to
   1.25 at 90, and 1.7 falling to 1.25 for the strokes, which is the familiar
   GBD age pattern. Hypertensive heart disease and CKD due to hypertension have
   PAFs of 100% by definition; I capped the PAF at 90%, which implies relative
   risks of 2 to 4 per 10 mmHg, and this cap is a choice the reader should know
   about. Within-age SBP standard deviations are assumed (11 mmHg at 25 rising
   to 20 at 75+) because the GBD extract carries only the mean.
3. **Time lags.** The paper applies the mean population impact fraction over the
   last 3 years for cardiovascular disease and over the last 2 to 20 years for
   stomach cancer. I apply the same moving averages to the proportional sodium
   reduction, which is equivalent to first order.
4. **Hypertensive heart disease incidence.** GBD 2023 publishes no incidence for
   it, so I derived one from the prevalence age gradient and the excess
   mortality, the DisMod identity `I = (dP/da + P·EMR) / (1 − P)`. Checked
   against the causes that do have incidence, this derivation lands within
   0.8 to 1.2 of GBD's figure at ages 55 and over for most of them.

The disability weight of each disease is its GBD YLD rate divided by its
prevalence, and the disability of everything not modeled is the all-cause YLD
rate minus the modeled causes', applied as a background weight in
post-processing. HALYs are person-years times one minus total disability,
discounted at 3% to 2024, and the gain is the difference from business as usual
under the same random seed, which vivarium's common random numbers make far less
noisy than the raw death counts would suggest.

## What replicates and what does not

The ordering of the nine interventions replicates exactly, and so does the
scale: immediate 30% potassium substitution is worth about nine times the
Australian reformulation targets in the paper and thirteen times in my model,
and the UK mass media campaign is two orders of magnitude below the WHO
targets in both. The absolute gains are lower than the paper's across the
board, by a factor that is fairly stable at 0.65 to 0.88 for the seven
scenarios with meaningful effect sizes and drops to about half for the two
UK programs, where the sodium reductions are under 2% and seed noise is of
the same size as the signal. Six of the nine model estimates fall inside the
paper's 95% uncertainty interval, and the three that do not (Australian
targets, the mass media campaign, and the UK program) sit just below its lower
end.

A shortfall this uniform points to the parts of the chain shared by every
scenario rather than to the scenario schedules, and the decomposition in
`results/haly_decomposition.csv` says where to look. Under the WHO targets my
model gains 23,000 discounted life-years, loses 7,500 of them to the
background disability of the extra years lived, and gains 13,800 from less
disease-specific disability. The mortality reductions are half to two-thirds
of the paper's, while the disability gains look about right, so the mortality
pathway is the suspect. Three differences could account for it. GBD 2023
models twelve outcomes of high SBP where GBD 2019 modeled seventeen, and the
paper uses the 2019 set; rheumatic heart disease, cardiomyopathy, endocarditis,
other cardiovascular disease, and CKD due to type 1 diabetes are missing from
my model entirely. The relative risks I backed out of the GBD 2023 PAFs may be
flatter than the GBD 2019 ones the paper used, especially where I capped the
PAF at 90%. And the annual time step delays each death from an acute event to
the year after onset, which for subarachnoid hemorrhage and stomach cancer
means the first years of each intervention show less mortality benefit than
a quarterly step would. The paper also lets net migration grow the population,
which adds person-years and therefore HALYs that my closed-plus-births cohort
does not have; I would guess this accounts for a few percent.

The mortality comparison carries a second caveat: my rate is crude and
averaged over 2040 to 2044, the paper's is age-standardised for 2044 alone,
and with 200,000 simulants one death is worth 1,350 real deaths, so a seed
standard deviation of 0.02 to 0.07 percentage points is the best I can do
without a larger run.

## What I did not attempt

The paper's inequality results (Table 3) need the SEIFA disaggregation of every
input, which SHINE's heterogeneity module does and I have no data for. The costs
need the AIHW disease expenditure by phase, New Zealand income loss estimates,
and the Grattan intervention costings, none of which are in the main text at a
usable level of detail. The 95% uncertainty intervals in the paper come from
2,000 draws over every input; what I report is the spread over three random
seeds, which is Monte Carlo noise only and should not be read as an uncertainty
interval. Migration and the trends in disease rates after 2023 are also left
out: the simulated population only ages, is born, and dies.

## Quickstart

```bash
cd sodium_reduction_australia_replication
uv venv -p 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m sodium_sim.build_artifact       # data/gbd/*.csv -> artifacts/australia.hdf
./run_all.sh 200000 0 1 2                            # 10 scenarios x 3 seeds, 4 in parallel, ~90 minutes
```

The raw per-run output lands in `results/raw/` (43 MB, not committed) and the
last command ends by writing `results/comparison.csv`,
`results/gains_by_seed.csv` and `results/halys_comparison.png`. A single scenario
runs with `python -m sodium_sim.run --scenarios mandate_who --seeds 0 --population 20000`
in about four minutes. Tests: `.venv/bin/python -m pytest`.

## Files

- `data/gbd/` GBD 2023 extracts for Australia pulled through the IHME GBD MCP
  server (population, live births, all-cause deaths and YLDs, and for each
  cause its incidence, prevalence, deaths, YLDs, and SBP or sodium PAF).
- `src/sodium_sim/build_artifact.py` turns those into a vivarium artifact and
  writes the risk-effect diagnostics to `results/risk_effect_diagnostics_*.csv`.
- `src/sodium_sim/scenarios.py` the nine intervention schedules from Table 1,
  and the paper's Figure 1 numbers for comparison.
- `src/sodium_sim/components.py`, `sbp_link.py` the model-specific components.
- `src/sodium_sim/model_spec.yaml` the vivarium model specification.
- `src/sodium_sim/run.py`, `analyze.py`, `plot.py` run, summarize, draw.

## Open directions

The largest open item is the SBP equation; with the authors' Supplemental
Table 3 the two reconstructions could be swapped for the real coefficients in
an afternoon. Costs are a bigger lift but a well-trodden one, since vivarium
models routinely carry per-state expenditure. The inequality analysis is the
hardest to reproduce without IHME-internal data. The annual time step also
deserves a check: it matches the paper's lifetable cycle, but a quarterly step
would let acute, highly fatal conditions such as subarachnoid hemorrhage kill
within the year of onset instead of the year after, and I have not measured how
much that moves the answer.
