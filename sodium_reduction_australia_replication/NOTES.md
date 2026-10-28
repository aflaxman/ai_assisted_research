# Running notes: sodium reduction replication

Working log for the replication of Dhungel et al. (2025). Newest entries at
the bottom. The README holds the write-up; this file holds what I tried, in
order, including the dead ends.

## 2026-10-27 (cloud session, summarized)

- Built the model on the Claude cloud sandbox from GBD 2023 pulled through
  the IHME GBD MCP server. Results: HALY gains 65% to 88% of the paper's
  medians for the seven main scenarios, about half for the two small UK
  programs; mortality reductions about half to two-thirds of the paper's.
- Suspects for the shortfall, from the decomposition: the mortality pathway
  (deaths averted), not the disability pathway.
- Supplementary material was not reachable from the sandbox (ScienceDirect 403).

## 2026-10-27 (cluster session)

Session moved to the IHME cluster. New inputs available: the paper's
supplementary appendix (uploaded by Abie), `vivarium_inputs` and `gbd_mapping`
against the GBD databases, the old `ihmeuw/vivarium_sodium_reduction` repo,
and the recent `vivarium_csu_mace_rct` project for data access patterns.

Question to answer: why are deaths averted smaller in my replication?

### Appendix, first read (mmc1.pdf, 28 pages)

- **Table S3 gives the SBP equation.** Per gram of sodium removed:
  `2.026 + 0.502 * (age - 50)/10 + 0.659 * (SBP - 140)/10 + 2.353 * p_nonwhite`,
  with p_nonwhite = 0 for Australia. My reconstruction from Huang et al. had
  0.371 per decade of age and 0.248 per 10 mmHg, so the paper's age slope is
  1.35x mine and its SBP slope 2.7x mine. At age 70 and SBP 150 the paper
  gets 3.69 mmHg per gram against my 3.0; at 80 and 160, 4.85 against 3.6.
  Deaths are concentrated exactly there, so this is a prime suspect.
- Table S5 confirms the equation is floored at zero (young, low-SBP cells
  show 0.00) and shows about 3.6 mmHg per gram at 70+ and SBP 140 to 149.
- **Sodium TMREL of 2 g/day for the SBP link**: only intake above 2 g/day
  counts, following Trieu et al. Needs checking against the sodium means
  (Table S1): if the paper's baseline intake is higher than GBD 2023's
  2.1 to 4.1 g/day, the absolute reduction (proportion x baseline) is larger
  in the paper for the same proportional cut.
- **Table S6, the 18 diseases**: IHD, ischemic stroke, ICH, SAH, hypertensive
  heart disease, atrial fibrillation, stomach cancer, non-rheumatic calcific
  aortic valve disease, other cardiomyopathy, CKD (T1 diabetes, T2 diabetes,
  hypertension, glomerulonephritis, other), rheumatic heart disease, aortic
  aneurysm, endocarditis, peripheral artery disease. Mine has 13 of these
  (no T1-diabetes CKD, calcific aortic valve, other cardiomyopathy, rheumatic
  heart disease, endocarditis, since GBD 2023 no longer attributes them to SBP).

### Baseline exposures: the paper's Tables S1 and S2 versus GBD 2023

Table S1 (Trieu et al.) gives sodium means by sex, age and SEIFA quintile;
averaging the five quintiles, men run from 4.4 g/day at 25 to 29 down to
3.1 at 80+, women from 3.2 to 2.6. GBD 2023's exposure for Australia, which
my first artifact used, is 3.6 to 2.4 for men and 2.8 to 2.2 for women. The
proportional reductions in Table 1 therefore remove 20 to 45% more sodium
in the paper than in my model, most of the gap at the ages where the deaths
are. Table S2's SBP means are close to GBD's for men but 5 to 7 mmHg lower
for women at older ages, with SDs of 12 (age 25) to 21 (age 85); my assumed
SDs were 11 to 20, so that part was fine.

Decision: switch the sodium and SBP exposure inputs to the paper's tables
(quintiles averaged with equal weight, since SEIFA quintiles are equal-size
by construction) and keep GBD for the disease rates and relative risks.
Transcribed tables are in `data/paper/`.

### Cluster resources

- `vivarium_inputs` 7.2.4 + `gbd_mapping` 5.0.4 in the
  `vivarium_csu_alzheimers_artifact` conda env, pointed at GBD 2023
  (release 16). Pulls for Australia are running in the background into
  the scratchpad; each measure takes 10 to 60 s against the database.
- GBD 2023 attributes high SBP to 12 causes (same set as GBD 2021); the
  paper's extra five are GBD 2019 (release 6) outcomes only.
- Slurm: partitions `all.q` and `long.q`, account `proj_simscience`
  (from the MACE RCT sbatch scripts). This node has 4 CPUs.

### What the appendix inputs do to the SBP reduction (no simulation yet)

Mean SBP reduction under the WHO targets in 2030, by sex and age, old inputs
(my Huang reconstruction, GBD 2023 sodium and SBP) against the appendix
inputs (Table S3 equation with the 2 g floor, Table S1 sodium, Table S2 SBP):

| | men 45 | men 60 | men 70 | men 80+ | women 50 | women 65 | women 80+ |
|---|---|---|---|---|---|---|---|
| old, mmHg | 0.76 | 0.84 | 0.85 | 1.0 to 1.2 | 0.57 | 0.71 | 0.9 to 1.1 |
| new, mmHg | 0.39 | 1.05 | 1.36 | 1.6 to 1.9 | 0.32 | 0.75 | 1.1 to 1.4 |
| ratio | 0.5 | 1.25 | 1.6 | 1.55 | 0.55 | 1.05 | 1.25 |

Under 40 the appendix equation gives zero (the age and SBP terms pull the
slope below zero, and Table S5 shows the same zeros), so the young lose
their small effect, while men over 60 gain 25 to 60% and women over 75
about 25%. Deaths sit at those ages, so this should close a good part of
the mortality gap. The sodium baseline matters more than the equation for
men over 60: with the new equation but GBD sodium the 70+ effect is 0.9 to
1.4, with the paper's sodium 1.4 to 1.9.

### GBD 2023 relative risks from vivarium_inputs

`get_measure(high_systolic_blood_pressure, "relative_risk", "Australia")`
returns a Burden-of-Proof curve on a 1,000-point grid from 90 to 200 mmHg
for seven causes only (AF, HHD, ICH, IHD, ischemic stroke, PAD, SAH); the
four CKD causes and aortic aneurysm are absent, and HHD is all zeros since
GBD treats it as a PAF-of-one cause. The curves are close to log-linear
above the TMREL: at age 65 the slope per 10 mmHg is 1.33 for IHD, 1.38 for
the three strokes, 1.19 for AF, 1.04 for PAD. My PAF-implied log-linear RRs
were 1.30, 1.38, 1.14 and 1.05 at that age, so the first artifact's relative
risks were not the problem. Plan: fit a log-linear slope to each curve
between 115 and 175 mmHg and keep the PAF-implied values for the causes
without curves.

### Experiments queued (cluster)

1. **v2 artifact**: GBD 2023 cause data and fitted relative risks from
   vivarium_inputs, the paper's sodium and SBP distributions, the Table S3
   SBP equation. Full 10 scenarios x 3 seeds x 200k on a 16-core sbatch job
   once the pull finishes (about 20 cause-measure pulls at 10 to 60 s each
   still to go).
2. **Time-step ablation** on the first artifact: BAU and WHO targets at
   100k simulants, seed 0, annual (365.25 d) versus quarterly (91.3 d)
   steps, to measure how much the annual cycle's one-year delay of acute
   deaths costs. Running now on this node, tags `v1_annual`, `v1_quarterly`.
   The raw results from the cloud session were never committed, so the
   annual run is needed as the matching baseline.

### Log-linear or the GBD curves?

Abie pointed out that `vivarium_public_health` has `NonLogLinearRiskEffect`
for curve-form relative risks (the MACE RCT project stores the GBD curve
under the standard `relative_risk` key with a numeric `parameter` column and
computes PAFs in a separate PAF-calculation spec). The paper used GBD 2019
incidence rate ratios per 10 mmHg, which are log-linear, so the log-linear
fit stays the main specification as the closer match to the published
method. The curves go into a second spec, `model_spec_nonloglinear.yaml`,
for the six SBP causes with usable curves (IHD, the three strokes, AF, PAD)
and for sodium on stomach cancer, with PAFs computed analytically from the
curve under the paper's exposure distribution (keys `*.relative_risk_curve`
and `*.population_attributable_fraction_curve`). Runs select the spec with
`--spec`; this is a sensitivity analysis, not the headline.

### The paper is GBD 2019 throughout (Abie, from Tony Blakely's group)

The group did not have time to download the updated GBD results, so every
GBD input in the paper is GBD 2019: the 18-cause list, the relative risks
per 10 mmHg (GBD 2019 appendix 2b), the PIFs, and the disease rates
forecast from 1990 to 2019 trends. My model is GBD 2023 (release 16).
Candidates this opens up: GBD 2019 relative risks were larger for some
causes than the 2023 Burden-of-Proof slopes; GBD 2023 revised Australian
CVD mortality and has the COVID years in the series; the five causes GBD
2023 no longer attributes to SBP. Pulling GBD 2019 (release 6) SBP relative
risks with `get_draws` for a direct comparison, and testing whether
`get_outputs(release_id=6)` can give the 2019 cause rates for a GBD 2019
version of the artifact.

### Time-step ablation (first artifact, 100k simulants, seed 0, BAU vs WHO targets)

| step | HALYs gained | deaths averted | ACMR change 2040-44 |
|---|---|---|---|
| annual (365.25 d) | 24,700 | 6,470 | -0.18% |
| quarterly (91.3 d) | 32,900 | 4,310 | -0.16% |

One seed only, and at 100k a single death is 270 real deaths, so the death
counts are noise; the HALY difference is mostly disability (prevalence
dynamics), which is much less noisy. The quarterly step gives a third more
HALYs. I had argued the annual step matched the paper's annual cycle, but
vivarium's discretization is not a PMSLT cycle: incidence in a year lands in
the state only at the next step, and the clipped probabilities delay acute
deaths. The quarterly step is the better vivarium choice; the production
runs on the cluster will use it (about four times the cost, affordable with
16 cores).

### GBD 2023 pull finished

115 cause-measure and risk pulls done (one benign failure: other
cardiomyopathy has no incidence in GBD 2023 and is not in the 2023 cause
set anyway). Condensing into `data/gbd_vi/` and building the v2 artifact.

### GBD 2019 versus GBD 2023 inputs for Australia (men; full table in results/gbd2019_vs_gbd2023_rates_males.csv)

| | age 55 | age 70 | age 85 |
|---|---|---|---|
| IHD deaths, 2019/2023 | 1.04 | 1.30 | 1.45 |
| IHD incidence, 2019/2023 | 4.5 | 4.1 | 2.4 |
| IHD prevalence, 2019/2023 | 1.6 | 2.6 | 2.4 |
| ischemic stroke deaths, 2019/2023 | 1.04 | 1.55 | 1.65 |
| ischemic stroke prevalence, 2019/2023 | 0.65 | 0.69 | 0.50 |
| all-cause deaths, 2019/2023 | 1.06 | 1.19 | 1.12 |

GBD 2023 cut Australian cardiovascular mortality at the ages that matter
by a third or more relative to GBD 2019, and the GBD 2019 IHD incidence is
a different quantity (it includes angina and chronic IHD, where GBD 2023
counts acute MI only). Both push the paper's deaths averted above mine.
The GBD 2019 artifact (`artifacts/australia_gbd2019.hdf`, 18 causes, GBD
2019 relative risks per 10 mmHg, paper exposures) is built; smoke test
running, then the full set on the cluster at quarterly steps.

### GBD 2019 smoke test (20k simulants, seed 0, annual steps, BAU vs WHO)

WHO targets: 46,900 HALYs gained (paper 40,100), 9,430 deaths averted,
crude mortality rate down 0.28% over 2040 to 2044 (the paper's Table 3
gives 0.29% in the most deprived quintile and 0.18% in the least). One
seed at 20k is noisy (one death is 1,350 real deaths), but the mortality
gap is gone with GBD 2019 inputs plus the appendix's SBP equation and
exposures. Baseline BAU deaths 198,000 in 2024 against GBD 2019's 2019
count of about 170,000 scaled to the 2023 population, which is the
expected shape (the 2023 population is older). Submitted the full set to
Slurm: 10 scenarios x 3 seeds x 200k, quarterly steps, tag `gbd2019`.

### GBD 2023 (v2) artifact: last data gaps

vivarium_inputs refuses aortic aneurysm nonfatal measures ("YLL only
cause"), has no HHD incidence (derived from prevalence and EMR, as in the
first artifact) and no other-cardiomyopathy incidence (not in the 2023 SBP
set anyway). The five peripheral arterial disease files were missing from
the first pull for no logged reason; re-pulling. The GBD 2019 production
set (Slurm job 64758034, 16 cores) is running.

### GBD 2023 (v2) artifact built

`artifacts/australia_v2.hdf`: 12 causes from vivarium_inputs (draw means),
HHD incidence derived, aortic aneurysm fatal-only, the paper's sodium and
SBP distributions, log-linear slopes fitted to the GBD 2023 curves for IHD,
the three strokes, AF and PAD, PAF-backed-out slopes for CKD and aortic
aneurysm (capped PAF for HHD and CKD due to hypertension), and curve-form
relative risks plus consistent PAFs for the non-log-linear sensitivity
spec. Smoke tests of both specs running (BAU and WHO, 20k, seed 0).

### Smoke tests side by side (BAU vs WHO targets, 20k simulants, seed 0, annual steps)

| inputs | HALYs gained | deaths averted | ACMR change 2040-44 | ratio to paper |
|---|---|---|---|---|
| first artifact (GBD 2023 MCP, my Huang reconstruction), 100k | 24,700 | 6,470 | -0.18% | 0.62 |
| GBD 2023 via vivarium_inputs, appendix equation and exposures | 22,400 | 5,390 | -0.29% | 0.56 |
| GBD 2019 via get_outputs, 18 causes, appendix equation and exposures | 46,900 | 9,430 | -0.28% | 1.17 |

Twenty thousand simulants is noisy (the GBD 2023 line's deaths averted
equals four simulant deaths), but the pattern is clear: the appendix's SBP
equation and exposures do not by themselves close the gap under GBD 2023,
while switching the disease rates and relative risks to GBD 2019, the
release the paper used, overshoots it. The GBD release is the dominant
driver. A bug found on the way: the slopes fitted to the GBD 2023 curves
were stored per 1 mmHg while the artifact applies them per 10 mmHg, which
made the first GBD 2023 smoke test show no effect at all; fixed before the
numbers above. Production sets for both releases are on Slurm (quarterly
steps, 200k, 3 seeds); the non-log-linear sensitivity follows.

### Caveat on the smoke tests, and the non-log-linear smoke

The non-log-linear spec's smoke test (GBD 2023 curves) gave 38,100 HALYs
against 22,400 for the log-linear fit on the same artifact, seed and
population, yet a direct calculation shows the two forms give the same
IHD incidence reduction per 1.3 mmHg (0.964 versus 0.965 at age 70). The
curves sit at RR 1.0 at 95 mmHg, 1.04 at 105, 1.25 at 115, so they add
little below the TMREL. The difference is therefore noise: at 20,000
simulants a single WHO run moved from 23,000 to 35,000 HALYs between 20k
and 200k in the cloud session. Smoke tests say whether a model runs and
roughly where it lands; the three Slurm sets (GBD 2019, GBD 2023
log-linear, GBD 2023 curves; jobs 64758034, 64763287, 64763402) are the
numbers to read.

### GBD 2019 production set, early read (seed 0 of 3, quarterly steps, 200k)

WHO targets 67,500 HALYs (paper 40,100), immediate 30% KCl 163,000
(103,000), UK program 10,700 (6,880): ratios of 1.5 to 1.7 across the
board, with the crude mortality change for the WHO targets at -0.34%
(paper -0.29% most deprived, -0.18% least). So the GBD 2019 inputs
overshoot where the GBD 2023 inputs undershoot. Two known differences
push the GBD 2019 variant up relative to the paper: the quarterly step (a
third more HALYs than annual in the ablation) and the absence of the
paper's BAU trends, which carry GBD 2019's 1990 to 2019 declines in
incidence and case fatality forward to 2034.

### BAU rate trends (fourth set)

`RateTrends` scales each cause's incidence and excess mortality, aortic
aneurysm mortality and the all-cause hazard by exp(apc x (min(year, 2034)
- 2019)), with apc the 2000 to 2019 log-linear slope of GBD 2019
age-standardised rates for Australia (`data/gbd2019/trends_apc.csv`,
capped at +/-4% a year; case fatality is deaths over prevalence). IHD
case fatality falls 3.6% a year and incidence 1%, ischemic stroke 3% and
1.8%, all-cause mortality 1.5%. By 2034 IHD excess mortality is 58% of
its 2019 level, so this should pull the GBD 2019 variant down toward the
paper. Spec `model_spec_gbd2019_trends.yaml`.

### GBD 2019 incidence is not prevalence-consistent; artifact rebuilt

Checking the DisMod identity I = (dP/da + P EMR)/(1 - P) against the
published GBD 2019 incidence: IHD's published incidence is 1.5 to 1.8 times
the consistent value (it counts angina and chronic entries), PAD's 1.3 to
1.7, while ischemic stroke and AF agree within 10%. Under an SI model the
published IHD incidence would inflate the IHD pool over 20 years and with
it every IHD death and every death the intervention averts. The GBD 2019
artifact now derives incidence from prevalence and EMR for every cause,
which is the consistency step SHINE's input processing performs. The
runs with the published incidence were stopped at 16 of 30 and kept under
tag `gbd2019_gbdinc` for the record; the GBD 2019 and GBD 2019 plus
trends sets were resubmitted. The baseline rise in deaths from about
190,000 in 2024 to 330,000 in 2044 is the same in every variant, so it is
the ageing of the 2023 population without migration, not an artefact of
one release.

`RateTrends` bug found in its smoke test: DiseaseModel components are named
`disease_model.<cause>`, so no cause-level modifier matched; fixed.

### GBD 2023 production sets, final (200k simulants, quarterly steps)

Log-linear fit to the GBD 2023 curves, 3 seeds; curve form
(NonLogLinearRiskEffect), 2 seeds. Ratio of mean HALYs gained to the
paper's median, and the crude mortality-rate change over 2040-44:

| intervention | log-linear | curves | paper 95% UI covers both? | ACMR change, log-linear / curves / paper (SEIFA 1, SEIFA 5) |
|---|---|---|---|---|
| Mandating Australia | 1.30 | 1.20 | yes | -0.03 / -0.05 / -0.08, -0.05 |
| Mandating UK | 1.17 | 1.17 | yes | -0.14 / -0.16 / -0.18, -0.11 |
| Mandating WHO | 1.18 | 1.16 | yes | -0.21 / -0.24 / -0.29, -0.18 |
| Australia then UK | 1.21 | 1.19 | yes | -0.13 / -0.16 / -0.17, -0.11 |
| 30% KCl immediate | 1.19 | 1.17 | yes | -0.44 / -0.47 / -0.66, -0.36 |
| 10% KCl over 10 y | 1.11 | 1.00 | yes | -0.13 / -0.17 / -0.19, -0.12 |
| 30% KCl discretionary | 1.44 | 1.34 | yes | -0.07 / -0.07 / -0.11, -0.07 |
| UK mass media | 1.35 | 1.58 | yes | 0.00 / 0.00 / 0.00, -0.01 |
| UK salt program | 1.05 | 1.23 | yes | 0.00 / -0.02 / -0.05, -0.03 |

Every estimate sits inside the paper's uncertainty interval and the
mortality reductions fall between the paper's two quintiles. The curve
form and the log-linear fit agree to within seed noise, as the direct
calculation said they should. Compared with the cloud session's 0.65 to
0.88, the move came from three things applied together: the appendix's SBP
equation and sodium baselines (the SBP reduction at 60+ up 25 to 60%), the
quarterly step (a third more HALYs than annual), and vivarium_inputs'
DisMod-consistent excess mortality in place of CSMR over prevalence.
