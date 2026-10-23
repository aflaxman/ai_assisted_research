# Plan: TB natural-history model-structure comparison

Status: **pre-code plan, awaiting go-ahead.** No model code has been written.
Everything below was checked against sources I could actually open on
2026-10-12; anything I could not open is marked **MISSING**.

Contents

1. The 12 Menzies (2018) structures, transcribed from Figure 1
2. Spectrum models and the 3 variants I propose
3. What each family can and cannot be fit to (the comparison design)
4. Transmission embedding, GBD targets, hold-outs, ACT3
5. Software design and milestones
6. Data inventory: obtained vs missing
7. Decisions (resolved 2026-10-12)

---

## 1. The 12 Menzies structures (Figure 1, PMC6070419)

I downloaded the Figure 1 image from PMC and transcribed it directly, so the
compartments and arrows below are read from the figure, not inferred.
The study counts are from the paper's Table (312 included studies; a study
can use more than one structure) and agree with the replication data on
Harvard Dataverse (doi:10.7910/DVN/STRA2P, 680 model-stratum rows).

Notation (from the Figure 1 legend): S susceptible; L latent; Ls slow
latent; Lf fast latent; I active disease; λ force of infection;
c progression rate from L or Ls; d progression rate from Lf;
e transfer Lf→Ls; f transfer Ls→Lf; a probability of immediate progression
on infection; b probability of entering Lf on infection.

| Structure | Compartments | Transitions (rate) | Free NH params | Notes | Studies (of 312) |
|---|---|---|---|---|---|
| A | S, L, I | S→L (λ); L→I (c) | c | constant progression hazard; = E with a=0 | 60 (19.2%) |
| B* | S, Lf1…Lfn, Ls, I | S→Lf1 (λ); Lf_i→Lf_{i+1} (tunnel, fixed dwell); Lf_i→I (d_i); Lfn→Ls; Ls→I (c) | d_1..d_n, c, n, dwell | "tunnel" recent-infection states, each with its own progression risk | 27 (8.7%) |
| C | S, Lf, Ls, I | S→Lf (λ); Lf→I (d); Lf→Ls (e); Ls→I (c) | d, e, c | = G with a=0 | 33 (10.6%) |
| D | S, I | S→I (λ) | none | immediate progression; used only for advanced HIV | 3 (1.0%) |
| E | S, L, I | S→I (λa); S→L (λ(1−a)); L→I (c) | a, c | the "standard" Blower-type structure | 153 (49.0%) |
| F | S, Ls, Lf, I | S→Ls (λ(1−b)); S→Lf (λb); Ls→I (c); Lf→I (d) | b, c, d | parallel latent compartments, no transfer | 35 (11.2%) |
| G | S, Lf, Ls, I | S→I (λa); S→Lf (λ(1−a)); Lf→I (d); Lf→Ls (e); Ls→I (c) | a, d, e, c | C plus immediate progression | 1 (0.3%) |
| H | S, Ls, Lf, I | S→Ls (λ(1−b)); S→Lf (λb); Ls→Lf (f); Lf→I (d) | b, f, d | progression only from Lf; Ls feeds Lf | 2 (0.6%) |
| I | S, Lf, Ls, I | S→Lf (λ); Lf→I (d); Lf→Ls (e); Ls→Lf (f) | d, e, f | progression only from Lf; Ls can return to Lf | 2 (0.6%) |
| J† | S, L1…Ln, I | S→L1 (λ); L_i→L_{i+1} (f_i); Ln→I (c) | f_1..f_{n−1}, c, n | progression risk *rises* with time since infection | 1 (0.3%) |
| K‡ | S, L, I | S→L (λ); L→I (c_t), c_t = x1·max(1, t)^{x2} | x1, x2 | t = time since infection; individual-based in the source (Blaser 2016) | 1 (0.3%) |
| L‡ | S, L, I | S→L (λ); L→I (c_t), c_t = x1·(x2 + exp(−x3·t)) | x1, x2, x3 | individual-based in the source | 1 (0.3%) |

Things the paper does **not** supply, which matter for us:

- **No recommended parameter values.** The paper is a validity review; it
  reports each model's predicted cumulative incidence and compares with two
  empirical curves (BCG-trial placebo arms, Ferebee 1970; Sutherland 1968).
  The Dataverse replication files do contain every reviewed model's
  parameter values (columns a, b, c, d, e, f, d1..d10, x1..x3, per stratum
  such as HIV+, children). I will use these as *literature priors*, clearly
  labelled as such, never as data.
- **No disease-exit transitions.** The structures end at I. To produce
  prevalence, duration and deaths we need I→recovered and I→dead. I propose
  one shared exit block for all 12 (self-cure and TB death rates informed by
  Tiemersma 2011 Table 3 survival data, see §3), so the comparison isolates
  the *entry* structure, which is what Menzies catalogued.
- **Reinfection was deliberately excluded** from the review. In the
  transmission model I will treat reinfection (L→Lf or L→I at λ·(1−p),
  recovered→Lf at λ·(1−p)) as an orthogonal switch applied identically to
  every structure, not as part of the structure definition.
- **B, J, K, L are not plain ODEs** in their sources (tunnel states, chains,
  time-since-infection hazards in individual-based models). In an ODE
  framework I will implement B and J as compartment chains (Erlang dwell
  times) and K and L as chains with piecewise-constant hazards on the
  time-since-infection grid, and I will say so in the write-up. The cohort
  (no-λ) versions of K and L are exact in closed form, so the cohort fit is
  not approximated; only the transmission embedding is.

## 2. Spectrum models

### What the two source models look like

**Richards 2023 (Lancet Glob Health), repo `natural_history_tb_disease`**:
Minimal ⇄ Subclinical ⇄ Clinical, Minimal→Recovery, Clinical→Death. No
infection state. Parameters (per year) and posterior medians from the repo's
`latest_param_table.csv`:

| param | meaning | median (2.5%, 97.5%) |
|---|---|---|
| min_out | Minimal→recovered | 0.19 (0.15, 0.25) |
| min_sub | Minimal→Subclinical | 0.25 (0.21, 0.29) |
| sub_min | Subclinical→Minimal | 1.50 (1.16, 1.93) |
| sub_clin | Subclinical→Clinical | 0.69 (0.55, 0.87) |
| clin_sub | Clinical→Subclinical | 0.58 (0.47, 0.73) |
| mort | Clinical→death | 0.33 (0.27, 0.38); prior N(0.389, 0.028) |
| phi | share of "infectious" that is subclinical at t=0 | 0.42 (0.15, 0.70) |

**Horton 2023 (PNAS), repo `Inf-Dis`**: adds an Infection state upstream.
Infection→Cleared (infclr), Infection→Minimal (infmin),
Infection→Subclinical (infsub); Minimal→Recovered (minrec); otherwise as
Richards. Posterior (SI Table 6, main model):

| transition | median (95% CrI) |
|---|---|
| Infection→Cleared | 1.83 (0.94–3.30) |
| Infection→Minimal | 0.10 (0.04–0.23) |
| Infection→Subclinical | 0.04 (0.01–0.10) |
| Minimal→Cleared/Recovered | 0.18 (0.14–0.23) |
| Minimal→Subclinical | 0.24 (0.20–0.28) |
| Subclinical→Minimal | 1.58 (1.22–2.05) |
| Subclinical→Clinical | 0.72 (0.56–0.95) |
| Clinical→Subclinical | 0.58 (0.46–0.73) |
| Clinical→Death | 0.33 (0.28–0.38) |

Horton's SI also reports four development structures (A: no
Infection→Subclinical; B: the main model; C: adds an intermediate state
before Minimal; D: adds Infection→Clinical) with DIC 657.4 / 663.8 /
657.5 / 665.1. The full 150,000-draw posterior trace is in the repo
(`02_Cohort/param_traces.zip`), so I can validate my re-implementation
against it.

### The three variants I propose

| id | states | infection exits | regression exits | purpose |
|---|---|---|---|---|
| SP-full | Inf, Min, Sub, Clin, Clr, Rec, Dead | Inf→Clr, Inf→Min, Inf→Sub | Min→Rec | Horton main model (reference) |
| SP-noclear | same minus Clr | Inf→Min, Inf→Sub | Min→Rec | removes self-clearance; Inf is a permanent latent pool like the classic L |
| SP-nomin | Inf, Sub, Clin, Clr, Rec, Dead | Inf→Clr, Inf→Sub | Sub→Rec | removes the minimal state; the "two-state spectrum" (Richards without Min) |

Optional fourth if cheap: SP-nomin-noclear (both removed), which is the
closest spectrum analogue of structure A and makes the family boundary
explicit. Horton's structure D (direct Infection→Clinical) is another
candidate but their trace plots suggested it is not supported by the data.

## 3. What each family can be fit to (the comparison design)

This is the design question that decides whether question (a) "fit to
natural-history cohort data" is even well-posed, so I want to flag it
before coding.

The Richards likelihood is built on 54 observations from 22 cohorts
(5,942 people; I confirmed the count from the CSV) of transitions
**between Min, Sub and Clin**. A classic structure has one disease state
I, so it cannot say anything about Min⇄Sub⇄Clin transitions. Fitting the
classic family to the Richards data would require a mapping such as
I = Sub + Clin, which is an extra modelling assumption, not a structure
comparison.

Proposed resolution: build **one likelihood with three blocks** and report
model comparison on the blocks each family can address.

| block | data | classic family | spectrum family |
|---|---|---|---|
| (i) infection→disease by time since infection | Horton SI Table 3 adjusted converter cohorts: Daniels 1944 (248 converters), Madsen 1942 (208), NTI Bangalore 1961–68 (674 + 451 + 413); plus the Menzies Fig. 5 empirical curves (Ferebee 1970 placebo arms, Sutherland 1968) from Dataverse | yes (I = any disease, or I = infectious disease, as the cohort's case definition dictates) | yes (Min for Daniels/Madsen; Sub+Clin for NTI, exactly as Horton did) |
| (ii) untreated disease duration and case fatality | Tiemersma 2011 Table 3: 5- and 10-year survival for 16 smear-positive and 5 smear-negative cohorts (n from 185 to 11,797); Richards' duration target (N(2.0, 0.5) y for infectious disease) | yes, via the shared exit block | yes |
| (iii) Min⇄Sub⇄Clin transitions | Richards `included_data.csv`, 54 rows, 22 cohorts | no | yes (reuse Richards' binomial observation model and sub-structure initialisation verbatim) |

Deliverable for (a): LOO/WAIC on blocks (i)+(ii) for all 15 structures
(12 classic + 3 spectrum), and on (iii) for the spectrum variants only;
pointwise log-likelihoods stored so other subsets can be compared later.
Identifiability: posterior pairwise correlations, prior-vs-posterior
contraction ratios, and profile likelihoods for parameters with
contraction below a threshold.

Note on the Horton data adjustments: Horton reduce reported cases by 25%
and redistribute onset times with a truncated Cauchy delay (their
`InfDis_DataPrep.R`). I will take their *adjusted* table (SI Table 3)
as the data and not re-run the adjustment, and will state this.

## 4. Transmission embedding, GBD targets, hold-outs, ACT3

**Model.** Each natural-history structure × HIV stratum (HIV−, HIV+ no
ART, HIV+ on ART) × treatment/recovered compartments. λ = β·Σ(relative
infectiousness_i · I_i)/N with relative infectiousness 1 for Clin, a fitted
or prior-informed value for Sub (the epidemery pipeline estimates this;
I will reproduce its summary from the repo), 0 for Min/latent. Case
detection: Clin→treatment at a country-year rate; Sub→treatment only via
active screening. HIV: progression multipliers by stratum taken as
literature priors from the Dataverse review data (strata HIV+, AIDS,
HIV+ (ARV)). Demography: births and background mortality from GBD
population.

**Targets (fitting, never validation):** GBD 2023 via the IHME MCP,
1990–2023, for Viet Nam, South Africa, India, Kenya, Netherlands:
TB incidence, prevalence, deaths (all-ages and age-standardised rates,
counts, with UI); Latent TB infection prevalence; the HIV/AIDS–TB split
(causes 948/949/950 vs 934/946/947); the DS/MDR/XDR split; HIV/AIDS
prevalence as the HIV covariate. All of these returned data in test calls
today. Age-specific pulls also work (`age_group='by_age'`, 25 GBD age
bands), but see §7 on whether v1 is age-structured.

**Fitting approach for v1:** natural-history parameters fixed at draws
from the cohort posterior (§3); country parameters (β, case-detection
trajectory, HIV progression multipliers, reinfection protection) fitted by
weighted least squares to the GBD targets with GBD UIs as weights; 
uncertainty by repeating over posterior draws. A full joint Bayesian ODE
fit for 15 structures × 5 countries is the v2 option if you want it.

**Hold-out 1, subclinical fraction.** The epidemery
`prevalence_survey_data.csv` has 15 surveys with subclinical/clinical and
smear counts. Of the proposed panel only **Viet Nam** (2007, 2018) is in
it. South Africa 2018, Kenya 2016 and India 2019–21 surveys exist but
their symptom breakdowns are not in any file I have (**MISSING**, see §6).
Netherlands has no survey. Also usable for Viet Nam: the ACT3 secondary
analysis (PMC12332464) reports 153/236 (64.8%) of screen-detected cases
asymptomatic before diagnosis.

**Hold-out 2, ACT3.** Annual Xpert screening of adults ≥15 in 60
intervention sub-communes, Ca Mau, 2014–2017, 60 control sub-communes.
What I have:

| year | intervention (per 100,000) | control | source |
|---|---|---|---|
| 1 (2014) | 389 | – | scoping review PMC13268051 text; matches the bar chart in Marks' TB-MAC 2019 slides |
| 2 (2015) | 308 | – | same |
| 3 (2016) | 176 | – | same |
| 4 (2017) | 126 (53/42,150 tested) | 226 (94/41,680) | NEJM abstract via Europe PMC; PR 0.56 (0.40–0.78) |

Per-round numerators and denominators for years 1–3, and per-round
participation, are **MISSING** (NEJM full text and supplement return 403;
ResearchGate and medRxiv mirrors also blocked). The scoping review quotes
both "80% screened each year" and "an average of 21,849 adults (42%)
produced sputum each year", which are not the same thing; I will model
screening coverage as the sputum-producing fraction and treat it as a
sensitivity parameter until the paper's Table can be obtained. The
simulation will apply screening to bacteriologically positive
compartments (Sub + Clin in spectrum models; I in classic models, which
is itself a finding: classic models cannot represent the asymptomatic
yield) with Xpert sensitivity as a parameter.

## 5. Software design and milestones

- `uv` project in `tb_model_structures/` with pinned `uv.lock`.
  Verified today: `uv pip install numpy scipy duckdb` works from PyPI;
  Quarto is not preinstalled but `quarto-cli` 1.10.19 installs from PyPI
  through uv, so it goes in as a dev dependency; `pdftotext` is available.
- Declarative structures: a `Structure` is a list of compartments with
  flags (infectious, relative infectiousness, symptomatic, bacteriologically
  positive, dead, recovered) and a list of transitions
  `(from, to, rate_expr)` where `rate_expr` is a small expression over the
  parameter dict and λ. Chains (B, J, K, L) are generated by a helper from
  `n` and a hazard function. Adding a structure = adding one dict in a
  registry module; the README will show the 10-line example.
- Cohort mode: λ = 0 makes the system a linear ODE, so transition
  probabilities come from `expm(Q·t)` (exact, fast, differentiable in JAX).
  Likelihood: binomial counts per cohort row exactly as Richards/Horton
  (time-to-event rows use cumulative probability from a known start state;
  cross-sectional rows use state occupancy).
- Bayesian calibration with numpyro (NUTS; expm via `jax.scipy.linalg`),
  pointwise log-likelihoods saved for arviz LOO/WAIC.
  Acceptance test: reproduce Richards' and Horton's posterior medians from
  their own data to within MCMC error (their traces are in the repos).
- Transmission mode: `scipy.integrate.solve_ivp` on the same transition
  list with λ(t) and time-varying country inputs.
- DuckDB: tables `inputs_*` (every raw file with provenance columns),
  `posterior_*` (draws), `fit_metrics`, `sim_*` (trajectories, screening
  scenarios).
- Quarto document `report.qmd` renders tables/figures from DuckDB.

Milestones, each a commit: M1 this plan → M2 structure registry + ODE
engine + unit tests (mass conservation; A≡E(a=0) and C≡G(a=0) equivalence;
expm vs solve_ivp agreement) → M3 cohort likelihood + calibration +
reproduction of the two published posteriors → M4 transmission model +
GBD fits for 5 countries → M5 hold-outs + ACT3 simulation → M6 Quarto
report, DuckDB, README on adding a structure.

## 6. Data inventory

Obtained (cached in this session's scratch space; nothing copied into the
repo yet pending the licensing note below):

| source | what | how obtained | license |
|---|---|---|---|
| Richards repo `alexandrasrichards/natural_history_tb_disease` @ cf0ed16 | `included_data.csv` (54 rows, 22 cohorts, 5,942 people), LibBi model `results_model.bi`, posterior traces and param table | git clone | **no LICENSE file** |
| Horton repo `ERC-TBornotTB/Inf-Dis` @ 8566eea | `included_studies_shrunk1.csv` (byte-identical to Richards'), `InfDis_Model.bi`, four alternative-structure models, `param_traces.zip` (150,000 draws × 10 params), `InfDis_DataPrep.R` with the raw converter-cohort counts | git clone | **no LICENSE file** |
| Horton SI appendix `pnas.2221186120.sapp.pdf` | Tables S1 (exclusions), S3 (raw and adjusted converter cohorts), S4 (incidence), S6 (posteriors under NTI sensitivity) | Europe PMC `supplementaryFiles` zip (PMC and PNAS direct links are reCAPTCHA/403) | CC BY 4.0 |
| epidemery repo `epidemery/subclinical_transmission` @ 6d000e5 | `prevalence_survey_data.csv` (15 surveys), `duration_data.csv`, household-contact tables (Viet Nam, Philippines, Bangladesh, ACT3), Stan model | git clone | MIT |
| Menzies 2018 Figure 1 | structure diagram (transcribed in §1) | PMC CDN image | article is NIHMS author manuscript |
| Menzies replication data, Dataverse doi:10.7910/DVN/STRA2P | `LTBI Review Data_Clean` (680 rows × 66 cols: structure, stratum, a..f, d1..d10, x1..x3 per model), citation table, `Ferebee 1970` and `Sutherland 1968` empirical cumulative-incidence tables, reactivation model results, R scripts | Dataverse API | CC0 |
| Tiemersma 2011 | Tables 1–4 parsed from the PLOS XML (Table 2 study list, Table 3 survival rates, Table 4 WHO CFRs) | PLOS manuscript XML | CC BY |
| ACT3 | year-4 counts (abstract), years 1–3 prevalence points (secondary), asymptomatic fraction among screen-detected cases (PMC12332464) | Europe PMC, PMC | – |
| GBD 2023 | TB and sub-causes, LTBI prevalence, HIV/AIDS and HIV–TB causes, by year, all-ages and by age; 27 covariates catalogued (none for HIV or ART) | IHME MCP; quota 50,000 rows per 24 h, ~130 used | IHME |

Missing or unverifiable:

| item | status | consequence / stub |
|---|---|---|
| Menzies supplementary webappendix PDF | PMC download blocked (reCAPTCHA); Europe PMC has no full text for this NIHMS manuscript | not needed: Figure 1 and the Dataverse carry the structure and parameter data |
| NEJM ACT3 full text and supplement (per-round numerators, denominators, participation) | 403 everywhere I tried | use secondary prevalence points; coverage as sensitivity parameter; stub table flagged `source=secondary` |
| Prevalence-survey symptom breakdown for South Africa 2018, Kenya 2016, India 2019–21 | not in any obtained file; would need the survey reports or papers | hold-out 1 covers Viet Nam only unless you approve fetching those reports or swapping panel countries |
| GBD ART coverage covariate | not served by the MCP covariate catalogue | treat ART coverage as a country parameter with a wide prior, or approve an external source (UNAIDS) |
| Relative infectiousness of subclinical TB (number) | computable from the epidemery pipeline, but I have not run it and will not quote a value until I do | parameter with prior to be derived in M4 |
| Licenses for Richards and Horton repos | none stated | I will not commit copies of their files; a pinned-commit fetch script will populate `data/raw/` locally, and DuckDB will be built from it. Dataverse (CC0) and epidemery (MIT) files can be vendored |

## 7. Decisions (resolved 2026-10-12)

| # | decision | resolution |
|---|---|---|
| 1 | Shared disease-exit block for classic structures | **Yes.** One common I→self-cure, I→death block, informed by Tiemersma 2011 Table 3 |
| 2 | Likelihood blocks | **Blocks**, as in §3. A standalone HTML explainer (`docs/likelihood_blocks.html`) sets out the reasoning |
| 3 | Country panel | **Keep** Viet Nam, South Africa, India, Kenya, Netherlands. Fetch the South Africa 2018, Kenya 2016 and India 2019–21 survey symptom data; if any cannot be fetched, report it so the user can supply it |
| 4 | Age structure | **None in v1** |
| 5 | ART coverage | **Try UNAIDS** (directly or via the World Bank mirror of UNAIDS estimates). The data layer keeps a `source` column so a GBD ART covariate pulled on the IHME cluster (vivarium `gbd_mapping`) can replace it |
| 6 | MDR | **Proportion split in v1**, but strain is implemented as a generic stratum (like HIV) with per-stratum multipliers and an optional treatment-amplification transition, so a two-strain model is a config change later |
| 7 | Unlicensed repos | **Fetch script** at pinned commits into `data/raw/` (git-ignored); CC0 and MIT files vendored |

### Original questions (kept for the record)

#### Questions as originally posed

1. **Shared disease-exit block for the classic structures** (I→self-cure,
   I→death, informed by Tiemersma Table 3): acceptable, or do you want the
   12 structures left exactly as catalogued and compared on cumulative
   incidence only?
2. **Likelihood blocks as in §3**: cross-family comparison on converter
   cohorts + duration, spectrum-only comparison on Min⇄Sub⇄Clin
   transitions. Alternatively I can map classic I = Sub+Clin and force all
   15 through the full Richards likelihood, but I would call that a weaker
   comparison.
3. **Country panel vs survey availability**: keep Viet Nam, South Africa,
   India, Kenya, Netherlands (I then need to fetch three survey reports
   for hold-out 1), or add Philippines and Cambodia, whose survey symptom
   data are already in hand?
4. **Age structure**: v1 fits all-ages and age-standardised GBD rates with
   no age structure; age-structured v2 if the comparison warrants it.
5. **ART coverage**: free parameter, or approve an external source?
6. **MDR**: a post-hoc proportion split of incident TB (v1), rather than a
   second strain?
7. **Fetch script for unlicensed repos** (pinned commits, local cache)
   rather than vendoring their CSVs: OK?

If you reply "go" with no changes I will proceed with: yes to 1, 2, 4, 6,
7; panel as you listed it with survey reports fetched where I can; ART as
a free parameter.
