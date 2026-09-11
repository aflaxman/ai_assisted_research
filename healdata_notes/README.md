# HEAL Data Platform: Data for an OUD/MOUD Simulation Model

Notes from a survey of [healdata.org](https://healdata.org/) (the NIH HEAL Data
Platform), conducted 2026-08-26, asking: **what data does the platform offer
that could parameterize, calibrate, or validate a simulation model of opioid
use disorder (OUD) and medications for OUD (MOUD) treatment?**

## TL;DR

- The HEAL Data Platform is a **Gen3-based discovery layer, not a data
  warehouse**. It catalogs 625 registered HEAL studies (via its public
  metadata API) and points to data in ~19 federated repositories (NAHDAP/ICPSR,
  NIDA Data Share, Vivli, NIMH Data Archive, JCOIN Data Commons, Harvard
  Dataverse, and others). You request data from those repositories, each with
  its own access process.
- **Most HEAL data is not yet released**: of 625 studies, 382 report data
  release "not started", 46 "started", 20 "finished". Plan around the released
  subset.
- The single most simulation-relevant asset is the **HEALing Communities Study
  (HCS)** — a cluster-randomized trial of community-level scale-up of MOUD,
  naloxone distribution, and safer prescribing in 67 communities (~10M people)
  across KY, MA, NY, OH, with archival at ICPSR. It is close to a natural
  benchmark/validation target for any community-intervention OUD model.
- HEAL already funds OUD simulation models — **RESPOND**, **RESCUE**, the
  **Data2Action Modeling and Economic Resource Center (MERC)**, and two system
  dynamics projects — worth reviewing before building a new model, both for
  parameter values and for structure comparison.
- The catalog is fully queryable programmatically:
  `https://healdata.org/mds/metadata?_guid_type=discovery_metadata&data=True&limit=2000`
  returns all study metadata as JSON (~25 MB). `query_heal_mds.py` in this
  directory reproduces the analysis below.

## How the platform works

The platform (run by University of Chicago's Center for Translational Data
Science on Gen3) provides:

1. **Discovery pages** at `https://healdata.org/portal/discovery/HDPxxxxx`
   (client-side rendered; the underlying JSON comes from the `/mds/` API).
2. **Study-level metadata** — description, research program, data availability,
   repository links, ClinicalTrials.gov IDs, NIH RePORTER abstracts.
3. **Variable-level metadata** — 172 of 625 studies have machine-readable data
   dictionaries (e.g., HCS exposes 1,653 variable names across five REDCap
   instruments). "HEAL Semantic Search" searches across these variables.
4. **Secure workspaces** for analyzing released data in the cloud.

Data access itself goes through the federated repository holding each dataset.
Repository representation among the 625 studies: NAHDAP (127), Vivli (43),
NIMH Data Archive (33), Harvard Dataverse (29), ICPSR (28), NIDA Data
Share (24), JCOIN Data Commons (19), plus Figshare, OSF, Dryad, dbGaP, etc.

## Mapping datasets to simulation model components

Thinking of a state-transition or agent-based model with states like
*no OUD → OUD (untreated) → MOUD (by medication) → discontinued/relapse →
overdose (fatal/non-fatal) → recovery*, plus justice-system and community
intervention layers:

### 1. Community-level intervention effects (calibration/validation target)

**HEALing Communities Study** — NCT04111939, the flagship. Communities
randomized to a package of overdose education + naloxone distribution, MOUD
expansion, and safer prescribing. Outcomes include overdose deaths, MOUD
uptake, and naloxone distribution at the community level. The published main
result (a smaller-than-projected effect on overdose deaths) is itself an
important validation story for OUD models that predicted larger effects.

| HDP ID | Study |
|---|---|
| HDP00360 | Data Coordinating Center (RTI) — 1,653 variables in data dictionaries; archival at ICPSR |
| HDP00036 | MassHEAL (Massachusetts sites) |
| HDP00074 | OHiO (Ohio sites) |
| HDP00084 | Kentucky CAN HEAL |
| HDP00150 | CHASE (New York sites) |

The DCC metadata says the shared file is a de-identified common-data-model
extract with masked geography. Archival was expected early-to-mid 2025, so it
should be checked at ICPSR/NAHDAP now.

### 2. MOUD initiation, retention, and comparative effectiveness

The densest cluster of studies. Highlights with data available or scheduled:

| HDP ID | Study | Notes |
|---|---|---|
| HDP01290 | **ED-INNOVATION** (CTN-0099, NCT04225598) | ED-initiated extended-release vs. sublingual buprenorphine; availability "all", release listed for late 2025 at NIDA Data Share |
| HDP01296 | Standard vs. high-dose ED buprenorphine induction | NIDA Data Share |
| HDP01317 | **MOMs** (NCT03918850) | XR vs. SL buprenorphine in pregnancy; NIDA Data Share |
| HDP01319 | SWIFT | Improving XR-naltrexone initiation (the induction hurdle) |
| HDP01310 | VA buprenorphine discontinuation study | Release finished — reasons for discontinuation |
| HDP00965 / HDP01187 | Methadone take-home flexibility | Effect of COVID-era regulatory relaxation on retention |
| HDP01337 | TREETOP | Retention/engagement for OUD + pain; 354 variables documented |
| HDP01430 | ReTAIN | Patient-reported outcomes to individualize treatment and improve retention |
| HDP00332 et al. | Contingency management + behavioral economics for buprenorphine | Several BRIM-program trials |
| HDP00104 | **STAR-COD** (NCT05138614, n≈1000) | MOUD + co-occurring mental illness; NIMH Data Archive #4128; release started, 4,485 variables documented |

These are the natural sources for treatment-arm transition rates: initiation
success by setting and formulation, retention curves by medication, and
discontinuation reasons.

### 3. Justice-involved populations (JCOIN)

Justice involvement is a key stratum in most OUD models (high post-release
overdose mortality, low MOUD access). JCOIN has 13+ studies, its own data
commons ([jcoin.datacommons.io](https://jcoin.datacommons.io)) with common
data elements across hubs, and deposits at NAHDAP.

| HDP ID | Study | Notes |
|---|---|---|
| HDP00091 | **EXIT-CJS** (NCT04219540, n=301) | XR-buprenorphine vs. XR-naltrexone in jails/prisons, 5 sites |
| HDP00362 | ROMI (NCT04925427, n=600) | Hub-and-spoke MOUD + naloxone + syringe services for justice-involved, urban/rural Illinois |
| HDP00183 | TCU Clinical Research Center | Implementation strategies at reentry |
| HDP01577 | Methadone dispensing in carceral facilities | |
| HDP01587–89 | JCOIN phase II (jail ECHO, drug courts, prisons) | |

Also from JCOIN: the **Opioid Environment Policy Scan (OEPS)** — an open
(freely downloadable) warehouse of county/zip/tract-level covariates on MOUD
access, overdose risk environment, and policy. Immediately usable for spatial
heterogeneity in a model, no DUA needed.

### 4. Overdose surveillance and epi parameters

The "Translating Data 2 Action" and "Leveraging Existing and Real-Time Data"
programs fund surveillance-style work useful for overdose incidence and fatal
fractions in the fentanyl era:

- HDP00975 — RADOR-KY: rapid actionable overdose data in Kentucky
- HDP00927 — Predicting fatal and non-fatal overdose in Los Angeles County
- HDP00887 / HDP01487 — FORTRESS: fatal overdose review team data systems
- HDP01001 — Near-real-time suspected overdose deaths from death investigations
- HDP01005 — O-SUDDEn opioid/SUD data enclave

The nine "Harm Reduction Approaches" studies (HDP01052–HDP01073) cover
naloxone distribution channels, mail-delivered harm reduction, and overdose
responder networks — inputs for naloxone-coverage scenario parameters.

### 5. Existing HEAL-funded simulation models (review before building)

| HDP ID | Project | Notes |
|---|---|---|
| HDP00516 | **RESPOND** (Boston Medical Center/BU) | A published state-transition/microsimulation OUD model (Massachusetts). Its HEAL deposit at Harvard Dataverse (doi:10.7910/DVN/F66ZIW) is the ICD-9/10 code lists for identifying OUD, overdose, and comorbidities in claims — directly reusable for case definitions. Release finished. |
| HDP01021 | **HEAL Data2Action MERC** | Modeling and economic resource center; maintains "a dynamic simulation model of OUD" and consults for HEAL projects — a potential collaboration/comparison point. |
| HDP01343 | **RESCUE** | Simulation-based planning for naloxone/harm-reduction scale-up with equity focus; code/data release planned 2027 at ICPSR/GitHub/Zenodo. |
| HDP00907 / HDP01573 | System dynamics modeling for real-time connections to care | |
| HDP01624 | OUD care trajectories via population-based data linkage | Multi-state transition estimates from linked administrative data — very close to what a microsimulation needs for care-cascade transitions. |

### 6. Special populations and adjacent parameters

- **Pregnancy/NOWS**: ACT NOW program (HDP00002 HELP for NOWS and siblings,
  NICHD DASH), MOMs (above), XR-naltrexone in pregnancy (HDP00210).
- **Prevention/incidence side**: 16 "Preventing OUD" studies (youth
  prevention trials, ED-based prevention) — weaker fit for incidence rates;
  national surveys (NSDUH etc., outside HEAL) remain the better source.
- **OUD + chronic pain comorbidity**: IMPOWR network (HDP01045, HDP01046,
  HDP01048), relevant if the model includes a pain–opioid pathway.

## Practical access notes

1. **Programmatic catalog**: the Gen3 metadata service requires no auth:
   `curl "https://healdata.org/mds/metadata?_guid_type=discovery_metadata&data=True&limit=2000"`.
   Each record has `gen3_discovery` (study metadata, tags, repository links,
   data-availability fields) and `variable_level_metadata` (data dictionary
   references). `query_heal_mds.py` here downloads and summarizes it.
2. **Requesting data** goes through each repository: NAHDAP/ICPSR (public +
   restricted-use files, standard ICPSR DUA), NIDA Data Share (open
   de-identified trial data), NIMH Data Archive (NDA access request), Vivli
   (proposal-based), JCOIN Data Commons (login, JCOIN data-analysis proposal).
3. **Timing caveat**: `data_release_status` in the catalog is self-reported
   and often stale (HCS says "expected February 2025"). Verify at the
   repository before planning on any dataset.
4. **Beyond HEAL**: NIDA Data Share also hosts the older CTN comparative
   effectiveness trials (e.g., X:BOT, NCT02032433, XR-naltrexone vs.
   buprenorphine-naloxone) that most published OUD models use for retention
   and relapse parameters — same access path, not HEAL-badged.

## Suggested next steps

1. Check ICPSR/NAHDAP for the HCS archive and request it — community-level
   MOUD uptake, naloxone distribution, and overdose mortality under a known
   intervention design is the best available calibration/validation target.
2. Pull the OEPS warehouse (open download) for county-level MOUD access
   covariates.
3. Review RESPOND and MERC model publications for structure and parameter
   tables; reuse the RESPOND ICD code lists from Dataverse.
4. Request 2–3 retention datasets (ED-INNOVATION, VA discontinuation,
   STAR-COD) to estimate medication-specific retention/discontinuation
   hazards.
