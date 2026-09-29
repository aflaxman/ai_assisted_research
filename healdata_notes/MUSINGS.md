# Musings: Unique Simulation Opportunities in the HEAL Data Portfolio

Companion to [README.md](README.md). The question here is not "what parameters
can we scrape" but "what becomes possible that wasn't before." Nine ideas,
roughly ordered from validation science to infrastructure.

## 1. HCS as a held-out test set for OUD models

Every published OUD simulation model was calibrated to observational
surveillance and "validated" by reproducing trends it was more or less fit to.
The HEALing Communities Study breaks that circularity: 67 communities
randomized to a documented intervention package, with process data (which
evidence-based practices each community selected from the menu, reach counts)
and measured overdose mortality. Model-based projections motivating the study
anticipated reductions approaching 40%; the trial observed roughly 9%,
not statistically significant.

Opportunity: a retrospective model tournament. Give each model only baseline
data and the intervention documentation, ask it to predict the trial's effect,
then compare with the observed result. The gap diagnoses structural
assumptions — the lag from MOUD initiation to mortality benefit, fentanyl
supply shocks, COVID disruption, saturation of the reachable population.
A model that can retrodict HCS earns credibility no fit to surveillance
curves can confer.

## 2. Sixty-seven communities calibrate the variance, not just the mean

Stochastic simulations emit distributions of outcomes across simulated
communities, but almost no one has replicated real communities under a common
intervention to check that dispersion against. HCS provides between-community
heterogeneity in response, with covariates. Hierarchical Bayesian calibration
treating each community as a replicate could make an OUD model's uncertainty
intervals honest for the first time — a methods contribution independent of
the substantive one.

## 3. A fentanyl-era parameter refresh, maintained like software

RESPOND, SOURCE, and related models lean on pre-fentanyl or early-fentanyl
parameters; overdose case fatality, naloxone reversal probability, and
remission rates have shifted under fentanyl and xylazine. HEAL surveillance
projects (RADOR-KY, LA County fatal/non-fatal overdose prediction, FORTRESS)
are the re-estimation inputs. The larger idea: a versioned, provenance-tracked
library of OUD model parameters — a GBD-style shared input layer for opioid
models — which nobody currently maintains. The platform's machine-readable
data dictionaries make parameter-to-variable provenance traceable. MERC
(HDP01021) is positioned to host it; an outside prototype could force the
issue.

## 4. The jail-to-community seam as a first-class state machine

The weeks after release from incarceration carry an order-of-magnitude
overdose mortality spike, and the levers are about timing: initiate MOUD in
custody, at release, or after; which formulation; with what linkage support.
JCOIN supplies medication-specific effectiveness in carceral settings
(EXIT-CJS), linkage models (ROMI hub-and-spoke), and methadone dispensing in
jails. Combined with the linked-administrative-data care-trajectory study
(HDP01624), one could estimate continuous-time transition intensities across
the justice/health boundary — enabling the first microsimulation where the
criminal-legal system is a coupled dynamic subsystem rather than a static
"justice-involved" covariate.

## 5. Tract-level allocation as simulation-optimization (startable today)

OEPS offers open-download tract/county data on MOUD access and overdose risk
environment; RESCUE targets equitable naloxone allocation. A spatial synthetic
population married to those layers turns "where does the next mobile methadone
van or naloxone vending machine go?" into a simulation-embedded optimization
with equity constraints — a concrete decision product for health departments.
No DUA required, which makes it the natural first project.

## 6. Two timescales in one agent

Cascade models run on months; relapse happens on days. Several HEAL studies
collect ecological momentary assessment and digital phenotyping (craving,
sleep, stress) in people on MOUD. An agent-based model whose within-person
daily dynamics are fit to EMA data and whose population dynamics are
calibrated to surveillance would bridge scales nobody has credibly
connected — and would let fast-timescale interventions (contingency
management, just-in-time adaptive apps) be evaluated in terms of
slow-timescale population payoff.

## 7. Natural experiments as mechanism tests

COVID-era methadone take-home flexibility (HDP00965/HDP01187), Oregon's
Measure 110 expansion-plus-decriminalization (HDP01110/HDP01483), and
hydrocodone rescheduling (HDP01022) each yield quasi-experimental effect
estimates that a mechanistic model should reproduce without being fit to
them. A model that passes several such out-of-sample tests occupies a
different epistemic class than one that passes none; the HEAL portfolio
conveniently supplies the test battery.

## 8. A CISNET moment for opioids

HEAL simultaneously funds multiple models (RESPOND, RESCUE, MERC's dynamic
model, system-dynamics projects), a shared metadata platform, and a natural
benchmark trial. These are the preconditions that made CISNET work for cancer
screening and the Scenario Modeling Hub for COVID. A coordinated multi-model
comparison — same scenarios, same targets, structured adjudication of
disagreements — would do more for policy credibility than any single model.

## 9. AI-assisted evidence synthesis (this repo's theme)

The 172 machine-readable data dictionaries and the semantic-search layer
invite automation of the worst part of simulation modeling: evidence
synthesis and variable harmonization. An LLM-agent pipeline that reads
dictionaries, maps fields onto a common schema (JCOIN's common data elements
are a ready-made target), drafts crosswalks for human review, and emits
parameter estimates with provenance would compress months of tedium into
days. It feeds naturally into a Vivarium implementation whose evidence
pipeline regenerates when upstream repositories release new data — and it is
a well-scoped AI-assisted-research experiment (and blog post) in its own
right.

## Ranking by leverage-per-effort

- **Quick win:** #5 — OEPS spatial allocation (open data, immediate start).
- **Scientifically deepest:** #1/#2 — HCS as test set and variance calibrator.
- **Compounds:** #9 — automated harmonization makes every other idea cheaper.
