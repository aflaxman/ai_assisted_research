# Prevention front-end: does vitamin D come out cost-saving?

This is the step where the paper's actual result gets reproduced. It takes the
reconstructed N=4176 prediabetes cohort, moves people through
prediabetes → diabetes → death year by year, applies the vitamin-D effect, tallies
discounted costs and QALYs, and computes the ICER and net monetary benefit — then
checks them against Briody Table 3.

```
uv run python onset_model.py          # needs ../nhanes_cohort/outputs/cohort_canonical_n4176.csv
```

## TL;DR — it reproduces the paper's headline

| Lifetime, per person | This model | Briody 2026 |
|---|---|---|
| Incremental cost | **−$3,318** | −$3,208 |
| Incremental QALY | +0.149 | +0.120 |
| ICER ($/QALY) | **−$22,270** | −$26,134 |
| NMB @ $100k/QALY | $18,218 | $15,483 |
| **Conclusion** | **cost-saving (dominant)** | cost-saving (dominant) |

At 10 years: incremental cost −$500 (paper −$401), QALY +0.014 (+0.010). The economic
conclusion — vitamin D lowers cost *and* raises QALYs, so it dominates no-supplement —
reproduces robustly. Two outputs stay off (life-years gained, and the size of the
lifetime incidence reduction); both are honest limitations of the reduced form, explained
at the end.

## Why a new model at all

The public CDC/RTI engine (see the parent [`README.md`](../README.md)) ships the
*diabetes-complications* module but **not** the *prediabetes→diabetes onset* front-end —
and onset is exactly what vitamin D acts on. So the front-end has to be built. Rather
than reconstruct the proprietary 17-equation engine, this is a transparent **3-state
microsimulation** that captures the paper's mechanism directly.

## How it works

**States and clock.** Each of the 4176 people starts prediabetic at their real NHANES
age with their real sex and BMI. The model steps one year at a time to death or age 100.
Each year it (1) accrues that year's cost and QALY for the person's current state,
(2) lets prediabetics develop diabetes, and (3) applies mortality; survivors age a year.

**Onset, with a susceptible fraction.** A single constant onset hazard cannot fit the
paper's incidence curve — 20% of the cohort has diabetes by 10 years but only 32% ever
do, so the at-risk pool must deplete. This matches reality: many people with prediabetes
never progress. So the model makes a fraction `f` of the cohort *ever-susceptible* (they
face an annual onset hazard) and the rest never progress. Calibration lands `f ≈ 0.36`
with a susceptible hazard of ≈ 0.079/yr — which reproduces **both** the 10-year (20.2 vs
20.14) and lifetime (32.3 vs 32.35) incidence.

**The vitamin-D effect.** For people eligible at baseline (BMI ≥ 25, per the paper), the
onset hazard is multiplied by `(1 − adherence × RRR) = (1 − 0.85 × 0.15) = 0.872`.
Supplementation costs $60/yr while a person is still prediabetic, and stops at diabetes
onset. Non-susceptible eligible people still take (and pay for) the supplement for life
with no benefit — as they would in a real program. Both arms share the same random
draws (*common random numbers*), so the incremental estimate has very low Monte-Carlo
noise.

**Costs and QALYs.** Prediabetes costs $3,244/yr; diabetes costs $11,322/yr of treatment
**plus a complication cost that grows with diabetes duration**, and its utility starts at
0.935 and **falls with duration** as complications accumulate. Representing the 17
complications this way — in aggregate, ramping with time-in-diabetes — is the key
modelling shortcut. It encodes the paper's actual mechanism ("the longer you have
diabetes the more complications cost you, so delaying onset saves money and quality of
life") without the individual complication equations. Everything is discounted 3%/yr and
reported per person, survey-weighted to be nationally representative.

## Calibration — and why it's an honest test

The model has five free parameters. All five are tuned to the **control-arm** rows of
Table 3 — i.e. to *descriptive* facts about the no-supplement world, **not** to the
intervention effect:

| Parameter | Tuned to hit (control arm) | Value |
|---|---|---|
| background-mortality scale | remaining life-years 36.45 | 0.63 |
| susceptible fraction `f` | lifetime incidence 32.35/100 | 0.363 |
| onset hazard | 10-year incidence 20.14/100 | 0.079/yr |
| diabetes complication cost ramp | lifetime cost $163,210 | $1,023/yr·dur |
| diabetes complication disutility ramp | lifetime QALYs 16.35 | 0.050/yr·dur |

Diabetes excess mortality is fixed at HR 1.8 (a central literature value), not calibrated.

Because nothing was tuned to the *intervention* rows, the incremental cost, QALY,
life-years, ICER, and NMB are **emergent** — they fall out of applying the 15% hazard
reduction to a calibrated control world. That they land close to the paper (cost −$3,318
vs −$3,208; ICER −$22k vs −$26k) is the real test, and it passes.

## What it does *not* reproduce, and why

Two rows are honestly off:

- **Incremental life-years: 0.043 vs 0.270.** The paper's supplement buys a lot of
  survival; mine buys little. In this reduced form, susceptibles carry a fairly high
  hazard, so a 13% hazard reduction *delays* onset only briefly, and a brief delay against
  an HR-1.8 mortality gap yields few life-years. The paper's fuller model spreads onset
  out and adds a **reversion-to-normoglycemia** channel (vitamin D also pushes people back
  to normal glucose, rate ratio 1.30) that this model omits — both give more room for a
  survival benefit.
- **Lifetime incidence reduction: 2.2% vs 8%.** Same root cause — my structure mostly
  *delays* onset among fast-converting susceptibles rather than *preventing* it, so the
  lifetime case count barely moves even though the 10-year count does (5.8% reduction).

Notably, the **QALY gain still matches** (it comes through the avoided-complication
channel rather than the survival channel), and the QALY gain is what drives the ICER and
NMB — so the economic conclusion is robust even though its decomposition differs from the
paper's.

Two modelling flags carried over from the feasibility review, worth keeping in view:
- **ITT × adherence risks double-counting.** The 15% RRR is an intention-to-treat estimate
  that already embeds trial adherence; multiplying by a further 0.85 (as the paper does,
  and as this model mirrors for comparability) may penalise adherence twice.
- **Reversion to normoglycemia is omitted** — adding it would raise the modelled benefit.

## How to make it more faithful (next fidelity steps)

1. Add a **normoglycemia** state with prediabetes⇄normal transitions, and let vitamin D
   raise reversion (RR 1.30) — this should lift both the incidence reduction and the
   life-years gained toward the paper.
2. Replace the aggregate complication ramp with the **real CDC/RTI complication and
   mortality equations** (public in `t2d/`; see parent README) run on each person after
   onset — turning this into a genuine two-module (prevention + complications) model.
3. Add **probabilistic sensitivity analysis** (draw the 15% RRR from its CI, HR 0.85
   [0.75–0.96], plus cost/utility SEs from Table 2) to produce uncertainty intervals like
   the paper's, and run the paper's subgroup/scenario analyses.

## Files

- `onset_model.py` — the model, calibration, and Table 3 comparison.
- `outputs/cea_results.csv` — incremental cost, QALY, life-years, ICER, NMB (both horizons).
