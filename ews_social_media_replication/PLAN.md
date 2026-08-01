# Replication Plan: Early-Warning Signals for Infectious Diseases with a Social-Media Compartment

> **Note on the name.** The Google Scholar alert labeled this paper "LINK3D," but that
> is a mismatch. The actual article is *"Early-warning signals for infectious diseases
> with a social-media compartment"* (Olajide, Lutscher & Smith, PLOS ONE, 28 Jul 2026,
> DOI [10.1371/journal.pone.0354091](https://doi.org/10.1371/journal.pone.0354091)).

## 1. What the paper is about

Can we see an epidemic coming before it arrives? The paper tests **early-warning
signals (EWSs)** — rising variance and rising lag-1 autocorrelation, the classic
"critical slowing down" fingerprints of a system approaching a tipping point — on a
disease model that is slowly pushed past its emergence threshold (R₀ crossing 1).

Two things make this paper distinctive:

1. **It adds a social-media compartment.** Beyond the usual epidemiological states, the
   model tracks the volume of social-media posts about the disease. The authors ask
   whether online chatter carries an earlier or stronger warning than case counts do.
2. **It takes delayed emergence seriously.** In stochastic models the observable
   transition often lags the theoretical bifurcation. Most EWS studies ignore this gap.
   This paper measures the lag with changepoint detection and asks whether EWSs still
   work when the transition is late.

### Headline findings to reproduce

- **Variance beats autocorrelation** as a warning signal across most noise conditions —
  the opposite of some earlier studies (O'Regan & Burton; Chakraborty et al.).
- **Social-media dynamics give no usable warning.** Variance and autocorrelation on the
  post-volume series perform no better than a coin flip (AUC ≈ 0.5) under every noise
  type and intensity.
- **Transition timing depends on the noise.** Additive and multiplicative noise tend to
  *delay* the observed changepoint past the bifurcation; demographic stochasticity tends
  to *advance* it. Unreported infections (W) turn before reported infections (I).
- **Warning quality degrades with noise.** Variance on reported infections reaches
  AUC ≈ 0.99 at low multiplicative noise but falls toward chance as noise rises.

### How it fits the literature

- **Extends critical-slowing-down theory** (Scheffer et al.; Dakos et al.) from ecology
  to epidemic emergence, and specifically to a novel social-media state variable.
- **Engages a live disagreement** over whether variance or autocorrelation is the better
  EWS, and lands on variance — attributing conflicting prior results to differences in
  model dimensionality and noise structure.
- **Builds on bifurcation-delay work** (O'Regan & Drake; O'Regan & Burton) by quantifying
  how far the observable transition trails the analytic threshold, and showing EWSs can
  still succeed despite that lag.
- **Departs from data-driven social-media epidemiology** (Google Trends / Twitter +
  machine learning) by asking the mechanistic question instead: if posts are generated
  by a coupled dynamical process, do *classical* EWSs read them? Answer: no.

## 2. The model (SWIRM)

Five coupled compartments: **S**usceptible, **W** (unreported infected), **I** (reported
infected), **R**ecovered, **M** (social-media posts). N = S + W + I + R.

```
dS/dt = Λ − β(t)·S·(W+I)/N − μS + ωR
dW/dt = β(t)·S·(W+I)/N − (γ_W + σ + μ)W
dI/dt = σW − (γ_I + μ)I
dR/dt = γ_W·W + γ_I·I − (ω + μ)R
dM/dt = Σ_x α_x·X − M·Σ_x β_x·X − δ_M·M       (x ∈ {S,W,I,R})
```

Only the **reported** class I appears in the force of infection `βSI` (mass-action, no
`/N`); each new infection is split fraction `f` into I and `1−f` into W. Emergence is
driven by a **time-varying transmission rate** `β(t) = β₀ + r·t`. The bifurcation
parameter is R₀(t); the analytic crossing time is the reference against which we measure
changepoint delay/advance.

**R₀ (next-generation, verified against the S1 appendix):**

```
R₀ = [ α(1−f)·βS̄ + f·βS̄·(γ_W+α+μ) ] / [ (γ_W+α+μ)(γ_I+μ) ],   S̄ = π/μ
```

> **✅ Parameters confirmed** against the full text (Table 1) and S1 appendix. Plugging
> Table 1 into R₀(t)=1 gives the bifurcation at **t = 14.50 days** — exactly the paper's
> value — which validates the whole model setup.

| Param | Meaning | Confirmed value | Source |
|-------|---------|-----------------|--------|
| π | birth rate | 14.28 /day | calc. for S̄=1000 |
| μ | natural death | 0.01428 /day | ref |
| f | fraction reporting | 0.8 | assumed |
| γ_W | recovery, unreported | 0.07 /day | ref |
| γ_I | recovery, reported | 0.024 /day | assumed |
| ϕ | waning immunity | 0.01 /day | ref |
| α | progression W→I | 0.033 /day | ref |
| ε | media engagement | 0.25 | calc. |
| δ | media posting | 0.5 | ref |
| μ̄ | media "fickle"/decay | 0.1428 | ref |
| β₀ | initial transmission | 5e-6 | ref |
| r | transmission ramp | 2.739e-6 | calc. (R₀=1 at t=14.5) |

Media compartment: `dM/dt = δ·Σ X + ε·M·Σ X − μ̄·M²` (quadratic decay, bilinear
engagement). Media is one-directionally coupled — it reads disease state but never feeds
back — so it never enters R₀ and cannot change the epidemic dynamics.

### Stochastic variants (Euler–Maruyama, dt = 0.005, 2900 pts = 14.5 days to BP)

- **Additive noise:** same intensity σ added to all five equations, `dXᵢ = fᵢ dt + σ dW`.
- **Multiplicative noise:** `dXᵢ = fᵢ dt + σ·Xᵢ dW` (same σ all compartments).
- **Demographic stochasticity:** chemical-Langevin form of the 13-process reaction table
  (S1 Table 3); intensity is set by the rates, **no free knob**.
- Negative states are rejected (revert to previous step). **300 realizations** per scenario.

> **⚠️ One remaining gap:** the *numeric* low/medium/high intensity values for additive and
> multiplicative noise live only in the figure captions (stripped from the machine-readable
> text). We pick defensible values, expose them as parameters, and flag that exact Table 2
> matching needs those captions. The qualitative findings are robust across reasonable σ.

### EWS pipeline

1. Gaussian-detrend each series, then compute **rolling variance** and **lag-1
   autocorrelation** over a window = 50% of the segment.
2. Analyze two segments: baseline→bifurcation and baseline→changepoint.
3. Summarize each series' trend with **Kendall-τ** (positive = warning).
4. Score detection with **ROC/AUC** across the 300-realization ensemble, per compartment
   (I, W, M) × noise type × intensity.
5. Detect the observed transition with **AMOC mean-changepoint** detection; compare its
   timing to the analytic bifurcation.

## 3. Notebook architecture — keep the substance in front

Goal: the notebook reads like the paper's argument, not like plumbing. All machinery
lives in small, tested modules; each notebook cell shows one idea, one figure, or one
number, wrapped in prose that says *what* we find, *how*, and *how it fits the literature*.

```
ews_social_media_replication/
├── PLAN.md                  ← this file
├── README.md                ← short blog-style write-up once results are in
├── pyproject.toml           ← uv-managed deps (numpy, scipy, pandas, ruptures,
│                              statsmodels, matplotlib, ptitprince[raincloud], tqdm)
├── swirm/
│   ├── model.py             ← parameters (dataclass) + deterministic RHS + R₀(t)
│   ├── simulate.py          ← Euler–Maruyama; additive/multiplicative/demographic noise
│   ├── ews.py               ← Gaussian detrend, rolling variance/AC1, Kendall-τ
│   ├── changepoint.py       ← AMOC mean changepoint (ruptures), delay-vs-bifurcation
│   ├── evaluate.py          ← ensemble runner → ROC/AUC tables
│   └── plots.py             ← raincloud, ROC, time-series-with-EWS panels
├── tests/                   ← pytest: R₀ crossing time, EWS on known series, AUC sanity
└── replicate_ews.ipynb      ← the narrative
```

**Design rules that keep the notebook clean:**

- *One import, no logic in the notebook.* Cells call `swirm.*` functions and plot. Any
  cell longer than ~15 lines is a smell — push it into a module.
- *Config is data.* A single `Params` dataclass and a `SCENARIOS` list drive everything,
  so a reader sees the experiment design at a glance.
- *Deterministic seeds.* Every ensemble takes an explicit RNG seed; results reproduce.
- *Cache the expensive ensemble.* 300 × several scenarios × ~1.16M steps is heavy — run
  once, save to `.parquet`/`.npz`, and let the notebook load results so re-runs are fast.
- *Each figure earns a paragraph.* Markdown above every plot states the claim it supports
  and cites the paper's corresponding figure/finding.

### Notebook narrative (section → paper element)

1. **The question & the literature** — critical slowing down, the variance-vs-AC debate,
   why social media might help. *(prose only)*
2. **The SWIRM model** — equations, then `deterministic_run()` showing emergence. Plot
   R₀(t) crossing 1.
3. **Adding noise** — one figure, three panels: additive / multiplicative / demographic
   sample paths.
4. **When does the epidemic *look* like it started?** — changepoint vs. bifurcation;
   reproduce the delay/advance result and W-before-I ordering.
5. **Reading the warning signals** — rolling variance & AC1 on I, W, M; Kendall-τ.
6. **Do the signals work?** — ROC/AUC tables and raincloud plots. Reproduce
   *variance > autocorrelation* and *social media ≈ chance*.
7. **What it means** — synthesize findings, connect back to the literature, note the
   simulation-only caveat and the authors' proposed EpiEstim validation on real outbreaks.

## 4. Milestones

1. **Confirm parameters** from the PDF and S1 supplement (fix the ⚠️ table).
2. **Deterministic core** (`model.py`) + test that R₀(t)=1 at the stated time.
3. **Stochastic integrator** (`simulate.py`) + the three noise types, with a smoke plot.
4. **EWS + changepoint** modules with unit tests on synthetic signals.
5. **Ensemble + AUC** (`evaluate.py`); cache results.
6. **Notebook narrative** stitched over cached results; match the paper's key numbers.
7. **README** write-up in the house blog style once figures agree.

## 5. Open questions to resolve from the full text

- Exact `r`, β₀, and the R₀=1 time (the HTML extraction is approximate).
- Which compartments receive noise (all five, or infected states only?).
- Demographic-noise construction (τ-leaping vs. Gaussian per-reaction amplitude).
- Gaussian-detrending bandwidth used by the R `earlywarnings` package, to match exactly.
- Definition of the "changepoint" segment endpoint feeding the second EWS window.
