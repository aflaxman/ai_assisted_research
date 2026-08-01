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

Emergence is driven by a **time-varying transmission rate** `β(t) = β₀ + r·t`, tuned so
R₀(t) crosses 1 at a known time. The bifurcation parameter is R₀(t); the analytic
crossing time is the reference point against which we measure changepoint delay/advance.

> **⚠️ Verify parameters against the PDF + S1 supplement before trusting results.**
> The values below were extracted via an automated read of the HTML and must be
> confirmed. Treat them as a starting scaffold, not ground truth.

| Param | Meaning | Provisional value |
|-------|---------|-------------------|
| Λ | recruitment | 20 /day |
| μ | natural death | 0.01 /day |
| ω | immunity loss | 0.005 /day |
| σ | W→I progression | 1/3 /day |
| γ_W, γ_I | recovery (W, I) | 1/7 /day |
| α_S,α_W,α_I,α_R | post generation | 0.05, 0.1, 0.1, 0.05 |
| β_S,β_W,β_I,β_R | post engagement | 0.025 (0.05 for I) |
| δ_M | post decay | 0.05 /day |
| β₀ | initial transmission | 3.5e-4 /day |
| r | transmission ramp | ~2.4e-5 /day² (R₀=1 near t≈14.5) |

### Stochastic variants (Euler–Maruyama, dt = 0.005)

- **Additive noise:** `dXᵢ = fᵢ dt + σ_add dWᵢ`, with σ_add ∈ {0.1, 1, 2, 4}.
- **Multiplicative noise:** `dXᵢ = fᵢ dt + σ_mult·Xᵢ dWᵢ`, with σ_mult ∈ {0.01, 0.05, 0.1, 0.2}.
- **Demographic stochasticity:** noise amplitude set by the reaction rates (√rate per
  process); no free intensity knob.
- Negative states are rejected (revert to previous step). **300 realizations** per scenario.

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
