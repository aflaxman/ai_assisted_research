# Frimpong & Bauch (2026) with camdl: pandemic waves from coupled behaviour-disease dynamics

Replicates: Sefah Frimpong & Chris T. Bauch, *Pandemic waves as the
outcome of coupled behaviour and disease dynamics: a mathematical
modelling study*, J. Theor. Biol. 2026
([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0022519326001955);
medRxiv [2026.02.05.26345658](https://doi.org/10.64898/2026.02.05.26345658);
[authors' code + data](https://github.com/SefahF/Pandemic-waves-as-the-outcome-of-coupled-social-and-disease-dynamics)).

The paper fits two deterministic ODE models to the Spring-2020
SARS-CoV-2 wave in 13 European countries and tests how well each
retrodicts the Fall-2020 second wave:

- **SIR** ("disease model"): seasonal-transmission SIR fitted to daily
  reported case incidence only.
- **SIRx** ("coupled model"): the same SIR plus an imitation-dynamics
  behaviour state x (the proportion supporting NPIs) that suppresses
  transmission by (1 − εx) and grows or decays with the payoff
  difference between mitigating (cost c) and not (perceived reported
  prevalence, decaying as e^(−λt)); fitted jointly to case incidence
  and the Oxford stringency index (OSI/100).

Its headline: the SIR model fits the first wave slightly better, but
badly overshoots the second wave; the coupled model predicts the second
wave's magnitude far better and is preferred by AICc in every country.

## RESULTS_PLACEHOLDER

## Model translation notes (paper → camdl)

- **Code beats supplement where they disagree.** The supplement writes
  the seasonal forcing as `cos(t − φ)` (period 2π days) and gives
  φ ∈ [−90°, 90°], λ ∈ [0, 0.03], c ∈ [0, 0.1], κ ≤ 10,000/day. The
  authors' scripts use `cos(2π(t − φ)/365)` with φ ∈ [−135, −45] days,
  λ ∈ [0, 0.2], c ∈ [0, 0.01], κ ∈ [100, 5000] — and the behaviour
  payoff uses **reported** prevalence I/η₀, not I. This replication
  follows the code.
- **The SIR variant carries a hidden 0.8.** The paper's SIR script fixes
  ε = 1, x = 0.2, multiplying transmission by a constant (1 − 0.2) = 0.8
  — a pure rescale of β, reproduced literally so fitted β is comparable.
  The paper's R0 = β/γ convention omits it.
- **Counts, not proportions.** camdl compartments are populations, so
  S, I, R are counts with N = pop and the data TSVs are the paper's
  proportion series × pop. x becomes a real-valued compartment
  X = pop·x with an `ode {}` law; identical dynamics up to scale.
  (The paper's script assigns the Netherlands Germany's population — a
  bug kept for fidelity; it only rescales the count units.)
- **camdl's `normal` likelihood is a discretized-count density**, so the
  OSI stream is fitted on a permille (0–1000) scale where rounding costs
  <0.1%.
- **Likelihood replaces ABC tolerances.** The paper runs ABC-SMC with
  separate acceptance tolerances for the incidence and behaviour
  streams; here each stream gets a normal likelihood, whose sd is the
  tolerance's analogue. The case-stream sd σ_i is estimated; the
  behaviour-stream sd is **fixed** at 300‰ — the paper's own accepted
  particles have behaviour RMS 0.29–0.50, and with a *free* scale the
  ML solution inflates it to wash the OSI stream out entirely (the
  paper's Belarus/Finland failure mode, for every country).
- **Ii0 (initial reported incidence) is dropped**: it affects only the
  day-1 observation, which is ≈0 everywhere. Parameter-count parity
  with the paper holds anyway (SIR 7, SIRx 12 fitted parameters).
- **Fitting**: camdl's ODE backend, NLopt-Sbplx multi-start scout
  (LHS starts) then 4-chain adaptive Metropolis-Hastings, 80k
  iterations, uniform priors on the paper's ranges. The training window
  is each country's Table-S1 cutoff (200–250 days); the rest is
  out-of-sample. (camdl parses `[data].holdout_after` but does not yet
  apply it — the split is materialised in `*_train.tsv` files.)

## Files

- `sir_covid.camdl`, `sirx_covid.camdl` — the two models
- `prep_data.py` — builds `data/*.tsv` from the authors' repository
- `gen_fits.py` — writes the 26 per-country fit configs into `fits/`
- `run_fits.sh` — runs all fits (~2.5 h on 4 cores)
- `analyze.py` — posterior trajectories + the paper's metrics
  (peaks, area, AICc, Table S3) via a scipy re-integration validated
  against camdl's ODE backend to <0.2%
- `plot_waves.py` — the three figures
- `results/summary_stats.tsv`, `results/table_s3_comparison.tsv`

## Quickstart

```bash
python3 prep_data.py /path/to/authors-repo-clone
python3 gen_fits.py
./run_fits.sh
uv run python analyze.py     # from camdl_replication/
uv run python plot_waves.py
```
