# Early-warning signals with a social-media compartment — a replication

A clean-room replication of

> Olajide F, Lutscher F, Smith SR (2026). *Early-warning signals for infectious
> diseases with a social-media compartment.* PLOS ONE 20(7): e0354091.
> https://doi.org/10.1371/journal.pone.0354091

Run [`replicate_ews.ipynb`](replicate_ews.ipynb) for the narrated version. This
README is the short tour.

> The paper was mislabeled "LINK3D" in the Google Scholar alert that prompted
> this work; that name has nothing to do with the article.

## The question

As a system nears a tipping point it recovers from shocks more and more slowly —
**critical slowing down** — so its **variance** and **lag-1 autocorrelation**
climb. Those rising statistics are **early-warning signals (EWSs)**. The paper
forces a disease model past its emergence threshold ($R_0$ crossing 1) and asks:
do the signals fire in time, and does a **social-media** signal warn us earlier
than case counts?

## The model (SWIRM)

Five compartments — **S**usceptible, **W** (unreported infected), **I** (reported
infected), **R**ecovered, **M** (social-media posts) — with emergence forced by a
transmission rate that ramps in time, $\beta(t)=\beta_0+rt$. Only reported
infections transmit; media is a one-way *observer* that never feeds back. The
model is simulated under three noise types (additive, multiplicative,
demographic) via Euler–Maruyama, 300 realizations each.

**A reassuring check:** plugging the Table 1 parameters into the next-generation
$R_0$ puts the bifurcation at **t = 14.50 days** — exactly the paper's value.
`tests/` enforces this.

## What we reproduce

| Finding | Paper | This replication |
|---|---|---|
| Bifurcation time (R₀=1) | 14.5 d | **14.50 d** |
| Emergence is delayed past the bifurcation | yes | yes (spike ~90–110 d) |
| Variance AUC, reported *I*, multiplicative-low | ≈0.99 | **0.96** |
| Autocorrelation AUC, reported *I* | poor | **0.45** |
| Variance > autocorrelation for *I* and *W* | yes | **yes** |
| Social-media *M* AUC (all noise) | ≈0.5 | **0.47–0.53** |

The two headline conclusions — **variance beats autocorrelation for reported
infections**, and **social media carries no usable warning** — reproduce
cleanly.

## How it fits the literature

- **Contradicts** O'Regan & Burton and Chakraborty et al. (autocorrelation-favoring)
  and **sides with** the variance-favoring studies; the reconciler is model
  dimensionality and noise structure.
- **Extends** O'Regan & Drake on bifurcation delay: EWSs computed on the approach
  to the (late) changepoint still fire.
- **Departs** from data-driven Twitter/Google-Trends epidemiology by asking the
  mechanistic question — and finding classical EWSs cannot read the posts.

## Where this replication differs from the paper

1. **Noise intensities** (low/medium/high for additive & multiplicative) appear
   only in figure captions, absent from the machine-readable text. We chose
   values, which shifts AUC *magnitudes* — additive-noise and *W* AUCs come out
   lower than published — without changing the qualitative story.
2. **Demographic stochasticity** here yields mostly *delays*; the paper reports
   ~70% *advances*. Our AMOC penalty is Monte-Carlo-calibrated to a 5%
   false-positive rate, so it skips the spurious early changepoints that the
   paper notes high demographic noise can produce — the likely source of most of
   their "advances."
3. **Tooling.** `changepoint.py` and `ews.py` are transparent NumPy
   reimplementations of R's `cpt.mean` (AMOC) and `earlywarnings` (Gaussian
   detrend + rolling window), not the original R packages; small numeric
   differences follow.

## Layout

```
swirm/
  model.py        deterministic SWIRM, R0(t), bifurcation
  simulate.py     Euler–Maruyama: additive / multiplicative / demographic noise
  changepoint.py  AMOC mean-shift detection
  ews.py          Gaussian detrend, rolling variance / lag-1 AC, Kendall tau
  evaluate.py     ensemble runner -> delays + ROC/AUC (Table 2, Figs 7–10)
  plots.py        thin plotting helpers
tests/            analytic + numerical checks (R0=1 at 14.5 d, etc.)
replicate_ews.ipynb   the narrative
PLAN.md           the replication plan and confirmed parameters
```

## Quickstart

```bash
uv venv
uv pip install -e ".[dev,notebook]"
uv run pytest -q                 # analytic + numerical checks
uv run jupyter lab replicate_ews.ipynb
```
