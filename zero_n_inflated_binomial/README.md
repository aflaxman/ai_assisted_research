# Zero- and N-inflated Binomial in NumPyro

A small experiment: fit a zero- and N-inflated Binomial model in NumPyro to
simulated data, then progressively misspecify the data-generating process and
watch the fit deteriorate.

## Model

For each observation `y_i ∈ {0, 1, ..., N}`:

```
y_i = 0      with probability π_0  (structural zero)
y_i = N      with probability π_N  (structural max)
y_i ~ Bin(N, p)  with probability 1 - π_0 - π_N
```

Priors: `π = (π_0, π_N, π_bin) ~ Dirichlet(1, 1, 1)`, `p ~ Beta(1, 1)`.
The likelihood marginalises the discrete component label with `logsumexp`.

See `zni_binomial.py:30`.

## Stages of misspecification

1. **Matched.** Data drawn from the same model. Posterior recovers
   `π_0 = 0.15`, `π_N = 0.10`, `p = 0.35`; posterior predictive matches the
   histogram.
2. **Beta-Binomial overdispersion.** Per-obs `p_i ~ Beta(2, 4)`. The body is
   wider than any single Binomial(N, p); the fit absorbs the spread by
   inflating `π_0` and `π_N` and missing the shoulders.
3. **Two-component mixture in p.** Half the obs use `p = 0.15`, half use
   `p = 0.75`. The body is bimodal, but the model can only place one
   unimodal Binomial bump - the worst fit of the four.
4. **Variable trial counts.** `N_i` varies in `[8, 20]`; the model assumes
   `N = 20`. The structural-N inflation collapses (`π_N ≈ 0`) because no
   observation reaches 20 unless drawn from the body.

## Run

```bash
uv sync
uv run python zni_binomial.py
```

Outputs `frames/stage_*.png` and `zni_binomial_stages.gif`.

![stages](zni_binomial_stages.gif)
