# Copula-based latent variable models in NumPyro

Notes from a UW seminar by Sunbeom Kwon (College of Education postdoc),
*Copula-Based Latent Variable Modeling for Asymmetric Dependence*, 2026-10-14.
The talk covers two studies:

1. **Kwon & Zhang, "A copula-based joint model for speed and accuracy"**
   (Psychometrika, in press; presented at IMPS 2025 and NCME 2026). Van der
   Linden's hierarchical model for item responses and response times, with the
   bivariate-normal link between latent ability and latent speed replaced by a
   **Gumbel copula** (upper-tail dependence).
2. **Kwon & Zhang, "Predicting binary risk outcomes via copula-based SEM"**
   (in preparation). The same idea in an SEM with a rare binary outcome such as
   suicide attempt, where Gaussian dependence underestimates joint risk.

Neither paper is public yet, so this directory simulates data from each setting
as I understand it from the abstracts, fits the copula models and the usual
Gaussian-dependence versions with NumPyro (NUTS), and compares them.

RESULTS_TLDR

## Why a copula

A multivariate normal says the dependence between two latent variables is
linear, symmetric, and tail-independent: knowing that someone is extreme on one
trait tells you nothing extra about whether they are extreme on the other,
beyond the correlation. Sklar's theorem splits any joint distribution into
margins plus a *copula* that carries the dependence, so you can keep normal
margins for the latent traits and swap in a copula with the dependence shape you
want. The Gumbel copula has upper-tail dependence
`lambda_U = 2 - 2^(1/delta)`: high values travel together, low values do not.

![Gumbel vs Gaussian persons](figures/study1_dgp.png)

Both panels have the same margins and the same Kendall's tau. In the Gumbel
panel (left) the fast-and-able corner is tight while speed among low-ability
persons is spread out, which is the empirical pattern Kwon & Zhang cite as
motivation (Domingue et al., 2022: more response-time variability among lower
ability test-takers).

## The trick in six lines of NumPyro

If `theta` and `tau_std` have standard-normal margins and copula density `c`,
their joint density is `phi(theta) phi(tau_std) c(Phi(theta), Phi(tau_std))`.
So two independent `Normal(0, 1)` draws supply the margins and a
`numpyro.factor` adds the log copula density:

```python
with numpyro.plate("persons", n_persons):
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))
    tau_std = numpyro.sample("tau_std", dist.Normal(0.0, 1.0))
ktau = numpyro.sample("ktau", dist.Beta(2.0, 2.0))              # Kendall's tau
numpyro.factor("copula", jnp.sum(cp.logpdf_z(theta, tau_std, "gumbel", ktau)))
```

See `speed_accuracy.py:88`. The copula functions live in `copulas.py`
(Gaussian, Gumbel, Clayton; rotations by 90/180/270 degrees for negative
dependence or the opposite tail; all parameterised by Kendall's tau so the
families are comparable). Two numerical details matter in practice:

- Everything works from `norm.logcdf(z)` rather than `Phi(z)`, and the Gumbel
  and Clayton formulas are rearranged so the small tail terms are never
  absorbed into large ones (`copulas.py:78`). The first draft computed
  `P(Y=1 | eta)` as exactly zero at `eta = -7`, which NUTS does not forgive.
- 64-bit JAX (`numpyro.enable_x64()`) is required; float32 loses the tails.

For a binary outcome (Study 2) there is a second ingredient. If the outcome is
`Y = 1{y* > c}` with `y*` latent and `(eta, y*)` joined by a copula, then

```
P(Y = 1 | eta) = 1 - h(Phi(c) | Phi(eta)),   h(v | u) = dC(u, v) / du
```

The conditional CDF `h` is closed-form for all three families
(`copulas.log_hfunc_z`), so `y*` never has to be sampled; the model is a
Bernoulli likelihood with a copula-shaped link (`risk_sem.py:85`). Under a
Gaussian copula this reduces to the familiar probit regression on `eta`.

`test_copulas.py` checks that each density integrates to one, that `h` is the
integral of the density, that the samplers reproduce Kendall's tau and the
right tail, and that gradients stay finite far into the tails
(`uv run pytest`).

## Study 1: speed and accuracy

**Data.** 1,000 persons by 20 items. Responses follow a two-parameter logistic
model, log response times a log-normal model with item time intensity and time
precision, as in van der Linden (2007):

```
Y_ij    ~ Bernoulli(logit^-1(a_j (theta_i - b_j)))
log T_ij ~ Normal(beta_j - tau_i, 1 / alpha_j)
theta_i ~ N(0, 1),  tau_i ~ N(0, 0.4^2),  (Phi(theta_i), Phi(tau_i / 0.4)) ~ Gumbel(delta = 2)
```

Kendall's tau between ability and speed is 0.5 and the upper-tail dependence
coefficient is 0.59.

**Models.** Four NumPyro models on the same data set, NUTS with 2 chains of
1,000 warmup and 1,000 draws each:

| model | dependence between ability and speed |
|---|---|
| `2pl_only` | response times ignored |
| `gaussian` | bivariate normal (van der Linden's hierarchical model) |
| `gumbel` | Gumbel copula, the speaker's model |
| `clayton` | Clayton copula, an asymmetric copula with the *wrong* (lower) tail |

**What is compared.** The three claims in the abstract, in order: ability
recovery (RMSE of the posterior-mean ability against the truth, overall and by
tercile), model fit (per-person WAIC), and the empirical pattern that motivates
the model, as a posterior predictive check: the SD across persons of their mean
log response time, within quintiles of observed score.

RESULTS_STUDY1

## Study 2: a rare binary risk outcome

This paper is not public, so this is one reading of the abstract. One latent
risk factor `eta` (think depressive symptoms) is measured by six continuous
indicators; the outcome is rare (3% prevalence) and is linked to the risk
factor through a copula between `eta` and the latent outcome propensity `y*`:

```
x_ik ~ N(nu_k + lambda_k eta_i, sigma_k^2),  k = 1..6
(Phi(eta_i), Phi(y*_i)) ~ Gumbel(delta = 2),  Y_i = 1{y*_i > Phi^-1(0.97)}
```

Under a Gaussian copula with latent correlation `rho` this is the standard
probit SEM, `P(Y=1 | eta) = Phi((rho eta - c) / sqrt(1 - rho^2))`. Under the
Gumbel copula `P(Y=1 | eta)` rises much more steeply in the upper tail: at the
99th percentile of the risk factor the true outcome probability is 0.69, while
a Gaussian copula with the same Kendall's tau gives 0.37.

**Models.** Gaussian, Gumbel and Clayton copula SEMs, each fit to 3,000
training persons, then used to predict 20,000 held-out persons from their
indicators alone (`eta | x` is Gaussian under the linear measurement model, so
the prediction integrates `P(Y=1 | eta)` over it by Gauss-Hermite quadrature,
averaged over posterior draws).

RESULTS_STUDY2

## Files

| file | what |
|---|---|
| `copulas.py` | Gaussian / Gumbel / Clayton copulas in JAX: log density, conditional CDF, samplers, tail coefficients, rotations |
| `test_copulas.py` | 33 tests (`uv run pytest`) |
| `speed_accuracy.py` | Study 1: simulation, four NumPyro models, metrics, posterior predictive check |
| `risk_sem.py` | Study 2: simulation, three copula SEMs, out-of-sample prediction |
| `make_figures.py` | the figures above, from `results/` |
| `results/study*/summary.json` | every number quoted here |

```bash
uv sync
uv run pytest
uv run python speed_accuracy.py      # ~RUNTIME1 on 4 CPU cores
uv run python risk_sem.py            # ~RUNTIME2
uv run python make_figures.py
```

## Caveats, and questions for the speaker

CAVEATS

## Sources

- Seminar announcement (abstract quoted in the task); CV of Sunbeom Kwon,
  https://sunbeomk.github.io/assets/pdf/Kwon_Sunbeom_CV.pdf (lists both papers)
- IMPS 2025 abstract, "A copula-based joint model for item responses and
  response times" (Kwon & Zhang), p. 40 of
  https://www.psychometricsociety.org/sites/main/files/file-attachments/imps2025-abstracts.pdf
- Illinois Psychology news item on the IMPS 2025 travel award:
  https://psychology.illinois.edu/news/2025-05-27/sunbeom-kwon-trung-le-are-heading-international-meeting-psychometric-society
- van der Linden, W. J. (2007). A hierarchical framework for modeling speed
  and accuracy on test items. *Psychometrika*, 72, 287-308.
  https://link.springer.com/article/10.1007/s11336-006-1478-z
- Hofert, M. (2011). Efficiently sampling nested Archimedean copulas.
  *Computational Statistics & Data Analysis* (the Marshall-Olkin / positive
  stable sampler used in `copulas.sample`).
- Nikoloulopoulos & Joe, factor copula models for item response data,
  *Psychometrika* 2015, for copulas in IRT more broadly:
  https://www.cambridge.org/core/journals/psychometrika/article/abs/factor-copula-models-for-item-response-data/8ECB92F29C0DC97EED08CBE85376869C
