# Replicating He, Ionides & King (2010) with camdl

Notes from a first end-to-end run of [camdl](https://github.com/vsbuffalo/camdl)
([intro blog post](https://vincebuffalo.com/blog/introducing-camdl/)), a DSL +
compiler + inference stack for stochastic compartmental models, developed at
the Institute for Disease Modeling.

**Status: draft — results filled in below as runs complete.**

## What this replicates

He, Ionides & King (2010), *Plug-and-play inference for disease dynamics:
measles in large and small populations as a case study*, J. R. Soc. Interface
7:271–283 ([doi:10.1098/rsif.2009.0151](https://doi.org/10.1098/rsif.2009.0151)).
The London weekly measles notification series (1944–1965), fit with a
stochastic SEIR model featuring school-term forcing, a cohort school-entry
pulse, inhomogeneous mixing, and extra-demographic transmission noise.

This directory exercises camdl's workflow at the paper's published London MLE
(R0 = 56.8, amplitude = 0.554, rho = 0.488, ...):

1. `camdl check` — compile-time validation (units, dimensions)
2. `camdl simulate` — forward simulation at the MLE vs the observed series
3. `camdl pfilter` — particle-filter log-likelihood at the MLE, compared
   against the reference implementation (R's `pomp`, the package from the
   original authors)
4. `camdl fit run` — a small IF2 "scout" fit re-estimating three headline
   parameters from the data

## Files

- `he2010_london.camdl` — the model (from camdl's external-validation suite,
  Apache-2.0; annotated line-by-line correspondence with the paper)
- `params/he2010_london_mle.toml` — published MLE values
- `data/he2010_london_cases.tsv` — London weekly notifications, 1096 weeks
- `data/he2010_london_covariates.tsv` — spline-smoothed population and
  (4-year-lagged) birthrate covariates
- `fit_he2010.toml` — IF2 scout-fit configuration
- `plot_replication.py` — figure: observed vs simulated series + pfilter
  log-likelihoods vs the pomp reference

## Results

(to be filled in)

## Quickstart

```bash
./install.sh   # in a camdl clone — installs OCaml + Rust toolchains and camdl
camdl check he2010_london.camdl
camdl simulate he2010_london.camdl --params params/he2010_london_mle.toml \
    --backend chain_binomial --dt 1.0 --seed 42 --obs-only results/sim_obs.tsv
camdl pfilter he2010_london.camdl --params params/he2010_london_mle.toml \
    --data data/he2010_london_cases.tsv --particles 2000 --replicates 10 \
    --seed 42 --output results/pfilter_logliks.tsv
uv run python plot_replication.py
```
