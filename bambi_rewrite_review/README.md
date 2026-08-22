# Bambi rewrite review (PR #1002)

Notes and experiments on [bambinos/bambi#1002](https://github.com/bambinos/bambi/pull/1002),
tomicapretto's draft rewrite of Bambi (families decoupled from PyMC, new defaults system,
pluggable backend, `model_components` → `parameters`).

`simple_regression_before_after.ipynb` fits the same simple linear regression (`y ~ x`) under
Bambi 0.20.0 (current PyPI release) and under the rewrite branch at commit `3fc7196`, and shows
every user-visible difference as executed output.

## Quickstart

```bash
cd bambi_rewrite_review
./setup.sh
.venv-new/bin/jupyter nbconvert --to notebook --execute --inplace simple_regression_before_after.ipynb
```

## What setup.sh does

The notebook needs two environments, built with `uv`:

- `.venv-old`: `bambi==0.20.0` from PyPI, plus `matplotlib-inline` (the notebook kernel exports
  `MPLBACKEND=module://matplotlib_inline.backend_inline` to the `%%script` subprocess, so the old
  env must be able to import it).
- `.venv-new`: the PR branch, cloned into `.bambi-pr1002/` at commit `3fc7196`, **with a one-line
  patch**: at that commit, `bambi/backend/pymc/model.py:557` reads
  `except RuntimeError, ValueError:` (Python 2 syntax), so the package does not import until the
  exception types are parenthesized. The rest of the branch works — the notebook runs on it.

The notebook kernel runs in `.venv-new`; the "before" cells shell out to `.venv-old` via
`%%script`, which is also why the old fit uses `cores=1` (code fed over stdin breaks
multiprocessing spawn).

## Findings

- The surface API for a simple regression is nearly unchanged: same formula interface, identical
  automatic priors, matching posteriors.
- Internals consolidated: `model.components` / `response_component` /
  `constant_components` / `distributional_components` are replaced by a single
  `model.parameters` dict.
- The rewrite centers predictors in the computation graph: PyMC samples `Intercept_centered` and
  recovers `Intercept`; the result `DataTree` gains a `constant_data` group.
- `fit()` defaults to `inference_method=None` (backend selects); `bmb.Link` drops
  `linkinv`/`linkinv_backend` for a single `inverse_link`.
- The ecosystem-level changes (fit returning `xarray.DataTree`, ArviZ 1.x plotting) already landed
  in Bambi 0.20.0 and are *not* part of the PR.
