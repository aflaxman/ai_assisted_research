"""MAP + Laplace calibration of a structure to the cohort likelihood.

NUTS struggles with the classic structures because several are non-identifiable
from the available cohort data (a ridge in the posterior), which is itself a
finding of the Menzies review.  MAP optimisation with a Laplace (Gaussian)
approximation handles that cleanly and fast: it returns a point estimate,
approximate credible intervals, and -- through the Hessian eigenvalue spectrum
-- a direct identifiability diagnostic (near-zero eigenvalues = unconstrained
parameter directions).

Parameters are optimised in an unconstrained space (log for rates, logit for
probabilities), with weakly-informative Gaussian priors there that regularise
the non-identified directions while letting the data dominate where they
constrain.  The implied prior on a rate is log-normal; on a probability, a mild
shrink toward 0.5.  The TB-death rate keeps its informative prior.

Validated to reproduce the SP-full NUTS posterior (and hence Horton 2023).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from . import likelihood as L
from .core import Structure
from .data import load_converter_cohorts, load_ferebee_converter, load_spectrum_transitions

jax.config.update("jax_enable_x64", True)

import pandas as pd

# unconstrained-space prior (mean, sd) and the transform for each parameter kind
LOGRATE_PRIOR = (np.log(0.5), 2.0)
LOGIT_PRIOR = (0.0, 1.5)
X2_PRIOR = (0.0, 1.5)
LOGX1_PRIOR = (np.log(0.1), 1.5)
MORT_PRIOR = (np.log(0.389), 0.072)   # ~ Normal(0.389, 0.028) on the rate


def _spec(name: str):
    """Return (transform, inverse, prior_mean, prior_sd) for a parameter."""
    if name in ("mort", "tb_death"):
        return (jnp.exp, np.log, *MORT_PRIOR)
    if name in ("a", "b", "phi"):
        sig = lambda u: 1.0 / (1.0 + jnp.exp(-u))
        logit = lambda x: np.log(x / (1 - x))
        return (sig, logit, *LOGIT_PRIOR)
    if name == "x1":
        return (jnp.exp, np.log, *LOGX1_PRIOR)
    if name == "x2":
        return (lambda u: u, lambda x: x, *X2_PRIOR)
    if name == "x3":
        return (jnp.exp, np.log, *LOGRATE_PRIOR)
    return (jnp.exp, np.log, *LOGRATE_PRIOR)  # generic positive rate


def _data(s: Structure):
    conv = pd.concat([load_converter_cohorts(), load_ferebee_converter()], ignore_index=True)
    cohorts = L.prep_converter(conv)
    if s.family == "classic":
        cohorts = [c for c in cohorts if c["band"] == "infectious"]
    trans = L.prep_transitions(load_spectrum_transitions()) if s.family == "spectrum" else None
    return cohorts, trans


def build_potential(s: Structure):
    names = list(s.param_names)
    specs = [_spec(n) for n in names]
    transforms = [sp[0] for sp in specs]
    pmean = jnp.array([sp[2] for sp in specs])
    psd = jnp.array([sp[3] for sp in specs])
    cohorts, trans = _data(s)

    def to_dict(u):
        return {n: transforms[i](u[i]) for i, n in enumerate(names)}

    def loglik(u):
        p = to_dict(u)
        ll = L.loglik_converters(s, p, cohorts)
        if s.family == "classic":
            ll = ll + L.loglik_case_fatality(s, p)
        else:
            ll = ll + L.loglik_transitions(s, p, trans)
        return ll

    def logprior(u):
        return jnp.sum(-0.5 * ((u - pmean) / psd) ** 2)

    def neg_log_post(u):
        return -(loglik(u) + logprior(u))

    return names, to_dict, neg_log_post


def calibrate_map(s: Structure, n_restarts: int = 8, n_draws: int = 2000, seed: int = 0):
    names, to_dict, nlp = build_potential(s)
    k = len(names)
    val_grad = jax.jit(jax.value_and_grad(nlp))

    def scipy_obj(u):
        v, g = val_grad(jnp.asarray(u))
        return float(v), np.asarray(g, dtype=float)

    rng = np.random.default_rng(seed)
    prior_mean = np.array([_spec(n)[2] for n in names])
    best = None
    for r in range(n_restarts):
        u0 = prior_mean + rng.normal(0.0, 1.0, size=k)  # jittered start near prior mean
        try:
            res = minimize(scipy_obj, u0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=500))
            if not np.isfinite(res.fun):
                raise ValueError("non-finite objective")
        except Exception:  # noqa: BLE001
            # gradient-free fallback for awkward (e.g. time-varying-hazard) models
            try:
                res = minimize(lambda u: scipy_obj(u)[0], u0, method="Nelder-Mead",
                               options=dict(maxiter=4000, xatol=1e-5, fatol=1e-5))
            except Exception:  # noqa: BLE001
                continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError(f"{s.name}: MAP optimisation failed")

    u_map = jnp.asarray(best.x)
    H = np.asarray(jax.hessian(nlp)(u_map))
    # regularise for inversion; eigenvalues of H are the identifiability spectrum
    evals = np.linalg.eigvalsh((H + H.T) / 2)
    try:
        cov = np.linalg.inv((H + H.T) / 2 + 1e-6 * np.eye(k))
        cov = (cov + cov.T) / 2
    except np.linalg.LinAlgError:
        cov = np.eye(k)
    # Laplace draws in unconstrained space, then transform
    draws_u = np.random.default_rng(seed + 1).multivariate_normal(
        np.asarray(u_map), cov, size=n_draws, check_valid="ignore")
    transforms = [_spec(n)[0] for n in names]
    draws = {n: np.asarray(transforms[i](jnp.asarray(draws_u[:, i]))) for i, n in enumerate(names)}
    map_params = {n: float(np.asarray(_spec(n)[0](u_map[i]))) for i, n in enumerate(names)}
    return dict(names=names, map=map_params, draws=draws,
                hessian_evals=evals, neg_log_post=float(best.fun),
                cov=cov, u_map=np.asarray(u_map))
