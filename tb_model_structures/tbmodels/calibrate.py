"""Bayesian calibration of each structure to the cohort likelihood blocks.

Priors follow the two source models where they overlap (Horton 2023 for the
spectrum rates and the TB-death prior; wide uniforms for the classic
progression/transfer rates).  The log-likelihood is assembled from the blocks
each family can address (:mod:`tbmodels.likelihood`).  NUTS runs via numpyro.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from . import likelihood as L
from .core import Structure
from .data import load_converter_cohorts, load_spectrum_transitions

jax.config.update("jax_enable_x64", True)

# TB-death prior shared by both families (Richards/Horton: N(0.389, 0.028)).
MORT_LOC, MORT_SCALE = 0.389, 0.028


def _sample_param(name: str, family: str):
    """Draw one parameter from its prior."""
    if name in ("mort", "tb_death"):
        return numpyro.sample(name, dist.TruncatedNormal(MORT_LOC, MORT_SCALE, low=1e-3))
    if name in ("a", "b", "phi"):
        return numpyro.sample(name, dist.Uniform(0.0, 1.0))
    if name == "x1":
        return numpyro.sample(name, dist.Uniform(0.0, 2.0))
    if name == "x2":
        # K uses x2 as a negative exponent; L as a non-negative offset.
        return numpyro.sample(name, dist.Uniform(-3.0, 3.0))
    if name == "x3":
        return numpyro.sample(name, dist.Uniform(0.0, 3.0))
    if name == "self_cure":
        return numpyro.sample(name, dist.Uniform(0.0, 3.0))
    if family == "spectrum":
        hi = 6.0 if name == "infclr" else 3.0
        return numpyro.sample(name, dist.Uniform(0.0, hi))
    # classic progression / transfer rates
    return numpyro.sample(name, dist.Uniform(0.0, 12.0))


def build_model(s: Structure):
    cohorts = L.prep_converter(load_converter_cohorts())
    transitions = L.prep_transitions(load_spectrum_transitions()) if s.family == "spectrum" else None

    def model():
        p = {name: _sample_param(name, s.family) for name in s.param_names}
        # converter cohorts: classic sees only infectious-band cohorts as "I";
        # spectrum sees both bands with its own target mapping.
        if s.family == "classic":
            use = [c for c in cohorts if c["band"] == "infectious"]
        else:
            use = cohorts
        numpyro.factor("ll_converters", L.loglik_converters(s, p, use))
        if s.family == "classic":
            # classic has only one disease state, so duration + case fatality
            # pin the self-cure / TB-death exit rates
            numpyro.factor("ll_duration", L.loglik_duration(s, p))
            numpyro.factor("ll_casefatality", L.loglik_case_fatality(s, p))
        else:
            # spectrum rates (incl. infectious-disease duration) are pinned by
            # the Min/Sub/Clin transition data and the mort prior; duration is
            # reported as a derived diagnostic, not a fitting target
            numpyro.factor("ll_transitions", L.loglik_transitions(s, p, transitions))

    return model


def calibrate(
    s: Structure,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    seed: int = 0,
    target_accept: float = 0.9,
) -> MCMC:
    model = build_model(s)
    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False, chain_method="sequential")
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def pointwise_loglik(s: Structure, samples: dict):
    """Per-observation log-likelihoods for LOO/WAIC.

    Returns a dict block -> array of shape (n_draws, n_obs).  Converter obs are
    per interval; transition obs per observation row; duration/case-fatality are
    single summary targets.
    """
    import numpy as np
    from . import jaxcore as J

    cohorts = L.prep_converter(load_converter_cohorts())
    if s.family == "classic":
        cohorts = [c for c in cohorts if c["band"] == "infectious"]
    transitions = L.prep_transitions(load_spectrum_transitions()) if s.family == "spectrum" else []

    names = list(s.param_names)
    draws = {k: np.asarray(samples[k]) for k in names}
    n_draws = len(draws[names[0]])

    conv_ll, dur_ll, cf_ll, trans_ll = [], [], [], []
    for d in range(n_draws):
        p = {k: jnp.asarray(draws[k][d]) for k in names}
        # converters: one log-lik per interval
        row = []
        for co in cohorts:
            target = L.converter_target(s, co["band"])
            tgrid = jnp.asarray(np.concatenate([[0.0], co["year"]]))
            ci = J.jax_first_passage_KL(s.name, p, tgrid) if s.name in ("K", "L") \
                else J.jax_first_passage(s, p, tgrid, target)
            surv = 1.0 - ci
            q = jnp.clip((surv[:-1] - surv[1:]) / jnp.clip(surv[:-1], 1e-9, 1.0), 1e-9, 1 - 1e-9)
            ev, n = jnp.asarray(co["events"]), jnp.asarray(co["at_risk"])
            row.extend(np.asarray(ev * jnp.log(q) + (n - ev) * jnp.log1p(-q)))
        conv_ll.append(row)
        dur_ll.append(float(L.loglik_duration(s, p)))
        if s.family == "classic":
            cf_ll.append(float(L.loglik_case_fatality(s, p)))
        else:
            trow = []
            for grp in transitions:
                trow.append(float(L.loglik_transitions(s, p, [grp])))
            trans_ll.append(trow)

    out = {"converters": np.array(conv_ll), "duration": np.array(dur_ll)[:, None]}
    if s.family == "classic":
        out["case_fatality"] = np.array(cf_ll)[:, None]
    else:
        out["transitions"] = np.array(trans_ll)
    return out
