"""Differentiable (JAX) forward evaluations of a Structure, for calibration.

The scipy engine in :mod:`tbmodels.core` is the reference; this module mirrors
it in JAX so numpyro/NUTS can differentiate through it.  Two quantities are
needed by the cohort likelihood:

* **first-passage cumulative incidence** -- the fraction of a cohort that has
  ever reached a target band by time t, with the target made absorbing;
* **sub-model occupancy** -- the fraction in an observed state at time t given
  the cohort started in a named state (for the Min/Sub/Clin transition data).

Constant-rate structures use ``jax.scipy.linalg.expm``.  Structures K and L
have time-since-infection hazards; their single-latent first-passage is given
in closed form (they appear only in the classic cohort fit, where the target is
the single disease state).

``jax_generator`` is validated against the scipy generator in the tests.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from .core import Structure

jax.config.update("jax_enable_x64", True)


def jax_generator(s: Structure, p: dict, absorbing: frozenset[str] | None = None):
    """Build the generator matrix in JAX from a dict of jnp scalar params.

    Only valid for constant-rate structures (lambda=0, t=0): the rate closures
    are evaluated once.  Structures K and L are handled by closed form elsewhere.
    """
    n = s.n
    Q = jnp.zeros((n, n))
    counter = {i for i, c in enumerate(s.compartments) if c.kind == "counter"}
    for tr in s.transitions:
        if absorbing is not None and tr.src in absorbing:
            continue
        r = tr.rate(p, 0.0, 0.0)
        i, j = s.idx(tr.src), s.idx(tr.dst)
        Q = Q.at[j, i].add(r)
        if i not in counter:
            Q = Q.at[i, i].add(-r)
    return Q


def _entry_vector(s: Structure, p: dict):
    x0 = jnp.zeros(s.n)
    for name, prob in s.entry(p).items():
        x0 = x0.at[s.idx(name)].add(prob)
    return x0


def jax_first_passage(s: Structure, p: dict, times, target):
    """Fraction ever reaching any ``target`` compartment by each time
    (constant-rate structures)."""
    tset = frozenset(target)
    tidx = jnp.array([s.idx(n) for n in target])
    Q = jax_generator(s, p, absorbing=tset)
    x0 = _entry_vector(s, p)
    occ = jnp.stack([expm(Q * t) @ x0 for t in times])
    return occ[:, tidx].sum(axis=1)


def jax_occupancy_from(s: Structure, p: dict, times, start: str, observe):
    """Occupancy of the ``observe`` compartments at each time, given the cohort
    started entirely in ``start`` (no absorbing states)."""
    oidx = jnp.array([s.idx(n) for n in observe])
    Q = jax_generator(s, p, absorbing=None)
    x0 = jnp.zeros(s.n).at[s.idx(start)].add(1.0)
    occ = jnp.stack([expm(Q * t) @ x0 for t in times])
    return occ[:, oidx].sum(axis=1)


def jax_first_passage_from(s: Structure, p: dict, times, start: str, target):
    """First-passage cumulative incidence to ``target`` given the cohort started
    in ``start`` (target absorbing)."""
    tset = frozenset(target)
    tidx = jnp.array([s.idx(n) for n in target])
    Q = jax_generator(s, p, absorbing=tset)
    x0 = jnp.zeros(s.n).at[s.idx(start)].add(1.0)
    occ = jnp.stack([expm(Q * t) @ x0 for t in times])
    return occ[:, tidx].sum(axis=1)


# -- closed-form first passage for the single-latent time-varying structures --
def cumhaz_K(p: dict, t):
    """integral_0^t x1 * max(1, s)^x2 ds."""
    x1, x2 = p["x1"], p["x2"]
    # for t <= 1: x1*t ; for t > 1: x1*(1 + (t^(x2+1) - 1)/(x2+1))
    below = x1 * t
    above = x1 * (1.0 + (t ** (x2 + 1.0) - 1.0) / (x2 + 1.0))
    return jnp.where(t <= 1.0, below, above)


def cumhaz_L(p: dict, t):
    """integral_0^t x1 * (x2 + exp(-x3 s)) ds = x1*(x2*t + (1-exp(-x3 t))/x3)."""
    x1, x2, x3 = p["x1"], p["x2"], p["x3"]
    return x1 * (x2 * t + (1.0 - jnp.exp(-x3 * t)) / x3)


def jax_first_passage_KL(name: str, p: dict, times):
    cum = cumhaz_K(p, times) if name == "K" else cumhaz_L(p, times)
    return 1.0 - jnp.exp(-cum)
