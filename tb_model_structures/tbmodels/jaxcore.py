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

from .core import Structure

jax.config.update("jax_enable_x64", True)


_EXPM_SQUARINGS = 16     # 2^16 = 65536 >> any ||M*t|| encountered here
_EXPM_TAYLOR = 6         # Taylor order for the scaled matrix (||.|| ~ 1e-3)


def _expm_action(M, x0, times):
    """Return expm(M*t) @ x0 for each t in times, shape (len(times), n).

    Fixed-count scaling and squaring, batched over times: scale M*t down by
    2^16, approximate the exponential of the tiny matrix with a 6th-order Taylor
    series (exact to machine precision at that norm), then square 16 times.  All
    pure matmuls with a data-independent operation count, so it is fast to
    evaluate and to differentiate in NUTS (unlike jax.scipy.linalg.expm, whose
    dynamic squaring makes the reverse-mode gradient expensive), and robust for
    defective (non-diagonalisable) generators such as the chain structures.
    """
    times = jnp.asarray(times, float)
    n = M.shape[0]
    A = times[:, None, None] * M[None]            # (T, n, n) = M*t
    As = A / (2.0 ** _EXPM_SQUARINGS)
    eye = jnp.eye(n)[None]
    # Horner form of I + X + X^2/2! + ... + X^k/k!
    E = eye + As / _EXPM_TAYLOR
    for k in range(_EXPM_TAYLOR - 1, 0, -1):
        E = eye + (As @ E) / k
    for _ in range(_EXPM_SQUARINGS):
        E = E @ E
    return (E @ x0[None, :, None])[:, :, 0]        # (T, n)


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
    tidx = jnp.array([s.idx(n) for n in target])
    Q = jax_generator(s, p, absorbing=frozenset(target))
    occ = _expm_action(Q, _entry_vector(s, p), jnp.asarray(times, float))
    return occ[:, tidx].sum(axis=1)


def jax_occupancy_from(s: Structure, p: dict, times, start: str, observe):
    """Occupancy of the ``observe`` compartments at each time, given the cohort
    started entirely in ``start`` (no absorbing states)."""
    oidx = jnp.array([s.idx(n) for n in observe])
    Q = jax_generator(s, p, absorbing=None)
    x0 = jnp.zeros(s.n).at[s.idx(start)].add(1.0)
    occ = _expm_action(Q, x0, jnp.asarray(times, float))
    return occ[:, oidx].sum(axis=1)


def jax_first_passage_from(s: Structure, p: dict, times, start: str, target):
    """First-passage cumulative incidence to ``target`` given the cohort started
    in ``start`` (target absorbing)."""
    tidx = jnp.array([s.idx(n) for n in target])
    Q = jax_generator(s, p, absorbing=frozenset(target))
    x0 = jnp.zeros(s.n).at[s.idx(start)].add(1.0)
    occ = _expm_action(Q, x0, jnp.asarray(times, float))
    return occ[:, tidx].sum(axis=1)


# -- closed-form first passage for the single-latent time-varying structures --
def cumhaz_K(p: dict, t):
    """integral_0^t x1 * max(1, s)^x2 ds."""
    x1, x2 = p["x1"], p["x2"]
    # for t <= 1: x1*t ; for t > 1: x1*(1 + (t^(x2+1) - 1)/(x2+1)).
    # Use a safe (>=1) base in the upper branch so its value and gradient are
    # finite even where the branch is unused (t=0 would give 0**negative = inf,
    # and jnp.where would then propagate a NaN gradient).
    t_safe = jnp.maximum(t, 1.0)
    below = x1 * t
    above = x1 * (1.0 + (t_safe ** (x2 + 1.0) - 1.0) / (x2 + 1.0))
    return jnp.where(t <= 1.0, below, above)


def cumhaz_L(p: dict, t):
    """integral_0^t x1 * (x2 + exp(-x3 s)) ds = x1*(x2*t + (1-exp(-x3 t))/x3)."""
    x1, x2, x3 = p["x1"], p["x2"], p["x3"]
    return x1 * (x2 * t + (1.0 - jnp.exp(-x3 * t)) / x3)


def jax_first_passage_KL(name: str, p: dict, times):
    cum = cumhaz_K(p, times) if name == "K" else cumhaz_L(p, times)
    return 1.0 - jnp.exp(-cum)
