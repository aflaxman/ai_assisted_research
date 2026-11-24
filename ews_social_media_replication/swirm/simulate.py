"""Euler-Maruyama simulation of the three stochastic SWIRM models.

The paper drives the deterministic skeleton with three kinds of noise:

* **additive** white noise -- the same intensity ``sigma`` added to every equation;
* **multiplicative** white noise -- intensity ``sigma * X_i`` (scales with the state);
* **demographic** stochasticity -- the chemical-Langevin form of the reaction
  table in S1 (Table 3); its size is fixed by the rates, so there is no free knob.

Every integrator is vectorized across the 300 realizations at once, and negative
states are rejected by reverting that realization to its previous step, exactly
as the paper describes.
"""

from __future__ import annotations

import numpy as np

from .model import Params, deterministic_rhs, initial_state


def _steps(t_max: float, dt: float) -> np.ndarray:
    n = int(round(t_max / dt))
    return np.arange(n + 1) * dt


# --- demographic stochasticity: the 13 reactions of S1 Table 3 -----------------
#
# Each row is (indices affected, stoichiometry, propensity). Coupled transitions
# (e.g. R -> S) touch two compartments with a shared random increment, which
# preserves the correlations the paper's formulation builds in.
def _demographic_propensities(x: np.ndarray, p: Params, t: float) -> np.ndarray:
    """Return an ``(n, 13)`` array of reaction propensities."""
    s, w, i, r, m = (x[..., k] for k in range(5))
    beta = p.beta(t)
    inf = beta * s * i
    total = s + w + i + r
    media_birth = p.delta * total + p.epsilon * total * m
    a = np.stack([
        np.full_like(s, p.pi),        # 0  birth        -> S+
        p.phi * r,                    # 1  waning   R-> S
        (1 - p.f) * inf,              # 2  infection S-> W
        p.f * inf,                    # 3  infection S-> I
        p.mu * s,                     # 4  death S
        p.gamma_W * w,                # 5  recovery W-> R
        p.alpha * w,                  # 6  progression W-> I
        p.mu * w,                     # 7  death W
        p.gamma_I * i,                # 8  recovery I-> R
        p.mu * i,                     # 9  death I
        p.mu * r,                     # 10 death R
        media_birth,                  # 11 media birth  -> M+
        p.mu_bar * m * m,             # 12 media death  -> M-
    ], axis=-1)
    return np.maximum(a, 0.0)


# stoichiometry[j] = list of (state_index, +/-1) touched by reaction j
_DEMOG_STOICH = [
    [(0, +1)],           # 0 birth
    [(3, -1), (0, +1)],  # 1 waning R->S
    [(0, -1), (1, +1)],  # 2 infection S->W
    [(0, -1), (2, +1)],  # 3 infection S->I
    [(0, -1)],           # 4 death S
    [(1, -1), (3, +1)],  # 5 recovery W->R
    [(1, -1), (2, +1)],  # 6 progression W->I
    [(1, -1)],           # 7 death W
    [(2, -1), (3, +1)],  # 8 recovery I->R
    [(2, -1)],           # 9 death I
    [(3, -1)],           # 10 death R
    [(4, +1)],           # 11 media birth
    [(4, -1)],           # 12 media death
]


def _demographic_step(x, p, t, dt, rng):
    """One chemical-Langevin step: drift + sqrt(rate) * dW per reaction."""
    a = _demographic_propensities(x, p, t)              # (n, 13)
    dW = rng.standard_normal(a.shape) * np.sqrt(dt)     # (n, 13)
    increment = a * dt + np.sqrt(a) * dW                # (n, 13)
    dx = np.zeros_like(x)
    for j, touches in enumerate(_DEMOG_STOICH):
        for idx, sign in touches:
            dx[..., idx] += sign * increment[..., j]
    return dx


def simulate_ensemble(
    p: Params,
    noise_type: str,
    sigma: float = 0.0,
    n_real: int = 300,
    t_max: float = 150.0,
    dt: float = 0.005,
    seed: int = 0,
    seed_infected: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_real`` realizations of a stochastic SWIRM model.

    Parameters
    ----------
    noise_type : {"additive", "multiplicative", "demographic"}
    sigma : noise intensity for additive/multiplicative (ignored for demographic).

    Returns
    -------
    times : ``(n_steps,)`` array of time points (days).
    traj  : ``(n_real, n_steps, 5)`` array of states, order ``S, W, I, R, M``.
    """
    if noise_type not in {"additive", "multiplicative", "demographic"}:
        raise ValueError(f"unknown noise_type: {noise_type!r}")

    times = _steps(t_max, dt)
    n_steps = times.size
    rng = np.random.default_rng(seed)

    x = np.tile(initial_state(p, seed_infected), (n_real, 1))  # (n_real, 5)
    traj = np.empty((n_real, n_steps, 5), dtype=float)
    traj[:, 0, :] = x
    sqrt_dt = np.sqrt(dt)

    for k in range(1, n_steps):
        t = times[k - 1]
        drift = deterministic_rhs(x, t, p)
        if noise_type == "additive":
            dx = drift * dt + sigma * sqrt_dt * rng.standard_normal(x.shape)
        elif noise_type == "multiplicative":
            dx = drift * dt + sigma * x * sqrt_dt * rng.standard_normal(x.shape)
        else:  # demographic
            dx = _demographic_step(x, p, t, dt, rng)

        x_new = x + dx
        # Reject non-physical states: revert those realizations to the prior step.
        bad = np.any(x_new < 0, axis=1)
        x_new[bad] = x[bad]
        x = x_new
        traj[:, k, :] = x

    return times, traj
