"""The deterministic SWIRM model and its bifurcation.

SWIRM extends SIR with two ideas from the paper:

* infected people split into **W** (unreported) and **I** (reported), with a
  fraction ``f`` of new infections reported, and
* a **social-media** compartment **M** that *reads* the epidemic (posts and
  engagement scale with the compartments) but never feeds back into it.

Only the reported class ``I`` transmits: the force of infection is the
mass-action term ``beta * S * I`` (no ``/N``). Emergence is forced by letting
the transmission rate rise linearly in time, ``beta(t) = beta0 + r * t``, which
carries the basic reproduction number R0(t) up through 1.

All parameter values come from Table 1 of the paper; the R0 formula and the
t = 14.5 day bifurcation are verified in ``tests/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Order of the state vector used everywhere in this package.
STATE_NAMES = ("S", "W", "I", "R", "M")
S, W, I, R, M = range(5)  # noqa: E741 - single-letter names match the paper


@dataclass(frozen=True)
class Params:
    """SWIRM parameters (Table 1 of the paper).

    Media posting (``delta``) and engagement (``epsilon``) are given as single
    values in Table 1; we apply them to every compartment. Because M is
    one-directionally coupled, these choices never affect the epidemic itself.
    """

    # Demographic
    pi: float = 14.28        # birth rate (chosen so S* = pi/mu = 1000)
    mu: float = 0.01428      # natural death rate

    # Disease
    f: float = 0.8           # fraction of new infections that are reported
    gamma_W: float = 0.07    # recovery rate, unreported
    gamma_I: float = 0.024   # recovery rate, reported
    phi: float = 0.01        # waning-immunity rate
    alpha: float = 0.033     # progression W -> I

    # Social media
    delta: float = 0.5       # posting rate (per individual)
    epsilon: float = 0.25    # engagement rate (per individual, scales with M)
    mu_bar: float = 0.1428   # "fickle" rate: posts fade, quadratically

    # Time-varying transmission beta(t) = beta0 + r * t
    beta0: float = 5e-6
    r: float = 2.739e-6

    def beta(self, t: float | np.ndarray) -> float | np.ndarray:
        """Transmission rate at time ``t`` (days)."""
        return self.beta0 + self.r * t

    @property
    def S_star(self) -> float:
        """Susceptibles at the disease-free equilibrium, pi / mu."""
        return self.pi / self.mu

    @property
    def M_star(self) -> float:
        """Media posts at the disease-free equilibrium.

        With only S present, ``mu_bar M^2 - epsilon S* M - delta S* = 0``; we
        take the positive root.
        """
        a, b, c = self.mu_bar, -self.epsilon * self.S_star, -self.delta * self.S_star
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    def R0(self, t: float | np.ndarray) -> float | np.ndarray:
        """Basic reproduction number R0(t) (next-generation matrix).

        Verified against the S1 appendix:
        ``R0 = [alpha(1-f) + f(gamma_W+alpha+mu)] * beta S* / [(gamma_W+alpha+mu)(gamma_I+mu)]``.
        """
        b = self.beta(t) * self.S_star
        num = self.alpha * (1 - self.f) * b + self.f * b * (self.gamma_W + self.alpha + self.mu)
        den = (self.gamma_W + self.alpha + self.mu) * (self.gamma_I + self.mu)
        return num / den

    def bifurcation_time(self) -> float:
        """Time at which R0(t) = 1 (the transcritical bifurcation).

        R0 is linear in beta and beta is linear in t, so we solve directly.
        """
        # R0(t) = k * beta(t); find beta with R0 = 1, then invert beta(t).
        k = self.R0(0.0) / self.beta(0.0)
        beta_crit = 1.0 / k
        return (beta_crit - self.beta0) / self.r


def deterministic_rhs(state: np.ndarray, t: float, p: Params) -> np.ndarray:
    """Right-hand side of the deterministic SWIRM system.

    ``state`` may be a single 5-vector or an ``(n, 5)`` array of realizations;
    the same code serves both, which keeps the stochastic integrator simple.
    """
    state = np.asarray(state, dtype=float)
    s, w, i, r, m = (state[..., k] for k in range(5))
    beta = p.beta(t)
    infections = beta * s * i
    total = s + w + i + r

    d = np.empty_like(state)
    d[..., S] = p.pi - infections - p.mu * s + p.phi * r
    d[..., W] = (1 - p.f) * infections - (p.gamma_W + p.alpha + p.mu) * w
    d[..., I] = p.f * infections + p.alpha * w - (p.gamma_I + p.mu) * i
    d[..., R] = p.gamma_W * w + p.gamma_I * i - (p.mu + p.phi) * r
    d[..., M] = p.delta * total + p.epsilon * total * m - p.mu_bar * m * m
    return d


def initial_state(p: Params, seed_infected: float = 1.0) -> np.ndarray:
    """Disease-free equilibrium with a small reported-infection seed."""
    x = np.array([p.S_star, 0.0, seed_infected, 0.0, p.M_star], dtype=float)
    return x
