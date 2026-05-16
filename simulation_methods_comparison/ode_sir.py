"""Lens #1: ODE (deterministic SIR).

Strengths:
- Fast: a full 120-day trajectory solves in milliseconds.
- Smooth: clean curves are easy to inspect and explain.
- Analytic insights: R0, herd-immunity threshold, final-size equation.

Weaknesses:
- Mean-field: every susceptible meets every infected, fractionally.
- Deterministic: with R0 > 1 an outbreak always happens.
- Hides queues, networks, individuals.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from scenario import Scenario, DEFAULT


def simulate(scn: Scenario = DEFAULT, vaccinated: bool = False) -> dict:
    """Run the SIR ODE and return time series plus summary stats."""
    immune_frac = 1.0 - scn.effective_susceptible_fraction(vaccinated)
    S0 = scn.N * (1.0 - immune_frac) - scn.initial_infected
    I0 = scn.initial_infected
    R0_state = scn.N - S0 - I0  # vaccinated people start in R

    def rhs(_t, y):
        S, I, _R = y
        new_inf = scn.beta * S * I / scn.N
        recov = scn.gamma * I
        return [-new_inf, new_inf - recov, recov]

    sol = solve_ivp(
        rhs,
        (0, scn.horizon_days),
        [S0, I0, R0_state],
        t_eval=np.linspace(0, scn.horizon_days, scn.horizon_days * 4 + 1),
        rtol=1e-8,
        atol=1e-8,
    )
    t, S, I, R = sol.t, sol.y[0], sol.y[1], sol.y[2]

    # New infections per day, for downstream models
    new_infections_per_day = scn.beta * S * I / scn.N

    return {
        "t": t,
        "S": S,
        "I": I,
        "R": R,
        "new_infections_per_day": new_infections_per_day,
        "peak_prevalence": float(I.max()),
        "peak_day": float(t[I.argmax()]),
        "total_infected": float(R[-1] - R0_state),  # exclude pre-immune
        "vaccinated": vaccinated,
    }


def final_size(R0: float, susceptible_frac: float = 1.0) -> float:
    """Analytic final-size equation: fraction of the initially susceptible
    population infected by the end of the epidemic. Useful as a sanity check."""
    if R0 * susceptible_frac <= 1.0:
        return 0.0
    # Solve 1 - r = exp(-R0 * s0 * r)
    f = lambda r: 1.0 - r - np.exp(-R0 * susceptible_frac * r)
    return brentq(f, 1e-9, 1.0 - 1e-9)
