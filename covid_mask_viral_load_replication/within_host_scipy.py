"""Independent scipy implementation of the within-host model.

This is the cross-check for ``within_host.camdl``: the same six equations
(Goyal et al. 2021, Supplementary methods, eq. 1), integrated with LSODA as
the authors do in R (``deSolve::lsodar``).  It is not used to produce the
results; ``compare_integrators.py`` runs it against the camdl output.

    dS/dt  = -beta V S
    dI/dt  =  beta V S - delta I^k I - m E^r / (E^r + phi^r) I
    dV/dt  =  pi I - c V
    dM1/dt =  omega I M1 - q M1
    dM2/dt =  q (M1 - M2)
    dE/dt  =  q M2 - delta_E E

An antiviral of efficacy ``eps`` started ``t_treat`` days after exposure
multiplies ``pi`` by ``1 - eps`` from then on.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

S0, I0, M1_0 = 1.0e7, 1.0, 1.0


def rhs(t, x, beta, delta, k, pi, m, omega, phi, r, q, delta_e, c):
    S, I, V, M1, M2, E = x
    Ipos = max(I, 0.0)
    kill = m * E**r / (E**r + phi**r)
    infection = beta * V * S
    return [
        -infection,
        infection - delta * Ipos**k * I - kill * I,
        pi * I - c * V,
        omega * I * M1 - q * M1,
        q * (M1 - M2),
        q * M2 - delta_e * E,
    ]


def viral_load(par: dict, days: int = 20, tau: float = 0.5, eps: float = 0.0,
               t_treat: float = np.inf, rtol: float = 1e-8, atol: float = 1e-8) -> np.ndarray:
    """V at tau, tau+1, ..., tau+days (days since exposure) for one person."""
    args = lambda pi: (par["beta"], par["delta"], par["k"], pi, par["m"], par["w"],
                       par["E50"], par["r"], par["q"], par["de"], par["c"])
    x0 = [S0, I0, par["p"] * I0 / par["c"], M1_0, 0.0, 0.0]
    t_eval = tau + np.arange(days + 1.0)
    t_end = t_eval[-1]
    if not (tau < t_treat < t_end) or eps == 0.0:
        pi = par["p"] * (1.0 - eps) if t_treat <= tau else par["p"]
        sol = solve_ivp(rhs, (tau, t_end), x0, method="LSODA", t_eval=t_eval,
                        args=args(pi), rtol=rtol, atol=atol)
        return sol.y[2]
    before = t_eval[t_eval <= t_treat]
    after = t_eval[t_eval > t_treat]
    a = solve_ivp(rhs, (tau, t_treat), x0, method="LSODA", t_eval=before,
                  args=args(par["p"]), rtol=rtol, atol=atol, dense_output=True)
    xs = a.sol(t_treat)
    b = solve_ivp(rhs, (t_treat, t_end), xs, method="LSODA", t_eval=after,
                  args=args(par["p"] * (1.0 - eps)), rtol=rtol, atol=atol)
    return np.concatenate([a.y[2], b.y[2]])
