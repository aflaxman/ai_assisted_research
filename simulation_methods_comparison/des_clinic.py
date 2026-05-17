"""Lens #3: Discrete Event Simulation (clinic queueing during the outbreak).

Strengths:
- Resources, queues, and waiting are first-class citizens.
- Captures congestion and capacity that smooth ODE curves cannot.
- Natural for any system where 'who is served next' matters.

Weaknesses:
- Slower than ODEs; many trajectories needed for stable averages.
- A DES is only as honest as its arrival process - so we feed it case arrivals
  derived from the SIR dynamics (a deliberate model hand-off, not a fudge).

This DES treats *case arrivals* as exogenous (driven by the ODE infection rate)
and focuses on what the ODE hides: the queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import simpy

from scenario import Scenario, DEFAULT
from ode_sir import simulate as ode_simulate


@dataclass
class ClinicRun:
    arrivals: List[float] = field(default_factory=list)         # time of arrival (days)
    waits_hours: List[float] = field(default_factory=list)      # time queued before service
    served: List[float] = field(default_factory=list)           # time of service start
    queue_samples: List[tuple[float, int]] = field(default_factory=list)
    # for daily aggregation
    arrivals_per_day: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _patient(env, name, clinic, run, service_rate_per_hr):
    arrival = env.now
    run.arrivals.append(arrival)
    with clinic.request() as req:
        yield req
        wait = (env.now - arrival) * 24.0  # convert days -> hours
        run.waits_hours.append(wait)
        run.served.append(env.now)
        service_time = np.random.exponential(1.0 / service_rate_per_hr) / 24.0
        yield env.timeout(service_time)


def _queue_monitor(env, clinic, run, sample_dt=0.05):
    while True:
        run.queue_samples.append((env.now, len(clinic.queue)))
        yield env.timeout(sample_dt)


def _arrival_process(env, clinic, run, daily_rate_fn, scn):
    """Inhomogeneous Poisson arrivals: thinning against the daily peak."""
    horizon = scn.horizon_days
    # Find a safe upper bound for thinning
    ts = np.linspace(0, horizon, 1000)
    lam_max = max(daily_rate_fn(t) for t in ts) + 1e-9
    n = 0
    while env.now < horizon:
        yield env.timeout(np.random.exponential(1.0 / lam_max))
        if env.now >= horizon:
            break
        if np.random.random() < daily_rate_fn(env.now) / lam_max:
            n += 1
            env.process(_patient(env, f"p{n}", clinic, run,
                                 service_rate_per_hr=60.0 / scn.service_minutes))


def simulate(scn: Scenario = DEFAULT, vaccinated: bool = False, seed: int = 0) -> ClinicRun:
    """Run one realization of the clinic DES, driven by SIR-derived arrivals."""
    np.random.seed(seed)
    ode = ode_simulate(scn, vaccinated=vaccinated)
    rate = ode["new_infections_per_day"] * scn.care_seeking_prob
    # Linear interp of the daily care-seeking rate
    t_grid, rate_grid = ode["t"], rate

    def daily_rate(t: float) -> float:
        return float(np.interp(t, t_grid, rate_grid))

    env = simpy.Environment()
    clinic = simpy.Resource(env, capacity=scn.clinic_servers)
    run = ClinicRun()
    env.process(_arrival_process(env, clinic, run, daily_rate, scn))
    env.process(_queue_monitor(env, clinic, run))
    env.run(until=scn.horizon_days)

    # Daily arrival counts for plotting alongside the ODE
    if run.arrivals:
        bins = np.arange(scn.horizon_days + 1)
        run.arrivals_per_day, _ = np.histogram(run.arrivals, bins=bins)
    else:
        run.arrivals_per_day = np.zeros(scn.horizon_days, dtype=int)
    return run


def summarize(run: ClinicRun) -> dict:
    if not run.waits_hours:
        return {"n_served": 0, "mean_wait_hr": 0.0, "p95_wait_hr": 0.0, "max_queue": 0}
    waits = np.array(run.waits_hours)
    queue = np.array([q for _t, q in run.queue_samples])
    return {
        "n_served": len(run.served),
        "n_arrivals": len(run.arrivals),
        "mean_wait_hr": float(waits.mean()),
        "p95_wait_hr": float(np.percentile(waits, 95)),
        "max_queue": int(queue.max()) if queue.size else 0,
    }
