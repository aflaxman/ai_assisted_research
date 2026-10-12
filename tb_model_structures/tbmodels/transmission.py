"""Embed a natural-history structure in a population transmission model.

Each natural-history structure (its post-infection compartments and within-host
transitions, with parameters fixed at the cohort-calibration posterior) is given
a susceptible pool, a force of infection over the infectious compartments,
reinfection, passive case detection, and demography.

HIV is handled in reduced form (v1, PLAN.md decision 4/5): rather than
stratifying the state space, progression and TB-mortality rates carry a
population-average multiplier that rises with HIV prevalence and falls with ART
coverage, both supplied as time-varying inputs from GBD / UNAIDS.  Full HIV and
strain stratification is a generic extension (decision 6): the same machinery
takes a list of strata with per-stratum multipliers; v1 runs a single stratum
with the HIV multiplier folded in, and applies the MDR split post hoc.

Structures K and L are excluded here: their progression hazard depends on time
since infection, which has no autonomous-ODE transmission form.  They appear
only in the cohort fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .core import Structure

# structures with time-since-infection hazards: cohort-only
TRANSMISSION_EXCLUDED = {"K", "L"}


@dataclass
class TransmissionInputs:
    """Time-varying country inputs (years as absolute calendar or model time)."""
    hiv_prevalence: Callable[[float], float] = lambda t: 0.0   # fraction 0-1
    art_coverage: Callable[[float], float] = lambda t: 0.0     # fraction 0-1
    detection_rate: Callable[[float], float] = lambda t: 0.0   # per year, passive
    birth_rate: float = 1.0 / 65.0     # per capita per year
    bg_mortality: float = 1.0 / 65.0   # per capita per year (non-TB)


@dataclass
class TransmissionParams:
    beta: float = 10.0            # transmission coefficient
    reinf_protection: float = 0.79  # protection against reinfection (Andrews 2012)
    hiv_prog_rr: float = 20.0     # progression rate ratio for untreated HIV+
    hiv_mort_rr: float = 2.0      # TB-mortality rate ratio for untreated HIV+
    art_efficacy: float = 0.65    # fractional reduction of the HIV effect on ART


@dataclass
class TransmissionModel:
    nh: Structure
    nh_params: dict
    # derived at build
    living: list[str] = field(default_factory=list)
    idx: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.nh.name in TRANSMISSION_EXCLUDED:
            raise ValueError(f"{self.nh.name}: excluded from transmission (time-since-infection hazard)")
        # living post-infection compartments (exclude the NH susceptible and dead)
        self.living = [c.name for c in self.nh.compartments
                       if c.kind not in ("susceptible", "dead")]
        order = ["S"] + self.living
        self.idx = {name: i for i, name in enumerate(order)}
        self.n = len(order)
        self._comp = {c.name: c for c in self.nh.compartments}
        # classify within-host transitions (exclude infection from S and mortality)
        self._within, self._mort = [], []
        for tr in self.nh.transitions:
            if tr.src == "S":
                continue
            if self._comp[tr.dst].kind == "dead":
                self._mort.append(tr)
            else:
                self._within.append(tr)
        # entry distribution for (re)infection
        self._entry = self.nh.entry(self.nh_params)
        # infectious compartments and their relative infectiousness
        self._inf = [(c.name, (self.nh_params[c.rel_inf] if isinstance(c.rel_inf, str) else c.rel_inf))
                     for c in self.nh.compartments if c.infectious]
        # states available for passive detection (symptomatic disease)
        self._symptomatic = [c.name for c in self.nh.compartments if c.symptomatic]
        self._bactpos = [c.name for c in self.nh.compartments if c.bact_positive]
        # reinfection-eligible compartments (latent / cleared / recovered)
        self._reinf_src = [c.name for c in self.nh.compartments
                           if c.kind in ("latent", "cleared", "recovered")]

    # -- helpers ---------------------------------------------------------
    def initial_state(self, N: float = 1e5, seed_prev: float = 1e-3) -> np.ndarray:
        x = np.zeros(self.n)  # living states only; simulate() appends the counters
        x[self.idx["S"]] = N * (1 - seed_prev)
        # seed a little infectious disease
        seed_comp = "Clin" if "Clin" in self.idx else ("I" if "I" in self.idx else self.living[-1])
        x[self.idx[seed_comp]] = N * seed_prev
        return x

    def hiv_multipliers(self, t: float, inp: TransmissionInputs, tp: TransmissionParams):
        h = inp.hiv_prevalence(t)
        art = inp.art_coverage(t)
        # fraction of the HIV effect still active after ART
        active = 1.0 - art * tp.art_efficacy
        prog_mult = 1.0 + h * active * (tp.hiv_prog_rr - 1.0)
        mort_mult = 1.0 + h * active * (tp.hiv_mort_rr - 1.0)
        return prog_mult, mort_mult

    def rhs(self, t, y, inp: TransmissionInputs, tp: TransmissionParams,
            screening_rate: Callable[[float], float] | None = None):
        x = y[: self.n]
        N = x.sum()
        if N <= 0:
            return np.zeros_like(y)
        lam = tp.beta * sum(ri * x[self.idx[name]] for name, ri in self._inf) / N
        prog_mult, mort_mult = self.hiv_multipliers(t, inp, tp)
        delta = inp.detection_rate(t)
        screen = screening_rate(t) if screening_rate is not None else 0.0

        dx = np.zeros(self.n)
        d_inc = d_notif = d_death = 0.0
        p = self.nh_params

        # (re)infection of susceptibles and reinfection-eligible compartments
        foi_S = lam
        for name, prob in self._entry.items():
            flow = foi_S * x[self.idx["S"]] * prob
            dx[self.idx["S"]] -= flow
            dx[self.idx[name]] += flow
            if self._comp[name].kind == "disease":
                d_inc += flow
        for src in self._reinf_src:
            foi_r = lam * (1 - tp.reinf_protection)
            tot = foi_r * x[self.idx[src]]
            dx[self.idx[src]] -= tot
            for name, prob in self._entry.items():
                dx[self.idx[name]] += tot * prob
                if self._comp[name].kind == "disease":
                    d_inc += tot * prob

        # within-host transitions (progression carries the HIV multiplier)
        for tr in self._within:
            r = tr.rate(p, 0.0, t)
            if self._comp[tr.src].kind == "latent" and self._comp[tr.dst].kind == "disease":
                r = r * prog_mult
                is_inc = True
            else:
                is_inc = False
            flow = r * x[self.idx[tr.src]]
            dx[self.idx[tr.src]] -= flow
            dx[self.idx[tr.dst]] += flow
            if is_inc:
                d_inc += flow

        # TB mortality (HIV multiplier) -> leaves the living population
        for tr in self._mort:
            flow = tr.rate(p, 0.0, t) * mort_mult * x[self.idx[tr.src]]
            dx[self.idx[tr.src]] -= flow
            d_death += flow

        # passive case detection: symptomatic disease -> recovered
        rec = "Rec" if "Rec" in self.idx else None
        for name in self._symptomatic:
            flow = delta * x[self.idx[name]]
            dx[self.idx[name]] -= flow
            if rec:
                dx[self.idx[rec]] += flow
            d_notif += flow
        # active screening: all bacteriologically positive disease -> recovered
        if screen:
            for name in self._bactpos:
                flow = screen * x[self.idx[name]]
                dx[self.idx[name]] -= flow
                if rec:
                    dx[self.idx[rec]] += flow
                d_notif += flow

        # demography: births into S, background mortality from all living
        dx[self.idx["S"]] += inp.birth_rate * N
        dx -= inp.bg_mortality * x

        return np.concatenate([dx, [d_inc, d_notif, d_death]])

    def simulate(self, times, inp: TransmissionInputs, tp: TransmissionParams,
                 x0=None, screening_rate=None):
        times = np.asarray(times, dtype=float)
        if x0 is None:
            x0 = self.initial_state()
        y0 = np.concatenate([x0, [0.0, 0.0, 0.0]])
        sol = solve_ivp(
            lambda t, y: self.rhs(t, y, inp, tp, screening_rate),
            (times.min(), times.max()), y0, t_eval=times,
            method="LSODA", rtol=1e-7, atol=1e-7,
        )
        if not sol.success:
            raise RuntimeError(f"{self.nh.name}: transmission ODE failed: {sol.message}")
        return sol.y.T

    # -- observable outputs ---------------------------------------------
    def outputs(self, traj: np.ndarray, times: np.ndarray) -> dict:
        """Derive per-100k rates and the subclinical fraction from a trajectory."""
        living = traj[:, : self.n]
        N = living.sum(axis=1)
        prev_states = [self.idx[c.name] for c in self.nh.compartments if c.bact_positive]
        prevalence = living[:, prev_states].sum(axis=1) / N * 1e5
        # GBD LTBI = IGRA/TST positivity, which persists after infection clears or
        # resolves, so it maps to the ever-infected (immunoreactive) fraction = 1 - S/N,
        # not the live-infection compartment.
        ltbi = (1.0 - living[:, self.idx["S"]] / N)
        # annual incidence / mortality from the cumulative counters (diff per year)
        cum_inc, cum_notif, cum_death = traj[:, self.n], traj[:, self.n + 1], traj[:, self.n + 2]
        inc_rate = np.gradient(cum_inc, times) / N * 1e5
        death_rate = np.gradient(cum_death, times) / N * 1e5
        # subclinical fraction among bacteriologically positive
        sub_states = [self.idx[c.name] for c in self.nh.compartments
                      if getattr(c, "band", None) == "subclinical"]
        if sub_states and prev_states:
            subfrac = living[:, sub_states].sum(axis=1) / living[:, prev_states].sum(axis=1)
        else:
            subfrac = np.full_like(N, np.nan)
        return dict(N=N, prevalence=prevalence, ltbi=ltbi, incidence=inc_rate,
                    mortality=death_rate, subclinical_fraction=subfrac)
