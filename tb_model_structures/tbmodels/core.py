"""Declarative compartmental TB structures with a shared ODE interface.

A :class:`Structure` is a list of :class:`Compartment` objects and a list of
:class:`Transition` objects.  The same object is used two ways:

* **cohort mode** (force of infection ``lam = 0``): follow a cohort from the
  instant of infection.  The system is linear, so for time-constant rates the
  transition probabilities are ``expm(Q * t)`` applied to the initial
  distribution; for time-varying rates (structures K and L) we integrate the
  linear ODE.  Used to fit natural-history cohort data.

* **transmission mode**: the same transition list is integrated with
  ``solve_ivp``; the force of infection is recomputed from the infectious
  compartments at each step, so the system is nonlinear.

Adding a structure means adding one :class:`Structure` to the registry; nothing
in this module hard-codes any particular model.  See ``README`` /
``docs/adding_a_structure`` for the worked example.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# A rate is a function of (params, force_of_infection, time) -> float.
RateFn = Callable[[dict, float, float], float]

# Compartment kinds.  "counter" compartments are cumulative-incidence
# accumulators: they only receive inflow and never lose mass, so they do not
# represent people and are excluded from mass-conservation checks.
KINDS = {
    "susceptible",
    "latent",
    "disease",
    "cleared",
    "recovered",
    "dead",
    "counter",
}

# Spectrum disease bands, ordered from least to most advanced.
BANDS = ("minimal", "subclinical", "clinical")


@dataclass(frozen=True)
class Compartment:
    name: str
    kind: str
    band: str | None = None  # one of BANDS for spectrum disease states
    infectious: bool = False
    rel_inf: float | str = 0.0  # relative infectiousness; str => a parameter name
    symptomatic: bool = False
    bact_positive: bool = False  # detectable by a sputum-based test (Xpert/smear/culture)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.band is not None and self.band not in BANDS:
            raise ValueError(f"{self.name}: unknown band {self.band!r}")


def const(name: str) -> RateFn:
    """A rate equal to a single named parameter."""
    return lambda p, lam, t: p[name]


def foi(coeff: float = 1.0) -> RateFn:
    """A rate proportional to the force of infection (used for S -> ... entry)."""
    return lambda p, lam, t: coeff * lam


@dataclass
class Transition:
    src: str
    dst: str
    rate: RateFn
    label: str = ""


@dataclass
class Structure:
    name: str
    family: str  # "classic" or "spectrum"
    compartments: Sequence[Compartment]
    transitions: Sequence[Transition]
    param_names: Sequence[str]
    # entry(p) -> {compartment_name: probability}: where a newly infected cohort
    # lands at t=0 (cohort mode).  Must sum to 1.
    entry: Callable[[dict], dict]
    # Compartments that count as "active TB disease" for the classic cumulative
    # incidence curve (Menzies Figure 3/5).  For spectrum models this is the set
    # of bacteriologically positive states.
    disease_states: Sequence[str] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        self._index = {c.name: i for i, c in enumerate(self.compartments)}
        names = set(self._index)
        for tr in self.transitions:
            if tr.src not in names:
                raise ValueError(f"{self.name}: transition from unknown {tr.src!r}")
            if tr.dst not in names:
                raise ValueError(f"{self.name}: transition to unknown {tr.dst!r}")
        if not self.disease_states:
            self.disease_states = tuple(c.name for c in self.compartments if c.kind == "disease")

    # -- indexing --------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.compartments)

    def idx(self, name: str) -> int:
        return self._index[name]

    def names(self) -> list[str]:
        return [c.name for c in self.compartments]

    def mass_compartments(self) -> list[int]:
        """Indices of compartments that hold people (everything but counters)."""
        return [i for i, c in enumerate(self.compartments) if c.kind != "counter"]

    # -- generator matrix ------------------------------------------------
    def generator(
        self, p: dict, lam: float, t: float, absorbing: set[str] | None = None
    ) -> np.ndarray:
        """Generator Q with dx/dt = Q @ x.

        Counter compartments receive inflow but have no outflow, so they do not
        conserve mass by design.  If ``absorbing`` is given, transitions whose
        source is in that set are dropped, making those compartments absorbing;
        their occupancy is then the first-passage ("ever reached") fraction.
        """
        n = self.n
        Q = np.zeros((n, n))
        counter = {i for i, c in enumerate(self.compartments) if c.kind == "counter"}
        for tr in self.transitions:
            if absorbing is not None and tr.src in absorbing:
                continue
            r = tr.rate(p, lam, t)
            i, j = self._index[tr.src], self._index[tr.dst]
            Q[j, i] += r
            if i not in counter:
                Q[i, i] -= r
        return Q

    # -- cohort initial condition ---------------------------------------
    def initial_cohort(self, p: dict) -> np.ndarray:
        x = np.zeros(self.n)
        dist = self.entry(p)
        total = sum(dist.values())
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"{self.name}: entry distribution sums to {total}, not 1")
        for name, prob in dist.items():
            x[self._index[name]] = prob
        return x

    # -- cohort simulation ----------------------------------------------
    def is_time_varying(self, p: dict) -> bool:
        """True if any rate depends on t (checked at two sample times)."""
        for tr in self.transitions:
            if tr.rate(p, 0.0, 0.0) != tr.rate(p, 0.0, 1.7):
                return True
        return False

    def simulate_cohort(
        self,
        p: dict,
        times: np.ndarray,
        method: str = "auto",
    ) -> np.ndarray:
        """Return an array of shape (len(times), n) of cohort occupancy.

        method: "expm" (constant rates only), "ode" (solve_ivp), or "auto".
        """
        times = np.asarray(times, dtype=float)
        x0 = self.initial_cohort(p)
        if method == "auto":
            method = "ode" if self.is_time_varying(p) else "expm"
        if method == "expm":
            Q = self.generator(p, 0.0, 0.0)
            out = np.empty((len(times), self.n))
            for k, t in enumerate(times):
                out[k] = expm(Q * t) @ x0
            return out
        # time-varying linear ODE
        def deriv(t, x):
            return self.generator(p, 0.0, t) @ x
        sol = solve_ivp(
            deriv, (0.0, float(times.max())), x0, t_eval=times,
            method="LSODA", rtol=1e-9, atol=1e-12, max_step=0.25,
        )
        if not sol.success:
            raise RuntimeError(f"{self.name}: cohort ODE failed: {sol.message}")
        return sol.y.T

    def cumulative_incidence(
        self,
        p: dict,
        times: np.ndarray,
        target: Sequence[str] | None = None,
    ) -> np.ndarray:
        """First-passage cumulative fraction: the share of the cohort that has
        **ever** reached any ``target`` compartment by each time.

        Computed by making the target compartments absorbing (dropping their
        outflows) and reading their total occupancy, which counts each cohort
        member at most once even when the model allows regression back out of
        disease.  target defaults to ``disease_states``.
        """
        target = list(target) if target is not None else list(self.disease_states)
        tset = set(target)
        tidx = [self._index[name] for name in target]
        times = np.asarray(times, dtype=float)
        x0 = self.initial_cohort(p)

        if self.is_time_varying(p):
            def deriv(t, x):
                return self.generator(p, 0.0, t, absorbing=tset) @ x
            sol = solve_ivp(
                deriv, (0.0, float(times.max())), x0, t_eval=times,
                method="LSODA", rtol=1e-10, atol=1e-13, max_step=0.25,
            )
            if not sol.success:
                raise RuntimeError(f"{self.name}: cumulative-incidence ODE failed: {sol.message}")
            occ = sol.y.T
        else:
            Q = self.generator(p, 0.0, 0.0, absorbing=tset)
            occ = np.array([expm(Q * t) @ x0 for t in times])
        return occ[:, tidx].sum(axis=1)

    # -- transmission simulation ----------------------------------------
    def force_of_infection(self, p: dict, x: np.ndarray, beta: float, N: float) -> float:
        """beta * sum(rel_inf_i * x_i) / N over infectious compartments."""
        s = 0.0
        for i, c in enumerate(self.compartments):
            if c.infectious:
                ri = p[c.rel_inf] if isinstance(c.rel_inf, str) else c.rel_inf
                s += ri * x[i]
        return beta * s / N

    def simulate_transmission(
        self,
        p: dict,
        times: np.ndarray,
        x0: np.ndarray,
        beta: float,
        extra_deriv: Callable[[float, np.ndarray, float], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Integrate the transmission model (nonlinear via FOI).

        ``extra_deriv(t, x, lam)`` may add demographic/intervention flows
        (births, background death, screening); it returns an increment to dx.
        """
        times = np.asarray(times, dtype=float)
        N0 = x0[self.mass_compartments()].sum()

        def deriv(t, x):
            N = x[self.mass_compartments()].sum()
            lam = self.force_of_infection(p, x, beta, N if N > 0 else N0)
            dx = self.generator(p, lam, t) @ x
            if extra_deriv is not None:
                dx = dx + extra_deriv(t, x, lam)
            return dx

        sol = solve_ivp(
            deriv, (float(times.min()), float(times.max())), x0, t_eval=times,
            method="LSODA", rtol=1e-8, atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(f"{self.name}: transmission ODE failed: {sol.message}")
        return sol.y.T
