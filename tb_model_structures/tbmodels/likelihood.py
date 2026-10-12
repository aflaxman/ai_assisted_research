"""The three-block cohort likelihood (PLAN.md section 3; HTML explainer in
docs/likelihood_blocks.html).

Block (i)  converter cohorts: infection -> disease by time since infection.
           Both families.  Classic target = the single disease state I; spectrum
           target = any disease for the minimal-band (radiographic) cohorts and
           the bacteriologically positive states for the infectious-band cohorts.
Block (ii) duration of infectious disease and (classic only) case fatality.
Block (iii) Min<->Sub<->Clin transitions.  Spectrum only.

Each block returns a scalar JAX log-likelihood given a dict of jnp-scalar
parameters, so the whole thing is differentiable for NUTS.  The reference
quantities are the first-passage and occupancy functions in tbmodels.jaxcore,
validated against the scipy engine.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from . import jaxcore as J
from .core import Structure

# ---- binomial log-pmf (events out of n at probability pr) ------------------
def _binom_loglik(events, n, pr):
    pr = jnp.clip(pr, 1e-9, 1 - 1e-9)
    return jnp.sum(events * jnp.log(pr) + (n - events) * jnp.log1p(-pr))


# ===========================================================================
# Block (i): converter cohorts
# ===========================================================================
def converter_target(s: Structure, band: str) -> tuple[str, ...]:
    """Which compartments count as 'progressed' for a converter cohort of the
    given band, under this structure's family."""
    if s.family == "classic":
        return ("I",)
    if band == "minimal":
        # radiographic disease onset = any disease state
        return tuple(c.name for c in s.compartments if c.kind == "disease")
    # infectious cohort: bacteriologically positive states
    return tuple(c.name for c in s.compartments if getattr(c, "bact_positive", False))


def prep_converter(df) -> list[dict]:
    """Group the converter rows into per-study arrays of (year, events, at_risk)."""
    out = []
    for (study, band), g in df.groupby(["study", "band"]):
        g = g.sort_values("year")
        out.append(dict(
            study=study, band=band,
            year=g["year"].to_numpy(dtype=float),
            events=g["events"].to_numpy(dtype=float),
            at_risk=g["at_risk"].to_numpy(dtype=float),
        ))
    return out


def loglik_converters(s: Structure, p: dict, cohorts: list[dict]) -> jnp.ndarray:
    """Interval-binomial likelihood: in year [y-1, y], an at-risk individual
    progresses with probability [S(y-1)-S(y)]/S(y-1), S = 1 - cumulative
    incidence to the band's target."""
    ll = 0.0
    for co in cohorts:
        target = converter_target(s, co["band"])
        yrs = co["year"]
        tgrid = jnp.asarray(np.concatenate([[0.0], yrs]))
        if s.name in ("K", "L"):
            ci = J.jax_first_passage_KL(s.name, p, tgrid)
        else:
            ci = J.jax_first_passage(s, p, tgrid, target)
        surv = 1.0 - ci
        # conditional progression probability within each interval
        q = (surv[:-1] - surv[1:]) / jnp.clip(surv[:-1], 1e-9, 1.0)
        ll = ll + _binom_loglik(jnp.asarray(co["events"]), jnp.asarray(co["at_risk"]), q)
    return ll


# ===========================================================================
# Block (ii): duration of infectious disease, and (classic) case fatality
# ===========================================================================
def _infectious_states(s: Structure) -> list[str]:
    return [c.name for c in s.compartments if c.infectious]


def duration_infectious(s: Structure, p: dict, start: str | None = None) -> jnp.ndarray:
    """Expected time spent in the infectious set before leaving it permanently
    (to minimal/recovered/dead), starting from ``start`` (default: the first
    infectious state reached, i.e. Sub, or I for classic)."""
    inf = _infectious_states(s)
    idx = {name: k for k, name in enumerate(inf)}
    m = len(inf)
    T = jnp.zeros((m, m))
    for tr in s.transitions:
        r = tr.rate(p, 0.0, 0.0)
        if tr.src in idx:
            i = idx[tr.src]
            T = T.at[i, i].add(-r)  # all outflow reduces sojourn
            if tr.dst in idx:
                T = T.at[idx[tr.dst], i].add(r)  # stays within the infectious set
    sojourn = jnp.linalg.solve(-T, jnp.ones(m))  # expected time per start state
    start = start or inf[0]
    return sojourn[idx[start]]


def case_fatality_classic(s: Structure, p: dict) -> jnp.ndarray:
    """Fraction of active-disease episodes ending in TB death (classic I)."""
    return p["tb_death"] / (p["tb_death"] + p["self_cure"])


def loglik_duration(s: Structure, p: dict, target_years=2.0, sd=0.5) -> jnp.ndarray:
    dur = duration_infectious(s, p)
    return norm.logpdf(target_years, loc=dur, scale=sd)


def loglik_case_fatality(s: Structure, p: dict, target=0.43, sd=0.10) -> jnp.ndarray:
    """Soft target on untreated case fatality (Tiemersma 2011).  Classic only;
    pins self_cure given tb_death."""
    cf = case_fatality_classic(s, p)
    return norm.logpdf(target, loc=cf, scale=sd)


# ===========================================================================
# Block (iii): Min <-> Sub <-> Clin transitions (spectrum only)
# ===========================================================================
# Map the Transition column to (start, observe/target, kind).
# kind "occ" = cross-sectional occupancy; kind "fp" = first-passage cumulative.
# "infectious" target expands to the bacteriologically positive states.
_TRANSITION_MAP = {
    "Submin":        ("Min", ("Sub",),  "occ"),
    "Minsub":        ("Sub", ("Min",),  "occ"),
    "Clinmin":       ("Min", ("Clin",), "occ"),
    "Minclin":       ("Clin", ("Min",), "occ"),
    "Submin_sub":    ("Min", ("Sub",),  "fp"),
    "Clinmin_clin":  ("Min", ("Clin",), "fp"),
    "Minclin_min":   ("Clin", ("Min",), "fp"),
    "InfminI":       ("Min", "INF",     "occ"),
    "MininfI":       ("INF", ("Min",),  "occ"),
    "Infmin_infI":   ("Min", "INF",     "fp"),
    "Mininf_minI":   ("INF", ("Min",),  "fp"),
}


# Richards/Horton (results_data_prep.R / InfDis_DataPrep.R) apply a 3/4
# interval-censoring factor to the denominators of these observables, because a
# case first seen positive at a survey was positive for only part of the
# interval.  We reproduce it so the weighting matches the source models.
_THREE_QUARTER = {"Submin", "Clinmin", "Clinmin_clin", "Submin_sub", "InfminI", "Infmin_infI"}


def prep_transitions(df) -> list[dict]:
    """Group the spectrum transition rows by observable into arrays."""
    out = []
    for trans, g in df.groupby("Transition"):
        if trans not in _TRANSITION_MAP:
            continue
        g = g.sort_values("years")
        denom = g["effective.d"].to_numpy(dtype=float)
        if trans in _THREE_QUARTER:
            denom = np.round(denom * 0.75)
        events = np.minimum(g["effective.n"].to_numpy(dtype=float), denom)
        out.append(dict(
            trans=trans,
            years=g["years"].to_numpy(dtype=float),
            events=events,
            denom=denom,
        ))
    return out


def loglik_transitions(s: Structure, p: dict, groups: list[dict]) -> jnp.ndarray:
    """Binomial likelihood for the Min/Sub/Clin transition observations.

    For observables whose start is 'INF' (started infectious), the infectious
    initial state is split Sub:Clin as phi:(1-phi), matching the source models.
    """
    bact = tuple(c.name for c in s.compartments if getattr(c, "bact_positive", False))
    ll = 0.0
    for grp in groups:
        start, observe, kind = _TRANSITION_MAP[grp["trans"]]
        obs = bact if observe == "INF" else observe
        times = jnp.asarray(grp["years"])
        if start == "INF":
            # start split phi Sub, (1-phi) Clin
            pr = p["phi"] * _traj(s, p, times, "Sub", obs, kind) \
                + (1 - p["phi"]) * _traj(s, p, times, "Clin", obs, kind)
        else:
            pr = _traj(s, p, times, start, obs, kind)
        ll = ll + _binom_loglik(jnp.asarray(grp["events"]), jnp.asarray(grp["denom"]), pr)
    return ll


def _traj(s, p, times, start, observe, kind):
    if kind == "occ":
        return J.jax_occupancy_from(s, p, times, start, observe)
    return J.jax_first_passage_from(s, p, times, start, observe)
