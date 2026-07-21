"""Policy-informed behavioral-change stochastic epidemic model.

A faithful re-implementation of the framework described in

    Pathak, W., Pokharel, G., Hossain, S. (2026).
    "Modeling Infectious Disease Transmission Dynamics with Policy-Informed
    Behavioral Responses." In *Statistical Science: From Theory to Applied
    Research IV* (SIS-FENStatS 2026), pp. 293-299. Springer.
    https://doi.org/10.1007/978-3-032-30665-4_48

The chapter itself is paywalled (abstract only), so the concrete equations here
follow its open-access methodological backbone, reference [6] of the chapter:

    Ward, C., Deardon, R., Schmidt, A.M. (2023).
    "Bayesian modeling of dynamic behavioral change during an epidemic."
    Infectious Disease Modelling 8(4), 947-963.
    https://doi.org/10.1016/j.idm.2023.08.002   (arXiv:2211.00122)

We extend that alarm-modulated chain-binomial SIR in the direction the 2026
abstract states: the behavioral response becomes *policy-informed* -- a
stochastic, policy-dependent process with heterogeneous compliance and delayed
adaptation, layered on top of the endogenous (incidence-driven) alarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Alarm functions  f(x) -> [0, 1]   (Ward, Deardon & Schmidt 2023, sec. 2.2)
#
# x is m-day average incidence; the endogenous alarm rises as x grows.
# ---------------------------------------------------------------------------


def power_alarm(x: np.ndarray, k: float, N: int) -> np.ndarray:
    """Power alarm: f(x) = 1 - (1 - x/N)^(1/k), k > 0."""
    frac = np.clip(x / N, 0.0, 1.0)
    return 1.0 - (1.0 - frac) ** (1.0 / k)


def threshold_alarm(x: np.ndarray, H: float, delta: float) -> np.ndarray:
    """Threshold alarm: f(x) = delta * 1(x > H)."""
    return delta * (np.asarray(x, dtype=float) > H)


def hill_alarm(x: np.ndarray, delta: float, x0: float, nu: float) -> np.ndarray:
    """Modified Hill alarm: f(x) = delta / (1 + (x0/x)^nu)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = delta / (1.0 + (x0 / x[pos]) ** nu)
    return out


# ---------------------------------------------------------------------------
# Policy signal: the "policy-informed" extension of the 2026 chapter.
#
# Policies are (start, end, intensity) windows.  intensity in [0, 1] encodes
# the *type/strength* of the intervention (e.g. 0.4 = advisory, 0.9 = lockdown).
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    """A single time-limited intervention with a strength in [0, 1]."""

    start: int
    end: int
    intensity: float


def policy_signal(policies: list[Policy], n_days: int) -> np.ndarray:
    """Per-day policy intensity p(t) in [0, 1] (max over overlapping policies)."""
    p = np.zeros(n_days)
    for pol in policies:
        lo, hi = max(0, pol.start), min(n_days, pol.end)
        p[lo:hi] = np.maximum(p[lo:hi], pol.intensity)
    return p


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelParams:
    """Transmission, removal, and behavioral-response parameters."""

    beta: float = 0.6           # baseline transmission rate
    gamma: float = 0.2          # removal rate (1/gamma = mean infectious period)

    # Endogenous (incidence-driven) alarm -- Hill form by default.
    endog_delta: float = 0.6    # max endogenous alarm (asymptote)
    endog_x0: float = 200.0     # half-response incidence
    endog_nu: float = 3.0       # steepness
    m: int = 7                  # smoothing window for average incidence (days)

    # Policy-informed alarm.
    policy_weight: float = 0.7  # how strongly policy translates into alarm
    adapt_rate: float = 0.25    # rho: delayed adaptation (1 = instantaneous)
    compliance_sd: float = 0.05  # heterogeneous-compliance noise SD (sim only)


@dataclass
class Epidemic:
    """Realized trajectory of a simulated / observed epidemic."""

    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    new_infections: np.ndarray   # I*_t
    new_removals: np.ndarray     # R*_t
    alarm: np.ndarray            # realized a_t
    policy: np.ndarray           # policy signal p(t)
    N: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Behavioral response: combine endogenous + policy alarm with delayed adaptation
# ---------------------------------------------------------------------------


def endogenous_alarm_value(avg_incidence: float, p: ModelParams) -> float:
    """Endogenous alarm from smoothed incidence (Hill form)."""
    return float(hill_alarm(np.array([avg_incidence]), p.endog_delta, p.endog_x0, p.endog_nu)[0])


def target_alarm(avg_incidence: float, policy_intensity: float, p: ModelParams) -> float:
    """Instantaneous *target* alarm before delayed adaptation.

    Endogenous fear and policy pressure combine and saturate at 1.  We combine
    them so neither channel alone can exceed 1 and the two reinforce:
        a* = 1 - (1 - endog)(1 - policy_weight * policy)
    """
    endog = endogenous_alarm_value(avg_incidence, p)
    pol = p.policy_weight * policy_intensity
    return float(np.clip(1.0 - (1.0 - endog) * (1.0 - pol), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Simulation (stochastic chain-binomial SIR)
# ---------------------------------------------------------------------------


def simulate(
    params: ModelParams,
    policies: list[Policy],
    n_days: int = 120,
    N: int = 100_000,
    I0: int = 5,
    rng: np.random.Generator | None = None,
) -> Epidemic:
    """Simulate one stochastic epidemic with policy-informed behavioral change.

    Transitions (Ward, Deardon & Schmidt 2023):
        I*_t ~ Binomial(S_t, pi_SI_t),   pi_SI_t = 1 - exp(-beta (1 - a_t) I_t / N)
        R*_t ~ Binomial(I_t, pi_IR),     pi_IR   = 1 - exp(-gamma)
    """
    if rng is None:
        rng = np.random.default_rng()

    p_sig = policy_signal(policies, n_days)
    pi_IR = 1.0 - np.exp(-params.gamma)

    S = np.zeros(n_days, dtype=int)
    I = np.zeros(n_days, dtype=int)
    R = np.zeros(n_days, dtype=int)
    new_inf = np.zeros(n_days, dtype=int)
    new_rem = np.zeros(n_days, dtype=int)
    alarm = np.zeros(n_days)

    S[0], I[0], R[0] = N - I0, I0, 0
    a_prev = 0.0

    for t in range(n_days):
        # Delayed adaptation toward the target alarm (geometric / exp. smoothing).
        window = new_inf[max(0, t - params.m):t]
        avg_inc = window.mean() if window.size else 0.0
        a_star = target_alarm(avg_inc, p_sig[t], params)
        a_mean = (1 - params.adapt_rate) * a_prev + params.adapt_rate * a_star
        # Heterogeneous compliance: realized alarm is a noisy version of the mean.
        a_t = float(np.clip(a_mean + rng.normal(0.0, params.compliance_sd), 0.0, 1.0))
        alarm[t] = a_t
        a_prev = a_mean  # adaptation tracks the mean, not the noisy realization

        if t + 1 < n_days:
            pi_SI = 1.0 - np.exp(-params.beta * (1.0 - a_t) * I[t] / N)
            i_star = rng.binomial(S[t], pi_SI)
            r_star = rng.binomial(I[t], pi_IR)
            new_inf[t] = i_star
            new_rem[t] = r_star
            S[t + 1] = S[t] - i_star
            I[t + 1] = I[t] + i_star - r_star
            R[t + 1] = R[t] + r_star

    return Epidemic(S, I, R, new_inf, new_rem, alarm, p_sig, N,
                    meta={"params": params, "policies": policies})


# ---------------------------------------------------------------------------
# Deterministic alarm path (used by the likelihood: mean behavior, no comp. noise)
# ---------------------------------------------------------------------------


def alarm_path(new_inf: np.ndarray, policy: np.ndarray, p: ModelParams) -> np.ndarray:
    """Reconstruct the mean alarm path a_t from an observed incidence series."""
    n = len(new_inf)
    a = np.zeros(n)
    a_prev = 0.0
    for t in range(n):
        window = new_inf[max(0, t - p.m):t]
        avg_inc = window.mean() if window.size else 0.0
        a_star = target_alarm(avg_inc, policy[t], p)
        a_prev = (1 - p.adapt_rate) * a_prev + p.adapt_rate * a_star
        a[t] = a_prev
    return a


# ---------------------------------------------------------------------------
# Likelihood (both I* and R* observed -> no data augmentation needed)
# ---------------------------------------------------------------------------

_EPS = 1e-12


def log_likelihood(theta: dict, epi: Epidemic, p_template: ModelParams) -> float:
    """Complete-data log-likelihood of the chain-binomial SIR with alarm.

    ``theta`` holds the free parameters being estimated; any parameter absent
    from ``theta`` is taken from ``p_template`` (treated as known/fixed).
    """
    from scipy.stats import binom

    p = ModelParams(**{**p_template.__dict__, **theta})
    if p.beta <= 0 or p.gamma <= 0 or not (0 < p.gamma < 1 + _EPS):
        return -np.inf
    if not (0 <= p.endog_delta <= 1) or p.endog_x0 <= 0 or p.endog_nu <= 0:
        return -np.inf
    if not (0 <= p.policy_weight <= 1) or not (0 < p.adapt_rate <= 1):
        return -np.inf

    N = epi.N
    a = alarm_path(epi.new_infections, epi.policy, p)
    pi_IR = 1.0 - np.exp(-p.gamma)

    T = len(epi.S) - 1
    S, I = epi.S[:T], epi.I[:T]
    i_star, r_star = epi.new_infections[:T], epi.new_removals[:T]

    pi_SI = 1.0 - np.exp(-p.beta * (1.0 - a[:T]) * I / N)
    pi_SI = np.clip(pi_SI, _EPS, 1 - _EPS)

    ll_inf = binom.logpmf(i_star, S, pi_SI).sum()
    ll_rem = binom.logpmf(r_star, I, np.clip(pi_IR, _EPS, 1 - _EPS)).sum()
    ll = ll_inf + ll_rem
    return float(ll) if np.isfinite(ll) else -np.inf
