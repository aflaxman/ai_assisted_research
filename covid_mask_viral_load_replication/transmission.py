"""Transmission layer of Goyal et al. (2021): masks, contacts, dose response.

The within-host model (``within_host.camdl``) gives each transmitter a viral
load on each day since exposure.  Everything here is the paper's daily
bookkeeping on top of that viral load, following the Supplementary methods
and, where those are silent, the authors' public R code for the companion
eLife paper (``SARS_CoV_2_Params_search_Curated.R``):

* a day's exposure contacts are ``eta ~ Gamma(shape=theta/rho, scale=rho)``;
* contagiousness and infectiousness are the same dose-response curve
  ``P(V) = V^alpha / (lambda^alpha + V^alpha)`` of the exposure viral load;
* a mask of efficacy ``eps`` multiplies the exposure viral load by ``1 - eps``,
  once for a masked transmitter and once more for masked contacts;
* a day's secondary infections are ``Ber(P) * P * eta``, a real number;
* a transmitter's individual R0 is the sum over days, rounded to an integer
  (the eLife code rounds; the masking supplement does not say).

Two ways of getting the population R_e are provided.  ``simulate`` is the
paper's Monte Carlo over transmitters.  ``expected_re`` is the expectation the
Monte Carlo estimates, which has a closed form given the viral loads because
``E[Ber(P) * P * eta] = theta * P^2`` and the masking draws are independent of
everything else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: [alpha, lambda, tau, theta, rho] from the Supplementary methods.
ALPHA = 0.8
LAMBDA = 1.0e7
TAU = 0.5
THETA = 4.0
RHO = 40.0

#: Pair types, in the order the paper's Figure 4 legend lists them.
PAIRS = ["unmasked_to_unmasked", "unmasked_to_masked", "masked_to_unmasked", "masked_to_masked"]


def dose_response(v, lam: float = LAMBDA, alpha: float = ALPHA):
    """Contagiousness (= infectiousness) at exposure viral load ``v``."""
    va = np.maximum(v, 0.0) ** alpha
    return va / (lam**alpha + va)


def transmission_risk(v, eps_t=0.0, eps_e=0.0):
    """Per-contact transmission probability: contagiousness x infectiousness."""
    return dose_response(np.asarray(v) * (1.0 - eps_t) * (1.0 - eps_e)) ** 2


# ---------------------------------------------------------------------------
# The expectation
# ---------------------------------------------------------------------------

def pair_weights(coverage: float) -> np.ndarray:
    """Probability of each pair type when transmitter and contacts are each
    masked with probability ``coverage`` (= compliance x adherence)."""
    c = coverage
    return np.array([(1 - c) * (1 - c), (1 - c) * c, c * (1 - c), c * c])


def expected_re_by_pair(V: np.ndarray, coverage: float, eps_t: float, eps_e: float,
                        theta: float = THETA) -> np.ndarray:
    """Expected R_e contributed by each pair type (order of ``PAIRS``).

    ``V`` is transmitters x days.  No rounding: this is the mean of the
    real-valued sum the Monte Carlo draws.
    """
    w = pair_weights(coverage)
    factors = [1.0, 1.0 - eps_e, 1.0 - eps_t, (1.0 - eps_t) * (1.0 - eps_e)]
    return np.array([
        wi * theta * (dose_response(V * f) ** 2).sum(axis=1).mean()
        for wi, f in zip(w, factors)
    ])


def expected_re(V, coverage, eps_t, eps_e, theta: float = THETA) -> float:
    return float(expected_re_by_pair(V, coverage, eps_t, eps_e, theta).sum())


# ---------------------------------------------------------------------------
# The Monte Carlo
# ---------------------------------------------------------------------------

@dataclass
class Draws:
    """One Monte Carlo run over ``n`` transmitters and ``d`` days."""
    v: np.ndarray          # n x d   transmitter viral load
    v_exposure: np.ndarray  # n x d  viral load after masks
    contacts: np.ndarray   # n x d   exposure contacts eta
    pair: np.ndarray       # n x d   index into PAIRS
    infections: np.ndarray  # n x d  Ber(P) * P * eta, real-valued

    @property
    def total(self) -> np.ndarray:
        """Real-valued individual reproduction number."""
        return self.infections.sum(axis=1)

    @property
    def r_individual(self) -> np.ndarray:
        """Individual R0 as the eLife code reports it: the sum, rounded."""
        return np.round(self.total)

    def by_pair(self) -> np.ndarray:
        """n x 4 real-valued infections by pair type."""
        out = np.zeros((self.infections.shape[0], 4))
        for j in range(4):
            out[:, j] = np.where(self.pair == j, self.infections, 0.0).sum(axis=1)
        return out


def simulate(V: np.ndarray, rng: np.random.Generator, *, compliance: float = 0.0,
             adherence: float = 0.0, eps_t: float = 0.0, eps_e: float = 0.0,
             theta: float = THETA, rho: float = RHO, always_masked_above: float | None = None,
             always_masked_if_ever_above: float | None = None, follow_up_days: int = 31) -> Draws:
    """The paper's procedure for one scenario, for the transmitters in ``V``.

    compliance (sigma)  probability a person ever wears a mask
    adherence (vartheta) probability a mask wearer has it on in a given day
    eps_t, eps_e        mask efficacy for transmitter and for exposed contacts
    always_masked_above Supplementary figure 5, read as person-days: on days
                        with more than this many exposure contacts, transmitter
                        and contacts are masked whatever their compliance
    always_masked_if_ever_above
                        Supplementary figure 5, read as people: a transmitter
                        with more than this many exposure contacts on any of
                        the ``follow_up_days`` the authors simulate is masked
                        on every day, as are that transmitter's contacts
    """
    n, d = V.shape
    v = V

    compliant = rng.random(n) < compliance
    t_masked = compliant[:, None] & (rng.random((n, d)) < adherence)
    contacts = rng.gamma(shape=theta / rho, scale=rho, size=(n, d))
    e_masked = rng.random((n, d)) < compliance * adherence
    if always_masked_above is not None:
        busy = contacts > always_masked_above
        t_masked = t_masked | busy
        e_masked = e_masked | busy
    if always_masked_if_ever_above is not None:
        # The authors draw contacts for 31 days although viral load is zero
        # after day 20; the extra days count towards "ever".
        later = rng.gamma(shape=theta / rho, scale=rho, size=(n, max(follow_up_days - d, 0)))
        busy_person = (contacts.max(axis=1) > always_masked_if_ever_above) | (
            later.max(axis=1, initial=0.0) > always_masked_if_ever_above)
        t_masked = t_masked | busy_person[:, None]
        e_masked = e_masked | busy_person[:, None]

    v_exposure = v * np.where(t_masked, 1.0 - eps_t, 1.0) * np.where(e_masked, 1.0 - eps_e, 1.0)
    p = dose_response(v_exposure)
    success = rng.random((n, d)) < p
    infections = np.where(success, p * contacts, 0.0)
    pair = 2 * t_masked.astype(int) + e_masked.astype(int)
    return Draws(v=v, v_exposure=v_exposure, contacts=contacts, pair=pair, infections=infections)


def summarize(r: np.ndarray) -> dict[str, float]:
    """Population summaries of integer individual reproduction numbers."""
    total = r.sum()
    out = {
        "mean_R": r.mean(),
        "sd_R": r.std(ddof=1),
        "pct_transmit": 100.0 * (r >= 1).mean(),
    }
    for k in (5, 10, 20, 50):
        out[f"pct_all_ge{k}"] = 100.0 * (r >= k).mean()
        out[f"pct_transmitters_ge{k}"] = 100.0 * (r >= k).sum() / max((r >= 1).sum(), 1)
        out[f"share_infections_ge{k}"] = 100.0 * r[r >= k].sum() / total if total > 0 else np.nan
    return out
