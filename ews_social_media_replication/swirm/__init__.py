"""Replication of Olajide, Lutscher & Smith (2026), PLOS ONE 10.1371/journal.pone.0354091:
early-warning signals for infectious diseases with a social-media compartment.

The package is deliberately small and layered so a notebook can tell the paper's
story one function at a time:

* ``model``      -- the deterministic SWIRM system, R0(t), and the bifurcation.
* ``simulate``   -- Euler-Maruyama for additive, multiplicative, demographic noise.
* ``changepoint``-- AMOC mean-shift detection (when does the epidemic *look* like it started?).
* ``ews``        -- rolling variance / lag-1 autocorrelation and their Kendall-tau trends.
* ``evaluate``   -- run an ensemble, measure delay, score EWS performance (Table 2, ROC/AUC).
* ``plots``      -- thin plotting helpers.
"""

from .model import Params, deterministic_rhs, initial_state, STATE_NAMES
from .simulate import simulate_ensemble
from .changepoint import amoc_meanshift, changepoint_time
from .ews import ews_trends, gaussian_detrend
from .evaluate import Scenario, run_scenario, summarize, roc_curve

__all__ = [
    "Params", "deterministic_rhs", "initial_state", "STATE_NAMES",
    "simulate_ensemble",
    "amoc_meanshift", "changepoint_time",
    "ews_trends", "gaussian_detrend",
    "Scenario", "run_scenario", "summarize", "roc_curve",
]
