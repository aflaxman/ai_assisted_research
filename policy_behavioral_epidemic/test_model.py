"""Tests for the policy-informed behavioral-change epidemic model."""

import numpy as np
import pytest

from model import (
    Epidemic,
    ModelParams,
    Policy,
    alarm_path,
    hill_alarm,
    log_likelihood,
    policy_signal,
    power_alarm,
    simulate,
    target_alarm,
    threshold_alarm,
)


# --- Alarm functions map into [0, 1] and are monotone in incidence -----------

def test_alarm_functions_in_unit_interval():
    x = np.linspace(0, 5000, 200)
    for a in (power_alarm(x, 0.001, 100_000),
              threshold_alarm(x, 200, 0.8),
              hill_alarm(x, 0.7, 200, 3)):
        assert a.min() >= 0.0 and a.max() <= 1.0


def test_hill_alarm_monotone_increasing():
    x = np.linspace(1, 5000, 200)
    a = hill_alarm(x, 0.7, 200, 3)
    assert np.all(np.diff(a) >= -1e-12)


def test_hill_alarm_zero_at_zero_incidence():
    assert hill_alarm(np.array([0.0]), 0.7, 200, 3)[0] == 0.0


# --- Policy signal -----------------------------------------------------------

def test_policy_signal_windows_and_overlap():
    sig = policy_signal([Policy(10, 20, 0.5), Policy(15, 30, 0.9)], 40)
    assert sig[5] == 0.0
    assert sig[12] == 0.5
    assert sig[17] == 0.9          # overlap takes the max
    assert sig[25] == 0.9
    assert sig[35] == 0.0


def test_target_alarm_increases_with_policy():
    p = ModelParams()
    a_none = target_alarm(0.0, 0.0, p)
    a_strong = target_alarm(0.0, 1.0, p)
    assert a_strong > a_none
    assert 0.0 <= a_strong <= 1.0


# --- Simulation invariants ---------------------------------------------------

def test_simulation_conserves_population():
    epi = simulate(ModelParams(), [Policy(30, 90, 0.8)], n_days=120,
                   N=50_000, I0=5, rng=np.random.default_rng(0))
    total = epi.S + epi.I + epi.R
    assert np.all(total == 50_000)
    assert np.all(epi.S >= 0) and np.all(epi.I >= 0) and np.all(epi.R >= 0)


def test_behavior_reduces_final_size():
    """A strong behavioral response must lower the attack rate."""
    rng = np.random.default_rng(3)
    no_behavior = ModelParams(endog_delta=0.0, policy_weight=0.0, compliance_sd=0.0)
    strong = ModelParams(endog_delta=0.85, policy_weight=0.9, compliance_sd=0.0)
    pol = [Policy(20, 120, 0.9)]
    fs_none, fs_strong = [], []
    for s in range(15):
        r1 = np.random.default_rng(s)
        r2 = np.random.default_rng(s)
        fs_none.append(simulate(no_behavior, pol, 150, 50_000, 5, r1).R[-1])
        fs_strong.append(simulate(strong, pol, 150, 50_000, 5, r2).R[-1])
    assert np.mean(fs_strong) < np.mean(fs_none)


# --- Likelihood --------------------------------------------------------------

def test_loglik_finite_at_truth():
    p = ModelParams()
    epi = simulate(p, [Policy(30, 120, 0.8)], n_days=120, N=50_000, I0=5,
                   rng=np.random.default_rng(7))
    ll = log_likelihood({"beta": p.beta, "gamma": p.gamma}, epi, p)
    assert np.isfinite(ll)


def test_loglik_peaks_near_true_beta():
    """Profile log-likelihood in beta should peak near the generating value."""
    p = ModelParams(endog_delta=0.5, policy_weight=0.6)
    epi = simulate(p, [Policy(25, 200, 0.85)], n_days=200, N=200_000, I0=10,
                   rng=np.random.default_rng(11))
    betas = np.linspace(0.3, 0.9, 25)
    lls = [log_likelihood({"beta": b}, epi, p) for b in betas]
    b_hat = betas[int(np.argmax(lls))]
    assert abs(b_hat - p.beta) < 0.12


def test_loglik_rejects_invalid_params():
    p = ModelParams()
    epi = simulate(p, [], n_days=60, N=20_000, I0=5, rng=np.random.default_rng(1))
    assert log_likelihood({"beta": -1.0}, epi, p) == -np.inf
    assert log_likelihood({"endog_delta": 2.0}, epi, p) == -np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
