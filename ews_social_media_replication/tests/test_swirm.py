"""Fast checks on the analytic and numerical core.

The most valuable test is the first one: with Table 1 parameters, the model's
R0(t) must cross 1 at t = 14.5 days -- the paper's stated bifurcation. If that
holds, the equations and constants are wired up correctly.
"""

import numpy as np

from swirm import Params, deterministic_rhs, initial_state, amoc_meanshift, gaussian_detrend
from swirm.ews import _rolling_variance, _rolling_lag1_autocorr


def test_bifurcation_at_14_5_days():
    p = Params()
    assert abs(p.bifurcation_time() - 14.5) < 0.1
    assert abs(p.R0(p.bifurcation_time()) - 1.0) < 1e-9


def test_susceptible_equilibrium_is_1000():
    assert abs(Params().S_star - 1000.0) < 1e-6


def test_dfe_is_a_fixed_point_without_seed():
    p = Params()
    x = initial_state(p, seed_infected=0.0)
    d = deterministic_rhs(x, 0.0, p)
    # S, W, I, R, M derivatives all ~0 at the disease-free equilibrium.
    assert np.allclose(d, 0.0, atol=1e-6)


def test_rolling_variance_matches_numpy():
    y = np.random.default_rng(0).normal(size=200)
    win = 50
    got = _rolling_variance(y, win)
    want = np.array([y[k:k + win].var() for k in range(y.size - win + 1)])
    assert np.allclose(got, want)


def test_rolling_autocorr_matches_naive_loop():
    y = np.random.default_rng(4).normal(size=200)
    win = 50
    got = _rolling_lag1_autocorr(y, win)

    def naive(w):
        w = w - w.mean()
        return np.dot(w[:-1], w[1:]) / np.dot(w, w)

    want = np.array([naive(y[k:k + win]) for k in range(y.size - win + 1)])
    assert np.allclose(got, want, atol=1e-10)


def test_amoc_finds_obvious_step():
    y = np.concatenate([np.zeros(100), np.ones(100) * 5]) + \
        np.random.default_rng(1).normal(0, 0.1, 200)
    cp = amoc_meanshift(y)
    assert cp is not None and abs(cp - 100) <= 2


def test_amoc_false_positive_rate_near_5pct():
    # The penalty is calibrated to a 5% false-positive rate on flat noise.
    rng = np.random.default_rng(2)
    fires = sum(amoc_meanshift(rng.normal(0, 1, 300)) is not None for _ in range(400))
    assert fires / 400 < 0.10


def test_detrend_removes_linear_trend():
    t = np.arange(300)
    y = 0.05 * t + np.random.default_rng(3).normal(0, 0.1, 300)
    resid = gaussian_detrend(y, bandwidth_frac=0.1)
    assert abs(np.polyfit(t, resid, 1)[0]) < 0.01
