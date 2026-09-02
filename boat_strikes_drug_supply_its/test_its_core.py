"""Tests for the rolling-window ITS estimator.

The important one is `test_recovers_known_step`: simulate latent monthly counts
with a step change of known size, publish only the 12-month-ending totals, and
check the estimator gets the step back. `test_naive_regression_is_attenuated`
records why the machinery is needed at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import its_core as ic

INTERVENTION = pd.Timestamp("2025-09-01")


# --------------------------------------------------------------------------
# spline basis
# --------------------------------------------------------------------------
def test_spline_is_linear_beyond_boundary_knots():
    knots = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    x = np.linspace(100.0, 140.0, 41)
    basis = ic.natural_spline_basis(x, knots)
    # A linear function has a constant second difference of zero.
    second = np.diff(basis, n=2, axis=0)
    assert np.abs(second).max() < 1e-6


def test_spline_column_count():
    knots = np.array([0.0, 10.0, 20.0, 30.0])
    basis = ic.natural_spline_basis(np.linspace(0, 30, 31), knots)
    assert basis.shape[1] == len(knots) - 1


def test_spline_requires_three_knots():
    with pytest.raises(ValueError):
        ic.natural_spline_basis(np.arange(5.0), np.array([0.0, 1.0]))


# --------------------------------------------------------------------------
# design and moving-sum operator
# --------------------------------------------------------------------------
def _window_ends(n=135, start="2015-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def test_moving_sum_operator_sums_twelve_months():
    design = ic.build_design(_window_ends(24), INTERVENTION)
    assert design.A.shape == (24, len(design.months))
    assert np.allclose(design.A.sum(1), 12.0)
    # The first window ends at the first published month and reaches 11 back.
    assert design.months[0] == pd.Timestamp("2014-02-01")
    assert design.A[0, :12].sum() == 12.0
    assert design.A[0, 12:].sum() == 0.0


def test_covariance_matches_closed_form():
    design = ic.build_design(_window_ends(30), INTERVENTION)
    rng = np.random.default_rng(0)
    mu = rng.uniform(50, 150, len(design.months))
    sigma = ic._covariance(mu, design.A)
    assert np.allclose(sigma, sigma.T)
    # Windows more than 11 apart share no months.
    assert abs(sigma[0, 12]) < 1e-9
    # Diagonal is the sum of the 12 months in the window.
    assert np.isclose(sigma[0, 0], mu[:12].sum())
    # Off-diagonal is the sum over the shared months.
    assert np.isclose(sigma[0, 3], mu[3:12].sum())


def test_step_and_ramp_switch_on_at_intervention():
    design = ic.build_design(_window_ends(135), INTERVENTION, effect="step_ramp")
    step = design.X[:, design.names.index("step")]
    ramp = design.X[:, design.names.index("ramp_per_year")]
    pre = design.months < INTERVENTION
    assert step[pre].max() == 0.0
    assert step[~pre].min() == 1.0
    assert ramp[pre].max() == 0.0
    assert np.isclose(ramp[design.months == INTERVENTION][0], 1 / 12)


def test_lag_shifts_onset():
    design = ic.build_design(_window_ends(135), INTERVENTION,
                             effect="step", lag_months=3)
    step = design.X[:, design.names.index("step")]
    onset = design.months[step > 0][0]
    assert onset == pd.Timestamp("2025-12-01")


def test_spline_knots_stay_in_pre_period():
    """The counterfactual must extrapolate, not interpolate through the campaign."""
    design = ic.build_design(_window_ends(135), INTERVENTION)
    n_pre = int((design.months < INTERVENTION).sum())
    # Beyond the last pre-period month the baseline columns must be linear.
    tail = design.X[n_pre:, 1:design.names.index("step")]
    if tail.shape[0] > 3:
        assert np.abs(np.diff(tail, n=2, axis=0)).max() < 1e-6


# --------------------------------------------------------------------------
# recovery of a known effect
# --------------------------------------------------------------------------
def _simulate(step_log_rr, n_windows=135, seed=1, seasonal=0.0,
              intervention=INTERVENTION, dispersion=1.0, level=7.0,
              noise=True):
    """Latent monthly counts with a known step, published as rolling sums."""
    window_ends = _window_ends(n_windows)
    months = pd.date_range(window_ends[0] - pd.DateOffset(months=11),
                           window_ends[-1], freq="MS")
    t = np.arange(len(months), dtype=float)
    # A hump then a decline, loosely like the real fentanyl-era trajectory.
    log_mu = level + 0.9 * np.sin(np.pi * t / len(t)) - 0.004 * t
    log_mu = log_mu + step_log_rr * np.asarray(months >= intervention, dtype=float)
    log_mu = log_mu + seasonal * np.cos(2 * np.pi * np.asarray(months.month) / 12)
    mu = np.exp(np.asarray(log_mu, dtype=float))

    rng = np.random.default_rng(seed)
    if not noise:
        y = mu
    elif dispersion <= 1.0:
        y = rng.poisson(mu)
    else:                                    # negative binomial with var = phi*mu
        p = 1.0 / dispersion
        y = rng.negative_binomial(mu * p / (1 - p), p)
    rolling = np.array([y[i:i + 12].sum() for i in range(len(window_ends))],
                       dtype=float)
    return window_ends, rolling, mu


@pytest.mark.parametrize("truth", [0.0, -0.10, 0.08])
def test_recovers_known_step(truth):
    window_ends, rolling, _ = _simulate(truth, seed=7)
    design = ic.build_design(window_ends, INTERVENTION, effect="step")
    fit = ic.fit_its(rolling, design)
    est, se = fit.coef("step")
    assert abs(est - truth) < max(3.0 * se, 0.02)


def test_recovery_is_unbiased_across_replicates():
    truth = -0.12
    ests = []
    for seed in range(25):
        window_ends, rolling, _ = _simulate(truth, seed=seed)
        design = ic.build_design(window_ends, INTERVENTION, effect="step")
        ests.append(ic.fit_its(rolling, design).coef("step")[0])
    bias = float(np.mean(ests) - truth)
    assert abs(bias) < 0.03, f"mean estimate {np.mean(ests):.3f} vs truth {truth}"


def test_additive_periodic_seasonality_is_annihilated_exactly():
    """A count-scale 12-month-periodic term sums to a constant in every window."""
    window_ends = _window_ends(40)
    months = pd.date_range(window_ends[0] - pd.DateOffset(months=11),
                           window_ends[-1], freq="MS")
    season = 40.0 * np.cos(2 * np.pi * np.asarray(months.month) / 12)
    rolling = np.array([season[i:i + 12].sum() for i in range(len(window_ends))])
    assert np.ptp(rolling) < 1e-9


def test_window_leverage_is_k_over_twelve():
    design = ic.build_design(_window_ends(135), INTERVENTION, effect="step")
    lev = ic.window_leverage(design)
    post = lev[lev["post_fraction"] > 0]
    assert len(post) == 7          # Sep 2025 .. Mar 2026
    assert np.allclose(post["post_fraction"].to_numpy(),
                       np.arange(1, 8) / 12.0)
    assert 0.3 < post["post_fraction"].mean() < 0.4


def test_multiplicative_seasonality_is_amplified_into_the_step():
    """The honest limitation: a small wobble in the rolling series becomes a
    large apparent step, because the step is identified with ~1/3 leverage."""
    truth = -0.10
    ends_a, roll_plain, _ = _simulate(truth, seed=3, seasonal=0.0, noise=False)
    _, roll_season, _ = _simulate(truth, seed=3, seasonal=0.25, noise=False)
    wobble = np.abs(np.log(roll_season / roll_plain)).max()
    assert wobble < 0.05, "seasonality should barely move the rolling totals"

    est_plain = ic.fit_its(
        roll_plain, ic.build_design(ends_a, INTERVENTION, effect="step")
    ).coef("step")[0]
    est_season = ic.fit_its(
        roll_season, ic.build_design(ends_a, INTERVENTION, effect="step")
    ).coef("step")[0]
    assert abs(est_plain - truth) < 0.02
    # A <5% wobble displaces the step by far more than the wobble itself.
    assert abs(est_season - truth) > 4 * wobble


def test_controlled_contrast_survives_shared_seasonality():
    """Why the contrast is the primary estimator: shared seasonality cancels."""
    truth = -0.10
    for seasonal in (0.0, 0.25):
        errors = []
        for seed in range(6):
            ends, roll_t, _ = _simulate(truth, seed=seed, seasonal=seasonal)
            _, roll_c, _ = _simulate(0.0, seed=500 + seed, seasonal=seasonal,
                                     level=7.3)
            design = ic.build_design(ends, INTERVENTION, effect="step")
            ft = ic.fit_its(roll_t, design)
            fc = ic.fit_its(roll_c, design)
            treated = dict(zip(("log_rr", "se"), ft.coef("step")))
            control = dict(zip(("log_rr", "se"), fc.coef("step")))
            out = ic.contrast_effects(treated, control)
            errors.append(out["log_rr"] - truth)
        assert abs(np.mean(errors)) < 0.06, (
            f"contrast bias {np.mean(errors):+.3f} at seasonal={seasonal}")


def test_naive_regression_on_rolling_series_is_attenuated():
    """Why the latent model exists: segmented regression on the published
    rolling series smears a step over the following 12 windows and shrinks it."""
    truth = -0.20
    window_ends, rolling, _ = _simulate(truth, seed=11, n_windows=135)
    post = (window_ends >= INTERVENTION).astype(float)
    t = np.arange(len(window_ends), dtype=float)
    knots = np.quantile(t[window_ends < INTERVENTION], [0, 0.25, 0.5, 0.75, 1.0])
    naive_X = np.column_stack([np.ones_like(t),
                               ic.natural_spline_basis(t, knots), post])
    beta, *_ = np.linalg.lstsq(naive_X, np.log(rolling), rcond=None)
    naive_step = beta[-1]

    design = ic.build_design(window_ends, INTERVENTION, effect="step")
    proper_step = ic.fit_its(rolling, design).coef("step")[0]

    assert abs(naive_step) < 0.5 * abs(truth), "expected naive estimate to attenuate"
    assert abs(proper_step - truth) < abs(naive_step - truth)


def test_dispersion_is_detected():
    _, rolling_p, _ = _simulate(0.0, seed=5, dispersion=1.0)[0:3]
    window_ends, rolling_nb, _ = _simulate(0.0, seed=5, dispersion=4.0)
    design = ic.build_design(window_ends, INTERVENTION, effect="step")
    phi_nb = ic.fit_its(rolling_nb, design).dispersion
    phi_p = ic.fit_its(rolling_p, design).dispersion
    assert phi_nb > phi_p


# --------------------------------------------------------------------------
# effect summaries
# --------------------------------------------------------------------------
def test_relative_effect_matches_step_when_no_ramp():
    window_ends, rolling, _ = _simulate(-0.15, seed=2)
    design = ic.build_design(window_ends, INTERVENTION, effect="step")
    fit = ic.fit_its(rolling, design)
    eff = ic.relative_effect(fit)
    assert np.isclose(eff["log_rr"], fit.coef("step")[0])
    assert eff["lo"] < eff["rr"] < eff["hi"]


def test_cumulative_excess_sign_follows_the_effect():
    for truth, sign in ((-0.15, -1), (0.15, 1)):
        window_ends, rolling, _ = _simulate(truth, seed=4)
        design = ic.build_design(window_ends, INTERVENTION, effect="step")
        fit = ic.fit_its(rolling, design)
        out = ic.cumulative_excess(fit, INTERVENTION)
        assert out["post_months"] == int((design.months >= INTERVENTION).sum())
        assert np.sign(out["excess_deaths"]) == sign


def test_contrast_of_identical_effects_is_null():
    treated = {"log_rr": -0.10, "se": 0.04}
    control = {"log_rr": -0.10, "se": 0.03}
    out = ic.contrast_effects(treated, control)
    assert np.isclose(out["log_rr"], 0.0)
    assert np.isclose(out["se"], np.hypot(0.04, 0.03))
    assert out["pct_lo"] < 0 < out["pct_hi"]


def test_unknown_effect_rejected():
    with pytest.raises(ValueError):
        ic.build_design(_window_ends(30), INTERVENTION, effect="nonsense")


def test_dose_effect_requires_dose():
    with pytest.raises(ValueError):
        ic.build_design(_window_ends(30), INTERVENTION, effect="dose")
