"""Checks on the transmission layer that do not need camdl.

    uv run pytest
"""
import numpy as np

import transmission as tr


def toy_viral_loads(n=400, seed=0):
    """Viral loads with the right shape: a rise to a peak near 1e7, then decay."""
    rng = np.random.default_rng(seed)
    days = np.arange(21.0)
    peak_day = rng.uniform(2.0, 5.0, size=(n, 1))
    peak = rng.normal(7.0, 0.5, size=(n, 1))
    return 10.0 ** (peak - 0.8 * np.abs(days - peak_day))


def test_dose_response_is_half_at_lambda():
    assert np.isclose(tr.dose_response(tr.LAMBDA), 0.5)
    assert np.isclose(tr.transmission_risk(tr.LAMBDA), 0.25)
    assert tr.dose_response(-1.0) == 0.0


def test_masks_enter_only_through_the_product():
    v = np.logspace(5, 10, 11)
    assert np.allclose(tr.transmission_risk(v, 0.3, 0.6), tr.transmission_risk(v, 0.6, 0.3))
    assert np.allclose(tr.transmission_risk(v, 0.5, 0.5), tr.transmission_risk(v, 0.75, 0.0))


def test_pair_weights_sum_to_one():
    for c in (0.0, 0.0625, 0.5625, 1.0):
        assert np.isclose(tr.pair_weights(c).sum(), 1.0)


def test_expectation_reduces_to_baseline_without_masks_or_with_useless_ones():
    V = toy_viral_loads()
    base = tr.expected_re(V, 0.0, 0.0, 0.0)
    assert np.isclose(tr.expected_re(V, 0.8, 0.0, 0.0), base)
    assert np.isclose(tr.expected_re(V, 0.0, 0.9, 0.9), base)
    assert np.isclose(tr.expected_re(V, 1.0, 1.0, 0.3), 0.0)


def test_monte_carlo_mean_matches_the_expectation():
    V = toy_viral_loads()
    rng = np.random.default_rng(1)
    kwargs = dict(compliance=0.75, adherence=0.75, eps_t=0.5, eps_e=0.3)
    totals, by_pair = [], []
    for _ in range(400):
        d = tr.simulate(V, rng, **kwargs)
        totals.append(d.total.mean())
        by_pair.append(d.by_pair().mean(axis=0))
    exact = tr.expected_re_by_pair(V, 0.75 * 0.75, 0.5, 0.3)
    se = np.std(totals) / np.sqrt(len(totals))
    assert abs(np.mean(totals) - exact.sum()) < 4 * se
    assert np.allclose(np.mean(by_pair, axis=0), exact, atol=4 * np.std(by_pair, axis=0).max() / 20)


def test_individual_counts_are_integers_and_summaries_add_up():
    V = toy_viral_loads()
    r = tr.simulate(V, np.random.default_rng(2)).r_individual
    assert np.all(r == np.round(r))
    s = tr.summarize(r)
    assert np.isclose(s["mean_R"], r.mean())
    assert s["pct_all_ge5"] <= s["pct_transmit"] <= 100.0


def test_targeted_masking_readings_are_ordered():
    """Masking busy days cannot beat masking those same people on every day."""
    V = toy_viral_loads(2000)
    rng = np.random.default_rng(3)
    kw = dict(compliance=0.25, adherence=0.25, eps_t=0.6, eps_e=0.6)
    plain = np.mean([tr.simulate(V, rng, **kw).total.mean() for _ in range(40)])
    days = np.mean([tr.simulate(V, rng, always_masked_above=10.0, **kw).total.mean() for _ in range(40)])
    ever = np.mean([tr.simulate(V, rng, always_masked_if_ever_above=10.0, **kw).total.mean() for _ in range(40)])
    assert ever < days < plain
