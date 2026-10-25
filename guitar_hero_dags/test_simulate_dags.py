"""Tests: each DAG reproduces the observed contrast; the tools that
distinguish them behave as the theory says."""

import math

import numpy as np
import pytest

from simulate_dags import (
    h2_superstar_manager,
    h2b_general_hustle,
    h3_busy_stores,
    h4_sales_to_demo,
    h5_noticed,
    max_fakeable_rr,
    run_all,
    sim_equal_strength_boundary,
)

N = 200_000


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_evalue_inverts_bounding_factor():
    # E = RR + sqrt(RR(RR-1)) is exactly the strength whose equal-arm
    # bounding factor recovers RR.
    for rr in (1.5, 4.2, 9.0):
        e = rr + math.sqrt(rr * (rr - 1))
        assert math.isclose(max_fakeable_rr(e), rr, rel_tol=1e-9)


def test_boundary_world_attains_the_bound(rng):
    # A zero-effect world with confounder strength 17.49 on both arms
    # shows an observed RR of about 9 -- the E-value claim, simulated.
    obs = sim_equal_strength_boundary(17.49, N, rng)
    assert obs == pytest.approx(max_fakeable_rr(17.49), rel=0.05)
    assert obs == pytest.approx(9.0, rel=0.06)


def test_every_scenario_shows_the_anecdote(rng):
    for res in run_all(n=N, seed=7):
        assert res["observed"] > 7.0, res["name"]


def test_h2_fakes_nine_with_no_effect(rng):
    res = h2_superstar_manager(N, rng)
    assert res["observed"] == pytest.approx(9.0, rel=0.1)
    assert res["rollout"] == 1.0
    # Both confounder associations must exceed the RR being faked.
    assert res["rr_eu"] > 9.0 and res["rr_ud"] > 9.0


def test_negative_control_separates_traffic_from_specific(rng):
    specific = h2_superstar_manager(N, rng)
    hustle = h2b_general_hustle(N, rng)
    traffic = h3_busy_stores(N, rng)
    assert specific["madden"] == pytest.approx(1.0, abs=0.1)
    assert 1.5 < hustle["madden"] < 4.0
    assert traffic["madden"] > 6.0


def test_h4_shows_regression_to_the_mean(rng):
    res = h4_sales_to_demo(N, rng)
    assert res["observed"] > 5.0  # cross-section still looks impressive
    assert res["within_store"] < 0  # but demo stores FALL after installing


def test_h5_noticing_exaggerates_a_real_effect(rng):
    res = h5_noticed(N, rng)
    assert res["fair_rr"] == pytest.approx(3.0, rel=0.1)
    assert res["observed"] > 2 * res["fair_rr"]
    assert res["within_store"] < 0
