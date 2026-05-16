"""Sanity tests that pin down behaviour we relied on in the blog post.

Run with: .venv/bin/python -m pytest test_models.py -v
"""
import numpy as np
import pytest

from scenario import DEFAULT, Scenario
import ode_sir
import decision_tree
import des_clinic
import abm_network


def test_ode_no_vax_runs_a_full_epidemic():
    r = ode_sir.simulate(DEFAULT, vaccinated=False)
    # With R0=2, no intervention should infect a clear majority.
    assert r["total_infected"] > 0.7 * DEFAULT.N


def test_ode_vaccination_suppresses_outbreak():
    r = ode_sir.simulate(DEFAULT, vaccinated=True)
    assert r["total_infected"] < 50


def test_final_size_matches_simulated_total():
    # The analytic equation should be within 1% of the simulated final size.
    r = ode_sir.simulate(DEFAULT, vaccinated=False)
    analytic = ode_sir.final_size(DEFAULT.R0) * DEFAULT.N
    assert abs(r["total_infected"] - analytic) / analytic < 0.01


def test_decision_tree_prefers_vaccination_at_default_costs():
    tree = decision_tree.build_tree(DEFAULT)
    summary = decision_tree.summarize(tree)
    assert summary["best"] == "Mass-vaccinate"


def test_decision_tree_flips_when_vaccine_is_outrageously_expensive():
    scn = Scenario(cost_vaccine_dose=10_000.0)
    tree = decision_tree.build_tree(scn)
    summary = decision_tree.summarize(tree)
    assert summary["best"] == "Do nothing"


def test_des_queue_is_nonempty_without_vaccination():
    run = des_clinic.simulate(DEFAULT, vaccinated=False, seed=1)
    s = des_clinic.summarize(run)
    # The clinic should have hours-long waits and a real queue at the peak.
    assert s["max_queue"] > 10
    assert s["mean_wait_hr"] > 1.0


def test_des_queue_stays_empty_with_vaccination():
    run = des_clinic.simulate(DEFAULT, vaccinated=True, seed=1)
    s = des_clinic.summarize(run)
    assert s["max_queue"] <= 2  # small Poisson variance is fine
    assert s["mean_wait_hr"] < 1.0


def test_abm_shows_some_extinction_without_vaccination():
    runs = abm_network.simulate_many(DEFAULT, vaccinated=False, n_runs=40, base_seed=7)
    p_ext = abm_network.extinction_probability(runs)
    # With R0=2 and only three seeds, classical branching-process theory predicts
    # roughly (1/R0)^seeds = 1/8 = 12.5% extinction; allow a generous band.
    assert 0.05 < p_ext < 0.5


def test_abm_vaccination_mostly_prevents_takeoff():
    runs = abm_network.simulate_many(DEFAULT, vaccinated=True, n_runs=40, base_seed=7)
    p_ext = abm_network.extinction_probability(runs)
    assert p_ext > 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
