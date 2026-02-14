"""Tests for the Wesson sampling method simulation."""

import numpy as np
import pandas as pd
from simulation import (
    generate_population,
    draw_samples,
    predict_cross_probabilities,
    combine_samples,
    estimate_means,
)


def test_population_generation():
    pop = generate_population(n=1000, seed=0)
    assert len(pop) == 1000
    assert all(col in pop.columns for col in ["age", "sheltered", "service_use", "network_size", "subgroup", "health_score", "p_vbs", "p_rds"])
    assert pop["age"].between(18, 70).all()
    assert pop["sheltered"].isin([0, 1]).all()
    assert pop["p_vbs"].between(0, 1).all()
    assert pop["p_rds"].between(0, 1).all()
    assert pop["subgroup"].isin([0, 1, 2, 3]).all()


def test_sampling_produces_correct_sizes():
    pop = generate_population(n=5000, seed=1)
    vbs, rds = draw_samples(pop, n_vbs=300, n_rds=100, seed=1)
    assert len(vbs) == 300
    assert len(rds) == 100
    assert (vbs["method"] == "VBS").all()
    assert (rds["method"] == "RDS").all()


def test_vbs_oversamples_sheltered():
    """VBS should have a higher proportion of sheltered individuals than the population."""
    pop = generate_population(n=10_000, seed=2)
    vbs, _ = draw_samples(pop, n_vbs=800, n_rds=200, seed=2)
    pop_sheltered_rate = pop["sheltered"].mean()
    vbs_sheltered_rate = vbs["sheltered"].mean()
    assert vbs_sheltered_rate > pop_sheltered_rate


def test_rds_oversamples_marginalized_subgroups():
    """RDS should have a higher proportion of farmworkers + DV survivors than VBS."""
    pop = generate_population(n=10_000, seed=3)
    vbs, rds = draw_samples(pop, n_vbs=800, n_rds=200, seed=3)
    vbs_marginal = vbs["subgroup"].isin([2, 3]).mean()
    rds_marginal = rds["subgroup"].isin([2, 3]).mean()
    assert rds_marginal > vbs_marginal


def test_cross_probabilities_have_expected_columns():
    pop = generate_population(n=5000, seed=4)
    vbs, rds = draw_samples(pop, n_vbs=300, n_rds=100, seed=4)
    vbs, rds, diag = predict_cross_probabilities(pop, vbs, rds, seed=4)
    assert "pred_p_rds" in vbs.columns
    assert "pred_p_vbs" in vbs.columns
    assert "pred_p_rds" in rds.columns
    assert "pred_p_vbs" in rds.columns
    assert 0.5 < diag["auc_vbs"] < 1.0
    assert 0.5 < diag["auc_rds"] < 1.0


def test_combined_inclusion_probability():
    """P(overall) = P(VBS) + P(RDS) - P(VBS)*P(RDS) should hold."""
    pop = generate_population(n=5000, seed=5)
    vbs, rds = draw_samples(pop, n_vbs=300, n_rds=100, seed=5)
    vbs, rds, _ = predict_cross_probabilities(pop, vbs, rds, seed=5)
    combined = combine_samples(vbs, rds)

    expected = combined["pred_p_vbs"] + combined["pred_p_rds"] - combined["pred_p_vbs"] * combined["pred_p_rds"]
    np.testing.assert_allclose(combined["p_overall"], expected)
    assert (combined["p_overall"] > 0).all()
    assert (combined["p_overall"] <= 1).all()


def test_combined_weighted_reduces_bias():
    """The combined weighted estimator should have less bias than VBS unweighted."""
    pop = generate_population(n=10_000, seed=42)
    vbs, rds = draw_samples(pop, n_vbs=800, n_rds=200, seed=42)
    vbs, rds, _ = predict_cross_probabilities(pop, vbs, rds, seed=42)
    combined = combine_samples(vbs, rds)
    estimates = estimate_means(pop, vbs, rds, combined)

    true = estimates["true_mean"]
    vbs_bias = abs(estimates["vbs_unweighted"] - true)
    combined_bias = abs(estimates["combined_weighted"] - true)
    assert combined_bias < vbs_bias, f"Combined bias {combined_bias:.2f} >= VBS bias {vbs_bias:.2f}"


if __name__ == "__main__":
    import sys
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test in test_funcs:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{len(test_funcs)} tests passed")
    sys.exit(0 if passed == len(test_funcs) else 1)
