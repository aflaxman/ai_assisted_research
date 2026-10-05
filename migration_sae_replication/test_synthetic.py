import numpy as np
from geography import load_geography
from synthetic import (covariate_cells, calibrate_intercepts, base_pattern, district_logits,
                       generate_population, individual_probs, migration_structure, simulate_moves,
                       composition_counts, sample_known_status, build_cells, PATTERNS, BASES)
from power_prior import power_prior_binomial, draw_scalars

geo = load_geography()
X, cell_p, prev = covariate_cells()


def test_cells_and_calibration():
    assert abs(cell_p.sum() - 1) < 1e-12
    target = np.linspace(0.1, 0.4, geo.r)
    mu = calibrate_intercepts(target, X, cell_p)
    p = 1 / (1 + np.exp(-(mu[:, None] + (X @ np.array([0, .1, .1, 0, -.3, 1.2]))[None, :])))
    assert np.allclose((p * cell_p).sum(1), target, atol=1e-8)


def test_base_patterns_in_range():
    for b in BASES:
        v = base_pattern(b, geo)
        assert v.min() >= 0.1 - 1e-9 and v.max() <= 0.4 + 1e-9 and v.shape == (geo.r,)


def test_smoothed_pattern_tracks_input():
    rng = np.random.default_rng(0)
    for b in BASES:
        base = base_pattern(b, geo)
        target, _ = district_logits(base, geo, rng, X, cell_p)
        assert np.corrcoef(base, target)[0, 1] > 0.95


def test_migration_structures():
    for pat in PATTERNS:
        E, P = migration_structure(pat, geo)
        assert np.allclose(P[E].sum(1), 1.0)
        assert np.all(np.diag(P) == 0) and E.any()


def test_moves_and_composition():
    rng = np.random.default_rng(1)
    district, cell = generate_population(geo, rng, cell_p)
    E, P = migration_structure("neighbors", geo)
    dest = simulate_moves(district, E, P, 0.4, rng)
    moved = dest != district
    assert abs(moved.mean() - 0.4) < 0.01
    assert np.all(geo.contiguity[district[moved], dest[moved]] == 1)
    C = composition_counts(district, dest, geo.r)
    assert C.sum() == district.size and np.allclose(C.sum(0), np.bincount(district, minlength=geo.r))
    a = draw_scalars(C, rng, 5)
    assert np.allclose(a.sum(2), 1.0) and np.all(a[:, C == 0] == 0)


def test_power_prior_reduces_to_tempered_prior_without_migration():
    rng = np.random.default_rng(2)
    r = 3
    C = np.diag([100.0, 200.0, 50.0])
    draws = rng.beta(20, 60, size=(2000, r))
    y, n = np.array([25.0, 50.0, 10.0]), np.array([100.0, 200.0, 50.0])
    pp = power_prior_binomial(draws, y, n, C, rng, n_draws=4000)
    # a_jj = 1 exactly: posterior Beta(alpha + y, beta + n - y) is tighter than the baseline
    assert np.all(pp.std(0) < draws.std(0))
    assert np.all(np.abs(pp.mean(0) - 0.25) < 0.02)


def test_build_cells_consistency():
    rng = np.random.default_rng(3)
    district, cell = generate_population(geo, rng, cell_p)
    base = base_pattern("block", geo)
    target, mu = district_logits(base, geo, rng, X, cell_p)
    p = individual_probs(mu, district, cell, X)
    z = rng.uniform(size=p.size) < p
    sampled, w = sample_known_status(cell, prev, rng)
    cells, pop = build_cells(district, cell, sampled, w, z.astype(float), X, geo.r)
    assert np.isclose(cells["n"].sum(), sampled.sum()) and np.isclose(cells["y"].sum(), z[sampled].sum())
    assert np.allclose(pop["N"], cells["n"].reshape(geo.r, -1).sum(1) + pop["m"].reshape(geo.r, -1).sum(1))
    assert np.isclose((cells["w"] * cells["n"]).sum(), sampled.sum(), rtol=1e-6)
    assert 0.08 < sampled.mean() < 0.12
    realized = np.bincount(district, weights=z, minlength=geo.r) / pop["N"]
    assert np.corrcoef(realized, target)[0, 1] > 0.9
