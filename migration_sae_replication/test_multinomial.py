import numpy as np
from multinomial import (category_probs, draw_categories, build_cells_multi, fit_multinomial,
                         predict_multinomial, records_multi, H)
from models import laplacian_eigen, rhat
from power_prior import power_prior_multinomial


def test_category_probs_and_draws():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.4, 20000)
    P = category_probs(p)
    assert np.allclose(P.sum(1), 1.0)
    z = draw_categories(P, rng)
    freq = np.bincount(z, minlength=H) / z.size
    assert np.allclose(freq, P.mean(0), atol=0.01)


def test_small_multinomial_fit_recovers_shares():
    rng = np.random.default_rng(1)
    r, K = 12, 4
    pts = rng.uniform(size=(r, 2))
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    adj = ((d < 0.45) & (d > 0)).astype(float)
    eig = laplacian_eigen(adj)
    X = np.column_stack([np.ones(K), [0, 1, 0, 1], [0, 0, 1, 1]]).astype(float)
    area = np.repeat(np.arange(r), 400)
    cell = rng.integers(0, K, area.size)
    p_rr = 1 / (1 + np.exp(-(rng.normal(-1.0, 0.4, r)[area] + X[cell] @ np.array([0, 0.3, 0.8]))))
    z = draw_categories(category_probs(p_rr), rng)
    sampled = rng.uniform(size=area.size) < 0.5
    cells, pop = build_cells_multi(area, cell, sampled, np.ones(area.size), z, X, r)
    fit = fit_multinomial(cells, eig, num_warmup=200, num_samples=200, num_chains=2, seed=3)
    assert sum(fit["diverging"]) == 0 and rhat(fit["beta"]).max() < 1.1
    draws = predict_multinomial(fit, pop, rng)
    assert draws.shape == (400, r, H) and np.allclose(draws.sum(2), 1.0)
    truth = np.zeros((r, H))
    np.add.at(truth, (area, z), 1.0)
    truth /= truth.sum(1, keepdims=True)
    assert np.abs(draws.mean(0) - truth).max() < 0.06
    rec = records_multi(draws, truth, method="naive")
    assert set(rec["category"]) == set(["DS", "RR", "OtherDR", "preXDR_XDR", "TVD"])
    C = np.diag(np.bincount(area, minlength=r).astype(float))
    pp = power_prior_multinomial(draws, pop["N"][:, None] * draws.mean(0), C, rng, 500)
    assert pp.shape == (500, r, H) and np.allclose(pp.sum(2), 1.0)
    assert np.abs(pp.mean(0) - draws.mean(0)).max() < 0.03
