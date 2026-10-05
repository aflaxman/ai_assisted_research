"""Parameter recovery for the numpyro unit-level SAE model on a small synthetic graph."""
import numpy as np

from models import laplacian_eigen, fit_sae, predict_district_proportions, rhat


def _toy(r=24, K=6, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(r, 2))
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    adj = ((d < 0.3) & (d > 0)).astype(float)
    eig = laplacian_eigen(adj)
    area = np.repeat(np.arange(r), K)
    X = np.column_stack([np.ones(r * K), rng.integers(0, 2, r * K), rng.normal(size=r * K)])
    beta = np.array([-1.0, 0.8, 0.3])
    eta = rng.normal(0, 0.6, r)
    psi = X @ beta + eta[area]
    n = rng.poisson(40, r * K)
    y = rng.binomial(n, 1 / (1 + np.exp(-psi)))
    cells = dict(X=X, area=area, n=n, y=y, w=np.ones(r * K))
    return rng, eig, cells, beta, eta, adj


def test_recovers_fixed_effects_and_converges():
    rng, eig, cells, beta, eta, adj = _toy()
    fit = fit_sae(cells, eig, num_warmup=300, num_samples=300, num_chains=2, seed=1)
    assert rhat(fit["beta"]).max() < 1.1 and rhat(fit["eta"]).max() < 1.1
    assert sum(fit["diverging"]) == 0
    est = fit["beta"].reshape(-1, 3).mean(0)
    assert np.all(np.abs(est - beta) < 0.25)
    assert np.corrcoef(fit["eta"].reshape(-1, len(eta)).mean(0), eta)[0, 1] > 0.8


def test_amatrix_fit_and_prediction_shapes():
    rng, eig, cells, beta, eta, adj = _toy()
    r = len(eta)
    A = 0.7 * np.eye(r) + 0.3 * adj / adj.sum(1, keepdims=True)
    fit = fit_sae(cells, eig, A_draws=[A, A], num_warmup=200, num_samples=200, num_chains=2, seed=2)
    pop = dict(X=cells["X"], area=cells["area"], m=np.full(cells["X"].shape[0], 10.0),
               y_obs=np.bincount(cells["area"], weights=cells["y"], minlength=r),
               N=np.bincount(cells["area"], weights=cells["n"], minlength=r) + 10.0 * 6)
    draws = predict_district_proportions(fit, pop, rng)
    assert draws.shape == (400, r) and np.all((draws >= 0) & (draws <= 1))
