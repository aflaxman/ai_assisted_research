"""numpyro implementations of the unit-level Bayesian SAE models in Dupuis (2026).

Two models share one function:

* migration-naive unit-level model (Dupuis Ch. 1, after Parker et al.):
      logit(theta_i) = X_i beta + eta_{area(i)}
      eta | sigma_eta^2 ~ N(0, sigma_eta^2 Sigma_r),  Sigma_r = (L + I / sigma_eta^2)^{-1}
      beta | sigma_beta^2 ~ N(0, sigma_beta^2 I),  sigma^2 ~ InverseGamma
* migration-informed "A-matrix" model (Dupuis Ch. 3):
      logit(theta_i) = X_i beta + (A eta)_{area(i)}
  where A is the row-stochastic post-migration composition matrix. Following
  the dissertation, A is *not* updated by the data: it is drawn from its
  row-wise Dirichlet prior and held fixed within a chain (one draw per chain),
  so pooling chains integrates over the prior on A.

The weighted binomial pseudo-likelihood prod_i Bin(Z_i | 1, theta_i)^{w_i} is
evaluated on aggregated cells (district x covariate pattern), which is exact
when weights are constant within a cell, and makes each gradient evaluation
cost O(#cells) instead of O(#individuals).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

jax.config.update("jax_enable_x64", False)

HYPER_DEFAULT = dict(a=0.5, b=0.5, c=0.5, d=0.5)  # IG shape/scale for sigma_eta^2 and sigma_beta^2


def laplacian_eigen(adjacency: np.ndarray):
    """Eigendecomposition of the graph Laplacian L = D - A (symmetric 0/1 adjacency)."""
    A = np.asarray(adjacency, dtype=np.float64)
    L = np.diag(A.sum(1)) - A
    evals, evecs = np.linalg.eigh(L)
    evals = np.clip(evals, 0.0, None)
    return evals.astype(np.float32), evecs.astype(np.float32)


def unit_level_sae(X, area, n, y, w, eigvals, eigvecs, A=None, hyper=None):
    """Aggregated-cell version of the Dupuis/Parker unit-level model.

    X        (C, p) covariate matrix of the cells (includes intercept column)
    area     (C,)   district index of each cell
    n, y     (C,)   sampled individuals and sampled positives in the cell
    w        (C,)   normalized sampling weight shared by the cell's individuals
    eigvals, eigvecs : eigendecomposition of the Laplacian of the spatial graph
    A        (r, r) optional composition matrix (rows sum to one)
    """
    h = dict(HYPER_DEFAULT, **(hyper or {}))
    p = X.shape[1]
    r = eigvecs.shape[0]
    sigma_eta2 = numpyro.sample("sigma_eta2", dist.InverseGamma(h["a"], h["b"]))
    sigma_beta2 = numpyro.sample("sigma_beta2", dist.InverseGamma(h["c"], h["d"]))
    beta = numpyro.sample("beta", dist.Normal(0.0, jnp.sqrt(sigma_beta2)).expand([p]).to_event(1))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([r]).to_event(1))
    # eta = sigma_eta * Sigma_r^{1/2} z  with Sigma_r = (L + I/sigma_eta^2)^{-1} = V diag(1/(lam + 1/s2)) V'
    scale = jnp.sqrt(sigma_eta2) / jnp.sqrt(eigvals + 1.0 / sigma_eta2)
    eta = numpyro.deterministic("eta", eigvecs @ (scale * z))
    area_eff = eta if A is None else A @ eta
    psi = X @ beta + area_eff[area]
    # weighted binomial pseudo-likelihood, sum_i w_i [Z_i psi_i - log(1 + e^{psi_i})]
    numpyro.factor("pseudo_loglik", jnp.sum(w * (y * psi - n * jax.nn.softplus(psi))))


_MCMC_CACHE = {}


def _cached_mcmc(model, variant, num_warmup, num_samples, progress_bar=False):
    """One MCMC object per (model variant, settings) per process. With jit_model_args=True
    numpyro reuses the compiled sampler for new data of the same shapes, which avoids
    recompiling (and exhausting XLA's JIT code memory) across hundreds of fits."""
    key = (model.__name__, variant, num_warmup, num_samples, progress_bar)
    if key not in _MCMC_CACHE:
        kernel = NUTS(model, init_strategy=init_to_median(), target_accept_prob=0.8)
        _MCMC_CACHE[key] = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1,
                                progress_bar=progress_bar, jit_model_args=True)
    return _MCMC_CACHE[key]


def fit_sae(cells, eig, A_draws=None, num_warmup=500, num_samples=500, num_chains=2,
            seed=0, hyper=None, progress_bar=False):
    """Run NUTS. If A_draws is given (num_chains, r, r), chain c uses A_draws[c]."""
    eigvals, eigvecs = eig
    data = dict(
        X=jnp.asarray(cells["X"], jnp.float32),
        area=jnp.asarray(cells["area"], jnp.int32),
        n=jnp.asarray(cells["n"], jnp.float32),
        y=jnp.asarray(cells["y"], jnp.float32),
        w=jnp.asarray(cells["w"], jnp.float32),
        eigvals=jnp.asarray(eigvals), eigvecs=jnp.asarray(eigvecs),
    )
    mcmc = _cached_mcmc(unit_level_sae, "amatrix" if A_draws is not None else "naive",
                        num_warmup, num_samples, progress_bar)
    out = {"beta": [], "eta": [], "sigma_eta2": [], "sigma_beta2": [], "diverging": [], "A": []}
    rng = jax.random.PRNGKey(seed)
    for c in range(num_chains):
        rng, sub = jax.random.split(rng)
        A_c = None if A_draws is None else jnp.asarray(A_draws[c], jnp.float32)
        mcmc.run(sub, A=A_c, hyper=hyper, **data)
        s = mcmc.get_samples()
        for k in ("beta", "eta", "sigma_eta2", "sigma_beta2"):
            out[k].append(np.asarray(s[k]))
        out["diverging"].append(int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
        out["A"].append(None if A_draws is None else np.asarray(A_draws[c]))
    # stack by chain: (chains, samples, ...)
    for k in ("beta", "eta", "sigma_eta2", "sigma_beta2"):
        out[k] = np.stack(out[k])
    return out


def rhat(x):
    """Split-free Gelman-Rubin R-hat for an array (chains, samples, ...)."""
    x = np.asarray(x)
    m, n = x.shape[0], x.shape[1]
    chain_means = x.mean(1)
    chain_vars = x.var(1, ddof=1)
    B = n * chain_means.var(0, ddof=1)
    W = chain_vars.mean(0)
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def predict_district_proportions(fit, pop_cells, rng, A_draws=None):
    """Posterior predictive finite-population proportions per district.

    pop_cells: dict with 'X' (C,p), 'area' (C,), 'm' (C,) unsampled individuals per
    cell, 'y_obs' (r,) observed positives among sampled individuals in each district,
    'N' (r,) total individuals per district.
    Returns array (chains*samples, r) of district proportions.
    """
    X = np.asarray(pop_cells["X"], np.float32)
    area = np.asarray(pop_cells["area"])
    m = np.rint(np.asarray(pop_cells["m"])).astype(np.int64)
    r = len(pop_cells["N"])
    draws = []
    for c in range(fit["beta"].shape[0]):
        beta = fit["beta"][c]            # (S, p)
        eta = fit["eta"][c]              # (S, r)
        A_c = fit["A"][c] if A_draws is None else A_draws[c]
        area_eff = eta if A_c is None else eta @ np.asarray(A_c).T   # (S, r): row s = A eta_s
        psi = beta @ X.T + area_eff[:, area]                          # (S, C)
        theta = 1.0 / (1.0 + np.exp(-psi))
        Z = rng.binomial(m[None, :], theta)                           # (S, C)
        counts = np.zeros((Z.shape[0], r))
        np.add.at(counts.T, area, Z.T)                                 # sum cells into districts
        draws.append((pop_cells["y_obs"][None, :] + counts) / pop_cells["N"][None, :])
    return np.concatenate(draws, 0)
