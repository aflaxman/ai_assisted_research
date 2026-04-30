"""ZNI-Binomial regression with logit-linked p, plus RCT simulators.

Used by 03_treatment_effect_rct.ipynb to evaluate how three methods recover
a treatment effect under increasing model misspecification:

  - naive logistic regression (ignores zero/N inflation)
  - mcount::mznib (marginalised ZNI binomial, frequentist)
  - NumPyro ZNI-Binomial regression with logit link on p (Bayesian)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize

numpyro.enable_validation(False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def zni_binomial_regression_model(N, X, y=None, n_obs=None):
    """ZNI-Binomial with logit-linked success probability.

    p_i = sigmoid(X_i @ beta).  pi_0 and pi_N are constant across rows.
    """
    if y is not None:
        n_obs = y.shape[0]
    n_features = X.shape[1]

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(3)))
    pi0, piN, pibin = pi[0], pi[1], pi[2]
    beta = numpyro.sample("beta", dist.Normal(0.0, 2.0).expand([n_features]).to_event(1))
    p = jax.nn.sigmoid(X @ beta)
    binom = dist.Binomial(total_count=N, probs=p)

    if y is None:
        with numpyro.plate("obs", n_obs):
            component = numpyro.sample("component", dist.Categorical(probs=pi))
            y_bin = numpyro.sample("y_bin", binom)
        y_out = jnp.where(component == 0, 0, jnp.where(component == 1, N, y_bin))
        numpyro.deterministic("y", y_out)
        return y_out

    log_pi0 = jnp.log(pi0)
    log_piN = jnp.log(piN)
    log_pibin = jnp.log(pibin)
    log_lik_bin = binom.log_prob(y) + log_pibin
    log_lik_zero = jnp.where(y == 0, log_pi0, -jnp.inf)
    log_lik_N = jnp.where(y == N, log_piN, -jnp.inf)
    log_lik = jax.scipy.special.logsumexp(
        jnp.stack([log_lik_zero, log_lik_N, log_lik_bin], axis=0), axis=0
    )
    numpyro.factor("zni_binomial_lik", log_lik.sum())


# ---------------------------------------------------------------------------
# RCT simulators - control p = 0.40, treated p = 0.55 by default
# ---------------------------------------------------------------------------
DEFAULTS = dict(n_per_arm=300, N=30, p_ctl=0.40, p_tx=0.55, pi0=0.10, piN=0.05)


def _design(rng, n_per_arm):
    n_obs = 2 * n_per_arm
    tx = np.concatenate([np.zeros(n_per_arm, int), np.ones(n_per_arm, int)])
    perm = rng.permutation(n_obs)
    return tx[perm], n_obs


def simulate_rct_matched(rng, **kw):
    cfg = {**DEFAULTS, **kw}
    tx, n_obs = _design(rng, cfg["n_per_arm"])
    p_each = np.where(tx == 1, cfg["p_tx"], cfg["p_ctl"])
    pibin = 1.0 - cfg["pi0"] - cfg["piN"]
    component = rng.choice(3, size=n_obs, p=[cfg["pi0"], cfg["piN"], pibin])
    y_bin = rng.binomial(cfg["N"], p_each)
    y = np.where(component == 0, 0, np.where(component == 1, cfg["N"], y_bin))
    return y, tx


def simulate_rct_inflation_shift(rng, pi0_ctl=0.20, pi0_tx=0.05, **kw):
    """Treatment also reduces structural-zero rate (e.g. reminders pull non-engagers in)."""
    cfg = {**DEFAULTS, **kw}
    tx, n_obs = _design(rng, cfg["n_per_arm"])
    pi0_each = np.where(tx == 1, pi0_tx, pi0_ctl)
    p_each = np.where(tx == 1, cfg["p_tx"], cfg["p_ctl"])
    piN = cfg["piN"]
    y = np.empty(n_obs, dtype=int)
    for i in range(n_obs):
        probs = [pi0_each[i], piN, 1.0 - pi0_each[i] - piN]
        c = rng.choice(3, p=probs)
        if c == 0:
            y[i] = 0
        elif c == 1:
            y[i] = cfg["N"]
        else:
            y[i] = rng.binomial(cfg["N"], p_each[i])
    return y, tx


def simulate_rct_heterogeneous(rng, p_tx_responder=0.75, p_tx_nonresp=0.40,
                               prop_resp=0.5, **kw):
    """Treatment works in only a fraction of treated subjects."""
    cfg = {**DEFAULTS, **kw}
    tx, n_obs = _design(rng, cfg["n_per_arm"])
    is_responder = (tx == 1) & (rng.uniform(size=n_obs) < prop_resp)
    p_each = np.where(
        tx == 0, cfg["p_ctl"],
        np.where(is_responder, p_tx_responder, p_tx_nonresp),
    )
    pibin = 1.0 - cfg["pi0"] - cfg["piN"]
    component = rng.choice(3, size=n_obs, p=[cfg["pi0"], cfg["piN"], pibin])
    y_bin = rng.binomial(cfg["N"], p_each)
    y = np.where(component == 0, 0, np.where(component == 1, cfg["N"], y_bin))
    return y, tx


def simulate_rct_overdisp(rng, kappa=8.0, **kw):
    """Per-subject p drawn from Beta around the arm mean (subject heterogeneity)."""
    cfg = {**DEFAULTS, **kw}
    tx, n_obs = _design(rng, cfg["n_per_arm"])
    mu = np.where(tx == 1, cfg["p_tx"], cfg["p_ctl"])
    a = kappa * mu
    b = kappa * (1.0 - mu)
    p_each = rng.beta(a, b)
    pibin = 1.0 - cfg["pi0"] - cfg["piN"]
    component = rng.choice(3, size=n_obs, p=[cfg["pi0"], cfg["piN"], pibin])
    y_bin = rng.binomial(cfg["N"], p_each)
    y = np.where(component == 0, 0, np.where(component == 1, cfg["N"], y_bin))
    return y, tx


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------
def fit_naive(y, tx, N):
    """Vanilla Binomial GLM with logit link, ignoring inflation."""
    def nll(params):
        a, b = params
        logit = a + b * tx
        # log-sigmoid stable form
        log_p = -np.logaddexp(0.0, -logit)
        log_1mp = -np.logaddexp(0.0, logit)
        return -(y * log_p + (N - y) * log_1mp).sum()

    res = minimize(nll, [0.0, 0.0], method="BFGS")
    a, b = res.x
    H = res.hess_inv
    se_b = float(np.sqrt(H[1, 1]))
    return dict(intercept=a, tx=b, tx_se=se_b,
                tx_ci=(b - 1.96 * se_b, b + 1.96 * se_b))


def fit_numpyro(y, tx, N, key=0, num_warmup=600, num_samples=1200):
    X = np.column_stack([np.ones_like(tx, dtype=float), tx.astype(float)])
    kernel = NUTS(zni_binomial_regression_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(key), N=int(N), X=jnp.asarray(X), y=jnp.asarray(y))
    s = mcmc.get_samples()
    beta = np.asarray(s["beta"])
    pi = np.asarray(s["pi"])
    p_ctl = 1.0 / (1.0 + np.exp(-beta[:, 0]))
    p_tx = 1.0 / (1.0 + np.exp(-(beta[:, 0] + beta[:, 1])))
    em_ctl = pi[:, 1] + pi[:, 2] * p_ctl
    em_tx = pi[:, 1] + pi[:, 2] * p_tx
    return dict(
        tx_logit=beta[:, 1],
        em_ctl=em_ctl, em_tx=em_tx,
        em_diff=em_tx - em_ctl,
        pi=pi, p_ctl=p_ctl, p_tx=p_tx,
    )


def summarise(samples):
    return float(np.mean(samples)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
