"""Study 2: copula-based SEM for predicting a rare binary risk outcome.

A reconstruction, from the seminar abstract, of Kwon & Zhang's "Predicting
binary risk outcomes via copula-based SEM" (in preparation).  The paper is
not public, so this is one plausible reading of it: a one-factor
measurement model for a latent risk factor, and an asymmetric copula
between that factor and the latent propensity behind a rare binary outcome.

Data-generating process (person i, indicator k):

    eta_i  ~ N(0, 1)                                 latent risk factor
    x_ik   ~ N(nu_k + lambda_k eta_i, sigma_k^2)     K continuous indicators (CFA)
    y*_i   ~ N(0, 1)                                 latent outcome propensity
    (Phi(eta_i), Phi(y*_i)) ~ C(. ; ktau)            copula between the latent variables
    Y_i    = 1{y*_i > Phi^-1(1 - p)}                 rare outcome, prevalence p

Under a Gaussian copula this is the usual probit SEM with latent correlation
rho:  P(Y=1 | eta) = Phi((rho eta - c) / sqrt(1 - rho^2)).  Under a Gumbel
copula P(Y=1 | eta) = 1 - h(v_c | Phi(eta)), with h the copula's conditional
CDF, which rises far more steeply in the upper tail of eta.  Only the
conditional CDF is needed, so y* never has to be sampled.

Run:  uv run python risk_sem.py --out results/study2
      uv run python risk_sem.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from jax.scipy.special import ndtri  # noqa: E402
from numpyro.diagnostics import summary as numpyro_summary  # noqa: E402
from numpyro.infer import MCMC, NUTS, log_likelihood  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

import copulas as cp  # noqa: E402

FAMILIES = ("gaussian", "gumbel", "clayton")
GLOBAL_SITES = ("lam", "nu", "sig", "prev", "ktau", "rho")


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(n, n_indicators=6, ktau=0.5, family="gumbel", rotation=0,
             prevalence=0.03, seed=0, params=None):
    rng = np.random.default_rng(seed)
    if params is None:
        params = dict(lam=rng.uniform(0.5, 0.9, n_indicators),
                      nu=rng.normal(0.0, 0.5, n_indicators),
                      sig=rng.uniform(0.5, 0.8, n_indicators))
    lam, nu, sig = params["lam"], params["nu"], params["sig"]
    u, v = cp.sample(family, ktau, n, rng, rotation)
    eta = sps.norm.ppf(u)
    Y = (v > 1.0 - prevalence).astype(np.int64)
    X = nu + lam * eta[:, None] + rng.normal(0.0, sig, size=(n, n_indicators))
    return dict(X=X, Y=Y, eta=eta, ystar=sps.norm.ppf(v), lam=lam, nu=nu, sig=sig,
                ktau=ktau, family=family, rotation=rotation, prevalence=prevalence)


# --------------------------------------------------------------------------
# NumPyro model
# --------------------------------------------------------------------------

def model(X, Y=None, family="gumbel", rotation=0):
    n, k = X.shape
    lam = numpyro.sample("lam", dist.LogNormal(0.0, 0.5).expand([k]).to_event(1))  # loadings > 0
    nu = numpyro.sample("nu", dist.Normal(0.0, 2.0).expand([k]).to_event(1))
    sig = numpyro.sample("sig", dist.HalfNormal(1.0).expand([k]).to_event(1))
    prev = numpyro.sample("prev", dist.Beta(1.0, 1.0))        # marginal prevalence of Y
    if family == "gaussian":
        rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
        ktau = numpyro.deterministic("ktau", 2.0 / jnp.pi * jnp.arcsin(rho))
    else:
        ktau = numpyro.sample("ktau", dist.Beta(2.0, 2.0))

    with numpyro.plate("persons", n):
        eta = numpyro.sample("eta", dist.Normal(0.0, 1.0))
    with numpyro.plate("indicators", k, dim=-1), numpyro.plate("obs_persons", n, dim=-2):
        numpyro.sample("X", dist.Normal(nu + lam * eta[:, None], sig), obs=X)

    # Structural part: P(Y = 1 | eta) = P(y* > c | eta) from the copula's conditional CDF
    c = ndtri(1.0 - prev)
    log_h = cp.log_hfunc_z(eta, c, family, ktau, rotation)            # log P(Y = 0 | eta)
    log_s = cp.log_survival_hfunc_z(eta, c, family, ktau, rotation)   # log P(Y = 1 | eta)
    with numpyro.plate("persons_y", n):
        numpyro.sample("Y", dist.Bernoulli(logits=log_s - log_h), obs=Y)


def fit(X, Y, family, rotation, warmup, samples, chains, seed):
    mcmc = MCMC(NUTS(model), num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.asarray(X), Y=jnp.asarray(Y),
             family=family, rotation=rotation)
    return mcmc, time.time() - t0


# --------------------------------------------------------------------------
# Prediction for new persons
# --------------------------------------------------------------------------

def prob_outcome_given_eta(eta, prev, family, ktau, rotation=0):
    c = ndtri(1.0 - prev)
    return jnp.exp(cp.log_survival_hfunc_z(eta, c, family, ktau, rotation))


def predict_risk(X_new, draws, family, rotation=0, n_quad=24, chunk=25):
    """Posterior predictive P(Y = 1 | x) for new persons.

    eta | x is Gaussian under the linear measurement model, so the integral
    over eta is done with Gauss-Hermite quadrature; the posterior over the
    model parameters is handled by averaging over draws.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    weights = weights / weights.sum()
    nodes, weights = jnp.asarray(nodes), jnp.asarray(weights)
    X_new = jnp.asarray(X_new)

    def one_draw(lam, nu, sig, prev, ktau):
        prec = 1.0 + jnp.sum(lam ** 2 / sig ** 2)
        mean = jnp.sum(lam * (X_new - nu) / sig ** 2, axis=1) / prec
        sd = 1.0 / jnp.sqrt(prec)
        eta = mean[:, None] + sd * nodes[None, :]
        p = prob_outcome_given_eta(eta, prev, family, ktau, rotation)
        return jnp.sum(p * weights[None, :], axis=1)

    batched = jax.jit(jax.vmap(one_draw))
    out = []
    n_draws = len(draws["prev"])
    for start in range(0, n_draws, chunk):
        sl = slice(start, start + chunk)
        out.append(np.asarray(batched(draws["lam"][sl], draws["nu"][sl], draws["sig"][sl],
                                      draws["prev"][sl], draws["ktau"][sl])))
    return np.concatenate(out, axis=0).mean(0)


def evaluate_predictions(p, y, top_fracs=(0.01, 0.05, 0.20)):
    order = np.argsort(-p)
    out = dict(auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, p)),
               brier=float(brier_score_loss(y, p)), mean_pred=float(p.mean()), prevalence=float(y.mean()))
    for f in top_fracs:
        top = order[: int(round(f * len(p)))]
        out[f"top{int(f * 100)}pct"] = dict(
            observed_rate=float(y[top].mean()), predicted_rate=float(p[top].mean()),
            share_of_events=float(y[top].sum() / y.sum()))
    return out


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def global_diagnostics(mcmc):
    by_chain = mcmc.get_samples(group_by_chain=True)
    sub = {k: v for k, v in by_chain.items() if k in GLOBAL_SITES}
    summ = numpyro_summary(sub, prob=0.9)
    out = dict(max_rhat_global=max(float(np.nanmax(summ[k]["r_hat"])) for k in sub),
               min_ess_global=min(float(np.nanmin(summ[k]["n_eff"])) for k in sub))
    s = numpyro_summary({"eta": by_chain["eta"]}, prob=0.9)["eta"]
    out.update(max_rhat_eta=float(np.nanmax(s["r_hat"])), min_ess_eta=float(np.nanmin(s["n_eff"])))
    return out


def waic(ll):
    n_draws = ll.shape[0]
    lppd = logsumexp(ll, axis=0) - np.log(n_draws)
    p_waic = ll.var(axis=0, ddof=1)
    elpd_i = lppd - p_waic
    return dict(elpd_waic=float(elpd_i.sum()), p_waic=float(p_waic.sum()),
                se=float(np.sqrt(len(elpd_i) * elpd_i.var(ddof=1)))), elpd_i


def thin(samples, n_draws):
    n_total = next(iter(samples.values())).shape[0]
    idx = np.linspace(0, n_total - 1, min(n_draws, n_total)).astype(int)
    return {k: v[idx] for k, v in samples.items()}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--n-indicators", type=int, default=6)
    ap.add_argument("--ktau", type=float, default=0.5)
    ap.add_argument("--prevalence", type=float, default=0.03)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--out", default="results/study2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_test = 500, 2000
        args.warmup, args.samples, args.chains = 150, 150, 1
        args.out = "results/study2_smoke"
    os.makedirs(args.out, exist_ok=True)

    train = simulate(args.n_train, args.n_indicators, ktau=args.ktau, prevalence=args.prevalence,
                     seed=args.seed)
    params = dict(lam=train["lam"], nu=train["nu"], sig=train["sig"])
    test = simulate(args.n_test, args.n_indicators, ktau=args.ktau, prevalence=args.prevalence,
                    seed=args.seed + 1, params=params)
    np.savez(os.path.join(args.out, "data.npz"),
             **{f"train_{k}": v for k, v in train.items() if isinstance(v, np.ndarray)},
             **{f"test_{k}": v for k, v in test.items() if isinstance(v, np.ndarray)})

    eta_grid = np.linspace(-3.0, 4.0, 141)
    truth_curve = np.asarray(prob_outcome_given_eta(jnp.asarray(eta_grid), args.prevalence, "gumbel", args.ktau))
    oracle = dict(lam=jnp.asarray(train["lam"])[None], nu=jnp.asarray(train["nu"])[None],
                  sig=jnp.asarray(train["sig"])[None], prev=jnp.asarray([args.prevalence]),
                  ktau=jnp.asarray([args.ktau]))
    p_oracle = predict_risk(test["X"], oracle, "gumbel")
    p_eta_oracle = np.asarray(prob_outcome_given_eta(jnp.asarray(test["eta"]), args.prevalence, "gumbel", args.ktau))
    results = dict(
        config=vars(args),
        truth=dict(family="gumbel", ktau=args.ktau, prevalence=args.prevalence,
                   lambda_upper=cp.tail_dependence("gumbel", args.ktau)[1],
                   train_events=int(train["Y"].sum()), test_events=int(test["Y"].sum()),
                   joint_tail_p_eta_gt_q90_and_Y=float(cp.joint_upper_tail(0.9, 1 - args.prevalence, "gumbel", args.ktau)),
                   joint_tail_p_eta_gt_q99_and_Y=float(cp.joint_upper_tail(0.99, 1 - args.prevalence, "gumbel", args.ktau)),
                   oracle_from_indicators=evaluate_predictions(p_oracle, test["Y"]),
                   oracle_from_true_eta=evaluate_predictions(p_eta_oracle, test["Y"])),
        models={},
    )
    np.savez(os.path.join(args.out, "curves_truth.npz"), eta_grid=eta_grid, truth=truth_curve,
             p_oracle=p_oracle, p_eta_oracle=p_eta_oracle)
    elpd_pointwise = {}
    for family in args.families.split(","):
        print(f"[{family}] fitting ...", flush=True)
        mcmc, secs = fit(train["X"], train["Y"], family, 0, args.warmup, args.samples, args.chains, args.seed)
        samples = mcmc.get_samples()
        diag = global_diagnostics(mcmc)
        kt = np.asarray(samples["ktau"])
        prev = np.asarray(samples["prev"])
        entry = dict(seconds=secs, diagnostics=diag,
                     ktau=dict(mean=float(kt.mean()), q05=float(np.quantile(kt, 0.05)), q95=float(np.quantile(kt, 0.95))),
                     prevalence=dict(mean=float(prev.mean()), q05=float(np.quantile(prev, 0.05)), q95=float(np.quantile(prev, 0.95))),
                     loadings_mean=np.asarray(samples["lam"]).mean(0).tolist())
        if family != "gaussian":
            lam_tail = np.array([cp.tail_dependence(family, k) for k in kt])
            entry["lambda_lower"] = float(lam_tail[:, 0].mean())
            entry["lambda_upper"] = float(lam_tail[:, 1].mean())
        # model-implied joint tail probabilities
        for q in (0.9, 0.99):
            jt = np.array([cp.joint_upper_tail(q, 1 - p, family, k) for p, k in zip(prev[::10], kt[::10])])
            entry[f"joint_tail_p_eta_gt_q{int(q * 100)}_and_Y"] = float(jt.mean())
        # P(Y = 1 | eta) curve with posterior band
        sub = thin(samples, 400)
        curves = np.asarray(jax.vmap(lambda p, k: prob_outcome_given_eta(jnp.asarray(eta_grid), p, family, k))(
            sub["prev"], sub["ktau"]))
        # per-person WAIC (train)
        ll_dict = log_likelihood(model, thin(samples, 500), X=jnp.asarray(train["X"]), Y=jnp.asarray(train["Y"]),
                                 family=family, rotation=0)
        ll = sum(np.asarray(ll_dict[k]).reshape(ll_dict[k].shape[0], ll_dict[k].shape[1], -1).sum(-1)
                 for k in ("X", "Y"))
        w, elpd_i = waic(ll)
        elpd_pointwise[family] = elpd_i
        entry["waic"] = w
        # out-of-sample prediction
        p_test = predict_risk(test["X"], thin(samples, 200), family)
        entry["test"] = evaluate_predictions(p_test, test["Y"])
        results["models"][family] = entry
        np.savez(os.path.join(args.out, f"posterior_{family}.npz"), ktau=kt, prev=prev,
                 curves=curves, p_test=p_test, elpd_i=elpd_i,
                 eta_mean=np.asarray(samples["eta"]).mean(0))
        print(f"[{family}] {secs:.0f}s ktau {kt.mean():.3f} prev {prev.mean():.4f} "
              f"AUC {entry['test']['auc']:.3f} logloss {entry['test']['log_loss']:.4f} "
              f"top1% obs {entry['test']['top1pct']['observed_rate']:.3f} pred {entry['test']['top1pct']['predicted_rate']:.3f} "
              f"diag {diag}", flush=True)
        with open(os.path.join(args.out, "summary.json"), "w") as fh:
            json.dump(results, fh, indent=2)

    names = list(elpd_pointwise)
    comparisons = {}
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            d = elpd_pointwise[m1] - elpd_pointwise[m2]
            comparisons[f"{m1}_minus_{m2}"] = dict(elpd_diff=float(d.sum()), se=float(np.sqrt(len(d) * d.var(ddof=1))))
    results["waic_comparisons"] = comparisons
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
