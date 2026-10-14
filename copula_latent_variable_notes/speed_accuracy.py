"""Study 1: a copula-based joint model for speed and accuracy.

Simulation version of Kwon & Zhang (Psychometrika, in press; IMPS 2025):
van der Linden's (2007) hierarchical framework, but the dependence between
latent ability (theta) and latent speed (tau) is a Gumbel copula with
upper-tail dependence instead of a bivariate normal.

Data-generating process (person i, item j):

    Y_ij   ~ Bernoulli(logit^-1(a_j (theta_i - b_j)))      two-parameter logistic
    logT_ij ~ Normal(beta_j - tau_i, 1 / alpha_j)           log-normal response time
    theta_i ~ N(0, 1),  tau_i ~ N(0, sigma_tau^2)           margins
    (Phi(theta_i), Phi(tau_i / sigma_tau)) ~ C_gumbel(.; delta)   copula

Four NumPyro models are fit to the same data set:
    2pl_only   responses only (no response-time information)
    gaussian   van der Linden's bivariate-normal hierarchical model
    gumbel     the speaker's model (upper-tail dependence)
    clayton    an asymmetric copula with the *wrong* tail (lower)

Run:  uv run python speed_accuracy.py --out results/study1
      uv run python speed_accuracy.py --smoke          # 1-minute check
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
from numpyro.diagnostics import summary as numpyro_summary  # noqa: E402
from numpyro.infer import MCMC, NUTS, log_likelihood  # noqa: E402
from scipy.special import expit, logsumexp  # noqa: E402

import copulas as cp  # noqa: E402

MODELS = {
    "2pl_only": dict(family=None, use_rt=False),
    "gaussian": dict(family="gaussian", use_rt=True),
    "gumbel": dict(family="gumbel", use_rt=True),
    "clayton": dict(family="clayton", use_rt=True),
}
GLOBAL_SITES = ("a", "b", "beta", "alpha", "sigma_tau", "ktau", "rho")


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(n_persons, n_items, ktau=0.5, family="gumbel", rotation=0,
             sigma_tau=0.4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.0, 0.3, n_items)            # discrimination
    b = rng.normal(0.0, 1.0, n_items)               # difficulty
    beta = rng.normal(4.0, 0.4, n_items)            # time intensity (log seconds)
    alpha = rng.lognormal(np.log(2.0), 0.2, n_items)  # time precision (1 / residual sd)
    z1, z2 = cp.sample_z(family, ktau, n_persons, rng, rotation)
    theta, tau = z1, sigma_tau * z2
    p = expit(a * (theta[:, None] - b))
    Y = rng.binomial(1, p).astype(np.int64)
    logT = rng.normal(beta - tau[:, None], 1.0 / alpha)
    return dict(Y=Y, logT=logT, theta=theta, tau=tau, a=a, b=b, beta=beta,
                alpha=alpha, ktau=ktau, family=family, rotation=rotation,
                sigma_tau=sigma_tau)


# --------------------------------------------------------------------------
# NumPyro model
# --------------------------------------------------------------------------

def model(Y, logT=None, family=None, rotation=0):
    """Joint model.  logT=None drops the response-time part (2PL only);
    family=None with logT gives independent theta and tau."""
    n_persons, n_items = Y.shape
    use_rt = logT is not None

    a = numpyro.sample("a", dist.LogNormal(0.0, 0.5).expand([n_items]).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, 2.0).expand([n_items]).to_event(1))
    if use_rt:
        beta = numpyro.sample("beta", dist.Normal(4.0, 2.0).expand([n_items]).to_event(1))
        alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0).expand([n_items]).to_event(1))
        sigma_tau = numpyro.sample("sigma_tau", dist.HalfNormal(1.0))

    if use_rt and family == "gaussian":
        # van der Linden (2007): (theta, tau) bivariate normal
        rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
        numpyro.deterministic("ktau", 2.0 / jnp.pi * jnp.arcsin(rho))
        scale_tril = jnp.stack([
            jnp.stack([jnp.ones_like(rho), jnp.zeros_like(rho)]),
            jnp.stack([rho * sigma_tau, sigma_tau * jnp.sqrt(1.0 - rho ** 2)]),
        ])
        with numpyro.plate("persons", n_persons):
            z = numpyro.sample("z", dist.MultivariateNormal(jnp.zeros(2), scale_tril=scale_tril))
        theta = numpyro.deterministic("theta", z[:, 0])
        tau = z[:, 1]
    else:
        with numpyro.plate("persons", n_persons):
            theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))
            if use_rt:
                tau_std = numpyro.sample("tau_std", dist.Normal(0.0, 1.0))
        if use_rt:
            tau = sigma_tau * tau_std
            if family is not None:
                # Copula on the two latent variables: the joint density is
                # phi(theta) phi(tau_std) c(Phi(theta), Phi(tau_std)); the two
                # Normal(0, 1) draws above supply the margins, this factor the copula.
                ktau = numpyro.sample("ktau", dist.Beta(2.0, 2.0))
                numpyro.factor("copula", jnp.sum(
                    cp.logpdf_z(theta, tau_std, family, ktau, rotation)))

    with numpyro.plate("items", n_items, dim=-1), numpyro.plate("obs_persons", n_persons, dim=-2):
        numpyro.sample("Y", dist.Bernoulli(logits=a * (theta[:, None] - b)), obs=Y)
        if use_rt:
            numpyro.sample("logT", dist.Normal(beta - tau[:, None], 1.0 / alpha), obs=logT)


def fit(Y, logT, family, rotation, use_rt, warmup, samples, chains, seed):
    mcmc = MCMC(NUTS(model), num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), Y=jnp.asarray(Y),
             logT=jnp.asarray(logT) if use_rt else None, family=family, rotation=rotation)
    return mcmc, time.time() - t0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def global_diagnostics(mcmc):
    by_chain = mcmc.get_samples(group_by_chain=True)
    sub = {k: v for k, v in by_chain.items() if k in GLOBAL_SITES}
    summ = numpyro_summary(sub, prob=0.9)
    rhat = max(float(np.nanmax(summ[k]["r_hat"])) for k in sub)
    ess = min(float(np.nanmin(summ[k]["n_eff"])) for k in sub)
    out = dict(max_rhat_global=rhat, min_ess_global=ess)
    if "theta" in by_chain:
        s = numpyro_summary({"theta": by_chain["theta"]}, prob=0.9)["theta"]
        out.update(max_rhat_theta=float(np.nanmax(s["r_hat"])), min_ess_theta=float(np.nanmin(s["n_eff"])))
    return out


def theta_recovery(theta_hat, theta_true):
    err = theta_hat - theta_true
    terciles = np.quantile(theta_true, [1 / 3, 2 / 3])
    groups = np.digitize(theta_true, terciles)
    return dict(
        rmse=float(np.sqrt(np.mean(err ** 2))),
        corr=float(np.corrcoef(theta_hat, theta_true)[0, 1]),
        rmse_by_tercile=[float(np.sqrt(np.mean(err[groups == g] ** 2))) for g in range(3)],
        bias_by_tercile=[float(np.mean(err[groups == g])) for g in range(3)],
    )


def pointwise_loglik(mcmc, Y, logT, family, rotation, use_rt, n_draws=500):
    samples = mcmc.get_samples()
    n_total = next(iter(samples.values())).shape[0]
    idx = np.linspace(0, n_total - 1, min(n_draws, n_total)).astype(int)
    sub = {k: v[idx] for k, v in samples.items()}
    ll = log_likelihood(model, sub, Y=jnp.asarray(Y),
                        logT=jnp.asarray(logT) if use_rt else None,
                        family=family, rotation=rotation)
    # only the data sites: numpyro.factor registers the copula term as an observed site too
    return sum(np.asarray(ll[k]).sum(-1) for k in ("Y", "logT") if k in ll)   # (draws, persons)


def waic(ll):
    """Per-person WAIC from a (draws, persons) log-likelihood matrix."""
    n_draws = ll.shape[0]
    lppd = logsumexp(ll, axis=0) - np.log(n_draws)
    p_waic = ll.var(axis=0, ddof=1)
    elpd_i = lppd - p_waic
    return dict(elpd_waic=float(elpd_i.sum()), p_waic=float(p_waic.sum()),
                se=float(np.sqrt(len(elpd_i) * elpd_i.var(ddof=1)))), elpd_i


def psis_loo(ll):
    """PSIS-LOO via ArviZ if its API is available; otherwise None."""
    try:
        import arviz as az
        from arviz_base import from_dict
        idata = from_dict({"log_likelihood": {"obs": ll[None]}})
        res = az.loo(idata, var_name="obs")
        return dict(elpd_loo=float(res.elpd), se=float(res.se), p_loo=float(res.p),
                    pareto_k_max=float(np.max(np.asarray(res.pareto_k))),
                    frac_pareto_k_gt_0_7=float(np.mean(np.asarray(res.pareto_k) > 0.7)))
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc))


def rt_dispersion_by_score(Y, logT, n_bins=5):
    """Within score-quintile SD (and mean) of each person's mean log RT.

    This is the empirical pattern that motivates the model (Domingue et al.,
    2022): low scorers vary more in how long they take.
    """
    n = Y.shape[0]
    score = Y.sum(1)
    mean_logt = logT.mean(1)
    order = np.argsort(score, kind="stable")
    bins = np.empty(n, dtype=int)
    bins[order] = np.arange(n) * n_bins // n
    sd = np.array([mean_logt[bins == k].std(ddof=1) for k in range(n_bins)])
    mean = np.array([mean_logt[bins == k].mean() for k in range(n_bins)])
    return sd, mean


def posterior_predictive_dispersion(mcmc, data, family, n_reps=200, seed=0, n_bins=5):
    """Replicate the dispersion statistic under the fitted joint model."""
    samples = mcmc.get_samples()
    n_persons, n_items = data["Y"].shape
    n_total = samples["a"].shape[0]
    idx = np.linspace(0, n_total - 1, n_reps).astype(int)
    rng = np.random.default_rng(seed)
    sds, means = [], []
    for s in idx:
        a, b = np.asarray(samples["a"][s]), np.asarray(samples["b"][s])
        beta, alpha = np.asarray(samples["beta"][s]), np.asarray(samples["alpha"][s])
        sigma_tau = float(samples["sigma_tau"][s])
        if family is None:
            z1 = rng.normal(size=n_persons)
            z2 = rng.normal(size=n_persons)
        else:
            ktau = float(samples["ktau"][s])
            z1, z2 = cp.sample_z(family, ktau, n_persons, rng, data["rotation"])
        theta, tau = z1, sigma_tau * z2
        Y = rng.binomial(1, expit(a * (theta[:, None] - b)))
        logT = rng.normal(beta - tau[:, None], 1.0 / alpha)
        sd, mean = rt_dispersion_by_score(Y, logT, n_bins)
        sds.append(sd)
        means.append(mean)
    return np.array(sds), np.array(means)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-persons", type=int, default=1000)
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--ktau", type=float, default=0.5)
    ap.add_argument("--sigma-tau", type=float, default=0.4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default="results/study1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_persons, args.n_items = 200, 10
        args.warmup, args.samples, args.chains = 150, 150, 1
        args.out = "results/study1_smoke"
    os.makedirs(args.out, exist_ok=True)

    data = simulate(args.n_persons, args.n_items, ktau=args.ktau, sigma_tau=args.sigma_tau,
                    seed=args.seed)
    np.savez(os.path.join(args.out, "data.npz"), **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    obs_sd, obs_mean = rt_dispersion_by_score(data["Y"], data["logT"])
    lam_l, lam_u = cp.tail_dependence("gumbel", args.ktau)
    results = dict(
        config=vars(args),
        truth=dict(ktau=args.ktau, family="gumbel", sigma_tau=args.sigma_tau,
                   lambda_upper=lam_u, delta=float(cp.ktau_to_param("gumbel", args.ktau)),
                   observed_rt_sd_by_score_quintile=obs_sd.tolist(),
                   observed_rt_mean_by_score_quintile=obs_mean.tolist()),
        models={},
    )
    elpd_pointwise = {}
    for name in args.models.split(","):
        spec = MODELS[name]
        print(f"[{name}] fitting ...", flush=True)
        mcmc, secs = fit(data["Y"], data["logT"], spec["family"], 0, spec["use_rt"],
                         args.warmup, args.samples, args.chains, args.seed)
        samples = mcmc.get_samples()
        theta_hat = np.asarray(samples["theta"]).mean(0)
        theta_sd = np.asarray(samples["theta"]).std(0)
        rec = theta_recovery(theta_hat, data["theta"])
        diag = global_diagnostics(mcmc)
        entry = dict(seconds=secs, theta=rec, diagnostics=diag)
        if spec["use_rt"]:
            kt = np.asarray(samples["ktau"])
            entry["ktau"] = dict(mean=float(kt.mean()), sd=float(kt.std()),
                                 q05=float(np.quantile(kt, 0.05)), q95=float(np.quantile(kt, 0.95)))
            if spec["family"] in ("gumbel", "clayton"):
                lam = np.array([cp.tail_dependence(spec["family"], k) for k in kt])
                entry["lambda_lower"] = dict(mean=float(lam[:, 0].mean()), q05=float(np.quantile(lam[:, 0], 0.05)), q95=float(np.quantile(lam[:, 0], 0.95)))
                entry["lambda_upper"] = dict(mean=float(lam[:, 1].mean()), q05=float(np.quantile(lam[:, 1], 0.05)), q95=float(np.quantile(lam[:, 1], 0.95)))
            st = np.asarray(samples["sigma_tau"])
            entry["sigma_tau"] = dict(mean=float(st.mean()), sd=float(st.std()))
            ll = pointwise_loglik(mcmc, data["Y"], data["logT"], spec["family"], 0, True)
            w, elpd_i = waic(ll)
            elpd_pointwise[name] = elpd_i
            entry["waic"] = w
            entry["loo"] = psis_loo(ll)
            ppc_sd, ppc_mean = posterior_predictive_dispersion(mcmc, data, spec["family"], seed=args.seed)
            entry["ppc_rt_sd_by_score_quintile"] = dict(
                mean=ppc_sd.mean(0).tolist(), q05=np.quantile(ppc_sd, 0.05, axis=0).tolist(),
                q95=np.quantile(ppc_sd, 0.95, axis=0).tolist())
            np.savez(os.path.join(args.out, f"posterior_{name}.npz"), ktau=kt, sigma_tau=st,
                     theta_mean=theta_hat, theta_sd=theta_sd, ppc_sd=ppc_sd, ppc_mean=ppc_mean,
                     elpd_i=elpd_i, rho=np.asarray(samples["rho"]) if "rho" in samples else np.array([]))
        else:
            np.savez(os.path.join(args.out, f"posterior_{name}.npz"), theta_mean=theta_hat, theta_sd=theta_sd)
        results["models"][name] = entry
        print(f"[{name}] {secs:.0f}s  theta RMSE {rec['rmse']:.3f}  by tercile {np.round(rec['rmse_by_tercile'], 3)}  "
              f"ktau {entry.get('ktau', {}).get('mean', float('nan')):.3f}  diag {diag}", flush=True)
        with open(os.path.join(args.out, "summary.json"), "w") as fh:
            json.dump(results, fh, indent=2)

    # Pairwise elpd differences (per-person WAIC) with standard errors
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
