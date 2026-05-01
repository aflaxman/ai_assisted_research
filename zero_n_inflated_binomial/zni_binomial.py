"""Zero- and N-inflated Binomial model in NumPyro.

Stages of increasing data complexity:
  1. Matched: data drawn from the same ZNI-Binomial used in fitting.
  2. Heterogeneous p: each observation has its own p drawn from a Beta.
  3. Mixture of p: two latent groups with very different success rates.
  4. Variable N: trial counts vary per observation (model assumes fixed N).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_host_device_count(1)
numpyro.enable_validation(False)  # log_factor uses logsumexp with -inf placeholders


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def zni_binomial_model(N, y=None, n_obs=None):
    """Zero- and N-inflated Binomial.

    Mixture: with probability pi0 emit 0, with pi_N emit N, otherwise Binomial(N, p).
    """
    if y is not None:
        n_obs = y.shape[0]

    pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(3)))
    pi0, piN, pibin = pi[0], pi[1], pi[2]
    p = numpyro.sample("p", dist.Beta(1.0, 1.0))
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
# Simulators
# ---------------------------------------------------------------------------
def simulate_matched(rng, n_obs, N, pi0=0.15, piN=0.10, p=0.35):
    """Stage 1: data exactly from the ZNI-Binomial."""
    pibin = 1.0 - pi0 - piN
    component = rng.choice(3, size=n_obs, p=[pi0, piN, pibin])
    y_bin = rng.binomial(N, p, size=n_obs)
    y = np.where(component == 0, 0, np.where(component == 1, N, y_bin))
    return y, np.full(n_obs, N)


def simulate_overdispersed(rng, n_obs, N, pi0=0.15, piN=0.10, alpha=2.0, beta=4.0):
    """Stage 2: each obs draws its own p from Beta(alpha, beta) -> Beta-Binomial body."""
    pibin = 1.0 - pi0 - piN
    component = rng.choice(3, size=n_obs, p=[pi0, piN, pibin])
    p_each = rng.beta(alpha, beta, size=n_obs)
    y_bin = rng.binomial(N, p_each)
    y = np.where(component == 0, 0, np.where(component == 1, N, y_bin))
    return y, np.full(n_obs, N)


def simulate_mixture(rng, n_obs, N, pi0=0.15, piN=0.10, p_low=0.15, p_high=0.75, mix=0.5):
    """Stage 3: body is a mixture of two binomials with different p."""
    pibin = 1.0 - pi0 - piN
    component = rng.choice(3, size=n_obs, p=[pi0, piN, pibin])
    group = rng.binomial(1, mix, size=n_obs)
    p_each = np.where(group == 1, p_high, p_low)
    y_bin = rng.binomial(N, p_each)
    y = np.where(component == 0, 0, np.where(component == 1, N, y_bin))
    return y, np.full(n_obs, N)


def simulate_variable_N(rng, n_obs, N_max, pi0=0.15, piN=0.10, p=0.35, N_min=8):
    """Stage 4: N varies per observation in [N_min, N_max]; model assumes fixed N_max."""
    pibin = 1.0 - pi0 - piN
    component = rng.choice(3, size=n_obs, p=[pi0, piN, pibin])
    Ns = rng.integers(N_min, N_max + 1, size=n_obs)
    y_bin = rng.binomial(Ns, p)
    y = np.where(component == 0, 0, np.where(component == 1, Ns, y_bin))
    return y, Ns


# ---------------------------------------------------------------------------
# Fitting + posterior predictive
# ---------------------------------------------------------------------------
def fit_and_ppc(y, N, key, num_warmup=800, num_samples=1500, n_ppc=400):
    kernel = NUTS(zni_binomial_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(key, N=int(N), y=jnp.asarray(y))
    samples = mcmc.get_samples()

    pred_key = jax.random.fold_in(key, 1)
    predictive = Predictive(zni_binomial_model, posterior_samples=samples, num_samples=n_ppc)
    ppc = predictive(pred_key, N=int(N), n_obs=y.shape[0])
    return samples, np.asarray(ppc["y"])


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def build_stages(rng, n_obs=600, N=20):
    return [
        dict(
            name="Stage 1 - Matched",
            blurb="Data drawn from the model used to fit.",
            data=simulate_matched(rng, n_obs, N),
            true_p=0.35,
        ),
        dict(
            name="Stage 2 - Beta-Binomial overdispersion",
            blurb="Per-obs p ~ Beta(2, 4); body is now overdispersed.",
            data=simulate_overdispersed(rng, n_obs, N),
            true_p=2.0 / (2.0 + 4.0),
        ),
        dict(
            name="Stage 3 - Two-component mixture in p",
            blurb="Half the obs use p=0.15, half use p=0.75; bimodal body.",
            data=simulate_mixture(rng, n_obs, N),
            true_p=0.5 * 0.15 + 0.5 * 0.75,
        ),
        dict(
            name="Stage 4 - Variable trial counts N_i",
            blurb="N varies per obs; the model assumes a single fixed N.",
            data=simulate_variable_N(rng, n_obs, N),
            true_p=0.35,
        ),
    ]


# ---------------------------------------------------------------------------
# Plotting one frame
# ---------------------------------------------------------------------------
def plot_frame(stage, samples, ppc, N, out_path):
    y, _ = stage["data"]
    bins = np.arange(0, N + 2) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(y, bins=bins, density=True, alpha=0.55, color="#3366cc",
            edgecolor="white", label="Observed")
    ppc_flat = ppc.reshape(-1)
    ax.hist(ppc_flat, bins=bins, density=True, histtype="step",
            color="#cc3333", linewidth=2.0, label="Posterior predictive")
    ax.set_xlabel("y (successes out of N)")
    ax.set_ylabel("density")
    ax.set_title(stage["name"])
    ax.legend(loc="upper center")
    ax.set_xlim(-0.5, N + 0.5)
    ax.text(0.5, -0.22, stage["blurb"], transform=ax.transAxes,
            ha="center", fontsize=9, style="italic")

    ax = axes[1]
    pi = np.asarray(samples["pi"])
    p = np.asarray(samples["p"])
    labels = [r"$\pi_0$", r"$\pi_N$", "p"]
    data = [pi[:, 0], pi[:, 1], p]
    parts = ax.violinplot(data, positions=range(3), showmedians=True, widths=0.8)
    for body, color in zip(parts["bodies"], ["#dd8855", "#55aa88", "#5577dd"]):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Posterior parameters")
    ax.axhline(0.15, ls="--", color="#dd8855", alpha=0.6, lw=1)
    ax.axhline(0.10, ls="--", color="#55aa88", alpha=0.6, lw=1)
    ax.axhline(stage["true_p"], ls="--", color="#5577dd", alpha=0.6, lw=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir="frames", gif_path="zni_binomial_stages.gif"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    N = 20

    stages = build_stages(rng, n_obs=600, N=N)

    frame_paths = []
    for i, stage in enumerate(stages):
        y, _ = stage["data"]
        print(f"Fitting {stage['name']} (n={len(y)}, mean={y.mean():.2f})")
        sub_key = jax.random.fold_in(key, i)
        samples, ppc = fit_and_ppc(y, N, sub_key)
        path = os.path.join(out_dir, f"stage_{i:02d}.png")
        plot_frame(stage, samples, ppc, N, path)
        frame_paths.append(path)

    import imageio.v2 as imageio

    images = [imageio.imread(p) for p in frame_paths]
    images_held = []
    for img in images:
        images_held.extend([img] * 3)
    imageio.mimsave(gif_path, images_held, duration=1.2, loop=0)
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()
