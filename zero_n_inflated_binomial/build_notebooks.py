"""Generate and execute the two walkthrough notebooks.

Run with: uv run python build_notebooks.py
"""

from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


# ---------------------------------------------------------------------------
# Notebook 1: NumPyro walkthrough
# ---------------------------------------------------------------------------
nb1_cells = [
    md(
        """
# NumPyro walkthrough: Zero- and N-inflated Binomial

This notebook walks through the NumPyro implementation of a ZNI-Binomial
model. We:

1. Define the generative model.
2. Simulate matched data and confirm the fit recovers truth.
3. Progressively misspecify the data and watch the posterior predictive
   degrade.

The companion notebook `02_compare_mcount.ipynb` repeats the same fits with
the R package `mcount::mznib` and compares.
"""
    ),
    md(
        """
## 1. The model

For a count `y_i ∈ {0, 1, ..., N}`,

```
y_i = 0      with prob  π_0     (structural zero)
y_i = N      with prob  π_N     (structural max)
y_i ~ Bin(N, p)  with prob  1 - π_0 - π_N
```

Priors: `π = (π_0, π_N, π_bin) ~ Dirichlet(1, 1, 1)`, `p ~ Beta(1, 1)`. The
discrete component label is marginalised analytically with `logsumexp`, so
NUTS sees a smooth log-density in `(π, p)`.
"""
    ),
    code(
        """
import numpy as np
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt

from zni_binomial import (
    zni_binomial_model, fit_and_ppc,
    simulate_matched, simulate_overdispersed,
    simulate_mixture, simulate_variable_N,
)

np.set_printoptions(precision=3, suppress=True)
"""
    ),
    md(
        """
## 2. Simulate matched data and fit

True parameters: `π_0 = 0.15`, `π_N = 0.10`, `p = 0.35`, `N = 20`.
"""
    ),
    code(
        """
rng = np.random.default_rng(0)
N = 20
y, _ = simulate_matched(rng, n_obs=600, N=N)

bins = np.arange(0, N + 2) - 0.5
plt.figure(figsize=(6, 3))
plt.hist(y, bins=bins, color="#3366cc", edgecolor="white")
plt.title("Matched data: ZNI-Binomial(N=20, π_0=0.15, π_N=0.10, p=0.35)")
plt.xlabel("y"); plt.ylabel("count")
plt.show()

print("mean y/N:", (y / N).mean(), "  zeros:", (y == 0).mean(),
      "  N's:", (y == N).mean())
"""
    ),
    code(
        """
samples, ppc = fit_and_ppc(y, N, jax.random.PRNGKey(0))
print("posterior means:")
print(f"  π_0 = {samples['pi'][:,0].mean():.3f}")
print(f"  π_N = {samples['pi'][:,1].mean():.3f}")
print(f"  p   = {samples['p'].mean():.3f}")
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(y, bins=bins, density=True, alpha=0.55, color="#3366cc",
        edgecolor="white", label="observed")
ax.hist(np.asarray(ppc).reshape(-1), bins=bins, density=True,
        histtype="step", color="#cc3333", linewidth=2.0,
        label="posterior predictive")
ax.legend(); ax.set_xlabel("y"); ax.set_title("Stage 1 - matched")
plt.show()
"""
    ),
    md(
        """
## 3. Progressively break the model

We feed four datasets of increasing complexity into the same model and
inspect posterior predictive checks.
"""
    ),
    code(
        """
stages = {
    "Stage 1: matched":
        lambda r: simulate_matched(r, 600, N),
    "Stage 2: Beta-Binomial overdispersion":
        lambda r: simulate_overdispersed(r, 600, N),
    "Stage 3: two-component mixture in p":
        lambda r: simulate_mixture(r, 600, N),
    "Stage 4: variable trial counts":
        lambda r: simulate_variable_N(r, 600, N),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
key = jax.random.PRNGKey(0)
for i, (name, gen) in enumerate(stages.items()):
    rng = np.random.default_rng(i)
    yi, _ = gen(rng)
    s, p = fit_and_ppc(yi, N, jax.random.fold_in(key, i))
    ax = axes.flat[i]
    ax.hist(yi, bins=bins, density=True, alpha=0.55, color="#3366cc",
            edgecolor="white", label="observed")
    ax.hist(np.asarray(p).reshape(-1), bins=bins, density=True,
            histtype="step", color="#cc3333", linewidth=2.0,
            label="post. pred.")
    ax.set_title(name); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""
    ),
    md(
        """
**Reading the panels.** Stage 1 fits perfectly. Stage 2 (Beta-Binomial body)
has heavier tails than any single Binomial; the model mostly absorbs that as
extra zero/N inflation. Stage 3 (bimodal `p` mixture) is the worst — the
model can only put one bump in the body. Stage 4 (varying N_i) collapses
`π_N` to zero because no observation reaches the assumed maximum unless drawn
from the body component.

For a side-by-side animated view, see `zni_binomial_stages.gif`. For the same
data fit with `mcount::mznib` in R, see `02_compare_mcount.ipynb`.
"""
    ),
]

nb1 = nbf.v4.new_notebook(cells=nb1_cells)
nb1["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}


# ---------------------------------------------------------------------------
# Notebook 2: comparison with mcount::mznib
# ---------------------------------------------------------------------------
nb2_cells = [
    md(
        """
# Comparison: NumPyro vs `mcount::mznib`

The R package
[`mcount`](https://cran.r-project.org/package=mcount) (Zhou et al., 2024)
implements a *marginalized* zero- and N-inflated Binomial via `mznib()`.
It fits by maximum likelihood and reports nonparametric bootstrap p-values.

This notebook fits the same simulated data with both approaches and lays
their outputs side by side.

| Aspect | NumPyro `zni_binomial_model` | `mcount::mznib` |
|---|---|---|
| Inference | Bayesian (NUTS) | Frequentist MLE |
| Uncertainty | Posterior samples / credible intervals | Bootstrap CIs |
| Parameters reported | π_0, π_N, p individually | logit(E[y/N]) regression coefs |
| Covariate effects act on | latent `p` (and `π` via separate priors) | the marginal proportion `E[y/N]` |
| Speed | seconds (NUTS warm-up + samples) | seconds (MLE) + bootstrap |

The marginalized parameterization is the key conceptual difference:
`mznib` describes how covariates shift the *observed* proportion, while the
NumPyro model describes how they shift the *latent* success probability.
For an intercept-only fit on simulated data, both should agree on
`E[y/N]`.
"""
    ),
    code(
        """
import numpy as np, pandas as pd
import jax
import matplotlib.pyplot as plt
from zni_binomial import (
    fit_and_ppc, simulate_matched, simulate_overdispersed,
    simulate_mixture, simulate_variable_N,
)

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
pandas2ri.activate(); numpy2ri.activate()
ro.r('suppressPackageStartupMessages(library(mcount))')
print("mcount version:", ro.r('as.character(packageVersion(\"mcount\"))')[0])
"""
    ),
    md(
        """
## 1. Helper: fit both models on the same dataset
"""
    ),
    code(
        """
N = 20
bins = np.arange(0, N + 2) - 0.5

def fit_mznib(y, Ns, R=200, seed=1):
    df = pd.DataFrame({"y": y.astype(int), "N_i": Ns.astype(int)})
    ro.globalenv["df"] = df
    ro.r(f'fit <- mznib(y ~ 1, data=df, N_i=df$N_i, R={R}, seed={seed}L)')
    intercept = float(ro.r('fit$coef$estimate[1]')[0])  # logit(E[y/N])
    se        = float(ro.r('fit$coef$SE[1]')[0])
    ci_lo     = float(ro.r('fit$coef$CI_lower[1]')[0])
    ci_hi     = float(ro.r('fit$coef$CI_upper[1]')[0])
    p_marg    = 1.0 / (1.0 + np.exp(-intercept))
    return dict(logit=intercept, se=se, ci=(ci_lo, ci_hi), p_marg=p_marg)


def fit_numpyro(y, N, key=0):
    samples, ppc = fit_and_ppc(y, N, jax.random.PRNGKey(key))
    pi = np.asarray(samples["pi"])
    p = np.asarray(samples["p"])
    pi0, piN = pi[:, 0], pi[:, 1]
    pibin = pi[:, 2]
    p_marg = piN + pibin * p          # E[y/N] under the model
    return dict(
        pi0=(pi0.mean(), np.quantile(pi0, [.025, .975])),
        piN=(piN.mean(), np.quantile(piN, [.025, .975])),
        p=(p.mean(), np.quantile(p, [.025, .975])),
        p_marg=(p_marg.mean(), np.quantile(p_marg, [.025, .975])),
        ppc=np.asarray(ppc),
    )
"""
    ),
    md(
        """
## 2. Stage 1 — matched data

Both methods should be in close agreement on `E[y/N]`.
"""
    ),
    code(
        """
rng = np.random.default_rng(0)
y, Ns = simulate_matched(rng, 600, N)

m = fit_mznib(y, Ns, R=200)
b = fit_numpyro(y, N, key=0)

print("Empirical mean y/N : ", (y / N).mean())
print("mznib  E[y/N]      : {:.3f}  (95% CI [{:.3f}, {:.3f}])".format(
    m["p_marg"], 1/(1+np.exp(-m["ci"][0])), 1/(1+np.exp(-m["ci"][1]))))
print("NumPyro E[y/N]     : {:.3f}  (95% CrI [{:.3f}, {:.3f}])".format(
    b["p_marg"][0], *b["p_marg"][1]))
print()
print("NumPyro structural zero π_0 : {:.3f}  CrI {}".format(*b["pi0"]))
print("NumPyro structural N    π_N : {:.3f}  CrI {}".format(*b["piN"]))
print("NumPyro success rate    p   : {:.3f}  CrI {}".format(*b["p"]))
print("(mcount does not expose π_0, π_N, p separately for an intercept-only fit.)")
"""
    ),
    md(
        """
## 3. All four stages — side-by-side comparison
"""
    ),
    code(
        """
stages = [
    ("Stage 1: matched",
        lambda r: simulate_matched(r, 600, N)),
    ("Stage 2: Beta-Bin overdispersion",
        lambda r: simulate_overdispersed(r, 600, N)),
    ("Stage 3: bimodal p mixture",
        lambda r: simulate_mixture(r, 600, N)),
    ("Stage 4: variable N_i",
        lambda r: simulate_variable_N(r, 600, N)),
]

records = []
fig, axes = plt.subplots(4, 1, figsize=(8, 11))
for i, (name, gen) in enumerate(stages):
    yi, Nsi = gen(np.random.default_rng(i))
    m = fit_mznib(yi, Nsi, R=200, seed=i+1)
    b = fit_numpyro(yi, N, key=i)
    records.append({
        "stage": name,
        "empirical mean y/N": (yi / Nsi).mean(),
        "mznib E[y/N]": m["p_marg"],
        "NumPyro E[y/N]": b["p_marg"][0],
        "NumPyro pi0": b["pi0"][0],
        "NumPyro piN": b["piN"][0],
        "NumPyro p": b["p"][0],
    })
    ax = axes[i]
    ax.hist(yi, bins=bins, density=True, alpha=0.5, color="#3366cc",
            edgecolor="white", label="observed")
    ax.hist(b["ppc"].reshape(-1), bins=bins, density=True, histtype="step",
            color="#cc3333", lw=2, label="NumPyro PPC")
    ax.axvline(m["p_marg"] * N, color="#33aa33", ls="--", lw=2,
               label=f"mznib N·E[y/N]={m['p_marg']*N:.2f}")
    ax.axvline((yi / Nsi).mean() * N, color="black", ls=":", lw=1,
               label="empirical N·mean")
    ax.set_title(name); ax.legend(fontsize=8, loc="upper center")
plt.tight_layout(); plt.show()

pd.DataFrame(records).set_index("stage").round(3)
"""
    ),
    md(
        """
## 4. Reading the comparison

- **Stage 1.** Both estimates of `E[y/N]` match the empirical mean. NumPyro
  additionally recovers the three separate components (`π_0 ≈ 0.15`,
  `π_N ≈ 0.10`, `p ≈ 0.35`) — `mznib` does not, by design.
- **Stage 2 (overdispersion).** The marginal mean is still well-estimated by
  both: averaging is robust to overdispersion. The misfit is hidden in the
  shape, which only the NumPyro PPC reveals.
- **Stage 3 (bimodal p).** Same story: the marginal mean is right, the shape
  is very wrong. This is the case where reporting *only* a regression
  coefficient on `E[y/N]` would mislead a reader into thinking the model fits.
- **Stage 4 (variable N_i).** The data-generating max is < 20 for many obs,
  so `mznib`'s `N_i` argument actually receives the true per-row N — meaning
  this stage is *not* misspecified for `mznib`. It is misspecified for our
  NumPyro model, which assumes a single fixed `N`. So `mznib` should fit
  noticeably better here.

## 5. Takeaways

- For pure regression on a bounded count proportion, `mznib` is the right
  tool: simple formula interface, fast bootstrap inference, and direct
  interpretation on the proportion scale.
- For mechanism-level questions ("how big is the structural-zero rate?",
  "what is the underlying success probability after removing inflation?"),
  the NumPyro mixture parameterization is the better fit and gives full
  posterior intervals.
- Posterior predictive checks (NumPyro) and per-row N (`mznib`) are the
  diagnostics that catch misspecification each tool is silent about.
"""
    ),
]

nb2 = nbf.v4.new_notebook(cells=nb2_cells)
nb2["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
}


# ---------------------------------------------------------------------------
# Write + execute
# ---------------------------------------------------------------------------
def execute(nb, path):
    nbf.write(nb, path)
    nb = nbf.read(path, as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name="python3",
        resources={"metadata": {"path": "."}},
    ).execute()
    nbf.write(nb, path)
    print(f"executed and saved: {path}")


execute(nb1, "01_numpyro_walkthrough.ipynb")
execute(nb2, "02_compare_mcount.ipynb")
