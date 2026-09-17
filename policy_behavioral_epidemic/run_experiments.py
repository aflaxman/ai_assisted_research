"""Reproduce the framework's claimed benefits and generate all figures.

Run:  uv run python run_experiments.py

Produces (in ./figures):
  1. alarm_functions.png     -- the three alarm forms from Ward et al. (2023)
  2. policy_behavior.png      -- how a policy reshapes the epidemic vs no response
  3. estimation.png           -- MLE / posterior recovery of true parameters
  4. forecast.png             -- short-term posterior-predictive forecast + coverage
and prints a numeric summary table to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference import fit_mcmc, fit_mle, forecast
from model import (
    ModelParams,
    Policy,
    hill_alarm,
    power_alarm,
    simulate,
    threshold_alarm,
)

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

# A single ground-truth scenario used throughout.
TRUTH = ModelParams(
    beta=0.55, gamma=0.2,
    endog_delta=0.5, endog_x0=250.0, endog_nu=3.0, m=7,
    policy_weight=0.7, adapt_rate=0.3, compliance_sd=0.05,
)
POLICIES = [Policy(start=25, end=200, intensity=0.85)]
N, I0, N_DAYS = 200_000, 10, 200


# ---------------------------------------------------------------------------

def fig_alarm_functions():
    x = np.linspace(0, 2000, 500)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, power_alarm(x, 0.0008, N), label="Power (k=0.0008)")
    ax.plot(x, threshold_alarm(x, 500, 0.7), label="Threshold (H=500, δ=0.7)")
    ax.plot(x, hill_alarm(x, 0.7, 400, 4), label="Hill (δ=0.7, x₀=400, ν=4)")
    ax.set_xlabel("m-day average incidence  x")
    ax.set_ylabel("alarm  f(x)")
    ax.set_title("Alarm functions (Ward, Deardon & Schmidt 2023)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "alarm_functions.png", dpi=130)
    plt.close(fig)


def fig_policy_behavior():
    """Same policy, three behavioral regimes -> flattening the curve."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    regimes = {
        "No behavioral response": ModelParams(
            beta=TRUTH.beta, gamma=TRUTH.gamma,
            endog_delta=0.0, policy_weight=0.0, compliance_sd=0.0),
        "Endogenous fear only": ModelParams(
            beta=TRUTH.beta, gamma=TRUTH.gamma,
            endog_delta=TRUTH.endog_delta, endog_x0=TRUTH.endog_x0,
            endog_nu=TRUTH.endog_nu, policy_weight=0.0, compliance_sd=0.0),
        "Policy-informed": ModelParams(**{**TRUTH.__dict__, "compliance_sd": 0.0}),
    }
    for label, params in regimes.items():
        # Average several stochastic runs for a smooth mean curve.
        curves = np.array([
            simulate(params, POLICIES, N_DAYS, N, I0,
                     np.random.default_rng(s)).new_infections
            for s in range(30)])
        ax[0].plot(curves.mean(0), label=label)
    ax[0].axvspan(POLICIES[0].start, N_DAYS, color="grey", alpha=0.12,
                  label="policy active")
    ax[0].set_xlabel("day"); ax[0].set_ylabel("new infections / day")
    ax[0].set_title("Policy-informed behavior flattens the curve")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    epi = simulate(regimes["Policy-informed"], POLICIES, N_DAYS, N, I0,
                   np.random.default_rng(0))
    ax[1].plot(epi.policy, "--", color="firebrick", label="policy p(t)")
    ax[1].plot(epi.alarm, color="navy", label="realized alarm a(t)")
    ax[1].set_xlabel("day"); ax[1].set_ylabel("intensity / alarm")
    ax[1].set_title("Delayed adaptation & heterogeneous compliance")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "policy_behavior.png", dpi=130)
    plt.close(fig)


def run_inference():
    """Fit an epidemic simulated from TRUTH; recover parameters."""
    epi = simulate(TRUTH, POLICIES, N_DAYS, N, I0, np.random.default_rng(42))
    fit_names = ["beta", "gamma", "endog_delta", "policy_weight", "adapt_rate"]

    mle = fit_mle(epi, fit_names, TRUTH, n_restarts=10, seed=0)
    prop = _proposal_from_se(mle, fit_names)
    mcmc = fit_mcmc(epi, fit_names, TRUTH, n_iter=25_000, burn=6_000, thin=5,
                    init=mle["estimate"], proposal_sd=prop, seed=1)
    return epi, fit_names, mle, mcmc


def _proposal_from_se(mle, fit_names, scale=0.7):
    """Random-walk proposal SDs scaled to the MLE standard errors."""
    prop = {}
    for n in fit_names:
        se = mle["se"].get(n, np.nan)
        lo, hi = 0.0, 1.0
        fallback = 0.01
        prop[n] = scale * se if np.isfinite(se) and se > 0 else fallback
    return prop


def fig_estimation(fit_names, mle, mcmc):
    truth_vals = {n: getattr(TRUTH, n) for n in fit_names}
    fig, axes = plt.subplots(1, len(fit_names), figsize=(3.0 * len(fit_names), 3.6))
    for ax, i in zip(axes, range(len(fit_names))):
        n = fit_names[i]
        col = mcmc["samples"][:, i]
        ax.hist(col, bins=40, color="steelblue", alpha=0.7, density=True)
        ax.axvline(truth_vals[n], color="black", lw=2, label="truth")
        ax.axvline(mle["estimate"][n], color="firebrick", ls="--", lw=2, label="MLE")
        ax.set_title(n, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Parameter recovery: posterior (blue), MLE (red), truth (black)")
    fig.tight_layout()
    fig.savefig(FIG / "estimation.png", dpi=130)
    plt.close(fig)


def fig_forecast(fit_names):
    """Fit on the first 60 days, forecast the next 40, check coverage."""
    full = simulate(TRUTH, POLICIES, N_DAYS, N, I0, np.random.default_rng(99))
    cut = 60
    from model import Epidemic
    obs = Epidemic(full.S[:cut], full.I[:cut], full.R[:cut],
                   full.new_infections[:cut], full.new_removals[:cut],
                   full.alarm[:cut], full.policy[:cut], N)
    mle = fit_mle(obs, fit_names, TRUTH, n_restarts=8, seed=4)
    prop = _proposal_from_se(mle, fit_names)
    mcmc = fit_mcmc(obs, fit_names, TRUTH, n_iter=20_000, burn=5_000, thin=5,
                    init=mle["estimate"], proposal_sd=prop, seed=5)
    horizon = 40
    fc = forecast(obs, mcmc, TRUTH, POLICIES, horizon, n_paths=500, seed=7)

    actual = full.new_infections[cut:cut + horizon]
    days = np.arange(cut, cut + horizon)
    covered = np.mean((actual >= fc["lo"]) & (actual <= fc["hi"]))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(np.arange(cut), obs.new_infections, color="black", label="observed (fit)")
    ax.plot(days, actual, color="green", lw=2, label="actual (held out)")
    ax.plot(days, fc["median"], color="firebrick", ls="--", label="forecast median")
    ax.fill_between(days, fc["lo"], fc["hi"], color="firebrick", alpha=0.2,
                    label="95% predictive interval")
    ax.axvline(cut, color="grey", ls=":", label="forecast origin")
    ax.set_xlabel("day"); ax.set_ylabel("new infections / day")
    ax.set_title(f"Short-term forecast — 95% PI covers {covered:.0%} of held-out days")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "forecast.png", dpi=130)
    plt.close(fig)
    return covered


def _coverage_run(compliance_sd, n_sims, seed0):
    """Fit beta & gamma by MLE on `n_sims` epidemics; return estimates + coverage."""
    names = ["beta", "gamma"]
    truth = {"beta": TRUTH.beta, "gamma": TRUTH.gamma}
    tmpl = ModelParams(**{**TRUTH.__dict__, "compliance_sd": compliance_sd})
    ests = {n: [] for n in names}
    cover = {n: 0 for n in names}
    z = 1.959963985
    for s in range(n_sims):
        epi = simulate(tmpl, POLICIES, N_DAYS, N, I0, np.random.default_rng(seed0 + s))
        mle = fit_mle(epi, names, tmpl, n_restarts=2, seed=s)
        for n in names:
            e, se = mle["estimate"][n], mle["se"][n]
            ests[n].append(e)
            if np.isfinite(se) and (e - z * se) <= truth[n] <= (e + z * se):
                cover[n] += 1
    return ests, {n: cover[n] / n_sims for n in names}


def fig_calibration(n_sims=50):
    """Coverage study: are Wald 95% CIs for beta and gamma calibrated?

    Two regimes make the point.  When compliance is homogeneous (sigma_c = 0)
    the chain-binomial likelihood is correctly specified and coverage is ~95%.
    When compliance is heterogeneous (sigma_c > 0) the extra process noise is
    unmodeled, the likelihood is overconfident, and coverage drops -- a concrete
    illustration of why the framework foregrounds uncertainty quantification.
    """
    names = ["beta", "gamma"]
    truth = {"beta": TRUTH.beta, "gamma": TRUTH.gamma}
    regimes = [("Homogeneous compliance (σ_c=0, correctly specified)", 0.0, 1000),
               ("Heterogeneous compliance (σ_c=0.05, unmodeled)", 0.05, 3000)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    out = {}
    for row, (label, sd, seed0) in enumerate(regimes):
        ests, cov = _coverage_run(sd, n_sims, seed0)
        out[f"sd_{sd}"] = cov
        for col, n in enumerate(names):
            ax = axes[row, col]
            ax.hist(ests[n], bins=18, color="mediumseagreen", alpha=0.75)
            ax.axvline(truth[n], color="black", lw=2, label="truth")
            ax.axvline(np.mean(ests[n]), color="firebrick", ls="--", lw=2,
                       label=f"mean={np.mean(ests[n]):.3f}")
            ax.set_title(f"{n}: 95% CI coverage = {cov[n]:.0%}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Estimator calibration over {n_sims} epidemics per regime")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=130)
    plt.close(fig)
    return out


def main():
    print("Generating alarm-function figure ...")
    fig_alarm_functions()
    print("Generating policy/behavior figure ...")
    fig_policy_behavior()

    print("Running estimation (MLE + MCMC) ...")
    epi, fit_names, mle, mcmc = run_inference()
    fig_estimation(fit_names, mle, mcmc)

    print("Running forecast experiment ...")
    coverage = fig_forecast(fit_names)

    print("Running calibration / coverage study ...")
    calib = fig_calibration(n_sims=50)

    # ---- numeric summary table -------------------------------------------
    truth_vals = {n: getattr(TRUTH, n) for n in fit_names}
    rows = []
    print("\n" + "=" * 78)
    print(f"{'param':<14}{'truth':>10}{'MLE':>10}{'MLE SE':>10}"
          f"{'post.mean':>11}{'95% CI':>22}")
    print("-" * 78)
    for n in fit_names:
        s = mcmc["summary"][n]
        ci = f"[{s['q2.5']:.3f}, {s['q97.5']:.3f}]"
        print(f"{n:<14}{truth_vals[n]:>10.3f}{mle['estimate'][n]:>10.3f}"
              f"{mle['se'][n]:>10.3f}{s['mean']:>11.3f}{ci:>22}")
        rows.append({"param": n, "truth": truth_vals[n],
                     "mle": mle["estimate"][n], "mle_se": mle["se"][n],
                     "post_mean": s["mean"], "ci_lo": s["q2.5"], "ci_hi": s["q97.5"]})
    print("=" * 78)
    print(f"MCMC acceptance rate: {mcmc['accept_rate']:.2f}")
    print(f"Forecast 95% PI empirical coverage: {coverage:.0%}")
    print("Wald 95% CI coverage (50 sims/regime):")
    for reg, cov in calib.items():
        print(f"  {reg}: beta={cov['beta']:.0%}, gamma={cov['gamma']:.0%}")

    summary = {"truth": truth_vals, "mle": mle["estimate"], "mle_se": mle["se"],
               "posterior": mcmc["summary"], "accept_rate": mcmc["accept_rate"],
               "forecast_coverage": float(coverage), "calibration": calib,
               "rows": rows}
    (Path(__file__).parent / "results_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nWrote results_summary.json and 4 figures to {FIG}/")


if __name__ == "__main__":
    main()
