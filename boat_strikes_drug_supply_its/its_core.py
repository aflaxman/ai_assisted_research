"""Interrupted time series on a 12-month-ending rolling series.

The problem
-----------
CDC releases provisional overdose deaths only as 12-month-ending totals. Running
a segmented regression on that series directly is wrong twice over: consecutive
observations share 11 of 12 months, so they are massively autocorrelated by
construction, and the rolling window smears a step change at month T0 across the
following 12 observations, biasing the estimated step toward zero.

The fix
-------
Model the latent monthly count and let the rolling sum fall out of the model:

    log mu(m) = a + spline(m) + effect(m)          latent monthly mean
    S(t)      = sum_{k=0..11} mu(t - k)            what CDC publishes

Fit S by generalised least squares. The covariance of overlapping sums of
counts is available in closed form: two windows covary through the months they
share, so with dispersion phi,

    Cov(S(t), S(t')) = phi * sum_{m in overlap(t, t')} mu(m)

which is a banded matrix of bandwidth 11. That is the whole trick — no
deconvolution, no seed estimation, no invented monthly data.

Seasonality, and why it still bites
-----------------------------------
Seasonality that is *additive on the count scale* and strictly 12-month
periodic sums to the same constant over every window, so it is annihilated
exactly and can bias nothing. Real overdose seasonality is closer to
multiplicative on a trending baseline, which survives the window as a small
residual wobble — under 3% of the rolling total in simulation.

That wobble is not harmless. A window ending k months after the intervention
contains only k of 12 post-intervention months, so the step is identified with
leverage `window_leverage()` — averaging about 1/3 across the seven windows
available in 2026 — and model error is amplified by roughly its inverse. In
simulation a 2.6% wobble becomes a 30% apparent step, and 95% interval coverage
for a single series falls from nominal to zero.

The defence is the controlled contrast: comparator series that share the
seasonal shape cancel it, and `contrast_effects()` holds nominal coverage in
the same simulations. Treat single-series estimates from this data as
descriptive, and the contrast as the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = 12


# --------------------------------------------------------------------------
# natural cubic spline basis
# --------------------------------------------------------------------------
def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (Hastie/Tibshirani/Friedman parameterisation).

    Returns len(knots) - 1 columns, excluding the intercept. The basis is linear
    outside the boundary knots, so extrapolating past the last knot -- which is
    what an ITS counterfactual does -- is a straight line in log space rather
    than a runaway cubic.
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(sorted(knots), dtype=float)
    if len(knots) < 3:
        raise ValueError("need at least 3 knots")
    k_first, k_last = knots[0], knots[-1]

    def d(z: float) -> np.ndarray:
        pos_last = np.clip(x - k_last, 0, None) ** 3
        return (np.clip(x - z, 0, None) ** 3 - pos_last) / (k_last - z)

    cols = [x]
    for z in knots[1:-1]:
        cols.append(d(z) - d(k_first))
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------
@dataclass
class Design:
    """Latent-monthly design matrix plus the months it is defined on."""

    months: pd.DatetimeIndex   # latent monthly grid, length M
    X: np.ndarray              # M x p
    names: list[str]
    effect_cols: list[int]     # indices of the intervention terms
    window_ends: pd.DatetimeIndex   # observations, length n
    A: np.ndarray              # n x M moving-sum operator


def build_design(
    window_ends: pd.DatetimeIndex,
    intervention: pd.Timestamp,
    spline_knot_spacing: int = 18,
    effect: str = "step_ramp",
    dose: pd.Series | None = None,
    lag_months: int = 0,
) -> Design:
    """Assemble the latent monthly design and the moving-sum operator.

    Parameters
    ----------
    window_ends
        Months labelling each published 12-month-ending total.
    intervention
        First month of the campaign; effects are switched on at
        ``intervention + lag_months``.
    spline_knot_spacing
        Months between interior baseline knots. Knots are placed only in the
        pre-intervention period so the counterfactual is an extrapolation of
        pre-campaign behaviour, not a curve fitted through the campaign.
    effect
        ``"step"``, ``"step_ramp"``, or ``"dose"``.
    dose
        Monthly exposure (e.g. cumulative vessels struck), required for
        ``effect="dose"``.
    lag_months
        Delay between a supply shock and any change in deaths.
    """
    window_ends = pd.DatetimeIndex(window_ends)
    # Latent months run 11 before the first window end (so S is defined) to the last.
    months = pd.date_range(
        window_ends[0] - pd.DateOffset(months=WINDOW - 1), window_ends[-1], freq="MS"
    )
    t = np.arange(len(months), dtype=float)

    onset = intervention + pd.DateOffset(months=lag_months)
    post = (months >= onset).astype(float)
    since = np.where(post > 0, (months.year - onset.year) * 12
                     + (months.month - onset.month) + 1.0, 0.0)

    pre_mask = months < intervention
    pre_t = t[pre_mask]
    n_interior = max(1, int(len(pre_t) // spline_knot_spacing) - 1)
    quantiles = np.linspace(0, 1, n_interior + 2)
    knots = np.quantile(pre_t, quantiles)
    knots = np.unique(knots)

    baseline = natural_spline_basis(t, knots)
    # Centre and scale for conditioning.
    baseline = (baseline - baseline.mean(0)) / (baseline.std(0) + 1e-12)

    cols = [np.ones(len(months))] + [baseline[:, j] for j in range(baseline.shape[1])]
    names = ["intercept"] + [f"spline{j}" for j in range(baseline.shape[1])]

    first_effect_col = len(cols)
    if effect == "step":
        cols.append(post)
        names.append("step")
    elif effect == "step_ramp":
        cols += [post, since / 12.0]
        names += ["step", "ramp_per_year"]
    elif effect == "dose":
        if dose is None:
            raise ValueError("effect='dose' requires a dose series")
        aligned = dose.reindex(months).fillna(0.0).to_numpy(dtype=float)
        if lag_months:
            aligned = np.concatenate([np.zeros(lag_months), aligned])[: len(months)]
        scale = aligned.max() if aligned.max() > 0 else 1.0
        cols.append(aligned / scale)
        names.append("dose_full_campaign")
    else:
        raise ValueError(f"unknown effect {effect!r}")
    effect_cols = list(range(first_effect_col, len(cols)))

    X = np.column_stack(cols)

    # Moving-sum operator: row t picks out the 12 latent months ending at t.
    month_pos = {m: i for i, m in enumerate(months)}
    A = np.zeros((len(window_ends), len(months)))
    for r, end in enumerate(window_ends):
        for k in range(WINDOW):
            A[r, month_pos[end - pd.DateOffset(months=k)]] = 1.0

    return Design(months, X, names, effect_cols, window_ends, A)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
@dataclass
class Fit:
    design: Design
    theta: np.ndarray
    vcov: np.ndarray
    dispersion: float
    fitted_rolling: np.ndarray
    latent: np.ndarray            # fitted monthly mu
    counterfactual: np.ndarray    # monthly mu with effect terms zeroed
    n_obs: int

    def coef(self, name: str) -> tuple[float, float]:
        i = self.design.names.index(name)
        return self.theta[i], float(np.sqrt(self.vcov[i, i]))

    def summary(self) -> pd.DataFrame:
        se = np.sqrt(np.diag(self.vcov))
        return pd.DataFrame(
            {
                "term": self.design.names,
                "estimate": self.theta,
                "se": se,
                "z": self.theta / se,
            }
        )


def _covariance(mu: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Cov of overlapping moving sums of counts with mean mu, unit dispersion.

    Cov(S(t), S(t')) = sum over months shared by the two windows of mu(m),
    which is exactly A diag(mu) A'.
    """
    return (A * mu) @ A.T


def fit_its(
    rolling: np.ndarray,
    design: Design,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> Fit:
    """Iterated generalised least squares for the latent monthly model."""
    y = np.asarray(rolling, dtype=float)
    X, A = design.X, design.A
    n, p = len(y), X.shape[1]

    # Start from a log-linear least squares fit to the rolling mean.
    theta = np.zeros(p)
    theta[0] = np.log(max(y.mean() / WINDOW, 1e-6))

    for _ in range(max_iter):
        eta = X @ theta
        mu = np.exp(np.clip(eta, -50, 50))
        pred = A @ mu
        sigma = _covariance(mu, A)
        # Ridge the covariance a hair for numerical safety.
        sigma = sigma + np.eye(n) * 1e-8 * np.trace(sigma) / n
        L = np.linalg.cholesky(sigma)

        J = A @ (mu[:, None] * X)                 # n x p Jacobian of pred wrt theta
        rw = np.linalg.solve(L, y - pred)
        Jw = np.linalg.solve(L, J)
        step, *_ = np.linalg.lstsq(Jw, rw, rcond=None)

        # Damped Gauss-Newton: shrink the step until the objective improves.
        obj0 = float(rw @ rw)
        alpha = 1.0
        for _ in range(40):
            cand = theta + alpha * step
            mu_c = np.exp(np.clip(X @ cand, -50, 50))
            r_c = np.linalg.solve(L, y - A @ mu_c)
            if float(r_c @ r_c) <= obj0:
                break
            alpha *= 0.5
        theta = theta + alpha * step
        if np.max(np.abs(alpha * step)) < tol:
            break

    eta = X @ theta
    mu = np.exp(eta)
    pred = A @ mu
    sigma = _covariance(mu, A)
    sigma = sigma + np.eye(n) * 1e-8 * np.trace(sigma) / n
    L = np.linalg.cholesky(sigma)
    J = A @ (mu[:, None] * X)
    Jw = np.linalg.solve(L, J)
    rw = np.linalg.solve(L, y - pred)
    dispersion = float(rw @ rw) / max(n - p, 1)
    vcov = np.linalg.pinv(Jw.T @ Jw) * dispersion

    theta_cf = theta.copy()
    theta_cf[design.effect_cols] = 0.0
    counterfactual = np.exp(X @ theta_cf)

    return Fit(design, theta, vcov, dispersion, pred, mu, counterfactual, n)


# --------------------------------------------------------------------------
# effect summaries
# --------------------------------------------------------------------------
def relative_effect(fit: Fit, at: pd.Timestamp | None = None) -> dict:
    """Proportional change in the monthly rate attributable to the campaign.

    Evaluated in the final observed month unless ``at`` is given, so a
    step+ramp model is summarised where the data actually end rather than
    extrapolated.
    """
    months = fit.design.months
    at = months[-1] if at is None else at
    i = int(np.where(months == at)[0][0])

    contrast = np.zeros(len(fit.theta))
    contrast[fit.design.effect_cols] = fit.design.X[i, fit.design.effect_cols]
    log_rr = float(contrast @ fit.theta)
    se = float(np.sqrt(contrast @ fit.vcov @ contrast))
    return {
        "month": at,
        "log_rr": log_rr,
        "se": se,
        "rr": float(np.exp(log_rr)),
        "lo": float(np.exp(log_rr - 1.96 * se)),
        "hi": float(np.exp(log_rr + 1.96 * se)),
        "pct_change": 100.0 * (np.exp(log_rr) - 1.0),
        "pct_lo": 100.0 * (np.exp(log_rr - 1.96 * se) - 1.0),
        "pct_hi": 100.0 * (np.exp(log_rr + 1.96 * se) - 1.0),
    }


def cumulative_excess(fit: Fit, intervention: pd.Timestamp) -> dict:
    """Fitted minus counterfactual deaths summed over the campaign months."""
    mask = fit.design.months >= intervention
    excess = float((fit.latent[mask] - fit.counterfactual[mask]).sum())
    baseline = float(fit.counterfactual[mask].sum())

    # Delta method on g(theta) = sum_post [exp(X theta) - exp(X theta_cf)].
    # d/dtheta exp(X_i theta)     = mu_i * X_i
    # d/dtheta exp(X_i theta_cf)  = cf_i * X_i with the effect columns zeroed,
    # since theta_cf holds those columns at 0 regardless of theta.
    X = fit.design.X
    X_cf = X.copy()
    X_cf[:, fit.design.effect_cols] = 0.0
    idx = np.where(mask)[0]
    grad = (fit.latent[idx, None] * X[idx]).sum(0) - \
           (fit.counterfactual[idx, None] * X_cf[idx]).sum(0)
    se = float(np.sqrt(grad @ fit.vcov @ grad))
    return {
        "post_months": int(mask.sum()),
        "counterfactual_deaths": baseline,
        "excess_deaths": excess,
        "excess_lo": excess - 1.96 * se,
        "excess_hi": excess + 1.96 * se,
    }


def counterfactual_rolling(fit: Fit) -> pd.DataFrame:
    """The counterfactual on the published rolling scale, with a 95% band.

    Delta method: d/dtheta [A mu_cf] = A diag(mu_cf) X_cf, where X_cf holds the
    effect columns at zero.
    """
    X_cf = fit.design.X.copy()
    X_cf[:, fit.design.effect_cols] = 0.0
    grad = fit.design.A @ (fit.counterfactual[:, None] * X_cf)
    var = np.einsum("ij,jk,ik->i", grad, fit.vcov, grad)
    centre = fit.design.A @ fit.counterfactual
    se = np.sqrt(np.clip(var, 0, None))
    return pd.DataFrame({
        "window_end": fit.design.window_ends,
        "counterfactual": centre,
        "lo": centre - 1.96 * se,
        "hi": centre + 1.96 * se,
        "fitted": fit.fitted_rolling,
    })


def window_leverage(design: Design) -> pd.DataFrame:
    """How much of each published window is actually post-intervention.

    This is the quantity that governs how badly the rolling window amplifies
    model error into the step estimate: a window containing k post-intervention
    months out of 12 responds to a step with weight k/12, so a systematic
    distortion of size d in the rolling series can masquerade as a step of
    roughly d * 12/k.
    """
    step_like = np.zeros(design.X.shape[1])
    idx = [i for i, n in enumerate(design.names) if n in ("step", "dose_full_campaign")]
    if not idx:
        raise ValueError("design has no step term to measure leverage against")
    step_like[idx[0]] = 1.0
    indicator = design.X @ step_like
    frac = (design.A @ indicator) / WINDOW
    return pd.DataFrame({"window_end": design.window_ends, "post_fraction": frac})


def contrast_effects(treated: dict, control: dict) -> dict:
    """Controlled ITS contrast: treated effect minus comparator effect.

    Cocaine-involved and opioid-involved deaths overlap heavily (most
    cocaine-involved deaths also involve fentanyl), so the two series are
    positively correlated and adding their variances is conservative.
    """
    log_rr = treated["log_rr"] - control["log_rr"]
    se = float(np.hypot(treated["se"], control["se"]))
    return {
        "log_rr": log_rr,
        "se": se,
        "ratio_of_ratios": float(np.exp(log_rr)),
        "pct_change": 100.0 * (np.exp(log_rr) - 1.0),
        "pct_lo": 100.0 * (np.exp(log_rr - 1.96 * se) - 1.0),
        "pct_hi": 100.0 * (np.exp(log_rr + 1.96 * se) - 1.0),
        "z": log_rr / se,
    }
