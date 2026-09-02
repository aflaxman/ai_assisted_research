"""What effect could this design actually have detected?

A null is only as informative as the study's power. This simulates from the
*fitted* cocaine and comparator models -- real latent trajectory, real
estimated dispersion, real number of post-campaign windows -- injects a known
effect into the cocaine series only, and asks how often the controlled contrast
rejects the null.

It also reports how power grows as CDC releases more months, which answers the
practical question: when will this question be answerable?

Usage:
    python -P power_analysis.py [--reps 400]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import its_core as ic
from run_analysis import CAMPAIGN_START, load_outcomes

HERE = Path(__file__).parent
EFFECTS_PCT = [0, -5, -10, -15, -20, -25, -30, -40]


def _simulate_rolling(mu: np.ndarray, A: np.ndarray, phi: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Draw a rolling-sum series from the fitted window-level error model.

    Mean A mu, covariance phi * A diag(mu) A' -- the same covariance the
    estimator assumes, so this measures sampling power rather than
    misspecification.
    """
    mean = A @ mu
    cov = phi * ((A * mu) @ A.T)
    cov = cov + np.eye(len(mean)) * 1e-6 * np.trace(cov) / len(mean)
    draw = rng.multivariate_normal(mean, cov, method="cholesky")
    return np.clip(draw, 1.0, None)


def power_curve(wide: pd.DataFrame, comparator: str = "synthetic_opioids",
                reps: int = 400, horizon: pd.Timestamp | None = None,
                seed: int = 0) -> pd.DataFrame:
    """Rejection rate of the controlled contrast against injected effects."""
    if horizon is not None:
        wide = wide.loc[wide.index <= horizon]
    window_ends = pd.DatetimeIndex(wide.index)
    design = ic.build_design(window_ends, CAMPAIGN_START, effect="step")

    # Counterfactual latent trajectory and dispersion from the real data.
    truth = {}
    for drug in ("cocaine", comparator):
        fit = ic.fit_its(wide[drug].to_numpy(dtype=float), design)
        truth[drug] = (fit.counterfactual.copy(), fit.dispersion)

    post = (design.months >= CAMPAIGN_START).astype(float)
    rng = np.random.default_rng(seed)

    rows = []
    for pct in EFFECTS_PCT:
        log_rr = np.log1p(pct / 100.0)
        rejections, estimates = 0, []
        for _ in range(reps):
            mu_t = truth["cocaine"][0] * np.exp(log_rr * post)
            s_t = _simulate_rolling(mu_t, design.A, truth["cocaine"][1], rng)
            s_c = _simulate_rolling(truth[comparator][0], design.A,
                                    truth[comparator][1], rng)
            ft = ic.fit_its(s_t, design)
            fc = ic.fit_its(s_c, design)
            con = ic.contrast_effects(
                dict(zip(("log_rr", "se"), ft.coef("step"))),
                dict(zip(("log_rr", "se"), fc.coef("step"))),
            )
            estimates.append(con["log_rr"])
            rejections += abs(con["z"]) > 1.96
        rows.append({
            "true_effect_pct": pct,
            "power": rejections / reps,
            "mean_estimate_pct": 100.0 * (np.exp(np.mean(estimates)) - 1.0),
            "n_windows": len(window_ends),
            "post_windows": int((window_ends >= CAMPAIGN_START).sum()),
        })
    return pd.DataFrame(rows)


def minimum_detectable_effect(curve: pd.DataFrame, target: float = 0.8) -> float:
    """Interpolate the effect size reaching `target` power."""
    df = curve.sort_values("true_effect_pct", ascending=False)
    x, y = df["true_effect_pct"].to_numpy(), df["power"].to_numpy()
    if y.max() < target:
        return float("nan")
    for i in range(1, len(x)):
        if y[i] >= target:
            lo_x, lo_y, hi_x, hi_y = x[i - 1], y[i - 1], x[i], y[i]
            if hi_y == lo_y:
                return float(hi_x)
            return float(lo_x + (target - lo_y) * (hi_x - lo_x) / (hi_y - lo_y))
    return float(x[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()

    wide = load_outcomes("rolling12")
    print("=" * 78)
    print("POWER: controlled contrast, cocaine vs synthetic opioids")
    print("=" * 78)
    curve = power_curve(wide, reps=args.reps)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    mde = minimum_detectable_effect(curve)
    print(f"\nfalse positive rate at no effect: {curve.iloc[0]['power']:.3f}")
    print(f"minimum detectable effect at 80% power: "
          f"{mde:.1f}%" if np.isfinite(mde)
          else "\n80% power is not reached anywhere in the tested range")

    print("\n" + "=" * 78)
    print("POWER AS DATA ACCUMULATES")
    print("=" * 78)
    print("CDC publishes 12-month-ending totals about 4 months in arrears, so")
    print("each extra release adds one window. Extrapolating the current series:")
    rows = []
    for extra in (0, 6, 12, 18):
        extended = _extend(wide, extra)
        c = power_curve(extended, reps=max(120, args.reps // 3),
                        horizon=None, seed=1 + extra)
        m = minimum_detectable_effect(c)
        rows.append({
            "extra_months": extra,
            "last_window": extended.index.max().strftime("%Y-%m"),
            "post_windows": int((extended.index >= CAMPAIGN_START).sum()),
            "power_at_-15pct": float(c.loc[c["true_effect_pct"] == -15, "power"].iloc[0]),
            "power_at_-30pct": float(c.loc[c["true_effect_pct"] == -30, "power"].iloc[0]),
            "mde_pct": m,
        })
    horizon = pd.DataFrame(rows)
    print(horizon.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    curve.to_csv(HERE / "outputs" / "power_curve.csv", index=False)
    horizon.to_csv(HERE / "outputs" / "power_horizon.csv", index=False)
    print(f"\nwrote power tables to {HERE / 'outputs'}")


def _extend(wide: pd.DataFrame, extra_months: int) -> pd.DataFrame:
    """Append hypothetical future windows continuing each series' recent slope.

    Used only to project how power improves with more follow-up; the values are
    a placeholder trajectory, not a forecast.
    """
    if extra_months == 0:
        return wide
    recent = np.log(wide.tail(13))
    slope = (recent.iloc[-1] - recent.iloc[0]) / (len(recent) - 1)
    future_index = pd.date_range(wide.index.max() + pd.DateOffset(months=1),
                                 periods=extra_months, freq="MS")
    steps = np.arange(1, extra_months + 1)
    future = pd.DataFrame(
        {col: wide[col].iloc[-1] * np.exp(slope[col] * steps) for col in wide.columns},
        index=future_index,
    )
    out = pd.concat([wide, future])
    out.index.name = wide.index.name
    return out


if __name__ == "__main__":
    main()
