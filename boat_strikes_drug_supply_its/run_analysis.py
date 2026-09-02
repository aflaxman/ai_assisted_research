"""Fit the controlled ITS for the maritime strike campaign.

Primary estimator: the difference between the cocaine step and a comparator
step, where the comparator is a drug whose supply chain the strikes do not
touch. Cocaine reaching the US crosses the Caribbean and Eastern Pacific by
boat; fentanyl and methamphetamine are made in and moved overland from Mexico.
If the campaign cut cocaine supply, cocaine-involved deaths should move
relative to those comparators.

Single-series estimates are printed too, but they are descriptive: see the
`its_core` docstring and `test_multiplicative_seasonality_is_amplified_into_the_step`
for why the rolling-window data cannot support them on their own.

Usage:
    python -P run_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import its_core as ic

HERE = Path(__file__).parent
CAMPAIGN_START = pd.Timestamp("2025-09-01")

TREATED = "cocaine"
COMPARATORS = ["synthetic_opioids", "psychostimulants"]
ALL_SERIES = [TREATED] + COMPARATORS + ["heroin", "all_drug"]

# A supply shock cannot reach a death certificate instantly: cargo already
# ashore has to clear, prices and purity have to move, and deaths follow.
LAGS = [0, 2, 4]


def load_outcomes(value_col: str = "rolling12") -> pd.DataFrame:
    df = pd.read_csv(HERE / "data" / "overdose_12mo.csv", parse_dates=["window_end"])
    return df.pivot(index="window_end", columns="drug", values=value_col)


def load_dose() -> pd.Series:
    df = pd.read_csv(HERE / "data" / "strikes_monthly.csv", parse_dates=["month"])
    return df.set_index("month")["cum_vessels"].astype(float)


def fit_one(series: pd.Series, intervention: pd.Timestamp, effect: str,
            lag: int = 0, dose: pd.Series | None = None,
            knot_spacing: int = 18) -> ic.Fit:
    design = ic.build_design(
        pd.DatetimeIndex(series.index),
        intervention,
        spline_knot_spacing=knot_spacing,
        effect=effect,
        dose=dose,
        lag_months=lag,
    )
    return ic.fit_its(series.to_numpy(dtype=float), design)


def controlled_table(wide: pd.DataFrame, intervention: pd.Timestamp,
                     effect: str = "step", knot_spacing: int = 18,
                     dose: pd.Series | None = None) -> pd.DataFrame:
    """Single-series effects plus every cocaine-vs-comparator contrast."""
    fits, effects = {}, {}
    for lag in LAGS:
        for drug in ALL_SERIES:
            fit = fit_one(wide[drug], intervention, effect, lag, dose, knot_spacing)
            fits[(drug, lag)] = fit
            effects[(drug, lag)] = ic.relative_effect(fit)

    rows = []
    for lag in LAGS:
        for drug in ALL_SERIES:
            eff = effects[(drug, lag)]
            rows.append({
                "kind": "single series", "series": drug, "lag_months": lag,
                "pct_change": eff["pct_change"], "lo": eff["pct_lo"],
                "hi": eff["pct_hi"], "z": eff["log_rr"] / eff["se"],
                "dispersion": fits[(drug, lag)].dispersion,
            })
        for comp in COMPARATORS:
            con = ic.contrast_effects(effects[(TREATED, lag)], effects[(comp, lag)])
            rows.append({
                "kind": "contrast", "series": f"{TREATED} vs {comp}",
                "lag_months": lag, "pct_change": con["pct_change"],
                "lo": con["pct_lo"], "hi": con["pct_hi"], "z": con["z"],
                "dispersion": np.nan,
            })
    return pd.DataFrame(rows)


def placebo_test(wide: pd.DataFrame) -> pd.DataFrame:
    """Run the estimator against intervention dates when nothing happened.

    Any 'effect' found at a placebo date is the design misreading ordinary
    curvature in the series, and bounds how much of the real estimate to
    believe. Placebo dates are spaced a year apart in the pre-campaign period
    and each fit uses only data available up to the same horizon (7 windows of
    follow-up) so the comparison is like for like.
    """
    rows = []
    for placebo in [pd.Timestamp(f"{y}-09-01") for y in (2022, 2023, 2024)]:
        horizon = placebo + pd.DateOffset(months=6)
        truncated = wide.loc[wide.index <= horizon]
        effects = {}
        for drug in [TREATED] + COMPARATORS:
            fit = fit_one(truncated[drug], placebo, "step")
            effects[drug] = ic.relative_effect(fit)
            rows.append({"placebo_date": placebo, "kind": "single series",
                         "series": drug, "pct_change": effects[drug]["pct_change"],
                         "lo": effects[drug]["pct_lo"], "hi": effects[drug]["pct_hi"]})
        for comp in COMPARATORS:
            con = ic.contrast_effects(effects[TREATED], effects[comp])
            rows.append({"placebo_date": placebo, "kind": "contrast",
                         "series": f"{TREATED} vs {comp}",
                         "pct_change": con["pct_change"],
                         "lo": con["pct_lo"], "hi": con["pct_hi"]})
    return pd.DataFrame(rows)


def sensitivity(wide_pred: pd.DataFrame, wide_rep: pd.DataFrame,
                dose: pd.Series) -> pd.DataFrame:
    """Vary the choices an analyst could reasonably make differently."""
    specs = [
        ("primary: step, predicted counts, 18-month knots",
         wide_pred, "step", 18, None),
        ("reported counts (no pending-investigation adjustment)",
         wide_rep, "step", 18, None),
        ("step + ramp", wide_pred, "step_ramp", 18, None),
        ("dose-response in cumulative vessels struck",
         wide_pred, "dose", 18, dose),
        ("stiffer baseline (30-month knots)", wide_pred, "step", 30, None),
        ("wigglier baseline (12-month knots)", wide_pred, "step", 12, None),
    ]
    rows = []
    for label, wide, effect, spacing, dose_arg in specs:
        effects = {}
        for drug in [TREATED] + COMPARATORS:
            fit = fit_one(wide[drug], CAMPAIGN_START, effect, 0, dose_arg, spacing)
            effects[drug] = ic.relative_effect(fit)
        for comp in COMPARATORS:
            con = ic.contrast_effects(effects[TREATED], effects[comp])
            rows.append({"specification": label, "contrast": f"{TREATED} vs {comp}",
                         "pct_change": con["pct_change"], "lo": con["pct_lo"],
                         "hi": con["pct_hi"]})
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ("pct_change", "lo", "hi", "z", "dispersion"):
        if col in out:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}")
    return out.to_string(index=False)


def main() -> None:
    wide = load_outcomes("rolling12")
    wide_rep = load_outcomes("rolling12_reported")
    dose = load_dose()

    print("=" * 78)
    print("DATA")
    print("=" * 78)
    print(f"12-month-ending windows: {wide.index.min():%Y-%m} .. "
          f"{wide.index.max():%Y-%m}  (n={len(wide)})")
    print(f"campaign begins {CAMPAIGN_START:%Y-%m}; "
          f"{(wide.index >= CAMPAIGN_START).sum()} windows overlap it")

    design = ic.build_design(pd.DatetimeIndex(wide.index), CAMPAIGN_START,
                             effect="step")
    lev = ic.window_leverage(design)
    post = lev[lev["post_fraction"] > 0]
    print("\nhow much of each post-campaign window is actually post-campaign:")
    print("  " + "  ".join(f"{r.window_end:%Y-%m}={r.post_fraction:.2f}"
                           for r in post.itertuples()))
    print(f"  mean leverage {post['post_fraction'].mean():.2f} -> model error is "
          f"amplified about {1 / post['post_fraction'].mean():.1f}x")

    print("\n" + "=" * 78)
    print("PRIMARY: step change at campaign start, by lag")
    print("=" * 78)
    main_tbl = controlled_table(wide, CAMPAIGN_START, "step")
    print(_fmt(main_tbl))

    print("\n" + "=" * 78)
    print("PLACEBO: same estimator, intervention dates when nothing happened")
    print("=" * 78)
    plac = placebo_test(wide)
    print(_fmt(plac))

    print("\n" + "=" * 78)
    print("SENSITIVITY: contrasts under alternative specifications")
    print("=" * 78)
    sens = sensitivity(wide, wide_rep, dose)
    print(_fmt(sens))

    outdir = HERE / "outputs"
    main_tbl.to_csv(outdir / "its_main.csv", index=False)
    plac.to_csv(outdir / "its_placebo.csv", index=False)
    sens.to_csv(outdir / "its_sensitivity.csv", index=False)
    lev.to_csv(outdir / "window_leverage.csv", index=False)
    print(f"\nwrote 4 tables to {outdir}")


if __name__ == "__main__":
    main()
