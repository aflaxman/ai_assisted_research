"""King County regression of Goyal et al. (2021): Figure 6.

The paper regresses the log of the effective reproductive number on mobility
and mask wearing (Supplementary methods, eq. 4),

    ln Re ~ ln(1 - f_h) + ln(1 - f_M) + A ln S + zeta,

and reads the effect of masks off the fit by setting f_M = 0.  Inputs here:

* Re: the paper's own estimate (Thakkar & Famulare, from cases, admissions
  and deaths held under a data use agreement), read off Figure 6C by
  ``digitize/digitize_fig6.py``.  This is a curve traced from a blurred image,
  good to about 0.03.
* f_h: Google's residential "percent change from baseline" for King County,
  the public data.  The paper's smoothed line follows the weekday level (the
  weekend dips sit below it), so weekends are interpolated over before a
  centred 7-day mean.
* f_M: the paper's smoothed "always masking" line, read off Figure 6B.  The
  SCAN survey begins on 7 May; before that the line is the authors'
  interpolation, linear in the logit, from an assumed 1% on 1 March.
* S: more than 90% throughout and reported as negligible; left out.

Serial correlation is extreme (Re is a smooth curve sampled daily), so
standard errors are Newey-West with a 21-day lag.

    uv run python run_king_county.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
SURVEY_START = "2020-05-07"
HAC_LAGS = 21


def load() -> pd.DataFrame:
    google = pd.read_csv(DATA / "google_mobility_king_county_2020.tsv", sep="\t",
                         parse_dates=["date"]).set_index("date")
    paper = pd.read_csv(DATA / "paper_fig6_digitized.tsv", sep="\t", parse_dates=["date"]).set_index("date")
    home = google.residential / 100.0
    weekday = home.where(home.index.dayofweek < 5).interpolate()
    df = paper.join(weekday.rolling(7, center=True, min_periods=1).mean().clip(lower=0.0).rename("f_h"))
    df["f_M"] = df.mask_always.rolling(7, center=True, min_periods=1).mean()
    df["ln_Re"] = np.log(df.Re)
    df["ln_mobility"] = np.log(1.0 - df.f_h)
    df["ln_masks"] = np.log(1.0 - df.f_M)
    df["day"] = np.arange(len(df), dtype=float)
    return df


def alternative_masks(df: pd.DataFrame, start_level: float) -> pd.Series:
    """ln(1 - f_M) had mask wearing on 1 March been ``start_level`` instead of
    1%, with the same logit-linear rise to the first survey value."""
    f = df.f_M.copy()
    pre = df.index < SURVEY_START
    logit = lambda p: np.log(p / (1.0 - p))
    end = logit(f[SURVEY_START])
    frac = np.arange(pre.sum()) / pre.sum()
    f[pre] = 1.0 / (1.0 + np.exp(-(logit(start_level) + frac * (end - logit(start_level)))))
    return np.log(1.0 - f)


def fit(df: pd.DataFrame, columns: list[str]):
    X = sm.add_constant(df[columns])
    return sm.OLS(df.ln_Re, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def describe(label, model, df, mask_col="ln_masks") -> dict:
    last = df.iloc[-1]
    row = {"model": label, "days": int(model.nobs), "R2": model.rsquared, "AIC": model.aic,
           "durbin_watson": sm.stats.durbin_watson(model.resid)}
    for name in ("ln_mobility", mask_col, "day"):
        if name in model.params:
            key = "ln_masks" if name == mask_col else name
            row[f"coef_{key}"] = model.params[name]
            row[f"se_{key}"] = model.bse[name]
    if mask_col in model.params:
        # Removing masks multiplies Re by (1 - f_M)^(-coef).
        b, se = model.params[mask_col], model.bse[mask_col]
        x = last[mask_col]
        row["Re_fitted_Aug30"] = float(np.exp(model.predict(sm.add_constant(df[list(model.params.index[1:])]))).iloc[-1])
        row["Re_no_masks_Aug30"] = row["Re_fitted_Aug30"] * np.exp(-b * x)
        row["mask_reduction_pct"] = 100.0 * (1.0 - np.exp(b * x))
        row["mask_reduction_pct_low"] = 100.0 * (1.0 - np.exp((b - 1.96 * se) * x))
        row["mask_reduction_pct_high"] = 100.0 * (1.0 - np.exp((b + 1.96 * se) * x))
    return row


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load()
    survey = df[SURVEY_START:]
    rows = []

    main_fit = fit(df, ["ln_mobility", "ln_masks"])
    rows.append(describe("mobility + masks (the paper's regression)", main_fit, df))
    rows.append(describe("mobility only", fit(df, ["ln_mobility"]), df))
    rows.append(describe("mobility + linear time trend (placebo)", fit(df, ["ln_mobility", "day"]), df))
    rows.append(describe("mobility + masks + time trend", fit(df, ["ln_mobility", "ln_masks", "day"]), df))
    rows.append(describe("mobility + masks, survey period only (7 May on)", fit(survey, ["ln_mobility", "ln_masks"]), survey))
    rows.append(describe("mobility + time trend, survey period only", fit(survey, ["ln_mobility", "day"]), survey))
    for level in (0.10, 0.30):
        alt = df.assign(ln_masks_alt=alternative_masks(df, level))
        rows.append(describe(f"mobility + masks, {level:.0%} masked on 1 March instead of 1%",
                             fit(alt, ["ln_mobility", "ln_masks_alt"]), alt, mask_col="ln_masks_alt"))

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "fig6_king_county_regression.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))

    b = main_fit.params
    out = df[["Re", "Re_low", "Re_high", "f_h", "f_M"]].copy()
    out["Re_fitted"] = np.exp(b["const"] + b["ln_mobility"] * df.ln_mobility + b["ln_masks"] * df.ln_masks)
    out["Re_no_masks"] = np.exp(b["const"] + b["ln_mobility"] * df.ln_mobility)
    placebo = fit(df, ["ln_mobility", "day"])
    out["Re_fitted_time_trend"] = np.exp(placebo.predict(sm.add_constant(df[["ln_mobility", "day"]])))
    out.to_csv(RESULTS / "fig6_king_county_fit.tsv", sep="\t", float_format="%.4f")
    print("\nfitted vs counterfactual, selected days (paper on 30 August: 1.08 and 1.48):")
    print(out.loc[["2020-06-01", "2020-06-25", "2020-07-15", "2020-08-15", "2020-08-30"],
                  ["Re", "Re_fitted", "Re_no_masks", "f_h", "f_M"]].round(3).to_string())


if __name__ == "__main__":
    main()
