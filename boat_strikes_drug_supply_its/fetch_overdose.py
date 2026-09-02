"""Pull CDC VSRR provisional drug-overdose death counts.

Every publicly released provisional series is a *12-month-ending* rolling total,
not a monthly count. The rolling window is the central data problem in this
analysis and `deconvolve.py` handles it explicitly; this module only fetches.

Two value columns matter:
  data_value       reported deaths in the 12 months ending that month
  predicted_value  the same, inflated for deaths still pending investigation

CDC advises the predicted value for recent months, where pending cases are
concentrated. Because the strike campaign begins in the most recent months,
using the reported value alone would manufacture a downward artefact at exactly
the point of interest, so `predicted_value` is the primary series here.

Usage:
    python -P fetch_overdose.py
    python -P fetch_overdose.py --offline
"""

from __future__ import annotations

import argparse
import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

RESOURCE = "https://data.cdc.gov/resource/xkb8-kh2a.csv"

# Indicator label -> short name. The comparators are chosen for route contrast:
# cocaine reaches the US mainly by sea from South America, whereas fentanyl and
# methamphetamine are overwhelmingly produced in and moved overland from Mexico.
INDICATORS = {
    "Cocaine (T40.5)": "cocaine",
    "Synthetic opioids, excl. methadone (T40.4)": "synthetic_opioids",
    "Psychostimulants with abuse potential (T43.6)": "psychostimulants",
    "Heroin (T40.1)": "heroin",
    "Number of Drug Overdose Deaths": "all_drug",
}

HERE = Path(__file__).parent
OUT = HERE / "data" / "overdose_12mo.csv"


def fetch(state: str = "US") -> pd.DataFrame:
    frames = []
    for label, short in INDICATORS.items():
        query = urllib.parse.urlencode(
            {
                "state": state,
                "indicator": label,
                "$select": "year,month,data_value,predicted_value,"
                           "percent_complete,percent_pending_investigation",
                "$order": "year,month",
                "$limit": "5000",
            }
        )
        with urllib.request.urlopen(f"{RESOURCE}?{query}", timeout=120) as resp:
            frame = pd.read_csv(io.BytesIO(resp.read()))
        frame["drug"] = short
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["window_end"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01",
        format="%Y-%B-%d",
    )
    df = df.dropna(subset=["data_value"])
    df["rolling12_reported"] = df["data_value"].astype(float)
    df["rolling12"] = df["predicted_value"].fillna(df["data_value"]).astype(float)
    keep = ["drug", "window_end", "rolling12", "rolling12_reported",
            "percent_complete", "percent_pending_investigation"]
    return (df[keep]
            .sort_values(["drug", "window_end"])
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="reuse the committed CSV instead of refetching")
    args = ap.parse_args()

    if args.offline:
        df = pd.read_csv(OUT, parse_dates=["window_end"])
    else:
        df = tidy(fetch())
        df.to_csv(OUT, index=False)

    print(f"{len(df)} rows, {df['drug'].nunique()} drugs")
    for drug, grp in df.groupby("drug"):
        last = grp.iloc[-1]
        print(f"  {drug:18} {grp['window_end'].min():%Y-%m} .. "
              f"{grp['window_end'].max():%Y-%m}  "
              f"latest 12-mo total {last['rolling12']:,.0f}")


if __name__ == "__main__":
    main()
