"""Condense raw vivarium_inputs pulls into the small CSVs under data/gbd_vi/.

Usage: python -m sodium_sim.prepare_gbd_vi <raw_pull_dir>

The raw pulls carry 250 draws per row and the SBP relative risk is a curve on
a 1,000-point exposure grid per cause, age and sex (about 1 GB as CSV). This
keeps the draw mean (and, for the main tables, the 2.5th and 97.5th
percentiles) and for the relative-risk curves both a thinned copy of the
mean curve and a log-linear slope fitted between 115 and 175 mmHg, which is
the form the paper's GBD 2019 relative risks took (one RR per 10 mmHg).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sodium_sim.build_artifact import ROOT

OUT = ROOT / "data" / "gbd_vi"
# Exposure range for the log-linear fit: above the TMREL and across the bulk of the population.
FIT_RANGE = {"high_systolic_blood_pressure": (115.0, 175.0), "diet_high_in_sodium": (2.0, 6.0)}


def condense(df: pd.DataFrame) -> pd.DataFrame:
    draws = [c for c in df.columns if c.startswith("draw_")]
    if not draws:
        return df
    out = df.drop(columns=draws)
    out["value"] = df[draws].mean(axis=1)
    out["lower"] = df[draws].quantile(0.025, axis=1)
    out["upper"] = df[draws].quantile(0.975, axis=1)
    return out


def condense_rr_curve(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (thinned mean curve, fitted log-linear RR per unit) for a curve-form RR file."""
    risk = path.stem.split(".")[1]
    lo, hi = FIT_RANGE[risk]
    keys = ["location", "sex", "age_start", "age_end", "year_start", "year_end",
            "affected_entity", "affected_measure", "parameter"]
    parts = []
    for chunk in pd.read_csv(path, chunksize=20000):
        draws = [c for c in chunk.columns if c.startswith("draw_")]
        parts.append(chunk[keys].assign(value=chunk[draws].mean(axis=1)))
    curve = pd.concat(parts, ignore_index=True)
    group_cols = [c for c in keys if c != "parameter"]
    thinned = curve[curve.groupby(group_cols).cumcount() % 20 == 0].reset_index(drop=True)
    fits = []
    for key, g in curve.groupby(group_cols):
        g = g[(g["parameter"] >= lo) & (g["parameter"] <= hi) & (g["value"] > 0)]
        if g.empty:
            slope = 0.0
        else:
            slope = np.polyfit(g["parameter"], np.log(g["value"]), 1)[0]
        fits.append(dict(zip(group_cols, key), value=float(np.exp(slope))))
    return thinned, pd.DataFrame(fits)


def main(raw: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw.glob("*.csv")):
        key = path.stem
        if key.endswith(".relative_risk") and path.stat().st_size > 5_000_000:
            if (OUT / f"{key}_per_unit.csv").exists() and (OUT / f"{key}_per_unit.csv").stat().st_mtime > path.stat().st_mtime:
                continue
            thinned, fit = condense_rr_curve(path)
            thinned.to_csv(OUT / f"{key}_curve.csv", index=False)
            fit.to_csv(OUT / f"{key}_per_unit.csv", index=False)
            print(f"{key}: curve {thinned.shape}, per-unit fit {fit.shape}", flush=True)
            continue
        df = condense(pd.read_csv(path))
        df.to_csv(OUT / f"{key}.csv", index=False)
        print(f"{key}: {df.shape}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
