"""Scenario-level metrics from the long-format simulation output, following Dupuis (2026):

per district j: bias_j = mean over replications of (posterior mean - truth);
                bias2_j = bias_j^2;  rmse_j = sqrt(mean over replications of squared error)
scenario level: averages of these over districts; Spearman correlation between estimates and
                truth computed per replication and averaged; 95% interval coverage over all
                (district, replication) pairs; mean posterior sd.
Differences are reported as (method - naive) for bias, bias2 and rmse and (method - naive)
for Spearman and coverage, so negative error differences and positive correlation
differences favour the migration-adjusted method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["study", "base", "pattern", "level", "method"]


def scenario_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["err"] = df["est"] - df["truth"]
    df["covered"] = (df["lo"] <= df["truth"]) & (df["truth"] <= df["hi"])
    per_district = df.groupby(KEYS + ["district"]).agg(
        bias=("err", "mean"), mse=("err", lambda e: np.mean(np.square(e))),
        coverage=("covered", "mean"), sd=("sd", "mean"), n_rep=("rep", "nunique")).reset_index()
    per_district["bias2"] = per_district["bias"] ** 2
    per_district["rmse"] = np.sqrt(per_district["mse"])
    scen = per_district.groupby(KEYS).agg(bias=("bias", "mean"), bias2=("bias2", "mean"), rmse=("rmse", "mean"),
                                          coverage=("coverage", "mean"), sd=("sd", "mean"),
                                          n_rep=("n_rep", "max")).reset_index()
    sp = (df.groupby(KEYS + ["rep"])[["est", "truth"]]
            .apply(lambda g: g["est"].corr(g["truth"], method="spearman")).rename("spearman").reset_index())
    sp = sp.groupby(KEYS)["spearman"].mean().reset_index()
    scen = scen.merge(sp, on=KEYS)
    return scen, per_district


def differences(scen: pd.DataFrame, reference="naive") -> pd.DataFrame:
    ref = scen[scen["method"] == reference].drop(columns=["method"])
    cols = ["bias", "bias2", "rmse", "coverage", "sd", "spearman"]
    out = scen[scen["method"] != reference].merge(ref, on=["study", "base", "pattern", "level"], suffixes=("", "_ref"))
    for c in cols:
        out[f"d_{c}"] = out[c] - out[f"{c}_ref"]
    return out


def main(paths):
    for p in paths:
        p = Path(p)
        df = pd.read_parquet(p)
        scen, per_district = scenario_metrics(df)
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        scen.to_csv(out_dir / f"{p.stem}_scenario_metrics.csv", index=False)
        per_district.to_csv(out_dir / f"{p.stem}_district_metrics.csv", index=False)
        diff = differences(scen)
        diff.to_csv(out_dir / f"{p.stem}_differences.csv", index=False)
        pd.set_option("display.width", 200)
        print(f"== {p.stem}: {scen.n_rep.max()} replications")
        print(scen.groupby(["level", "method"])[["bias", "bias2", "rmse", "spearman", "coverage", "sd"]].mean().round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
