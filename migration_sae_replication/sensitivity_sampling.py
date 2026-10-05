"""How much does the power prior's under-coverage depend on the share of cases with known RR status?

Dupuis (2026) never states the sampling fraction. Because the power prior re-uses each area's
full (observed + imputed) case count as likelihood counts, its posterior precision is inflated by
roughly (1 + 1/f) relative to the baseline, where f is the sampled fraction. This script reruns
the prediction study for one base pattern at three sampling fractions.

Usage: uv run python sensitivity_sampling.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np
import pandas as pd

import synthetic
from simulation import prediction_task
from summarize import scenario_metrics

ROWS = []
for scale, label in [(1.0, "~10% known"), (2.5, "~25% known"), (5.0, "~50% known")]:
    synthetic.PI_KNOWN = {0: 0.08 * scale, 1: 0.16 * scale}
    for rep in range(3):
        df, _ = prediction_task("block", 100 + rep, ["neighbors", "distance", "crisis_idp"], ["medium"])
        df["sampling"] = label
        ROWS.append(df)
raw = pd.concat(ROWS, ignore_index=True)
out = []
for label, sub in raw.groupby("sampling"):
    scen, _ = scenario_metrics(sub.drop(columns="sampling"))
    scen["sampling"] = label
    out.append(scen)
res = pd.concat(out)[["sampling", "pattern", "method", "rmse", "bias", "coverage", "sd"]]
res = res[res["method"].isin(["naive", "pp", "pp_sample"])]
os.makedirs("results", exist_ok=True)
res.to_csv("results/sensitivity_sampling_fraction.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
