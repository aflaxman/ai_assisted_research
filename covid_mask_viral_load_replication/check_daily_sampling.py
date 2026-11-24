"""Does it matter that the paper reads viral load once a day?

The authors count a day's secondary infections from the viral load at one
instant of that day, "assuming that viral load, and subsequent risk, does not
change substantially within a day".  Viral load rises five orders of magnitude
in two days, so the assumption is false for any one person.  This script asks
what it does to the population mean: camdl writes the viral load of the first
2,000 published parameter sets every 0.05 days, and the expected R0 is
computed from daily readings at each possible time of day and from the
continuous integral, theta times the time integral of P(V)^2.

    uv run python check_daily_sampling.py
"""
from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np
import pandas as pd

import transmission as tr
import viral_load as vl

N, STEP = 2000, 0.05


def main():
    par = vl.authors_parameters().iloc[:N]
    fixed = {**{name: par[name].iloc[0] for name in vl.FIXED}, **vl.NO_TREATMENT}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "fixed.toml").write_text("".join(f"{k} = {float(v)!r}\n" for k, v in fixed.items()))
        par[vl.SWEPT].to_csv(tmp / "design.tsv", sep="\t", index=False, float_format="%.17g")
        subprocess.run(["camdl", "simulate", str(vl.MODEL), "--params", str(tmp / "fixed.toml"),
                        "--sweep-file", str(tmp / "design.tsv"), "--backend", "ode", "--columns", "V",
                        "--output-every", str(STEP), "--no-progress", "--parallel", "0",
                        "--store", str(tmp / "store"), "-o", str(tmp / "out.tsv")],
                       check=True, capture_output=True)
        out = pd.read_csv(tmp / "out.tsv", sep="\t", comment="#", usecols=["point_id", "t", "V"])
    wide = out.pivot(index="point_id", columns="t", values="V")
    risk = tr.dose_response(wide.to_numpy()) ** 2
    per_day = int(round(1.0 / STEP))
    rows = [{"viral_load_read": f"once a day, {offset * STEP:.2f} d after the paper's time",
             "mean_R0": tr.THETA * risk[:, offset::per_day].sum(axis=1).mean()}
            for offset in range(0, per_day, 4)]
    rows.append({"viral_load_read": "continuously (integral)",
                 "mean_R0": tr.THETA * np.trapezoid(risk, dx=STEP, axis=1).mean()})
    table = pd.DataFrame(rows)
    table.to_csv(vl.ROOT / "results" / "daily_sampling_check.tsv", sep="\t", index=False, float_format="%.4f")
    print(table.to_string(index=False))
    peak_time = wide.columns.to_numpy()[wide.to_numpy().argmax(axis=1)]
    print(f"peak viral load: median 10^{np.median(np.log10(wide.max(axis=1))):.2f} copies, "
          f"{np.median(peak_time):.1f} days after exposure "
          f"(interquartile range {np.percentile(peak_time, 25):.1f} to {np.percentile(peak_time, 75):.1f})")


if __name__ == "__main__":
    main()
