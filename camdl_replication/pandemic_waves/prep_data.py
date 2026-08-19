"""Prepare camdl observation TSVs from Frimpong & Bauch's published data.

Source: https://github.com/SefahF/Pandemic-waves-as-the-outcome-of-coupled-social-and-disease-dynamics
(the paper's own repository; clone it and pass its path as argv[1]).

The paper's estimation scripts read two series per country:
- {country}Avg7ma.csv    — 7-day moving average of OWID daily reported
  cases, divided by population (a proportion)
- {country}Avgosi7ma.csv — Oxford stringency index / 100 (a proportion)

camdl's `normal(...)` likelihood is the discretized-count Normal (values
are rounded to the nearest integer before scoring), so both series are
rescaled to count-like magnitudes where that rounding is negligible:
cases as persons (proportion x population), OSI as permille (x 1000).

Time convention matches the paper's fitting code: data row i (0-based)
is compared against the model at day t = i + 1, so the TSV time column
is i + 1. The fit window is time <= cutoff (the paper's Table S1), the
rest is the out-of-sample prediction window.
"""

import csv
import sys
from pathlib import Path

# (name, paper job-array index, population, fit cutoff, x0 upper bound)
# from sirx_model_estimations.py in the paper's repository. Note the
# Netherlands population is Germany's (a bug in the paper's script);
# kept as-is for fidelity — it only scales the count units and the
# I0 prior, not the proportion-space dynamics.
COUNTRIES = [
    ("austria",     "austria",     9006398,  200, 0.20),
    ("belarus",     "belarus",     9449323,  230, 0.05),
    ("belgium",     "belgium",     11589623, 220, 0.20),
    ("denmark",     "denmark",     5792202,  230, 0.10),
    ("finland",     "finland",     5540720,  200, 0.05),
    ("france",      "france",      65411076, 200, 0.10),
    ("germany",     "germany",     83783942, 200, 0.10),
    ("ireland",     "ireland",     4937786,  200, 0.20),
    ("netherlands", "netherlands", 83783942, 200, 0.10),
    ("norway",      "norway",      5421241,  200, 0.20),
    ("portugal",    "portugal",    10196709, 200, 0.10),
    ("uk",          "uk",          67886011, 250, 0.10),
    ("switzerland", "switzerland", 8654622,  200, 0.10),
]


def read_series(path: Path) -> list[float]:
    with open(path) as f:
        return [float(row[1]) for row in csv.reader(f)]


def write_tsv(path: Path, name: str, values: list[float]) -> None:
    with open(path, "w") as f:
        f.write(f"time\t{name}\n")
        for i, v in enumerate(values):
            f.write(f"{i + 1}\t{v:.6f}\n")


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "../../..") / "Countries"
    if not (src / "austriaAvg7ma.csv").exists():
        sys.exit(f"source data not found under {src} — pass the paper repo clone path")
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    for name, stem, pop, cutoff, _ in COUNTRIES:
        cases_prop = read_series(src / f"{stem}Avg7ma.csv")
        osi_prop = read_series(src / f"{stem}Avgosi7ma.csv")
        n = min(len(cases_prop), len(osi_prop))
        cases = [p * pop for p in cases_prop[:n]]
        osi = [p * 1000.0 for p in osi_prop[:n]]
        write_tsv(out / f"{name}_cases.tsv", "cases", cases)
        write_tsv(out / f"{name}_osi.tsv", "osi", osi)
        # camdl parses but does not yet apply [data].holdout_after, so the
        # first-wave training window is materialised as separate files
        write_tsv(out / f"{name}_cases_train.tsv", "cases", cases[:cutoff])
        write_tsv(out / f"{name}_osi_train.tsv", "osi", osi[:cutoff])
        print(f"{name}: {n} days, cutoff {cutoff}, pop {pop}")


if __name__ == "__main__":
    main()
