"""Antiviral results of Goyal et al. (2021): Figure 7.

Treatment multiplies viral production by ``1 - eps_treat`` from ``t_treat``
days after exposure (``within_host.camdl``).  Three families of scenario:

* early symptomatic therapy, 100% efficacy, started 0.5 to 5 days after
  symptom onset, where the incubation period is Gamma(3.45, rate 0.66) as in
  the authors' eLife code;
* post-exposure prophylaxis (PEP) of efficacy 10% to 100%, started 0.5 to 5
  days after exposure;
* fully effective PEP started on a given day after exposure (panel G).  The
  paper plots six points at 0.5, 1.5, ..., 5.5 days and does not say how the
  start time was set.  Two readings are run: a start at exactly that time,
  and a start uniform over the day around it.

Uptake is an independent coin flip per infected person, so every population
summary here (mean R_e, the share who transmit, the share who infect five or
more) is linear in uptake: the value at uptake u is (1 - u) times the
untreated value plus u times the fully treated one.  Only the two endpoints
need simulating.

    uv run python run_antivirals.py        # about 45 minutes the first time
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

import transmission as tr
import viral_load as vl

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
N_PAPER = 3000
REPEATS = 60            # passes over the 10,000 parameter sets for distributional summaries


def summaries(V: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Exact mean R_e, and Monte Carlo shares of transmitters and super-spreaders."""
    r = np.concatenate([tr.simulate(V, rng).r_individual for _ in range(REPEATS)])
    return {
        "Re_exact": tr.expected_re(V, 0.0, 0.0, 0.0),
        "Re_mc": r.mean(),
        "se_3000": r.std() / np.sqrt(N_PAPER),
        "pct_transmit": 100.0 * (r >= 1).mean(),
        "pct_all_ge5": 100.0 * (r >= 5).mean(),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(20210512)
    V0 = vl.untreated()
    n = len(V0)

    # One set of draws per person, reused across scenarios.
    incubation = rng.gamma(shape=3.45, scale=1.0 / 0.66, size=n)
    delay = rng.uniform(0.5, 5.0, size=n)
    within_day = rng.uniform(-0.5, 0.5, size=n)

    rows = [{"scenario": "untreated", "efficacy": 0.0, "start": "none", **summaries(V0, rng)}]

    V = vl.treated("viral_load_symptomatic_e100", incubation + delay, 1.0)
    rows.append({"scenario": "early symptomatic therapy", "efficacy": 1.0,
                 "start": "0.5-5 d after symptom onset", **summaries(V, rng)})

    for eff in np.round(np.arange(0.1, 1.01, 0.1), 1):
        V = vl.treated(f"viral_load_pep_e{int(round(eff * 100)):03d}", delay, eff)
        rows.append({"scenario": "post-exposure prophylaxis", "efficacy": eff,
                     "start": "0.5-5 d after exposure", **summaries(V, rng)})

    for day in np.arange(0.5, 6.0, 1.0):
        V = vl.treated(f"viral_load_pep_at_{day:.1f}", np.full(n, day), 1.0)
        rows.append({"scenario": "PEP on a given day, start at that time", "efficacy": 1.0,
                     "start": f"{day:.1f}", **summaries(V, rng)})
        V = vl.treated(f"viral_load_pep_during_{day:.1f}", day + within_day, 1.0)
        rows.append({"scenario": "PEP on a given day, start uniform over the day", "efficacy": 1.0,
                     "start": f"{day:.1f}", **summaries(V, rng)})

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    # Uptake at which R_e falls to 1, by linearity in uptake.
    out["uptake_for_Re_1"] = (base.Re_exact - 1.0) / (base.Re_exact - out.Re_exact)
    out.loc[(out.uptake_for_Re_1 > 1.0) | (out.uptake_for_Re_1 < 0.0), "uptake_for_Re_1"] = np.nan
    out.to_csv(RESULTS / "fig7_antivirals.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
