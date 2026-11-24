"""Cross-check: camdl's ODE backend against scipy's LSODA on the same model.

The authors integrate with ``deSolve::lsodar`` in R.  ``within_host.camdl``
runs on camdl's adaptive Runge-Kutta; ``within_host_scipy.py`` is the same
six equations written a second time for LSODA.  This compares the daily viral
loads of the first 1,000 published parameter sets, untreated and under an
antiviral of 50% efficacy started at a different time for each person, where
the switch in viral production tests the integrators at a discontinuity.

    uv run python compare_integrators.py
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

import viral_load as vl
from within_host_scipy import viral_load as scipy_viral_load

N = 1000
RAW = pd.read_csv(vl.AUTHORS_PARAMS).iloc[:N].to_dict("records")
T_TREAT = np.random.default_rng(7).uniform(0.5, 5.0, N)


def _untreated(par):
    return scipy_viral_load(par, days=vl.DAYS, tau=vl.TAU, rtol=1e-11, atol=1e-11)


def _treated(args):
    par, t = args
    return scipy_viral_load(par, days=vl.DAYS, tau=vl.TAU, eps=0.5, t_treat=t, rtol=1e-11, atol=1e-11)


def report(label, camdl, ref):
    relevant = ref > 1.0e3          # below 1,000 copies the transmission risk is under 1e-6
    rel = np.abs(camdl - ref)[relevant] / ref[relevant]
    print(f"{label:<28} {relevant.sum():>6} values above 1e3 copies: "
          f"median relative difference {np.median(rel):.1e}, largest {rel.max():.1e}")
    return {"case": label, "values_compared": int(relevant.sum()),
            "median_relative_difference": float(np.median(rel)), "max_relative_difference": float(rel.max())}


def main():
    par = vl.authors_parameters().iloc[:N]
    fixed = {name: par[name].iloc[0] for name in vl.FIXED}
    with Pool() as pool:
        ref0 = np.array(pool.map(_untreated, RAW, chunksize=50))
        ref1 = np.array(pool.map(_treated, list(zip(RAW, T_TREAT)), chunksize=50))
    cam0 = vl.run_camdl(par[vl.SWEPT], {**fixed, **vl.NO_TREATMENT})
    cam1 = vl.run_camdl(par[vl.SWEPT].assign(t_treat=T_TREAT), {**fixed, "eps_treat": 0.5}, treat=True)
    rows = [report("untreated", cam0, ref0), report("50% antiviral, random start", cam1, ref1)]
    pd.DataFrame(rows).to_csv(vl.ROOT / "results" / "integrator_check.tsv", sep="\t", index=False,
                              float_format="%.3g")


if __name__ == "__main__":
    main()
