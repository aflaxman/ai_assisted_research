#!/usr/bin/env python3
"""Fit each structure's transmission model to GBD incidence + prevalence, then
predict deaths, LTBI prevalence, and the subclinical fraction.

Natural-history parameters are fixed at the cohort-calibration posterior medians.
Only beta (transmission) and the passive detection rate are fitted, to the GBD
incidence and prevalence at the target year; deaths, LTBI and the subclinical
fraction are predictions.  Reinfection protection and the HIV progression RR are
held at literature values.  The model is run to endemic equilibrium with the
target-year HIV prevalence, ART coverage and demography held constant.

Usage:
    uv run python scripts/fit_transmission.py --all
    uv run python scripts/fit_transmission.py SP-full --countries "Viet Nam,India"
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "data" / "derived" / "posteriors"
OUT = ROOT / "data" / "derived" / "transmission_fits.csv"

REINF_PROTECTION = 0.5
HIV_PROG_RR = 20.0
HIV_MORT_RR = 2.0
BURN_YEARS = 400.0


def detection_from_haq(h: float) -> float:
    """Passive case-detection rate (per year) as a function of the GBD Healthcare
    Access and Quality index.  Higher HAQ -> faster detection.  A transparent
    linear map, documented as an assumption (not fitted)."""
    if not np.isfinite(h):
        h = 40.0
    return 0.8 + 2.2 * (h / 100.0)


def load_medians(name: str) -> dict | None:
    f = POST / name / "samples.npz"
    if not f.exists():
        return None
    s = np.load(f)
    p = {k: float(np.median(s[k])) for k in s.files}
    return p


def equilibrium(tm, hiv_prev, art_cov, beta, detection):
    from tbmodels.transmission import TransmissionInputs, TransmissionParams
    inp = TransmissionInputs(
        hiv_prevalence=lambda t: hiv_prev, art_coverage=lambda t: art_cov,
        detection_rate=lambda t: detection, birth_rate=1 / 65, bg_mortality=1 / 65)
    tp = TransmissionParams(beta=beta, reinf_protection=REINF_PROTECTION,
                            hiv_prog_rr=HIV_PROG_RR, hiv_mort_rr=HIV_MORT_RR)
    times = np.linspace(0, BURN_YEARS, int(BURN_YEARS) + 1)
    traj = tm.simulate(times, inp, tp)
    out = tm.outputs(traj, times)
    # use the final-year increments for incidence/mortality
    N = out["N"][-1]
    cum_inc, cum_death = traj[:, tm.n], traj[:, tm.n + 2]
    inc = (cum_inc[-1] - cum_inc[-2]) / N * 1e5
    dth = (cum_death[-1] - cum_death[-2]) / N * 1e5
    return dict(incidence=inc, prevalence=out["prevalence"][-1], deaths=dth,
                ltbi=out["ltbi"][-1], subclinical_fraction=out["subclinical_fraction"][-1])


def fit_beta_to_incidence(tm, tgt, hiv_prev, art_cov, detection):
    """1D fit of beta so the equilibrium incidence matches GBD incidence.

    Detection is fixed (from HAQ).  Incidence is monotone increasing in beta
    above the R0=1 threshold, so we grid-scan to bracket the target, then refine
    with brentq on the log-residual.  Returns (beta, equilibrium, converged)."""
    inc_t = tgt["incidence"]["rate"]

    def inc_at(beta):
        return equilibrium(tm, hiv_prev, art_cov, beta, detection)["incidence"]

    grid = np.geomspace(2.0, 250.0, 24)
    incs = np.array([inc_at(b) for b in grid])
    # find the bracket where modelled incidence crosses the target
    below = incs < inc_t
    if below.all():
        return grid[-1], equilibrium(tm, hiv_prev, art_cov, grid[-1], detection), False
    if (~below).all():
        return grid[0], equilibrium(tm, hiv_prev, art_cov, grid[0], detection), False
    k = np.where(below[:-1] & ~below[1:])[0]
    lo, hi = (grid[k[0]], grid[k[0] + 1]) if len(k) else (grid[0], grid[-1])
    try:
        beta = brentq(lambda b: inc_at(b) - inc_t, lo, hi, xtol=1e-3, rtol=1e-5, maxiter=60)
        conv = True
    except Exception:  # noqa: BLE001
        beta = grid[np.argmin(np.abs(incs - inc_t))]
        conv = False
    return beta, equilibrium(tm, hiv_prev, art_cov, beta, detection), conv


def main(argv):
    from tbmodels.registry import all_structures
    from tbmodels.transmission import TransmissionModel, TRANSMISSION_EXCLUDED
    from tbmodels.gbd_targets import targets, hiv_inputs, haq, COUNTRIES

    if "--all" in argv:
        names = [n for n in all_structures() if n not in TRANSMISSION_EXCLUDED]
    else:
        names = [a for a in argv[1:] if not a.startswith("--")]
    countries = COUNTRIES
    if "--countries" in argv:
        countries = argv[argv.index("--countries") + 1].split(",")
    year = 2019

    structs = all_structures()
    rows = []
    for name in names:
        med = load_medians(name)
        if med is None:
            print(f"{name}: no cohort posterior yet, skipping")
            continue
        nh = structs[name]
        if nh.family == "spectrum":
            med.setdefault("rel_inf_sub", 0.5)
        tm = TransmissionModel(nh, med)
        for country in countries:
            tgt = targets(country, year)
            if not tgt["incidence"] or not tgt["prevalence"]:
                continue
            hp, ac = hiv_inputs(country)
            hiv_prev, art_cov = hp(year), ac(year)
            det = detection_from_haq(haq(country, year))
            try:
                beta, eq, conv = fit_beta_to_incidence(tm, tgt, hiv_prev, art_cov, det)
            except Exception as e:  # noqa: BLE001
                print(f"{name}/{country}: fit failed: {e}")
                continue
            prev_gbd = tgt["prevalence"].get("rate", float("nan"))
            ltbi_gbd = tgt["ltbi_fraction"].get("rate", float("nan"))
            rows.append(dict(
                structure=name, family=nh.family, country=country, year=year,
                beta=round(beta, 3), detection_rate=round(det, 3), converged=conv,
                hiv_prev=round(hiv_prev, 4), art_cov=round(art_cov, 3),
                inc_model=round(eq["incidence"], 1), inc_gbd=round(tgt["incidence"]["rate"], 1),
                prev_model=round(eq["prevalence"], 1), prev_gbd=round(prev_gbd, 1),
                prev_ratio=round(eq["prevalence"] / prev_gbd, 2) if prev_gbd else "",
                deaths_model=round(eq["deaths"], 2), deaths_gbd=round(tgt["deaths"]["rate"], 2),
                ltbi_model=round(eq["ltbi"], 4), ltbi_gbd=round(ltbi_gbd, 4),
                subclinical_fraction=round(eq["subclinical_fraction"], 3)
                if np.isfinite(eq["subclinical_fraction"]) else "",
            ))
            print(f"{name}/{country}: beta={beta:.1f} det={det:.2f} conv={conv} "
                  f"inc {eq['incidence']:.0f}/{tgt['incidence']['rate']:.0f} "
                  f"prev {eq['prevalence']:.0f}/{prev_gbd:.0f} "
                  f"deaths {eq['deaths']:.1f}/{tgt['deaths']['rate']:.1f} "
                  f"LTBI {eq['ltbi']:.2f}/{ltbi_gbd:.2f} "
                  f"subfrac={eq['subclinical_fraction']:.2f}")

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
