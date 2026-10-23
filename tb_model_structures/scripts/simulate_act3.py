#!/usr/bin/env python3
"""Simulate the ACT3 community-wide screening trial and compare each structure's
predicted prevalence decline to the observed rounds.

ACT3 (Marks 2019, NEJM 381:1347): in Ca Mau, Viet Nam, adults were screened for
pulmonary TB once a year for three years (2014-2016) with Xpert on spontaneously
expectorated sputum, regardless of symptoms; a prevalence survey followed in
year 4 (2017).  Observed Xpert-positive prevalence (per 100k):

    round 1 (2014): intervention 389
    round 2 (2015): intervention 308
    round 3 (2016): intervention 176
    year 4  (2017): intervention 126, control 226   (prevalence ratio 0.56)

Each annual round removes a fraction f = coverage * Xpert-sensitivity of the
*bacteriologically positive* prevalent pool (subclinical and clinical) to
treatment, regardless of symptoms.  That is the structural lever: a spectrum
model has an asymptomatic subclinical pool that active screening reaches but
passive detection misses, while a classic model's single active-TB state is
already partly found by passive care-seeking.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POST = ROOT / "data" / "derived" / "posteriors"
FITS = ROOT / "data" / "derived" / "transmission_fits.csv"
OUT = ROOT / "data" / "derived" / "act3_simulation.csv"

OBSERVED = {1: 389.0, 2: 308.0, 3: 176.0, 4: 126.0}
OBSERVED_CONTROL_Y4 = 226.0
COUNTRY = "Viet Nam"
YEAR0 = 2019  # calibrate at the GBD reference year, then run the trial protocol


def load_fit_row(structure: str) -> dict | None:
    if not FITS.exists():
        return None
    with open(FITS) as f:
        for r in csv.DictReader(f):
            if r["structure"] == structure and r["country"] == COUNTRY:
                return r
    return None


def bactpos_prevalence(tm, x) -> float:
    living = x[: tm.n]
    N = living.sum()
    states = [tm.idx[c.name] for c in tm.nh.compartments if c.bact_positive]
    return living[states].sum() / N * 1e5


def run(structure: str, coverage: float, xpert_sens: float) -> list[dict]:
    from tbmodels.registry import get
    from tbmodels.transmission import TransmissionModel, TransmissionInputs, TransmissionParams, TRANSMISSION_EXCLUDED
    from tbmodels.gbd_targets import hiv_inputs

    if structure in TRANSMISSION_EXCLUDED:
        return []
    med_f = POST / structure / "samples.npz"
    fit = load_fit_row(structure)
    if not med_f.exists() or fit is None:
        return []
    med = {k: float(np.median(v)) for k, v in np.load(med_f).items()}
    nh = get(structure)
    if nh.family == "spectrum":
        med.setdefault("rel_inf_sub", 0.5)
    tm = TransmissionModel(nh, med)

    hp, ac = hiv_inputs(COUNTRY)
    beta = float(fit["beta"]); det = float(fit["detection_rate"])
    inp = TransmissionInputs(hiv_prevalence=lambda t: hp(YEAR0), art_coverage=lambda t: ac(YEAR0),
                             detection_rate=lambda t: det, birth_rate=1 / 65, bg_mortality=1 / 65)
    tp = TransmissionParams(beta=beta, reinf_protection=0.5, hiv_prog_rr=20.0, hiv_mort_rr=2.0)

    # equilibrate (fast steady-state solve)
    x = tm.equilibrium(inp, tp)["state"].copy()
    bact_states = [tm.idx[c.name] for c in tm.nh.compartments if c.bact_positive]
    rec_i = tm.idx["Rec"] if "Rec" in tm.idx else None
    f = coverage * xpert_sens

    rows = []
    # rounds 1-3: measure prevalence, then remove fraction f of bact-positive
    for rnd in (1, 2, 3):
        prev = bactpos_prevalence(tm, np.concatenate([x, [0, 0, 0]]))
        removed = 0.0
        for s_i in bact_states:
            take = f * x[s_i]
            x[s_i] -= take
            removed += take
        if rec_i is not None:
            x[rec_i] += removed
        rows.append(dict(structure=structure, family=nh.family, round=rnd,
                         model_prevalence=round(prev, 1), observed_prevalence=OBSERVED[rnd]))
        # integrate one year of natural dynamics + passive detection (no screening)
        seg = tm.simulate(np.array([0.0, 1.0]), inp, tp, x0=x)
        x = seg[-1, : tm.n]
    # year 4 survey (no screening pulse this year)
    prev4 = bactpos_prevalence(tm, np.concatenate([x, [0, 0, 0]]))
    rows.append(dict(structure=structure, family=nh.family, round=4,
                     model_prevalence=round(prev4, 1), observed_prevalence=OBSERVED[4]))
    # control arm = equilibrium prevalence with no screening
    xc = tm.equilibrium(inp, tp)["state"]
    prev_ctrl = bactpos_prevalence(tm, np.concatenate([xc, [0, 0, 0]]))
    rows.append(dict(structure=structure, family=nh.family, round="control_y4",
                     model_prevalence=round(prev_ctrl, 1), observed_prevalence=OBSERVED_CONTROL_Y4))
    return rows


def main(argv):
    from tbmodels.registry import all_structures
    from tbmodels.transmission import TRANSMISSION_EXCLUDED
    coverage = 0.65
    xpert_sens = 0.80
    if "--coverage" in argv:
        coverage = float(argv[argv.index("--coverage") + 1])
    if "--sens" in argv:
        xpert_sens = float(argv[argv.index("--sens") + 1])
    if "--all" in argv:
        names = [n for n in all_structures() if n not in TRANSMISSION_EXCLUDED]
    else:
        names = [a for a in argv[1:] if not a.startswith("--")]

    all_rows = []
    for name in names:
        rows = run(name, coverage, xpert_sens)
        all_rows.extend(rows)
        for r in rows:
            print(f"{name:16s} round {str(r['round']):10s} model={r['model_prevalence']:7.1f} "
                  f"observed={r['observed_prevalence']}")
    if all_rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["structure", "family", "round",
                                              "model_prevalence", "observed_prevalence"])
            w.writeheader(); w.writerows(all_rows)
        print(f"wrote {OUT} ({len(all_rows)} rows); coverage={coverage} xpert_sens={xpert_sens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
