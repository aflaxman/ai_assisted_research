"""
GBD re-base of the vitamin-D / T2D prevention model.

GOAL 2 ("how do results change under GBD evidence") + GOAL 3 ("other locations").
Instead of a parametric age-dependent onset hazard calibrated to the CDC/RTI paper,
this drives diabetes onset's AGE PATTERN and LEVEL, and background MORTALITY, from a
location's epidemiology curve (gbd_inputs.csv). Swapping the location swaps those
curves — which is exactly what a real GBD re-base does, one location_id at a time.

WHAT IS REAL vs ILLUSTRATIVE.
- Real: the pipeline, the cohort, the vitamin-D effect, the calibration logic, and the
  seam that consumes age/sex/location-specific incidence + mortality in the same shape
  GBD provides (see README for the artifact keys / get_draws recipe).
- Illustrative: the numbers in gbd_inputs.csv (GBD draws are unreachable from this
  sandbox). They are GBD/US-surveillance-anchored magnitudes, to be replaced by real
  draws. Costs are held at the paper's US values (location costing is separate goal-3
  work needing WHO-CHOICE / DEX), so this isolates the EPIDEMIOLOGY re-base.

METHOD.
- onset hazard(age) = PREDIAB_RR * gbd_incidence(age, loc) * REL_i   (REL = per-person
  marker heterogeneity from HbA1c/FPG/BMI, as in the onset+reversion model). Prediabetics
  progress faster than the general population, captured by the scalar PREDIAB_RR.
- Calibrate on the USA curve to the paper's control-arm Table 3 (10-yr & lifetime
  incidence, cost, QALY) via PREDIAB_RR, marker-heterogeneity scale, and the complication
  cost/disutility ramps. Then HOLD all of those fixed and swap only the location curve,
  so differences across locations come purely from GBD epidemiology.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT = os.path.join(HERE, "..", "nhanes_cohort", "outputs", "cohort_canonical_n4176.csv")
GBD = os.path.join(HERE, "gbd_inputs.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# fixed parameters (paper Table 2 / methods) -- costs held at US values
DISCOUNT = 0.03; V = 1 / (1 + DISCOUNT)
RRR = 0.15; ADHERENCE = 0.85; REV_RR = 1.30
C_NORMO, C_PREDIAB, C_DIAB_TX, C_VITD = 2636.0, 3244.0, 11322.0, 60.0
U_NORMO, U_PREDIAB, U_DIAB_BASE = 0.935, 0.920, 0.935
WTP = 100_000.0; DM_MORT_HR = 1.8; R_REV = 0.05; RELAPSE = 0.10
B_A1C, B_FPG, B_BMI = 1.5, 0.03, 0.03
MAX_AGE = 100; SEED = 2026
M_ONSET = 1 - ADHERENCE * RRR
M_REV = 1 + ADHERENCE * (REV_RR - 1)
# Diabetes complication cost/disutility that ACCRUE per year of diabetes duration.
# FIXED at the paper-consistent values from the US age-dependent onset model -- NOT
# re-solved to the paper's cost/QALY totals here, because those totals embed the paper's
# (higher) life expectancy; under GBD mortality the totals should be free to differ.
C_COMP, U_COMP = 1557.0, 0.0663
# US onset calibration targets (paper Table 3 control arm): incidence only.
TGT_INC10, TGT_INC = 0.2014, 0.3235

# ---- cohort & marker heterogeneity ------------------------------------------
c = pd.read_csv(COHORT)
AGE0 = c["age"].to_numpy(float); FEMALE = c["female"].to_numpy(float)
BMI = c["bmi"].to_numpy(float); A1C = c["hba1c"].to_numpy(float); FPG = c["fpg"].to_numpy(float)
ELIG = BMI >= 25
W = c["weight"].to_numpy(float); W = W / W.sum()
N = len(c); T = MAX_AGE - int(AGE0.min()) + 1


def dev(x, ref):
    d = x - ref
    return np.where(np.isnan(d), 0.0, d)


LP0 = (B_A1C * dev(A1C, np.nanmean(A1C)) + B_FPG * dev(FPG, np.nanmean(FPG))
       + B_BMI * dev(BMI, np.nanmean(BMI)))

_rng = np.random.default_rng(SEED)
U_ONSET = _rng.random((N, T)); U_REV = _rng.random((N, T))
U_RELAPSE = _rng.random((N, T)); U_MORT = _rng.random((N, T))

# ---- GBD location curves ----------------------------------------------------
gbd = pd.read_csv(GBD, comment="#")
LOC = {}
for loc, g in gbd.groupby("location"):
    g = g.sort_values("age_start")
    LOC[loc] = dict(age=g["age_start"].to_numpy(float),
                    inc=g["t2d_incidence_per1000"].to_numpy(float) / 1000.0,
                    acmr=g["acmr_per1000"].to_numpy(float) / 1000.0)


def inc_of(loc, age):
    return np.interp(age, LOC[loc]["age"], LOC[loc]["inc"])


def acmr_of(loc, age):
    return np.interp(age, LOC[loc]["age"], LOC[loc]["acmr"])


def simulate(arm, loc, prediab_rr, het):
    REL = np.exp(het * LP0)
    state = np.ones(N, int); dur = np.zeros(N)
    acc = {k: np.zeros(N) for k in ("py_n", "py_p", "py_d", "dur_d", "vitd",
                                    "py_n10", "py_p10", "py_d10", "dur_d10", "vitd10", "ly")}
    diab10 = np.zeros(N, bool); diab_life = np.zeros(N, bool)
    for t in range(T):
        age = AGE0 + t; disc = V ** t
        alive = state != 3
        n = alive & (state == 0); p = alive & (state == 1); d = alive & (state == 2)
        acc["py_n"][n] += disc; acc["py_p"][p] += disc
        acc["py_d"][d] += disc; acc["dur_d"][d] += disc * dur[d]; acc["ly"][alive] += 1
        if arm == "vitd":
            acc["vitd"][(n | p) & ELIG] += disc * C_VITD
        if t < 10:
            acc["py_n10"][n] += disc; acc["py_p10"][p] += disc
            acc["py_d10"][d] += disc; acc["dur_d10"][d] += disc * dur[d]
            if arm == "vitd":
                acc["vitd10"][(n | p) & ELIG] += disc * C_VITD
        # mortality from the location's all-cause rate; diabetics carry excess HR
        q = acmr_of(loc, age)
        q_eff = q.copy(); q_eff[d] = 1 - (1 - q[d]) ** DM_MORT_HR
        died = alive & ((U_MORT[:, t] < q_eff) | (age >= MAX_AGE)); state[died] = 3
        surv = alive & ~died
        sp = surv & (state == 1); sn = surv & (state == 0)
        # onset age-pattern & level from GBD incidence, x prediabetes RR x marker het
        h_on = prediab_rr * inc_of(loc, age) * REL
        r_ev = np.full(N, R_REV)
        if arm == "vitd":
            h_on = np.where(ELIG, h_on * M_ONSET, h_on)
            r_ev = np.where(ELIG, r_ev * M_REV, r_ev)
        onset = sp & (U_ONSET[:, t] < 1 - np.exp(-h_on))
        revert = sp & ~onset & (U_REV[:, t] < 1 - np.exp(-r_ev))
        relapse = sn & (U_RELAPSE[:, t] < 1 - np.exp(-RELAPSE))
        state[onset] = 2; diab_life[onset] = True
        if t < 10:
            diab10[onset] = True
        state[revert] = 0; state[relapse] = 1
        dur[state == 2] += 1
    return dict(**acc, diab10=diab10, diab_life=diab_life)


def wm(x):
    return float(np.sum(W * x))


def cq(r, C_COMP, U_COMP):
    cost = (C_NORMO * r["py_n"] + C_PREDIAB * r["py_p"] + C_DIAB_TX * r["py_d"]
            + C_COMP * r["dur_d"] + r["vitd"])
    qaly = (U_NORMO * r["py_n"] + U_PREDIAB * r["py_p"] + U_DIAB_BASE * r["py_d"]
            - U_COMP * r["dur_d"])
    cost10 = (C_NORMO * r["py_n10"] + C_PREDIAB * r["py_p10"] + C_DIAB_TX * r["py_d10"]
              + C_COMP * r["dur_d10"] + r["vitd10"])
    return cost, qaly, cost10


def calibrate_usa():
    """PREDIAB_RR -> 10-yr incidence; HET -> lifetime incidence; C_COMP,U_COMP -> totals."""
    het, rr = 1.0, 5.0
    for _ in range(3):
        lo, hi = 0.2, 8.0                      # HET -> lifetime incidence
        for _ in range(18):
            hm = 0.5 * (lo + hi)
            # refit rr to 10-yr inside
            a, b = 0.5, 60.0
            for _ in range(18):
                rm = 0.5 * (a + b)
                i10 = wm(simulate("control", "USA", rm, hm)["diab10"].astype(float))
                a, b = (rm, b) if i10 < TGT_INC10 else (a, rm)
            rrm = 0.5 * (a + b)
            il = wm(simulate("control", "USA", rrm, hm)["diab_life"].astype(float))
            lo, hi = (hm, hi) if il > TGT_INC else (lo, hm)
        het = 0.5 * (lo + hi)
        a, b = 0.5, 60.0
        for _ in range(20):
            rm = 0.5 * (a + b)
            i10 = wm(simulate("control", "USA", rm, het)["diab10"].astype(float))
            a, b = (rm, b) if i10 < TGT_INC10 else (a, rm)
        rr = 0.5 * (a + b)
    return dict(prediab_rr=rr, het=het, C_COMP=C_COMP, U_COMP=U_COMP)


def run_location(loc, p):
    ctrl = simulate("control", loc, p["prediab_rr"], p["het"])
    vitd = simulate("vitd", loc, p["prediab_rr"], p["het"])
    cc, cqa, cc10 = cq(ctrl, p["C_COMP"], p["U_COMP"])
    vc, vqa, vc10 = cq(vitd, p["C_COMP"], p["U_COMP"])
    dC = wm(vc) - wm(cc); dQ = wm(vqa) - wm(cqa)
    return dict(loc=loc, inc_c=100 * wm(ctrl["diab_life"].astype(float)),
                inc_v=100 * wm(vitd["diab_life"].astype(float)),
                ly=wm(ctrl["ly"]), dC=dC, dQ=dQ,
                icer=dC / dQ if dQ else np.nan, nmb=dQ * WTP - dC)


def main():
    p = calibrate_usa()
    print("==== USA calibration (onset age-pattern from GBD incidence) ====")
    print(f"  prediabetes progression RR (vs gen-pop incidence): {p['prediab_rr']:.1f}x")
    print(f"  marker-heterogeneity scale HET: {p['het']:.2f}")
    print(f"  complication cost ${p['C_COMP']:.0f}/yr.dur ; disutility {p['U_COMP']:.4f}/yr.dur\n")
    rows = [run_location(loc, p) for loc in ["USA", "HighBurden"]]
    print(f"  {'location':12s} {'incid C/100':>11s} {'incid V/100':>11s} {'red.%':>6s} "
          f"{'rem.LY':>7s} {'dCost':>8s} {'dQALY':>6s} {'ICER':>9s} {'NMB':>8s}")
    for r in rows:
        red = 100 * (r["inc_v"] - r["inc_c"]) / r["inc_c"]
        print(f"  {r['loc']:12s} {r['inc_c']:11.2f} {r['inc_v']:11.2f} {red:6.1f} "
              f"{r['ly']:7.1f} {r['dC']:8,.0f} {r['dQ']:6.3f} {r['icer']:9,.0f} {r['nmb']:8,.0f}")
    print("\n  (USA is calibrated to the paper; HighBurden swaps only the GBD epidemiology curve,")
    print("   holding the intervention effect, progression RR, heterogeneity, and US costs fixed.)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "gbd_rebase_results.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUT, 'gbd_rebase_results.csv')}")


if __name__ == "__main__":
    main()
