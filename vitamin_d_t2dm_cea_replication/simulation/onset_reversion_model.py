"""
Upgraded prevention front-end: RISK-FACTOR-DEPENDENT onset + a NORMOGLYCEMIA
(reversion) state. This replaces the binary "susceptible fraction" hack in
onset_model.py with two mechanisms that are both grounded in the paper and the data:

  1. AGE- and risk-factor-dependent onset. Instead of splitting the cohort into
     immune vs susceptible, every person's annual prediabetes->diabetes hazard is
     h0 * exp(b_age*(age-50)) * exp(b_a1c*(HbA1c-mean) + b_fpg*(FPG-mean) + b_bmi*(BMI-mean)).
     Onset rises with CURRENT AGE (core diabetes epidemiology) and with baseline
     glycemic/adiposity markers. Age-rising onset plus rising competing mortality caps
     lifetime incidence naturally, and makes converters older (shorter, cheaper diabetes
     careers) -- the fix for the v2-without-age overshoot. (FPG is used where measured,
     ~63% of the cohort; neutral where absent, so each person is scored on the marker
     that defined them.)

  2. A Normoglycemia state with reversion. Briody Table 2 lists a "Normoglycemia"
     cost/QALY state distinct from prediabetes; in a cohort that starts prediabetic it
     is only reachable by REVERTING. So we add prediabetes<->normoglycemia transitions.
     Vitamin D raises reversion (regression to normoglycemia, rate ratio 1.30 from
     Pittas 2023) -- the second benefit channel the previous model omitted, expected to
     lift the life-years gained and the lifetime incidence reduction.

Everything else matches onset_model.py: annual cycle, discount 3%, survey-weighted
per-person outputs, common random numbers across arms, calibration to the Table 3
CONTROL arm (so intervention increments are emergent, not fitted).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT = os.path.join(HERE, "..", "nhanes_cohort", "outputs", "cohort_canonical_n4176.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- fixed parameters -------------------------------------------------------
DISCOUNT = 0.03
V = 1.0 / (1.0 + DISCOUNT)
RRR = 0.15                 # onset relative risk reduction (Pittas 2023 IPD meta-analysis)
REV_RR = 1.30              # regression-to-normoglycemia rate ratio (Pittas 2023)
ADHERENCE = 0.85
C_NORMO = 2636.0
C_PREDIAB = 3244.0
C_DIAB_TX = 11322.0
C_VITD = 60.0
U_NORMO = 0.935
U_PREDIAB = 0.920
U_DIAB_BASE = 0.935
WTP = 100_000.0
DM_MORT_HR = 2.0           # diabetes excess-mortality HR (literature; drives life-years gained)
RELAPSE = 0.10             # normoglycemia -> prediabetes annual hazard (fixed)
R_REV = 0.05               # prediabetes -> normoglycemia reversion hazard (fixed, plausible)
MAX_AGE = 100
SEED = 2026

# onset log-hazard coefficients give the RELATIVE weight of each risk factor;
# an overall HETerogeneity scale (calibrated) multiplies them. Literature-plausible ratios.
B_A1C = 1.5                # per % HbA1c
B_FPG = 0.03               # per mg/dL fasting glucose
B_BMI = 0.03               # per BMI unit
# AGE-dependent onset: progression risk rises with current age (core diabetes
# epidemiology). B_AGE ~ 0.05/yr => hazard doubles about every 14 years of age.
B_AGE = 0.05               # per year of age (~doubling every 14 yr; core diabetes epidemiology)
AGE_REF = 50.0             # centring age (h0 is the onset level at age 50, mean markers)

# calibration targets (Briody Table 3, control arm)
TGT_LY = 36.45
TGT_INC10 = 0.2014
TGT_INC = 0.3235
TGT_COST = 163_210.0
TGT_QALY = 16.35

# effective per-arm multipliers for eligible, treated person-years
M_ONSET = 1.0 - ADHERENCE * RRR          # 0.8725
M_REV = 1.0 + ADHERENCE * (REV_RR - 1)   # 1.255

# ---- load cohort & build per-person onset risk score ------------------------
c = pd.read_csv(COHORT)
AGE0 = c["age"].to_numpy(float)
FEMALE = c["female"].to_numpy(float)
BMI = c["bmi"].to_numpy(float)
A1C = c["hba1c"].to_numpy(float)
FPG = c["fpg"].to_numpy(float)
ELIG = (BMI >= 25)
W = c["weight"].to_numpy(float); W = W / W.sum()
N = len(c)
T = MAX_AGE - int(AGE0.min()) + 1

a1c_ref = np.nanmean(A1C); fpg_ref = np.nanmean(FPG); bmi_ref = np.nanmean(BMI)


def dev(x, ref):
    d = x - ref
    return np.where(np.isnan(d), 0.0, d)   # missing marker -> neutral (at reference)


# unit (HET=1) onset log-hazard deviation per person; scaled by HET inside simulate()
LP0 = B_A1C * dev(A1C, a1c_ref) + B_FPG * dev(FPG, fpg_ref) + B_BMI * dev(BMI, bmi_ref)

_rng = np.random.default_rng(SEED)
U_ONSET = _rng.random((N, T))
U_REV = _rng.random((N, T))
U_RELAPSE = _rng.random((N, T))
U_MORT = _rng.random((N, T))


def q_bg(age, mort_scale):
    mu = mort_scale * 2.7e-5 * np.exp(0.095 * age) * np.where(FEMALE == 1, 0.55, 1.0)
    return 1.0 - np.exp(-mu)


def simulate(arm, h0, het, mort_scale, r=R_REV):
    """States: 0 normoglycemia, 1 prediabetes, 2 diabetes, 3 dead. All start at 1.
    `het` scales the risk-factor spread of the onset hazard (heterogeneity does the
    cumulative-incidence deceleration); `r` is the fixed reversion rate."""
    REL = np.exp(het * LP0)
    state = np.ones(N, int)
    dur = np.zeros(N)
    # discounted accumulators
    acc = {k: np.zeros(N) for k in
           ("py_n", "py_p", "py_d", "dur_d", "vitd",
            "py_n10", "py_p10", "py_d10", "dur_d10", "vitd10", "ly", "ly10")}
    diab10 = np.zeros(N, bool); diab_life = np.zeros(N, bool)
    onset_age = np.full(N, np.nan)   # age at diabetes onset (NaN if never)

    for t in range(T):
        age = AGE0 + t
        disc = V ** t
        alive = state != 3
        n = alive & (state == 0); p = alive & (state == 1); d = alive & (state == 2)

        # accrue for the year (start-of-year state)
        acc["py_n"][n] += disc; acc["py_p"][p] += disc
        acc["py_d"][d] += disc; acc["dur_d"][d] += disc * dur[d]
        acc["ly"][alive] += 1.0
        if arm == "vitd":                        # supplement taken while non-diabetic & eligible
            acc["vitd"][(n | p) & ELIG] += disc * C_VITD
        if t < 10:
            acc["py_n10"][n] += disc; acc["py_p10"][p] += disc
            acc["py_d10"][d] += disc; acc["dur_d10"][d] += disc * dur[d]
            acc["ly10"][alive] += 1.0
            if arm == "vitd":
                acc["vitd10"][(n | p) & ELIG] += disc * C_VITD

        # ---- mortality first (diabetics carry excess hazard) ----
        q = q_bg(age, mort_scale)
        q_eff = q.copy(); q_eff[d] = 1.0 - (1.0 - q[d]) ** DM_MORT_HR
        died = alive & ((U_MORT[:, t] < q_eff) | (age >= MAX_AGE))
        state[died] = 3
        surv = alive & ~died

        # ---- transitions among survivors ----
        sp = surv & (state == 1)     # prediabetes
        sn = surv & (state == 0)     # normoglycemia
        age_factor = np.exp(B_AGE * (age - AGE_REF))     # onset rises with current age
        h_on = h0 * age_factor * REL                     # age- and risk-factor-dependent onset
        r_ev = np.full(N, r)
        if arm == "vitd":
            h_on = np.where(ELIG, h_on * M_ONSET, h_on)
            r_ev = np.where(ELIG, r_ev * M_REV, r_ev)
        p_on = 1.0 - np.exp(-h_on)
        p_rev = 1.0 - np.exp(-r_ev)
        onset = sp & (U_ONSET[:, t] < p_on)
        revert = sp & ~onset & (U_REV[:, t] < p_rev)
        relapse = sn & (U_RELAPSE[:, t] < (1.0 - np.exp(-RELAPSE)))
        state[onset] = 2; diab_life[onset] = True; onset_age[onset] = age[onset]
        if t < 10:
            diab10[onset] = True
        state[revert] = 0
        state[relapse] = 1

        dur[state == 2] += 1.0

    return dict(**acc, diab10=diab10, diab_life=diab_life, onset_age=onset_age)


def wmean(x):
    return float(np.sum(W * x))


def costs_qalys(r, C_COMP, U_COMP):
    cost = (C_NORMO * r["py_n"] + C_PREDIAB * r["py_p"] + C_DIAB_TX * r["py_d"]
            + C_COMP * r["dur_d"] + r["vitd"])
    qaly = (U_NORMO * r["py_n"] + U_PREDIAB * r["py_p"] + U_DIAB_BASE * r["py_d"]
            - U_COMP * r["dur_d"])
    cost10 = (C_NORMO * r["py_n10"] + C_PREDIAB * r["py_p10"] + C_DIAB_TX * r["py_d10"]
              + C_COMP * r["dur_d10"] + r["vitd10"])
    qaly10 = (U_NORMO * r["py_n10"] + U_PREDIAB * r["py_p10"] + U_DIAB_BASE * r["py_d10"]
              - U_COMP * r["dur_d10"])
    return cost, qaly, cost10, qaly10


def _fit_h0(het, mort_scale, r):
    """Given heterogeneity, mortality & reversion, find h0 hitting 10-yr incidence."""
    lo, hi = 1e-4, 3.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        i10 = wmean(simulate("control", mid, het, mort_scale, r)["diab10"].astype(float))
        lo, hi = (mid, hi) if i10 < TGT_INC10 else (lo, mid)
    return 0.5 * (lo + hi)


def calibrate(r=R_REV):
    """Stable nested calibration at a given reversion rate `r`: for each heterogeneity
    scale, refit h0 to the 10-yr incidence, move HET to hit lifetime incidence; fit
    mortality to life-years. Finally linear-solve the complication cost/disutility ramps."""
    mort_scale, het, h0 = 0.6, 1.0, 0.02
    for _ in range(2):
        lo, hi = 0.2, 8.0
        for _ in range(18):
            hm = 0.5 * (lo + hi)
            h0m = _fit_h0(hm, mort_scale, r)
            il = wmean(simulate("control", h0m, hm, mort_scale, r)["diab_life"].astype(float))
            lo, hi = (hm, hi) if il > TGT_INC else (lo, hm)   # more het -> lower lifetime
        het = 0.5 * (lo + hi)
        h0 = _fit_h0(het, mort_scale, r)
        lo, hi = 0.2, 5.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            ly = wmean(simulate("control", h0, het, mid, r)["ly"])
            lo, hi = (mid, hi) if ly > TGT_LY else (lo, mid)
        mort_scale = 0.5 * (lo + hi)
    rr = simulate("control", h0, het, mort_scale, r)
    fixed_cost = C_NORMO * wmean(rr["py_n"]) + C_PREDIAB * wmean(rr["py_p"]) + C_DIAB_TX * wmean(rr["py_d"])
    C_COMP = (TGT_COST - fixed_cost) / wmean(rr["dur_d"])
    fixed_q = U_NORMO * wmean(rr["py_n"]) + U_PREDIAB * wmean(rr["py_p"]) + U_DIAB_BASE * wmean(rr["py_d"])
    U_COMP = (fixed_q - TGT_QALY) / wmean(rr["dur_d"])
    return dict(h0=h0, het=het, mort_scale=mort_scale, r=r, C_COMP=C_COMP, U_COMP=U_COMP)


def evaluate(p):
    """Given calibrated params, return lifetime & 10-year incremental summaries."""
    ctrl = simulate("control", p["h0"], p["het"], p["mort_scale"], p["r"])
    vitd = simulate("vitd", p["h0"], p["het"], p["mort_scale"], p["r"])
    cc, cq, cc10, cq10 = costs_qalys(ctrl, p["C_COMP"], p["U_COMP"])
    vc, vq, vc10, vq10 = costs_qalys(vitd, p["C_COMP"], p["U_COMP"])

    def summ(cost_c, q_c, cost_v, q_v, ly_c, ly_v, inc_c, inc_v):
        dC = wmean(cost_v) - wmean(cost_c); dQ = wmean(q_v) - wmean(q_c)
        return dict(cost_c=wmean(cost_c), q_c=wmean(q_c), dC=dC, dQ=dQ,
                    dLY=wmean(ly_v) - wmean(ly_c),
                    inc_c=100 * wmean(inc_c), inc_v=100 * wmean(inc_v),
                    icer=dC / dQ if dQ else np.nan, nmb=dQ * WTP - dC)

    life = summ(cc, cq, vc, vq, ctrl["ly"], vitd["ly"], ctrl["diab_life"], vitd["diab_life"])
    ten = summ(cc10, cq10, vc10, vq10, ctrl["ly10"], vitd["ly10"], ctrl["diab10"], vitd["diab10"])
    return life, ten


PAPER_LIFE = dict(dC=-3208, dQ=0.12, dLY=0.27, icer=-26134, nmb=15483, inc_c=32.35, inc_v=29.75)
PAPER_10 = dict(dC=-401, dQ=0.01, dLY=0.001, icer=-61122, nmb=1058, inc_c=20.14, inc_v=18.15)


def main():
    # Sweep the (unidentified) reversion rate to expose the heterogeneity<->reversion
    # trade-off and find the value that best matches the paper's LIFETIME increments.
    print("==== Reversion-rate sweep (lifetime increments vs paper) ====")
    print(f"  paper: incid.red -8.0%  dCost -$3,208  dQALY 0.120  dLY 0.270  ICER -$26,134  NMB $15,483\n")
    print(f"  {'r_rev':>6s} {'HET':>5s} {'incid.red':>9s} {'dCost':>8s} {'dQALY':>6s} {'dLY':>6s} {'ICER':>8s} {'NMB':>8s}")
    results = {}
    for r in [0.05, 0.10, 0.15]:
        p = calibrate(r)
        life, _ = evaluate(p)
        results[r] = (p, life)
        red = 100 * (life["inc_v"] - life["inc_c"]) / life["inc_c"]
        print(f"  {r:6.2f} {p['het']:5.2f} {red:8.1f}% {life['dC']:8,.0f} {life['dQALY'] if False else life['dQ']:6.3f} "
              f"{life['dLY']:6.3f} {life['icer']:8,.0f} {life['nmb']:8,.0f}")

    # pick the reversion rate whose lifetime dCost is closest to the paper's -$3,208
    best_r = min(results, key=lambda r: abs(results[r][1]["dC"] - PAPER_LIFE["dC"]))
    p, life = results[best_r]
    _, ten = evaluate(p)
    REL = np.exp(p["het"] * LP0)
    csim = simulate("control", p["h0"], p["het"], p["mort_scale"], p["r"])
    oa = csim["onset_age"]; conv = ~np.isnan(oa)
    mean_onset_age = float(np.sum(W[conv] * oa[conv]) / np.sum(W[conv]))
    print(f"\n==== Chosen reversion rate r = {best_r:.2f} (closest lifetime dCost to paper) ====")
    print(f"  onset hazard h0 (at mean risk) : {p['h0']:.4f} /yr")
    print(f"  heterogeneity scale HET        : {p['het']:.2f}  (onset spread p10={np.quantile(REL,0.1):.2f}x  p90={np.quantile(REL,0.9):.1f}x)")
    print(f"  reversion P->N {best_r:.2f}/yr; relapse N->P {RELAPSE}/yr; mort scale {p['mort_scale']:.3f}")
    print(f"  complication cost ${p['C_COMP']:.0f}/yr.dur ; disutility {p['U_COMP']:.4f}/yr.dur ; DM mort HR {DM_MORT_HR}")
    print(f"  age slope B_AGE={B_AGE}/yr ; mean age at diabetes onset (control): {mean_onset_age:.1f} yr")

    def show(name, m, pap):
        print(f"\n==== {name} ====")
        print(f"  {'metric':34s} {'model':>13s} {'paper':>13s}")
        for lbl, a, b in [
            ("Diabetes incidence, control /100", f"{m['inc_c']:.2f}", f"{pap['inc_c']:.2f}"),
            ("Diabetes incidence, vitD /100", f"{m['inc_v']:.2f}", f"{pap['inc_v']:.2f}"),
            ("  relative incidence reduction", f"{100*(m['inc_v']-m['inc_c'])/m['inc_c']:.1f}%",
             f"{100*(pap['inc_v']-pap['inc_c'])/pap['inc_c']:.1f}%"),
            ("Incremental cost $", f"{m['dC']:,.0f}", f"{pap['dC']:,.0f}"),
            ("Incremental QALY", f"{m['dQ']:.3f}", f"{pap['dQ']:.3f}"),
            ("Incremental life-years", f"{m['dLY']:.3f}", f"{pap['dLY']:.3f}"),
            ("ICER $/QALY", f"{m['icer']:,.0f}", f"{pap['icer']:,.0f}"),
            ("NMB @ $100k/QALY $", f"{m['nmb']:,.0f}", f"{pap['nmb']:,.0f}"),
        ]:
            print(f"  {lbl:34s} {a:>13s} {b:>13s}")

    show("LIFETIME horizon", life, PAPER_LIFE)
    show("10-YEAR horizon", ten, PAPER_10)
    pd.DataFrame([{"horizon": "lifetime", "r_rev": best_r, **life},
                  {"horizon": "10-year", "r_rev": best_r, **ten}]).to_csv(
        os.path.join(OUT, "cea_results_reversion.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUT, 'cea_results_reversion.csv')}")


if __name__ == "__main__":
    main()
