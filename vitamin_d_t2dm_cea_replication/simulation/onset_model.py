"""
Prevention front-end: a reduced-form microsimulation of vitamin D for preventing
type 2 diabetes in the reconstructed NHANES prediabetes cohort.

WHAT THIS IS. The public CDC/RTI engine ships the *diabetes-complications* module but
NOT the *prediabetes->diabetes onset* front-end that the vitamin-D effect acts on. This
script builds that front-end from scratch as a transparent 3-state annual-cycle
microsimulation (prediabetes -> diabetes -> dead), parameterised from Briody 2026
Table 2 and CALIBRATED to Briody Table 3, so we can test whether the paper's headline
result (vitamin D is cost-saving) reproduces mechanistically.

WHAT THIS IS NOT. It is not a bit-for-bit rerun of the proprietary 17-complication
CDC/RTI engine. Complications are represented in AGGREGATE: a diabetes-state cost and
disutility that ramp with diabetes duration (calibrated so the control arm reproduces
the paper's lifetime cost and QALY). That captures "complications accumulate the longer
you have diabetes, so delaying onset saves money and quality of life" -- the exact
mechanism behind the paper's finding -- without the individual complication equations.

MODEL AT A GLANCE
- Each person starts prediabetic at their NHANES age with their real BMI/sex.
- Each year: (a) accrue cost + QALY for the current state; (b) prediabetics may develop
  diabetes with annual hazard H_ONSET; (c) everyone faces background mortality, with
  diabetics carrying an excess-mortality hazard ratio; survivors age one year.
- Vitamin-D arm: for people eligible at baseline (BMI >= 25), the onset hazard is
  multiplied by (1 - adherence x RRR) = (1 - 0.85 x 0.15); intervention costs $60/yr
  while still prediabetic. Common random numbers are shared across arms so the
  incremental estimate is low-variance.
- Costs and QALYs discounted 3%/yr. Reported per person, survey-weighted to be
  nationally representative. Horizons: lifetime and 10 years.

CALIBRATION (5 targets, all from the Table 3 CONTROL arm -- descriptive, not the result)
  1. background-mortality scale       -> remaining life-years 36.45
  2. susceptible fraction f           -> lifetime cumulative incidence 32.35/100
  3. onset hazard H_ONSET             -> 10-year cumulative incidence 20.14/100
  4. diabetes complication cost ramp  -> lifetime discounted cost $163,210
  5. diabetes complication disutility -> lifetime discounted QALYs 16.35
Diabetes excess-mortality HR is fixed at a literature value (1.8) and reported. The
intervention increments (cost, QALY, life-years, ICER, NMB) are then EMERGENT outputs
compared against the paper as the test of replication -- they are not calibrated to.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT = os.path.join(HERE, "..", "nhanes_cohort", "outputs", "cohort_canonical_n4176.csv")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- fixed parameters (from Briody 2026 Table 2 / Methods) -------------------
DISCOUNT = 0.03
V = 1.0 / (1.0 + DISCOUNT)
RRR = 0.15                 # relative risk reduction on onset (Pittas 2023)
ADHERENCE = 0.85
C_PREDIAB = 3244.0         # prediabetes base annual cost
C_DIAB_TX = 11322.0        # type 2 diabetes base annual (treatment) cost
C_VITD = 60.0              # annual vitamin D3 supplement cost
U_PREDIAB = 0.920          # prediabetes utility (0.935 baseline minus small decrements)
U_DIAB_BASE = 0.935        # diabetes baseline utility before complication accrual
WTP = 100_000.0
DM_MORT_HR = 1.8           # diabetes excess-mortality hazard ratio (literature central value)
MAX_AGE = 100
SEED = 2026

# calibration targets (Table 3, control / no-intervention, lifetime unless noted)
TGT_LY = 36.45
TGT_INC10 = 0.2014         # 10-year cumulative incidence (per person)
TGT_INC = 0.3235          # lifetime cumulative incidence (per person)
TGT_COST = 163_210.0
TGT_QALY = 16.35

# ---- load cohort ------------------------------------------------------------
c = pd.read_csv(COHORT)
AGE0 = c["age"].to_numpy(float)
FEMALE = c["female"].to_numpy(float)
ELIG = (c["bmi"].to_numpy(float) >= 25)          # baseline supplementation eligibility
W = c["weight"].to_numpy(float)
W = W / W.sum()                                   # normalise survey weights
N = len(c)
T = MAX_AGE - int(AGE0.min()) + 1                 # number of annual cycles to run

# shared random draws (common random numbers across arms)
_rng = np.random.default_rng(SEED)
U_ONSET = _rng.random((N, T))
U_MORT = _rng.random((N, T))
U_SUSC = _rng.random(N)   # latent susceptibility draw; susceptible if U_SUSC < f


def q_bg(age, mort_scale):
    """Gompertz background one-year mortality; women scaled down. `age` is an array."""
    mu = mort_scale * 2.7e-5 * np.exp(0.095 * age)
    mu = mu * np.where(FEMALE == 1, 0.55, 1.0)
    return 1.0 - np.exp(-mu)


def simulate(arm, f, H_ONSET, mort_scale):
    """Run one arm. Returns per-person accumulators (discounted unless noted).

    `f` = fraction of the cohort that is ever-susceptible to progression. Only
    susceptible prediabetics face the onset hazard H_ONSET; the rest never develop
    diabetes (they keep incurring prediabetes cost, and vitamin-D cost if in that arm).
    This "long-term survivor" structure reproduces the paper's decelerating cumulative
    incidence (20% by 10y, 32% lifetime) that a single constant hazard cannot.
    """
    susceptible = U_SUSC < f
    state = np.zeros(N, int)      # 0 prediabetes, 1 diabetes, 2 dead
    dur = np.zeros(N)             # years lived with diabetes so far
    # discounted person-years by state, and discounted (duration x person-year) for ramps
    py_pre = np.zeros(N); py_dm = np.zeros(N); dur_dm = np.zeros(N)
    py_pre10 = np.zeros(N); py_dm10 = np.zeros(N); dur_dm10 = np.zeros(N)
    vitd = np.zeros(N); vitd10 = np.zeros(N)
    ly = np.zeros(N); ly10 = np.zeros(N)
    diab10 = np.zeros(N, bool); diab_life = np.zeros(N, bool)

    for t in range(T):
        age = AGE0 + t
        disc = V ** t
        alive = (state != 2) & (age <= MAX_AGE)
        pre = alive & (state == 0)
        dm = alive & (state == 1)

        # ---- accrue for the year (based on start-of-year state) ----
        py_pre[pre] += disc; py_dm[dm] += disc; dur_dm[dm] += disc * dur[dm]
        ly[alive] += 1.0
        if arm == "vitd":
            vitd[pre & ELIG] += disc * C_VITD
        if t < 10:
            py_pre10[pre] += disc; py_dm10[dm] += disc; dur_dm10[dm] += disc * dur[dm]
            ly10[alive] += 1.0
            if arm == "vitd":
                vitd10[pre & ELIG] += disc * C_VITD

        # ---- prediabetes -> diabetes onset (susceptibles only) ----
        h = np.where(susceptible, H_ONSET, 0.0)
        if arm == "vitd":
            h = np.where(ELIG, h * (1.0 - ADHERENCE * RRR), h)
        p_onset = 1.0 - np.exp(-h)
        onset = pre & (U_ONSET[:, t] < p_onset)
        state[onset] = 1
        diab_life[onset] = True
        if t < 10:
            diab10[onset] = True

        # ---- mortality (diabetics carry excess hazard) ----
        q = q_bg(age, mort_scale)
        q_eff = q.copy()
        is_dm = (state == 1)
        q_eff[is_dm] = 1.0 - (1.0 - q[is_dm]) ** DM_MORT_HR
        died = alive & (U_MORT[:, t] < q_eff)
        # force death at MAX_AGE
        died = died | (alive & (age >= MAX_AGE))
        state[died] = 2

        # survivors who have diabetes age one diabetes-year
        dur[(state == 1)] += 1.0

    return dict(py_pre=py_pre, py_dm=py_dm, dur_dm=dur_dm,
                py_pre10=py_pre10, py_dm10=py_dm10, dur_dm10=dur_dm10,
                vitd=vitd, vitd10=vitd10, ly=ly, ly10=ly10,
                diab10=diab10, diab_life=diab_life)


def wmean(x):
    return float(np.sum(W * x))


def costs_qalys(r, C_COMP, U_COMP):
    """Per-person cost & QALY from accumulators, given calibrated ramp coefficients."""
    cost = C_PREDIAB * r["py_pre"] + C_DIAB_TX * r["py_dm"] + C_COMP * r["dur_dm"] + r["vitd"]
    qaly = U_PREDIAB * r["py_pre"] + U_DIAB_BASE * r["py_dm"] - U_COMP * r["dur_dm"]
    cost10 = (C_PREDIAB * r["py_pre10"] + C_DIAB_TX * r["py_dm10"]
              + C_COMP * r["dur_dm10"] + r["vitd10"])
    qaly10 = U_PREDIAB * r["py_pre10"] + U_DIAB_BASE * r["py_dm10"] - U_COMP * r["dur_dm10"]
    return cost, qaly, cost10, qaly10


def calibrate():
    """Tune mort_scale (life-years), and (f, H_ONSET) jointly to the 10-year AND
    lifetime cumulative incidence; then linear-solve (C_COMP, U_COMP)."""
    mort_scale, f, H = 1.0, 0.36, 0.08
    for _ in range(4):
        # (a) mort_scale -> remaining life-years 36.45
        lo, hi = 0.2, 5.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            ly = wmean(simulate("control", f, H, mid)["ly"])
            lo, hi = (mid, hi) if ly > TGT_LY else (lo, mid)
        mort_scale = 0.5 * (lo + hi)
        # (b) f -> lifetime incidence 32.35/100 (outer)
        flo, fhi = 0.05, 0.95
        for _ in range(16):
            fmid = 0.5 * (flo + fhi)
            # (c) H_ONSET -> 10-year incidence 20.14/100 given this f (inner)
            hlo, hhi = 0.001, 3.0
            for _ in range(22):
                hmid = 0.5 * (hlo + hhi)
                inc10 = wmean(simulate("control", fmid, hmid, mort_scale)["diab10"].astype(float))
                hlo, hhi = (hmid, hhi) if inc10 < TGT_INC10 else (hlo, hmid)
            Hc = 0.5 * (hlo + hhi)
            inc_life = wmean(simulate("control", fmid, Hc, mort_scale)["diab_life"].astype(float))
            flo, fhi = (fmid, fhi) if inc_life < TGT_INC else (flo, fmid)
        f = 0.5 * (flo + fhi)
        H = Hc
    # (d,e) linear solve for complication cost & disutility ramps from control arm
    r = simulate("control", f, H, mort_scale)
    fixed_cost = C_PREDIAB * wmean(r["py_pre"]) + C_DIAB_TX * wmean(r["py_dm"])
    C_COMP = (TGT_COST - fixed_cost) / wmean(r["dur_dm"])
    fixed_qaly = U_PREDIAB * wmean(r["py_pre"]) + U_DIAB_BASE * wmean(r["py_dm"])
    U_COMP = (fixed_qaly - TGT_QALY) / wmean(r["dur_dm"])
    return dict(f=f, H_ONSET=H, mort_scale=mort_scale, C_COMP=C_COMP, U_COMP=U_COMP)


def main():
    p = calibrate()
    print("==== Calibrated parameters ====")
    print(f"  susceptible fraction f     : {p['f']:.3f} (rest never progress)")
    print(f"  onset hazard H_ONSET       : {p['H_ONSET']:.4f} /yr (among susceptibles)")
    print(f"  background mortality scale : {p['mort_scale']:.3f}")
    print(f"  diabetes complication cost : ${p['C_COMP']:.0f} /yr per year of diabetes duration")
    print(f"  diabetes complication disutility: {p['U_COMP']:.4f} /yr per year of diabetes duration")
    print(f"  (fixed) diabetes excess-mortality HR: {DM_MORT_HR}")

    ctrl = simulate("control", p["f"], p["H_ONSET"], p["mort_scale"])
    vitd = simulate("vitd", p["f"], p["H_ONSET"], p["mort_scale"])
    cc, cq, cc10, cq10 = costs_qalys(ctrl, p["C_COMP"], p["U_COMP"])
    vc, vq, vc10, vq10 = costs_qalys(vitd, p["C_COMP"], p["U_COMP"])

    def summ(cost_c, q_c, cost_v, q_v, ly_c, ly_v, inc_c, inc_v):
        dC = wmean(cost_v) - wmean(cost_c)
        dQ = wmean(q_v) - wmean(q_c)
        dLY = wmean(ly_v) - wmean(ly_c)
        icer = dC / dQ if dQ != 0 else np.nan
        nmb = dQ * WTP - dC
        return dict(cost_c=wmean(cost_c), cost_v=wmean(cost_v), dC=dC,
                    q_c=wmean(q_c), q_v=wmean(q_v), dQ=dQ, dLY=dLY,
                    inc_c=100 * wmean(inc_c), inc_v=100 * wmean(inc_v),
                    icer=icer, nmb=nmb)

    life = summ(cc, cq, vc, vq, ctrl["ly"], vitd["ly"], ctrl["diab_life"], vitd["diab_life"])
    ten = summ(cc10, cq10, vc10, vq10, ctrl["ly10"], vitd["ly10"], ctrl["diab10"], vitd["diab10"])

    paper_life = dict(dC=-3208, dQ=0.12, dLY=0.27, icer=-26134, nmb=15483,
                      inc_c=32.35, inc_v=29.75, cost_c=163210, q_c=16.35)
    paper_10 = dict(dC=-401, dQ=0.01, dLY=0.001, icer=-61122, nmb=1058,
                    inc_c=20.14, inc_v=18.15, cost_c=50709, q_c=6.90)

    def show(name, m, pap):
        print(f"\n==== {name} ====")
        print(f"  {'metric':32s} {'model':>14s} {'paper':>14s}")
        rows = [
            ("Diabetes incidence, control /100", f"{m['inc_c']:.2f}", f"{pap['inc_c']:.2f}"),
            ("Diabetes incidence, vitD /100", f"{m['inc_v']:.2f}", f"{pap['inc_v']:.2f}"),
            ("Total cost, control $", f"{m['cost_c']:,.0f}", f"{pap['cost_c']:,.0f}"),
            ("Total QALY, control", f"{m['q_c']:.2f}", f"{pap['q_c']:.2f}"),
            ("Incremental cost $", f"{m['dC']:,.0f}", f"{pap['dC']:,.0f}"),
            ("Incremental QALY", f"{m['dQ']:.3f}", f"{pap['dQ']:.3f}"),
            ("Incremental life-years", f"{m['dLY']:.3f}", f"{pap['dLY']:.3f}"),
            ("ICER $/QALY", f"{m['icer']:,.0f}", f"{pap['icer']:,.0f}"),
            ("NMB @ $100k/QALY $", f"{m['nmb']:,.0f}", f"{pap['nmb']:,.0f}"),
        ]
        for lbl, a, b in rows:
            print(f"  {lbl:32s} {a:>14s} {b:>14s}")

    show("LIFETIME horizon", life, paper_life)
    show("10-YEAR horizon", ten, paper_10)

    out = pd.DataFrame([
        {"horizon": "lifetime", **life},
        {"horizon": "10-year", **ten},
    ])
    out.to_csv(os.path.join(OUT, "cea_results.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUT, 'cea_results.csv')}")


if __name__ == "__main__":
    main()
