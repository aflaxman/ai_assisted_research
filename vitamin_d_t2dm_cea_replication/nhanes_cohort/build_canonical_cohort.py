"""
Build the canonical N=4176 prediabetes cohort to carry forward to the simulation.

Decision (see README "Locking N=4176"): a grid search proved no rule-based public-data
definition lands exactly on the paper's N=4176 -- cohort sizes cluster at ~3800-3990
(HbA1c-driven) or jump to ~5150-5930 (once the fasting-FPG criterion is added), with a
dead zone in between. The paper's stated definition (HbA1c 5.7-6.4 OR FPG 100-125 OR
OGTT 140-199 on the full exam sample) is the closest match to its risk-factor PROFILE
and age; it yields N=5603 here. To move forward at the paper's size, we take that
literal-definition cohort and draw exactly 4176 individuals with probability
proportional to the MEC survey weight (fixed seed), i.e. a survey-weighted subsample.
The draw is representative (selection prob = weight) so the weighted profile is preserved.

Output: outputs/cohort_canonical_n4176.csv  (simulation-ready; 4176 rows).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from reconstruct_cohort import egfr_ckdepi_2021, wmean, wsd, wprop
from find_definition import build

SEED = 2026
N_TARGET = 4176


def main():
    df = build()

    # extra derived vars the canonical file should carry (build() has the core set)
    df["race_nhwhite"] = (df["RIDRETH3"] == 3).astype(float)
    df["race_nhblack"] = (df["RIDRETH3"] == 4).astype(float)
    df["race_hispanic"] = df["RIDRETH3"].isin([1, 2]).astype(float)
    df["race_other"] = df["RIDRETH3"].isin([6, 7]).astype(float)
    df["postsec_ed"] = df["DMDEDUC2"].isin([4, 5]).astype(float)
    df["current_smoker"] = ((df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))).astype(float)
    df["hdl_mmol"] = df["LBDHDD"] / 38.67
    df["ldl_mmol"] = df["LBDLDL"] / 38.67
    df["trig_mmol"] = df["LBXTR"] / 88.57
    df["scr_umol"] = df["LBXSCR"] * 88.42
    df["vitd_nmol"] = df["LBXVIDMS"]
    df["acr"] = df["URDACT"]
    df["egfr"] = egfr_ckdepi_2021(df["scr"], df["age"], df["female"])
    df["ckd_3_5"] = (df["egfr"] < 60).astype(float)
    df["ckd_4_5"] = (df["egfr"] < 30).astype(float)
    df["microalb"] = ((df["acr"] >= 30) & (df["acr"] <= 300)).astype(float)
    df["macroalb"] = (df["acr"] > 300).astype(float)
    df["mi"] = (df["MCQ160E"] == 1).astype(float)
    df["stroke"] = (df["MCQ160F"] == 1).astype(float)
    df["chf"] = (df["MCQ160B"] == 1).astype(float)
    df["angina"] = (df["MCQ160D"] == 1).astype(float)
    df["dialysis"] = (df["KIQ025"] == 1).astype(float)
    df["famhx_dm"] = (df["MCQ300C"] == 1).astype(float)

    # ---- literal-definition cohort (paper's stated criteria, full MEC) -------
    adult = df["age"] >= 18
    hb_pre = (df["hba1c"] >= 5.7) & (df["hba1c"] <= 6.4)
    fpg_pre = (df["fpg"] >= 100) & (df["fpg"] <= 125)
    ogtt_pre = (df["ogtt"] >= 140) & (df["ogtt"] <= 199)
    lab_diab = (df["DIQ010"] == 1) | (df["hba1c"] >= 6.5) | (df["fpg"] >= 126) | (df["ogtt"] >= 200)
    union = df[adult & ~lab_diab & (hb_pre | fpg_pre | ogtt_pre)].copy()
    print(f"Literal-definition MEC-union cohort: N = {len(union)} (weighted age "
          f"{wmean(union['age'], union['wtmec6']):.1f})")

    # ---- uniform draw to exactly N=4176, keeping original survey weights -----
    # Uniform (not weight-proportional) so that carrying each person's original
    # WTMEC2YR/3 weight keeps the subsample an unbiased representation of the
    # same population -- the weighted profile is preserved (just noisier).
    rng = np.random.default_rng(SEED)
    idx = rng.choice(union.index.to_numpy(), size=N_TARGET, replace=False)
    canon = union.loc[idx].copy()
    canon["weight"] = canon["wtmec6"]

    print(f"Canonical cohort: N = {len(canon)}  (seed={SEED}, uniform draw, original weights kept)")

    # ---- verify the profile is preserved ------------------------------------
    def wm(c, col, wt="weight"):
        return f"{wmean(c[col], c[wt]):.1f} ({wsd(c[col], c[wt]):.1f})"

    def wp(c, col, wt="weight"):
        return f"{100*wprop(c[col], c[wt]):.1f}%"

    checks = [
        ("N", str(len(canon)), "4176"),
        ("Age, years", wm(canon, "age"), "53.3 (17.0)"),
        ("Female", wp(canon, "female"), "49%"),
        ("NH White", wp(canon, "race_nhwhite"), "62%"),
        ("NH Black", wp(canon, "race_nhblack"), "12%"),
        ("Hispanic", wp(canon, "race_hispanic"), "16%"),
        ("BMI, kg/m2", wm(canon, "bmi"), "30.3 (7.3)"),
        ("HbA1c, %", wm(canon, "hba1c"), "5.7 (0.5)"),
        ("FPG, mmol/L", wm(canon, "fpg_mmol"), "5.9 (0.8)"),
        ("Systolic BP", wm(canon, "sbp"), "128 (19)"),
        ("Microalbuminuria", wp(canon, "microalb"), "8.2%"),
        ("Current smoker", wp(canon, "current_smoker"), "18%"),
    ]
    print("\n==== Canonical cohort profile vs. paper ====")
    print(f"  {'Variable':20s} {'Canonical (weighted)':22s} {'Paper':12s}")
    for lbl, val, pap in checks:
        print(f"  {lbl:20s} {val:22s} {pap:12s}")

    cols = ["SEQN", "age", "female", "race_nhwhite", "race_nhblack", "race_hispanic",
            "race_other", "postsec_ed", "current_smoker", "hba1c", "fpg", "fpg_mmol",
            "ogtt", "bmi", "sbp", "hdl_mmol", "ldl_mmol", "trig_mmol", "scr", "egfr",
            "vitd_nmol", "acr", "microalb", "macroalb", "ckd_3_5", "ckd_4_5",
            "dialysis", "mi", "stroke", "chf", "angina", "famhx_dm",
            "WTMEC2YR", "weight", "SDMVPSU", "SDMVSTRA"]
    canon[cols].to_csv("outputs/cohort_canonical_n4176.csv", index=False)
    print(f"\nSaved: outputs/cohort_canonical_n4176.csv ({len(canon)} rows)")


if __name__ == "__main__":
    main()
