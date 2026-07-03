"""
Reconstruct the Briody et al. 2026 NHANES 2013-2018 prediabetes cohort (Table 1).

Target (paper Table 1): N = 4176; mean age 53.3 (SD 17.0); 49% female;
NH White 62% / NH Black 12% / Hispanic 16% / Other 10%; BMI 30.3 (7.3);
HbA1c 5.7 (0.5)%; FPG 5.9 (0.8) mmol/L; SBP 128 (19); etc.

Prediabetes definition (any of):
  - HbA1c 5.7-6.4%   (LBXGH)
  - fasting plasma glucose 100-125 mg/dL  (LBXGLU)
  - 2-h OGTT glucose 140-199 mg/dL        (LBXGLT, 2013-16 only; no OGTT_J)
Exclusions: diagnosed diabetes (DIQ010==1), or HbA1c>=6.5, or FPG>=126, or OGTT>=200.
Restrict to adults age >= 18.

Weighting: FPG/OGTT are collected only in the morning FASTING subsample, so the
analytic domain is the fasting subsample. We weight with the fasting-subsample
weight WTSAF2YR divided by 3 (three 2-year cycles), per NHANES analytic guidelines.
(Note: WTMECPRP applies to the 2017-2020 COMBINED file, not these three standalone
2-year cycles -- so it is the wrong weight here.)

Data are downloaded from the public NHANES site and cached under ./nhanes_data/.
"""
from __future__ import annotations
import io
import os
import sys
import urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "nhanes_data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# cycle -> (year folder, file suffix)
CYCLES = {"2013-2014": ("2013", "H"), "2015-2016": ("2015", "I"), "2017-2018": ("2017", "J")}
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}_{suf}.XPT"

# file stem -> columns we need (SEQN always kept)
NEEDED = {
    "DEMO":   ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2",
               "WTMEC2YR", "WTINT2YR", "SDMVPSU", "SDMVSTRA"],
    "GHB":    ["LBXGH"],                       # HbA1c %
    "GLU":    ["LBXGLU", "WTSAF2YR"],          # fasting glucose mg/dL + fasting subsample weight
    "OGTT":   ["LBXGLT"],                       # 2-h OGTT mg/dL (H, I only)
    "BMX":    ["BMXBMI"],
    "BPX":    ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"],
    "HDL":    ["LBDHDD"],                       # mg/dL
    "TRIGLY": ["LBXTR", "LBDLDL"],             # triglycerides + LDL (fasting)
    "BIOPRO": ["LBXSCR"],                       # serum creatinine mg/dL
    "VID":    ["LBXVIDMS"],                     # 25(OH)D nmol/L
    "ALB_CR": ["URDACT"],                       # urine albumin/creatinine ratio mg/g
    "SMQ":    ["SMQ020", "SMQ040"],
    "DIQ":    ["DIQ010"],
    "MCQ":    ["MCQ160E", "MCQ160F", "MCQ160B", "MCQ160D"],  # MI, stroke, CHF, angina
    "KIQ_U":  ["KIQ025"],                       # dialysis in past 12 mo
    "MCQ_FH": ["MCQ300C"],                      # family history (also in MCQ file)
}
# MCQ300C lives in the MCQ file; handle as an alias so we download MCQ once.
FILE_ALIAS = {"MCQ_FH": "MCQ"}


def fetch(stem: str, year: str, suf: str) -> pd.DataFrame | None:
    real = FILE_ALIAS.get(stem, stem)
    url = BASE.format(year=year, stem=real, suf=suf)
    path = os.path.join(CACHE, f"{real}_{suf}.XPT")
    if not os.path.exists(path):
        try:
            print(f"  downloading {real}_{suf} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 5000:  # redirect stub, not a real XPT
                print(f"    !! {real}_{suf} too small ({len(data)}B) -> treat as missing")
                return None
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:  # noqa: BLE001
            print(f"    !! {real}_{suf} not available: {e}")
            return None
    try:
        df = pd.read_sas(path, format="xport")
    except Exception as e:  # noqa: BLE001
        print(f"    !! could not parse {real}_{suf}: {e}")
        return None
    return df


def load_cycle(cycle: str, year: str, suf: str) -> pd.DataFrame:
    print(f"Cycle {cycle} (suffix _{suf}):")
    demo = fetch("DEMO", year, suf)
    if demo is None:
        raise RuntimeError(f"DEMO missing for {cycle}")
    keep = ["SEQN"] + [c for c in NEEDED["DEMO"] if c in demo.columns]
    merged = demo[keep].copy()
    for stem, cols in NEEDED.items():
        if stem == "DEMO":
            continue
        df = fetch(stem, year, suf)
        if df is None:
            for c in cols:  # keep column shape consistent across cycles
                merged[c] = np.nan
            continue
        sub = df[["SEQN"] + [c for c in cols if c in df.columns]].copy()
        for c in cols:
            if c not in sub.columns:
                sub[c] = np.nan
        merged = merged.merge(sub, on="SEQN", how="left")
    merged["cycle"] = cycle
    return merged


def wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def wsd(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna()
    if m.sum() == 0:
        return np.nan
    mu = np.average(x[m], weights=w[m])
    var = np.average((x[m] - mu) ** 2, weights=w[m])
    return float(np.sqrt(var))


def wprop(mask: pd.Series, w: pd.Series) -> float:
    """Weighted proportion of `mask` among rows with defined mask AND weight."""
    valid = mask.notna() & w.notna()
    if (w[valid].sum()) == 0:
        return np.nan
    return float((w[valid] * mask[valid].astype(float)).sum() / w[valid].sum())


def egfr_ckdepi_2021(scr, age, female):
    """CKD-EPI 2021 race-free creatinine equation (mL/min/1.73 m^2)."""
    scr = np.asarray(scr, float)
    age = np.asarray(age, float)
    female = np.asarray(female, float)
    kappa = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** (-1.200)
            * (0.9938 ** age)
            * np.where(female == 1, 1.012, 1.0))
    return egfr


def main():
    frames = [load_cycle(c, y, s) for c, (y, s) in CYCLES.items()]
    df = pd.concat(frames, ignore_index=True)
    print(f"\nRaw pooled records: {len(df)}")

    # ---- derived variables --------------------------------------------------
    df["age"] = df["RIDAGEYR"]
    df["female"] = (df["RIAGENDR"] == 2).astype(float)
    reth = df["RIDRETH3"]
    df["race_nhwhite"] = (reth == 3).astype(float)
    df["race_nhblack"] = (reth == 4).astype(float)
    df["race_hispanic"] = reth.isin([1, 2]).astype(float)
    df["race_other"] = reth.isin([6, 7]).astype(float)
    df["postsec_ed"] = df["DMDEDUC2"].isin([4, 5]).astype(float)  # among 20+; 18-19 use DMDEDUC3
    df["current_smoker"] = ((df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2]))).astype(float)

    df["hba1c"] = df["LBXGH"]
    df["fpg_mgdl"] = df["LBXGLU"]
    df["fpg_mmol"] = df["LBXGLU"] / 18.0
    df["ogtt"] = df["LBXGLT"]
    df["bmi"] = df["BMXBMI"]
    sbp = df[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].replace(0, np.nan)
    df["sbp"] = sbp.mean(axis=1, skipna=True)
    df["hdl_mgdl"] = df["LBDHDD"]
    df["hdl_mmol"] = df["LBDHDD"] / 38.67
    df["ldl_mmol"] = df["LBDLDL"] / 38.67
    df["trig_mmol"] = df["LBXTR"] / 88.57
    df["scr_mgdl"] = df["LBXSCR"]
    df["scr_umol"] = df["LBXSCR"] * 88.42
    df["vitd_nmol"] = df["LBXVIDMS"]
    df["acr"] = df["URDACT"]

    df["egfr"] = egfr_ckdepi_2021(df["scr_mgdl"], df["age"], df["female"])
    df["ckd_3_5"] = (df["egfr"] < 60).astype(float)
    df["ckd_4_5"] = (df["egfr"] < 30).astype(float)
    df["microalb"] = ((df["acr"] >= 30) & (df["acr"] <= 300)).astype(float)
    df["macroalb"] = (df["acr"] > 300).astype(float)
    df["vitd_lt30"] = (df["vitd_nmol"] < 30).astype(float)
    df["vitd_30_50"] = ((df["vitd_nmol"] >= 30) & (df["vitd_nmol"] < 50)).astype(float)
    df["vitd_ge50"] = (df["vitd_nmol"] >= 50).astype(float)

    df["mi"] = (df["MCQ160E"] == 1).astype(float)
    df["stroke"] = (df["MCQ160F"] == 1).astype(float)
    df["chf"] = (df["MCQ160B"] == 1).astype(float)
    df["angina"] = (df["MCQ160D"] == 1).astype(float)
    df["dialysis"] = (df["KIQ025"] == 1).astype(float)
    df["famhx_dm"] = (df["MCQ300C"] == 1).astype(float)

    # multi-cycle fasting-subsample weight
    df["wtsaf6"] = df["WTSAF2YR"] / 3.0
    df["wtmec6"] = df["WTMEC2YR"] / 3.0

    # ---- cohort definition --------------------------------------------------
    adult = df["age"] >= 18
    diagnosed = df["DIQ010"] == 1
    lab_diab = (df["hba1c"] >= 6.5) | (df["fpg_mgdl"] >= 126) | (df["ogtt"] >= 200)
    prediab = (
        ((df["hba1c"] >= 5.7) & (df["hba1c"] <= 6.4))
        | ((df["fpg_mgdl"] >= 100) & (df["fpg_mgdl"] <= 125))
        | ((df["ogtt"] >= 140) & (df["ogtt"] <= 199))
    )
    base = adult & ~diagnosed & ~lab_diab & prediab

    fasting = df["WTSAF2YR"].notna() & (df["WTSAF2YR"] > 0)
    cohort_fast = df[base & fasting].copy()   # primary: fasting-subsample domain
    cohort_all = df[base].copy()               # sensitivity: all MEC examinees

    # ---- diagnostic: which definition reproduces N=4176 and mean age 53.3? ---
    hb_pre = (df["hba1c"] >= 5.7) & (df["hba1c"] <= 6.4)
    fpg_pre = (df["fpg_mgdl"] >= 100) & (df["fpg_mgdl"] <= 125)
    ogtt_pre = (df["ogtt"] >= 140) & (df["ogtt"] <= 199)
    no_diab = adult & ~diagnosed & ~lab_diab
    # complete data on the covariates a microsim needs to initialise a person
    complete = (df["bmi"].notna() & df["sbp"].notna() & df["hba1c"].notna()
                & df["scr_mgdl"].notna())
    defs = {
        "A fasting, HbA1c|FPG|OGTT (WTSAF/3)":  (no_diab & (hb_pre | fpg_pre | ogtt_pre) & fasting, "wtsaf6"),
        "B all MEC, HbA1c|FPG|OGTT (WTMEC/3)":  (no_diab & (hb_pre | fpg_pre | ogtt_pre), "wtmec6"),
        "C all MEC, HbA1c only       (WTMEC/3)": (adult & ~diagnosed & ~(df["hba1c"] >= 6.5) & hb_pre, "wtmec6"),
        "D all MEC, HbA1c|FPG        (WTMEC/3)": (no_diab & (hb_pre | fpg_pre), "wtmec6"),
        "F MEC+complete, HbA1c|FPG|OGTT":       (no_diab & (hb_pre | fpg_pre | ogtt_pre) & complete, "wtmec6"),
        "G MEC+complete, HbA1c|FPG":            (no_diab & (hb_pre | fpg_pre) & complete, "wtmec6"),
        "H MEC+complete, HbA1c only":           (adult & ~diagnosed & ~(df["hba1c"] >= 6.5) & hb_pre & complete, "wtmec6"),
    }
    print("\n==== Cohort definition diagnostics (target: N=4176, age 53.3) ====")
    print(f"  {'definition':42s} {'N':>6s} {'wt.age':>7s} {'wt.FPGsd':>9s}")
    for name, (mask, wcol) in defs.items():
        sub = df[mask]
        a = wmean(sub["age"], sub[wcol])
        fsd = wsd(sub["fpg_mmol"], sub[wcol])
        print(f"  {name:42s} {len(sub):6d} {a:7.1f} {fsd:9.2f}")

    # ---- Table 1: two bracketing reconstructions vs. the paper --------------
    # No single public-data definition reproduces (N=4176, age 53.3, HbA1c SD 0.5,
    # FPG SD 0.8) simultaneously. We report the two most defensible readings, which
    # bracket the paper's N:
    #   (1) fasting-subsample UNION  = def A: every person screened on HbA1c, FPG
    #       and OGTT (all measured in the morning fasting sample), WTSAF/3.
    #   (2) full-MEC UNION           = def B: HbA1c|FPG|OGTT on all examinees
    #       (FPG/OGTT contribute where measured), WTMEC/3.
    cohort_fasting = df[no_diab & (hb_pre | fpg_pre | ogtt_pre) & fasting].copy()
    cohort_mec = df[no_diab & (hb_pre | fpg_pre | ogtt_pre)].copy()

    CONT = [
        ("Age, years", "age"), ("HbA1c, %", "hba1c"), ("BMI, kg/m2", "bmi"),
        ("Systolic BP, mm Hg", "sbp"), ("HDL, mmol/L", "hdl_mmol"),
        ("LDL, mmol/L", "ldl_mmol"), ("Triglycerides, mmol/L", "trig_mmol"),
        ("Serum creatinine, umol/L", "scr_umol"),
        ("Fasting plasma glucose, mmol/L", "fpg_mmol"),
    ]
    PROP = [
        ("Female", "female"), ("NH White", "race_nhwhite"), ("NH Black", "race_nhblack"),
        ("Hispanic", "race_hispanic"), ("Other race/ethnicity", "race_other"),
        ("Postsecondary education (20+)", "postsec_ed"), ("Current smoker", "current_smoker"),
        ("25(OH)D <30 nmol/L", "vitd_lt30"), ("25(OH)D 30-50 nmol/L", "vitd_30_50"),
        ("25(OH)D >=50 nmol/L", "vitd_ge50"), ("Microalbuminuria", "microalb"),
        ("Macroalbuminuria", "macroalb"), ("CKD stage 3-5 (eGFR<60)", "ckd_3_5"),
        ("CKD stage 4-5 (eGFR<30)", "ckd_4_5"), ("Dialysis", "dialysis"),
        ("Myocardial infarction", "mi"), ("Stroke", "stroke"),
        ("Congestive heart failure", "chf"), ("Angina", "angina"),
        ("Family history of diabetes", "famhx_dm"),
    ]

    def table1(cohort, wcol):
        w = cohort[wcol]
        vals = {"N (unweighted)": str(len(cohort))}
        for lbl, col in CONT:
            vals[lbl] = f"{wmean(cohort[col], w):.1f} ({wsd(cohort[col], w):.1f})"
        for lbl, col in PROP:
            vals[lbl] = f"{100*wprop(cohort[col], w):.1f}%"
        return vals

    paper = {
        "N (unweighted)": "4176", "Age, years": "53.3 (17.0)", "Female": "49%",
        "NH White": "62%", "NH Black": "12%", "Hispanic": "16%", "Other race/ethnicity": "10%",
        "Postsecondary education (20+)": "61%", "Current smoker": "18%", "HbA1c, %": "5.7 (0.5)",
        "BMI, kg/m2": "30.3 (7.3)", "Systolic BP, mm Hg": "128 (19)", "HDL, mmol/L": "1.35 (0.39)",
        "LDL, mmol/L": "3.00 (0.93)", "Triglycerides, mmol/L": "1.12 (0.67)",
        "Serum creatinine, umol/L": "80 (35)", "Fasting plasma glucose, mmol/L": "5.9 (0.8)",
        "25(OH)D <30 nmol/L": "7.6%", "25(OH)D 30-50 nmol/L": "21.9%", "25(OH)D >=50 nmol/L": "70.5%",
        "Microalbuminuria": "8.2%", "Macroalbuminuria": "0.8%", "CKD stage 3-5 (eGFR<60)": "9.7%",
        "CKD stage 4-5 (eGFR<30)": "0.5%", "Dialysis": "<0.1%", "Myocardial infarction": "2.5%",
        "Stroke": "3.1%", "Congestive heart failure": "1.3%", "Angina": "2.8%",
        "Family history of diabetes": "27%",
    }

    order = (["N (unweighted)"] + [l for l, _ in CONT] + [l for l, _ in PROP])
    t_fast = table1(cohort_fasting, "wtsaf6")
    t_mec = table1(cohort_mec, "wtmec6")
    out = pd.DataFrame(
        [(lbl, paper.get(lbl, ""), t_fast.get(lbl, ""), t_mec.get(lbl, "")) for lbl in order],
        columns=["Variable", "Paper Table 1", "Recon: fasting-union (A)", "Recon: MEC-union (B)"],
    )
    csv_path = os.path.join(OUT, "table1_reconstructed.csv")
    out.to_csv(csv_path, index=False)

    print("\n==== Reconstructed Table 1 vs. paper (two bracketing definitions) ====")
    with pd.option_context("display.max_rows", None, "display.width", 130,
                           "display.max_colwidth", 40):
        print(out.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    # persist both analytic cohorts for downstream simulation work
    keepcols = ["SEQN", "cycle", "age", "female", "race_nhwhite", "race_nhblack",
                "race_hispanic", "race_other", "postsec_ed", "current_smoker",
                "hba1c", "fpg_mgdl", "fpg_mmol", "ogtt", "bmi", "sbp",
                "hdl_mmol", "ldl_mmol", "trig_mmol", "scr_mgdl", "egfr",
                "vitd_nmol", "acr", "microalb", "macroalb", "ckd_3_5", "ckd_4_5",
                "dialysis", "mi", "stroke", "chf", "angina", "famhx_dm",
                "WTSAF2YR", "wtsaf6", "WTMEC2YR", "wtmec6", "SDMVPSU", "SDMVSTRA"]
    cohort_fasting[keepcols].to_csv(os.path.join(OUT, "cohort_fasting_union.csv"), index=False)
    cohort_mec[keepcols].to_csv(os.path.join(OUT, "cohort_mec_union.csv"), index=False)
    print(f"Saved analytic cohorts: cohort_fasting_union.csv ({len(cohort_fasting)} rows), "
          f"cohort_mec_union.csv ({len(cohort_mec)} rows)")


if __name__ == "__main__":
    main()
