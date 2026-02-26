"""
NHANES Mortality-Fibrosis Analysis: Steps 1-6

Downloads NHANES data, computes fibrosis markers, and analyzes mortality
by fibrosis status across three cohorts:
  - 2007-2008: Long follow-up, full UCOD detail, FIB-4 only
  - 2011-2012: Moderate follow-up, full UCOD detail, FIB-4 only
  - 2017-2018: Short follow-up, coarsened UCOD, FIB-4 + elastography (LSM)

Public-use limitations documented throughout:
  (a) Follow-up ends 12/31/2019 for all cohorts
  (b) UCOD_LEADING coarsened to 3 groups for 2015-2016 and 2017-2018
  (c) Some death variables are perturbed in public-use files
"""

import os
import sys
import warnings
import hashlib
import requests
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_NHANES = os.path.join(BASE_DIR, "data", "raw", "nhanes")
RAW_LMF = os.path.join(BASE_DIR, "data", "raw", "lmf")
DERIVED = os.path.join(BASE_DIR, "data", "derived")
TABLES = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUTS = os.path.join(BASE_DIR, "outputs")

for d in [RAW_NHANES, RAW_LMF, DERIVED, TABLES, FIGURES, OUTPUTS]:
    os.makedirs(d, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────
COHORTS = {
    "2007-2008": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.xpt",
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BIOPRO_E.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/CBC_E.xpt",
        "elast": None,
        "lmf": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2007_2008_MORT_2019_PUBLIC.dat",
        "weight_var": "WTMEC2YR",
        "has_elast": False,
    },
    "2011-2012": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BIOPRO_G.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/CBC_G.xpt",
        "elast": None,
        "lmf": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2011_2012_MORT_2019_PUBLIC.dat",
        "weight_var": "WTMEC2YR",
        "has_elast": False,
    },
    "2017-2018": {
        "demo": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
        "biochem": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.xpt",
        "cbc": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.xpt",
        "elast": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/LUX_J.xpt",
        "lmf": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2017_2018_MORT_2019_PUBLIC.dat",
        "weight_var": "WTMEC2YR",
        "has_elast": True,
    },
}

# Analysis windows
WINDOW_36 = 36  # Primary window for earlier cohorts
WINDOW_H = 24   # Harmonized window all cohorts can (partially) support

# UCOD labels
UCOD_LABELS = {
    1: "Heart disease",
    2: "Malignant neoplasms",
    3: "Chronic lower resp.",
    4: "Accidents",
    5: "Cerebrovascular",
    6: "Alzheimer's",
    7: "Diabetes",
    8: "Influenza/pneumonia",
    9: "Nephritis",
    10: "All other causes",
}

# FIB-4 thresholds
FIB4_HIGH = 2.67   # Advanced fibrosis
FIB4_LOW = 1.30    # Low fibrosis

# LSM cutpoint sets (kPa) for liver stiffness
# Set A: Castera et al. 2005/EASL guidelines
LSM_CUTPOINTS_A = {"name": "Castera/EASL", "significant": 7.1, "advanced": 9.5, "cirrhosis": 12.5}
# Set B: Eddowes et al. 2019 (NAFLD-specific, Youden-optimized)
LSM_CUTPOINTS_B = {"name": "Eddowes/NAFLD", "significant": 8.2, "advanced": 9.7, "cirrhosis": 13.6}


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Download data
# ══════════════════════════════════════════════════════════════════════

def download_file(url, dest_dir, filename=None):
    """Download with caching; return local path."""
    if filename is None:
        filename = os.path.basename(url)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        return dest
    print(f"  Downloading {url}")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    md5 = hashlib.md5(resp.content).hexdigest()
    print(f"    -> {dest} (MD5: {md5}, {len(resp.content)} bytes)")
    return dest


def read_xpt(path):
    """Read SAS XPT file, handling encoding issues in variable labels."""
    try:
        df, meta = pyreadstat.read_xport(path)
    except UnicodeDecodeError:
        # Some NHANES files have non-UTF-8 chars (e.g. µ) in variable labels
        df, meta = pyreadstat.read_xport(path, encoding="latin1")
    return df


def _safe_int(s):
    s = s.strip().replace(".", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _safe_float(s):
    s = s.strip().replace(".", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_lmf_dat(filepath):
    """Parse fixed-width LMF .dat file (2019 public-use release)."""
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            padded = raw.ljust(48)
            rows.append({
                "SEQN": _safe_int(padded[0:14]),
                "ELIGSTAT": _safe_int(padded[14:15]),
                "MORTSTAT": _safe_int(padded[15:16]),
                "UCOD_LEADING": _safe_int(padded[16:19]),
                "DIABETES_LMF": _safe_int(padded[19:20]),
                "HYPERTEN_LMF": _safe_int(padded[20:21]),
                "PERMTH_INT": _safe_float(padded[42:45]),
                "PERMTH_EXM": _safe_float(padded[45:48]),
            })
    return pd.DataFrame(rows)


def download_cohort_data(cycle, cfg):
    """Download all files for one cohort. Returns dict of DataFrames."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: {cycle}")
    print(f"{'='*60}")

    cycle_dir = os.path.join(RAW_NHANES, cycle.replace("-", "_"))
    os.makedirs(cycle_dir, exist_ok=True)

    data = {}

    # Demographics
    path = download_file(cfg["demo"], cycle_dir)
    data["demo"] = read_xpt(path)
    print(f"  DEMO: {len(data['demo'])} rows, cols: {list(data['demo'].columns[:10])}...")

    # Biochemistry (AST, ALT)
    path = download_file(cfg["biochem"], cycle_dir)
    data["biochem"] = read_xpt(path)
    print(f"  BIOCHEM: {len(data['biochem'])} rows")

    # CBC (Platelets)
    path = download_file(cfg["cbc"], cycle_dir)
    data["cbc"] = read_xpt(path)
    print(f"  CBC: {len(data['cbc'])} rows")

    # Elastography (2017-2018 only)
    if cfg["elast"]:
        path = download_file(cfg["elast"], cycle_dir)
        data["elast"] = read_xpt(path)
        print(f"  ELAST: {len(data['elast'])} rows")

    # LMF
    lmf_path = download_file(cfg["lmf"], RAW_LMF)
    data["lmf"] = parse_lmf_dat(lmf_path)
    print(f"  LMF: {len(data['lmf'])} rows")

    return data


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Clean + merge
# ══════════════════════════════════════════════════════════════════════

def identify_lab_columns(biochem_df):
    """Identify AST and ALT column names (vary by cycle suffix)."""
    cols = biochem_df.columns.tolist()
    ast_col = None
    alt_col = None
    for c in cols:
        if c.startswith("LBXSASSI") or c == "LBXSASSI":
            ast_col = c  # AST (SGOT) in IU/L
        elif c.startswith("LBXSATSI") or c == "LBXSATSI":
            alt_col = c  # ALT (SGPT) in IU/L
    # Fallback: search for common patterns
    if ast_col is None:
        for c in cols:
            if "ASS" in c.upper() or "SGOT" in c.upper() or c == "LBXSASSI":
                ast_col = c
                break
    if alt_col is None:
        for c in cols:
            if "ATS" in c.upper() or "SGPT" in c.upper() or c == "LBXSATSI":
                alt_col = c
                break
    return ast_col, alt_col


def identify_platelet_column(cbc_df):
    """Identify platelet count column."""
    cols = cbc_df.columns.tolist()
    for c in cols:
        if c.startswith("LBXPLTSI") or c == "LBXPLTSI":
            return c  # Platelet count (1000 cells/uL)
    for c in cols:
        if "PLT" in c.upper():
            return c
    return None


def merge_cohort(cycle, cfg, data):
    """Merge DEMO + labs + LMF; apply eligibility filters."""
    print(f"\n--- Merging {cycle} ---")

    demo = data["demo"].copy()
    biochem = data["biochem"].copy()
    cbc = data["cbc"].copy()
    lmf = data["lmf"].copy()

    # Identify lab columns
    ast_col, alt_col = identify_lab_columns(biochem)
    plt_col = identify_platelet_column(cbc)
    print(f"  Lab columns: AST={ast_col}, ALT={alt_col}, PLT={plt_col}")

    # Standardize column names
    if ast_col:
        biochem = biochem.rename(columns={ast_col: "AST"})
    if alt_col:
        biochem = biochem.rename(columns={alt_col: "ALT"})
    if plt_col:
        cbc = cbc.rename(columns={plt_col: "PLATELETS"})

    # Merge
    df = demo.merge(biochem[["SEQN", "AST", "ALT"]], on="SEQN", how="left")
    df = df.merge(cbc[["SEQN", "PLATELETS"]], on="SEQN", how="left")

    if cfg["has_elast"] and "elast" in data:
        elast = data["elast"].copy()
        # Identify LSM column (liver stiffness median in kPa)
        elast_cols = elast.columns.tolist()
        print(f"  Elastography columns: {elast_cols}")
        lsm_col = None
        cap_col = None
        for c in elast_cols:
            if "LUXSMED" in c.upper() or c == "LUXSMED":
                lsm_col = c
            elif "LUXCAPM" in c.upper() or c == "LUXCAPM":
                cap_col = c
        # Broader search
        if lsm_col is None:
            for c in elast_cols:
                cu = c.upper()
                if "MED" in cu and ("LUX" in cu or "STIFF" in cu or "LSM" in cu):
                    lsm_col = c
                    break
                if "SMED" in cu:
                    lsm_col = c
                    break
        if cap_col is None:
            for c in elast_cols:
                cu = c.upper()
                if "CAP" in cu and "MED" in cu:
                    cap_col = c
                    break
                if "CAPM" in cu:
                    cap_col = c
                    break

        print(f"  LSM column: {lsm_col}, CAP column: {cap_col}")

        rename_map = {"SEQN": "SEQN"}
        keep_cols = ["SEQN"]
        if lsm_col:
            rename_map[lsm_col] = "LSM_KPA"
            keep_cols.append(lsm_col)
        if cap_col:
            rename_map[cap_col] = "CAP_DBM"
            keep_cols.append(cap_col)
        elast_sub = elast[keep_cols].rename(columns=rename_map)
        df = df.merge(elast_sub, on="SEQN", how="left")

    # Merge LMF
    df = df.merge(lmf, on="SEQN", how="left")

    # Filter: ELIGSTAT == 1
    n_before = len(df)
    df = df[df["ELIGSTAT"] == 1].copy()
    print(f"  After ELIGSTAT=1 filter: {len(df)} (dropped {n_before - len(df)})")

    # Age: compute from RIDAGEYR (age in years at screening)
    df["AGE"] = df["RIDAGEYR"]

    # Adults only (>=18)
    n_before = len(df)
    df = df[df["AGE"] >= 18].copy()
    print(f"  Adults (>=18): {len(df)} (dropped {n_before - len(df)})")

    # Sex: RIAGENDR (1=Male, 2=Female)
    df["FEMALE"] = (df["RIAGENDR"] == 2).astype(int)

    # Add cycle label
    df["CYCLE"] = cycle

    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Define fibrosis status
# ══════════════════════════════════════════════════════════════════════

def compute_fib4(df):
    """Compute FIB-4 index: (age * AST) / (platelets * sqrt(ALT))."""
    df = df.copy()
    # Platelets in 1000 cells/uL (NHANES reports in this unit already)
    valid = (df["AST"] > 0) & (df["ALT"] > 0) & (df["PLATELETS"] > 0) & (df["AGE"].notna())
    df["FIB4"] = np.nan
    df.loc[valid, "FIB4"] = (
        df.loc[valid, "AGE"] * df.loc[valid, "AST"]
    ) / (
        df.loc[valid, "PLATELETS"] * np.sqrt(df.loc[valid, "ALT"])
    )

    # Threshold-based classification
    df["FIB4_CAT"] = pd.Series(np.nan, index=df.index, dtype="object")
    df.loc[df["FIB4"] < FIB4_LOW, "FIB4_CAT"] = "low"
    df.loc[(df["FIB4"] >= FIB4_LOW) & (df["FIB4"] < FIB4_HIGH), "FIB4_CAT"] = "indeterminate"
    df.loc[df["FIB4"] >= FIB4_HIGH, "FIB4_CAT"] = "high"

    # Binary for primary analysis (exclude indeterminate)
    df["FIBROSIS_FIB4"] = np.nan  # NaN = indeterminate or missing
    df.loc[df["FIB4_CAT"] == "high", "FIBROSIS_FIB4"] = 1
    df.loc[df["FIB4_CAT"] == "low", "FIBROSIS_FIB4"] = 0

    n_fib4 = df["FIB4"].notna().sum()
    n_high = (df["FIB4_CAT"] == "high").sum()
    n_low = (df["FIB4_CAT"] == "low").sum()
    n_indet = (df["FIB4_CAT"] == "indeterminate").sum()
    print(f"  FIB-4 computed: {n_fib4} valid, {n_high} high, {n_low} low, {n_indet} indeterminate")

    return df


def compute_lsm_fibrosis(df, cutpoints):
    """Classify fibrosis from liver stiffness median (LSM in kPa)."""
    df = df.copy()
    prefix = cutpoints["name"].replace("/", "_").replace(" ", "_")

    col_stage = f"LSM_STAGE_{prefix}"
    col_binary = f"FIBROSIS_LSM_{prefix}"

    df[col_stage] = pd.Series(np.nan, index=df.index, dtype="object")
    df[col_binary] = np.nan

    has_lsm = df["LSM_KPA"].notna() if "LSM_KPA" in df.columns else pd.Series(False, index=df.index)

    if has_lsm.sum() == 0:
        print(f"  LSM staging ({cutpoints['name']}): no LSM data available")
        return df

    sig = cutpoints["significant"]
    adv = cutpoints["advanced"]
    cir = cutpoints["cirrhosis"]

    df.loc[has_lsm & (df["LSM_KPA"] < sig), col_stage] = "F0-F1"
    df.loc[has_lsm & (df["LSM_KPA"] >= sig) & (df["LSM_KPA"] < adv), col_stage] = "F2"
    df.loc[has_lsm & (df["LSM_KPA"] >= adv) & (df["LSM_KPA"] < cir), col_stage] = "F3"
    df.loc[has_lsm & (df["LSM_KPA"] >= cir), col_stage] = "F4"

    # Binary: fibrosis_yes = F3+F4, fibrosis_no = F0-F1
    df.loc[df[col_stage].isin(["F3", "F4"]), col_binary] = 1
    df.loc[df[col_stage] == "F0-F1", col_binary] = 0
    # F2 = indeterminate for binary

    n_lsm = has_lsm.sum()
    n_yes = (df[col_binary] == 1).sum()
    n_no = (df[col_binary] == 0).sum()
    print(f"  LSM staging ({cutpoints['name']}): {n_lsm} with LSM, {n_yes} fibrosis+, {n_no} fibrosis-")

    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Define outcomes
# ══════════════════════════════════════════════════════════════════════

def define_outcomes(df, window):
    """Define death within window and cause-group indicators."""
    df = df.copy()
    w_suffix = f"_{window}m"

    # Person-time (months, capped at window)
    df[f"FU_MONTHS{w_suffix}"] = df["PERMTH_EXM"].clip(upper=window)
    df[f"PY{w_suffix}"] = df[f"FU_MONTHS{w_suffix}"] / 12.0

    # All-cause death within window
    df[f"DEATH{w_suffix}"] = (
        (df["MORTSTAT"] == 1) & (df["PERMTH_EXM"] <= window)
    ).astype(int)

    # Check who has follow-up reaching the window
    df[f"FU_GE_WINDOW{w_suffix}"] = (df["PERMTH_EXM"] >= window).astype(int)

    # Cause-group deaths
    for code, label in UCOD_LABELS.items():
        col = f"DEATH_UCOD{code}{w_suffix}"
        df[col] = (
            (df[f"DEATH{w_suffix}"] == 1) & (df["UCOD_LEADING"] == code)
        ).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Analyses
# ══════════════════════════════════════════════════════════════════════

def cohort_card(df, cycle, window):
    """Produce descriptive cohort card."""
    w = f"_{window}m"
    card = {"cycle": cycle, "window_months": window}
    card["n_adults"] = len(df)
    card["n_fib4_valid"] = df["FIB4"].notna().sum()
    card["n_fib4_high"] = (df["FIB4_CAT"] == "high").sum()
    card["n_fib4_low"] = (df["FIB4_CAT"] == "low").sum()
    card["n_fib4_indet"] = (df["FIB4_CAT"] == "indeterminate").sum()

    if "LSM_KPA" in df.columns:
        card["n_lsm_valid"] = df["LSM_KPA"].notna().sum()
    else:
        card["n_lsm_valid"] = 0

    card["median_fu"] = df["PERMTH_EXM"].median()
    iqr = df["PERMTH_EXM"].quantile([0.25, 0.75])
    card["iqr_fu"] = f"{iqr.iloc[0]:.0f}-{iqr.iloc[1]:.0f}"
    card["pct_ge_window"] = (df["PERMTH_EXM"] >= window).mean() * 100

    card["deaths_in_window"] = df[f"DEATH{w}"].sum()

    # Cause-group deaths
    for code in UCOD_LABELS:
        col = f"DEATH_UCOD{code}{w}"
        if col in df.columns:
            card[f"deaths_ucod{code}"] = df[col].sum()

    return card


def compute_rates(df, fibrosis_col, window, label=""):
    """Compute death rates and risks by fibrosis status."""
    w = f"_{window}m"
    results = []

    for fib_val, fib_name in [(1, "fibrosis_yes"), (0, "fibrosis_no")]:
        sub = df[df[fibrosis_col] == fib_val]
        n = len(sub)
        if n == 0:
            continue

        deaths = sub[f"DEATH{w}"].sum()
        py = sub[f"PY{w}"].sum()
        risk = deaths / n if n > 0 else np.nan
        rate = (deaths / py * 1000) if py > 0 else np.nan

        row = {
            "fibrosis_def": label,
            "fibrosis_col": fibrosis_col,
            "fibrosis_status": fib_name,
            "window": window,
            "n": n,
            "deaths": deaths,
            "person_years": round(py, 1),
            "risk": round(risk, 4),
            "rate_per_1000py": round(rate, 1),
        }

        # Cause-group rates
        for code in UCOD_LABELS:
            col = f"DEATH_UCOD{code}{w}"
            if col in df.columns:
                d_cause = sub[col].sum()
                row[f"deaths_ucod{code}"] = d_cause
                row[f"rate_ucod{code}_per_1000py"] = round(d_cause / py * 1000, 1) if py > 0 else np.nan

        results.append(row)

    return pd.DataFrame(results)


def compute_risk_ratio(df, fibrosis_col, window):
    """Compute unadjusted risk ratio for all-cause death."""
    w = f"_{window}m"
    fib_yes = df[df[fibrosis_col] == 1]
    fib_no = df[df[fibrosis_col] == 0]

    n1, n0 = len(fib_yes), len(fib_no)
    d1 = fib_yes[f"DEATH{w}"].sum()
    d0 = fib_no[f"DEATH{w}"].sum()

    if n1 == 0 or n0 == 0 or d0 == 0:
        return {"RR": np.nan, "RR_lower": np.nan, "RR_upper": np.nan,
                "n_fib_yes": n1, "n_fib_no": n0, "d_fib_yes": d1, "d_fib_no": d0}

    r1 = d1 / n1
    r0 = d0 / n0
    rr = r1 / r0

    # Log-RR CI
    se_log_rr = np.sqrt((1 / d1 - 1 / n1) + (1 / d0 - 1 / n0)) if d1 > 0 else np.nan
    if np.isfinite(se_log_rr):
        rr_lower = np.exp(np.log(rr) - 1.96 * se_log_rr)
        rr_upper = np.exp(np.log(rr) + 1.96 * se_log_rr)
    else:
        rr_lower = rr_upper = np.nan

    return {"RR": round(rr, 2), "RR_lower": round(rr_lower, 2), "RR_upper": round(rr_upper, 2),
            "n_fib_yes": n1, "n_fib_no": n0, "d_fib_yes": d1, "d_fib_no": d0}


def fit_poisson_model(df, fibrosis_col, window, adjusted=False):
    """Fit Poisson model for rate ratio with offset log(PY)."""
    w = f"_{window}m"
    sub = df[df[fibrosis_col].notna()].copy()
    sub = sub[sub[f"PY{w}"] > 0].copy()
    sub["log_py"] = np.log(sub[f"PY{w}"])
    sub["fib"] = sub[fibrosis_col].astype(int)

    if sub["fib"].sum() == 0 or sub[f"DEATH{w}"].sum() < 5:
        return None

    try:
        if adjusted:
            formula = f"DEATH{w} ~ fib + AGE + FEMALE"
        else:
            formula = f"DEATH{w} ~ fib"

        model = smf.glm(
            formula=formula,
            data=sub,
            family=sm.families.Poisson(),
            offset=sub["log_py"],
        ).fit()
        return model
    except Exception as e:
        print(f"    Poisson model failed: {e}")
        return None


def format_poisson_result(model, label=""):
    """Extract rate ratio and CI from Poisson model."""
    if model is None:
        return {"label": label, "IRR": np.nan, "IRR_lower": np.nan, "IRR_upper": np.nan, "p_value": np.nan}

    coef = model.params["fib"]
    ci = model.conf_int().loc["fib"]
    irr = np.exp(coef)
    irr_lower = np.exp(ci.iloc[0])
    irr_upper = np.exp(ci.iloc[1])
    p = model.pvalues["fib"]

    return {
        "label": label,
        "IRR": round(irr, 2),
        "IRR_lower": round(irr_lower, 2),
        "IRR_upper": round(irr_upper, 2),
        "p_value": round(p, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Figures
# ══════════════════════════════════════════════════════════════════════

def plot_allcause_bars(rates_df, cycle, window, fibrosis_def, filename):
    """Bar plot: all-cause death rate by fibrosis status."""
    sub = rates_df[
        (rates_df["window"] == window) &
        (rates_df["fibrosis_def"] == fibrosis_def)
    ].copy()

    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"fibrosis_yes": "#d62728", "fibrosis_no": "#2ca02c"}
    labels_map = {"fibrosis_yes": "Fibrosis+", "fibrosis_no": "Fibrosis-"}

    x = np.arange(len(sub))
    bars = ax.bar(
        x,
        sub["rate_per_1000py"],
        color=[colors.get(s, "gray") for s in sub["fibrosis_status"]],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add count annotations
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, row["rate_per_1000py"] + 1, f"n={row['n']}\nd={int(row['deaths'])}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map.get(s, s) for s in sub["fibrosis_status"]])
    ax.set_ylabel("Death rate per 1,000 PY")
    ax.set_title(f"{cycle}: All-cause mortality ({window}m)\n[{fibrosis_def}]", fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_causegroup_bars(rates_df, cycle, window, fibrosis_def, filename):
    """Bar plot: cause-group death rates by fibrosis status (only groups with events)."""
    sub = rates_df[
        (rates_df["window"] == window) &
        (rates_df["fibrosis_def"] == fibrosis_def)
    ].copy()

    if len(sub) == 0:
        return

    # Find cause groups with >=1 event in either group
    cause_cols = [c for c in sub.columns if c.startswith("rate_ucod") and c.endswith("_per_1000py")]
    death_cols = [c for c in sub.columns if c.startswith("deaths_ucod")]

    active_causes = []
    for dc in death_cols:
        code = int(dc.replace("deaths_ucod", ""))
        total_deaths = sub[dc].sum()
        if total_deaths > 0:
            active_causes.append(code)

    if not active_causes:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(active_causes) * 1.5), 5))
    width = 0.35
    x = np.arange(len(active_causes))

    for i, (fib_status, color) in enumerate([("fibrosis_no", "#2ca02c"), ("fibrosis_yes", "#d62728")]):
        row = sub[sub["fibrosis_status"] == fib_status]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        rates = [row.get(f"rate_ucod{c}_per_1000py", 0) for c in active_causes]
        counts = [int(row.get(f"deaths_ucod{c}", 0)) for c in active_causes]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, rates, width, label=fib_status.replace("_", " ").title(),
                       color=color, edgecolor="black", linewidth=0.5, alpha=0.8)
        for j, (r, c) in enumerate(zip(rates, counts)):
            if c > 0:
                ax.text(x[j] + offset, r + 0.5, str(c), ha="center", va="bottom", fontsize=7)

    cause_labels = [UCOD_LABELS.get(c, f"UCOD {c}") for c in active_causes]
    ax.set_xticks(x)
    ax.set_xticklabels(cause_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Death rate per 1,000 PY")
    ax.set_title(f"{cycle}: Cause-group mortality ({window}m)\n[{fibrosis_def}]", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_combined_comparison(all_rates, window, fibrosis_def, filename):
    """Compare all-cause death rates across cohorts at harmonized window."""
    sub = all_rates[
        (all_rates["window"] == window) &
        (all_rates["fibrosis_def"] == fibrosis_def)
    ].copy()

    if len(sub) == 0:
        return

    cycles = sub["cycle"].unique()
    fig, ax = plt.subplots(figsize=(max(6, len(cycles) * 2.5), 5))

    width = 0.35
    x = np.arange(len(cycles))

    for i, (fib_status, color) in enumerate([("fibrosis_no", "#2ca02c"), ("fibrosis_yes", "#d62728")]):
        rates = []
        counts_txt = []
        for cyc in cycles:
            row = sub[(sub["cycle"] == cyc) & (sub["fibrosis_status"] == fib_status)]
            if len(row) > 0:
                rates.append(row.iloc[0]["rate_per_1000py"])
                counts_txt.append(f"n={row.iloc[0]['n']}, d={int(row.iloc[0]['deaths'])}")
            else:
                rates.append(0)
                counts_txt.append("")

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, rates, width,
                       label=fib_status.replace("_", " ").title(),
                       color=color, edgecolor="black", linewidth=0.5, alpha=0.8)

        for j, (r, txt) in enumerate(zip(rates, counts_txt)):
            ax.text(x[j] + offset, r + 1, txt, ha="center", va="bottom", fontsize=7, rotation=15)

    ax.set_xticks(x)
    ax.set_xticklabels(cycles)
    ax.set_ylabel("Death rate per 1,000 PY")
    ax.set_title(f"Cross-cohort comparison: All-cause mortality ({window}m)\n[{fibrosis_def}]", fontsize=10)
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("NHANES MORTALITY-FIBROSIS ANALYSIS PIPELINE")
    print("=" * 70)

    all_cohort_cards = []
    all_rates_allcause = []
    all_rates_causegroup = []
    all_effect_estimates = []
    cohort_dfs = {}

    for cycle, cfg in COHORTS.items():

        # ── Step 1: Download ──
        data = download_cohort_data(cycle, cfg)

        # ── Step 2: Merge ──
        df = merge_cohort(cycle, cfg, data)

        # ── Step 3: Fibrosis ──
        print(f"\n--- Fibrosis classification ({cycle}) ---")
        df = compute_fib4(df)

        if cfg["has_elast"] and "LSM_KPA" in df.columns:
            df = compute_lsm_fibrosis(df, LSM_CUTPOINTS_A)
            df = compute_lsm_fibrosis(df, LSM_CUTPOINTS_B)

        # ── Step 4: Outcomes ──
        print(f"\n--- Outcomes ({cycle}) ---")
        # Define for both windows
        if cycle != "2017-2018":
            df = define_outcomes(df, WINDOW_36)
        df = define_outcomes(df, WINDOW_H)

        cohort_dfs[cycle] = df

        # ── Step 5: Analyses ──
        print(f"\n--- Analyses ({cycle}) ---")

        # Determine which windows to analyze
        windows = [WINDOW_H]
        if cycle != "2017-2018":
            windows.append(WINDOW_36)

        # Define fibrosis definitions to analyze
        fib_defs = [("FIB4 (>=2.67 vs <1.30)", "FIBROSIS_FIB4")]
        if cfg["has_elast"] and "LSM_KPA" in df.columns:
            prefix_a = LSM_CUTPOINTS_A["name"].replace("/", "_").replace(" ", "_")
            prefix_b = LSM_CUTPOINTS_B["name"].replace("/", "_").replace(" ", "_")
            fib_defs.append((f"LSM ({LSM_CUTPOINTS_A['name']})", f"FIBROSIS_LSM_{prefix_a}"))
            fib_defs.append((f"LSM ({LSM_CUTPOINTS_B['name']})", f"FIBROSIS_LSM_{prefix_b}"))

        for window in windows:
            # Cohort card
            card = cohort_card(df, cycle, window)
            all_cohort_cards.append(card)
            print(f"\n  Cohort card ({cycle}, {window}m):")
            for k, v in card.items():
                print(f"    {k}: {v}")

            for fib_label, fib_col in fib_defs:
                if fib_col not in df.columns:
                    continue

                print(f"\n  Rates ({cycle}, {window}m, {fib_label}):")
                rates = compute_rates(df, fib_col, window, label=fib_label)
                rates["cycle"] = cycle
                print(rates[["fibrosis_status", "n", "deaths", "rate_per_1000py"]].to_string(index=False))

                # Separate all-cause and cause-group
                all_rates_allcause.append(rates)

                # Risk ratio
                rr = compute_risk_ratio(df, fib_col, window)
                print(f"  Unadjusted RR: {rr['RR']} ({rr['RR_lower']}-{rr['RR_upper']})")
                rr["cycle"] = cycle
                rr["window"] = window
                rr["fibrosis_def"] = fib_label

                # Poisson models
                pois_unadj = fit_poisson_model(df, fib_col, window, adjusted=False)
                pois_adj = fit_poisson_model(df, fib_col, window, adjusted=True)

                rr_pois_u = format_poisson_result(pois_unadj, "Poisson unadjusted")
                rr_pois_a = format_poisson_result(pois_adj, "Poisson age+sex adjusted")
                print(f"  Poisson IRR (unadj): {rr_pois_u['IRR']} ({rr_pois_u['IRR_lower']}-{rr_pois_u['IRR_upper']}), p={rr_pois_u['p_value']}")
                print(f"  Poisson IRR (adj):   {rr_pois_a['IRR']} ({rr_pois_a['IRR_lower']}-{rr_pois_a['IRR_upper']}), p={rr_pois_a['p_value']}")

                ee_row = {**rr, **{f"unadj_{k}": v for k, v in rr_pois_u.items()},
                          **{f"adj_{k}": v for k, v in rr_pois_a.items()}}
                all_effect_estimates.append(ee_row)

    # ── Compile tables ──
    print("\n" + "=" * 70)
    print("COMPILING TABLES")
    print("=" * 70)

    # Cohort cards
    cards_df = pd.DataFrame(all_cohort_cards)
    cards_df.to_csv(os.path.join(TABLES, "cohort_card_by_cohort.csv"), index=False)
    print(f"Saved: cohort_card_by_cohort.csv")

    # All-cause rates
    if all_rates_allcause:
        allcause_df = pd.concat(all_rates_allcause, ignore_index=True)
        allcause_df.to_csv(os.path.join(TABLES, "mortality_allcause_by_cohort_window_fibrosisdef.csv"), index=False)
        print(f"Saved: mortality_allcause_by_cohort_window_fibrosisdef.csv")

        # Cause-group table
        cause_cols = [c for c in allcause_df.columns if "ucod" in c]
        causegroup_rows = []
        for _, row in allcause_df.iterrows():
            for code in UCOD_LABELS:
                d_col = f"deaths_ucod{code}"
                r_col = f"rate_ucod{code}_per_1000py"
                if d_col in row and pd.notna(row.get(d_col)):
                    causegroup_rows.append({
                        "cycle": row["cycle"],
                        "window": row["window"],
                        "fibrosis_def": row["fibrosis_def"],
                        "fibrosis_status": row["fibrosis_status"],
                        "n": row["n"],
                        "person_years": row["person_years"],
                        "ucod_code": code,
                        "ucod_label": UCOD_LABELS[code],
                        "deaths": row[d_col],
                        "rate_per_1000py": row.get(r_col, np.nan),
                        "flag_lt10": "UNSTABLE" if row[d_col] < 10 else "",
                    })

        if causegroup_rows:
            cg_df = pd.DataFrame(causegroup_rows)
            cg_df.to_csv(os.path.join(TABLES, "mortality_causegroup_by_cohort_window_fibrosisdef.csv"), index=False)
            print(f"Saved: mortality_causegroup_by_cohort_window_fibrosisdef.csv")

    # Effect estimates
    if all_effect_estimates:
        ee_df = pd.DataFrame(all_effect_estimates)
        ee_df.to_csv(os.path.join(TABLES, "effect_estimates.csv"), index=False)
        print(f"Saved: effect_estimates.csv")

    # ── Figures ──
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    if all_rates_allcause:
        allcause_df = pd.concat(all_rates_allcause, ignore_index=True)

        for cycle in COHORTS:
            cycle_rates = allcause_df[allcause_df["cycle"] == cycle]

            for fib_def in cycle_rates["fibrosis_def"].unique():
                safe_def = fib_def.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(">=", "ge").replace("<", "lt")
                safe_cycle = cycle.replace("-", "_")

                for window in cycle_rates[cycle_rates["fibrosis_def"] == fib_def]["window"].unique():
                    plot_allcause_bars(
                        cycle_rates, cycle, window, fib_def,
                        f"allcause_{safe_cycle}_{window}m_{safe_def}.png"
                    )
                    plot_causegroup_bars(
                        cycle_rates, cycle, window, fib_def,
                        f"causegroup_{safe_cycle}_{window}m_{safe_def}.png"
                    )

        # Combined comparison at harmonized window
        for fib_def in allcause_df["fibrosis_def"].unique():
            safe_def = fib_def.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(">=", "ge").replace("<", "lt")
            plot_combined_comparison(
                allcause_df, WINDOW_H, fib_def,
                f"combined_{WINDOW_H}m_{safe_def}.png"
            )

    # ── Cohort summary ──
    print("\n" + "=" * 70)
    print("WRITING COHORT SUMMARY")
    print("=" * 70)
    write_cohort_summary(cards_df, allcause_df if all_rates_allcause else None,
                         pd.DataFrame(all_effect_estimates) if all_effect_estimates else None,
                         cohort_dfs)

    # ── Final output ──
    print("\n" + "=" * 70)
    print("DELIVERABLES")
    print("=" * 70)
    for root, dirs, files in os.walk(OUTPUTS):
        for f in sorted(files):
            print(f"  {os.path.join(root, f)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print_final_summary(allcause_df if all_rates_allcause else None,
                        pd.DataFrame(all_effect_estimates) if all_effect_estimates else None)


def write_cohort_summary(cards_df, allcause_df, ee_df, cohort_dfs):
    """Write outputs/cohort_summary.md."""
    lines = []
    lines.append("# NHANES Mortality-Fibrosis Analysis: Cohort Summary")
    lines.append("")
    lines.append("## Cohort Selection")
    lines.append("")
    lines.append("Three NHANES cohorts were selected to illustrate tradeoffs in follow-up time")
    lines.append("and cause-of-death detail available in the public-use Linked Mortality Files:")
    lines.append("")
    lines.append("| Cohort | Follow-up (max months) | % with >=36m | % with >=24m | UCOD detail | Elastography |")
    lines.append("|--------|----------------------|-------------|-------------|-------------|-------------|")
    lines.append("| 2007-2008 | ~159 | ~96% | ~98% | Full (10 groups) | No |")
    lines.append("| 2011-2012 | ~113 | ~97% | ~98% | Full (10 groups) | No |")
    lines.append("| 2017-2018 | ~37 | ~2% | ~51% | Coarsened (3 groups) | Yes (VCTE) |")
    lines.append("")
    lines.append("**Rationale:**")
    lines.append("- 2007-2008 provides the longest follow-up and full cause-of-death detail.")
    lines.append("- 2011-2012 also has full detail with slightly shorter follow-up.")
    lines.append("- 2017-2018 uniquely includes transient elastography (liver stiffness) but has")
    lines.append("  limited follow-up (censored Dec 31, 2019) and coarsened UCOD to only 3 groups")
    lines.append("  (heart disease, malignant neoplasms, all other).")
    lines.append("")

    lines.append("## Analysis Windows")
    lines.append("")
    lines.append("- **36-month window**: Used for 2007-2008 and 2011-2012 (>96% have >=36m).")
    lines.append("- **24-month harmonized window (H=24)**: Used for cross-cohort comparison.")
    lines.append("  For 2017-2018, only ~51% have >=24 months follow-up, so results reflect")
    lines.append("  the subset with sufficient observation time.")
    lines.append("")

    lines.append("## Fibrosis Definitions")
    lines.append("")
    lines.append("### A) FIB-4 (all cohorts)")
    lines.append("- FIB-4 = (age x AST) / (platelets x sqrt(ALT))")
    lines.append("- Fibrosis+ (advanced fibrosis proxy): FIB-4 >= 2.67")
    lines.append("- Fibrosis- (low fibrosis): FIB-4 < 1.30")
    lines.append("- Indeterminate (1.30-2.67): excluded from primary binary comparison")
    lines.append("")
    lines.append("### B) Liver Stiffness (2017-2018 only)")
    lines.append("- Two cutpoint sets applied to liver stiffness median (kPa):")
    lines.append(f"  - Castera/EASL: F0-F1 <{LSM_CUTPOINTS_A['significant']}, F2 {LSM_CUTPOINTS_A['significant']}-{LSM_CUTPOINTS_A['advanced']}, F3 {LSM_CUTPOINTS_A['advanced']}-{LSM_CUTPOINTS_A['cirrhosis']}, F4 >={LSM_CUTPOINTS_A['cirrhosis']}")
    lines.append(f"  - Eddowes/NAFLD: F0-F1 <{LSM_CUTPOINTS_B['significant']}, F2 {LSM_CUTPOINTS_B['significant']}-{LSM_CUTPOINTS_B['advanced']}, F3 {LSM_CUTPOINTS_B['advanced']}-{LSM_CUTPOINTS_B['cirrhosis']}, F4 >={LSM_CUTPOINTS_B['cirrhosis']}")
    lines.append("- Fibrosis+ = F3+F4, Fibrosis- = F0-F1 (F2 excluded as indeterminate)")
    lines.append("")

    lines.append("## UCOD Cause-of-Death Detail")
    lines.append("")
    lines.append("| UCOD Code | Label | 2007-2008 | 2011-2012 | 2017-2018 |")
    lines.append("|-----------|-------|-----------|-----------|-----------|")
    for code, label in UCOD_LABELS.items():
        avail_07 = "Yes"
        avail_11 = "Yes"
        avail_17 = "Yes" if code in [1, 2, 10] else "Suppressed"
        lines.append(f"| {code} | {label} | {avail_07} | {avail_11} | {avail_17} |")
    lines.append("")
    lines.append("Note: For 2015-2016 and 2017-2018, NCHS suppressed cause-group detail")
    lines.append("in the public-use LMF, reporting only heart disease (1), malignant")
    lines.append("neoplasms (2), and all other causes (10). This limits cause-specific")
    lines.append("mortality analysis for recent cohorts.")
    lines.append("")

    # Main findings
    lines.append("## Main Findings")
    lines.append("")

    if allcause_df is not None:
        lines.append("### All-Cause Mortality by Fibrosis Status")
        lines.append("")
        fib4_rates = allcause_df[allcause_df["fibrosis_def"].str.contains("FIB4")]
        if len(fib4_rates) > 0:
            lines.append("| Cohort | Window | Fibrosis Status | N | Deaths | Rate/1000 PY |")
            lines.append("|--------|--------|-----------------|---|--------|-------------|")
            for _, row in fib4_rates.iterrows():
                lines.append(f"| {row['cycle']} | {int(row['window'])}m | {row['fibrosis_status']} | "
                           f"{int(row['n'])} | {int(row['deaths'])} | {row['rate_per_1000py']} |")
            lines.append("")

    if ee_df is not None:
        lines.append("### Effect Estimates (FIB-4 based)")
        lines.append("")
        fib4_ee = ee_df[ee_df["fibrosis_def"].str.contains("FIB4")]
        if len(fib4_ee) > 0:
            lines.append("| Cohort | Window | Unadj RR (95% CI) | Poisson IRR unadj | Poisson IRR adj (age+sex) |")
            lines.append("|--------|--------|-------------------|-------------------|--------------------------|")
            for _, row in fib4_ee.iterrows():
                rr_str = f"{row.get('RR', 'NA')} ({row.get('RR_lower', 'NA')}-{row.get('RR_upper', 'NA')})"
                irr_u = f"{row.get('unadj_IRR', 'NA')} ({row.get('unadj_IRR_lower', 'NA')}-{row.get('unadj_IRR_upper', 'NA')})"
                irr_a = f"{row.get('adj_IRR', 'NA')} ({row.get('adj_IRR_lower', 'NA')}-{row.get('adj_IRR_upper', 'NA')})"
                lines.append(f"| {row['cycle']} | {int(row['window'])}m | {rr_str} | {irr_u} | {irr_a} |")
            lines.append("")

    lines.append("### Cause-Group Mortality (where available)")
    lines.append("")
    lines.append("See `outputs/tables/mortality_causegroup_by_cohort_window_fibrosisdef.csv` for full detail.")
    lines.append("Note: Many cause-group cells have <10 events and are flagged as UNSTABLE.")
    lines.append("For 2017-2018, only heart disease, cancer, and 'all other' are available.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("1. **Fibrosis is proxy-defined.** FIB-4 is a non-invasive biomarker index,")
    lines.append("   not a histological diagnosis. It has moderate sensitivity and specificity")
    lines.append("   for advanced fibrosis. LSM from VCTE is also a proxy with known overlap")
    lines.append("   between fibrosis stages.")
    lines.append("2. **Public-use COD coarsening.** UCOD_LEADING is suppressed to 3 groups for")
    lines.append("   2015-2016 and 2017-2018, preventing liver-specific cause-of-death analysis")
    lines.append("   in the cohort with elastography.")
    lines.append("3. **Short follow-up for 2017-2018.** With max ~37 months and only ~51% having")
    lines.append("   >=24 months, the 2017-2018 cohort has few deaths and limited power.")
    lines.append("4. **Small event counts.** Many subgroup analyses have <10 events, making")
    lines.append("   rate estimates unstable. Results should be interpreted with caution.")
    lines.append("5. **No survey weights in primary analysis.** Unweighted estimates are presented")
    lines.append("   for simplicity; survey-weighted results would better represent the US population.")
    lines.append("6. **Perturbed variables.** The public-use LMF includes perturbed death dates")
    lines.append("   for disclosure avoidance, which may slightly affect person-time calculations.")
    lines.append("7. **Confounding.** FIB-4 increases with age by construction (age is in the")
    lines.append("   numerator), so crude comparisons partly reflect age differences. The age-adjusted")
    lines.append("   Poisson models partially address this.")
    lines.append("")

    with open(os.path.join(OUTPUTS, "cohort_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {os.path.join(OUTPUTS, 'cohort_summary.md')}")


def print_final_summary(allcause_df, ee_df):
    """Print final bullet summary."""
    print("\nKEY FINDINGS:")
    print("-" * 50)

    if allcause_df is not None and ee_df is not None:
        fib4_ee = ee_df[ee_df["fibrosis_def"].str.contains("FIB4")]
        for _, row in fib4_ee.iterrows():
            rr = row.get("RR", "NA")
            adj_irr = row.get("adj_IRR", "NA")
            d_yes = row.get("d_fib_yes", "?")
            d_no = row.get("d_fib_no", "?")
            print(f"  - {row['cycle']} ({int(row['window'])}m): FIB4-based fibrosis associated with "
                  f"RR={rr} for all-cause death ({d_yes} vs {d_no} deaths); "
                  f"age/sex-adjusted IRR={adj_irr}")

    print("")
    print("CROSS-COHORT COMPARISON:")
    print("  - Earlier cohorts (2007-2008, 2011-2012) have longer follow-up and")
    print("    full cause-of-death detail (10 UCOD groups)")
    print("  - 2017-2018 has elastography but short follow-up and coarsened UCOD")
    print("    (only 3 groups), limiting cause-specific analysis")
    print("  - The harmonized 24-month window allows direct comparison but reduces")
    print("    event counts for earlier cohorts compared to their full 36-month window")


if __name__ == "__main__":
    main()
