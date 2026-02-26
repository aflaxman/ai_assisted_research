"""
Step 0: Document what's possible before coding estimates.

Downloads the Public-Use Linked Mortality Files for candidate NHANES cohorts,
examines follow-up time distributions and UCOD_LEADING cause-of-death codes,
and selects 3 cohorts for the analysis.

Public-Use LMF documentation:
  https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

Key LMF variables:
  SEQN        - Respondent sequence number (merge key)
  ELIGSTAT    - Mortality eligibility status (1=eligible for linkage)
  MORTSTAT    - Final mortality status (0=assumed alive, 1=assumed deceased)
  PERMTH_EXM  - Person-months from MEC exam to death or censoring
  PERMTH_INT  - Person-months from interview to death or censoring
  UCOD_LEADING - Underlying cause of death recode (leading cause groups)

UCOD_LEADING codes (when available):
  1  = Diseases of heart
  2  = Malignant neoplasms
  3  = Chronic lower respiratory diseases
  4  = Accidents (unintentional injuries)
  5  = Cerebrovascular diseases
  6  = Alzheimer's disease
  7  = Diabetes mellitus
  8  = Influenza and pneumonia
  9  = Nephritis, nephrotic syndrome, nephrosis
  10 = All other causes (residual)

Note: In some cohort files the UCOD recode may be more coarsened or have
fewer distinct values populated.
"""

import os
import requests
import hashlib
import pandas as pd
import pyreadstat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_LMF = os.path.join(BASE_DIR, "data", "raw", "lmf")
DERIVED = os.path.join(BASE_DIR, "data", "derived")
os.makedirs(RAW_LMF, exist_ok=True)
os.makedirs(DERIVED, exist_ok=True)

# Public-use LMF download URLs
# The 2019 public-use linked mortality files cover follow-up through 12/31/2019.
# Updated files (2022 release) extend through 12/31/2019 but with updated NDI matching.
# NOTE: NCHS released updated public-use LMF in 2022-2023 with follow-up through Dec 31, 2019.
# The URLs follow a consistent pattern on the CDC data-linkage site.

LMF_FILES = {
    "2007-2008": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "2009-2010": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "2011-2012": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "2013-2014": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "2015-2016": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "2017-2018": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2017_2018_MORT_2019_PUBLIC.dat",
}


def download_file(url, dest_path):
    """Download a file with caching."""
    if os.path.exists(dest_path):
        print(f"  Cached: {dest_path}")
        return
    print(f"  Downloading: {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    md5 = hashlib.md5(resp.content).hexdigest()
    print(f"  Saved: {dest_path} (MD5: {md5})")


def _safe_int(s):
    """Parse a string to int, returning None for missing/invalid."""
    s = s.strip().replace(".", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _safe_float(s):
    """Parse a string to float, returning None for missing/invalid."""
    s = s.strip().replace(".", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_lmf_dat(filepath):
    """
    Parse the fixed-width public-use LMF .dat file (2019 release).

    Verified column layout (0-indexed Python slicing):
      Col  1-14 (slice  0:14) PUBLICID / SEQN
      Col 15    (slice 14:15) ELIGSTAT
      Col 16    (slice 15:16) MORTSTAT
      Col 17-19 (slice 16:19) UCOD_LEADING (zero-padded, e.g. "010" = 10)
      Col 20    (slice 19:20) DIABETES
      Col 21    (slice 20:21) HYPERTEN
      Col 22-42 (padding)
      Col 43-45 (slice 42:45) PERMTH_INT
      Col 46-48 (slice 45:48) PERMTH_EXM

    Lines are padded to 48 chars; trailing spaces may be stripped in file.
    """
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            # Pad to 48 chars to handle stripped trailing spaces
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
    df = pd.DataFrame(rows)
    return df


def main():
    print("=" * 70)
    print("STEP 0: Document data availability for NHANES-LMF cohort selection")
    print("=" * 70)

    summaries = {}

    for cycle, url in LMF_FILES.items():
        print(f"\n--- {cycle} ---")
        fname = os.path.basename(url)
        dest = os.path.join(RAW_LMF, fname)
        download_file(url, dest)

        df = parse_lmf_dat(dest)
        print(f"  Total records: {len(df)}")

        # Restrict to linkage-eligible
        elig = df[df["ELIGSTAT"] == 1].copy()
        print(f"  Eligible (ELIGSTAT=1): {len(elig)}")

        # Follow-up from MEC exam
        fu = elig["PERMTH_EXM"].dropna()
        n_dead = (elig["MORTSTAT"] == 1).sum()

        pct_36 = (fu >= 36).mean() * 100 if len(fu) > 0 else 0
        pct_24 = (fu >= 24).mean() * 100 if len(fu) > 0 else 0
        pct_12 = (fu >= 12).mean() * 100 if len(fu) > 0 else 0

        print(f"  Deaths (MORTSTAT=1): {n_dead}")
        print(f"  PERMTH_EXM: median={fu.median():.1f}, max={fu.max():.1f}")
        print(f"  % with >=12 months: {pct_12:.1f}%")
        print(f"  % with >=24 months: {pct_24:.1f}%")
        print(f"  % with >=36 months: {pct_36:.1f}%")

        # UCOD_LEADING among deaths
        deaths = elig[elig["MORTSTAT"] == 1]
        ucod_counts = deaths["UCOD_LEADING"].value_counts().sort_index()
        print(f"  UCOD_LEADING distribution among {len(deaths)} deaths:")
        ucod_labels = {
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
        for code, count in ucod_counts.items():
            label = ucod_labels.get(code, f"Unknown({code})")
            print(f"    {code:>2} ({label}): {count}")

        ucod_missing = deaths["UCOD_LEADING"].isna().sum()
        if ucod_missing > 0:
            print(f"    Missing UCOD: {ucod_missing}")

        summaries[cycle] = {
            "n_total": len(df),
            "n_eligible": len(elig),
            "n_deaths": n_dead,
            "max_fu_months": fu.max() if len(fu) > 0 else 0,
            "median_fu_months": fu.median() if len(fu) > 0 else 0,
            "pct_ge_12m": pct_12,
            "pct_ge_24m": pct_24,
            "pct_ge_36m": pct_36,
            "ucod_codes": sorted(ucod_counts.index.tolist()),
            "ucod_n_missing": ucod_missing,
            "n_deaths_with_ucod": len(deaths) - ucod_missing,
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Candidate cohorts")
    print("=" * 70)
    summary_df = pd.DataFrame(summaries).T
    summary_df.index.name = "Cycle"
    print(summary_df[["n_eligible", "n_deaths", "max_fu_months",
                       "median_fu_months", "pct_ge_36m", "pct_ge_24m",
                       "ucod_codes"]].to_string())

    # Save for downstream use
    summary_df.to_csv(os.path.join(DERIVED, "lmf_cohort_availability.csv"))
    print(f"\nSaved: {os.path.join(DERIVED, 'lmf_cohort_availability.csv')}")

    print("\n" + "=" * 70)
    print("COHORT SELECTION RATIONALE")
    print("=" * 70)
    print("""
Selected 3 cohorts:

1) 2007-2008: Longest follow-up (up to ~144 months), full UCOD detail,
   all participants have >=36 months follow-up. FIB-4 computable from labs.

2) 2011-2012: Moderate follow-up (~96 months max), full UCOD detail,
   all have >=36 months. FIB-4 computable. Allows comparison with 2007-2008.

3) 2017-2018: Has transient elastography (VCTE/liver stiffness),
   but LIMITED follow-up (max ~24 months to Dec 2019 censoring).
   UCOD may be more restricted. Enables LSM-based fibrosis staging.

Harmonized window H:
   Because the 2017-2018 cohort's max follow-up may be ~24 months,
   we set H based on what 2017-2018 can support. If max is ~24 months,
   H = 24 months. If less, H = max practical window.
""")


if __name__ == "__main__":
    main()
