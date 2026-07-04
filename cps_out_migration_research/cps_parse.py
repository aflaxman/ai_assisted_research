"""Parse CPS Basic Monthly public-use fixed-width files into tidy DataFrames.

Column positions come from the 2024 Basic CPS Public Use Record Layout
(`layout2024.txt`). Positions in the layout are 1-indexed inclusive; pandas
`read_fwf` wants 0-indexed half-open (start-1, end) tuples.

Only the columns needed for month-to-month panel linkage and a nativity
breakdown are read, which keeps parsing fast on the ~120k-row national files.
"""

from __future__ import annotations

import gzip
import zipfile

import pandas as pd

# name -> (start_1indexed, end_1indexed_inclusive)
LAYOUT = {
    "HRHHID": (1, 15),      # household identifier part 1
    "HRMONTH": (16, 17),    # month of interview
    "HRYEAR4": (18, 21),    # year of interview
    "HUTYPEA": (41, 42),    # Type A reason: 1 NOH,2 temp abs,3 REFUSED,5 unable to locate
    "HUTYPB": (43, 44),     # Type B reason (vacancy states)
    "HUTYPC": (45, 46),     # Type C reason (demolished etc.)
    "HRINTSTA": (57, 58),   # interview status: 1=interview, 2=Type A, 3=B, 4=C
    "HRMIS": (63, 64),      # month-in-sample 1-8
    "HRLONGLK": (69, 70),   # longitudinal link: 0=no link, 2=MIS2-4/6-8, 3=MIS5
    "HRHHID2": (71, 75),    # household identifier part 2
    "GESTFIPS": (93, 94),   # state FIPS (53 = Washington)
    "GTCO": (101, 103),     # county FIPS (0 if not identified)
    "PERRP": (118, 119),    # relationship to reference person
    "PRTAGE": (122, 123),   # age (topcoded 85)
    "PEMARITL": (125, 126), # marital status: 1 married sp present, 2 married sp ABSENT
    "PESPOUSE": (127, 128), # line number of spouse (-1 none)
    "PESEX": (129, 130),    # sex 1=male 2=female
    "PTDTRACE": (139, 140), # race recode
    "PULINENO": (147, 148), # person line number (stable within household)
    "PRPERTYP": (161, 162), # 1=child,2=adult civilian,3=adult armed forces
    "PENATVTY": (163, 165), # country of birth
    "PEMNTVTY": (166, 168), # mother's country of birth
    "PEFNTVTY": (169, 171), # father's country of birth
    "PRCITSHP": (172, 173), # citizenship: 4,5 = foreign born
    "PRINUYER": (176, 177), # immigrant year-of-entry recode
    "PEMLR": (180, 181),    # monthly labor force recode
    "PWSSWGT": (613, 622),  # final person weight (4 implied decimals)
    "PWCMPWGT": (846, 855), # composited final person weight (4 implied decimals)
}

WEIGHT_COLS = ("PWSSWGT", "PWCMPWGT")
ID_COLS = ("HRHHID", "HRHHID2")  # keep as strings to preserve leading zeros


def colspecs_and_names():
    names = list(LAYOUT)
    colspecs = [(s - 1, e) for (s, e) in LAYOUT.values()]
    return colspecs, names


def _open_data_stream(path: str):
    """Return a decompressed binary stream for a CPS data file.

    The Census distributes these as ``*.dat.gz`` but the bytes are actually a
    ZIP archive (magic ``PK``) wrapping a single ``*.dat`` member. Handle both
    real gzip and the mislabeled zip, plus plain ``.dat``.
    """
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":
        zf = zipfile.ZipFile(path)
        member = zf.namelist()[0]
        return zf.open(member)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return open(path, "rb")


def read_month(path: str, state_fips: int | None = None) -> pd.DataFrame:
    """Read one monthly CPS basic file (.dat.gz/.zip/.dat).

    If ``state_fips`` is given, filter to that state. Weights are converted
    from their integer storage (4 implied decimals) to floats.
    """
    colspecs, names = colspecs_and_names()
    # Read id columns as strings; everything else numeric.
    dtype = {c: str for c in ID_COLS}
    stream = _open_data_stream(path)
    try:
        df = pd.read_fwf(
            stream,
            colspecs=colspecs,
            names=names,
            compression=None,
            dtype=dtype,
        )
    finally:
        stream.close()
    # Numeric coercion for non-id columns.
    for c in names:
        if c in ID_COLS:
            df[c] = df[c].astype(str).str.strip()
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Weights: 4 implied decimal places.
    for w in WEIGHT_COLS:
        df[w] = df[w] / 10000.0
    if state_fips is not None:
        df = df[df["GESTFIPS"] == state_fips].copy()
    return df


def household_key(df: pd.DataFrame) -> pd.Series:
    """Unique household key = HRHHID + '_' + HRHHID2 (the CPS standard)."""
    return df["HRHHID"].str.strip() + "_" + df["HRHHID2"].str.strip()


if __name__ == "__main__":
    import sys

    p = sys.argv[1]
    d = read_month(p)
    print("rows:", len(d))
    print("national PWSSWGT sum (millions):", round(d["PWSSWGT"].sum() / 1e6, 1))
    print("HRINTSTA value counts:\n", d["HRINTSTA"].value_counts(dropna=False))
    wa = d[d["GESTFIPS"] == 53]
    print("WA rows:", len(wa), "| WA interviewed persons:",
          int((wa["HRINTSTA"] == 1).sum()))
