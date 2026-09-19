"""Loader and helpers for the 2024 Nigeria Verbal and Social Autopsy (NVASA) microdata.

The data live on IHME's J: drive and are read-only. Nothing here writes to that
location. Set the ``NVASA_DIR`` environment variable to point at a different copy.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

FILE = "NGA_VASA8_2024_VA_NGVA8JFL_Y2026M09D14.DTA"

CANDIDATE_DIRS = [
    os.environ.get("NVASA_DIR"),
    "/home/j/DATA/DHS_PROG_VASA/NGA/2024",  # IHME cluster
    "/snfs1/DATA/DHS_PROG_VASA/NGA/2024",  # same mount, alternate path
    "/mnt/j/DATA/DHS_PROG_VASA/NGA/2024",  # WSL with J: mounted
    "J:/DATA/DHS_PROG_VASA/NGA/2024",  # Windows
]

# ICD-11 chapter 2 block for malignant neoplasms of breast (2C60-2C6Z).
BREAST_CANCER_ICD_PREFIX = "2C6"

# Letter codes for A4255 "main problems accessing care" (adult questionnaire).
PROBLEM_CODES = {
    "A": "Did not think sick enough to need care",
    "B": "No one available to go with her",
    "C": "Too much time away from duties",
    "D": "Someone else had to decide",
    "E": "Too far to travel",
    "F": "No transportation available",
    "G": "Cost of transportation",
    "H": "Cost of health care",
    "I": "Other cost",
    "J": "Not satisfied with available care",
    "K": "Problem required traditional care",
    "L": "Thought too sick to travel",
    "M": "Thought she would die despite care",
    "N": "Late at night, transport/provider unavailable",
    "O": "Fear of catching other diseases",
    "P": "Fear of exposure to male provider",
    "Q": "Other",
    "R": "Don't know",
}

# Provider-type columns for the up-to-nine care actions in the social autopsy.
PROVIDER_COLS = [f"a4252_2_{i}" for i in range(1, 10)]


def find_data_dir() -> Path:
    for d in CANDIDATE_DIRS:
        if d and (Path(d) / FILE).exists():
            return Path(d)
    raise FileNotFoundError(
        f"Could not find {FILE}. Set NVASA_DIR to the directory that contains it."
    )


def load(path: str | Path | None = None):
    """Read the Stata file. Returns (DataFrame, pyreadstat metadata).

    The file is Latin-1 encoded; the default UTF-8 read fails partway through.
    """
    path = Path(path) if path else find_data_dir() / FILE
    df, meta = pyreadstat.read_dta(str(path), apply_value_formats=False, encoding="LATIN1")
    return df, meta


class Labels:
    """Thin wrapper over pyreadstat metadata for variable and value labels."""

    def __init__(self, meta):
        self.var = meta.column_names_to_labels
        self.val = meta.variable_value_labels

    def label(self, var: str) -> str:
        return self.var.get(var, var)

    def decode(self, df: pd.DataFrame, var: str, na: str = "<NA>") -> pd.Series:
        m = self.val.get(var, {})
        return df[var].map(lambda x: m.get(x, x)).fillna(na)

    def tab(self, df: pd.DataFrame, var: str, weight: pd.Series | None = None) -> pd.DataFrame:
        """Frequency table with decoded labels; adds a weighted count if given."""
        s = self.decode(df, var)
        out = s.value_counts(dropna=False).rename("n").to_frame()
        if weight is not None:
            out["weighted_n"] = weight.groupby(s).sum().round(1)
        out.index.name = f"{var}: {self.label(var)}"
        return out


def weights(df: pd.DataFrame) -> pd.Series:
    """Survey weight, rescaled from 6 implied decimals. Mean is about 1.0."""
    return df["vaweight"] / 1e6


def adults(df: pd.DataFrame) -> pd.DataFrame:
    """Adult module: deaths of women aged 12-49 (questionnaire type 3)."""
    return df[pd.to_numeric(df["va1609t"], errors="coerce") == 3].copy()


def analytic_denominator(df: pd.DataFrame, age_range=(15, 49)) -> pd.Series:
    """Official NVASA denominator for women's deaths (see DENOMINATOR_DEFINITIONS.docx).

    NDHS-sampled households, death within 60 months of the household interview,
    adult questionnaire, age at death in range. Yields N=607 for 15-49, 654 for 12-49.
    """
    months_since = pd.to_numeric(df["vndhsc"], errors="coerce") - pd.to_numeric(
        df["va1606c"], errors="coerce"
    )
    age = pd.to_numeric(df["vadagen"], errors="coerce")
    return (
        (df["nsamp"] == 1)
        & months_since.between(0, 59)
        & (pd.to_numeric(df["va1609t"], errors="coerce") == 3)
        & age.between(*age_range)
    )


def flag_breast_cancer(df: pd.DataFrame) -> pd.Series:
    """True where the physician-certified ICD-11 cause is malignant neoplasm of breast."""
    return df["vicd"].astype(str).str.startswith(BREAST_CANCER_ICD_PREFIX)


def duration_months(df: pd.DataFrame, unit_col: str, num_col: str) -> pd.Series:
    """Convert a VASA duration pair (unit 1=days, 2=months, 3=years) to months.

    Codes 9/99/999 mean "don't know" and become NaN.
    """
    unit = pd.to_numeric(df[unit_col], errors="coerce")
    num = pd.to_numeric(df[num_col], errors="coerce")
    num = num.where(~num.isin([99, 999]))
    factor = unit.map({1: 1 / 30.4, 2: 1.0, 3: 12.0})
    return num * factor


def provider_sequence(df: pd.DataFrame, labels: Labels) -> pd.Series:
    """List of decoded provider types, in order, for each death's care actions."""
    m = labels.val.get(PROVIDER_COLS[0], {})
    return df[PROVIDER_COLS].apply(
        lambda row: [m.get(x, str(x)) for x in row if pd.notna(x)], axis=1
    )


def count_letters(series: pd.Series, codes: dict[str, str]) -> pd.Series:
    """Count multi-response letter codes (e.g. A4255) and decode them."""
    c = Counter(ch for s in series.dropna().astype(str) for ch in s.strip() if ch.isalpha())
    out = pd.Series(c, dtype=int).sort_values(ascending=False)
    out.index = [f"{k}: {codes.get(k, '?')}" for k in out.index]
    return out


def weighted_share(df: pd.DataFrame, flag: pd.Series, w: pd.Series) -> float:
    """Weighted percentage of rows where flag is True."""
    return float(100 * w[flag].sum() / w.sum()) if w.sum() else np.nan


# ---------------------------------------------------------------------------
# Statistical disclosure control
# ---------------------------------------------------------------------------

MIN_CELL = 5  # unweighted counts below this are never shown in committed output


def redact(table, min_cell: int = MIN_CELL, count_cols=None, other_label="Other (each <{k})"):
    """Suppress small unweighted cells before a table is displayed or committed.

    - Series of counts: categories with 0 < n < min_cell are pooled into one
      "Other" row (so no single small category is visible). If pooling would
      leave a single small category exposed, it is masked as "<k" instead.
    - DataFrame: every integer-like count column (or those named in
      ``count_cols``) has cells with 0 < n < min_cell replaced by "<k". Any
      other column in the same row (percentages, weighted counts) is masked
      too, because it would reveal the count.
    Zero cells are left as 0.
    """
    k = min_cell
    if isinstance(table, pd.Series):
        s = table.copy()
        small = (s > 0) & (s < k)
        if small.sum() == 0:
            return s
        if small.sum() == 1:
            s = s.astype(object)
            s[small] = f"<{k}"
            return s
        pooled = s[small].sum()
        out = s[~small].astype(object)
        out[other_label.format(k=k)] = int(pooled) if pooled >= k else f"<{k}"
        return out

    df = table.copy().astype(object)
    if count_cols is None:
        count_cols = [
            c for c in table.columns
            if pd.api.types.is_numeric_dtype(table[c])
            and np.allclose(table[c].dropna(), table[c].dropna().round())
            and not str(c).lower().endswith(("pct", "%", "rate", "ratio"))
        ]
    if not count_cols:
        return df
    small_rows = pd.Series(False, index=df.index)
    for c in count_cols:
        v = pd.to_numeric(table[c], errors="coerce")
        small = (v > 0) & (v < k)
        df.loc[small, c] = f"<{k}"
        small_rows |= small
    other_cols = [c for c in df.columns if c not in count_cols]
    for c in other_cols:
        df.loc[small_rows, c] = "-"
    return df


def redact_crosstab(ct: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Mask small cells in a two-way count table.

    Primary suppression masks cells with 0 < n < min_cell. Secondary
    suppression then masks the smallest remaining cell in any row or column
    that has exactly one masked cell, so a masked value cannot be recovered
    from its row or column total. Margins are dropped.
    """
    ct = ct.copy()
    for ax in (0, 1):
        if "All" in ct.axes[ax]:
            ct = ct.drop("All", axis=ax)
    v = ct.apply(pd.to_numeric, errors="coerce").astype(float)
    masked = (v > 0) & (v < min_cell)
    for _ in range(2):  # a couple of passes settle row/column interactions
        for ax in (0, 1):
            m = masked if ax == 1 else masked.T
            vals = v if ax == 1 else v.T
            for idx in m.index:
                row_mask = m.loc[idx]
                if row_mask.sum() == 1 and (~row_mask).sum() >= 1:
                    candidates = vals.loc[idx][~row_mask]
                    candidates = candidates[candidates > 0]
                    if len(candidates):
                        j = candidates.idxmin()
                        if ax == 1:
                            masked.loc[idx, j] = True
                        else:
                            masked.loc[j, idx] = True
    out = ct.astype(object).mask(masked, f"<{min_cell}")
    return out


def redact_bins(values: pd.Series, edges: list[float], min_cell: int = MIN_CELL):
    """Histogram counts on `edges`, merging bins from the top down until every
    non-empty bin holds at least `min_cell` observations. Returns (counts, edges)."""
    x = pd.Series(values).dropna()
    edges = list(edges)
    while True:
        counts, _ = np.histogram(x, bins=edges)
        small = [i for i, c in enumerate(counts) if 0 < c < min_cell]
        if not small or len(edges) <= 2:
            return counts, edges
        i = small[-1]
        # merge the small bin with its neighbour (upper neighbour if it exists)
        drop = i + 1 if i + 1 < len(edges) - 1 else i
        edges.pop(drop)
