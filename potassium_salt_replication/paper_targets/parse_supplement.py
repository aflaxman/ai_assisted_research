"""Parse the country tables in the supplement of Huang et al. (2026) into CSV.

Supplementary Tables 3 and 4 give, for each of 193 countries, the baseline
burden and the burden averted under each scenario.  These are the numbers this
replication is compared against, so they are parsed mechanically rather than
typed in.

The supplement is open access (CC BY 4.0):
https://static-content.springer.com/esm/art%3A10.1038%2Fs44360-026-00216-9/MediaObjects/44360_2026_216_MOESM1_ESM.pdf

Usage
-----
    python paper_targets/parse_supplement.py path/to/44360_2026_216_MOESM1_ESM.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "paper"

# Each sub-table starts on a new page under one of these headings.  The page
# numbers in the supplement's own table of contents are out of date, so the
# parser finds the headings instead of trusting them.
TABLE3_HEADINGS = {
    "a. Prevented CVD deaths": "cvd_deaths",
    "b. Prevented CVD new cases": "cvd_new_cases",
    "c. Prevented CVD DALYs": "cvd_dalys",
    "d. Prevented CKD deaths": "ckd_deaths",
    "e. Prevented CKD new cases": "ckd_new_cases",
    "f. Prevented CKD DALYs": "ckd_dalys",
}
TABLE4_HEADINGS = {
    "a. Prevented CVD deaths in patients with CKD": "prevented_cvd_deaths_in_ckd",
    "b. Additional deaths in patients with CKD": "hyperkalaemia_deaths_in_ckd",
    "c. Net effect in patients with CKD": "net_deaths_averted_in_ckd",
}
SCENARIOS = [
    "all_salt",
    "discretionary",
    "non_discretionary",
    "discretionary_diagnosed_hypertension",
    "discretionary_treated_hypertension",
]

NUMBER = r"[-−]?\d+(?:\.\d+)?"
CELL = re.compile(rf"^({NUMBER})\(({NUMBER})-({NUMBER})\)$")


def parse_cell(cell: str | None) -> tuple[float, float, float] | None:
    """'1358 (1082-1637)' -> (1358.0, 1082.0, 1637.0); None if not a value cell."""
    if not cell:
        return None
    text = re.sub(r"\s+", "", cell).replace("−", "-").replace("–", "-")
    match = CELL.match(text)
    if not match:
        return None
    median, lower, upper = (float(g) for g in match.groups())
    return median, lower, upper


def sections(pdf: pdfplumber.PDF) -> dict[int, tuple[str, str]]:
    """Map each pdf page index to (table, section) by tracking the headings."""
    page_section: dict[int, tuple[str, str]] = {}
    table, section = "", ""
    for index, page in enumerate(pdf.pages):
        lines = [line.strip() for line in (page.extract_text() or "").splitlines()]
        if any(line.startswith("Supplementary table 3.") for line in lines) and index > 5:
            table = "table3"
        if any(line.startswith("Supplementary table 4.") for line in lines) and index > 5:
            table = "table4"
        if any(line.startswith("Supplementary figure 1.") for line in lines) and index > 5:
            table, section = "", ""
        headings = TABLE3_HEADINGS if table == "table3" else TABLE4_HEADINGS
        for line in lines:
            if table and line in headings:
                section = headings[line]
        if table and section:
            page_section[index] = (table, section)
    return page_section


def iter_rows(pdf: pdfplumber.PDF, table_name: str):
    for index, (table, section) in sections(pdf).items():
        if table != table_name:
            continue
        for extracted in pdf.pages[index].extract_tables():
            for row in extracted:
                yield section, index, [(c or "").replace("\n", " ").strip() for c in row]


def parse_table3(pdf: pdfplumber.PDF) -> pd.DataFrame:
    records = []
    for outcome, page, row in iter_rows(pdf, "table3"):
        if len(row) < 4 or row[1] not in ("Female", "Male"):
            continue
        baseline = parse_cell(row[2])
        if baseline is None:
            continue
        is_global = row[0].lower().startswith("global")
        scale = 1e6 if is_global else 1.0
        record = {
            "outcome": outcome,
            "country": "Global" if is_global else row[0],
            "sex": row[1],
            "page": page,
            "baseline": baseline[0] * scale,
            "baseline_lower": baseline[1] * scale,
            "baseline_upper": baseline[2] * scale,
        }
        cells = row[3:]
        for k, scenario in enumerate(SCENARIOS):
            averted = parse_cell(cells[2 * k]) if 2 * k < len(cells) else None
            percent = parse_cell(cells[2 * k + 1]) if 2 * k + 1 < len(cells) else None
            for name, value, factor in (("averted", averted, scale), ("percent", percent, 1.0)):
                for suffix, j in (("", 0), ("_lower", 1), ("_upper", 2)):
                    record[f"{scenario}__{name}{suffix}"] = (
                        value[j] * factor if value is not None else float("nan")
                    )
        records.append(record)
    return pd.DataFrame.from_records(records)


def parse_table4(pdf: pdfplumber.PDF) -> pd.DataFrame:
    records = []
    for quantity, page, row in iter_rows(pdf, "table4"):
        values = [parse_cell(c) for c in row[1:4]] if len(row) >= 4 else []
        if len(values) != 3 or any(v is None for v in values):
            continue
        is_global = row[0].lower().startswith("global")
        scale = 1e6 if is_global and "million" in row[0].lower() else 1.0
        record = {"quantity": quantity, "country": "Global" if is_global else row[0],
                  "page": page}
        for scenario, value in zip(SCENARIOS[:3], values):
            for suffix, j in (("", 0), ("_lower", 1), ("_upper", 2)):
                record[f"{scenario}{suffix}"] = value[j] * scale
        records.append(record)
    return pd.DataFrame.from_records(records)


def main(pdf_path: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        table3 = parse_table3(pdf)
        table4 = parse_table4(pdf)
    table3.to_csv(OUT_DIR / "supp_table3_country_results.csv", index=False)
    table4.to_csv(OUT_DIR / "supp_table4_ckd_results.csv", index=False)
    print(table3.groupby("outcome").agg(rows=("country", "size"), countries=("country", "nunique")))
    print(table4.groupby("quantity").agg(rows=("country", "size"), countries=("country", "nunique")))


if __name__ == "__main__":
    main(sys.argv[1])
