"""Build a strike-by-strike record of the US maritime counter-narcotics campaign.

The campaign (Operation Southern Spear) began with a strike on 2 September 2025.
Wikipedia maintains a sourced, strike-level table compiled from the New York Times
and ABC News trackers; that table is the most complete public event list, so this
script parses it into a tidy CSV.

Usage:
    python -P fetch_strikes.py                 # refetch from Wikipedia
    python -P fetch_strikes.py --offline       # parse the committed snapshot
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import pandas as pd

PAGE = "United_States_strikes_on_alleged_drug_traffickers_during_Operation_Southern_Spear"
RAW_URL = f"https://en.wikipedia.org/w/index.php?title={PAGE}&action=raw"
UA = "ai-assisted-research/0.1 (research use; abie@ihme.washington.edu)"

HERE = Path(__file__).parent
SNAPSHOT = HERE / "data" / "wikitext_snapshot.txt"
OUT = HERE / "data" / "strikes.csv"


def fetch_wikitext(offline: bool = False) -> str:
    if offline:
        return SNAPSHOT.read_text(encoding="utf-8")
    req = urllib.request.Request(RAW_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    SNAPSHOT.write_text(text, encoding="utf-8")
    return text


def _strip_markup(cell: str) -> str:
    """Reduce a wikitable cell to plain text."""
    cell = re.sub(r"<ref[^>]*/>", "", cell)
    cell = re.sub(r"<ref.*?</ref>", "", cell, flags=re.S)
    cell = re.sub(r"<!--.*?-->", "", cell, flags=re.S)
    cell = re.sub(r"\{\{[^{}]*\}\}", "", cell)
    cell = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", cell)
    cell = re.sub(r"<[^>]+>", " ", cell)
    cell = cell.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", cell).strip()


def _int_or_zero(cell: str) -> int:
    """Counts are sometimes written as '>1' or left blank."""
    text = _strip_markup(cell)
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


def _strike_number_range(cell: str) -> tuple[int, int]:
    """A row can bundle same-day strikes, numbered e.g. '11-13'."""
    text = _strip_markup(cell).replace("–", "-").replace("—", "-")
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if not nums:
        return 0, 0
    return nums[0], nums[-1]


def _region(cell: str) -> str:
    text = _strip_markup(cell).lower()
    if "pacific" in text:
        return "Eastern Pacific"
    if "caribbean" in text:
        return "Caribbean"
    return "Unknown"


def parse_strike_table(wikitext: str) -> pd.DataFrame:
    """Extract the 'Strikes by the United States military' table."""
    start = wikitext.index('|+ Strikes by the United States military')
    table_open = wikitext.rindex("{|", 0, start)
    table_close = wikitext.index("\n|}", table_open)
    table = wikitext[table_open:table_close]

    rows = []
    # Rows are separated by "|-"; the date lives in a `scope="row"` header cell
    # carrying a machine-readable data-sort-value.
    for chunk in re.split(r"\n\|-", table)[1:]:
        sort_value = re.search(r'data-sort-value=\s*"?(\d{4}-\d{1,2}-\d{1,2})', chunk)
        if not sort_value:
            continue
        # Split the row into cells on newline-leading | or ! delimiters.
        cells = re.split(r"\n[|!]", chunk)
        cells = [c for c in cells if c.strip()]
        if len(cells) < 5:
            continue
        first, last = _strike_number_range(cells[0])
        rows.append(
            {
                "strike_no_first": first,
                "strike_no_last": last,
                "n_strikes": last - first + 1,
                "date": pd.Timestamp(sort_value.group(1)),
                "date_text": _strip_markup(cells[1].split("|", 1)[-1]),
                "region": _region(cells[2]),
                "vessels_struck": max(1, _int_or_zero(cells[3])),
                "killed": _int_or_zero(cells[4]),
                "captured": _int_or_zero(cells[5]) if len(cells) > 5 else 0,
                "missing": _int_or_zero(cells[6]) if len(cells) > 6 else 0,
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("parsed zero strikes; the source table layout changed")
    return df


def monthly_dose(strikes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strikes to the monthly grid the outcome models use."""
    by_month = (
        strikes.assign(month=strikes["date"].values.astype("datetime64[M]"))
        .groupby("month")
        .agg(
            strike_events=("n_strikes", "sum"),
            vessels=("vessels_struck", "sum"),
            killed=("killed", "sum"),
        )
    )
    full = pd.date_range(by_month.index.min(), by_month.index.max(), freq="MS")
    by_month = by_month.reindex(full, fill_value=0)
    by_month.index.name = "month"
    by_month["cum_strike_events"] = by_month["strike_events"].cumsum()
    by_month["cum_vessels"] = by_month["vessels"].cumsum()
    return by_month.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="parse the committed wikitext snapshot instead of refetching")
    args = ap.parse_args()

    strikes = parse_strike_table(fetch_wikitext(args.offline))
    strikes.to_csv(OUT, index=False)
    dose = monthly_dose(strikes)
    dose.to_csv(HERE / "data" / "strikes_monthly.csv", index=False)

    print(f"{strikes['n_strikes'].sum()} numbered strikes on {len(strikes)} dates, "
          f"{strikes['date'].min():%Y-%m-%d} to {strikes['date'].max():%Y-%m-%d}")
    print(f"vessels struck: {strikes['vessels_struck'].sum()}   "
          f"reported killed: {strikes['killed'].sum()}   "
          f"captured: {strikes['captured'].sum()}")
    print("\nby region:")
    print(strikes.groupby("region")[["vessels_struck", "killed"]].sum().to_string())
    print("\nmonthly dose:")
    print(dose.to_string(index=False))


if __name__ == "__main__":
    main()
