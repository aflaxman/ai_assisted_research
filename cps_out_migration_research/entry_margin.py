"""Entry-margin check for the fear thermometer.

Within-panel attrition measures who *leaves* a continuing panel. But
avoidance can also operate on *first contact*: eligible households never
join. This script compares the (weighted) foreign-born share of interviewed
adults across rotation groups each month:

  mis1    — fresh panels, first interview this month
  mis2_4  — recruited 1-3 months ago
  mis5    — returning after the 8-month rest
  mis6_8  — old panels, recruited ~15-19 months ago

Old panels' nativity composition is frozen at their (earlier) recruitment
plus attrition; fresh panels reflect current recruitment. A widening
fresh-minus-old wedge in the foreign-born share is recruitment-margin
avoidance (or genuine foreign-born population decline reaching new samples
first). Also records the all-household Type A noninterview rate per group.

Usage: uv run python entry_margin.py <data_dir> [start end]
Writes outputs/entry_margin.csv (date, mis_group, fb_share, n_adults,
typea_rate_all).
"""

from __future__ import annotations

import sys

import pandas as pd

from fear_thermometer import month_files, prep_month

MIS_GROUPS = {"mis1": {1}, "mis2_4": {2, 3, 4}, "mis5": {5},
              "mis6_8": {6, 7, 8}}


def main():
    data_dir = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "jan23"
    end = sys.argv[3] if len(sys.argv) > 3 else "may26"
    rows = []
    for tok in month_files(start, end):
        df = prep_month(f"{data_dir}/{tok}pub.dat.gz")
        if (df["HRINTSTA"] == 1).sum() < 10000:  # Oct 2025 shutdown stub
            print(f"{tok}: stub file, skipped", flush=True)
            continue
        date = pd.Timestamp(year=2000 + int(tok[3:]),
                            month=["jan", "feb", "mar", "apr", "may", "jun",
                                   "jul", "aug", "sep", "oct", "nov",
                                   "dec"].index(tok[:3]) + 1, day=15)
        for name, mis in MIS_GROUPS.items():
            grp = df[df["HRMIS"].isin(list(mis))]
            hh = grp.drop_duplicates("hhkey")
            # Type A share of ALL sampled units (incl. Type B/C vacancies)
            typea = float((hh["HRINTSTA"] == 2).sum() / max(len(hh), 1))
            ad = grp[(grp["HRINTSTA"] == 1) & (grp["PRTAGE"] >= 15)
                     & (grp["PULINENO"] > 0)]
            w = ad["PWSSWGT"].sum()
            fb = float(ad.loc[ad["foreign_born"], "PWSSWGT"].sum() / w) \
                if w > 0 else float("nan")
            rows.append({"date": date, "mis_group": name,
                         "fb_share": round(fb, 5), "n_adults": len(ad),
                         "typea_rate_all": round(typea, 5)})
        print(f"{tok} done", flush=True)
    pd.DataFrame(rows).to_csv("outputs/entry_margin.csv", index=False)
    print("wrote outputs/entry_margin.csv")


if __name__ == "__main__":
    main()
