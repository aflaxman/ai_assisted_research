"""Download CPS Basic Monthly public-use files (no API key required).

The Census distributes these at www2.census.gov as ``MONyrpub.dat.gz`` files
that are actually ZIP archives. Files land in ``--out`` (default: a local
``cps_data/`` directory). Each national monthly file is ~10 MB compressed.

Usage:
    python download_data.py --year 2024 --out cps_data
"""

from __future__ import annotations

import argparse
import os
import urllib.request

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def url_for(year: int, mon: str) -> str:
    yy = str(year)[2:]
    return (f"https://www2.census.gov/programs-surveys/cps/datasets/"
            f"{year}/basic/{mon}{yy}pub.dat.gz")


def layout_url(year: int) -> str:
    return (f"https://www2.census.gov/programs-surveys/cps/datasets/"
            f"{year}/basic/{year}_Basic_CPS_Public_Use_Record_Layout"
            f"_plus_IO_Code_list.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default="cps_data")
    ap.add_argument("--months", nargs="*", default=MONTHS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    yy = str(args.year)[2:]
    for mon in args.months:
        dest = os.path.join(args.out, f"{mon}{yy}pub.dat.gz")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"have {dest}")
            continue
        u = url_for(args.year, mon)
        print(f"downloading {u}")
        urllib.request.urlretrieve(u, dest)
        print(f"  -> {dest} ({os.path.getsize(dest)} bytes)")


if __name__ == "__main__":
    main()
