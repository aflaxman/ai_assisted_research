"""Download LODES8 (WA, 2023) and the 2020 block-to-place crosswalk for WA."""

from pathlib import Path
import sys
import urllib.request

YEAR = 2023
STATE = "wa"
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

LODES = "https://lehd.ces.census.gov/data/lodes/LODES8"
FILES = {
    f"{STATE}_od_main_JT00_{YEAR}.csv.gz": f"{LODES}/{STATE}/od/{STATE}_od_main_JT00_{YEAR}.csv.gz",
    f"{STATE}_od_aux_JT00_{YEAR}.csv.gz":  f"{LODES}/{STATE}/od/{STATE}_od_aux_JT00_{YEAR}.csv.gz",
    f"{STATE}_wac_S000_JT00_{YEAR}.csv.gz": f"{LODES}/{STATE}/wac/{STATE}_wac_S000_JT00_{YEAR}.csv.gz",
    f"{STATE}_rac_S000_JT00_{YEAR}.csv.gz": f"{LODES}/{STATE}/rac/{STATE}_rac_S000_JT00_{YEAR}.csv.gz",
    "BlockAssign_ST53_WA.zip": "https://www2.census.gov/geo/docs/maps-data/data/baf2020/BlockAssign_ST53_WA.zip",
}


def fetch(name: str, url: str) -> None:
    dest = DATA / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {name} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  [get]  {name}  <- {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"         {dest.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    for name, url in FILES.items():
        try:
            fetch(name, url)
        except Exception as e:
            print(f"  [fail] {name}: {e}", file=sys.stderr)
            sys.exit(1)
    print("done.")
