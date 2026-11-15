"""GDP per capita in 2021 for every GBD location, as the paper assembled it.

The paper feeds 2021 GDP per capita (current US$) into the regression of Bhat
et al. (2020), after dividing by 1.24 to bring it to the 2010 dollars the
regression was fitted in.  World Bank figures are used where they exist, and
International Monetary Fund figures fill the gaps.  The paper names two of its
gap-fills, and I follow them: World Bank 2020 for Cuba and IMF 2019 for Eritrea.

Sources (downloaded into data/worldbank/raw/ if not already there):
  https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&date=<year>&per_page=400
  https://www.imf.org/external/datamapper/api/v1/NGDPDPC

Usage
-----
    python inputs/prepare_gdp.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "worldbank" / "raw"
OUT = ROOT / "data" / "worldbank" / "gdp_per_capita_2021.csv"

WORLD_BANK = ("https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD"
              "?format=json&date={year}&per_page=400")
IMF = "https://www.imf.org/external/datamapper/api/v1/NGDPDPC"
PAPER_GAP_FILLS = {"CUB": ("world_bank", 2020), "ERI": ("imf", 2019)}


def fetch(url: str, path: Path) -> dict | list:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text())


def world_bank(year: int) -> dict[str, float]:
    rows = fetch(WORLD_BANK.format(year=year), RAW / f"gdp_pcap_cd_{year}.json")[1]
    return {r["countryiso3code"]: r["value"] for r in rows if r["value"] is not None}


def main() -> None:
    locations = pd.read_csv(ROOT / "data" / "locations" / "location_crosswalk.csv")
    bank = {year: world_bank(year) for year in (2021, 2020)}
    fund = fetch(IMF, RAW / "imf_ngdpdpc.json")["values"]["NGDPDPC"]

    records = []
    for iso3 in locations["iso3"]:
        if iso3 in PAPER_GAP_FILLS:
            source, year = PAPER_GAP_FILLS[iso3]
        elif iso3 in bank[2021]:
            source, year = "world_bank", 2021
        else:
            source, year = "imf", 2021
        value = bank[year].get(iso3) if source == "world_bank" else fund.get(iso3, {}).get(str(year))
        records.append({"iso3": iso3, "gdp_per_capita_current_usd": value,
                        "source": source, "source_year": year})
    table = locations[["location_id", "gbd_name", "iso3", "in_paper"]].merge(
        pd.DataFrame(records), on="iso3"
    )
    table.to_csv(OUT, index=False)
    missing = table[table["gdp_per_capita_current_usd"].isna()]
    print(table["source"].value_counts().to_string())
    print("no GDP figure:", missing["gbd_name"].tolist())
    print("of which in the paper:", missing.loc[missing["in_paper"], "gbd_name"].tolist())


if __name__ == "__main__":
    main()
