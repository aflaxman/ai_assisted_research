"""Build the crosswalk between the four country lists this model touches.

GBD (the connector's location names and ids), the paper's supplement, NCD-RisC,
and the World Bank / IMF all spell countries differently.  Names are matched
exactly first and then through the explicit tables below, so a wrong match
cannot slip in through fuzzy matching (the two Congos and the two Koreas are
the usual casualties).

Usage
-----
    python inputs/build_location_crosswalk.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "locations" / "location_crosswalk.csv"

# GBD location name -> ISO3, where the GBD name differs from NCD-RisC's name
# or NCD-RisC does not cover the location.
GBD_NAME_TO_ISO3 = {
    "Bolivia (Plurinational State of)": "BOL",
    "Brunei Darussalam": "BRN",
    "Cabo Verde": "CPV",
    "Czechia": "CZE",
    "Côte d'Ivoire": "CIV",
    "Democratic People's Republic of Korea": "PRK",
    "Democratic Republic of the Congo": "COD",
    "Eswatini": "SWZ",
    "Guam": "GUM",
    "Iran (Islamic Republic of)": "IRN",
    "Lao People's Democratic Republic": "LAO",
    "Micronesia (Federated States of)": "FSM",
    "Monaco": "MCO",
    "North Macedonia": "MKD",
    "Northern Mariana Islands": "MNP",
    "Palestine": "PSE",
    "Republic of Korea": "KOR",
    "Republic of Moldova": "MDA",
    "Russian Federation": "RUS",
    "San Marino": "SMR",
    "Syrian Arab Republic": "SYR",
    "Taiwan": "TWN",
    "Taiwan (Province of China)": "TWN",
    "Türkiye": "TUR",
    "United Republic of Tanzania": "TZA",
    "United States Virgin Islands": "VIR",
    "United States of America": "USA",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM",
    "Bahamas": "BHS",
    "Gambia": "GMB",
    "Guinea-Bissau": "GNB",
}
# The paper's country name -> ISO3, where it differs from NCD-RisC's name.
PAPER_NAME_TO_ISO3 = {
    "Bolivia": "BOL", "Brunei": "BRN", "Cape Verde": "CPV", "Czech Republic": "CZE",
    "Cote d'Ivoire": "CIV", "Democratic Republic of the Congo": "COD",
    "Federated States of Micronesia": "FSM", "Iran": "IRN", "Laos": "LAO",
    "Macedonia": "MKD", "Moldova": "MDA", "Russia": "RUS", "South Korea": "KOR",
    "Swaziland": "SWZ", "Syria": "SYR", "Taiwan (Province of China)": "TWN",
    "Tanzania": "TZA", "The Bahamas": "BHS", "The Gambia": "GMB", "Turkey": "TUR",
    "United States": "USA", "Venezuela": "VEN", "Vietnam": "VNM",
    "Guinea-Bissau": "GNB",
}


def main() -> None:
    gbd = (
        pd.read_csv(ROOT / "data" / "gbd_2023" / "exposure.csv")[["location_id", "location_name"]]
        .drop_duplicates().rename(columns={"location_name": "gbd_name"})
    )
    ncd = (
        pd.read_csv(ROOT / "data" / "ncdrisc" / "ncdrisc_hypertension_2019_age_specific.csv")
        [["country", "iso3"]].drop_duplicates().rename(columns={"country": "ncdrisc_name"})
    )
    paper = pd.DataFrame({"paper_name": sorted(
        set(pd.read_csv(ROOT / "data" / "paper" / "supp_table3_country_results.csv")["country"])
        - {"Global"}
    )})
    exact = dict(zip(ncd["ncdrisc_name"], ncd["iso3"]))

    gbd["iso3"] = gbd["gbd_name"].map(lambda name: GBD_NAME_TO_ISO3.get(name, exact.get(name)))
    paper["iso3"] = paper["paper_name"].map(
        lambda name: PAPER_NAME_TO_ISO3.get(name, exact.get(name))
    )
    for table, column in ((gbd, "gbd_name"), (paper, "paper_name")):
        unmatched = table.loc[table["iso3"].isna(), column].tolist()
        assert not unmatched, f"no ISO3 for {unmatched}"
        assert table["iso3"].is_unique, f"duplicate ISO3 among {column}"

    crosswalk = gbd.merge(paper, on="iso3", how="left").merge(ncd, on="iso3", how="left")
    crosswalk["in_paper"] = crosswalk["paper_name"].notna()
    assert crosswalk["in_paper"].sum() == len(paper) == 193
    crosswalk = crosswalk.sort_values("gbd_name")[
        ["location_id", "gbd_name", "iso3", "paper_name", "ncdrisc_name", "in_paper"]
    ]
    crosswalk.to_csv(OUT, index=False)
    print(f"{len(crosswalk)} GBD locations, {int(crosswalk['in_paper'].sum())} in the paper")
    print("GBD locations the paper leaves out:",
          crosswalk.loc[~crosswalk["in_paper"], "gbd_name"].tolist())
    print("paper countries without NCD-RisC data:",
          crosswalk.loc[crosswalk["in_paper"] & crosswalk["ncdrisc_name"].isna(), "gbd_name"].tolist())


if __name__ == "__main__":
    main()
