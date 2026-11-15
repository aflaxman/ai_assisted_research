"""Assign each GBD country to its GBD super-region.

Two of the seven super-regions (High-income; Central Europe, Eastern Europe,
and Central Asia) were listed by the IHME GBD connector before its CSV export
cap was reached.  The other five are entered here by hand from the published
GBD location hierarchy.  The script checks that the hand-entered names exist,
that nothing is assigned twice, and that all 204 locations are covered.

Usage
-----
    python inputs/build_super_regions.py path/to/partial_from_connector.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "locations" / "gbd_super_regions.csv"

BY_HAND = {
    "South Asia": ["Bangladesh", "Bhutan", "India", "Nepal", "Pakistan"],
    "North Africa and Middle East": [
        "Afghanistan", "Algeria", "Bahrain", "Egypt", "Iran (Islamic Republic of)", "Iraq",
        "Jordan", "Kuwait", "Lebanon", "Libya", "Morocco", "Oman", "Palestine", "Qatar",
        "Saudi Arabia", "Sudan", "Syrian Arab Republic", "Tunisia", "Türkiye",
        "United Arab Emirates", "Yemen",
    ],
    "Southeast Asia, East Asia, and Oceania": [
        # East Asia
        "China", "Democratic People's Republic of Korea", "Taiwan",
        # Southeast Asia
        "Cambodia", "Indonesia", "Lao People's Democratic Republic", "Malaysia", "Maldives",
        "Mauritius", "Myanmar", "Philippines", "Seychelles", "Sri Lanka", "Thailand",
        "Timor-Leste", "Viet Nam",
        # Oceania
        "American Samoa", "Cook Islands", "Fiji", "Guam", "Kiribati", "Marshall Islands",
        "Micronesia (Federated States of)", "Nauru", "Niue", "Northern Mariana Islands",
        "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tokelau", "Tonga", "Tuvalu",
        "Vanuatu",
    ],
    "Latin America and Caribbean": [
        # Caribbean
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Bermuda", "Cuba", "Dominica",
        "Dominican Republic", "Grenada", "Guyana", "Haiti", "Jamaica", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname",
        "Trinidad and Tobago", "United States Virgin Islands",
        # Andean, Central, and Tropical Latin America
        "Bolivia (Plurinational State of)", "Ecuador", "Peru",
        "Colombia", "Costa Rica", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua",
        "Panama", "Venezuela (Bolivarian Republic of)",
        "Brazil", "Paraguay",
    ],
    "Sub-Saharan Africa": [
        # Central
        "Angola", "Central African Republic", "Congo", "Democratic Republic of the Congo",
        "Equatorial Guinea", "Gabon",
        # Eastern
        "Burundi", "Comoros", "Djibouti", "Eritrea", "Ethiopia", "Kenya", "Madagascar", "Malawi",
        "Mozambique", "Rwanda", "Somalia", "South Sudan", "Uganda",
        "United Republic of Tanzania", "Zambia",
        # Southern
        "Botswana", "Eswatini", "Lesotho", "Namibia", "South Africa", "Zimbabwe",
        # Western
        "Benin", "Burkina Faso", "Cabo Verde", "Cameroon", "Chad", "Côte d'Ivoire", "Gambia",
        "Ghana", "Guinea", "Guinea-Bissau", "Liberia", "Mali", "Mauritania", "Niger", "Nigeria",
        "Sao Tome and Principe", "Senegal", "Sierra Leone", "Togo",
    ],
}


def main(partial_path: str) -> None:
    gbd = pd.read_csv(ROOT / "data" / "locations" / "location_crosswalk.csv")[
        ["location_id", "gbd_name"]
    ].rename(columns={"gbd_name": "location_name"})
    from_connector = pd.read_csv(partial_path)[["location_id", "super_region"]]
    from_connector["source"] = "IHME GBD connector"

    rows = []
    known = set(gbd["location_name"])
    for super_region, names in BY_HAND.items():
        missing = [name for name in names if name not in known]
        assert not missing, f"not GBD location names: {missing}"
        rows += [(name, super_region) for name in names]
    by_hand = pd.DataFrame(rows, columns=["location_name", "super_region"]).merge(
        gbd, on="location_name", validate="one_to_one"
    )
    by_hand["source"] = "GBD location hierarchy, entered by hand"

    table = pd.concat(
        [from_connector.merge(gbd, on="location_id", validate="one_to_one"), by_hand],
        ignore_index=True,
    )
    assert table["location_id"].is_unique, "a location is assigned twice"
    unassigned = sorted(known - set(table["location_name"]))
    assert not unassigned, f"unassigned: {unassigned}"
    assert len(table) == 204 and table["super_region"].nunique() == 7
    table = table.sort_values("location_name")[
        ["location_id", "location_name", "super_region", "source"]
    ]
    table.to_csv(OUT, index=False)
    print(table.groupby("super_region").size().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
