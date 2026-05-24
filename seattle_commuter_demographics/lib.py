"""Shared helpers: LODES8 column labels and Seattle / King County block sets."""

from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"
YEAR = 2023

KING_COUNTY_FIPS = "53033"
SEATTLE_PLACEFP = "63000"
WA_STATE_FIPS = "53"

# LODES8 column label maps
AGE = {"CA01": "≤29", "CA02": "30–54", "CA03": "≥55"}
EARN = {"CE01": "≤$1.25k/mo", "CE02": "$1.25–3.33k/mo", "CE03": ">$3.33k/mo"}
SEX = {"CS01": "Male", "CS02": "Female"}
RACE = {
    "CR01": "White alone",
    "CR02": "Black alone",
    "CR03": "AIAN alone",
    "CR04": "Asian alone",
    "CR05": "NHPI alone",
    "CR07": "Two+ races",
}
ETH = {"CT01": "Not Hispanic", "CT02": "Hispanic"}
EDU = {
    "CD01": "Less than HS",
    "CD02": "HS or equiv.",
    "CD03": "Some college / AA",
    "CD04": "Bachelor's or higher",
}
NAICS = {
    "CNS01": "Agriculture/Forestry",
    "CNS02": "Mining",
    "CNS03": "Utilities",
    "CNS04": "Construction",
    "CNS05": "Manufacturing",
    "CNS06": "Wholesale Trade",
    "CNS07": "Retail Trade",
    "CNS08": "Transportation/Warehousing",
    "CNS09": "Information",
    "CNS10": "Finance/Insurance",
    "CNS11": "Real Estate",
    "CNS12": "Professional/Scientific",
    "CNS13": "Management",
    "CNS14": "Admin/Waste mgmt",
    "CNS15": "Educational Services",
    "CNS16": "Health Care & Social Assistance",
    "CNS17": "Arts/Entertainment",
    "CNS18": "Accommodation/Food Services",
    "CNS19": "Other Services",
    "CNS20": "Public Administration",
}

# Industry groups targeted by the analysis (workplaces likely to have cafeterias
# subject to public-health-influenced menus).
INSTITUTIONAL = {
    "Hospitals & Health Care": ["CNS16"],
    "Educational Services":    ["CNS15"],
    "Public Administration":   ["CNS20"],
}


def load_seattle_blocks() -> set[str]:
    """Return set of 15-digit block GEOIDs that fall inside the City of Seattle."""
    f = DATA / "BlockAssign_ST53_WA_INCPLACE_CDP.txt"
    df = pd.read_csv(f, sep="|", dtype=str)
    return set(df.loc[df["PLACEFP"] == SEATTLE_PLACEFP, "BLOCKID"])


def classify_block(geoid: str, seattle_blocks: set[str]) -> str:
    """Bucket a 15-digit block GEOID into Seattle / Rest of King Co / Rest of WA / Out of state."""
    if geoid in seattle_blocks:
        return "Seattle"
    if geoid.startswith(KING_COUNTY_FIPS):
        return "King Co. (outside Seattle)"
    if geoid.startswith(WA_STATE_FIPS):
        return "WA (outside King Co.)"
    return "Out of state"


def load_wac() -> pd.DataFrame:
    df = pd.read_csv(DATA / f"wa_wac_S000_JT00_{YEAR}.csv.gz", dtype={"w_geocode": str})
    return df


def load_rac() -> pd.DataFrame:
    df = pd.read_csv(DATA / f"wa_rac_S000_JT00_{YEAR}.csv.gz", dtype={"h_geocode": str})
    return df


def load_od() -> pd.DataFrame:
    main = pd.read_csv(DATA / f"wa_od_main_JT00_{YEAR}.csv.gz",
                       dtype={"w_geocode": str, "h_geocode": str})
    aux  = pd.read_csv(DATA / f"wa_od_aux_JT00_{YEAR}.csv.gz",
                       dtype={"w_geocode": str, "h_geocode": str})
    return pd.concat([main, aux], ignore_index=True)


def share_table(df: pd.DataFrame, base: str = "C000") -> pd.DataFrame:
    """Express each column as a percent of the row's `base` column. Drop createdate."""
    out = df.copy()
    out = out.drop(columns=[c for c in ["createdate"] if c in out.columns])
    for c in out.columns:
        if c == base or c == "label":
            continue
        out[c] = (out[c] / out[base] * 100).round(1)
    return out
