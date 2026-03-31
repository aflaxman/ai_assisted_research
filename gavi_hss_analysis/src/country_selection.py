"""
Country selection for Gavi HSS analysis pilot.

Selects 12 countries with geographic diversity across WHO regions,
prioritizing countries where HSS proposal documents are publicly available.
"""

SELECTED_COUNTRIES = [
    {
        "country": "Ethiopia",
        "iso3": "ETH",
        "who_region": "AFRO",
        "rationale": "Large AFRO country; HSS proposal doc available on gavi.org; HSS evaluation country",
    },
    {
        "country": "Nigeria",
        "iso3": "NGA",
        "who_region": "AFRO",
        "rationale": "Largest African population; high-impact Gavi country; HSS proposal doc available",
    },
    {
        "country": "Democratic Republic of the Congo",
        "iso3": "COD",
        "who_region": "AFRO",
        "rationale": "Large conflict-affected AFRO country; HSS proposal doc available",
    },
    {
        "country": "Malawi",
        "iso3": "MWI",
        "who_region": "AFRO",
        "rationale": "Smaller AFRO country with strong immunization program; HSS proposal doc available",
    },
    {
        "country": "Kenya",
        "iso3": "KEN",
        "who_region": "AFRO",
        "rationale": "East Africa; HSS proposal referenced on gavi.org; HSS evaluation country",
    },
    {
        "country": "Pakistan",
        "iso3": "PAK",
        "who_region": "EMRO",
        "rationale": "Large EMRO country; high-impact Gavi country; HSS proposal doc available",
    },
    {
        "country": "Chad",
        "iso3": "TCD",
        "who_region": "AFRO",
        "rationale": "Francophone Sahel country; low immunization coverage; HSS proposal doc available",
    },
    {
        "country": "India",
        "iso3": "IND",
        "who_region": "SEARO",
        "rationale": "Largest birth cohort globally; 2017 HSS PDF available; SEARO representation",
    },
    {
        "country": "Lao PDR",
        "iso3": "LAO",
        "who_region": "WPRO",
        "rationale": "Small WPRO country; HSS proposal doc available; geographic diversity",
    },
    {
        "country": "Togo",
        "iso3": "TGO",
        "who_region": "AFRO",
        "rationale": "Small francophone West Africa country; HSS proposal doc available",
    },
    {
        "country": "Central African Republic",
        "iso3": "CAF",
        "who_region": "AFRO",
        "rationale": "Fragile state; 2017-2019 HSS PDF available; conflict-affected setting",
    },
    {
        "country": "Uzbekistan",
        "iso3": "UZB",
        "who_region": "EURO",
        "rationale": "EURO region representation; HSS PDF available; Central Asian diversity",
    },
]

# Additional backup countries if some selected countries have insufficient docs
BACKUP_COUNTRIES = [
    {"country": "Cambodia", "iso3": "KHM", "who_region": "WPRO"},
    {"country": "Somalia", "iso3": "SOM", "who_region": "EMRO"},
    {"country": "Ghana", "iso3": "GHA", "who_region": "AFRO"},
    {"country": "Bangladesh", "iso3": "BGD", "who_region": "SEARO"},
]

if __name__ == "__main__":
    print(f"Selected {len(SELECTED_COUNTRIES)} countries:")
    for i, c in enumerate(SELECTED_COUNTRIES, 1):
        print(f"  {i}. {c['country']} ({c['iso3']}, {c['who_region']}): {c['rationale']}")

    print(f"\nRegion distribution:")
    from collections import Counter
    regions = Counter(c["who_region"] for c in SELECTED_COUNTRIES)
    for r, n in sorted(regions.items()):
        print(f"  {r}: {n}")
