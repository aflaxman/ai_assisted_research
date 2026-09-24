"""Bidirectional USPS-style street-suffix standardization.

Both directions use the same dictionary of official forms, so neither
treatment gets an unfair vocabulary advantage. Typo'd suffixes ("stree",
"svv") are unknown to both cleaners, as they would be in practice.
"""

# canonical full word -> USPS abbreviation (subset covering pseudopeople data)
FULL_TO_ABBREV = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "boulevard": "blvd",
    "circle": "cir",
    "parkway": "pkwy",
    "highway": "hwy",
    "terrace": "ter",
    "trail": "trl",
    "square": "sq",
    "cove": "cv",
    "point": "pt",
    "ridge": "rdg",
    "creek": "crk",
    "crescent": "cres",
    "heights": "hts",
    "junction": "jct",
    "expressway": "expy",
    "freeway": "fwy",
    "gardens": "gdns",
    "grove": "grv",
    "harbor": "hbr",
    "hollow": "holw",
    "island": "is",
    "landing": "lndg",
    "meadows": "mdws",
    "mission": "msn",
    "plaza": "plz",
    "shore": "shr",
    "springs": "spgs",
    "station": "sta",
    "summit": "smt",
    "turnpike": "tpke",
    "valley": "vly",
    "village": "vlg",
    "vista": "vis",
}

ABBREV_TO_FULL = {v: k for k, v in FULL_TO_ABBREV.items()}


def _map_tokens(name, mapping):
    if not isinstance(name, str):
        return name
    out = []
    for tok in name.split():
        key = tok.rstrip(".")
        out.append(mapping.get(key, tok))
    return " ".join(out)


def abbreviate(name):
    """Standardize toward abbreviations: 'main street' -> 'main st'."""
    return _map_tokens(name, FULL_TO_ABBREV)


def expand(name):
    """Standardize toward full words: 'main st' -> 'main street'.

    Note this is the naive token mapping a regex-based cleaner performs:
    it cannot tell 'st clair ave' (Saint Clair) from 'main st', so it
    produces 'street clair avenue'. That collision cost is part of what
    the experiment measures.
    """
    return _map_tokens(name, ABBREV_TO_FULL)


TREATMENTS = {
    "none": lambda s: s,
    "abbreviate": abbreviate,
    "expand": expand,
}
