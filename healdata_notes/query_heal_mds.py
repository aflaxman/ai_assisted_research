"""Query the HEAL Data Platform metadata API and summarize the study catalog.

The HEAL Data Platform (healdata.org) is built on Gen3; its metadata service
(MDS) is public and returns the full discovery catalog as JSON. This script
downloads it, tabulates research programs / repositories / data-release
status, and flags studies relevant to an OUD/MOUD simulation model.

Usage:
    uv run --with requests query_heal_mds.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import requests

MDS_URL = (
    "https://healdata.org/mds/metadata"
    "?_guid_type=discovery_metadata&data=True&limit=2000"
)
CACHE = Path("heal_mds.json")

KEYWORDS = {
    "moud": r"\bmoud\b|medication[s]? for opioid|medication-assisted"
            r"|buprenorphine|methadone|naltrexone",
    "overdose": r"overdose|naloxone|narcan|fentanyl",
    "oud": r"opioid use disorder|\boud\b|opioid addiction|opioid dependence",
    "retention": r"retention|discontinuation|dropout|adherence|relapse",
    "simulation": r"simulation|agent-based|system dynamics|microsimulation"
                  r"|markov|state-transition|decision-analytic"
                  r"|cost-effectiveness",
    "justice": r"justice|incarcerat|jail|prison|reentry|re-entry",
}


def fetch_catalog() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    resp = requests.get(MDS_URL, timeout=300)
    resp.raise_for_status()
    CACHE.write_text(resp.text)
    return resp.json()


def extract_row(hdp: str, rec: dict) -> dict:
    g = rec.get("gen3_discovery", {}) or {}
    sm = g.get("study_metadata", {}) or {}
    mi = sm.get("minimal_info", {}) or {}
    da = sm.get("data_availability", {}) or {}
    tags = g.get("tags", []) or []
    nih = rec.get("nih_reporter", {}) or {}
    n_vars = len((g.get("variable_metadata", {}) or {}).get("field_names", []) or [])
    return {
        "hdp": hdp,
        "name": mi.get("study_name", "") or g.get("project_title", ""),
        "text": " ".join(
            [
                mi.get("study_name", "") or "",
                mi.get("study_description", "") or "",
                nih.get("abstract_text") or "",
                g.get("research_program", "") or "",
            ]
        ).lower(),
        "program": g.get("research_program", ""),
        "repos": [t["name"] for t in tags if t.get("category") == "Data Repository"],
        "data_available": da.get("data_available", ""),
        "release_status": da.get("data_release_status", ""),
        "n_vars": n_vars,
    }


def main() -> None:
    catalog = fetch_catalog()
    rows = [extract_row(hdp, rec) for hdp, rec in catalog.items()]
    print(f"total studies: {len(rows)}")

    print("\n== research programs (top 15) ==")
    for name, n in Counter(r["program"] for r in rows).most_common(15):
        print(f"{n:4d}  {name or '(none)'}")

    print("\n== data release status ==")
    for name, n in Counter(r["release_status"] for r in rows).most_common():
        print(f"{n:4d}  {name or '(blank)'}")

    print("\n== OUD/MOUD studies with data available and released ==")
    for r in rows:
        hits = {k for k, pat in KEYWORDS.items() if re.search(pat, r["text"])}
        released = r["release_status"] in ("started", "finished")
        if ("moud" in hits or "oud" in hits) and released:
            print(
                f"{r['hdp']}  rel={r['release_status']:8s} "
                f"vars={r['n_vars']:4d}  {r['name'][:80]}"
            )


if __name__ == "__main__":
    main()
