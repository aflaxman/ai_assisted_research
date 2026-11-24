"""Validate a saved IHME GBD connector result and file it under its plan id.

Large connector results are written to disk by the harness instead of being
shown inline.  This script reads such a file, works out which planned call it
answers from its *content* (cause, measure, age band, number of locations),
checks it, and copies it to ``data/gbd_2023/raw/<plan id>.txt``.

Usage
-----
    python gbd_pull/stash.py <saved result file> [<saved result file> ...]
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import ADULT_AGE_BANDS, RAW_DIR, YEAR, build_plan, is_done  # noqa: E402


def read_result(path: Path) -> tuple[str, list[dict]]:
    """Return ("exposure" | "burden", rows) for a saved connector result."""
    text = path.read_text()
    try:
        payload = json.loads(text)
        result = payload["result"] if isinstance(payload, dict) and "result" in payload else text
    except json.JSONDecodeError:
        result = text
    result = result.strip()
    if result.startswith("{"):
        rows = json.loads(result)["rows"]
        return ("exposure" if rows and "rei_name" in rows[0] else "burden"), rows
    return "burden", list(csv.DictReader(io.StringIO(result)))


def describe(kind: str, rows: list[dict]) -> dict:
    def only(column: str) -> str:
        values = {str(r[column]) for r in rows}
        return values.pop() if len(values) == 1 else f"<{len(values)} values>"

    locations = {str(r["location_id"]) for r in rows}
    years = {int(float(r["year"])) for r in rows}
    if years != {YEAR}:
        raise ValueError(f"expected year {YEAR}, found {sorted(years)}")
    if kind == "exposure":
        return {"kind": kind, "rei_name": only("rei_name"), "series_id": only("series_id"),
                "age_group": only("age_group"), "n_locations": len(locations),
                "n_rows": len(rows)}
    return {"kind": kind, "cause_name": only("cause_name"), "measure": only("measure"),
            "age_group": only("age_group"), "n_locations": len(locations),
            "n_rows": len(rows), "age_groups": {r["age_group"] for r in rows}}


def matches(found: dict, expect: dict) -> bool:
    if found["kind"] != expect["kind"] or found["n_locations"] != expect["n_locations"]:
        return False
    if found["kind"] == "exposure":
        if found["rei_name"].lower() != expect["rei_name"].lower():
            return False
        if expect["series_id"] is not None and found["series_id"] != str(expect["series_id"]):
            return False
        return found["age_group"] == expect["age_group"] and found["n_rows"] == expect["n_rows"]
    if found["cause_name"].lower() != expect["cause_name"].lower():
        return False
    if found["measure"].lower() != expect["measure"].lower():
        return False
    if expect["age_group"] == "by_age":
        # every adult band must be present for every location
        return set(ADULT_AGE_BANDS) <= found["age_groups"] and found["n_rows"] >= 8 * 15
    return found["age_group"] == expect["age_group"] and found["n_rows"] == expect["n_rows"]


def main(argv: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    for arg in argv[1:]:
        path = Path(arg)
        try:
            kind, rows = read_result(path)
            found = describe(kind, rows)
        except Exception as error:  # noqa: BLE001
            print(f"FAILED  {path.name}: could not parse ({type(error).__name__}: {error})")
            continue
        hits = [c for c in plan if matches(found, c["expect"])]
        if len(hits) != 1:
            shown = {k: v for k, v in found.items() if k != "age_groups"}
            print(f"FAILED  {path.name}: matched {len(hits)} planned calls; content was {shown}")
            continue
        call = hits[0]
        target = RAW_DIR / f"{call['id']}.txt"
        if target.exists():
            print(f"SKIPPED {call['id']} (already stashed)")
            continue
        shutil.copyfile(path, target)
        print(f"OK      {call['id']}  rows={found['n_rows']}")
    remaining = sum(not is_done(c) for c in plan)
    print(f"{remaining} of {len(plan)} planned calls remain")


if __name__ == "__main__":
    main(sys.argv)
