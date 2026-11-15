"""The list of IHME GBD connector calls that feed this replication.

The IHME GBD MCP connector serves GBD 2023 estimates, but it caps a user at
50,000 rows per 24 hours, and at 10,000 rows per 24 hours of full-precision
CSV, so the pulls have to be planned.  This module writes the plan down once,
as data, so that the pulls are reproducible and so that the row budget can be
checked before any call is made.

The plan has two parts because I found the CSV cap by running into it.  The
first part (sodium, blood pressure, and deaths by country and age for the
three largest causes) was pulled as planned, the deaths in CSV.  The second
part asks for JSON, which the connector rounds to publication precision, and
it asks for less: every remaining cause comes as a country total for ages 25
and over plus an age pattern by super-region.

Usage
-----
    python gbd_pull/plan.py summary      # row budget by block
    python gbd_pull/plan.py next 6       # next 6 calls that are not yet stashed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

YEAR = 2021
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "gbd_2023" / "raw"

# The five SDI-quintile member groups partition the 204 GBD countries and
# territories, so together they ask for every country in one call.
ALL_COUNTRIES = [
    "low_sdi_members",
    "low_middle_sdi_members",
    "middle_sdi_members",
    "high_middle_sdi_members",
    "high_sdi_members",
]
GLOBAL_AND_SUPER_REGIONS = ["Global", "super_regions"]

ADULT_AGE_BANDS = [
    "25 to 29", "30 to 34", "35 to 39", "40 to 44", "45 to 49", "50 to 54",
    "55 to 59", "60 to 64", "65 to 69", "70 to 74", "75 to 79", "80 to 84",
    "85 to 89", "90 to 94", "95 plus",
]

# name -> GBD cause_id.  The first three carry about four fifths of the deaths
# in the model, and their deaths are pulled for every country and five-year
# age band.
MAJOR_CAUSES = {
    "Ischemic heart disease": 493,
    "Ischemic stroke": 495,
    "Intracerebral hemorrhage": 496,
}
# The CSV cap arrived part-way through hypertensive heart disease: its deaths
# are in hand by country for these age bands only.
PARTIAL_CAUSES = {"Hypertensive heart disease": ADULT_AGE_BANDS[:5]}
# The rest are pulled as country totals for ages 25+ and given an age pattern
# from their super-region.
MINOR_CAUSES = {
    "Hypertensive heart disease": 498,
    "Chronic kidney disease": 589,
    "Subarachnoid hemorrhage": 497,
    "Rheumatic heart disease": 492,
    "Non-rheumatic calcific aortic valve disease": 968,
    "Other cardiomyopathy": 944,
    "Atrial fibrillation and flutter": 500,
    "Aortic aneurysm": 501,
    "Lower extremity peripheral arterial disease": 502,
    "Endocarditis": 503,
}
ALL_CAUSES = {**MAJOR_CAUSES, **MINOR_CAUSES}
# Causes for which GBD publishes incidence and the paper counts "new cases".
INCIDENCE_CAUSES = [
    "Ischemic heart disease", "Ischemic stroke", "Intracerebral hemorrhage",
    "Subarachnoid hemorrhage", "Atrial fibrillation and flutter",
    "Lower extremity peripheral arterial disease",
    "Non-rheumatic calcific aortic valve disease", "Chronic kidney disease",
]

EXPOSURES = {
    # slug -> (risk_name, series_id or None)
    "sodium": ("Diet high in sodium", None),
    "sbp": ("High systolic blood pressure", None),
    "ckd_stage3": ("Kidney dysfunction", 60),
}


def slug(text: str) -> str:
    return (
        text.lower().replace(" plus", "plus").replace(" to ", "to")
        .replace(" ", "_").replace("-", "_")
    )


def exposure_call(key: str, age: str) -> dict:
    risk_name, series_id = EXPOSURES[key]
    args = {
        "risk_name": risk_name, "locations": ALL_COUNTRIES, "year": YEAR,
        "age_group": age, "fmt": "json", "language": "en",
    }
    if series_id is not None:
        args["series_id"] = series_id
    return {
        "id": f"exposure__{key}__{slug(age)}",
        "tool": "mcp__IHME_GBD__get_risk_exposure",
        "args": args,
        "expect": {"kind": "exposure", "rei_name": risk_name, "series_id": series_id,
                   "age_group": age, "n_locations": 204, "n_rows": 204},
    }


def burden_call(cause: str, measure: str, age: str, where: str, fmt: str = "json") -> dict:
    locations = ALL_COUNTRIES if where == "countries" else GLOBAL_AND_SUPER_REGIONS
    n_locations = 204 if where == "countries" else 8
    return {
        "id": f"burden__{measure}__{slug(cause)}__{slug(age)}__{where}",
        "tool": "mcp__IHME_GBD__get_cause_burden",
        "args": {
            "cause_name": cause, "cause_id": ALL_CAUSES[cause], "locations": locations,
            "measure": measure, "year": YEAR, "age_group": age, "fmt": fmt,
            "language": "en",
        },
        "expect": {"kind": "burden", "cause_name": cause, "measure": measure,
                   "age_group": age, "n_locations": n_locations,
                   # by_age returns up to 25 life-table bands per location
                   "n_rows": n_locations if age != "by_age" else None},
    }


def build_plan() -> list[dict]:
    plan: list[dict] = []
    # Part one, pulled before the CSV cap was reached.
    # A. exposures that the blood pressure model cannot do without
    for key in ("sodium", "sbp"):
        plan += [exposure_call(key, age) for age in ADULT_AGE_BANDS]
    # B. deaths for the major causes, every country and age band
    for cause in MAJOR_CAUSES:
        plan += [burden_call(cause, "deaths", age, "countries", "csv") for age in ADULT_AGE_BANDS]
    for cause, ages in PARTIAL_CAUSES.items():
        plan += [burden_call(cause, "deaths", age, "countries", "csv") for age in ages]
    plan.append(burden_call("Subarachnoid hemorrhage", "deaths", "by_age", "regions", "csv"))

    # Part two, in JSON.
    # C. deaths for the other causes: country totals, super-region age pattern
    for cause in MINOR_CAUSES:
        plan.append(burden_call(cause, "deaths", "25 plus", "countries"))
    for cause in ALL_CAUSES:
        if cause != "Subarachnoid hemorrhage":
            plan.append(burden_call(cause, "deaths", "by_age", "regions"))
    # D. stage 3 chronic kidney disease prevalence (for the hyperkalaemia model)
    plan += [exposure_call("ckd_stage3", age) for age in ADULT_AGE_BANDS]
    # E. DALYs and new cases: country totals and super-region age patterns
    for measure, causes in (("dalys", list(ALL_CAUSES)), ("incidence", INCIDENCE_CAUSES)):
        for cause in causes:
            plan.append(burden_call(cause, measure, "25 plus", "countries"))
            plan.append(burden_call(cause, measure, "by_age", "regions"))
    ids = [c["id"] for c in plan]
    assert len(ids) == len(set(ids)), "duplicate call ids in plan"
    return plan


def is_done(call: dict) -> bool:
    return (RAW_DIR / f"{call['id']}.txt").exists()


def main(argv: list[str]) -> None:
    plan = build_plan()
    command = argv[1] if len(argv) > 1 else "summary"
    if command == "summary":
        rows = sum((c["expect"]["n_rows"] or 8 * 25) for c in plan)
        done = sum(is_done(c) for c in plan)
        print(f"{len(plan)} calls planned, {done} stashed, at most {rows:,} rows in total")
    elif command == "next":
        n = int(argv[2]) if len(argv) > 2 else 6
        todo = [c for c in plan if not is_done(c)][:n]
        if not todo:
            print("ALL DONE")
            return
        for call in todo:
            print(json.dumps({"id": call["id"], "tool": call["tool"], "args": call["args"]}))
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main(sys.argv)
