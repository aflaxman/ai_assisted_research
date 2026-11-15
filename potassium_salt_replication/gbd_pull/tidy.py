"""Turn the stashed IHME GBD connector results into tidy CSV extracts.

Reads ``data/gbd_2023/raw/<plan id>.txt`` (see ``plan.py`` and ``stash.py``)
and writes long-format tables, one row per location, age band, and sex:

    data/gbd_2023/exposure.csv             sodium, SBP, CKD stage 3 by country
    data/gbd_2023/burden_by_age.csv        deaths by country and age (major causes)
    data/gbd_2023/burden_25_plus.csv       country totals for ages 25+
    data/gbd_2023/burden_region_by_age.csv Global and super-region age patterns
    data/gbd_2023/population.csv           person counts implied by count / rate

Usage
-----
    python gbd_pull/tidy.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import ADULT_AGE_BANDS, RAW_DIR, build_plan  # noqa: E402

OUT_DIR = RAW_DIR.parent
SEXES = {"male": "Male", "female": "Female"}


def read_rows(path: Path) -> list[dict]:
    result = json.loads(path.read_text())["result"].strip()
    if result.startswith("{"):
        return json.loads(result)["rows"]
    return list(csv.DictReader(io.StringIO(result)))


def tidy_exposure(rows: list[dict], key: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    parts = []
    for suffix, sex in SEXES.items():
        parts.append(pd.DataFrame({
            "exposure": key,
            "location_id": frame["location_id"].astype(int),
            "location_name": frame["location_name"],
            "year": frame["year"].astype(int),
            "age_group": frame["age_group"],
            "sex": sex,
            "mean": frame[f"mean_{suffix}"].astype(float),
            "lower": frame[f"lower_{suffix}"].astype(float),
            "upper": frame[f"upper_{suffix}"].astype(float),
        }))
    return pd.concat(parts, ignore_index=True)


def tidy_burden(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    parts = []
    for suffix, sex in SEXES.items():
        parts.append(pd.DataFrame({
            "cause": frame["cause_name"],
            "measure": frame["measure"],
            "location_id": frame["location_id"].astype(int),
            "location_name": frame["location_name"],
            "year": frame["year"].astype(int),
            "age_group": frame["age_group"],
            "sex": sex,
            "count": frame[f"count_{suffix}"].astype(float),
            "count_lower": frame[f"count_lower_{suffix}"].astype(float),
            "count_upper": frame[f"count_upper_{suffix}"].astype(float),
            # rates are per 100,000 of the age band's own population
            "rate": frame[f"rate_{suffix}"].astype(float),
        }))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    exposure, by_age, plus25, regions = [], [], [], []
    missing = []
    for call in build_plan():
        path = RAW_DIR / f"{call['id']}.txt"
        if not path.exists():
            missing.append(call["id"])
            continue
        rows = read_rows(path)
        kind, *rest = call["id"].split("__")
        if kind == "exposure":
            exposure.append(tidy_exposure(rows, rest[0]))
        else:
            table = tidy_burden(rows)
            where, age = rest[-1], call["args"]["age_group"]
            if where == "regions":
                regions.append(table[table["age_group"].isin(ADULT_AGE_BANDS)])
            elif age == "25 plus":
                plus25.append(table)
            else:
                by_age.append(table)

    def write(parts: list[pd.DataFrame], name: str, keys: list[str]) -> pd.DataFrame:
        if not parts:
            print(f"{name}: nothing to write yet")
            return pd.DataFrame()
        table = pd.concat(parts, ignore_index=True).sort_values(keys).reset_index(drop=True)
        assert not table.duplicated(keys).any(), f"duplicate rows in {name}"
        table.to_csv(OUT_DIR / name, index=False, float_format="%.10g")
        print(f"{name}: {len(table):,} rows")
        return table

    write(exposure, "exposure.csv", ["exposure", "location_id", "age_group", "sex"])
    burden_keys = ["cause", "measure", "location_id", "age_group", "sex"]
    by_age_table = write(by_age, "burden_by_age.csv", burden_keys)
    write(plus25, "burden_25_plus.csv", burden_keys)
    write(regions, "burden_region_by_age.csv", burden_keys)

    # Person counts are not pulled separately: each burden row carries a count
    # and a rate, and their ratio is the population of that stratum.  Ischemic
    # heart disease has deaths in every adult stratum, so it gives the cleanest
    # denominator.
    if len(by_age_table):
        ihd = by_age_table[
            (by_age_table["cause"] == "Ischemic heart disease")
            & (by_age_table["measure"] == "deaths") & (by_age_table["rate"] > 0)
        ]
        population = ihd.assign(population=ihd["count"] / ihd["rate"] * 1e5)[
            ["location_id", "location_name", "year", "age_group", "sex", "population"]
        ]
        population.to_csv(OUT_DIR / "population.csv", index=False, float_format="%.10g")
        print(f"population.csv: {len(population):,} rows, "
              f"{population['population'].sum() / 1e9:.3f} billion adults aged 25+")
    if missing:
        print(f"{len(missing)} planned calls are not stashed yet, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
