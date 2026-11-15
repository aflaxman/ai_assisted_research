"""Tidy NCD-RisC's hypertension estimates for 2019 into the table the model reads.

NCD-RisC (Lancet 2021;398:957-80) publishes, for 200 countries, the prevalence
of hypertension and the proportions of people with hypertension who are
diagnosed, treated, and controlled, by sex and five-year age band from 30 to
79.  The paper uses the 2019 estimates.

NCD-RisC states no licence for its data files, so neither the download nor the
tidy table is kept in the repository.  Download

    https://ncdrisc.org/downloads/hypertension/NCD-RisC_Lancet_2021_Hypertension_age_specific_estimates_by_country.csv

(37 MB, linked from https://ncdrisc.org/data-downloads-hypertension.html) and run

    python inputs/prepare_hypertension.py <the downloaded file>

which writes ``data/ncdrisc/ncdrisc_hypertension_2019_age_specific.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUTPUT = (Path(__file__).resolve().parents[1] / "data" / "ncdrisc"
          / "ncdrisc_hypertension_2019_age_specific.csv")
YEAR = 2019
QUANTITIES = {
    "Prevalence of hypertension": "htn_prev",
    "Proportion of diagnosed hypertension among all hypertension": "diagnosed",
    "Proportion of treated hypertension among all hypertension": "treated",
    "Proportion of controlled hypertension among all hypertension": "controlled",
    "Proportion of untreated stage 2 hypertension among all hypertension": "untreated_stage2",
}
BOUNDS = {" lower 95% uncertainty interval": "_lower", " upper 95% uncertainty interval": "_upper"}


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    names = {"Country": "country", "ISO": "iso3", "Sex": "sex", "Year": "year", "Age": "age"}
    for long_name, short_name in QUANTITIES.items():
        names[long_name] = short_name
        for long_bound, short_bound in BOUNDS.items():
            names[long_name + long_bound] = short_name + short_bound
    missing = set(names) - set(raw.columns)
    if missing:
        raise ValueError(f"the download has no column(s) {sorted(missing)}")
    table = raw[raw["Year"] == YEAR].rename(columns=names)[list(names.values())]
    table["sex"] = table["sex"].map({"Men": "Male", "Women": "Female"})
    assert table["sex"].notna().all()
    assert table.groupby(["iso3", "sex"]).size().eq(10).all(), "expected ten age bands, 30-79"
    return table.reset_index(drop=True)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    table = tidy(pd.read_csv(sys.argv[1], encoding="utf-8-sig"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, float_format="%.6g")
    print(f"{OUTPUT}: {len(table):,} rows, {table['iso3'].nunique()} countries")


if __name__ == "__main__":
    main()
