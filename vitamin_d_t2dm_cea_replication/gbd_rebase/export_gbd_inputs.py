"""
Export real GBD 2021 age-specific epidemiology into gbd_inputs.csv.

RUN THIS ON THE IHME CLUSTER (needs get_draws / db_queries / an active central-comp env).
It writes ../gbd_rebase/gbd_inputs.csv in the exact schema gbd_rebase_model.py reads:

    location,age_start,t2d_incidence_per1000,acmr_per1000

so after running it (and committing the CSV) the GBD re-base is real with no model edits.
It pulls, per location:
  * type-2 diabetes INCIDENCE (general population, COMO)      -> t2d_incidence_per1000
  * ALL-CAUSE mortality (CoDCorrect deaths / population)      -> acmr_per1000
both-sex, collapsed to the posterior mean across draws, for GBD's 5-year adult age bands.
(The model applies its own calibrated prediabetes-progression multiplier on top of the
general-population incidence, so general-pop incidence is exactly the right input.)

------------------------------------------------------------------------------------
CONFIRM THESE 3 THINGS FOR YOUR RELEASE before running (they change occasionally):
  1. RELEASE_ID for GBD 2021.
  2. CAUSE_T2D  -- run the verification block below; it prints the cause name.
  3. That your get_draws accepts metric_id (see INCIDENCE_METRIC note).
------------------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
from get_draws.api import get_draws
from db_queries import get_population, get_age_metadata, get_ids  # central comp

# ---- CONFIG (edit) ----------------------------------------------------------
RELEASE_ID = 16            # <-- GBD 2021. Confirm with your team (get_ids('release')).
YEAR_ID = 2021
CAUSE_T2D = 976            # Diabetes mellitus type 2. Confirm (verification block prints it).
CAUSE_ALL = 294            # All causes (stable).
MIN_AGE_START = 20         # adults; the microsim starts at >=18, model interpolates.
# name -> GBD location_id. Add any locations you want (get_location_metadata for ids).
LOCATIONS = {
    "USA": 102,
    # "Mexico": 130, "India": 163, "China": 6, "United Kingdom": 95,
    # "Japan": 67, "Brazil": 135, "Nigeria": 214,
}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "gbd_inputs.csv")


def draw_mean(df):
    """Collapse draw_* columns to a single posterior-mean 'val'."""
    dcols = [c for c in df.columns if c.startswith("draw_")]
    keep = [c for c in ["age_group_id", "sex_id", "location_id", "year_id"] if c in df.columns]
    out = df[keep].copy()
    out["val"] = df[dcols].mean(axis=1)
    return out


def main():
    # --- verification: confirm the T2D cause id resolves to the right name ---
    try:
        causes = get_ids("cause")
        nm = causes.loc[causes.cause_id == CAUSE_T2D, "cause_name"]
        print(f"[verify] CAUSE_T2D={CAUSE_T2D} -> {nm.iloc[0] if len(nm) else '??? CHECK THIS'}")
    except Exception as e:  # noqa: BLE001
        print(f"[verify] could not resolve cause name ({e}); double-check CAUSE_T2D.")

    # --- adult age bands for this release (age_group_id -> age_start) ---
    age_meta = get_age_metadata(release_id=RELEASE_ID)
    adult = age_meta[age_meta["age_group_years_start"] >= MIN_AGE_START].copy()
    AGE_GROUPS = adult["age_group_id"].tolist()
    age_start = dict(zip(adult["age_group_id"], adult["age_group_years_start"].astype(int)))

    rows = []
    for name, loc in LOCATIONS.items():
        print(f"[pull] {name} (location_id={loc}) ...", flush=True)

        # type-2 diabetes incidence RATE (per person) by age/sex, from COMO.
        # INCIDENCE_METRIC note: metric_id=3 = rate. If your get_draws rejects metric_id
        # for source='como', drop it and pull number (metric_id=1) then divide by pop
        # like the mortality path below.
        inc = draw_mean(get_draws(
            gbd_id_type="cause_id", gbd_id=CAUSE_T2D, source="como",
            measure_id=6, metric_id=3, location_id=loc, age_group_id=AGE_GROUPS,
            sex_id=[1, 2], year_id=YEAR_ID, release_id=RELEASE_ID,
        )).rename(columns={"val": "inc_rate"})

        # all-cause deaths (NUMBER) by age/sex, from CoDCorrect.
        dth = draw_mean(get_draws(
            gbd_id_type="cause_id", gbd_id=CAUSE_ALL, source="codcorrect",
            measure_id=1, metric_id=1, location_id=loc, age_group_id=AGE_GROUPS,
            sex_id=[1, 2], year_id=YEAR_ID, release_id=RELEASE_ID,
        )).rename(columns={"val": "deaths"})

        # population denominator by age/sex.
        pop = get_population(
            location_id=loc, age_group_id=AGE_GROUPS, sex_id=[1, 2],
            year_id=YEAR_ID, release_id=RELEASE_ID,
        )[["age_group_id", "sex_id", "population"]]

        m = inc.merge(dth, on=["age_group_id", "sex_id"]).merge(pop, on=["age_group_id", "sex_id"])
        # collapse the two sexes to both-sex, population-weighting rates:
        #   incidence rate  = sum(rate_s * pop_s) / sum(pop_s)
        #   mortality rate  = sum(deaths_s)       / sum(pop_s)
        g = m.groupby("age_group_id").apply(lambda d: pd.Series({
            "t2d_incidence_per1000": 1000.0 * (d["inc_rate"] * d["population"]).sum() / d["population"].sum(),
            "acmr_per1000": 1000.0 * d["deaths"].sum() / d["population"].sum(),
        })).reset_index()
        g["location"] = name
        g["age_start"] = g["age_group_id"].map(age_start)
        rows.append(g[["location", "age_start", "t2d_incidence_per1000", "acmr_per1000"]])

    out = pd.concat(rows).sort_values(["location", "age_start"]).reset_index(drop=True)

    header = (
        "# REAL GBD 2021 age-specific epidemiology, exported by export_gbd_inputs.py on the IHME cluster.\n"
        f"# release_id={RELEASE_ID}, year_id={YEAR_ID}, cause_t2d={CAUSE_T2D}, both-sex, posterior mean of draws.\n"
        "# t2d_incidence_per1000 = general-population T2D incidence (/1000/yr); "
        "acmr_per1000 = all-cause mortality (/1000/yr).\n"
    )
    with open(OUT_CSV, "w") as f:
        f.write(header)
        out.to_csv(f, index=False)
    print(f"\n[done] wrote {len(out)} rows for {len(LOCATIONS)} location(s) -> {OUT_CSV}")
    print(out.to_string(index=False))
    print("\nNext: commit gbd_inputs.csv, then `uv run python gbd_rebase_model.py` (no cluster needed).")


if __name__ == "__main__":
    main()
