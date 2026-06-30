"""Link CPS basic monthly files across consecutive months and classify
each person as a "stayer" or a "leaver", and each household by its
re-interview outcome.

The point of this module is NOT to measure international out-migration
directly -- the CPS public file records no destination for anyone who
leaves. It is to measure the *gross* household-departure signal that the
rotating panel actually exposes, and to show how international emigration
sits buried inside it (alongside domestic moves, deaths, and institutional
entries) with no field that separates them.

Linkage method (the CPS standard, cf. Drew, Flood & Warren 2014 / IPUMS
CPSIDP): household = HRHHID + HRHHID2; person = line number PULINENO within
household; a candidate person-link is *validated* by requiring matching sex
and race and a plausible one-month age change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cps_parse import read_month

# Households in these months-in-sample are scheduled to be re-interviewed the
# next calendar month (MIS 4 and 8 rotate out; after 4 the unit is gone 8
# months, after 8 it is finished).
MIS_ELIGIBLE_NEXT = {1, 2, 3, 5, 6, 7}

AGE_TOL_LO, AGE_TOL_HI = -1, 2  # plausible one-month age change window


def _hhkey(df: pd.DataFrame) -> pd.Series:
    return df["HRHHID"].str.strip() + "_" + df["HRHHID2"].str.strip()


def _valid_person(df: pd.DataFrame) -> pd.Series:
    """A real interviewed person record (not a noninterview placeholder)."""
    return (
        (df["HRINTSTA"] == 1)
        & (df["PRPERTYP"].isin([1, 2, 3]))
        & (df["PRTAGE"] >= 0)
        & (df["PULINENO"] > 0)
    )


def link_pair(df_t: pd.DataFrame, df_t1: pd.DataFrame) -> dict:
    """Link month t to month t+1 (already filtered to one state).

    Returns a dict with the classified person-level frame, a household-level
    frame, and summary counts/weights.
    """
    df_t = df_t.copy()
    df_t1 = df_t1.copy()
    df_t["hhkey"] = _hhkey(df_t)
    df_t1["hhkey"] = _hhkey(df_t1)

    # --- Household-level interview status per month -------------------------
    # One row per household: its interview status. (Noninterview households
    # have a single placeholder record carrying HRINTSTA 2/3/4.)
    hh_t = (
        df_t.groupby("hhkey")
        .agg(intsta_t=("HRINTSTA", "min"), mis_t=("HRMIS", "max"))
        .reset_index()
    )
    hh_t1 = (
        df_t1.groupby("hhkey")
        .agg(intsta_t1=("HRINTSTA", "min"), mis_t1=("HRMIS", "max"))
        .reset_index()
    )
    hh = hh_t.merge(hh_t1, on="hhkey", how="left")

    # Households interviewed in t and *scheduled* to return next month.
    hh["eligible_next"] = hh["mis_t"].isin(MIS_ELIGIBLE_NEXT)
    interviewed_t = hh["intsta_t"] == 1
    sched = hh[interviewed_t & hh["eligible_next"]].copy()

    def _hh_outcome(row):
        if pd.isna(row["intsta_t1"]):
            return "gone_from_sample"      # no record at all next month
        s = int(row["intsta_t1"])
        return {1: "reinterviewed", 2: "typeA_noninterview",
                3: "typeB_vacant_ineligible", 4: "typeC_ineligible"}.get(s, "other")

    sched["hh_outcome"] = sched.apply(_hh_outcome, axis=1)

    # --- Person-level linkage within both-interviewed households -----------
    # Month-t side: real interviewed people, in households eligible to return.
    pt = df_t[_valid_person(df_t)].copy()
    eligible_hh = set(sched["hhkey"])
    pt = pt[pt["hhkey"].isin(eligible_hh)].copy()

    # Month-(t+1) match side: the full *roster* of interviewed households (any
    # person row with a real line number), NOT the strict _valid_person set.
    # Presence is roster-based so that a person who is still in the household but
    # carries an allocation-flagged record (e.g. PRTAGE = -1) is not mistaken for
    # a leaver. Identity is then validated to catch a line number reused by a
    # genuinely different (replacement) person.
    roster_t1 = df_t1[(df_t1["HRINTSTA"] == 1) & (df_t1["PULINENO"] > 0)]

    # Set of households that WERE re-interviewed next month.
    reint_hh = set(sched.loc[sched["hh_outcome"] == "reinterviewed", "hhkey"])

    pt1d = (
        roster_t1[["hhkey", "PULINENO", "PESEX", "PTDTRACE", "PRTAGE"]]
        .drop_duplicates(["hhkey", "PULINENO"])
        .rename(columns={"PESEX": "PESEX_1", "PTDTRACE": "PTDTRACE_1",
                         "PRTAGE": "PRTAGE_1"})
    )
    m = pt.merge(pt1d, on=["hhkey", "PULINENO"], how="left", indicator=True)
    in_reint = m["hhkey"].isin(reint_hh)
    matched = m["_merge"] == "both"

    # A *conflict* means the line is occupied next month by a demonstrably
    # different person. Only compare fields that are present (>=0 / valid); a
    # missing field cannot establish a conflict.
    sex_conflict = m["PESEX_1"].isin([1, 2]) & (m["PESEX_1"] != m["PESEX"])
    race_conflict = (m["PTDTRACE_1"] >= 1) & (m["PTDTRACE_1"] != m["PTDTRACE"])
    dage = m["PRTAGE_1"] - m["PRTAGE"]
    age_conflict = (m["PRTAGE_1"] >= 0) & ~dage.between(AGE_TOL_LO, AGE_TOL_HI)
    reuse = matched & (sex_conflict | race_conflict | age_conflict)
    present = matched & ~reuse

    # Person in a household that did not cleanly re-interview -> fate unobserved.
    # Otherwise: present (validated) -> stayed; absent or line-reused -> left.
    m["fate"] = np.select(
        [~in_reint, present],
        ["hh_not_reinterviewed", "stayed"],
        default="left",
    )
    m["foreign_born"] = m["PRCITSHP"].isin([4, 5])
    pt = m.drop(columns=["_merge"])

    return {
        "persons_t": pt,
        "households_scheduled": sched,
        "hh_table": hh,
    }


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1]
    a, b = sys.argv[2], sys.argv[3]
    dfa = read_month(f"{data_dir}/{a}", state_fips=53)
    dfb = read_month(f"{data_dir}/{b}", state_fips=53)
    out = link_pair(dfa, dfb)
    p = out["persons_t"]
    print(f"WA persons in eligible/interviewed t households: {len(p)}")
    print(p["fate"].value_counts(dropna=False))
    print("\nHousehold re-interview outcomes (scheduled households):")
    print(out["households_scheduled"]["hh_outcome"].value_counts())
