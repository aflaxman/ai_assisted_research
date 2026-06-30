"""Tests for the month-to-month CPS linkage logic.

Runs standalone (``python test_link_months.py``) or under pytest. Uses a small
synthetic two-month fixture rather than real CPS files so the expected
classification is known by construction.
"""

from __future__ import annotations

import pandas as pd

from link_months import link_pair

# Columns link_pair reads.
COLS = ["HRHHID", "HRHHID2", "HRINTSTA", "HRMIS", "HRLONGLK",
        "PRPERTYP", "PRTAGE", "PULINENO", "PESEX", "PTDTRACE",
        "PRCITSHP", "PWSSWGT"]


def person(hhid, hh2, intsta, mis, line, age, sex, race, pertyp=2,
           citshp=1, wgt=1000.0, longlk=2):
    return {
        "HRHHID": hhid, "HRHHID2": hh2, "HRINTSTA": intsta, "HRMIS": mis,
        "HRLONGLK": longlk, "PRPERTYP": pertyp, "PRTAGE": age,
        "PULINENO": line, "PESEX": sex, "PTDTRACE": race,
        "PRCITSHP": citshp, "PWSSWGT": wgt,
    }


def build_fixture():
    # Month t (e.g., January). MIS values 2 => eligible to return; 4 => rotates out.
    t = [
        # HH A: interviewed, two people. line1 stays, line2 leaves.
        person("A", "0001", 1, 2, 1, 40, 1, 1),
        person("A", "0001", 1, 2, 2, 38, 2, 1),
        # HH B: interviewed now; will be a Type A noninterview next month.
        person("B", "0001", 1, 2, 1, 50, 1, 2),
        # HH C: interviewed; line 1 will be REUSED by a different person.
        person("C", "0001", 1, 2, 1, 30, 1, 3, citshp=5),  # foreign-born
        # HH D: MIS 4 -> rotates out, must be EXCLUDED from the analysis.
        person("D", "0001", 1, 4, 1, 60, 2, 1),
    ]
    # Month t+1 (February).
    t1 = [
        # HH A: line1 present (same identity), line2 GONE.
        person("A", "0001", 1, 3, 1, 40, 1, 1),
        # HH B: Type A noninterview placeholder (no real person).
        person("B", "0001", 2, 3, 1, 0, 0, 0, pertyp=0),
        # HH C: line 1 now a DIFFERENT person (sex flips) -> original left.
        person("C", "0001", 1, 3, 1, 25, 2, 3, citshp=1),
    ]
    return pd.DataFrame(t, columns=COLS), pd.DataFrame(t1, columns=COLS)


def test_classification():
    df_t, df_t1 = build_fixture()
    out = link_pair(df_t, df_t1)
    p = out["persons_t"].set_index(["hhkey", "PULINENO"])
    fate = p["fate"].to_dict()

    # HH D rotated out -> not present at all.
    assert not any(k[0].startswith("D_") for k in fate), "rotated-out HH leaked in"

    assert fate[("A_0001", 1)] == "stayed"
    assert fate[("A_0001", 2)] == "left"            # individual leaver
    assert fate[("B_0001", 1)] == "hh_not_reinterviewed"  # Type A next month
    assert fate[("C_0001", 1)] == "left"            # line reused by replacement

    counts = out["persons_t"]["fate"].value_counts().to_dict()
    assert counts.get("stayed") == 1
    assert counts.get("left") == 2
    assert counts.get("hh_not_reinterviewed") == 1

    # Foreign-born flag carried through for C.
    fb = out["persons_t"].set_index(["hhkey", "PULINENO"])["foreign_born"].to_dict()
    assert fb[("C_0001", 1)] is True or fb[("C_0001", 1)] == True  # noqa: E712


def test_household_outcomes():
    df_t, df_t1 = build_fixture()
    out = link_pair(df_t, df_t1)
    sched = out["households_scheduled"].set_index("hhkey")["hh_outcome"].to_dict()
    # D excluded (MIS 4). A and C re-interviewed, B Type A.
    assert "D_0001" not in sched
    assert sched["A_0001"] == "reinterviewed"
    assert sched["C_0001"] == "reinterviewed"
    assert sched["B_0001"] == "typeA_noninterview"


if __name__ == "__main__":
    test_classification()
    test_household_outcomes()
    print("all tests passed")
