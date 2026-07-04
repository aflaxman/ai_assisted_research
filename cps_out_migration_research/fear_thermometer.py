"""Leaving the Country or Leaving the Survey? — a high-frequency
"fear thermometer" from CPS month-to-month panel attrition, with a
family-pointer decomposition of departures vs avoidance.

Index construction, per consecutive month-pair t -> t+1:

  fear_index = u_fb - u_2g

where u_g is the share of generation-g adults (15+) in eligible continuing
households (true MIS 1-3 or 5-7 at t) who fail person-level follow-up at
t+1. Second-generation adults (U.S.-born, >=1 foreign-born parent) are the
control, exactly as in Van Hook et al. (2006) — but here read as an
*avoidance index*: shared nonresponse cancels, so the gap moves only when
the foreign-born diverge from their control (departure abroad, or
foreign-born-specific survey withdrawal).

Family-pointer tier decomposition of each person's fate:

  T1 spouse_reported : matched stayer whose spouse (PESPOUSE pointer at t)
                       vanished from the roster AND whose own marital status
                       flipped married-spouse-present -> married-spouse-
                       ABSENT. An affirmative informant report of a specific
                       person's departure (resists concealment).
  T2 roster_confirmed: person absent (or line reused) from a household that
                       WAS re-interviewed — cannot be their own refusal.
  T3 moved_signal    : whole household Type B vacant, gone from the sample
                       files, or Type A "unable to locate" — the address
                       emptied: departure (somewhere), not refusal.
  T4 refusal_signal  : whole household Type A REFUSED / no-one-home /
                       temporarily absent — evidence of presence plus
                       avoidance.

T1 and T2 carry departure information; T4 is the purest avoidance signal;
T3 sits between (movers include domestic movers). All rates are weighted
(PWSSWGT) shares of generation-g adults at month t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cps_parse import read_month

MIS_ELIGIBLE_NEXT = {1, 2, 3, 5, 6, 7}
AGE_LO, AGE_HI = -1, 2

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def month_files(start: str, end: str) -> list[str]:
    """List of 'monYY' tokens from start to end inclusive, e.g. jan23..may26."""
    s_m, s_y = start[:3], int(start[3:])
    e_m, e_y = end[:3], int(end[3:])
    out = []
    y, mi = s_y, MONTHS.index(s_m)
    while (y, mi) <= (e_y, MONTHS.index(e_m)):
        out.append(f"{MONTHS[mi]}{y:02d}")
        mi += 1
        if mi == 12:
            mi = 0
            y += 1
    return out


def prep_month(path: str) -> pd.DataFrame:
    df = read_month(path)
    df["hhkey"] = df["HRHHID"].str.strip() + "_" + df["HRHHID2"].str.strip()
    df["foreign_born"] = df["PRCITSHP"].isin([4, 5])
    df["second_gen"] = (~df["foreign_born"]) & (
        (df["PEFNTVTY"] >= 100) | (df["PEMNTVTY"] >= 100))
    df["third_gen"] = (~df["foreign_born"]) & (~df["second_gen"])
    return df


def link_pair_tiers(df_t: pd.DataFrame, df_t1: pd.DataFrame,
                    eligible_mis: set | None = None) -> pd.DataFrame:
    """Classify every eligible adult at t into stayed / T1-T4 tiers.

    eligible_mis: months-in-sample at t scheduled to be interviewed at t+1.
    Default is the 1-month-gap set; a 2-month bridge (e.g. across the never-
    collected October 2025) must pass {1, 2, 5, 6}.
    """
    if eligible_mis is None:
        eligible_mis = MIS_ELIGIBLE_NEXT
    # Household outcome at t+1
    hh_t1 = (df_t1.groupby("hhkey")
             .agg(intsta_t1=("HRINTSTA", "min"), typea=("HUTYPEA", "max"))
             .reset_index())

    base = df_t[(df_t["HRINTSTA"] == 1)
                & df_t["HRMIS"].isin(list(eligible_mis))
                & (df_t["PRTAGE"] >= 15) & (df_t["PULINENO"] > 0)].copy()
    base = base.merge(hh_t1, on="hhkey", how="left")

    # Person-level match into re-interviewed households (roster presence)
    roster = df_t1[(df_t1["HRINTSTA"] == 1) & (df_t1["PULINENO"] > 0)]
    r = (roster[["hhkey", "PULINENO", "PESEX", "PRTAGE", "PEMARITL"]]
         .drop_duplicates(["hhkey", "PULINENO"])
         .rename(columns={"PESEX": "sex1", "PRTAGE": "age1",
                          "PEMARITL": "maritl1"}))
    m = base.merge(r, on=["hhkey", "PULINENO"], how="left")
    dage = m["age1"] - m["PRTAGE"]
    matched = m["sex1"].notna() & (m["sex1"] == m["PESEX"]) \
        & dage.between(AGE_LO, AGE_HI)

    reint = m["intsta_t1"] == 1
    typea = m["intsta_t1"] == 2
    vac_gone = (m["intsta_t1"].isna()) | (m["intsta_t1"] >= 3)
    unable = typea & (m["typea"] == 5)
    refusal_like = typea & m["typea"].isin([1, 2, 3, 4, 6])

    fate = np.select(
        [reint & matched,
         reint & ~matched,          # T2 roster-confirmed departure
         vac_gone | unable,          # T3 moved signal
         refusal_like],              # T4 refusal/avoidance signal
        ["stayed", "t2_roster_confirmed", "t3_moved_signal",
         "t4_refusal_signal"],
        default="other")
    m["fate"] = fate

    # T1: spouse-reported departure — among matched stayers whose spouse
    # line (at t) is a T2 departure AND whose marital status flipped to
    # "married, spouse absent" (PEMARITL 1 -> 2).
    sp = m[(m["PESPOUSE"] > 0)][["hhkey", "PESPOUSE"]].copy()
    t2_keys = set(map(tuple, m.loc[m["fate"] == "t2_roster_confirmed",
                                   ["hhkey", "PULINENO"]].values))
    spouse_left = [(hk, ln) in t2_keys for hk, ln in
                   zip(sp["hhkey"], sp["PESPOUSE"])]
    sp_left_idx = sp.index[np.array(spouse_left, dtype=bool)]
    flip = (m["PEMARITL"] == 1) & (m["maritl1"] == 2)
    m["t1_spouse_reported"] = False
    m.loc[m.index.isin(sp_left_idx) & flip & (m["fate"] == "stayed"),
          "t1_spouse_reported"] = True
    return m


def pair_summary(m: pd.DataFrame) -> dict:
    """Weighted tier rates by generation, plus the fear index."""
    out = {}
    for g, mask in [("fb", m["foreign_born"]), ("2g", m["second_gen"]),
                    ("3g", m["third_gen"])]:
        d = m[mask]
        w = d["PWSSWGT"].sum()
        if w == 0:
            continue
        out[f"n_{g}"] = len(d)
        for tier in ["t2_roster_confirmed", "t3_moved_signal",
                     "t4_refusal_signal"]:
            out[f"{g}_{tier}"] = float(
                d.loc[d["fate"] == tier, "PWSSWGT"].sum() / w)
        out[f"{g}_u"] = float(
            d.loc[d["fate"] != "stayed", "PWSSWGT"].sum() / w)
        out[f"{g}_t1_spouse_reported"] = float(
            d.loc[d["t1_spouse_reported"], "PWSSWGT"].sum() / w)
    out["fear_index"] = out.get("fb_u", np.nan) - out.get("2g_u", np.nan)
    out["fear_refusal"] = (out.get("fb_t4_refusal_signal", np.nan)
                           - out.get("2g_t4_refusal_signal", np.nan))
    out["depart_gap"] = (
        out.get("fb_t2_roster_confirmed", np.nan)
        + out.get("fb_t3_moved_signal", np.nan)
        - out.get("2g_t2_roster_confirmed", np.nan)
        - out.get("2g_t3_moved_signal", np.nan))
    return out


def run(data_dir: str, start: str, end: str,
        state_rows: list | None = None) -> pd.DataFrame:
    toks = month_files(start, end)
    rows = []
    prev = None
    prev_tok = None
    for tok in toks:
        cur = prep_month(f"{data_dir}/{tok}pub.dat.gz")
        # The October 2025 CPS was never collected (government shutdown);
        # its published file is a 376-row stub. Treat any month with almost
        # no interviews as missing: skip it, letting the next pair bridge
        # the gap (flagged via months_span).
        if (cur["HRINTSTA"] == 1).sum() < 10000:
            print(f"{tok}: file is a stub ({int((cur['HRINTSTA']==1).sum())} "
                  f"interviews) -- treating month as missing", flush=True)
            continue
        if prev is not None:
            mi_prev = MONTHS.index(prev_tok[:3]) + 12 * int(prev_tok[3:])
            mi_cur = MONTHS.index(tok[:3]) + 12 * int(tok[3:])
            span = mi_cur - mi_prev
            if span > 2:
                raise ValueError(f"gap {prev_tok}->{tok} spans {span} months;"
                                 " only 1- and 2-month links are supported")
            emis = {1, 2, 5, 6} if span == 2 else None
            m = link_pair_tiers(prev, cur, eligible_mis=emis)
            s = pair_summary(m)
            s["pair"] = f"{prev_tok}->{tok}"
            s["months_span"] = span
            s["date"] = pd.Timestamp(year=2000 + int(prev_tok[3:]),
                                     month=MONTHS.index(prev_tok[:3]) + 1,
                                     day=15)
            rows.append(s)
            if state_rows is not None:
                for fips, g in m.groupby("GESTFIPS"):
                    ss = pair_summary(g)
                    ss["pair"], ss["date"], ss["fips"] = s["pair"], s["date"], fips
                    ss["months_span"] = s["months_span"]
                    state_rows.append(ss)
            print(f"{s['pair']}: fear={s['fear_index']*100:+.2f}pp "
                  f"(fb u={s.get('fb_u', float('nan'))*100:.1f}%, "
                  f"n={s.get('n_fb', 0)}; "
                  f"2g u={s.get('2g_u', float('nan'))*100:.1f}%, "
                  f"n={s.get('n_2g', 0)})"
                  + (f" [span {s['months_span']}mo]"
                     if s["months_span"] > 1 else ""), flush=True)
        prev, prev_tok = cur, tok
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "jan23"
    end = sys.argv[3] if len(sys.argv) > 3 else "may26"
    state_rows: list = []
    nat = run(data_dir, start, end, state_rows)
    nat.to_csv("outputs/fear_monthly.csv", index=False)
    pd.DataFrame(state_rows).to_csv("outputs/fear_monthly_states.csv",
                                    index=False)
    print("wrote outputs/fear_monthly.csv, fear_monthly_states.csv")
