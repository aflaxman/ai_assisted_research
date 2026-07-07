"""
Grid-search the cohort-definition degrees of freedom to find the combination that
reproduces the paper's N = 4176 (and mean age 53.3). Reuses the cached NHANES files
downloaded by reconstruct_cohort.py.

Degrees of freedom searched:
  - sample domain:   MEC (all examinees) | FASTING (WTSAF2YR>0)
  - criteria:        HbA1c only | HbA1c|FPG | HbA1c|FPG|OGTT
  - exclusion:       diagnosed-only | +HbA1c>=6.5 | +FPG>=126 | full lab (incl OGTT>=200)
  - pregnancy:       keep all | exclude pregnant
  - complete-case:   off | require BMI+SBP+HbA1c+creatinine

Prints every combination ranked by |N - 4176|.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from reconstruct_cohort import load_cycle, CYCLES, egfr_ckdepi_2021, wmean, wsd

TARGET_N, TARGET_AGE = 4176, 53.3


def build():
    df = pd.concat([load_cycle(c, y, s) for c, (y, s) in CYCLES.items()], ignore_index=True)
    df["age"] = df["RIDAGEYR"]
    df["female"] = (df["RIAGENDR"] == 2).astype(float)
    df["hba1c"] = df["LBXGH"]
    df["fpg"] = df["LBXGLU"]
    df["fpg_mmol"] = df["LBXGLU"] / 18.0
    df["ogtt"] = df["LBXGLT"]
    df["bmi"] = df["BMXBMI"]
    sbp = df[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].replace(0, np.nan)
    df["sbp"] = sbp.mean(axis=1, skipna=True)
    df["scr"] = df["LBXSCR"]
    df["wtsaf6"] = df["WTSAF2YR"] / 3.0
    df["wtmec6"] = df["WTMEC2YR"] / 3.0
    df["pregnant"] = (df["RIDEXPRG"] == 1)
    df["complete"] = (df["bmi"].notna() & df["sbp"].notna()
                      & df["hba1c"].notna() & df["scr"].notna())
    return df


def main():
    df = build()
    adult = df["age"] >= 18
    diagnosed = df["DIQ010"] == 1
    hb_pre = (df["hba1c"] >= 5.7) & (df["hba1c"] <= 6.4)
    fpg_pre = (df["fpg"] >= 100) & (df["fpg"] <= 125)
    ogtt_pre = (df["ogtt"] >= 140) & (df["ogtt"] <= 199)
    fasting = df["WTSAF2YR"].notna() & (df["WTSAF2YR"] > 0)

    criteria = {
        "H": hb_pre,
        "H|F": hb_pre | fpg_pre,
        "H|F|O": hb_pre | fpg_pre | ogtt_pre,
    }
    exclusions = {
        "dx-only": diagnosed,
        "+a1c>=6.5": diagnosed | (df["hba1c"] >= 6.5),
        "+fpg>=126": diagnosed | (df["hba1c"] >= 6.5) | (df["fpg"] >= 126),
        "full-lab": diagnosed | (df["hba1c"] >= 6.5) | (df["fpg"] >= 126) | (df["ogtt"] >= 200),
    }
    domains = {"MEC": (pd.Series(True, index=df.index), "wtmec6"),
               "FAST": (fasting, "wtsaf6")}

    rows = []
    for (cname, crit), (ename, excl), (dname, (dom, wcol)), preg, cc in itertools.product(
            criteria.items(), exclusions.items(), domains.items(), [False, True], [False, True]):
        mask = adult & ~excl & crit & dom
        if preg:
            mask = mask & ~df["pregnant"]
        if cc:
            mask = mask & df["complete"]
        sub = df[mask]
        n = len(sub)
        age = wmean(sub["age"], sub[wcol]) if n else np.nan
        rows.append({
            "criteria": cname, "exclude": ename, "domain": dname,
            "no_preg": preg, "complete": cc, "weight": wcol,
            "N": n, "wt_age": round(age, 1), "dN": abs(n - TARGET_N),
        })

    res = pd.DataFrame(rows).sort_values(["dN", "criteria"]).reset_index(drop=True)
    print(f"\nTarget: N={TARGET_N}, age={TARGET_AGE}\n")
    print("==== Top 15 definitions closest to N=4176 ====")
    with pd.option_context("display.width", 130, "display.max_rows", None):
        print(res.head(15).to_string(index=False))
    res.to_csv("outputs/definition_search.csv", index=False)
    print("\nSaved full grid: outputs/definition_search.csv")


if __name__ == "__main__":
    main()
