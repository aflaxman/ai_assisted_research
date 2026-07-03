"""Multi-year driver: run the Van Hook CPS matching method on every ASEC pair
2019->2020 through 2024->2025, with state-level and stratified estimates.

Outputs:
  outputs/multiyear_summary.csv   national gross/net per pair, components, SEs
  outputs/state_rates.csv         per pair x state: standardized e, raw u_f, n
  outputs/strata_rates.csv        per pair x stratifier x level

Usage: python run_multiyear.py <data_dir> [t_start t_end]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from asec_matching_method import (entry_cats, load_asec, load_march_true_mis,
                                  matching_method)

STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def wavg(d, col, wcol="MARSUPWT"):
    d = d.dropna(subset=[col])
    if len(d) == 0 or d[wcol].sum() == 0:
        return np.nan
    return float(np.average(d[col].astype(float), weights=d[wcol]))


def main():
    data_dir = sys.argv[1]
    t0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
    t1_last = int(sys.argv[3]) if len(sys.argv) > 3 else 2024
    os.makedirs("outputs", exist_ok=True)

    years = list(range(t0, t1_last + 2))
    asec, mis = {}, {}
    for y in years:
        print(f"loading ASEC {y} ...", flush=True)
        asec[y] = load_asec(data_dir, y)
        mis[y] = load_march_true_mis(data_dir, y)

    nat_rows, state_rows, strata_rows = [], [], []

    for t in range(t0, t1_last + 1):
        t1 = t + 1
        top_t = int(asec[t].loc[asec[t]["foreign_born"], "PEINUSYR"].max())
        top_t1 = int(asec[t1].loc[asec[t1]["foreign_born"], "PEINUSYR"].max())
        dur04, dur59, _ = entry_cats(t, top_t)
        dur04_t1, dur59_t1, ret_cut = entry_cats(t1, top_t1)
        print(f"pair {t}->{t1}: top_cat={top_t}, dur04={sorted(dur04)}, "
              f"dur59={sorted(dur59)}, ret_cutoff={ret_cut}", flush=True)
        res = matching_method(asec[t], asec[t1], mis[t], mis[t1],
                              dur04, dur59, ret_cut,
                              dur_0_4_cats_t1=dur04_t1,
                              dur_5_9_cats_t1=dur59_t1)
        fb = res["fb_all"]
        pair = f"{t}->{t1}"
        nat_rows.append({
            "pair": pair, **{f"raw_{k}": v for k, v in res["raw"].items()},
            "gross_e": res["gross_e"], "gross_e_dur": res["gross_e_dur"],
            "raw_gross_e": res["raw_gross_e"],
            "gross_se_full": res["gross_se_full"], "gross_se_fixed": res["gross_se"],
            "ret_ratio": res["ret_ratio"], "net_e": res["net_e"],
            "fb_stock": res["fb_stock"], "n_fb_all": res["n_fb_all"],
        })
        print(f"  gross={res['gross_e']*100:.2f}% (raw {res['raw_gross_e']*100:.2f}%, "
              f"duration-aware {res['gross_e_dur']*100:.2f}%) "
              f"+/-{res['gross_se_full']*100:.2f}  net={res['net_e']*100:.2f}%",
              flush=True)

        # ---- states ------------------------------------------------------
        for fips, g in fb.groupby("GESTFIPS"):
            st = STATE_FIPS.get(int(fips), str(fips))
            stock = float(asec[t].loc[asec[t]["foreign_born"]
                                      & (asec[t]["GESTFIPS"] == fips),
                                      "MARSUPWT"].sum())
            g_ad = g[g["A_AGE"] >= 15]
            state_rows.append({
                "pair": pair, "state": st, "n": len(g),
                "n_adults": len(g_ad),
                "e_adj": wavg(g, "e_i"), "e_adj_dur": wavg(g, "e_i_dur"),
                "raw_u_f": wavg(g, "raw_nonfollowup"),
                "raw_u_f_adults": wavg(g_ad, "raw_nonfollowup"),
                "fb_stock": stock,
            })

        # ---- strata ------------------------------------------------------
        fb = fb.copy()
        fb["sex"] = np.where(fb["male"], "male", "female")
        fb["duration"] = np.select(
            [fb["PEINUSYR"].isin(dur04), fb["PEINUSYR"].isin(dur59),
             (fb["PEINUSYR"] > 0) & (fb["PEINUSYR"] < min(dur59))],
            ["0-4 yrs", "5-9 yrs", "10+ yrs"], default="unknown")
        # household-income quartiles among the foreign-born (weighted)
        qs = np.quantile(fb["HTOTVAL"], [0.25, 0.5, 0.75])
        fb["income_q"] = pd.cut(fb["HTOTVAL"],
                                [-np.inf, *qs, np.inf],
                                labels=["q1_low", "q2", "q3", "q4_high"])
        strats = ["sex", "agegrp", "citizenship", "region_birth", "race_eth",
                  "educ4", "income_q", "duration", "occ_major", "ind_major",
                  "student"]
        for s in strats:
            frame = fb
            if s == "educ4":
                frame = fb[fb["A_AGE"] >= 25]
            elif s in ("occ_major", "ind_major"):
                # employed adults with an occupation/industry code
                frame = fb[(fb["A_AGE"] >= 15) & fb[s].notna()]
            elif s == "student":
                # A_HSCOL enrollment universe
                frame = fb[fb["A_AGE"].between(16, 54)]
            for lvl, g in frame.groupby(s, observed=True):
                if len(g) == 0:
                    continue
                strata_rows.append({
                    "pair": pair, "stratifier": s, "level": str(lvl),
                    "n": len(g), "e_adj": wavg(g, "e_i"),
                    "e_adj_dur": wavg(g, "e_i_dur"),
                    "raw_u_f": wavg(g, "raw_nonfollowup"),
                    "wpop": float(g["MARSUPWT"].sum()),
                })

    pd.DataFrame(nat_rows).to_csv("outputs/multiyear_summary.csv", index=False)
    pd.DataFrame(state_rows).to_csv("outputs/state_rates.csv", index=False)
    pd.DataFrame(strata_rows).to_csv("outputs/strata_rates.csv", index=False)
    print("wrote outputs/multiyear_summary.csv, state_rates.csv, strata_rates.csv")


if __name__ == "__main__":
    main()
