"""Driver: run the Van Hook CPS matching method on ASEC 2023->2024 and
2024->2025, compare with the paper's circa-2000 estimates (Table 2), and
write outputs/replication_summary.{txt,csv}.

Usage: python run_replication.py <data_dir>
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from asec_matching_method import load_asec, load_march_true_mis, matching_method

# Van Hook, Zhang, Bean & Passel (2006), Table 2 (circa 2000)
PAPER = {
    "gross": 3.8, "return": 0.9, "net": 2.9,
    "male": 5.3, "female": 2.3,
    "in_us_0_4": 6.5, "in_us_5_9": 5.0, "in_us_10plus": 2.5,
    "gross_emigrants_000s": 1136, "net_emigrants_000s": 875,
}

# PEINUSYR 2-year entry-cohort mapping (verified empirically per file year):
#   2023 file: ..., 26=2018-19, 27=2020-23(top)
#   2024 file: ..., 26=2018-19, 27=2020-21, 28=2022-24(top)
#   2025 file: same as 2024 with 28=2022-25(top)
PAIRS = [
    dict(t=2023, t1=2024,
         dur_0_4_cats={26, 27}, dur_5_9_cats={24, 25}, ret_cutoff_cat=27),
    dict(t=2024, t1=2025,
         dur_0_4_cats={27, 28}, dur_5_9_cats={25, 26}, ret_cutoff_cat=27),
]


def fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{100*x:.2f}%"


def main():
    data_dir = sys.argv[1]
    os.makedirs("outputs", exist_ok=True)
    lines, rows = [], []

    asec = {y: load_asec(data_dir, y) for y in (2023, 2024, 2025)}
    true_mis = {y: load_march_true_mis(data_dir, y) for y in (2023, 2024, 2025)}

    for spec in PAIRS:
        t, t1 = spec["t"], spec["t1"]
        for scope in ["march_basic"]:
            res = matching_method(
                asec[t], asec[t1], true_mis[t], true_mis[t1],
                spec["dur_0_4_cats"], spec["dur_5_9_cats"],
                spec["ret_cutoff_cat"])
            raw = res["raw"]
            hdr = f"===== ASEC {t} -> {t1}  [{scope}, true-MIS from basic files] ====="
            lines += [
                hdr,
                (f"raw components (adults 15+): "
                 f"u_f={fmt_pct(raw['u_f'])} (n={raw['n_fb_adults']})  "
                 f"u_s={fmt_pct(raw['u_s'])} (n={raw['n_sg_adults']})  "
                 f"m_f={fmt_pct(raw['m_f'])} (n={raw['n_mig_f']})  "
                 f"m_s={fmt_pct(raw['m_s'])} (n={raw['n_mig_s']})"),
                (f"GROSS emigration e = {fmt_pct(res['gross_e'])} "
                 f"(boot SE {fmt_pct(res['gross_se'])})   "
                 f"[paper circa-2000: {PAPER['gross']}%]"),
                (f"return ratio = {fmt_pct(res['ret_ratio'])} "
                 f"(raw {fmt_pct(res['ret_ratio_raw'])}, "
                 f"n_unwt={res['n_return_unwt']})   "
                 f"[paper: {PAPER['return']}%]"),
                (f"NET emigration = {fmt_pct(res['net_e'])}   "
                 f"[paper: {PAPER['net']}%]"),
                (f"FB stock (full ASEC year-t, weighted): "
                 f"{res['fb_stock']/1e6:.1f}M -> implied gross emigrants "
                 f"{res['gross_e']*res['fb_stock']/1e3:,.0f}k/yr, net "
                 f"{res['net_e']*res['fb_stock']/1e3:,.0f}k/yr"),
            ]
            for k, label in [("male", "male"), ("female", "female"),
                             ("in_us_0_4", "0-4 yrs in US"),
                             ("in_us_5_9", "5-9 yrs in US"),
                             ("in_us_10plus", "10+ yrs in US"),
                             ("washington", "WASHINGTON")]:
                r, n = res["subgroups"][k]
                bench = PAPER.get(k)
                bench_s = f"   [paper: {bench}%]" if bench else ""
                lines.append(f"  {label:>14}: e = {fmt_pct(r)} (n={n}){bench_s}")
            wa_e = res["subgroups"]["washington"][0]
            if not pd.isna(wa_e):
                wa_net = wa_e - res["ret_ratio"]
                lines.append(
                    f"  WA: FB stock {res['wa_fb_stock']/1e3:,.0f}k; gross rate "
                    f"{fmt_pct(wa_e)}; net {fmt_pct(wa_net)} -> "
                    f"~{wa_net*res['wa_fb_stock']/1e3:,.1f}k net emigrants/yr")
            lines.append("")
            rows.append({
                "pair": f"{t}->{t1}", "scope": scope,
                **{f"raw_{k}": v for k, v in raw.items()},
                "gross_e": res["gross_e"], "gross_se": res["gross_se"],
                "ret_ratio": res["ret_ratio"], "net_e": res["net_e"],
                "fb_pop": res["fb_pop"], "wa_fb_pop": res["wa_fb_pop"],
                "fb_stock": res["fb_stock"], "wa_fb_stock": res["wa_fb_stock"],
                **{f"e_{k}": v[0] for k, v in res["subgroups"].items()},
                **{f"n_{k}": v[1] for k, v in res["subgroups"].items()},
            })

    text = "\n".join(lines)
    print(text)
    with open("outputs/replication_summary.txt", "w") as f:
        f.write(text)
    pd.DataFrame(rows).to_csv("outputs/replication_summary.csv", index=False)
    print("wrote outputs/replication_summary.{txt,csv}")


if __name__ == "__main__":
    main()
