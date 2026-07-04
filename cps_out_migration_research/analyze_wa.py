"""Full-year CPS month-to-month departure analysis for Washington State (and
the nation for context), 2024.

Outputs:
  outputs/pair_summary.csv        per-month-pair counts and weighted rates
  outputs/pooled_summary.txt      pooled WA + US figures and interpretation
  outputs/figure_departure.png    visual: where the "out-migration" signal goes
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from cps_parse import read_month
from link_months import link_pair

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
WA_FIPS = 53


def annualize(monthly_rate: float) -> float:
    return 1.0 - (1.0 - monthly_rate) ** 12


def analyze(data_dir: str, state_fips: int | None, label: str):
    """Pool all consecutive month-pairs for one geography."""
    pair_rows = []
    pooled_persons = []
    pooled_hh = []

    cache = {}

    def load(mon):
        if mon not in cache:
            f = f"{data_dir}/{mon}24pub.dat.gz"
            cache[mon] = read_month(f, state_fips=state_fips)
        return cache[mon]

    for i in range(len(MONTHS) - 1):
        a, b = MONTHS[i], MONTHS[i + 1]
        out = link_pair(load(a), load(b))
        p = out["persons_t"]
        sched = out["households_scheduled"]
        p = p.assign(pair=f"{a}->{b}")
        sched = sched.assign(pair=f"{a}->{b}")
        pooled_persons.append(p)
        pooled_hh.append(sched)

        # Persons in re-interviewed households (clean denominator for the
        # individual departure rate).
        clean = p[p["fate"].isin(["stayed", "left"])]
        left = clean["fate"] == "left"
        w = clean["PWSSWGT"]
        wr = w[left].sum() / w.sum() if w.sum() > 0 else np.nan
        pair_rows.append({
            "pair": f"{a}->{b}",
            "n_persons_clean": len(clean),
            "n_left": int(left.sum()),
            "leaver_rate_unwt": left.mean() if len(clean) else np.nan,
            "leaver_rate_wt": wr,
            "n_hh_scheduled": len(sched),
            "n_hh_reint": int((sched["hh_outcome"] == "reinterviewed").sum()),
            "n_hh_typeA": int((sched["hh_outcome"] == "typeA_noninterview").sum()),
            "n_hh_gone_or_vacant": int(sched["hh_outcome"].isin(
                ["gone_from_sample", "typeB_vacant_ineligible",
                 "typeC_ineligible"]).sum()),
        })

    persons = pd.concat(pooled_persons, ignore_index=True)
    hh = pd.concat(pooled_hh, ignore_index=True)
    pair_df = pd.DataFrame(pair_rows)

    # ---- Pooled individual departure rate (re-interviewed households) ------
    clean = persons[persons["fate"].isin(["stayed", "left"])].copy()
    left = clean["fate"] == "left"
    w = clean["PWSSWGT"]
    pooled = {
        "label": label,
        "n_person_pairs_clean": len(clean),
        "n_left": int(left.sum()),
        "monthly_leaver_rate_unwt": float(left.mean()),
        "monthly_leaver_rate_wt": float(w[left].sum() / w.sum()),
    }
    pooled["annual_leaver_rate_wt"] = annualize(pooled["monthly_leaver_rate_wt"])

    # ---- By nativity -------------------------------------------------------
    for name, sub in [("native", clean[~clean["foreign_born"]]),
                      ("foreign_born", clean[clean["foreign_born"]])]:
        l = sub["fate"] == "left"
        ww = sub["PWSSWGT"]
        pooled[f"monthly_leaver_rate_wt_{name}"] = (
            float(ww[l].sum() / ww.sum()) if ww.sum() > 0 else np.nan)
        pooled[f"n_{name}_clean"] = len(sub)
        pooled[f"n_{name}_left"] = int(l.sum())

    # ---- Household attrition (scheduled-to-return households) ---------------
    n_sched = len(hh)
    vc = hh["hh_outcome"].value_counts()
    pooled["n_hh_scheduled"] = n_sched
    for k in ["reinterviewed", "typeA_noninterview", "gone_from_sample",
              "typeB_vacant_ineligible", "typeC_ineligible"]:
        pooled[f"hh_{k}"] = int(vc.get(k, 0))
    pooled["hh_not_reinterviewed_rate"] = (
        1 - vc.get("reinterviewed", 0) / n_sched) if n_sched else np.nan

    return pair_df, pooled, persons, hh


def main():
    data_dir = sys.argv[1]
    os.makedirs("outputs", exist_ok=True)

    wa_pairs, wa_pooled, wa_persons, wa_hh = analyze(data_dir, WA_FIPS, "Washington")
    us_pairs, us_pooled, _, _ = analyze(data_dir, None, "United States")

    wa_pairs.to_csv("outputs/pair_summary_wa.csv", index=False)
    us_pairs.to_csv("outputs/pair_summary_us.csv", index=False)

    lines = []
    for pooled in (wa_pooled, us_pooled):
        lines.append(f"===== {pooled['label']} (CPS basic monthly, 2024, "
                     f"11 consecutive month-pairs) =====")
        lines.append(
            f"Individual departure rate from RE-INTERVIEWED households:\n"
            f"  unweighted: {pooled['monthly_leaver_rate_unwt']*100:.2f}% / month "
            f"({pooled['n_left']} leavers / {pooled['n_person_pairs_clean']} person-pairs)\n"
            f"  weighted  : {pooled['monthly_leaver_rate_wt']*100:.2f}% / month "
            f"-> annualized ~{pooled['annual_leaver_rate_wt']*100:.1f}% / year")
        lines.append(
            f"  by nativity (weighted monthly): "
            f"native {pooled['monthly_leaver_rate_wt_native']*100:.2f}% "
            f"(n_left={pooled['n_native_left']}/{pooled['n_native_clean']}) | "
            f"foreign-born {pooled['monthly_leaver_rate_wt_foreign_born']*100:.2f}% "
            f"(n_left={pooled['n_foreign_born_left']}/{pooled['n_foreign_born_clean']})")
        lines.append(
            f"Household re-interview outcomes (scheduled to return, n="
            f"{pooled['n_hh_scheduled']}):\n"
            f"  reinterviewed           : {pooled['hh_reinterviewed']}\n"
            f"  Type A noninterview     : {pooled['hh_typeA_noninterview']}\n"
            f"  gone from sample        : {pooled['hh_gone_from_sample']}\n"
            f"  Type B vacant/ineligible: {pooled['hh_typeB_vacant_ineligible']}\n"
            f"  Type C ineligible       : {pooled['hh_typeC_ineligible']}\n"
            f"  -> not cleanly reinterviewed: "
            f"{pooled['hh_not_reinterviewed_rate']*100:.1f}% of scheduled households")
        lines.append("")

    text = "\n".join(lines)
    with open("outputs/pooled_summary.txt", "w") as f:
        f.write(text)
    print(text)

    make_figure(wa_pooled)
    # Persist pooled dicts for the writeup.
    pd.DataFrame([wa_pooled, us_pooled]).to_csv("outputs/pooled_summary.csv", index=False)


def make_figure(wa):
    """A funnel showing the 'out-migration' signal dissolving into
    unidentifiable categories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    cats = [
        ("Persons observed\nleaving a continuing\nWA household", 100,
         "#2c7fb7"),
        ("Could be a domestic\nmove (another WA or\nU.S. address)", 70, "#7fb7d6"),
        ("Could be death or\nentry to institution/\nmilitary", 20, "#bcbddc"),
        ("Could be an\ninternational move\n(EMIGRATION)", 10, "#f03b20"),
    ]
    labels = [c[0] for c in cats]
    vals = [c[1] for c in cats]
    colors = [c[2] for c in cats]
    y = np.arange(len(cats))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="white")
    for yi, (lab, v, _) in zip(y, cats):
        ax.text(2, yi, lab, va="center", ha="left", fontsize=9,
                color="black", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Illustrative share of observed departures (NOT measured — "
                  "see text)", fontsize=9)
    ax.set_title("The CPS sees the departure, never the destination\n"
                 "Why the rotating panel cannot isolate international "
                 "out-migration", fontsize=11, fontweight="bold")
    note = (f"WA 2024: ~{wa['monthly_leaver_rate_wt']*100:.1f}% of people in a "
            f"re-interviewed household leave it each month\n"
            f"(~{wa['annual_leaver_rate_wt']*100:.0f}%/yr). The CPS public file "
            f"records NO destination for any of them.")
    ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center",
            fontsize=8.5, style="italic", color="#444")
    plt.tight_layout()
    plt.savefig("outputs/figure_departure.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_departure.png")


if __name__ == "__main__":
    main()
