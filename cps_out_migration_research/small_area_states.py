"""Small-area estimates of foreign-born emigration for ALL states.

Motivation: the per-state numbers in state_rates.csv `e_adj` are averages of
nationally-fit predictions over each state's records -- composition indices
that carry no state-specific behavior. A true state estimate needs the
state's own raw panel attrition, but most state cells are tiny. This module
therefore:

  1. builds a DIRECT state estimate from the state's own raw foreign-born
     adult non-follow-up, anchored to the national decomposition:

        e_s = e_nat + (u_fs - u_f,nat) / (1 - m_f,nat)

     i.e., the national standardized gross rate plus the state's deviation
     in raw attrition, passed through the national internal-migration
     denominator (assumes state deviations in attrition beyond the national
     level reflect emigration rather than state-specific response culture or
     mover rates -- stated caveat);

  2. shrinks the direct estimates toward the national mean with an
     empirical-Bayes random-effects model (Fay-Herriot form, DerSimonian-
     Laird tau^2): with sampling variance
        v_s = deff * u_fs (1 - u_fs) / n_s / (1 - m_nat)^2,   deff = 1.75
     the shrunk estimate is  e~_s = mu + B_s (e_s - mu),
     B_s = tau^2 / (tau^2 + v_s), posterior sd = sqrt(tau^2 v_s/(tau^2+v_s)).

Small states shrink almost fully to the national mean (as they should);
large states keep most of their own signal.

Usage: python small_area_states.py [pair]     (default 2024->2025)
Writes outputs/state_small_area.csv and prints the table.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

DEFF = 1.75  # household clustering of adults within state samples


def eb_shrink(e_direct: np.ndarray, v: np.ndarray):
    """DerSimonian-Laird random-effects shrinkage. Returns (mu, tau2, shrunk,
    posterior_sd)."""
    w = 1.0 / v
    mu_fixed = np.sum(w * e_direct) / np.sum(w)
    q = np.sum(w * (e_direct - mu_fixed) ** 2)
    k = len(e_direct)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / denom)
    w_star = 1.0 / (v + tau2)
    mu = np.sum(w_star * e_direct) / np.sum(w_star)
    b = tau2 / (tau2 + v)
    shrunk = mu + b * (e_direct - mu)
    post_sd = np.sqrt(tau2 * v / (tau2 + v)) if tau2 > 0 else np.sqrt(v * 0)
    return mu, tau2, shrunk, post_sd


def small_area(pair: str, st: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    natrow = nat[nat["pair"] == pair].iloc[0]
    e_nat = natrow["gross_e"]
    u_nat = natrow["raw_u_f"]
    m_nat = natrow["raw_m_f"]

    d = st[(st["pair"] == pair) & (st["n_adults"] >= 20)].copy()
    d["e_direct"] = e_nat + (d["raw_u_f_adults"] - u_nat) / (1.0 - m_nat)
    d["v"] = (DEFF * d["raw_u_f_adults"] * (1 - d["raw_u_f_adults"])
              / d["n_adults"] / (1.0 - m_nat) ** 2)
    mu, tau2, shrunk, post_sd = eb_shrink(d["e_direct"].values, d["v"].values)
    d["e_shrunk"] = shrunk
    d["post_sd"] = post_sd
    d["direct_se"] = np.sqrt(d["v"])
    d["shrink_weight_own"] = tau2 / (tau2 + d["v"])
    d.attrs["mu"], d.attrs["tau"] = mu, np.sqrt(tau2)
    return d.sort_values("e_shrunk", ascending=False)


def main():
    pair = sys.argv[1] if len(sys.argv) > 1 else "2024->2025"
    st = pd.read_csv("outputs/state_rates.csv")
    nat = pd.read_csv("outputs/multiyear_summary.csv")

    out_frames = []
    for p in [pair, "baseline_2019_24"]:
        if p == "baseline_2019_24":
            # Pool the pre-2025 pairs: state raw attrition pooled over pairs,
            # national references pooled the same way.
            pre = st[st["pair"] != pair]
            pooled = (pre.groupby("state")
                      .apply(lambda g: pd.Series({
                          "n_adults": g["n_adults"].sum(),
                          "n": g["n"].sum(),
                          "raw_u_f_adults": np.average(
                              g["raw_u_f_adults"], weights=g["n_adults"]),
                          "fb_stock": g["fb_stock"].mean()}),
                          include_groups=False)
                      .reset_index())
            pooled["pair"] = p
            natp = nat[nat["pair"] != pair]
            nat_pool = pd.DataFrame([{
                "pair": p,
                "gross_e": np.average(natp["gross_e"],
                                      weights=natp["raw_n_fb_adults"]),
                "raw_u_f": np.average(natp["raw_u_f"],
                                      weights=natp["raw_n_fb_adults"]),
                "raw_m_f": np.average(natp["raw_m_f"],
                                      weights=natp["raw_n_mig_f"]),
            }])
            res = small_area(p, pooled, nat_pool)
        else:
            res = small_area(p, st, nat)
        res["window"] = p
        print(f"\n===== Small-area state estimates [{p}] "
              f"(mu={res.attrs['mu']*100:.2f}%, tau={res.attrs['tau']*100:.2f}pp) =====")
        for r in res.itertuples():
            print(f"  {r.state}: shrunk {r.e_shrunk*100:+5.1f}% ± {r.post_sd*100:.1f} "
                  f"(direct {r.e_direct*100:+5.1f}% ± {r.direct_se*100:.1f}, "
                  f"n_adults={r.n_adults}, own-signal weight "
                  f"{r.shrink_weight_own*100:.0f}%)")
        out_frames.append(res)

    pd.concat(out_frames, ignore_index=True).to_csv(
        "outputs/state_small_area.csv", index=False)
    print("\nwrote outputs/state_small_area.csv")


if __name__ == "__main__":
    main()
