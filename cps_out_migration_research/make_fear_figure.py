"""Figures for the fear thermometer.

figure_fear.png     — Panel A: monthly national fear index (foreign-born
                      minus second-generation excess attrition), with
                      3-month moving average and the Jan 20, 2025 marker.
                      Panel B: decomposition into the refusal gap (T4,
                      avoidance) and the departure gap (T2+T3).
figure_fear_map.png — state tile-grid map: pooled 2025-26 fear index minus
                      the pooled 2023-24 baseline, for states with enough
                      sample.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# classic tile-grid coordinates (col, row); row 0 at top
TILE = {
    "AK": (0, 0), "ME": (10, 0),
    "VT": (9, 1), "NH": (10, 1),
    "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2), "MN": (4, 2),
    "IL": (5, 2), "WI": (6, 2), "MI": (7, 2), "NY": (8, 2), "RI": (9, 2),
    "MA": (10, 2),
    "OR": (0, 3), "NV": (1, 3), "WY": (2, 3), "SD": (3, 3), "IA": (4, 3),
    "IN": (5, 3), "OH": (6, 3), "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3),
    "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4),
    "KY": (5, 4), "WV": (6, 4), "VA": (7, 4), "MD": (8, 4), "DE": (9, 4),
    "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5),
    "NC": (6, 5), "SC": (7, 5), "DC": (8, 5),
    "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6),
    "HI": (0, 7), "TX": (3, 7), "FL": (8, 7),
}

MIN_N_FB, MIN_N_2G = 1500, 700


def main():
    nat = pd.read_csv("outputs/fear_monthly.csv", parse_dates=["date"])
    # Monthly-equivalent rates for any bridged pair (the never-collected
    # October 2025 forces a sep->nov two-month span).
    span = nat.get("months_span", pd.Series(1, index=nat.index)).fillna(1)
    for col in ["fear_index", "fear_refusal", "depart_gap"]:
        nat[col] = nat[col] / span

    # ---- figure_fear.png -------------------------------------------------
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True)
    x = nat["date"]
    fi = nat["fear_index"] * 100
    axA.axhline(0, color="#333", lw=0.8)
    axA.plot(x, fi, "o-", ms=3.5, lw=1, color="#b8cfe4",
             label="monthly fear index")
    ma = fi.rolling(3, center=True).mean()
    axA.plot(x, ma, lw=2.6, color="#d1495b", label="3-month moving average")
    inaug = pd.Timestamp("2025-01-20")
    axA.axvline(inaug, color="#333", ls="--", lw=1.2)
    axA.text(inaug, axA.get_ylim()[1] * 0.95, " Jan 20, 2025",
             fontsize=8.5, va="top")
    shut = pd.Timestamp("2025-10-15")
    axA.axvspan(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-31"),
                color="#ddd", zorder=0)
    axA.text(shut, axA.get_ylim()[0] * 0.9 if axA.get_ylim()[0] < 0 else 0.2,
             "Oct 2025 CPS\nnever collected", fontsize=7, ha="center",
             color="#666")
    axA.set_ylabel("Fear index (pp)\nFB − second-gen excess attrition")
    axA.set_title("A. The fear thermometer: monthly foreign-born excess "
                  "panel attrition, Jan 2023 – Apr 2026",
                  fontsize=11.5, fontweight="bold", loc="left")
    axA.legend(fontsize=8.5, loc="upper left")
    axA.grid(alpha=0.25)

    ref = (nat["fear_refusal"] * 100).rolling(3, center=True).mean()
    dep = (nat["depart_gap"] * 100).rolling(3, center=True).mean()
    axB.axhline(0, color="#333", lw=0.8)
    axB.plot(x, ref, lw=2.2, color="#f58518",
             label="refusal gap (T4: household present, refused/no answer) — avoidance")
    axB.plot(x, dep, lw=2.2, color="#4c78a8",
             label="departure gap (T2 roster-confirmed + T3 vacated) — includes emigration")
    axB.axvline(inaug, color="#333", ls="--", lw=1.2)
    axB.set_ylabel("Component gaps (pp, 3-mo MA)")
    axB.set_title("B. Decomposition: leaving the survey vs leaving the "
                  "address", fontsize=11.5, fontweight="bold", loc="left")
    axB.legend(fontsize=8.5, loc="upper left")
    axB.grid(alpha=0.25)
    fig.text(0.5, 0.005,
             "Adults 15+ in continuing-eligible households; weighted; "
             "second generation as control nets out shared nonresponse. "
             "Monthly SE ≈ ±0.5pp.",
             ha="center", fontsize=8, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig("outputs/figure_fear.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_fear.png")

    # ---- figure_fear_map.png ---------------------------------------------
    st = pd.read_csv("outputs/fear_monthly_states.csv", parse_dates=["date"])
    st["era"] = np.where(st["date"] >= "2025-01-01", "post", "base")

    def pool(g):
        wf, w2 = g["n_fb"].sum(), g["n_2g"].sum()
        if wf == 0 or w2 == 0:
            return pd.Series({"fear": np.nan, "n_fb": wf, "n_2g": w2})
        fb_u = np.average(g["fb_u"], weights=g["n_fb"])
        g2_u = np.average(g["2g_u"], weights=g["n_2g"])
        return pd.Series({"fear": fb_u - g2_u, "n_fb": wf, "n_2g": w2})

    pooled = (st.groupby(["fips", "era"])
              .apply(pool, include_groups=False).reset_index())
    wide = pooled.pivot(index="fips", columns="era",
                        values=["fear", "n_fb", "n_2g"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["st"] = wide["fips"].map(STATE_FIPS)
    wide["diff"] = (wide["fear_post"] - wide["fear_base"]) * 100
    ok = (wide["n_fb_post"] >= MIN_N_FB) & (wide["n_2g_post"] >= MIN_N_2G)

    fig, ax = plt.subplots(figsize=(13, 8.6))
    vmax = np.nanmax(np.abs(wide.loc[ok, "diff"])) or 1
    cmap = plt.get_cmap("RdBu_r")
    for r in wide.itertuples():
        if r.st not in TILE:
            continue
        cx, cy = TILE[r.st]
        y = 8 - cy
        if ok.loc[r.Index] and not np.isnan(r.diff):
            color = cmap(0.5 + r.diff / (2 * vmax))
            ax.add_patch(plt.Rectangle((cx, y), 0.94, 0.94, color=color))
            tcol = "white" if abs(r.diff) > vmax * 0.55 else "black"
            ax.text(cx + 0.47, y + 0.60, r.st, ha="center", fontsize=10,
                    fontweight="bold", color=tcol)
            ax.text(cx + 0.47, y + 0.26, f"{r.diff:+.1f}", ha="center",
                    fontsize=8.5, color=tcol)
        else:
            ax.add_patch(plt.Rectangle((cx, y), 0.94, 0.94, color="#eeeeee"))
            ax.text(cx + 0.47, y + 0.45, r.st, ha="center", fontsize=9,
                    color="#999")
    ax.set_xlim(-0.3, 11.3)
    ax.set_ylim(0.5, 9.3)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Change in fear index (pp):\n2025–26 pooled minus 2023–24 "
                 "baseline", fontsize=9)
    ax.set_title('Where the thermometer rose: state change in foreign-born '
                 'excess attrition\n(grey = insufficient sample; '
                 f'shown states have ≥{MIN_N_FB} FB adult obs post-2025)',
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figure_fear_map.png", dpi=140, bbox_inches="tight")
    print("wrote outputs/figure_fear_map.png")


if __name__ == "__main__":
    main()
