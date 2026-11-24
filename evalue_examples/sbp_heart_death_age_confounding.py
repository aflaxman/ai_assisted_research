"""Age confounding of the SBP / heart-disease-death association, NHANES.

A companion to the acetaminophen/ASD replication: an example where the
confounder (age) is measured, so we can watch the E-value logic work.
Pretend age were unmeasured. The crude association between
hypertensive-range systolic blood pressure (SBP >= 140 mm Hg) and death
from heart disease has an E-value; age's actual associations with both
exposure and outcome exceed it, so age alone could produce most of the
crude association -- and age adjustment indeed collapses the estimate.

Data (all openly available, already downloaded under
../nhanes_mortality_fibrosis/data/raw/):
- NHANES 2003-2010 demographics (DEMO) and blood pressure exam (BPX),
  https://wwwn.cdc.gov/nchs/nhanes/
- NCHS public-use Linked Mortality Files, follow-up through Dec 31 2019,
  https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

Outcome is death with underlying cause "diseases of heart"
(UCOD_LEADING = 1), the closest openly available proxy for IHD
incidence; the public file does not separate ischemic heart disease.

Survey weights: MEC exam weights (WTMEC2YR / 4 for four combined
cycles) are applied to every estimate. Point estimates only; design-based
CIs would additionally need SDMVPSU/SDMVSTRA Taylor linearization.

Run: python sbp_heart_death_age_confounding.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evalue import bias_factor, evalue

RAW = Path(__file__).parent.parent / "nhanes_mortality_fibrosis" / "data" / "raw"
OUT = Path(__file__).parent / "outputs"

CYCLES = {"2003_2004": "C", "2005_2006": "D", "2007_2008": "E", "2009_2010": "F"}
N_CYCLES = len(CYCLES)
WINDOW_MONTHS = 105  # ~9 years: fully observed for every cycle by Dec 2019
AGE_STRATA = [(20, 39), (40, 59), (60, 74), (75, 200)]
SBP_CUT = 140.0
OLD_AGE = 60  # the "confounder" dichotomy used for the bias-factor check

# Validated categorical palette, slots 1-2 (dataviz reference palette).
BLUE, ORANGE = "#2a78d6", "#eb6834"


def parse_lmf(path):
    """Parse an NCHS public-use linked mortality fixed-width file."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\r\n").ljust(48)

            def safe(s, to_type=int):
                s = s.strip().replace(".", "")
                return to_type(s) if s else None

            rows.append(
                {
                    "SEQN": safe(p[0:14]),
                    "ELIGSTAT": safe(p[14:15]),
                    "MORTSTAT": safe(p[15:16]),
                    "UCOD_LEADING": safe(p[16:19]),
                    "PERMTH_EXM": safe(p[45:48], float),
                }
            )
    return pd.DataFrame(rows)


def load_cohort():
    parts = []
    for cycle, suffix in CYCLES.items():
        demo = pd.read_sas(RAW / "nhanes" / cycle / f"DEMO_{suffix}.xpt", format="xport")
        bpx = pd.read_sas(RAW / "nhanes" / cycle / f"BPX_{suffix}.xpt", format="xport")
        lmf = parse_lmf(RAW / "lmf" / f"NHANES_{cycle}_MORT_2019_PUBLIC.dat")
        df = demo[["SEQN", "RIDAGEYR", "WTMEC2YR"]].merge(
            bpx[["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]], on="SEQN"
        )
        df = df.merge(lmf, on="SEQN")
        df["cycle"] = cycle
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    df["SBP"] = df[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1)
    df = df[
        (df.RIDAGEYR >= 20)
        & df.SBP.notna()
        & (df.WTMEC2YR > 0)
        & (df.ELIGSTAT == 1)
        & df.MORTSTAT.isin([0, 1])
    ].copy()
    df["W"] = df.WTMEC2YR / N_CYCLES
    df["exposed"] = df.SBP >= SBP_CUT
    df["heart_death"] = (
        (df.MORTSTAT == 1) & (df.UCOD_LEADING == 1) & (df.PERMTH_EXM <= WINDOW_MONTHS)
    )
    df["old"] = df.RIDAGEYR >= OLD_AGE
    return df


def wrisk(df):
    """Weighted cumulative risk of heart-disease death in the window."""
    return (df.W * df.heart_death).sum() / df.W.sum()


def main():
    OUT.mkdir(exist_ok=True)
    df = load_cohort()

    alive_min_fu = df.loc[df.MORTSTAT == 0, "PERMTH_EXM"].min()
    print(f"Adults 20+ with measured SBP and mortality linkage: {len(df):,}")
    print(f"Heart-disease deaths within {WINDOW_MONTHS} months: "
          f"{int(df.heart_death.sum()):,} (unweighted)")
    print(f"Minimum follow-up among survivors: {alive_min_fu:.0f} months")
    assert alive_min_fu >= WINDOW_MONTHS, "window not fully observed for all cycles"
    print()

    exp, unexp = df[df.exposed], df[~df.exposed]

    # -- Crude association -------------------------------------------------
    r1, r0 = wrisk(exp), wrisk(unexp)
    rr_crude = r1 / r0
    e_crude = evalue(rr_crude)
    print(f"Crude {WINDOW_MONTHS}-month heart-death risk: "
          f"{r1:.2%} (SBP>={SBP_CUT:.0f}) vs {r0:.2%} (SBP<{SBP_CUT:.0f})")
    print(f"  Crude RR: {rr_crude:.2f}   E-value: {e_crude:.2f}")
    print()

    # -- How strong a confounder is age, actually? -------------------------
    # RR_EU: prevalence of old age among exposed vs unexposed.
    p_old_exp = (exp.W * exp.old).sum() / exp.W.sum()
    p_old_unexp = (unexp.W * unexp.old).sum() / unexp.W.sum()
    rr_eu = p_old_exp / p_old_unexp
    # RR_UD: heart-death risk in old vs young, within each exposure group
    # (the bounding factor uses the maximum).
    rr_ud = max(
        wrisk(g[g.old]) / wrisk(g[~g.old]) for g in (exp, unexp)
    )
    b = bias_factor(rr_eu, rr_ud)
    print(f"Age {OLD_AGE}+ prevalence: {p_old_exp:.1%} among exposed vs "
          f"{p_old_unexp:.1%} among unexposed  -> RR_EU = {rr_eu:.2f}")
    print(f"Heart-death risk, age {OLD_AGE}+ vs <{OLD_AGE}: RR_UD = {rr_ud:.2f}")
    print(f"  Bounding factor B = {b:.2f} "
          f"({'>=' if b >= rr_crude else '<'} crude RR {rr_crude:.2f}); "
          f"E-value condition needs both >= {e_crude:.2f}: "
          f"{'met' if min(rr_eu, rr_ud) >= e_crude else 'not met'}")
    print(f"  A confounder this strong could shrink the crude RR to "
          f"{rr_crude / b:.2f} on its own.")
    print()

    # -- Age-stratified and age-standardized -------------------------------
    rows = []
    total_w = df.W.sum()
    std_r1 = std_r0 = 0.0
    for lo, hi in AGE_STRATA:
        s = df[(df.RIDAGEYR >= lo) & (df.RIDAGEYR <= hi)]
        s1, s0 = s[s.exposed], s[~s.exposed]
        risk1, risk0 = wrisk(s1), wrisk(s0)
        share = s.W.sum() / total_w
        std_r1 += share * risk1
        std_r0 += share * risk0
        label = f"{lo}-{hi}" if hi < 150 else f"{lo}+"
        deaths = int(s1.heart_death.sum()), int(s0.heart_death.sum())
        flag = " (small cell: interpret with caution)" if min(deaths) < 30 else ""
        rows.append(
            {"stratum": label, "risk_exposed": risk1, "risk_unexposed": risk0,
             "rr": risk1 / risk0, "deaths_exposed": deaths[0],
             "deaths_unexposed": deaths[1], "small_cell": min(deaths) < 30}
        )
        print(f"Age {label:>6}: {risk1:6.2%} vs {risk0:6.2%}  "
              f"RR {risk1 / risk0:4.2f}  "
              f"(unweighted deaths {deaths[0]}/{deaths[1]}){flag}")

    rr_std = std_r1 / std_r0
    print()
    print(f"Age-standardized risks: {std_r1:.2%} vs {std_r0:.2%}")
    print(f"  Age-standardized RR: {rr_std:.2f}   E-value: {evalue(rr_std):.2f}")
    print()
    print(f"Adjusting for age alone moves the RR from {rr_crude:.2f} to "
          f"{rr_std:.2f} -- most of the crude association was age.")
    print("Unlike the acetaminophen example, the adjusted association stays")
    print("above the null, consistent with SBP truly causing heart disease.")

    results = pd.DataFrame(
        rows
        + [
            {"stratum": "crude", "risk_exposed": r1, "risk_unexposed": r0,
             "rr": rr_crude,
             "deaths_exposed": int(exp.heart_death.sum()),
             "deaths_unexposed": int(unexp.heart_death.sum())},
            {"stratum": "age_standardized", "risk_exposed": std_r1,
             "risk_unexposed": std_r0, "rr": rr_std},
        ]
    )
    results.to_csv(OUT / "sbp_heart_death_results.csv", index=False)
    print(f"\nWrote {OUT / 'sbp_heart_death_results.csv'}")

    make_figure(rows, r1, r0, rr_crude, rr_std)


def make_figure(rows, crude_r1, crude_r0, rr_crude, rr_std):
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    labels = [r["stratum"] for r in rows] + ["all ages\n(crude)"]
    pairs = [(r["risk_unexposed"], r["risk_exposed"]) for r in rows]
    pairs.append((crude_r0, crude_r1))
    rr_labels = [f"RR {r['rr']:.1f}{'*' if r['small_cell'] else ''}" for r in rows]
    rr_labels.append(f"RR {rr_crude:.1f}")
    xs = list(range(len(rows))) + [len(rows) + 0.6]

    dot = dict(s=80, zorder=2, edgecolor="#fcfcfb", linewidth=1.5)
    for x, (lo, hi), rr_label in zip(xs, pairs, rr_labels):
        ax.plot([x, x], [lo * 100, hi * 100], color="#c3c2b7", lw=2, zorder=1)
        ax.scatter([x], [lo * 100], color=BLUE, **dot)
        ax.scatter([x], [hi * 100], color=ORANGE, **dot)
        ax.annotate(rr_label, (x, max(lo, hi) * 100), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#5c5b53")

    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    ax.set_ylim(0.05, 45)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{WINDOW_MONTHS}-month heart-disease mortality risk "
                  "(weighted, log scale)", fontsize=9, color="#5c5b53")
    ax.tick_params(colors="#5c5b53", labelsize=9)
    ax.grid(axis="y", color="#e8e7e0", lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

    ax.scatter([], [], s=80, color=ORANGE, label=f"SBP ≥ {SBP_CUT:.0f} mm Hg")
    ax.scatter([], [], s=80, color=BLUE, label=f"SBP < {SBP_CUT:.0f} mm Hg")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#1a1a19")

    ax.set_title(
        "Age confounds the SBP–heart-disease-death association:\n"
        f"within age strata RR ≤ {max(r['rr'] for r in rows):.1f}, "
        f"but crude RR {rr_crude:.1f} (age-standardized: {rr_std:.1f})",
        fontsize=11, color="#1a1a19", loc="left", pad=12,
    )
    fig.text(0.01, 0.005,
             "NHANES 2003–2010 adults 20+, MEC-weighted; "
             "NCHS public-use linked mortality through 2019. "
             "* fewer than 30 unweighted deaths in a cell.",
             fontsize=7.5, color="#8a897e")
    fig.tight_layout()
    path = OUT / "sbp_heart_death_by_age.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
