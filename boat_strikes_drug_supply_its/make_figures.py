"""Figures for the strike-campaign ITS.

Palette and mark specs follow the project data-viz conventions: categorical
slots assigned in fixed order (blue, orange, aqua), one y-scale per panel,
a legend plus direct labels for every multi-series panel, label text in ink
with a coloured marker carrying identity, recessive grid.

Usage:
    PYTHONPATH=. python -P make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import its_core as ic
import power_analysis as pa
import supply_arithmetic as sa
from run_analysis import CAMPAIGN_START, load_dose, load_outcomes, fit_one

HERE = Path(__file__).parent
OUT = HERE / "outputs"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
SERIES = {"cocaine": "#2a78d6",           # slot 1 blue
          "synthetic_opioids": "#eb6834",  # slot 2 orange
          "psychostimulants": "#1baf7a"}   # slot 3 aqua
LABELS = {"cocaine": "Cocaine",
          "synthetic_opioids": "Synthetic opioids\n(mostly fentanyl)",
          "psychostimulants": "Psychostimulants\n(mostly meth)"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": "#d8d6cf",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "grid.color": "#e8e6df",
    "grid.linewidth": 0.7,
})


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def _campaign_marker(ax, label=True):
    ax.axvline(CAMPAIGN_START, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate("first strike\n2 Sep 2025", xy=(CAMPAIGN_START, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=7.5, color=INK_2)


def _direct_label(ax, x, y, text, colour, dx=6, va="center"):
    """Label text in ink, identity carried by a coloured marker beside it."""
    ax.plot([x], [y], marker="o", ms=5, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6,
            clip_on=False, linestyle="none")
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                ha="left", va=va, fontsize=7.5, color=INK, clip_on=False)


# --------------------------------------------------------------------------
def fig1_campaign_and_outcome(wide: pd.DataFrame) -> Path:
    dose = pd.read_csv(HERE / "data" / "strikes_monthly.csv", parse_dates=["month"])
    fit = fit_one(wide["cocaine"], CAMPAIGN_START, "step")
    cf = ic.counterfactual_rolling(fit)

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(8.4, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.22})

    # Panel A: campaign intensity. One series, so no legend.
    # 24-day bars leave a visible surface gap between adjacent months.
    top.bar(dose["month"] + pd.Timedelta(days=15), dose["vessels"], width=24,
            color=SERIES["cocaine"], zorder=3)
    _grid(top)
    top.set_ylabel("Vessels struck\nper month")
    top.set_ylim(0, 17)
    top.set_title("A  The campaign: 70 vessels struck in 69 strikes, "
                  "Sep 2025 – Aug 2026", loc="left", fontsize=10, color=INK,
                  pad=8)
    top.axvline(CAMPAIGN_START, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    for _, row in dose.iterrows():
        if row["vessels"] > 0:
            top.annotate(f"{int(row['vessels'])}",
                         xy=(row["month"] + pd.Timedelta(days=15), row["vessels"]),
                         xytext=(0, 2), textcoords="offset points",
                         ha="center", fontsize=7, color=INK_2)

    # Panel B: outcome versus counterfactual, on the scale CDC publishes.
    obs = wide["cocaine"]
    bot.fill_between(cf["window_end"], cf["lo"], cf["hi"],
                     color=MUTED, alpha=0.18, lw=0, zorder=2,
                     label="Counterfactual 95% interval")
    bot.plot(cf["window_end"], cf["counterfactual"], color=INK_2, lw=1.6,
             ls=(0, (5, 3)), zorder=4, label="Counterfactual (pre-campaign trend)")
    bot.plot(obs.index, obs.to_numpy(), color=SERIES["cocaine"], zorder=5,
             label="Observed")
    _grid(bot)
    _campaign_marker(bot)
    bot.set_ylabel("Cocaine-involved overdose deaths,\n12 months ending")
    bot.set_title("B  Cocaine-involved deaths ran above the extrapolated trend — "
                  "but so did every other drug", loc="left", fontsize=10,
                  color=INK, pad=8)
    bot.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    # The campaign is 12 months long; a 2022 start keeps the peak and the whole
    # decline in view without crushing panel A into the right margin.
    bot.set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-11-01"))
    bot.set_ylim(14000, 32500)
    _direct_label(bot, obs.index[-1], obs.iloc[-1], "Observed",
                  SERIES["cocaine"])
    _direct_label(bot, cf["window_end"].iloc[-1], cf["counterfactual"].iloc[-1],
                  "Counterfactual", INK_2)

    fig.savefig(OUT / "fig1_campaign_and_outcome.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    return OUT / "fig1_campaign_and_outcome.png"


def fig2_deceleration_predates(wide: pd.DataFrame) -> Path:
    """The decline slowed for every drug months before the first strike."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    diff = wide.diff()          # = y(t) - y(t-12), an exact year-over-year change
    window = diff.loc[diff.index >= pd.Timestamp("2023-06-01")]

    # Nudge colliding end labels apart without moving the data.
    label_offsets = {"cocaine": 0.30, "psychostimulants": -0.30,
                     "synthetic_opioids": 0.0}
    for drug in ("cocaine", "synthetic_opioids", "psychostimulants"):
        # Each series is scaled by its own 12-month total, so one y-axis
        # serves all three.
        rel = 100.0 * window[drug] / wide[drug].shift(1).loc[window.index]
        ax.plot(window.index, rel, color=SERIES[drug],
                label=LABELS[drug].replace("\n", " "), zorder=4)
        _direct_label(ax, window.index[-1], rel.iloc[-1], LABELS[drug],
                      SERIES[drug])
        ax.texts[-1].set_y(rel.iloc[-1] + label_offsets[drug])

    ax.axhline(0, color=INK_2, lw=1.0, zorder=3)
    _grid(ax)
    _campaign_marker(ax)
    # The steepest year-over-year decline is Aug-Sep 2024 for all three drugs;
    # everything after that is deceleration, and it starts a year early.
    ax.axvspan(pd.Timestamp("2024-09-01"), CAMPAIGN_START,
               color=SERIES["cocaine"], alpha=0.06, lw=0, zorder=1)
    ax.annotate("decline already flattening for all three drugs,\n"
                "a full year before the first strike",
                xy=(pd.Timestamp("2025-01-01"), 0.35),
                ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_ylabel("Year-over-year change in monthly deaths,\n% of the 12-month total")
    ax.set_title("The deceleration the ITS mistakes for an effect begins a year "
                 "before the campaign", loc="left", fontsize=10, color=INK, pad=8)
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_xlim(window.index[0], pd.Timestamp("2026-07-01"))
    fig.savefig(OUT / "fig2_deceleration_predates.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)
    return OUT / "fig2_deceleration_predates.png"


def fig3_placebo(main: pd.DataFrame, placebo: pd.DataFrame) -> Path:
    """Single-series estimates fire at dates when nothing happened; contrasts don't."""
    real_single = main[(main["kind"] == "single series")
                       & (main["series"] == "cocaine")
                       & (main["lag_months"] == 0)].iloc[0]
    real_contrast = main[(main["kind"] == "contrast")
                         & (main["series"] == "cocaine vs synthetic_opioids")
                         & (main["lag_months"] == 0)].iloc[0]

    singles = placebo[(placebo["kind"] == "single series")
                      & (placebo["series"] == "cocaine")]
    contrasts = placebo[(placebo["kind"] == "contrast")
                        & (placebo["series"] == "cocaine vs synthetic_opioids")]

    rows = []
    for _, r in singles.iterrows():
        rows.append((f"placebo {pd.Timestamp(r['placebo_date']):%b %Y}",
                     r["pct_change"], r["lo"], r["hi"], "single"))
    rows.append(("REAL Sep 2025", real_single["pct_change"], real_single["lo"],
                 real_single["hi"], "single"))
    for _, r in contrasts.iterrows():
        rows.append((f"placebo {pd.Timestamp(r['placebo_date']):%b %Y}",
                     r["pct_change"], r["lo"], r["hi"], "contrast"))
    rows.append(("REAL Sep 2025", real_contrast["pct_change"], real_contrast["lo"],
                 real_contrast["hi"], "contrast"))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.9), sharex=True)
    for ax, kind, title, colour in (
        (axes[0], "single", "Cocaine series alone", SERIES["cocaine"]),
        (axes[1], "contrast", "Cocaine vs synthetic opioids", SERIES["synthetic_opioids"]),
    ):
        sub = [r for r in rows if r[4] == kind]
        ys = np.arange(len(sub))[::-1]
        for y, (label, est, lo, hi, _) in zip(ys, sub):
            real = label.startswith("REAL")
            ax.plot([lo, hi], [y, y], color=colour if real else MUTED,
                    lw=2.4 if real else 1.8, solid_capstyle="round", zorder=3)
            ax.plot([est], [y], marker="o", ms=8 if real else 6.5,
                    color=colour if real else MUTED, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, zorder=4, linestyle="none")
        ax.axvline(0, color=INK_2, lw=1.0, zorder=2)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in sub], fontsize=8,
                           color=INK)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
        _grid(ax, axis="x")
        ax.set_xlabel("Estimated step change in deaths, %")

    fig.suptitle("A placebo date should find nothing. The single series finds "
                 "±18%; the contrast finds nothing.",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig3_placebo.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return OUT / "fig3_placebo.png"


def fig4_power_vs_expected(curve: pd.DataFrame, observed: pd.Series) -> Path:
    """What the design can detect, against what the tonnage implies to expect."""
    grid = sa.grid()
    consistent = grid[grid["consistent"]]["expected_death_change_pct"]
    expected_lo, expected_hi = consistent.min(), consistent.max()
    q1, q3 = consistent.quantile(0.25), consistent.quantile(0.75)
    expected_med = consistent.median()
    mde = pa.minimum_detectable_effect(curve)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.axvspan(expected_lo, expected_hi, color=SERIES["psychostimulants"],
               alpha=0.08, lw=0, zorder=1,
               label="Effect the tonnage implies: full range")
    ax.axvspan(q1, q3, color=SERIES["psychostimulants"], alpha=0.20, lw=0,
               zorder=2, label="Effect the tonnage implies: middle half")
    ax.axvline(expected_med, color=SERIES["psychostimulants"], lw=1.4,
               ls=(0, (4, 3)), zorder=3)
    ax.plot(curve["true_effect_pct"], curve["power"], color=SERIES["cocaine"],
            marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
            zorder=5, label="Power of the controlled contrast")
    ax.axhline(0.8, color=INK_2, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.annotate("80% power", xy=(0.99, 0.8), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK_2)
    ax.annotate(f"detectable only\nbelow {mde:.0f}%",
                xy=(mde, 0.8), xytext=(-14, -46), textcoords="offset points",
                ha="center", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax.annotate(f"median expected {expected_med:.0f}%",
                xy=(expected_med, 0.02), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK)

    _grid(ax)
    ax.set_xlabel("True reduction in cocaine-involved deaths, %")
    ax.set_ylabel("Probability the analysis detects it")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(2, -45)
    ax.set_title("The design can only detect reductions larger than the ones "
                 "the tonnage predicts", loc="left", fontsize=10, color=INK,
                 pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(OUT / "fig4_power_vs_expected.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return OUT / "fig4_power_vs_expected.png"


def main() -> None:
    wide = load_outcomes("rolling12")
    main_tbl = pd.read_csv(OUT / "its_main.csv")
    placebo = pd.read_csv(OUT / "its_placebo.csv")
    curve = pd.read_csv(OUT / "power_curve.csv")

    paths = [
        fig1_campaign_and_outcome(wide),
        fig2_deceleration_predates(wide),
        fig3_placebo(main_tbl, placebo),
        fig4_power_vs_expected(curve, wide["cocaine"]),
    ]
    for p in paths:
        print("wrote", p.relative_to(HERE))


if __name__ == "__main__":
    main()
