"""Summary figure: the replication against what the paper printed.

    uv run python plot_replication.py

Six panels, each the model as stated (lines) against values read off the
paper's own figures (dots).  Colours are the first slots of a categorical
palette checked for colour-vision deficiency; every line is also labelled.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import transmission as tr

ROOT = pathlib.Path(__file__).resolve().parent
RES, DATA = ROOT / "results", ROOT / "data"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 26,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
})


def dots(ax, x, y, color, label=None, size=44, **kw):
    """Markers with a ring in the surface colour so they stay legible on lines."""
    return ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=1.4, zorder=4, label=label, **kw)


def subtitle(ax, text):
    ax.text(0.0, 1.035, text, transform=ax.transAxes, color=INK2, fontsize=8.8, va="bottom")


def end_label(ax, x, y, text, dy=0.0, ha="left", dx=4):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2,
                fontsize=8.6, va="center", ha=ha)


def panel_fig3_cells(ax):
    d = pd.read_csv(RES / "fig3_re_grid.tsv", sep="\t")
    summary = pd.read_csv(RES / "fig3_fit_summary.tsv", sep="\t")
    lim = (-0.05, 2.0)
    ax.plot(lim, lim, color=AXIS, lw=1.2, zorder=1)
    ax.scatter(d.Re_expected, d.Re_paper, s=13, color=BLUE, alpha=0.55, edgecolor="none", zorder=3)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("expected value under the stated model"), ax.set_ylabel("read off the published figure")
    ax.set_title("A  Mean R of every cell of Figure 3")
    chi2, n = summary.chi2_stated.sum(), int(summary.cells.sum())
    subtitle(ax, f"484 cells, each a 3,000-transmitter run in the paper")
    ax.text(0.97, 0.06, f"scatter about the line is what\nMonte Carlo error predicts:\nchi-square {chi2:.0f} on {n} cells",
            transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8.6)


def panel_fig3_diagonal(ax):
    d = pd.read_csv(RES / "fig3_re_grid.tsv", sep="\t")
    d = d[d.eps_T == d.eps_E]
    ax.axhline(1.0, color=AXIS, lw=1.2, zorder=1)
    labels = {"25/25": "25% of people,\n25% of the time", "50/50": "50%, 50%", "75/75": "75%, 75%", "100/100": "100%, 100%"}
    for (cov, text), color in zip(labels.items(), (BLUE, ORANGE, AQUA, YELLOW)):
        g = d[d.coverage == cov].sort_values("eps_T")
        ax.plot(g.eps_T, g.Re_exact, color=color, zorder=3, label=text.replace("\n", " "))
        dots(ax, g.eps_T, g.Re_paper, color, size=30)
        end_label(ax, 1.0, g.Re_exact.iloc[-1], text, dy={"25/25": 4}.get(cov, 0))
    ax.set_xlim(-0.03, 1.0), ax.set_ylim(-0.05, 2.0)
    ax.set_xlabel("mask efficacy, same for transmitter and contacts"), ax.set_ylabel("effective reproductive number")
    ax.set_title("B  The paper's main result holds")
    subtitle(ax, "lines: exact expectation; dots: the paper's cells on the diagonal")


def panel_pairwise(ax):
    pts = pd.read_csv(DATA / "paper_fig2_digitized.tsv", sep="\t")
    eps = np.linspace(0.0, 1.0, 201)
    v = 1.0e8
    single = tr.transmission_risk(v, eps, 0.0)
    dual = tr.transmission_risk(v, eps, eps)
    ax.plot(eps, single, color=BLUE, zorder=3, label="one mask, as stated")
    ax.plot(eps, dual, color=ORANGE, zorder=3, label="two masks, as stated")
    ax.plot(eps, (1 - eps) * dual, color=ORANGE, lw=1.3, ls=(0, (4, 3)), zorder=2,
            label="two masks, stated risk times (1 - efficacy)")
    for name, color in (("Fig 2B", BLUE), ("Fig 2D", ORANGE)):
        g = pts[(pts.panel == name) & (pts.log10_viral_load == 8.0)].iloc[::2]
        dots(ax, g.efficacy, g.risk, color, size=34)
    end_label(ax, 0.62, tr.transmission_risk(v, 0.62, 0.0), "one mask (Fig 2B)", dx=6, dy=10)
    end_label(ax, 0.50, tr.transmission_risk(v, 0.50, 0.50), "two masks,\nas stated", dx=8, dy=12)
    end_label(ax, 0.30, 0.70 * tr.transmission_risk(v, 0.30, 0.30), "what Fig 2D plots", dx=-8, dy=-16, ha="right")
    ax.set_xlim(0, 1.0), ax.set_ylim(0, 0.8)
    ax.set_xlabel("mask efficacy"), ax.set_ylabel("transmission risk per exposure contact")
    ax.set_title("C  One heat map plots a different formula")
    subtitle(ax, "viral load 10^8; lines: the Methods' formula; dots: read off Figure 2")


def panel_targeted(ax):
    d = pd.read_csv(RES / "suppfig5_targeted.tsv", sep="\t")
    g = d[(d.coverage == "25/25") & (d.eps_T == d.eps_E)].sort_values("eps_T")
    ax.axhline(1.0, color=AXIS, lw=1.2, zorder=1)
    ax.plot(g.eps_T, g.Re_untargeted, color=BLUE, zorder=3, label="no targeting")
    ax.plot(g.eps_T, g.Re_masked_on_busy_days, color=ORANGE, zorder=3, label="masked on days with more than 10 contacts")
    ax.plot(g.eps_T, g.Re_masked_if_ever_busy, color=AQUA, zorder=3, label="masked always if ever more than 10 contacts")
    dots(ax, g.eps_T, g.Re_paper, AQUA, size=34)
    end_label(ax, 1.0, g.Re_untargeted.iloc[-1], "no targeting")
    end_label(ax, 1.0, g.Re_masked_on_busy_days.iloc[-1], "masked on days with\nmore than 10 contacts", dy=6)
    end_label(ax, 1.0, g.Re_masked_if_ever_busy.iloc[-1], "masked always if any\nday has more than 10", dy=-4)
    ax.set_xlim(-0.03, 1.0), ax.set_ylim(-0.05, 2.0)
    ax.set_xlabel("mask efficacy, same for transmitter and contacts"), ax.set_ylabel("effective reproductive number")
    ax.set_title("D  'Targeted' masking covers 97% of people")
    subtitle(ax, "others: 25% of people, 25% of the time; dots: Supplementary Fig 5A")


def panel_antivirals(ax):
    d = pd.read_csv(RES / "fig7_antivirals.tsv", sep="\t")
    paper = pd.read_csv(DATA / "paper_fig7_read_by_eye.tsv", sep="\t", comment="#")
    base = d[d.scenario == "untreated"].Re_exact.iloc[0]
    ax.axhline(1.0, color=AXIS, lw=1.2, zorder=1)
    pep = d[d.scenario == "post-exposure prophylaxis"].sort_values("efficacy")
    x = np.concatenate([[0.0], pep.efficacy]) * 100
    ax.plot(x, np.concatenate([[base], pep.Re_exact]), color=BLUE, zorder=3, label="prophylaxis 0.5 to 5 days after exposure")
    pp = paper[paper.scenario == "post-exposure prophylaxis"]
    dots(ax, pp.efficacy * 100, pp.mean_R0, BLUE, size=34)
    sym = d[d.scenario == "early symptomatic therapy"].Re_exact.iloc[0]
    ax.plot([0, 100], [base, sym], color=ORANGE, zorder=3, label="therapy 0.5 to 5 days after symptom onset")
    dots(ax, [100], paper[paper.scenario == "early symptomatic therapy"].mean_R0, ORANGE, size=34)
    end_label(ax, 100, sym, "therapy after\nsymptom onset\n(100% efficacy\nat right)", dy=-2)
    end_label(ax, 100, pep.Re_exact.iloc[-1], "prophylaxis\nafter exposure")
    ax.set_xlim(-3, 100), ax.set_ylim(-0.05, 2.0)
    ax.set_xlabel("antiviral efficacy (%), everyone treated"), ax.set_ylabel("effective reproductive number")
    ax.set_title("E  Antivirals: timing matters more than potency")
    subtitle(ax, "lines: exact expectation; dots: read by eye off Figure 7A and 7D")


def panel_king_county(ax):
    d = pd.read_csv(RES / "fig6_king_county_fit.tsv", sep="\t", parse_dates=["date"])
    ax.axhline(1.0, color=AXIS, lw=1.2, zorder=1)
    ax.fill_between(d.date, d.Re_low, d.Re_high, color=MUTED, alpha=0.13, lw=0, zorder=1)
    ax.plot(d.date, d.Re, color=INK2, lw=1.4, zorder=3, label="R estimated from cases (paper's Fig 6C)")
    ax.plot(d.date, d.Re_fitted, color=BLUE, zorder=4, label="mobility + masks")
    ax.plot(d.date, d.Re_no_masks, color=ORANGE, zorder=4, label="same fit, masks set to zero")
    ax.plot(d.date, d.Re_fitted_time_trend, color=AQUA, zorder=4, label="mobility + linear time trend")
    last = d.iloc[-1]
    end_label(ax, last.date, last.Re_no_masks, "masks set\nto zero")
    end_label(ax, last.date, last.Re_fitted, "mobility\n+ masks", dy=7)
    end_label(ax, last.date, last.Re_fitted_time_trend, "mobility\n+ time", dy=-16)
    ax.annotate("estimated from\ncase data", (pd.Timestamp("2020-06-17"), 1.53), xytext=(-70, 16),
                textcoords="offset points", color=INK2, fontsize=8.6,
                arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1})
    ax.set_ylim(0, 3.2), ax.set_xlim(d.date.iloc[0], d.date.iloc[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator()), ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("effective reproductive number, King County 2020")
    ax.set_title("F  The King County regression replicates, and so does a placebo")
    subtitle(ax, "regressions refitted to the paper's R curve and Google mobility data")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2))
    for draw, ax in zip((panel_fig3_cells, panel_fig3_diagonal, panel_pairwise,
                         panel_targeted, panel_antivirals, panel_king_county), axes.ravel()):
        draw(ax)
        ax.tick_params(length=0)
    for ax in (axes[0, 2], axes[1, 2]):
        ax.legend(loc="upper right" if ax is axes[0, 2] else "upper center", fontsize=8.2,
                  labelcolor=INK2, handlelength=1.8)
    axes[1, 0].legend(loc="lower left", fontsize=8.2, labelcolor=INK2, handlelength=1.8)
    axes[1, 1].legend(loc="lower left", fontsize=8.2, labelcolor=INK2, handlelength=1.8)
    axes[0, 1].legend(loc="lower left", fontsize=8.2, labelcolor=INK2, handlelength=1.8, title="masks worn by",
                      title_fontsize=8.2)
    fig.subplots_adjust(left=0.05, right=0.925, top=0.93, bottom=0.07, wspace=0.42, hspace=0.42)
    fig.savefig(RES / "mask_replication.png", dpi=170)
    print("wrote", RES / "mask_replication.png")


if __name__ == "__main__":
    main()
