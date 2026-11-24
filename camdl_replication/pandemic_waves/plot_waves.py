"""Figures for the Frimpong & Bauch (2026) replication.

Three PNGs into results/:
- waves_incidence.png  — Fig 2 analogue: reported-incidence data, SIR and
  SIRx posterior medians + 95% CrI, per country, with the train/predict
  split marked
- waves_behaviour.png  — Fig 4 analogue: Oxford stringency index vs the
  SIRx mitigator proportion x
- waves_metrics.png    — Fig 3 + 5 analogue: second-wave peak magnitude
  and day, area between curves, and AICc, per country
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prep_data import COUNTRIES

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TRAJ = RESULTS / "trajectories"

# dataviz reference palette (light mode)
C_SIRX = "#2a78d6"   # blue: the coupled model, code's lam prior [0, 0.2]
C_SIRX2 = "#1baf7a"  # aqua: the coupled model, supplement's lam prior [0, 0.03]
C_SIR = "#eb6834"    # orange: the disease model
C_DATA = "#0b0b0b"
C_MUTE = "#898781"
C_GRID = "#e1e0d9"
C_SPLIT = "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})

NAMES = [c[0] for c in COUNTRIES]
TICK = {"austria": "Aut", "belarus": "Blr", "belgium": "Bel", "denmark": "Den",
        "finland": "Fin", "france": "Fra", "germany": "Ger", "ireland": "Irl",
        "netherlands": "Nld", "norway": "Nor", "portugal": "Por", "uk": "UK",
        "switzerland": "Swi"}
CUTOFF = {c[0]: c[3] for c in COUNTRIES}
LABELS = {n: n.upper() if n == "uk" else n.capitalize() for n in NAMES}


def grid(n=13):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5))
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_incidence():
    fig, axes = grid()
    for ax, name in zip(axes, NAMES):
        cutoff = CUTOFF[name]
        for model, color in (("sir", C_SIR), ("sirx", C_SIRX),
                             ("sirx_s2", C_SIRX2)):
            f = TRAJ / f"{name}_{model}.tsv"
            if not f.exists():
                continue
            d = pd.read_csv(f, sep="\t")
            scale = 1e5  # per 100k
            ax.fill_between(d.time, d.q025 * scale, d.q975 * scale,
                            color=color, alpha=0.18, lw=0)
            ax.plot(d.time, d.med * scale, color=color, lw=1.6)
        d = pd.read_csv(TRAJ / f"{name}_sirx.tsv", sep="\t")
        ax.plot(d.time, d.data * 1e5, ".", ms=2.2, color=C_DATA, alpha=0.55)
        ax.axvline(cutoff, color=C_SPLIT, lw=1.0, ls="--", alpha=0.7)
        top = max(d.data.max(), d.med.max()) * 1e5
        ax.set_ylim(0, 3.2 * d.data.max() * 1e5)
        ax.set_title(LABELS[name], fontsize=10, loc="left")
    axes[0].plot([], [], ".", color=C_DATA, label="reported (7-day MA)")
    axes[0].plot([], [], color=C_SIR, lw=1.6, label="SIR (disease only)")
    axes[0].plot([], [], color=C_SIRX, lw=1.6, label="SIRx (code λ prior)")
    axes[0].plot([], [], color=C_SIRX2, lw=1.6, label="SIRx (supplement λ prior)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.supylabel("daily reported cases per 100,000", fontsize=10)
    fig.supxlabel("days from first data point (dashed line = end of fitting window)",
                  fontsize=10)
    fig.suptitle("Second-wave prediction: coupled behaviour-disease (SIRx) vs "
                 "seasonal SIR — posterior median and 95% CrI", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "waves_incidence.png", dpi=150)
    plt.close(fig)


def plot_behaviour():
    fig, axes = grid()
    for ax, name in zip(axes, NAMES):
        for model, color in (("sirx", C_SIRX), ("sirx_s2", C_SIRX2)):
            f = TRAJ / f"{name}_{model}.tsv"
            if not f.exists():
                continue
            d = pd.read_csv(f, sep="\t")
            ax.fill_between(d.time, d.x_q025, d.x_q975, color=color, alpha=0.18, lw=0)
            ax.plot(d.time, d.x_med, color=color, lw=1.6)
        ax.plot(d.time, d.osi, ".", ms=2.2, color=C_DATA, alpha=0.55)
        ax.axvline(CUTOFF[name], color=C_SPLIT, lw=1.0, ls="--", alpha=0.7)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(LABELS[name], fontsize=10, loc="left")
    axes[0].plot([], [], ".", color=C_DATA, label="Oxford stringency / 100")
    axes[0].plot([], [], color=C_SIRX, lw=1.6, label="x (code λ prior)")
    axes[0].plot([], [], color=C_SIRX2, lw=1.6, label="x (supplement λ prior)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.supylabel("proportion", fontsize=10)
    fig.supxlabel("days from first data point (dashed line = end of fitting window)",
                  fontsize=10)
    fig.suptitle("Behaviour stream: stringency index vs fitted mitigator "
                 "proportion — posterior median and 95% CrI", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "waves_behaviour.png", dpi=150)
    plt.close(fig)


def _err(center, lo, hi):
    return [np.maximum(0, center - lo), np.maximum(0, hi - center)]


def plot_metrics():
    df = pd.read_csv(RESULTS / "summary_stats.tsv", sep="\t")
    sir = df[df.model == "sir"].set_index("country")
    sirx = df[df.model == "sirx"].set_index("country")
    sirx2 = df[df.model == "sirx_s2"].set_index("country")
    names = [n for n in NAMES if n in sir.index and n in sirx.index]
    xs = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    # (a) second-wave peak magnitude, log scale
    ax = axes[0, 0]
    ax.errorbar(xs - 0.18, sir.loc[names, "peak2_mag"],
                yerr=_err(sir.loc[names, "peak2_mag"], sir.loc[names, "peak2_mag_lo"],
                          sir.loc[names, "peak2_mag_hi"]),
                fmt="o", ms=4.5, color=C_SIR, lw=1.2, capsize=2, label="SIR")
    ax.errorbar(xs + 0.05, sirx.loc[names, "peak2_mag"],
                yerr=_err(sirx.loc[names, "peak2_mag"], sirx.loc[names, "peak2_mag_lo"],
                          sirx.loc[names, "peak2_mag_hi"]),
                fmt="o", ms=4.5, color=C_SIRX, lw=1.2, capsize=2, label="SIRx (code)")
    ax.errorbar(xs + 0.28, sirx2.loc[names, "peak2_mag"],
                yerr=_err(sirx2.loc[names, "peak2_mag"], sirx2.loc[names, "peak2_mag_lo"],
                          sirx2.loc[names, "peak2_mag_hi"]),
                fmt="o", ms=4.5, color=C_SIRX2, lw=1.2, capsize=2, label="SIRx (suppl.)")
    ax.plot(xs, sir.loc[names, "data_peak2_mag"], "_", ms=14, color=C_DATA,
            label="observed", zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.set_title("(a) second-wave peak magnitude (proportion/day)", loc="left")
    ax.legend(frameon=False, fontsize=8)

    # (b) second-wave peak day
    ax = axes[0, 1]
    ax.errorbar(xs - 0.18, sir.loc[names, "peak2_day"],
                yerr=_err(sir.loc[names, "peak2_day"], sir.loc[names, "peak2_day_lo"],
                          sir.loc[names, "peak2_day_hi"]),
                fmt="o", ms=4.5, color=C_SIR, lw=1.2, capsize=2)
    ax.errorbar(xs + 0.05, sirx.loc[names, "peak2_day"],
                yerr=_err(sirx.loc[names, "peak2_day"], sirx.loc[names, "peak2_day_lo"],
                          sirx.loc[names, "peak2_day_hi"]),
                fmt="o", ms=4.5, color=C_SIRX, lw=1.2, capsize=2)
    ax.errorbar(xs + 0.28, sirx2.loc[names, "peak2_day"],
                yerr=_err(sirx2.loc[names, "peak2_day"], sirx2.loc[names, "peak2_day_lo"],
                          sirx2.loc[names, "peak2_day_hi"]),
                fmt="o", ms=4.5, color=C_SIRX2, lw=1.2, capsize=2)
    ax.plot(xs, sir.loc[names, "data_peak2_day"], "_", ms=14, color=C_DATA, zorder=5)
    ax.set_title("(b) second-wave peak day", loc="left")

    # (c) area between curves, prediction window
    ax = axes[1, 0]
    ax.bar(xs - 0.25, sir.loc[names, "area_pred"], width=0.24, color=C_SIR,
           edgecolor="#fcfcfb", lw=0.5)
    ax.bar(xs, sirx.loc[names, "area_pred"], width=0.24, color=C_SIRX,
           edgecolor="#fcfcfb", lw=0.5)
    ax.bar(xs + 0.25, sirx2.loc[names, "area_pred"], width=0.24, color=C_SIRX2,
           edgecolor="#fcfcfb", lw=0.5)
    ax.set_yscale("log")
    ax.set_title("(c) area between model and data, prediction window", loc="left")

    # (d) AICc on the prediction window
    ax = axes[1, 1]
    ax.bar(xs - 0.25, sir.loc[names, "aicc_inc"], width=0.24, color=C_SIR,
           edgecolor="#fcfcfb", lw=0.5, label="SIR")
    ax.bar(xs, sirx.loc[names, "aicc_inc"], width=0.24, color=C_SIRX,
           edgecolor="#fcfcfb", lw=0.5, label="SIRx (code)")
    ax.bar(xs + 0.25, sirx2.loc[names, "aicc_inc"], width=0.24, color=C_SIRX2,
           edgecolor="#fcfcfb", lw=0.5, label="SIRx (suppl.)")
    ax.set_title("(d) AICc on incidence, prediction window (lower is better)",
                 loc="left")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks(xs)
        ax.set_xticklabels([TICK[n] for n in names], rotation=0, fontsize=8)
    fig.suptitle("Prediction metrics: SIR (orange) vs coupled SIRx (blue), "
                 "13 European countries", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "waves_metrics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_incidence()
    plot_behaviour()
    plot_metrics()
    print("wrote", *[p.name for p in RESULTS.glob("waves_*.png")])
