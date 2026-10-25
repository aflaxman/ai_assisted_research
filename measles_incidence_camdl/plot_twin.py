"""Static figure for the digital-twin replay: forecast fans over the
observed series, the hidden-state nowcast, and the scorecard.

Run after digital_twin_replay.py:  uv run python plot_twin.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
TWIN = HERE / "results" / "twin"
CITY = "london"
DAYS_PER_YEAR = 365.25
T0 = pd.read_csv(HERE / "data" / "city_meta.tsv", sep="\t").set_index("city").loc[
    CITY, "t0_decyear"
]

INK = "#222222"
TWIN_BLUE = "#1f77b4"
NOWCAST_ORANGE = "#e8710a"
GRID = dict(color="0.85", lw=0.6)


def decyear(t):
    return T0 + np.asarray(t, float) / DAYS_PER_YEAR


def main():
    obs = pd.read_csv(HERE / "data" / f"{CITY}_cases.tsv", sep="\t")
    nowcast = pd.read_csv(TWIN / f"{CITY}_nowcast.tsv", sep="\t")
    fq = pd.read_csv(TWIN / f"{CITY}_forecast_quantiles.tsv", sep="\t")
    scores = pd.read_csv(TWIN / f"{CITY}_scores.tsv", sep="\t")

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.1, 1.1, 1.3], hspace=0.42, wspace=0.35)
    ax_fan = fig.add_subplot(gs[0, :])
    ax_now = fig.add_subplot(gs[1, :], sharex=ax_fan)
    ax_crps = fig.add_subplot(gs[2, 0])
    ax_cov = fig.add_subplot(gs[2, 1])
    ax_pit = fig.add_subplot(gs[2, 2])

    # -- A: observed series + a subset of the 8-week forecast fans
    ax_fan.plot(decyear(obs["time"]), obs["weekly_cases"] / 1000, color=INK, lw=0.7,
                label="observed weekly cases")
    issues = sorted(fq["t_issue"].unique())
    for t_issue in issues[::4]:
        f = fq[fq["t_issue"] == t_issue].sort_values("t_target")
        x = decyear(f["t_target"])
        ax_fan.fill_between(x, f["q05"] / 1000, f["q95"] / 1000,
                            color=TWIN_BLUE, alpha=0.30, lw=0)
        ax_fan.plot(x, f["q50"] / 1000, color=TWIN_BLUE, lw=0.9)
    ax_fan.plot([], [], color=TWIN_BLUE, lw=2, alpha=0.6,
                label="twin: 8-week forecast fan (median, 90%), every 16th week shown")
    ax_fan.set_ylabel("weekly cases ($\\times 10^{-3}$)")
    ax_fan.set_ylim(bottom=0)
    ax_fan.legend(fontsize=8, loc="upper left", frameon=False)
    ax_fan.set_title(
        f"A retrospective digital twin of London measles: assimilate every 4 weeks, "
        f"forecast 8 weeks ahead ({len(issues)} forecasts)", fontsize=11)
    ax_fan.grid(axis="y", **GRID)

    # -- B: the nowcast of the hidden susceptible pool
    ax_now.fill_between(decyear(nowcast["t"]), nowcast["S_q05"] / 1000,
                        nowcast["S_q95"] / 1000, color=NOWCAST_ORANGE, alpha=0.25, lw=0)
    ax_now.plot(decyear(nowcast["t"]), nowcast["S_q50"] / 1000,
                color=NOWCAST_ORANGE, lw=1.2, label="filtered $S_t$ (median, 90%)")
    ax_now.set_ylabel("susceptibles ($\\times 10^{-3}$)")
    ax_now.set_xlabel("year")
    ax_now.legend(fontsize=8, loc="upper left", frameon=False)
    ax_now.grid(axis="y", **GRID)
    ax_now.text(0.99, 0.05,
                "nobody observes this directly — the twin's state estimate",
                transform=ax_now.transAxes, ha="right", fontsize=8, color="0.4")

    # -- C1: CRPS by horizon, twin vs the two point baselines
    g = scores.groupby("h_weeks")[["crps_twin", "crps_persistence", "crps_seasonal"]].mean()
    ax_crps.plot(g.index, g["crps_twin"], color=TWIN_BLUE, lw=1.8, marker="o", ms=4,
                 label="twin")
    ax_crps.plot(g.index, g["crps_persistence"], color="0.45", lw=1.2, ls="--",
                 marker="s", ms=3, label="persistence")
    ax_crps.plot(g.index, g["crps_seasonal"], color="0.45", lw=1.2, ls=":",
                 marker="^", ms=3, label="seasonal-naive")
    ax_crps.set_xlabel("forecast horizon (weeks)")
    ax_crps.set_ylabel("mean CRPS (cases)")
    ax_crps.set_ylim(bottom=0)
    ax_crps.legend(fontsize=7, frameon=False)
    ax_crps.set_title("sharpness + calibration", fontsize=9)
    ax_crps.grid(axis="y", **GRID)

    # -- C2: interval coverage by horizon
    cov = scores.groupby("h_weeks")[["in50", "in90"]].mean()
    ax_cov.plot(cov.index, cov["in90"], color=TWIN_BLUE, lw=1.8, marker="o", ms=4,
                label="90% interval")
    ax_cov.plot(cov.index, cov["in50"], color=TWIN_BLUE, lw=1.2, ls="--", marker="s",
                ms=3, alpha=0.6, label="50% interval")
    ax_cov.axhline(0.9, color="0.45", lw=0.8, ls=":")
    ax_cov.axhline(0.5, color="0.45", lw=0.8, ls=":")
    ax_cov.set_ylim(0, 1)
    ax_cov.set_xlabel("forecast horizon (weeks)")
    ax_cov.set_ylabel("empirical coverage")
    ax_cov.legend(fontsize=7, frameon=False, loc="lower left")
    ax_cov.set_title("are the intervals honest?", fontsize=9)
    ax_cov.grid(axis="y", **GRID)

    # -- C3: PIT histogram from camdl's one-step-ahead prequential trace
    preq_path = TWIN / f"{CITY}_prequential.tsv"
    if preq_path.exists():
        preq = pd.read_csv(preq_path, sep="\t", comment="#")
        if "stream" in preq.columns:  # file carries a per-stream and a joint row
            preq = preq[preq["stream"] == "weekly_cases"]
        pit_col = next((c for c in preq.columns if c.lower() == "pit"), None)
        if pit_col is not None:
            pit = preq[pit_col].dropna()
            ax_pit.hist(pit, bins=10, range=(0, 1), color=TWIN_BLUE, alpha=0.75,
                        edgecolor="white", density=True)
            ax_pit.axhline(1.0, color="0.45", lw=0.8, ls=":")
            ax_pit.set_title("one-step PIT (camdl prequential)", fontsize=9)
            ax_pit.set_xlabel("probability integral transform")
            ax_pit.set_ylabel("density")
            ax_pit.grid(axis="y", **GRID)

    out = HERE / "results" / "twin_replay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(scores.groupby("h_weeks")[["crps_twin", "crps_persistence",
                                     "crps_seasonal", "in50", "in90"]].mean().round(2))


if __name__ == "__main__":
    main()
