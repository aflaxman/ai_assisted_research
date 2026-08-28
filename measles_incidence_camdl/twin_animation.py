"""Animated GIF of the digital-twin replay, one frame per assimilation
date: data-so-far in ink, the unseen future in light grey, the 8-week
forecast fan in blue, and the running susceptible nowcast below.

Run after digital_twin_replay.py:  uv run python twin_animation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).parent
TWIN = HERE / "results" / "twin"
CITY = "london"
DAYS_PER_YEAR = 365.25
T0 = pd.read_csv(HERE / "data" / "city_meta.tsv", sep="\t").set_index("city").loc[
    CITY, "t0_decyear"
]

INK = "#222222"
FUTURE = "#c8c8c8"
TWIN_BLUE = "#1f77b4"
NOWCAST_ORANGE = "#e8710a"


def decyear(t):
    return T0 + np.asarray(t, float) / DAYS_PER_YEAR


def main():
    obs = pd.read_csv(HERE / "data" / f"{CITY}_cases.tsv", sep="\t")
    nowcast = pd.read_csv(TWIN / f"{CITY}_nowcast.tsv", sep="\t")
    fq = pd.read_csv(TWIN / f"{CITY}_forecast_quantiles.tsv", sep="\t")
    issues = sorted(fq["t_issue"].unique())[::2]  # every 8 weeks -> ~84 frames

    fig, (ax, ax_s) = plt.subplots(
        2, 1, figsize=(9, 5.6), dpi=100, sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12},
    )
    ax.plot(decyear(obs["time"]), obs["weekly_cases"] / 1000, color=FUTURE, lw=0.7)
    (past_line,) = ax.plot([], [], color=INK, lw=0.8)
    fan_band = [None]
    (fan_med,) = ax.plot([], [], color=TWIN_BLUE, lw=1.4)
    today_line = ax.axvline(decyear(issues[0]), color=TWIN_BLUE, lw=0.8, ls=":")
    title = ax.set_title("", fontsize=11)
    ax.set_ylabel("weekly cases ($\\times 10^{-3}$)")
    ax.set_ylim(0, obs["weekly_cases"].max() / 1000 * 1.05)
    ax.text(0.01, 0.93, "reported so far", color=INK, fontsize=8,
            transform=ax.transAxes)
    ax.text(0.01, 0.85, "unseen future", color="0.55", fontsize=8,
            transform=ax.transAxes)
    ax.text(0.01, 0.77, "twin forecast (median, 50%, 90%)", color=TWIN_BLUE,
            fontsize=8, transform=ax.transAxes)

    s_band = [None]
    (s_line,) = ax_s.plot([], [], color=NOWCAST_ORANGE, lw=1.2)
    ax_s.set_ylabel("susceptibles\n($\\times 10^{-3}$)", fontsize=9)
    ax_s.set_xlabel("year")
    ax_s.set_ylim(nowcast["S_q05"].min() / 1000 * 0.9,
                  nowcast["S_q95"].max() / 1000 * 1.05)
    ax_s.text(0.01, 0.85, "filtered susceptible pool (hidden state)",
              color=NOWCAST_ORANGE, fontsize=8, transform=ax_s.transAxes)

    def update(frame):
        t_now = issues[frame]
        past = obs[obs["time"] <= t_now]
        past_line.set_data(decyear(past["time"]), past["weekly_cases"] / 1000)

        f = fq[fq["t_issue"] == t_now].sort_values("t_target")
        x = decyear(f["t_target"])
        if fan_band[0] is not None:
            for coll in fan_band[0]:
                coll.remove()
        fan_band[0] = [
            ax.fill_between(x, f["q05"] / 1000, f["q95"] / 1000,
                            color=TWIN_BLUE, alpha=0.22, lw=0),
            ax.fill_between(x, f["q25"] / 1000, f["q75"] / 1000,
                            color=TWIN_BLUE, alpha=0.35, lw=0),
        ]
        fan_med.set_data(x, f["q50"] / 1000)
        today_line.set_xdata([decyear(t_now)])

        nc = nowcast[nowcast["t"] <= t_now]
        if s_band[0] is not None:
            s_band[0].remove()
        s_band[0] = ax_s.fill_between(decyear(nc["t"]), nc["S_q05"] / 1000,
                                      nc["S_q95"] / 1000, color=NOWCAST_ORANGE,
                                      alpha=0.25, lw=0)
        s_line.set_data(decyear(nc["t"]), nc["S_q50"] / 1000)

        title.set_text(
            f"London measles digital twin — assimilated through "
            f"{decyear(t_now):.2f}, forecasting 8 weeks ahead"
        )
        return []

    anim = FuncAnimation(fig, update, frames=len(issues))
    out = HERE / "results" / "twin_replay.gif"
    anim.save(out, writer=PillowWriter(fps=6))
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({len(issues)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
