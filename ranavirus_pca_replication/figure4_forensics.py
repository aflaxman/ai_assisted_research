"""Compare the published Figure 4 of Duffus et al. (2026) against replication runs.

The paper's Figure 4 caption says omega = 0.15, but the plotted curves do not
match that setting: with omega = 0.15 the mean Combined curve peaks near 37
frogs and the ponds are fully depopulated around iteration 38 on average,
while the published figure shows Combined peaking near 70 and a Dead curve
that is still around 89 frogs at iteration 40.  A parameter scan shows the
published curves match omega ~= 0.06 almost exactly.

This script overlays curve values digitized from the published PNG
(results/fig4_digitized.json, extracted by color-matching pixels) on
replication runs at omega = 0.15 (caption value) and omega = 0.06 (best fit).

Usage:  python figure4_forensics.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ranavirus_ca import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

STATE_STYLE = [
    (0, "S", "Susceptible", "green"),
    (1, "U", "Ulcerative", "xkcd:dark yellow"),
    (2, "H", "Hemorrhagic", "red"),
    (3, "C", "Combined", "orange"),
    (4, "D", "Dead", "black"),
]


def main():
    with open(os.path.join(OUT, "fig4_digitized.json")) as f:
        dig = json.load(f)
    dig_x = list(range(0, 121, 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, omega in zip(axes, [0.15, 0.06]):
        res = simulate(pd=omega, n_ponds=10_000, seed=42)
        x = range(res["mean"].shape[0])
        for k, key, label, color in STATE_STYLE:
            ax.plot(x, res["mean"][:, k], color=color, label=f"{label} (replication)")
            pts = [(xi, v) for xi, v in zip(dig_x, dig[key]) if v is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=4, color=color,
                    alpha=0.6)
        ax.set_title(f"Replication with $\\omega$ = {omega}\n(dots: digitized published Figure 4)")
        ax.set_xlabel("Time (iterations)")
    axes[0].set_ylabel("# Frogs in given status")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Published Figure 4 matches $\\omega \\approx 0.06$, not the caption's $\\omega = 0.15$")
    fig.tight_layout()
    path = os.path.join(OUT, "figure4_forensics.png")
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
