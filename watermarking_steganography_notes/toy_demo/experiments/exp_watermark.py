"""Watermark detection power grows with tokens and with entropy.

The distortion-free watermark leaves no trace in the output distribution, yet a
key holder accumulates evidence as the text lengthens. The rate of accumulation
depends on entropy: a confident, low-entropy model is forced to emit particular
tokens regardless of the secret scores, leaking little signal; a high-entropy
model is steered by the scores and leaks a lot. Text we did not generate stays
flat at the null.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stegowm import watermark  # noqa: E402
from stegowm.sources import SyntheticSource  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def z_curve(alpha, lengths, trials=12):
    """Median detection z-score vs text length for a given entropy regime."""
    zs = np.zeros((trials, len(lengths)))
    for t in range(trials):
        src = SyntheticSource(vocab=50, n_steps=max(lengths) + 5, alpha=alpha, seed=100 + t)
        key = bytes([t]) + b"-wm-key-00000000"[:15]
        toks = watermark.generate(src, key, n_tokens=max(lengths))
        for j, n in enumerate(lengths):
            zs[t, j] = watermark.detect(src, key, toks[:n])["z"]
    return np.median(zs, axis=0)


def control_curve(lengths, trials=12):
    zs = np.zeros((trials, len(lengths)))
    for t in range(trials):
        src = SyntheticSource(vocab=50, n_steps=max(lengths) + 5, alpha=1.0, seed=7 + t)
        key = b"detector-key-0001"
        rng = np.random.default_rng(500 + t)
        toks = [int(rng.integers(0, 50)) for _ in range(max(lengths))]
        for j, n in enumerate(lengths):
            zs[t, j] = watermark.detect(src, key, toks[:n])["z"]
    return np.median(zs, axis=0)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    lengths = [10, 25, 50, 100, 150, 200, 300, 400]

    regimes = [(0.05, "low entropy (alpha=0.05)", "#C44E52"),
               (0.3, "medium entropy (alpha=0.3)", "#DD8452"),
               (2.0, "high entropy (alpha=2.0)", "#4C72B0")]

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for alpha, label, color in regimes:
        zs = z_curve(alpha, lengths)
        ax.plot(lengths, zs, "o-", color=color, lw=2, label=label)
    ax.plot(lengths, control_curve(lengths), "s--", color="gray", lw=1.5,
            label="not watermarked (control)")

    ax.axhline(3.09, color="k", ls=":", lw=1)
    ax.text(lengths[-1], 3.4, "p = 0.001", ha="right", fontsize=9)
    ax.set_xlabel("tokens observed")
    ax.set_ylabel("detection z-score")
    ax.set_title("Watermark detection power vs length and entropy")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "watermark_power.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
