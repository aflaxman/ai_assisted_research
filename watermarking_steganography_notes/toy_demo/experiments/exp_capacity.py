"""Steganographic capacity tracks text entropy.

Embeds a long payload through the arithmetic-coding channel over text whose
per-token entropy varies widely, then plots how many payload bits each token
actually carried against that token's entropy. The points hug the y = x line:
the channel extracts essentially all the entropy the model offers, no more.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stegowm import stego  # noqa: E402
from stegowm.prf import bytes_to_bits, keystream  # noqa: E402
from stegowm.sources import SyntheticSource  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


class RampSource:
    """Distributions whose entropy ramps smoothly from low to high across steps."""

    def __init__(self, vocab=64, n_steps=8000, seed=11):
        rng = np.random.default_rng(seed)
        # Slow oscillation in log-alpha, so short windows sweep low->high entropy.
        i = np.arange(n_steps)
        log_alpha = np.log(0.4) + 2.6 * np.sin(2 * np.pi * i / 500.0)
        alphas = np.exp(log_alpha)
        self.dists = np.array([rng.dirichlet(np.full(vocab, a)) for a in alphas])

    def dist(self, context):
        return self.dists[len(context) % len(self.dists)]


def windowed_capacity(window=40):
    """Sliding-window (mean entropy, mean bits/token); arithmetic coding emits
    bits in bursts, so bits and entropy line up over a window, not per token."""
    src = RampSource(vocab=64, n_steps=8000, seed=11)
    payload = bytes((i * 37 + 11) % 256 for i in range(500))  # 4000 bits
    key = b"capacity-key-001"
    header = np.zeros(stego.HEADER_BITS, dtype=np.uint8)
    plain = np.concatenate([header, bytes_to_bits(payload)])
    cipher = np.bitwise_xor(plain, keystream(key, len(plain)))
    res = stego.embed_bits(src, cipher)
    H = np.array(res.entropy_per_token, dtype=float)
    k = np.array(res.committed_per_token, dtype=float)
    kern = np.ones(window) / window
    Hw = np.convolve(H, kern, mode="valid")
    kw = np.convolve(k, kern, mode="valid")
    return Hw, kw


def alpha_sweep():
    """Mean bits/token vs mean entropy across a range of entropy regimes."""
    xs, ys = [], []
    for alpha in [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0]:
        src = SyntheticSource(vocab=64, n_steps=8000, alpha=alpha, seed=3)
        payload = bytes((i * 53 + 7) % 256 for i in range(300))
        key = b"sweep-key-000001"
        header = np.zeros(stego.HEADER_BITS, dtype=np.uint8)
        plain = np.concatenate([header, bytes_to_bits(payload)])
        cipher = np.bitwise_xor(plain, keystream(key, len(plain)))
        res = stego.embed_bits(src, cipher)
        xs.append(float(np.mean(res.entropy_per_token)))
        ys.append(float(np.mean(res.committed_per_token)))
    return np.array(xs), np.array(ys)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    Hw, kw = windowed_capacity(window=40)
    sx, sy = alpha_sweep()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    lim = [0, max(Hw.max(), kw.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, label="capacity = entropy")
    ax[0].scatter(Hw, kw, s=10, alpha=0.25, color="#4C72B0", label="40-token window")
    ax[0].set_xlabel("windowed mean entropy  [bits]")
    ax[0].set_ylabel("windowed payload bits / token")
    ax[0].set_title("Capacity tracks entropy within one run")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)

    ax[1].plot([0, sx.max() * 1.05], [0, sx.max() * 1.05], "k--", lw=1,
               label="capacity = entropy")
    ax[1].plot(sx, sy, "o-", color="#55A868", lw=2, ms=7)
    ax[1].set_xlabel("mean token entropy  [bits]")
    ax[1].set_ylabel("mean payload bits / token")
    ax[1].set_title("Averaged over 8 entropy regimes")
    ax[1].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "capacity_vs_entropy.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print("alpha-sweep (entropy -> bits/token):")
    for x, y in zip(sx, sy):
        print(f"  H={x:5.2f}  ->  {y:5.2f} bits/token   ({100*y/x:.0f}% of entropy)")


if __name__ == "__main__":
    main()
