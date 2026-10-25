"""Fixed-point cumulative distributions and entropy helpers.

Both the watermark and the steganographic channel operate on a categorical
next-token distribution ``p``. The steganographic channel needs that
distribution as *exact integer* cumulative boundaries so that encoder and
decoder agree bit-for-bit; floating point would drift. This module builds those
boundaries with a largest-remainder rule that (a) sums exactly to ``2**precision``
and (b) gives every supported token a width of at least one, so no reachable
token is ever assigned an empty interval.
"""

from __future__ import annotations

import numpy as np


def entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy of a distribution in bits."""
    p = np.asarray(p, dtype=np.float64)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log2(nz)))


def integer_cumulative(p: np.ndarray, precision: int) -> np.ndarray:
    """Return integer cumulative boundaries ``C`` of length ``len(p)+1``.

    ``C[0] == 0``, ``C[-1] == 2**precision``, and ``C[i+1] - C[i] >= 1`` for every
    token with ``p[i] > 0``. The largest-remainder method keeps the integer
    widths as close as possible to ``p * 2**precision`` while summing exactly.
    """
    p = np.asarray(p, dtype=np.float64)
    total = 1 << precision
    support = p > 0
    n_support = int(support.sum())
    if n_support == 0:
        raise ValueError("distribution has no support")
    if n_support > total:
        raise ValueError("precision too small for this vocabulary")

    # Reserve one unit for each supported token, distribute the rest by weight.
    remaining = total - n_support
    raw = p * remaining
    widths = np.floor(raw).astype(np.int64)
    widths[support] += 1  # the reserved unit

    deficit = total - int(widths.sum())
    if deficit > 0:
        # Hand out the leftover units to the largest fractional remainders.
        frac = raw - np.floor(raw)
        frac[~support] = -1.0  # never give width to unsupported tokens
        order = np.argsort(frac)[::-1]
        widths[order[:deficit]] += 1
    elif deficit < 0:  # pragma: no cover - defensive
        raise AssertionError("over-allocated width")

    C = np.zeros(len(p) + 1, dtype=np.int64)
    np.cumsum(widths, out=C[1:])
    assert C[-1] == total
    return C


def find_interval(C: np.ndarray, value: int) -> int:
    """Return the token index whose interval ``[C[i], C[i+1])`` contains ``value``."""
    # searchsorted on the right edge, minus one, is the containing bucket.
    return int(np.searchsorted(C, value, side="right") - 1)


def common_prefix_len(a: int, b: int, precision: int) -> int:
    """Length of the shared high-bit prefix of two ``precision``-bit integers."""
    diff = a ^ b
    return precision - diff.bit_length()


def high_bits(value: int, k: int, precision: int) -> np.ndarray:
    """Return the top ``k`` bits of a ``precision``-bit integer, MSB first."""
    if k == 0:
        return np.empty(0, dtype=np.uint8)
    shift = precision - k
    top = value >> shift
    return np.array([(top >> (k - 1 - j)) & 1 for j in range(k)], dtype=np.uint8)
