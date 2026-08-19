"""At-Most-One-Changepoint (AMOC) mean-shift detection.

The paper finds the emergence changepoint with R's ``cpt.mean`` under the AMOC
method: test whether a single shift in the mean fits better than none. We
reproduce it directly. For a single split point ``k`` the maximum-likelihood
gain over "no change" (assuming common variance) is a function of the two
segment means; we scan every ``k`` in O(n) with cumulative sums and accept the
best split when its gain clears an asymptotic (SIC/BIC-like) penalty.
"""

from __future__ import annotations

import numpy as np


def amoc_meanshift(y: np.ndarray, penalty: float | None = None) -> int | None:
    """Return the index of a single mean-shift changepoint, or ``None``.

    The test statistic is ``-2 log LR`` for a change in the mean of a normal
    series with common (unknown) variance. Its null distribution is the maximum
    over all candidate splits, so the naive ``log(n)`` (SIC) penalty over-fires.
    We instead use ``7 + 0.5 log(n)``, a threshold calibrated by Monte Carlo to
    a 5% false-positive rate -- the same nominal type-I error (0.05) the paper
    targets with the ``changepoint`` package's "Asymptotic" penalty. The
    changepoint is reported as the first index of the second segment.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        return None
    if penalty is None:
        penalty = 7.0 + 0.5 * np.log(n)

    total = y.sum()
    total_sq = (y * y).sum()
    csum = np.cumsum(y)

    # For split after index k (1..n-1): SSE = total_sq - left_mean^2*k - right_mean^2*(n-k)
    k = np.arange(1, n)
    left_sum = csum[:-1]
    right_sum = total - left_sum
    sse = total_sq - left_sum ** 2 / k - right_sum ** 2 / (n - k)

    null_sse = total_sq - total ** 2 / n
    # Gaussian log-likelihood-ratio test statistic for a change in mean.
    best = np.argmin(sse)
    sse_min = sse[best]
    if sse_min <= 0:
        return int(best + 1)
    stat = n * (np.log(null_sse / n) - np.log(sse_min / n))
    return int(best + 1) if stat > penalty else None


def changepoint_time(times: np.ndarray, series: np.ndarray) -> float | None:
    """Time of the mean-shift changepoint in ``series`` (or ``None``)."""
    k = amoc_meanshift(series)
    return None if k is None else float(times[k])
