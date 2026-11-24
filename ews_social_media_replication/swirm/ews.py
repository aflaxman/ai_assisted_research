"""Early-warning signals: Gaussian detrending, rolling variance and lag-1
autocorrelation, and the Kendall-tau trend statistic.

These mirror the R ``earlywarnings`` package the paper uses: detrend each
segment with a Gaussian kernel, slide a window of 50% of the segment length,
compute variance and lag-1 autocorrelation in each window, and summarize the
trend of each signal with Kendall's tau. A rising signal (positive tau) is the
warning.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau


def gaussian_detrend(y: np.ndarray, bandwidth_frac: float = 0.10) -> np.ndarray:
    """Remove a Gaussian-kernel-smoothed trend from ``y``.

    ``bandwidth_frac`` is the kernel standard deviation as a fraction of the
    series length -- the ``earlywarnings`` package uses a comparable bandwidth.
    Returns the residual (detrended) series. Uses an edge-reflecting Gaussian
    filter so the whole thing is O(n).
    """
    y = np.asarray(y, dtype=float)
    bw = max(bandwidth_frac * y.size, 1.0)
    trend = gaussian_filter1d(y, sigma=bw, mode="reflect")
    return y - trend


def _rolling_variance(y: np.ndarray, win: int) -> np.ndarray:
    """Variance in a sliding window of length ``win`` (population variance)."""
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    s1 = c1[win:] - c1[:-win]
    s2 = c2[win:] - c2[:-win]
    mean = s1 / win
    return s2 / win - mean * mean


def _rolling_lag1_autocorr(y: np.ndarray, win: int) -> np.ndarray:
    """Lag-1 autocorrelation in a sliding window of length ``win``.

    Vectorized with cumulative sums: for each window the numerator is
    ``sum((x_t - m)(x_{t+1} - m))`` over the ``win-1`` adjacent pairs and the
    denominator is ``win`` times the window variance.
    """
    def _wsum(arr, w):  # rolling sum of length-w windows
        c = np.concatenate([[0.0], np.cumsum(arr)])
        return c[w:] - c[:-w]

    n_win = y.size - win + 1
    # window mean and variance (denominator)
    s1 = _wsum(y, win)
    s2 = _wsum(y * y, win)
    mean = s1 / win
    denom = s2 - win * mean * mean

    # numerator: sum over t of x_t x_{t+1}, minus mean-correction terms.
    prod = _wsum(y[:-1] * y[1:], win - 1)          # sum x_t x_{t+1}, win-1 terms
    lead = _wsum(y[:-1], win - 1)                   # sum of x_t   (t = k..k+win-2)
    trail = _wsum(y[1:], win - 1)                   # sum of x_{t+1}
    numer = prod[:n_win] - mean * (lead[:n_win] + trail[:n_win]) + (win - 1) * mean * mean

    out = np.divide(numer, denom, out=np.zeros(n_win), where=denom > 0)
    return out


def ews_trends(
    segment: np.ndarray,
    window_frac: float = 0.5,
    bandwidth_frac: float = 0.10,
) -> dict[str, float]:
    """Kendall-tau trend of rolling variance and lag-1 autocorrelation.

    Detrend the segment, slide a window of ``window_frac`` of its length, and
    return the Kendall tau of each rolling signal against time. Positive tau
    means the signal rises -- an early warning.
    """
    resid = gaussian_detrend(segment, bandwidth_frac)
    win = max(int(round(window_frac * resid.size)), 5)

    var = _rolling_variance(resid, win)
    ac1 = _rolling_lag1_autocorr(resid, win)
    idx = np.arange(var.size)

    tau_var = kendalltau(idx, var).statistic
    tau_ac = kendalltau(np.arange(ac1.size), ac1).statistic
    return {
        "tau_variance": float(np.nan_to_num(tau_var)),
        "tau_autocorr": float(np.nan_to_num(tau_ac)),
    }
