"""Bivariate copulas for latent-variable models, in JAX.

Families
--------
"gaussian"  symmetric, tail independent (the usual multivariate-normal choice)
"gumbel"    upper-tail dependence (the family used by Kwon & Zhang)
"clayton"   lower-tail dependence

The Archimedean families (gumbel, clayton) are parameterised by Kendall's
tau in (0, 1); the Gaussian by Kendall's tau in (-1, 1).  ``rotation`` in
{0, 90, 180, 270} rotates an Archimedean copula so that negative
dependence (90, 270) or the opposite tail (180, the "survival" copula) can
be modelled.

Everything that enters a NumPyro model takes standard-normal scores
(z1, z2) = (Phi^-1(u), Phi^-1(v)) and works with log-CDFs internally so the
tails stay numerically stable.  Enable 64-bit JAX (``numpyro.enable_x64()``)
before use; float32 loses the tails.

Joint density of two latent variables with N(0, 1) margins:

    log p(z1, z2) = log phi(z1) + log phi(z2) + logpdf_z(z1, z2, ...)

so a model can draw z1, z2 ~ Normal(0, 1) and add ``numpyro.factor`` with
``logpdf_z``.  ``log_hfunc_z`` gives log P(V <= v | U = u), the conditional
CDF, which is what a binary outcome with a latent threshold needs.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from scipy import stats as sps

FAMILIES = ("gaussian", "gumbel", "clayton")
ROTATIONS = (0, 90, 180, 270)


# --------------------------------------------------------------------------
# Parameter conversions
# --------------------------------------------------------------------------

def ktau_to_param(family, ktau):
    """Kendall's tau -> native parameter (rho for gaussian, delta otherwise)."""
    if family == "gaussian":
        return jnp.sin(jnp.pi * ktau / 2.0)
    if family == "gumbel":
        return 1.0 / (1.0 - ktau)
    if family == "clayton":
        return 2.0 * ktau / (1.0 - ktau)
    raise ValueError(f"unknown family {family!r}")


def param_to_ktau(family, param):
    """Native parameter -> Kendall's tau."""
    param = np.asarray(param, dtype=float)
    if family == "gaussian":
        return 2.0 / np.pi * np.arcsin(param)
    if family == "gumbel":
        return 1.0 - 1.0 / param
    if family == "clayton":
        return param / (param + 2.0)
    raise ValueError(f"unknown family {family!r}")


def tail_dependence(family, ktau, rotation=0):
    """(lambda_lower, lambda_upper) of the copula in its (u, v) orientation.

    For rotations 90 and 270 the dependence sits in the off-diagonal
    corners, so both diagonal coefficients are zero.
    """
    if family == "gaussian" or rotation in (90, 270):
        return (0.0, 0.0)
    d = float(ktau_to_param(family, ktau))
    if family == "gumbel":
        lo, up = 0.0, 2.0 - 2.0 ** (1.0 / d)
    else:
        lo, up = 2.0 ** (-1.0 / d), 0.0
    return (up, lo) if rotation == 180 else (lo, up)


# --------------------------------------------------------------------------
# Base (unrotated) log densities and conditional CDFs
# --------------------------------------------------------------------------

def _gumbel_parts(logu, logv, delta):
    """Pieces of the Gumbel copula, arranged so the tails do not cancel.

    With x = -log u, y = -log v, S = log(x^d + y^d) = d*log(max) + r where
    r = log1p(exp(-d*|log x - log y|)) is the small correction.  Keeping r
    separate avoids absorbing it into the large term.
    """
    x, y = -logu, -logv                      # x, y > 0
    lx, ly = jnp.log(x), jnp.log(y)
    lM, lm = jnp.maximum(lx, ly), jnp.minimum(lx, ly)
    M, m = jnp.maximum(x, y), jnp.minimum(x, y)
    r = jnp.log1p(jnp.exp(-delta * jnp.abs(lx - ly)))
    return x, y, lx, ly, M, m, lM, lm, r


def _gumbel_logpdf(logu, logv, delta):
    x, y, lx, ly, M, m, lM, lm, r = _gumbel_parts(logu, logv, delta)
    A = M * jnp.exp(r / delta)               # (x^d + y^d)^(1/d) = -log C(u, v)
    return (-M * jnp.expm1(r / delta) + m + (delta - 1.0) * lm - delta * lM
            + (1.0 / delta - 2.0) * r + jnp.log(A + delta - 1.0))


def _gumbel_logh(logu, logv, delta):
    """log dC/du = log P(V <= v | U = u)."""
    x, y, lx, ly, M, m, lM, lm, r = _gumbel_parts(logu, logv, delta)
    base = (delta - 1.0) * (lx - lM) + (1.0 / delta - 1.0) * r
    big = x >= y
    return jnp.where(big, -x * jnp.expm1(r / delta) + base,
                     -y * jnp.exp(r / delta) + x + base)


def _clayton_parts(logu, logv, delta):
    """log(u^-d + v^-d - 1) = -d*lmin + r with r = log1p(u_max^d (v_min^-d - 1))."""
    lmin, lmax = jnp.minimum(logu, logv), jnp.maximum(logu, logv)
    r = jnp.log1p(jnp.exp(delta * lmin) * jnp.expm1(-delta * lmax))
    return lmin, lmax, r


def _clayton_logpdf(logu, logv, delta):
    lmin, lmax, r = _clayton_parts(logu, logv, delta)
    return (jnp.log1p(delta) + delta * lmin - (1.0 + delta) * lmax
            - (2.0 + 1.0 / delta) * r)


def _clayton_logh(logu, logv, delta):
    lmin, lmax, r = _clayton_parts(logu, logv, delta)
    return (1.0 + delta) * (lmin - logu) - (1.0 + 1.0 / delta) * r


def _gaussian_logpdf(z1, z2, rho):
    r2 = rho ** 2
    return (-0.5 * jnp.log1p(-r2)
            - (r2 * (z1 ** 2 + z2 ** 2) - 2.0 * rho * z1 * z2) / (2.0 * (1.0 - r2)))


def _gaussian_logh(z1, z2, rho):
    return jnorm.logcdf((z2 - rho * z1) / jnp.sqrt(1.0 - rho ** 2))


def _base_logpdf(z1, z2, family, ktau):
    p = ktau_to_param(family, ktau)
    if family == "gaussian":
        return _gaussian_logpdf(z1, z2, p)
    logu, logv = jnorm.logcdf(z1), jnorm.logcdf(z2)
    if family == "gumbel":
        return _gumbel_logpdf(logu, logv, p)
    if family == "clayton":
        return _clayton_logpdf(logu, logv, p)
    raise ValueError(f"unknown family {family!r}")


def _base_logh(z1, z2, family, ktau):
    p = ktau_to_param(family, ktau)
    if family == "gaussian":
        return _gaussian_logh(z1, z2, p)
    logu, logv = jnorm.logcdf(z1), jnorm.logcdf(z2)
    if family == "gumbel":
        return _gumbel_logh(logu, logv, p)
    if family == "clayton":
        return _clayton_logh(logu, logv, p)
    raise ValueError(f"unknown family {family!r}")


def _log1mexp(a):
    """log(1 - exp(a)) for a <= 0, stable at both ends (double-where trick)."""
    big = a > -0.693
    a1 = jnp.where(big, a, -1.0)
    a2 = jnp.where(big, -1.0, a)
    return jnp.where(big, jnp.log(-jnp.expm1(a1)), jnp.log1p(-jnp.exp(a2)))


# --------------------------------------------------------------------------
# Public JAX API (standard-normal scores in, logs out)
# --------------------------------------------------------------------------

def _orient(z1, z2, rotation):
    if rotation == 0:
        return z1, z2
    if rotation == 90:
        return -z1, z2
    if rotation == 180:
        return -z1, -z2
    if rotation == 270:
        return z1, -z2
    raise ValueError(f"rotation must be one of {ROTATIONS}")


def logpdf_z(z1, z2, family, ktau, rotation=0):
    """log c(u, v) at u = Phi(z1), v = Phi(z2) for the (rotated) copula."""
    a, b = _orient(z1, z2, rotation)
    return _base_logpdf(a, b, family, ktau)


def log_hfunc_z(z1, z2, family, ktau, rotation=0):
    """log P(V <= Phi(z2) | U = Phi(z1)) for the (rotated) copula."""
    if rotation == 0:
        return _base_logh(z1, z2, family, ktau)
    if rotation == 90:
        return _base_logh(-z1, z2, family, ktau)
    if rotation == 180:
        return _log1mexp(_base_logh(-z1, -z2, family, ktau))
    if rotation == 270:
        return _log1mexp(_base_logh(z1, -z2, family, ktau))
    raise ValueError(f"rotation must be one of {ROTATIONS}")


def log_survival_hfunc_z(z1, z2, family, ktau, rotation=0):
    """log P(V > Phi(z2) | U = Phi(z1)) for the (rotated) copula."""
    return _log1mexp(log_hfunc_z(z1, z2, family, ktau, rotation))


# --------------------------------------------------------------------------
# NumPy helpers for simulation and analysis
# --------------------------------------------------------------------------

def _base_cdf(u, v, family, ktau):
    p = float(ktau_to_param(family, ktau))
    if family == "gaussian":
        z = np.stack([sps.norm.ppf(u), sps.norm.ppf(v)], axis=-1)
        mvn = sps.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, p], [p, 1.0]])
        return mvn.cdf(z)
    if family == "gumbel":
        x, y = -np.log(u), -np.log(v)
        return np.exp(-(x ** p + y ** p) ** (1.0 / p))
    if family == "clayton":
        return (u ** (-p) + v ** (-p) - 1.0) ** (-1.0 / p)
    raise ValueError(f"unknown family {family!r}")


def cdf(u, v, family, ktau, rotation=0):
    """C(u, v) of the (rotated) copula (NumPy, for analysis)."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    if rotation == 0:
        return _base_cdf(u, v, family, ktau)
    if rotation == 90:
        return v - _base_cdf(1.0 - u, v, family, ktau)
    if rotation == 180:
        return u + v - 1.0 + _base_cdf(1.0 - u, 1.0 - v, family, ktau)
    if rotation == 270:
        return u - _base_cdf(u, 1.0 - v, family, ktau)
    raise ValueError(f"rotation must be one of {ROTATIONS}")


def joint_upper_tail(q1, q2, family, ktau, rotation=0):
    """P(U > q1, V > q2) = 1 - q1 - q2 + C(q1, q2)."""
    return 1.0 - q1 - q2 + cdf(q1, q2, family, ktau, rotation)


def _positive_stable(alpha, n, rng):
    """Positive stable variates with Laplace transform exp(-t^alpha).

    Chambers-Mallows-Stuck, as used for Gumbel sampling (Hofert, 2011).
    """
    th = rng.uniform(0.0, np.pi, size=n)
    w = rng.exponential(size=n)
    return ((np.sin(alpha * th) / np.sin(th) ** (1.0 / alpha))
            * (np.sin((1.0 - alpha) * th) / w) ** ((1.0 - alpha) / alpha))


def sample(family, ktau, n, rng, rotation=0):
    """Draw n pairs (u, v) from the (rotated) copula with the given Kendall's tau.

    Archimedean families use the Marshall-Olkin frailty construction, so
    they need ktau > 0.
    """
    p = float(ktau_to_param(family, ktau))
    if family == "gaussian":
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, p], [p, 1.0]], size=n)
        u, v = sps.norm.cdf(z[:, 0]), sps.norm.cdf(z[:, 1])
    else:
        e = rng.exponential(size=(n, 2))
        if family == "gumbel":
            alpha = 1.0 / p
            frailty = _positive_stable(alpha, n, rng)
            uv = np.exp(-(e / frailty[:, None]) ** alpha)
        elif family == "clayton":
            frailty = rng.gamma(1.0 / p, 1.0, size=n)
            uv = (1.0 + e / frailty[:, None]) ** (-1.0 / p)
        else:
            raise ValueError(f"unknown family {family!r}")
        u, v = uv[:, 0], uv[:, 1]
    if rotation == 90:
        u = 1.0 - u
    elif rotation == 180:
        u, v = 1.0 - u, 1.0 - v
    elif rotation == 270:
        v = 1.0 - v
    elif rotation != 0:
        raise ValueError(f"rotation must be one of {ROTATIONS}")
    return u, v


def sample_z(family, ktau, n, rng, rotation=0):
    """Copula sample mapped to standard-normal scores (z1, z2)."""
    u, v = sample(family, ktau, n, rng, rotation)
    return sps.norm.ppf(u), sps.norm.ppf(v)
