"""Tests for copulas.py: densities, conditional CDFs, samplers, rotations."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import integrate  # noqa: E402
from scipy import stats as sps  # noqa: E402

import copulas as cp  # noqa: E402

CASES = [
    ("gaussian", 0.5, 0),
    ("gaussian", -0.4, 0),
    ("gumbel", 0.5, 0),
    ("gumbel", 0.3, 180),
    ("gumbel", 0.6, 90),
    ("clayton", 0.5, 0),
    ("clayton", 0.4, 270),
]


@pytest.mark.parametrize("family,ktau,rot", CASES)
def test_joint_density_integrates_to_one(family, ktau, rot):
    z = np.linspace(-8.0, 8.0, 1601)
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    logc = np.asarray(cp.logpdf_z(jnp.asarray(Z1), jnp.asarray(Z2), family, ktau, rot))
    dens = np.exp(logc + sps.norm.logpdf(Z1) + sps.norm.logpdf(Z2))
    total = integrate.simpson(integrate.simpson(dens, x=z, axis=1), x=z)
    assert abs(total - 1.0) < 2e-3, total


@pytest.mark.parametrize("family,ktau,rot", CASES)
def test_hfunc_is_integral_of_density(family, ktau, rot):
    for zu, zv in [(-1.2, 0.3), (0.5, 0.5), (1.8, -0.7), (2.5, 2.0), (-2.0, -2.2)]:
        def integrand(t):
            lc = cp.logpdf_z(jnp.asarray(zu), jnp.asarray(t), family, ktau, rot)
            return float(np.exp(lc + sps.norm.logpdf(t)))
        h_num, _ = integrate.quad(integrand, -12.0, zv, limit=200)
        h = float(jnp.exp(cp.log_hfunc_z(jnp.asarray(zu), jnp.asarray(zv), family, ktau, rot)))
        assert abs(h - h_num) < 1e-5, (zu, zv, h, h_num)
        s = float(jnp.exp(cp.log_survival_hfunc_z(jnp.asarray(zu), jnp.asarray(zv), family, ktau, rot)))
        assert abs(h + s - 1.0) < 1e-12


@pytest.mark.parametrize("family,ktau,rot", [c for c in CASES if c[0] != "gaussian"])
def test_archimedean_density_matches_mixed_derivative_of_cdf(family, ktau, rot):
    eps = 1e-4
    for u, v in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.6), (0.95, 0.9)]:
        c_fd = (cp.cdf(u + eps, v + eps, family, ktau, rot) - cp.cdf(u + eps, v - eps, family, ktau, rot)
                - cp.cdf(u - eps, v + eps, family, ktau, rot) + cp.cdf(u - eps, v - eps, family, ktau, rot)) / (4 * eps ** 2)
        c = float(jnp.exp(cp.logpdf_z(jnp.asarray(sps.norm.ppf(u)), jnp.asarray(sps.norm.ppf(v)), family, ktau, rot)))
        assert abs(c - c_fd) < 1e-4 * max(1.0, c), (u, v, c, c_fd)


def test_gaussian_copula_plus_normal_margins_equals_mvn():
    rho = 0.65
    ktau = cp.param_to_ktau("gaussian", rho)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(50, 2)) * 2
    lp = np.asarray(cp.logpdf_z(jnp.asarray(z[:, 0]), jnp.asarray(z[:, 1]), "gaussian", ktau)
                    + sps.norm.logpdf(z[:, 0]) + sps.norm.logpdf(z[:, 1]))
    ref = sps.multivariate_normal(cov=[[1, rho], [rho, 1]]).logpdf(z)
    assert np.allclose(lp, ref, atol=1e-10)


def test_positive_stable_laplace_transform():
    rng = np.random.default_rng(1)
    for alpha in (0.3, 0.5, 0.8):
        v = cp._positive_stable(alpha, 400_000, rng)
        for t in (0.5, 1.0, 2.0):
            assert abs(np.mean(np.exp(-t * v)) - np.exp(-t ** alpha)) < 3e-3


@pytest.mark.parametrize("family,ktau,rot", CASES)
def test_sampler_kendall_tau_and_rotation(family, ktau, rot):
    rng = np.random.default_rng(2)
    u, v = cp.sample(family, ktau, 40_000, rng, rot)
    assert u.min() > 0 and u.max() < 1 and v.min() > 0 and v.max() < 1
    tau_hat = sps.kendalltau(u, v).statistic
    sign = -1.0 if rot in (90, 270) else 1.0
    assert abs(tau_hat - sign * ktau) < 0.02, (tau_hat, ktau)


def test_tail_asymmetry_of_samples():
    rng = np.random.default_rng(3)
    q = 0.98
    u, v = cp.sample("gumbel", 0.5, 300_000, rng)
    up = np.mean((u > q) & (v > q)) / (1 - q)
    lo = np.mean((u < 1 - q) & (v < 1 - q)) / (1 - q)
    assert up > lo + 0.15                     # upper-tail dependence
    lam_l, lam_u = cp.tail_dependence("gumbel", 0.5)
    assert lam_l == 0.0 and abs(lam_u - (2 - 2 ** 0.5)) < 1e-12
    u, v = cp.sample("gumbel", 0.5, 300_000, rng, rotation=180)
    up2 = np.mean((u > q) & (v > q)) / (1 - q)
    lo2 = np.mean((u < 1 - q) & (v < 1 - q)) / (1 - q)
    assert lo2 > up2 + 0.15                   # survival Gumbel: lower tail
    assert cp.tail_dependence("gumbel", 0.5, 180) == (lam_u, 0.0)


def test_joint_upper_tail_matches_simulation():
    rng = np.random.default_rng(4)
    u, v = cp.sample("gumbel", 0.5, 400_000, rng)
    emp = np.mean((u > 0.9) & (v > 0.97))
    assert abs(cp.joint_upper_tail(0.9, 0.97, "gumbel", 0.5) - emp) < 2e-3


def test_param_roundtrip():
    for fam in cp.FAMILIES:
        for t in (0.2, 0.5, 0.7):
            p = float(cp.ktau_to_param(fam, t))
            assert abs(float(cp.param_to_ktau(fam, p)) - t) < 1e-12


def test_gradients_finite_in_tails():
    for fam in ("gumbel", "clayton"):
        f = lambda z1, z2: jnp.sum(cp.logpdf_z(z1, z2, fam, 0.5))  # noqa: E731
        g = jax.grad(f, argnums=(0, 1))
        z1 = jnp.array([-7.0, -3.0, 0.0, 3.0, 7.0])
        z2 = jnp.array([7.0, 3.0, 0.0, -3.0, -7.0])
        for arr in g(z1, z2):
            assert np.all(np.isfinite(np.asarray(arr)))
        hs = cp.log_survival_hfunc_z(z1, jnp.full(5, 1.88), fam, 0.5)
        assert np.all(np.isfinite(np.asarray(hs)))


def test_far_tails_of_conditional_cdf():
    # Clayton: as u -> 0, 1 - h(v|u) ~ (1 + 1/d) u^d (v^-d - 1)
    d = float(cp.ktau_to_param("clayton", 0.5))
    z1, z2 = -7.0, 1.88
    u, v = sps.norm.cdf(z1), sps.norm.cdf(z2)
    expected = np.log((1 + 1 / d) * u ** d * (v ** (-d) - 1))
    got = float(cp.log_survival_hfunc_z(jnp.asarray(z1), jnp.asarray(z2), "clayton", 0.5))
    assert abs(got - expected) < 1e-6, (got, expected)
    # Gumbel: finite and monotone far into both tails
    z = jnp.array([-35.0, -20.0, -7.0, 0.0, 7.0, 20.0, 35.0])
    ls = np.asarray(cp.log_survival_hfunc_z(z, jnp.full(7, 1.88), "gumbel", 0.5))
    assert np.all(np.isfinite(ls)) and np.all(np.diff(ls) > 0), ls
    lh = np.asarray(cp.log_hfunc_z(z, jnp.full(7, 1.88), "gumbel", 0.5))
    assert np.all(np.isfinite(lh)) and np.all(np.diff(lh) < 0), lh
    lc = np.asarray(cp.logpdf_z(z, -z, "gumbel", 0.5))
    assert np.all(np.isfinite(lc))
    # Clayton's 1 - h ~ u^d underflows float64 beyond z ~ -27 (u^d ~ e^-745), so stop at -20
    z = jnp.array([-20.0, -7.0, 0.0, 7.0, 20.0])
    ls = np.asarray(cp.log_survival_hfunc_z(z, jnp.full(5, 1.88), "clayton", 0.5))
    assert np.all(np.isfinite(ls)) and np.all(np.diff(ls) >= 0), ls
