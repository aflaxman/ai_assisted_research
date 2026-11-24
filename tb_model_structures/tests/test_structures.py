"""Acceptance tests for the structure registry and ODE engine (milestone M2)."""
from __future__ import annotations

import numpy as np
import pytest

from tbmodels.classic import all_classic
from tbmodels.registry import all_structures
from tbmodels.spectrum import all_spectrum

# a parameter dict covering every parameter used by any structure
PARAMS = dict(
    c=0.10, d=0.50, e=0.30, f=0.20, g=1.00, a=0.10, b=0.20,
    # structure K uses x2 as a (negative) exponent; structure L uses x2 as a
    # non-negative additive offset, so it is overridden in the L-specific test.
    x1=0.12, x2=-0.50, x3=0.50,
    self_cure=0.30, tb_death=0.30,
    # spectrum
    infclr=1.83, infmin=0.10, infsub=0.04, minrec=0.18, min_sub=0.24,
    sub_min=1.58, sub_clin=0.72, clin_sub=0.58, mort=0.33, sub_rec=0.20,
    rel_inf_sub=0.50,
)

TIMES = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
ALL = all_structures()


@pytest.mark.parametrize("name", list(ALL))
def test_builds_and_entry_sums_to_one(name):
    s = ALL[name]
    dist = s.entry(PARAMS)
    assert np.isclose(sum(dist.values()), 1.0)
    for comp in dist:
        assert comp in s.names()


@pytest.mark.parametrize("name", list(ALL))
def test_mass_conservation(name):
    s = ALL[name]
    traj = s.simulate_cohort(PARAMS, TIMES)
    mass = traj[:, s.mass_compartments()].sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-6), f"{name}: mass drift {mass}"
    assert (traj > -1e-9).all(), f"{name}: negative occupancy"


@pytest.mark.parametrize("name", list(ALL))
def test_cumulative_incidence_monotone_bounded(name):
    s = ALL[name]
    ci = s.cumulative_incidence(PARAMS, TIMES)
    assert np.all(np.diff(ci) >= -1e-8), f"{name}: non-monotone CI {ci}"
    assert ci[0] >= -1e-9 and ci[-1] <= 1.0 + 1e-6


def test_structure_A_analytic():
    """Cumulative incidence to I for structure A equals 1 - exp(-c t)."""
    s = all_classic()["A"]
    ci = s.cumulative_incidence(PARAMS, TIMES)
    expected = 1.0 - np.exp(-PARAMS["c"] * TIMES)
    assert np.allclose(ci, expected, atol=1e-5)


def test_A_equals_E_with_a_zero():
    A = all_classic()["A"]
    E = all_classic()["E"]
    p = dict(PARAMS, a=0.0)
    ciA = A.cumulative_incidence(p, TIMES)
    ciE = E.cumulative_incidence(p, TIMES)
    assert np.allclose(ciA, ciE, atol=1e-6)


def test_C_equals_G_with_a_zero():
    C = all_classic()["C"]
    G = all_classic()["G"]
    p = dict(PARAMS, a=0.0)
    ciC = C.cumulative_incidence(p, TIMES)
    ciG = G.cumulative_incidence(p, TIMES)
    assert np.allclose(ciC, ciG, atol=1e-6)


def test_expm_matches_ode_for_constant_rates():
    s = all_classic()["C"]
    expm_traj = s.simulate_cohort(PARAMS, TIMES, method="expm")
    ode_traj = s.simulate_cohort(PARAMS, TIMES, method="ode")
    assert np.allclose(expm_traj, ode_traj, atol=1e-6)


def test_structure_K_analytic():
    """Structure K hazard x1*max(1,t)^x2 integrates to a known cumulative."""
    s = all_classic()["K"]
    x1, x2 = PARAMS["x1"], PARAMS["x2"]
    t = np.array([0.5, 1.0, 2.0, 5.0, 10.0])

    def cum_hazard(tt):
        if tt <= 1.0:
            return x1 * tt
        return x1 * (1.0 + (tt ** (x2 + 1) - 1.0) / (x2 + 1))

    expected = np.array([1.0 - np.exp(-cum_hazard(tt)) for tt in t])
    ci = s.cumulative_incidence(PARAMS, t)
    assert np.allclose(ci, expected, atol=1e-4), f"K: {ci} vs {expected}"


def test_structure_L_analytic():
    """Structure L hazard x1*(x2+exp(-x3 t)) with x2>=0 integrates to a known
    cumulative: integral_0^t = x1*(x2*t + (1-exp(-x3 t))/x3)."""
    s = all_classic()["L"]
    p = dict(PARAMS, x1=0.1, x2=0.3, x3=0.5)
    t = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    cumh = p["x1"] * (p["x2"] * t + (1.0 - np.exp(-p["x3"] * t)) / p["x3"])
    expected = 1.0 - np.exp(-cumh)
    ci = s.cumulative_incidence(p, t)
    assert np.allclose(ci, expected, atol=1e-4), f"L: {ci} vs {expected}"


def test_classic_structures_run_in_transmission_mode():
    """Every classic structure (which carries S and the infection transitions)
    integrates as a transmission model and conserves people (births/deaths off).
    Spectrum structures get their susceptible pool in the M4 embedding."""
    times = np.linspace(0.0, 30.0, 7)
    for name, s in all_classic().items():
        x0 = np.zeros(s.n)
        x0[s.idx("S")] = 9000.0
        x0[s.idx("I")] = 100.0  # seed infectious disease so FOI is nonzero
        traj = s.simulate_transmission(dict(PARAMS), times, x0, beta=4.0)
        mass = traj[:, s.mass_compartments()].sum(axis=1)
        assert np.allclose(mass, mass[0], atol=1e-3), f"{name}: mass not conserved"


def test_spectrum_variants_present():
    sp = all_spectrum()
    assert set(sp) == {"SP-full", "SP-noclear", "SP-nomin", "SP-nomin-noclear"}
    # SP-full has a minimal compartment and a cleared compartment; SP-nomin-noclear has neither
    assert "Min" in sp["SP-full"].names() and "Clr" in sp["SP-full"].names()
    assert "Min" not in sp["SP-nomin"].names()
    assert "Clr" not in sp["SP-noclear"].names()


def test_spectrum_reproduces_horton_headline_numbers():
    """With Horton 2023's posterior medians, SP-full reproduces their published
    mechanism: ~92% self-clearance by 10y, ~71% of progressors via minimal
    disease, and the median-plug competing-risks TB fraction
    (infmin+infsub)/(infclr+infmin+infsub)."""
    from tbmodels.spectrum import spectrum_full
    s = spectrum_full()
    p = dict(infclr=1.83, infmin=0.10, infsub=0.04, minrec=0.18, min_sub=0.24,
             sub_min=1.58, sub_clin=0.72, clin_sub=0.58, mort=0.33, rel_inf_sub=0.5)
    t = np.array([2.0, 10.0])
    occ = s.simulate_cohort(p, t)
    cleared10 = occ[1, s.idx("Clr")]
    assert 0.90 < cleared10 < 0.94  # Horton: 92.0%
    via_min = p["infmin"] / (p["infmin"] + p["infsub"])
    assert abs(via_min - 0.709) < 0.02  # Horton: 70.9%
    ci_any = s.cumulative_incidence(p, t, target=("Min", "Sub", "Clin"))
    frac = (p["infmin"] + p["infsub"]) / (p["infclr"] + p["infmin"] + p["infsub"])
    assert abs(ci_any[1] - frac) < 0.005


def test_force_of_infection_counts_only_infectious():
    s = all_spectrum()["SP-full"]
    x = np.zeros(s.n)
    x[s.idx("Sub")] = 10.0
    x[s.idx("Clin")] = 5.0
    x[s.idx("Min")] = 100.0  # non-infectious, must not contribute
    lam = s.force_of_infection(PARAMS, x, beta=2.0, N=1000.0)
    expected = 2.0 * (PARAMS["rel_inf_sub"] * 10.0 + 1.0 * 5.0) / 1000.0
    assert np.isclose(lam, expected)
