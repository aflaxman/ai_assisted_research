"""Tests pin the E-value implementation to published values.

Sources:
- Andrade C. E-value regression. J Clin Psychiatry. 2026;87(1):26f16324.
- VanderWeele TJ, Ding P. Ann Intern Med. 2017;167(4):268-274.
"""

import math

import pytest

from evalue import bias_factor, evalue, evalue_ci


def test_andrade_acetaminophen_asd_point_estimate():
    # Andrade: adjusted HR 1.05 for gestational acetaminophen and ASD
    # (Ahlqvist et al. 2024) gives E-value 1.28.
    assert round(evalue(1.05), 2) == 1.28


def test_andrade_acetaminophen_asd_ci_bound():
    # Andrade: lower CI bound 1.02 gives E-value 1.16.
    assert round(evalue_ci(1.05, 1.02, 1.08), 2) == 1.16


def test_andrade_protective_or():
    # Andrade: an OR of 0.50 (reciprocal 2.00) gives E-value 3.41.
    assert round(evalue(0.50), 2) == 3.41


def test_andrade_lee_antidepressant():
    # Andrade: HR 1.46 from Lee et al. gives E-value 2.28.
    assert round(evalue(1.46), 2) == 2.28


def test_vanderweele_ding_breastfeeding_leukemia():
    # VanderWeele & Ding 2017: RR 3.9 (never vs ever breastfed, childhood
    # leukemia example) gives E-value about 7.26.
    assert round(evalue(3.9), 2) == 7.26


def test_null_needs_no_confounding():
    assert evalue(1.0) == 1.0
    assert evalue_ci(1.05, 0.98, 1.12) == 1.0


def test_protective_symmetry():
    assert math.isclose(evalue(0.5), evalue(2.0))


def test_ci_bound_choice_for_protective_estimate():
    # For RR < 1 the bound closer to the null is the upper bound.
    assert math.isclose(evalue_ci(0.5, 0.4, 0.8), evalue(0.8))


def test_evalue_solves_bias_factor():
    # The E-value is the confounder strength E with bias_factor(E, E) == RR.
    for rr in (1.05, 1.5, 3.9):
        e = evalue(rr)
        assert math.isclose(bias_factor(e, e), rr, rel_tol=1e-9)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        evalue(0)
    with pytest.raises(ValueError):
        evalue_ci(1.05, 1.08, 1.02)
    with pytest.raises(ValueError):
        bias_factor(0.9, 2.0)
