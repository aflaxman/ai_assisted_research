import numpy as np
import pandas as pd
import pytest

from sodium_sim import build_artifact as b


def test_rr_backed_out_from_paf_reproduces_paf():
    mean, sd, tmrel, scale = 135.0, 15.0, 112.5, 10.0
    for paf in (0.1, 0.3, 0.5):
        log_rr = b.solve_rr_from_paf(paf, mean, sd, tmrel, scale, np.log(4.0))
        model_paf = 1 - 1 / b.expected_relative_risk(log_rr, mean, sd, tmrel, scale)
        assert model_paf == pytest.approx(paf, abs=1e-4)


def test_rr_cap_and_zero_paf():
    assert b.solve_rr_from_paf(0.0, 135, 15, 112.5, 10, np.log(4)) == 0.0
    # A PAF above MAX_PAF is capped to MAX_PAF, and the log RR never exceeds its cap.
    log_rr = b.solve_rr_from_paf(0.999, 135, 15, 112.5, 10, np.log(4))
    assert log_rr <= np.log(4)
    assert 1 - 1 / b.expected_relative_risk(log_rr, 135, 15, 112.5, 10) == pytest.approx(b.MAX_PAF, abs=1e-4)
    assert b.solve_rr_from_paf(0.95, 120, 8, 112.5, 10, np.log(1.2)) == pytest.approx(np.log(1.2))


def test_incidence_from_prevalence_recovers_constant_hazard():
    # With constant incidence i per susceptible, no remission and no excess
    # mortality, prevalence is 1 - exp(-i * age); the derivation should return i.
    i = 0.01
    bins = b.age_bins()
    rows = []
    for sex in b.SEXES:
        for _, r in bins.iterrows():
            mid = (r.age_start + r.age_end) / 2
            rows.append((sex, r.age_start, r.age_end, b.YEAR_START, b.YEAR_END, 1 - np.exp(-i * mid)))
    prev = pd.DataFrame(rows, columns=["sex", "age_start", "age_end", "year_start", "year_end", "value"])
    emr = prev.copy()
    emr["value"] = 0.0
    inc = b.incidence_from_prevalence(prev, emr)
    adult = inc[(inc.age_start >= 10) & (inc.age_start < 90)]
    assert np.allclose(adult["value"], i, rtol=0.1)
