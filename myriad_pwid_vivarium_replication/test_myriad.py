"""Fast checks of inputs, transition logic and a tiny end-to-end run."""
import numpy as np
import pandas as pd
import pytest

from myriad import parameters as P
from myriad import simulation as S
from myriad.components import CAUSES, _transition_matrix


def test_annual_to_monthly_round_trip():
    p = 0.123
    assert 1 - (1 - P.annual_to_monthly(p)) ** 12 == pytest.approx(p)


def test_drug_transition_matrix_is_stochastic():
    m = _transition_matrix(P.DRUG_STATES, P.DRUG_TRANSITIONS)
    assert np.allclose(m.sum(axis=1), 1.0)
    assert (m.to_numpy() >= 0).all()
    assert m.loc["remission", "high"] == pytest.approx(0.0977)


def test_sbi_trend_reproduces_table1():
    # Table 1: 2.7% annual probability of SBI hospitalisation in 2018
    assert P.sbi_annual_probability_idu(2018) == pytest.approx(0.027, abs=5e-4)


def test_overdose_schedule_matches_supplement():
    base = P.overdose_schedule("base")
    assert 1 - (1 - base[2025]["high"]) ** 12 == pytest.approx(0.0117)
    assert 1 - (1 - base[2034]["low"]) ** 12 == pytest.approx(0.0242)
    decline = P.overdose_schedule("sustained_decline")
    assert decline[2020] == base[2020]  # burn-in years shared
    assert decline[2034]["high"] < base[2034]["high"]


def test_initial_distributions_sum_to_one():
    assert sum(P.HCV_INITIAL_FIBROSIS.values()) == pytest.approx(1.0)
    for v in P.HIV_INITIAL_CD4.values():
        assert sum(v) == pytest.approx(1.0)
    assert P.SBI_TYPE_P.sum() == pytest.approx(1.0)


@pytest.fixture(scope="module")
def tiny_run():
    monthly = S.run(n_simulants=3_000, seed=7, end_year=2020, verbose=False)
    return monthly, S.annualise(monthly)


def test_population_is_conserved(tiny_run):
    monthly, _ = tiny_run
    n = monthly["n_simulants"].iloc[0]
    scale = monthly["scale"].iloc[0]
    death_cols = [f"deaths_{c}" for c in CAUSES + ["other"]]
    deaths = monthly[death_cols].sum().sum()
    assert monthly["alive_end"].iloc[-1] + deaths == n
    assert monthly["person_months"].iloc[0] == n
    assert scale == pytest.approx(P.N_PWID_2018 / n)


def test_outcomes_in_plausible_range(tiny_run):
    _, annual = tiny_run
    y = annual.loc[2018]
    # S-B3 target for 2018: 30,226 IDU-related overdose deaths (tiny run, loose bounds)
    assert 15_000 < y["deaths_overdose"] < 50_000
    # ~2.7%/yr SBI among 3.7M PWID (x HIV multiplier) -> ~100-130k
    assert 70_000 < y["new_sbi"] < 170_000
    assert y["cost_hiv"] > 0 and y["cost_hcv"] > 0 and y["cost_sbi"] > 0
    assert (annual["alive_dec"] <= annual["alive_end_jan"]).all()


def test_discounting(tiny_run):
    _, annual = tiny_run
    f = annual.loc[2020, "cost_hiv_disc"] / annual.loc[2020, "cost_hiv"]
    assert f == pytest.approx(1.03 ** 5)  # 2020 is 5 years before the 2025 base year
