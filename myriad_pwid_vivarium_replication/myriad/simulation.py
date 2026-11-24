"""Build and run the MYRIAD replication; scale results to the national cohort."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from loguru import logger
from vivarium import InteractiveContext

from . import parameters as P
from .components import (
    HCV,
    HIV,
    Calendar,
    DrugUse,
    Mortality,
    OutcomeObserver,
    Population,
    SevereBacterialInfection,
)

COST_COLUMNS = ["cost_drug_use", "cost_sbi", "cost_hiv", "cost_hcv"]


def build_simulation(
    n_simulants: int, seed: int = 0, scenario: str = "base"
) -> tuple[InteractiveContext, OutcomeObserver]:
    calendar = Calendar(P.COHORT_START_YEAR)
    observer = OutcomeObserver(calendar)
    components = [
        calendar,
        Population(calendar),
        DrugUse(calendar, scenario),
        HIV(calendar),
        HCV(calendar),
        SevereBacterialInfection(calendar),
        Mortality(calendar),
        observer,
    ]
    configuration = {
        "time": {
            "start": {"year": P.COHORT_START_YEAR, "month": 1, "day": 1},
            "end": {"year": P.PROJECTION_END_YEAR + 2, "month": 12, "day": 31},
            "step_size": 365.25 / 12,  # days; the Calendar component counts months
        },
        "population": {"population_size": n_simulants},
        "randomness": {"random_seed": seed, "key_columns": []},
    }
    logger.disable("vivarium")  # the engine logs every time step at INFO level
    sim = InteractiveContext(components=components, configuration=configuration)
    return sim, observer


def run(
    n_simulants: int = 100_000,
    seed: int = 0,
    scenario: str = "base",
    end_year: int = P.PROJECTION_END_YEAR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run one replicate and return monthly records scaled to 3,694,500 PWID."""
    sim, observer = build_simulation(n_simulants, seed, scenario)
    n_steps = (end_year - P.COHORT_START_YEAR + 1) * 12
    t0 = time.time()
    for k in range(n_steps):
        sim.step()
        if verbose and (k + 1) % 12 == 0:
            print(f"  {scenario} seed={seed}: year {P.COHORT_START_YEAR + k // 12} done "
                  f"({time.time() - t0:.0f}s)", flush=True)
    monthly = observer.results()
    monthly["scenario"] = scenario
    monthly["seed"] = seed
    monthly["n_simulants"] = n_simulants
    monthly["scale"] = P.N_PWID_2018 / n_simulants
    return monthly


def annualise(monthly: pd.DataFrame) -> pd.DataFrame:
    """Sum monthly records to calendar years, scale to the national cohort and discount costs."""
    scale = monthly["scale"].iloc[0]
    count_cols = [c for c in monthly.columns
                  if c.startswith(("deaths_", "new_")) or c == "person_months"]
    annual = monthly.groupby("year")[count_cols + COST_COLUMNS].sum() * scale
    # prevalence-type columns: value at the start of the year (first month)
    first = monthly.groupby("year").first()
    for c in ["alive_end", "hiv_prevalent", "hiv_on_art", "hiv_cd4_low", "hcv_prevalent"] + [f"state_{s}" for s in P.DRUG_STATES]:
        annual[f"{c}_jan"] = first[c] * scale
    annual["alive_dec"] = monthly.groupby("year")["alive_end"].last() * scale
    annual["deaths_total"] = annual[[c for c in annual.columns if c.startswith("deaths_")]].sum(axis=1)
    annual["new_infections_total"] = annual[["new_sbi", "new_hiv", "new_hcv"]].sum(axis=1)
    annual["life_years"] = annual["person_months"] / 12.0
    factor = 1.0 / (1.0 + P.DISCOUNT_RATE) ** (annual.index - P.DISCOUNT_BASE_YEAR)
    for c in COST_COLUMNS:
        annual[f"{c}_disc"] = annual[c] * factor
    annual["cost_total_disc"] = annual[[f"{c}_disc" for c in COST_COLUMNS]].sum(axis=1)
    annual["scenario"] = monthly["scenario"].iloc[0]
    annual["seed"] = monthly["seed"].iloc[0]
    return annual


def projection_summary(annual: pd.DataFrame) -> dict[str, float]:
    """Ten-year totals for 2025-2034, in the layout of the paper's Table 2."""
    a = annual.loc[P.PROJECTION_START_YEAR:P.PROJECTION_END_YEAR]
    pwid_2025 = a.loc[P.PROJECTION_START_YEAR, "alive_end_jan"]
    out = {
        "pwid_2025": pwid_2025,
        "new_sbi": a["new_sbi"].sum(),
        "new_hiv": a["new_hiv"].sum(),
        "new_hcv": a["new_hcv"].sum(),
        "new_infections_total": a["new_infections_total"].sum(),
        "deaths_overdose": a["deaths_overdose"].sum(),
        "deaths_sbi": a["deaths_sbi"].sum(),
        "deaths_hiv": a["deaths_hiv"].sum(),
        "deaths_hcv": a["deaths_hcv"].sum(),
        "deaths_other": a["deaths_other"].sum(),
        "deaths_total": a["deaths_total"].sum(),
        "cost_drug_use_billion": a["cost_drug_use_disc"].sum() / 1e9,
        "cost_sbi_billion": a["cost_sbi_disc"].sum() / 1e9,
        "cost_hiv_billion": a["cost_hiv_disc"].sum() / 1e9,
        "cost_hcv_billion": a["cost_hcv_disc"].sum() / 1e9,
        "cost_total_billion": a["cost_total_disc"].sum() / 1e9,
        "life_years_total": a["life_years"].sum(),
        "life_years_per_pwid_2025": a["life_years"].sum() / pwid_2025,
        "life_years_per_pwid_2018": a["life_years"].sum() / P.N_PWID_2018,
        # the paper's $17,375 = total cost / (PWID in 2025 x 10 years)
        "per_person_annual_cost": a["cost_total_disc"].sum() / (pwid_2025 * 10),
        "cost_per_life_year": a["cost_total_disc"].sum() / a["life_years"].sum(),
    }
    return out
