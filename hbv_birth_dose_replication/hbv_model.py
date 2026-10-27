"""Shared machinery for the Seaman et al. (2020) hepatitis B birth-dose replication.

The disease model runs in camdl (``hbv_birth_dose.camdl``). This module holds
everything around it, ported from the authors' published Python:

* reading their input spreadsheet (``data/hbv_model_inputs_revision.xlsx``);
* the birth-dose arithmetic of Figure 1 -- how coverage of each delivery
  strategy, its timing distribution and vaccine effectiveness combine into the
  share of births protected (``model_initializer`` in their model.py);
* the coverage saturation levels, the five modelled scenarios and the vaccine
  cost functions (``coverage_saturation``, ``model_scenarios``,
  ``vaccine_costs``);
* the cost-coverage curves and the SLSQP optimisations (``costcoverage.py``,
  ``optimisation_model.py``);
* writing a parameter TOML, invoking ``camdl simulate`` on the ODE backend and
  reading back the ``quantities`` it reports.

Outcomes are per cohort; ``outcomes_per_1000`` rescales to the paper's unit.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = ROOT / "hbv_birth_dose.camdl"
XLSX = ROOT / "data" / "hbv_model_inputs_revision.xlsx"

#: Cohort size handed to camdl. The model is linear in its initial state, so
#: the size is arbitrary; a million births keeps every initial compartment far
#: above camdl's whole-number rounding of count-kind initial values.
COHORT = 1_000_000.0
PER = 1000.0  # the paper reports per 1000 births

#: Fixed RK4 step in years; exact in binary.
DT = 0.03125
HORIZON = 100.0

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

GLOBAL_RATE_ROWS = {
    # authors' `global` sheet symbol -> camdl parameter
    "r_LA": "r_LA", "r_AC": "r_AC", "p_AC": "p_ac0", "r_AZ": "r_AZ", "r_CZ": "r_CZ",
    "r_CCC": "r_CCC", "r_CDC": "r_CDC", "r_DCHCC": "r_DCHCC", "r_CHCC": "r_CHCC",
    "r_CHC": "r_CHC", "mu_A": "mu_A", "mu_C": "mu_C", "mu_CC": "mu_CC", "mu_DC": "mu_DC",
    "mu_HCC": "mu_HCC", "DALY_A": "dw_A", "DALY_C": "dw_C", "DALY_CC": "dw_CC",
    "DALY_DC": "dw_DC", "DALY_HCC": "dw_HCC",
}
VE_ROWS = ["eff_vacC1", "eff_vacC2", "eff_vacC3", "eff_vacC4", "eff_vacC5"]
ACM_COLS = ["mu_01", "mu_14"] + [f"mu_{a}{a + 4}" for a in range(5, 100, 5)] + ["mu_100"]
ACM_PARAMS = ["acm_0", "acm_1"] + [f"acm_{a}" for a in range(5, 100, 5)] + ["acm_100"]
COST_COLS = {"c_A": "c_A", "c_C": "c_C", "c_CC": "c_CC", "c_DC": "c_DC", "c_HCC": "c_HCC"}
VACCINE_COST_COLS = ["fac_cc", "fac_ctc", "com_cc", "com_ctc", "com_cpad"]


@dataclass
class Inputs:
    glob: pd.DataFrame      # symbol-indexed: Value, Lower bound, Upper bound
    regions: pd.DataFrame   # one row per setting
    regions_lb: pd.DataFrame
    regions_ub: pd.DataFrame

    @property
    def settings(self) -> list[str]:
        return list(self.regions["Setting"])


def load_inputs(level: str = "regions") -> Inputs:
    """Read the authors' spreadsheet. ``level`` is ``"regions"`` or ``"countries"``."""
    xls = pd.ExcelFile(XLSX)
    glob = pd.read_excel(xls, "global").iloc[0:30].set_index("Symbol")
    return Inputs(
        glob=glob,
        regions=pd.read_excel(xls, level),
        regions_lb=pd.read_excel(xls, level + "_lb"),
        regions_ub=pd.read_excel(xls, level + "_ub"),
    )


def global_point(inp: Inputs) -> dict[str, float]:
    """Point values of the global parameters (Table 1), keyed by camdl name."""
    g = inp.glob["Value"]
    out = {camdl: float(g[sym]) for sym, camdl in GLOBAL_RATE_ROWS.items()}
    out["disc"] = float(g["discount"])
    return out


def vaccine_effectiveness(inp: Inputs) -> np.ndarray:
    """Day 1, day 2, days 3-7, days 8-41, unvaccinated (Table 1)."""
    return np.array([float(inp.glob.loc[r, "Value"]) for r in VE_ROWS])


def regional_point(row: pd.Series) -> dict[str, float]:
    """Setting-specific camdl parameters (Table 2) from one spreadsheet row."""
    out = {p: float(row[c]) for p, c in zip(ACM_PARAMS, ACM_COLS)}
    out.update({p: float(row[c]) for p, c in COST_COLS.items()})
    out.update({
        "p_fac": float(row["Facility"]),
        "prev": float(row["prev"]),
        "hbe_prev": float(row["hbe_prev"]),
        "risk_pos": float(row["hbe_risk"]),
        "risk_neg": float(row["n_hbe_risk"]),
        "hbv3": float(row["HBV3"]),
    })
    return out


# ---------------------------------------------------------------------------
# Birth-dose arithmetic (Figure 1)
# ---------------------------------------------------------------------------

# Timing distributions over the four post-partum strata (+ "never"), per
# delivery strategy -- model.py `model_initializer`.
TIMING = {
    "fac_cc":   np.array([0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3, 0.0]),
    "fac_ctc":  np.array([0.8 + 0.05 * (0.2 / 3), 0.2 / 3, 0.2 / 3, 0.95 * (0.2 / 3), 0.0]),
    "com_cc":   np.array([0.3, 0.4, 0.15, 0.15, 0.0]),
    "com_ctc":  np.array([1.0, 0.375 / 0.66, 0.15 / 0.66, 0.135 / 0.66, 0.0]),
    "com_cpad": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
}


@dataclass
class Coverage:
    """Coverage of each delivery strategy, as a proportion of births in its location."""
    fac_cc: float
    fac_ctc: float
    com_cc: float
    com_ctc: float
    com_cpad: float

    def total(self, p_fac: float) -> float:
        return (self.fac_cc + self.fac_ctc) * p_fac + (self.com_cc + self.com_ctc + self.com_cpad) * (1 - p_fac)


def protection(cov: Coverage, ve: np.ndarray, com_sba: float, ve_warm: float = 1.0,
               clamp: bool = False) -> tuple[float, float]:
    """Share of facility and community births protected by an effective birth dose.

    Ports the effectiveness-weighting in ``model_initializer``. ``com_sba`` is
    the proportion of community births attended by a professional health
    worker; CTC vaccines in the community are given on day 1 to those births
    and later to the rest. ``clamp`` mirrors the ``max(., 0)`` the authors
    apply in their uncertainty-analysis branch but not in their point run.
    """
    prot_fac = cov.fac_cc * float(TIMING["fac_cc"] @ ve) \
        + cov.fac_ctc * float(TIMING["fac_ctc"] @ (ve * ve_warm))
    prot_com = cov.com_cc * float(TIMING["com_cc"] @ ve) \
        + cov.com_cpad * float(TIMING["com_cpad"] @ (ve * ve_warm))
    later = float(TIMING["com_ctc"][1:] @ (ve[1:] * ve_warm))
    if cov.com_ctc == 0:
        pass
    elif com_sba > 0.3 * cov.com_cc:
        day1 = com_sba - 0.3 * cov.com_cc
        rest = cov.com_ctc - day1
        if clamp:
            rest = max(rest, 0.0)
        prot_com += day1 * ve[0] * ve_warm + rest * later
    else:
        prot_com += cov.com_ctc * 0.34 * ve[0] * ve_warm + 0.66 * cov.com_ctc * later
    return prot_fac, prot_com


def weighted_baseline_coverage(birth_dose: float, p_fac: float, weight: float = 2.0) -> tuple[float, float]:
    """Split reported birth-dose coverage into facility and community coverage.

    Facility births are ``weight`` times as likely to be vaccinated
    (model_setup.py ``facility_weight``).
    """
    bd = birth_dose
    k = (weight - 1) / (weight + 1)
    base = bd if bd <= 0.5 else (1 - bd)
    fac = bd + base * k * 2 * (1 - p_fac)
    com = bd - base * k * 2 * p_fac
    return fac, com


def saturation_levels(fac_cov: float, com_cov: float, com_sba: float) -> tuple[np.ndarray, np.ndarray]:
    """Maximum coverage of each strategy (Table 2, "optimisation parameters").

    Returns ``(saturation, sat_costing)`` as in ``coverage_saturation``; the
    costing version sits 5% above saturation so the logistic cost curves
    never divide by zero.
    """
    sat = np.zeros(5)
    sat[0] = max(0.80, fac_cov + (1 - fac_cov) * 0.0001)
    sat[1] = max(0.90, fac_cov + (1 - fac_cov) * 0.05)
    sat[2] = com_cov
    if com_sba > com_cov * 0.3:
        sat[3] = min(0.95, com_sba + 0.7 * com_cov + 0.1 * (1 - com_cov))
    else:
        sat[3] = min(0.95, com_cov + (1 - com_cov) * 0.1)
    sat[4] = 0.95
    if sat[2] > sat[3]:
        sat[3] = sat[2] + (1 - sat[2]) * 0.1
        sat[4] = sat[3]
    cost = sat.copy()
    cost[2:] = sat[2:] + (1 - sat[2:]) * 0.05
    return sat, cost


SCENARIOS = ["baseline", "scenario1", "scenario2", "scenario3", "scenario4"]


def scenario_coverage(scen: int, fac_cov: float, com_cov: float, sat: np.ndarray) -> Coverage:
    """Coverage by strategy for the baseline and the four scenarios (``model_scenarios``)."""
    if scen == 0:
        return Coverage(fac_cov, 0.0, com_cov, 0.0, 0.0)
    fac_ctc = sat[1] - sat[0] if scen >= 2 else 0.0
    com_ctc = sat[3] - sat[2] if scen >= 3 else 0.0
    com_cpad = sat[4] - sat[3] if scen >= 4 else 0.0
    return Coverage(sat[0], fac_ctc, sat[2], com_ctc, com_cpad)


def vaccine_costs(cov: Coverage, unit: np.ndarray, p_fac: float, sat: np.ndarray,
                  sat_costing: np.ndarray, births: float = PER) -> np.ndarray:
    """Vaccine expenditure per strategy and in total (``vaccine_costs``).

    Facility strategies cost linearly; community strategies follow the inverse
    of the logistic cost-coverage curve. ``unit`` is the per-dose cost of
    each strategy in the order of ``VACCINE_COST_COLS``.
    """
    hosp_n = p_fac * births
    comm_n = births - hosp_n
    c = np.zeros(6)
    c[0] = min(hosp_n * cov.fac_cc * unit[0], hosp_n * sat_costing[0] * unit[0])
    c[1] = min(hosp_n * cov.fac_ctc * unit[1], hosp_n * (sat_costing[1] - sat_costing[0]) * unit[1])

    def logistic_inverse(cap: float, x: float, price: float) -> float:
        return abs(-0.5 * comm_n * cap * price * math.log((cap * comm_n - comm_n * x) / (cap * comm_n + comm_n * x)))

    c[2] = logistic_inverse(sat_costing[2], cov.com_cc, unit[2])
    c[3] = logistic_inverse(sat_costing[3] - sat[2], cov.com_ctc, unit[3])
    c[4] = logistic_inverse(sat_costing[4] - sat[3], cov.com_cpad, unit[4])
    c[5] = c[:5].sum()
    return c


# ---------------------------------------------------------------------------
# Cost-coverage curves and optimisation (costcoverage.py, optimisation_model.py)
# ---------------------------------------------------------------------------

class CostCoverage:
    """Births reached per strategy as a function of expenditure, per 1000 births."""

    def __init__(self, unit: np.ndarray, sat: np.ndarray, p_fac: float, births: float = PER):
        self.unit = unit
        self.sat = sat
        self.hosp_n = p_fac * births
        self.comm_n = births - self.hosp_n

    def _logistic(self, cap: float, spend: float, price: float) -> float:
        if cap <= 0:
            return 0.0
        return max(2 * cap / (1 + math.exp(-2 * spend / (self.comm_n * cap * price))) - cap, 0.0)

    def proportions(self, x: np.ndarray) -> np.ndarray:
        """Proportion of births in each location covered by each strategy."""
        p = np.zeros(5)
        p[0] = x[0] / (self.unit[0] * self.hosp_n)
        p[1] = x[1] / (self.unit[1] * self.hosp_n)
        p[2] = self._logistic(self.sat[2], x[2], self.unit[2])
        p[3] = self._logistic(self.sat[3] - p[2], x[3], self.unit[3])
        if self.sat[3] == self.sat[4]:
            p[4] = 0.0
        else:
            p[4] = self._logistic(self.sat[4] - p[2] - p[3], x[4], self.unit[4])
        return p

    def births_reached(self, x: np.ndarray) -> float:
        p = self.proportions(x)
        return (p[0] + p[1]) * self.hosp_n + (p[2] + p[3] + p[4]) * self.comm_n

    def constraints(self):
        s = self.sat
        return [
            {"type": "ineq", "fun": lambda x: s[0] - self.proportions(x)[0]},
            {"type": "ineq", "fun": lambda x: (s[1] - s[0]) - self.proportions(x)[1]},
            {"type": "ineq", "fun": lambda x: s[3] - sum(self.proportions(x)[2:4])},
            {"type": "ineq", "fun": lambda x: s[4] - sum(self.proportions(x)[2:5])},
        ]

    def optimise_budget(self, budget: float) -> np.ndarray:
        """Reallocate a fixed budget across strategies to maximise births reached."""
        cons = [{"type": "eq", "fun": lambda x: budget - x.sum()}] + self.constraints()
        x0 = np.full(5, budget / 5)
        sol = minimize(lambda x: -self.births_reached(x), x0, method="SLSQP",
                       constraints=cons, bounds=[(0, None)] * 5)
        return sol.x

    def optimise_target(self, target: float = 0.9) -> np.ndarray:
        """Minimum expenditure reaching ``target`` coverage."""
        cons = [{"type": "eq", "fun": lambda x: target * PER - self.births_reached(x)}] + self.constraints()
        sol = minimize(lambda x: x.sum(), np.ones(5), method="SLSQP",
                       constraints=cons, bounds=[(0, None)] * 5)
        return sol.x

    def coverage(self, x: np.ndarray) -> Coverage:
        return Coverage(*self.proportions(x))


# ---------------------------------------------------------------------------
# Running camdl
# ---------------------------------------------------------------------------

QUANTITIES = ["infections", "chronic_cases", "hbv_deaths", "alive_at_end", "yll", "yld", "disease_cost"]


def write_params(params: dict[str, float], path: pathlib.Path) -> None:
    with open(path, "w") as fh:
        for key, value in params.items():
            fh.write(f"{key} = {float(value)!r}\n")


def simulate(params: dict[str, float], store: pathlib.Path | None = None) -> dict[str, float]:
    """One deterministic cohort run; returns the model's ``quantities``."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        ptoml = tmp / "p.toml"
        write_params(params, ptoml)
        cmd = [
            "camdl", "simulate", str(MODEL),
            "--params", str(ptoml),
            "--backend", "ode", "--integrator", "rk4",
            "--dt", str(DT), "--to", str(HORIZON),
            "--output-every", str(DT),
            "--no-progress", "--no-flows",
            "--quantities-out", str(tmp / "q"),
            "--output-dir", str(store or tmp / "store"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"camdl simulate failed:\n{res.stdout}\n{res.stderr}")
        out = {}
        for name in QUANTITIES:
            tsv = pd.read_csv(tmp / "q" / "quantities" / f"{name}.tsv", sep="\t", comment="#")
            out[name] = float(tsv["value"].iloc[0])
        return out


def cohort_params(inp: Inputs, row: pd.Series, prot_fac: float, prot_com: float,
                  glob: dict[str, float] | None = None, reg: dict[str, float] | None = None) -> dict[str, float]:
    params = dict(glob or global_point(inp))
    params.update(reg or regional_point(row))
    params.update(n_births=COHORT, prot_fac=prot_fac, prot_com=prot_com)
    return params


def outcomes_per_1000(q: dict[str, float]) -> dict[str, float]:
    return {k: v * PER / COHORT for k, v in q.items()}


def simulate_many(param_sets: list[dict[str, float]], workers: int = 4) -> list[dict[str, float]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(simulate, param_sets))


def dump_json(obj, path: pathlib.Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))
