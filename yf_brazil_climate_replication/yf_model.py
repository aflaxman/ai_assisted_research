"""Shared machinery for the Sadeghieh et al. (2021) yellow fever replication.

Every simulation goes through the camdl binary: this module only builds
parameter files, invokes `camdl simulate` on the ODE backend, and reduces the
resulting trajectory into the five outcomes the paper's Table 4 reports.
"""

from __future__ import annotations

import csv
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = ROOT / "yf_brazil.camdl"
DATA = ROOT / "data" / "yf_brazil_201718_cases.tsv"

# dt for the fixed-step RK4 integrator. 1/16 week; exact in binary, so every
# integer week boundary is hit exactly (camdl requires period boundaries to
# land on recorded output times).
DT = 0.0625
HORIZON = 60  # weeks; long enough for every scenario to burn out

# ---------------------------------------------------------------------------
# Published parameter values (Tables 1 and 2)
# ---------------------------------------------------------------------------

N_REF = 209469000.0   # total humans, UN WPP 2019
M_REF = 418938000.0   # total mosquitoes, assumed 2x humans

#: Table 1 as published, converted to camdl's per-capita form where needed.
PUBLISHED = {
    "beta1": 5.22e-7 * M_REF,     # 218.685636 /week
    "beta2": 1.00e-12 * N_REF,    # 2.09469e-4 /week
    "phi": 0.45,
    "omega": 2.38,
    "lambda1": 1.31,
    "lambda2": 0.54,
    "gamma": 0.82,
    "mu": 0.36,
    "delta": 0.58,
    "immune": 0.80,
    "y0": 150.0,
    "z0": 995.0,
    "n_ref": N_REF,
    "m_ref": M_REF,
}

#: omega implied by the paper's OWN Table 4 output (426 deaths / 1254 cases).
#: Table 1's 2.38/week implies a 74% case-fatality ratio instead of 34%.
OMEGA_CFR = 0.42237


# ---------------------------------------------------------------------------
# Table 3 — temperature-dependent mosquito parameters
# ---------------------------------------------------------------------------

def eip_rate(temp_c: float) -> float:
    """lambda2, rate of viraemic onset in mosquitoes, per week."""
    return 1.0 / (96.69 - 2.8 * temp_c) * 7.0


def death_rate(temp_c: float) -> float:
    """mu, mosquito death rate, per week."""
    return 4.76 - 0.38 * temp_c + 0.0081 * temp_c ** 2


def development_rate(temp_c: float) -> float:
    """delta, mosquito development rate, per week."""
    return -0.32 + 0.03 * temp_c


def temperature_params(temp_c: float) -> dict[str, float]:
    return {
        "lambda2": eip_rate(temp_c),
        "mu": death_rate(temp_c),
        "delta": development_rate(temp_c),
    }


# ---------------------------------------------------------------------------
# Running camdl
# ---------------------------------------------------------------------------

def _write_params(params: dict[str, float], path: pathlib.Path) -> None:
    with open(path, "w") as fh:
        for key, value in params.items():
            fh.write(f"{key} = {value!r}\n")


def simulate(params: dict[str, float], horizon: int = HORIZON) -> dict[str, list[float]]:
    """Run one deterministic ODE simulation and return weekly series.

    Returns a dict with ``weeks`` (1-based), ``cases`` (weekly symptomatic
    incidence, the E->I flow accumulated over each week) and ``deaths``
    (cumulative YF deaths at the end of each week).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        ptoml, out = tmp / "p.toml", tmp / "traj.tsv"
        _write_params(params, ptoml)
        cmd = [
            "camdl", "simulate", str(MODEL),
            "--params", str(ptoml),
            "--backend", "ode", "--integrator", "rk4",
            "--dt", str(DT),
            "--to", str(horizon),
            "--output-every", "1",
            "--no-progress",
            "-o", str(out),
            "--output-dir", str(tmp / "store"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"camdl simulate failed:\n{res.stdout}\n{res.stderr}")
        rows = [r for r in csv.DictReader(
            (ln for ln in open(out) if not ln.startswith("#")), delimiter="\t")]

    # With --output-every 1 on a weekly model, `flow_<name>` is the flow
    # accumulated over the interval ENDING at that row's t (not a running
    # total), so row t = k is already the incidence of week k. Row t = 0 is
    # the zero-width interval at the start and is dropped.
    weekly = [float(r["flow_onset_sym"]) for r in rows[1:]]
    cum_dead = [float(r["Dead"]) for r in rows]
    return {
        "weeks": list(range(1, len(weekly) + 1)),
        "cases": weekly,
        "deaths": cum_dead[1:],
    }


# ---------------------------------------------------------------------------
# Table 4 outcome definitions
# ---------------------------------------------------------------------------

def outcomes(series: dict[str, list[float]], threshold: float = 1.0) -> dict[str, float]:
    """The five outcomes Table 4 reports.

    ``duration`` is the last week whose expected incidence is at least
    ``threshold`` cases. The paper never states its own definition; a 1-case
    floor reproduces its reported 21 weeks for the fitted baseline.
    """
    cases = series["cases"]
    peak = max(cases)
    peak_week = cases.index(peak) + 1
    above = [w for w, c in zip(series["weeks"], cases) if c >= threshold]
    return {
        "peak_incidence": peak,
        "peak_week": peak_week,
        "cumulative_incidence": sum(cases),
        "deaths": series["deaths"][-1],
        "duration": above[-1] if above else 0,
    }


def observed() -> dict[str, list[float]]:
    rows = list(csv.DictReader(open(DATA), delimiter="\t"))
    weeks = [int(r["time"]) for r in rows]
    cases = [float(r["cases"]) for r in rows]
    return {"weeks": weeks, "cases": cases, "deaths": [428.0] * len(weeks)}


def r0(params: dict[str, float]) -> float:
    """Vector-host basic reproduction number at the mosquito quasi-equilibrium.

    One infectious mosquito infects beta1*S/m_ref humans per week and lives
    1/mu weeks. One infected human infects beta2*X/n_ref mosquitoes per week
    for phi/(gamma+omega) + (1-phi)/gamma weeks, and a fraction
    lambda2/(lambda2+mu) of those survive the extrinsic incubation period.
    R0 is the geometric mean of the two half-cycles.

    X is taken at delta*m_ref/mu, the level the open mosquito population
    relaxes to - not m_ref, which is where it starts.
    """
    p = params
    s0 = p["n_ref"] * (1.0 - p["immune"])
    x_eq = p["delta"] * p["m_ref"] / p["mu"]
    human_per_mosq = p["beta1"] * s0 / p["m_ref"] / p["mu"]
    infectious_weeks = (p["phi"] / (p["gamma"] + p["omega"])
                        + (1.0 - p["phi"]) / p["gamma"])
    mosq_per_human = (p["beta2"] * x_eq / p["n_ref"] * infectious_weeks
                      * p["lambda2"] / (p["lambda2"] + p["mu"]))
    return (human_per_mosq * mosq_per_human) ** 0.5
