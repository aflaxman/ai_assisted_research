"""Viral loads from ``within_host.camdl`` for the authors' 10,000 parameter sets.

The authors published 10,000 draws of the within-host parameters with the
companion eLife paper (``data/SimulatedParameters.txt``); their R code picks a
row at random for each simulated transmitter.  This module runs every row
through camdl once, as a ``--sweep-file`` design on the ODE backend, and
caches the daily viral loads.  Antiviral scenarios add ``t_treat`` and
``eps_treat`` columns to the design.

Day ``j`` of a trajectory is ``TAU + j`` days after exposure, ``j = 0..20``.
The authors set viral load to zero after day 20.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = ROOT / "within_host.camdl"
AUTHORS_PARAMS = ROOT / "data" / "SimulatedParameters.txt"
CACHE = ROOT / "results" / "cache"

TAU = 0.5
DAYS = 20
#: Days since exposure at which viral load is read.
TIMES = TAU + np.arange(DAYS + 1.0)

#: authors' column -> camdl parameter
RENAME = {"beta": "beta", "delta": "delta", "k": "k", "m": "m", "w": "omega", "p": "prod",
          "de": "delta_E", "E50": "phi", "r": "r", "q": "q", "c": "c"}
SWEPT = ["beta", "delta", "k", "m", "omega", "prod"]
FIXED = ["delta_E", "phi", "r", "q", "c"]
#: Placeholder values for the treatment parameters in untreated runs, where
#: the `start_treatment` intervention is not enabled and they are not read.
NO_TREATMENT = {"eps_treat": 0.0, "t_treat": 1.0e6}


def authors_parameters() -> pd.DataFrame:
    """The 10,000 published parameter sets, under camdl's parameter names."""
    raw = pd.read_csv(AUTHORS_PARAMS)
    par = raw.rename(columns=RENAME)[SWEPT + FIXED]
    for name in FIXED:
        assert par[name].nunique() == 1, f"{name} varies in the authors' file"
    return par


def run_camdl(design: pd.DataFrame, fixed: dict[str, float], threads: int = 0,
              treat: bool = False) -> np.ndarray:
    """Run one camdl sweep; returns viral load, design points x days.

    ``treat`` enables the model's ``start_treatment`` intervention, which
    switches the antiviral on at each design point's ``t_treat``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        with open(tmp / "fixed.toml", "w") as fh:
            for key, value in fixed.items():
                fh.write(f"{key} = {float(value)!r}\n")
        design.to_csv(tmp / "design.tsv", sep="\t", index=False, float_format="%.17g")
        cmd = [
            "camdl", "simulate", str(MODEL),
            "--params", str(tmp / "fixed.toml"),
            "--sweep-file", str(tmp / "design.tsv"),
            "--backend", "ode", "--columns", "V", "--no-progress",
            "--parallel", str(threads),
            "--store", str(tmp / "store"),
            "-o", str(tmp / "out.tsv"),
        ]
        if treat:
            cmd += ["--enable", "start_treatment"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"camdl simulate failed:\n{res.stdout}\n{res.stderr}")
        out = pd.read_csv(tmp / "out.tsv", sep="\t", comment="#", usecols=["point_id", "t", "V"])
    wide = out.pivot(index="point_id", columns="t", values="V").sort_index()
    assert wide.shape == (len(design), len(TIMES)), wide.shape
    assert np.allclose(wide.columns.to_numpy(), TIMES)
    return wide.to_numpy()


def _cached(name: str, compute) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    values = compute()
    np.save(path, values)
    return values


def untreated(chunk: int = 2500) -> np.ndarray:
    """Viral load without treatment, 10,000 parameter sets x 21 days."""
    def compute():
        par = authors_parameters()
        fixed = {**{name: par[name].iloc[0] for name in FIXED}, **NO_TREATMENT}
        parts = [run_camdl(par[SWEPT].iloc[i:i + chunk], fixed) for i in range(0, len(par), chunk)]
        return np.vstack(parts)
    return _cached("viral_load_untreated", compute)


def treated(name: str, t_treat: np.ndarray, eps_treat: float, chunk: int = 2500) -> np.ndarray:
    """Viral load when parameter set ``i`` starts an antiviral of efficacy
    ``eps_treat`` at ``t_treat[i]`` days after exposure."""
    def compute():
        par = authors_parameters()
        fixed = {**{n: par[n].iloc[0] for n in FIXED}, "eps_treat": eps_treat}
        design = par[SWEPT].copy()
        # A scheduled effect before the run starts does not fire, so treatment
        # begun before the first infected cell appears begins with it.
        design["t_treat"] = np.maximum(t_treat, TAU)
        parts = [run_camdl(design.iloc[i:i + chunk], fixed, treat=True)
                 for i in range(0, len(design), chunk)]
        return np.vstack(parts)
    return _cached(name, compute)


def clear_cache() -> None:
    shutil.rmtree(CACHE, ignore_errors=True)


if __name__ == "__main__":
    import time
    start = time.time()
    V = untreated()
    print(f"{V.shape[0]} trajectories x {V.shape[1]} days in {time.time() - start:.0f} s")
    peak = np.log10(V.max(axis=1))
    print(f"peak log10 viral load: median {np.median(peak):.2f}, "
          f"2.5-97.5% {np.percentile(peak, 2.5):.2f}-{np.percentile(peak, 97.5):.2f}")
