"""Run the model for one or more scenarios and seeds, saving raw observer output.

Usage::

    python -m sodium_sim.run --scenarios bau mandate_who --seeds 0 1 --population 100000

Each (scenario, seed) run writes ``results/raw/<scenario>/seed_<n>/<measure>.csv``
with the stratified counts from the vivarium observers: ``ylds``, ``deaths``,
``ylls`` and ``person_time``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from vivarium.engine import InteractiveContext

from sodium_sim.build_artifact import ARTIFACT_PATH, DATA, ROOT
from sodium_sim.scenarios import SCENARIOS

MODEL_SPEC = Path(__file__).with_name("model_spec.yaml")
RAW_RESULTS = ROOT / "results" / "raw"


def run_one(scenario: str, seed: int, population: int, artifact: Path = ARTIFACT_PATH,
            verbose: bool = False, step_days: float | None = None) -> dict[str, pd.DataFrame]:
    """Run a single simulation and return its results with one row per year.

    The model takes one step per year. After each step the cumulative observer
    results are snapshotted and differenced, and the increment is labelled with
    the calendar year the step covered (an ``event_year`` column). This keeps the
    observers free of a year stratification, which is slow in the results
    system, and avoids the uneven year binning of fractional-day steps.
    """
    config = {
        "input_data": {"artifact_path": str(artifact)},
        "randomness": {"random_seed": seed},
        "population": {"population_size": population},
        "sodium_intervention": {"scenario": scenario},
    }
    if step_days is not None:
        config["time"] = {"step_size": step_days}
    sim = InteractiveContext(
        MODEL_SPEC, configuration=config, observe=True,
        sim_name=f"{scenario}_{seed}_{time.time_ns()}", logging_verbosity=1 if verbose else 0,
    )
    start_year = sim.configuration.time.start.year
    end_year = sim.configuration.time.end.year
    steps_per_year = round(365.25 / sim.configuration.time.step_size)
    previous: dict[str, pd.Series] = {}
    yearly: dict[str, list[pd.DataFrame]] = {}
    # Label steps by index: the clock drifts a few hours per step, so its year is unreliable.
    for year in range(start_year, end_year + 1):
        t0 = time.time()
        for _ in range(steps_per_year):
            sim.step()
        for measure, df in sim.get_results().items():
            if not isinstance(df, pd.DataFrame):
                continue
            keys = [c for c in df.columns if c != "value"]
            cumulative = df.set_index(keys)["value"]
            increment = cumulative.sub(previous.get(measure, 0.0), fill_value=0.0)
            previous[measure] = cumulative
            out = increment.reset_index()
            out["event_year"] = year
            yearly.setdefault(measure, []).append(out)
        if verbose:
            print(f"  {scenario} seed {seed} year {year}: {time.time() - t0:.0f}s", flush=True)
    sim.finalize()
    return {m: pd.concat(parts, ignore_index=True) for m, parts in yearly.items()}


def real_population() -> float:
    """Australia's 2023 population from the GBD extract."""
    return float(pd.read_csv(DATA / "population_by_age.csv")["value_combined"].sum())


def save(results: dict[str, pd.DataFrame], scenario: str, seed: int, population: int,
         tag: str = "") -> Path:
    out = (RAW_RESULTS if not tag else ROOT / "results" / f"raw_{tag}") / scenario / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for measure, df in results.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{measure}.csv", index=False)
    meta = {"scenario": scenario, "seed": seed, "population_size": population,
            "scale": real_population() / population}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--population", type=int, default=100_000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH), help="artifact HDF to simulate from")
    parser.add_argument("--tag", default="", help="results go to results/raw_<tag>/ instead of results/raw/")
    parser.add_argument("--step-days", type=float, default=None,
                        help="time step in days (default from the model spec: 365.25); 91.3125 for quarterly")
    args = parser.parse_args()
    for seed in args.seeds:
        for scenario in args.scenarios:
            t0 = time.time()
            results = run_one(scenario, seed, args.population, artifact=Path(args.artifact),
                              verbose=args.verbose, step_days=args.step_days)
            out = save(results, scenario, seed, args.population, args.tag)
            print(f"{scenario} seed {seed}: {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
