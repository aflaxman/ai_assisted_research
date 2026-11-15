"""Run the simulation for many locations, scenarios, and input draws.

A cluster would hand this job to ``psimulate``, one simulation per location,
scenario, draw, and seed.  This is the same loop for one machine.  Each
simulation is the model specification in ``model_specifications/`` with a
handful of configuration overrides, run for its single one-year step.

Usage
-----
    python -m vivarium_potassium_salt.tools.run_simulation --all --jobs 4 \\
        --output results/simulation/draw_0.parquet
    python -m vivarium_potassium_salt.tools.run_simulation China India --draws 0 1 2 3
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd

from vivarium_potassium_salt.components import observers
from vivarium_potassium_salt.constants import metadata, paths
from vivarium_potassium_salt.data import gbd_extract

SCENARIO_NAMES = [metadata.BASELINE.name] + [s.name for s in metadata.SCENARIOS]


def build_simulation(location: str, scenario: str, draw: int = 0, seed: int = 0,
                     population_size: int | None = None,
                     configuration: dict | None = None):
    """Set up one simulation, with its simulants created, and return it before any step.

    The result is a ``vivarium.interactive.Simulation``, so its population and
    every pipeline can be inspected: ``simulation.get_population([...])``.

    Parameters
    ----------
    location
        A GBD location name with an artifact in ``artifacts/``.
    scenario
        ``baseline`` or the name of one of ``metadata.SCENARIOS``.
    draw
        Input draw; 0 is the mean of every input.
    seed
        Random seed.  Simulations that share a seed share their simulants.
    population_size
        Number of simulants, if not the model specification's.
    configuration
        Further overrides, keyed by component name, e.g.
        ``{"potassium_enriched_salt": {"kcl_fraction": 0.15}}``.
    """
    from loguru import logger
    from vivarium.interactive import Simulation

    logger.disable("vivarium")
    warnings.filterwarnings("ignore", category=FutureWarning, module="vivarium")
    overrides: dict = {
        "input_data": {
            "artifact_path": str(paths.artifact_path(location)),
            "input_draw_number": draw,
        },
        "randomness": {"random_seed": seed},
        "potassium_enriched_salt": {"scenario": scenario},
    }
    if population_size is not None:
        overrides["population"] = {"population_size": population_size}
    for component, settings in (configuration or {}).items():
        overrides.setdefault(component, {}).update(settings)
    return Simulation(str(paths.MODEL_SPEC), configuration=overrides, observe=True)


def simulate(location: str, scenario: str, draw: int = 0, seed: int = 0,
             population_size: int | None = None,
             configuration: dict | None = None) -> dict[str, pd.DataFrame]:
    """Run one simulation for its single year and return what its observers collected.

    The arguments are those of ``build_simulation``.
    """
    simulation = build_simulation(location, scenario, draw, seed, population_size, configuration)
    simulation.step()
    results = simulation.get_results()
    return {name: results[name] for name in (observers.BURDEN_RESULTS, observers.POLICY_RESULTS)}


def run_location(location: str, draws=(0,), scenarios=SCENARIO_NAMES, seed: int = 0,
                 population_size: int | None = None,
                 configuration: dict | None = None) -> dict[str, pd.DataFrame]:
    """Every scenario and draw for one location, stacked into one table per observer.

    The scenarios of a location share a seed, and so their simulants.  Different
    locations get different seeds (``seed`` plus the GBD location id), so that
    sampling error averages out when countries are added up instead of
    repeating in every one of them.
    """
    location_seed = seed + gbd_extract.location_id(location)
    tables: dict[str, list[pd.DataFrame]] = {}
    for draw in draws:
        for scenario in scenarios:
            results = simulate(
                location, scenario, draw, location_seed, population_size, configuration)
            for name, table in results.items():
                tables.setdefault(name, []).append(
                    table.assign(location=location, scenario=scenario, draw=draw)
                )
    return {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}


def _run_and_save(location: str, checkpoint_dir: Path | None, **kwargs) -> dict[str, pd.DataFrame]:
    """``run_location``, kept on disk so that an interrupted batch can pick up where it stopped."""
    if checkpoint_dir is None:
        return run_location(location, **kwargs)
    files = {
        name: checkpoint_dir / f"{paths.sanitize(location)}.{name}.parquet"
        for name in (observers.BURDEN_RESULTS, observers.POLICY_RESULTS)
    }
    if all(file.exists() for file in files.values()):
        return {name: pd.read_parquet(file) for name, file in files.items()}
    results = run_location(location, **kwargs)
    for name, file in files.items():
        results[name].to_parquet(file, index=False)
    return results


def run(locations: list[str], jobs: int = 1, checkpoint_dir: Path | None = None,
        **kwargs) -> dict[str, pd.DataFrame]:
    """``run_location`` for many locations, in parallel.

    With a ``checkpoint_dir``, each location's tables are written there as it
    finishes, and locations already there are read back and not run again.  The
    directory belongs to one set of draws and settings; start a new one when
    those change.
    """
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    worker = partial(_run_and_save, checkpoint_dir=checkpoint_dir, **kwargs)
    if jobs == 1:
        per_location = [worker(location) for location in locations]
    else:
        with ProcessPoolExecutor(jobs) as pool:
            per_location = list(pool.map(worker, locations))
    return {
        name: pd.concat([result[name] for result in per_location], ignore_index=True)
        for name in per_location[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("locations", nargs="*", help="GBD location names")
    parser.add_argument("--all", action="store_true", help="every location in the paper")
    parser.add_argument("--draws", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--association", choices=["independent", "ranked"], default=None,
                        help="how hypertension lines up with blood pressure")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=paths.RESULTS_DIR / "simulation" / "results.parquet",
                        help="where to write; '<stem>.<observer>.parquet' per observer")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="keep each finished location here and skip it on a rerun")
    arguments = parser.parse_args()

    locations = gbd_extract.paper_locations() if arguments.all else arguments.locations
    configuration = None
    if arguments.association:
        configuration = {
            "hypertension_status": {"association_with_blood_pressure": arguments.association}
        }
    results = run(
        locations, jobs=arguments.jobs, checkpoint_dir=arguments.checkpoint_dir,
        draws=arguments.draws, seed=arguments.seed,
        population_size=arguments.population_size, configuration=configuration,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        path = arguments.output.with_suffix(f".{name}.parquet")
        table.to_parquet(path, index=False)
        print(f"{path}: {len(table):,} rows")


if __name__ == "__main__":
    main()
