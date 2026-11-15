"""Build the data artifact for a location.

An artifact is one HDF file holding every input the simulation of a location
needs, under the keys in ``constants/data_keys.py``.

Usage
-----
    python -m vivarium_potassium_salt.data.builder China India
    python -m vivarium_potassium_salt.data.builder --all
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from vivarium.artifact import Artifact

from vivarium_potassium_salt.constants import data_keys, metadata, paths
from vivarium_potassium_salt.data import gbd_extract, loader

STANDARD_KEYS = [
    value
    for group in (data_keys.Population, data_keys.SystolicBloodPressure, data_keys.Sodium,
                  data_keys.Hypertension, data_keys.KidneyDysfunction, data_keys.Covariates)
    for name, value in vars(group).items()
    if name.isupper() and name != "ENTITY"
]


def burden_keys() -> list[str]:
    """Keys for the burden measures the cached extract can supply."""
    return [
        data_keys.burden_rate(cause, measure)
        for measure in gbd_extract.available_measures()
        for cause in metadata.CAUSES_BY_MEASURE[measure]
    ]


def build_artifact(location: str, output_dir: Path = paths.ARTIFACT_DIR) -> Path:
    """Write the artifact for ``location`` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{paths.sanitize(location)}.hdf"
    if path.exists():
        path.unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "no artifact found, building new"
        artifact = Artifact(path)
    for key in STANDARD_KEYS + burden_keys():
        artifact.write(key, loader.get_data(key, location))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("locations", nargs="*", help="GBD location names")
    parser.add_argument("--all", action="store_true", help="every location in the paper")
    parser.add_argument("--output-dir", type=Path, default=paths.ARTIFACT_DIR)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()
    locations = gbd_extract.paper_locations() if arguments.all else arguments.locations
    output_dirs = [arguments.output_dir] * len(locations)
    if arguments.jobs == 1:
        built = map(build_artifact, locations, output_dirs)
    else:
        built = ProcessPoolExecutor(arguments.jobs).map(build_artifact, locations, output_dirs)
    for path in built:
        print(path)


if __name__ == "__main__":
    main()
