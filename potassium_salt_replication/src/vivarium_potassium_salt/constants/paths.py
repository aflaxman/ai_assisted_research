"""Where the model keeps its inputs and outputs."""
from __future__ import annotations

import os
from pathlib import Path

# src/vivarium_potassium_salt/constants/paths.py -> project root
PROJECT_ROOT = Path(
    os.environ.get("POTASSIUM_SALT_ROOT", Path(__file__).resolve().parents[3])
)

DATA_DIR = PROJECT_ROOT / "data"
GBD_EXTRACT_DIR = DATA_DIR / "gbd_2023"
LITERATURE_DIR = DATA_DIR / "literature"
NCDRISC_DIR = DATA_DIR / "ncdrisc"
WORLD_BANK_DIR = DATA_DIR / "worldbank"
LOCATION_DIR = DATA_DIR / "locations"
PAPER_DIR = DATA_DIR / "paper"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_SPEC_DIR = Path(__file__).resolve().parents[1] / "model_specifications"
MODEL_SPEC = MODEL_SPEC_DIR / "model_spec.yaml"


def artifact_path(location: str) -> Path:
    """Path of the artifact for one location, e.g. artifacts/viet_nam.hdf."""
    return ARTIFACT_DIR / f"{sanitize(location)}.hdf"


def sanitize(location: str) -> str:
    """'Côte d'Ivoire' -> 'cote_d_ivoire': a safe file stem for a location name."""
    import unicodedata

    text = unicodedata.normalize("NFKD", location).encode("ascii", "ignore").decode()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    return "_".join(part for part in cleaned.split("_") if part)
