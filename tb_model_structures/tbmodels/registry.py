"""The full set of structures under comparison."""
from __future__ import annotations

from .classic import all_classic
from .core import Structure
from .spectrum import all_spectrum


def all_structures() -> dict[str, Structure]:
    out: dict[str, Structure] = {}
    out.update(all_classic())
    out.update(all_spectrum())
    return out


def get(name: str) -> Structure:
    return all_structures()[name]
