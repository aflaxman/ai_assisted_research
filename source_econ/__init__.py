"""
SOURCE Economic Input Layer - Minimal Python Replication

A minimal, readable Python implementation of the economic/drug-market input
layer from the FDA SOURCE (Simulation of Opioid Use, Response, Consequences,
and Effects) opioid systems model.

This package implements:
  - Time series handling with interpolation and normalization
  - Heroin street price and derived availability index
  - Prescription opioid availability index (composite)
  - Relative attractiveness indices for Rx vs heroin

Usage:
    from source_econ import (
        TimeSeries,
        heroin_availability,
        rx_availability_index,
        relative_attractiveness,
        load_heroin_price_csv,
        load_rx_inputs_csv,
    )

Note:
    This is a minimal replication for educational/research purposes.
    It does NOT implement the full opioid compartment model—only the
    economic input layer that could feed into such a simulation.
"""

from .econ import (
    TimeSeries,
    RxAvailabilityWeights,
    heroin_availability,
    rx_availability_index,
    relative_attractiveness,
)

from .io import (
    load_heroin_price_csv,
    load_rx_inputs_csv,
    load_rx_price_csv,
    save_results_csv,
    validate_year_alignment,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "TimeSeries",
    "RxAvailabilityWeights",
    # Calculation functions
    "heroin_availability",
    "rx_availability_index",
    "relative_attractiveness",
    # I/O functions
    "load_heroin_price_csv",
    "load_rx_inputs_csv",
    "load_rx_price_csv",
    "save_results_csv",
    "validate_year_alignment",
]
