"""
CSV I/O utilities for SOURCE economic input layer.

This module provides functions to load and validate input data from CSV files,
converting them to TimeSeries objects for use in the economic calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .econ import TimeSeries


def load_heroin_price_csv(
    filepath: Union[str, Path],
    year_col: str = "year",
    price_col: str = "heroin_price_usd",
    validate: bool = True,
) -> TimeSeries:
    """
    Load heroin street price data from a CSV file.

    Expected CSV format:
        year,heroin_price_usd
        2010,400.0
        2011,375.0
        ...

    Args:
        filepath: Path to the CSV file.
        year_col: Name of the year column.
        price_col: Name of the price column.
        validate: If True, check for positive prices and valid years.

    Returns:
        TimeSeries of heroin prices by year.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Heroin price file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Check required columns
    for col in [year_col, price_col]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {filepath}. "
                f"Available columns: {list(df.columns)}"
            )

    # Convert to dictionary
    data: Dict[int, float] = {}
    for _, row in df.iterrows():
        year = int(row[year_col])
        price = float(row[price_col])

        if validate:
            if price <= 0:
                raise ValueError(
                    f"Invalid heroin price {price} for year {year}. "
                    "Price must be positive."
                )
            if year < 1900 or year > 2100:
                raise ValueError(f"Suspicious year value: {year}")

        data[year] = price

    return TimeSeries(data, name="heroin_price")


def load_rx_inputs_csv(
    filepath: Union[str, Path],
    year_col: str = "year",
    prescriptions_col: str = "prescriptions",
    patients_col: str = "patients",
    mme_col: str = "mme",
    adf_share_col: str = "adf_share",
    validate: bool = True,
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
    """
    Load prescription opioid input data from a CSV file.

    Expected CSV format:
        year,prescriptions,patients,mme,adf_share
        2010,250.0,45.0,100.0,0.05
        2011,255.0,46.0,105.0,0.08
        ...

    Units (suggested):
        - prescriptions: millions of annual prescriptions
        - patients: millions of patients receiving Rx opioids
        - mme: billions of morphine milligram equivalents
        - adf_share: fraction of prescriptions that are ADF (0-1)

    Args:
        filepath: Path to the CSV file.
        year_col: Name of the year column.
        prescriptions_col: Name of the prescriptions column.
        patients_col: Name of the patients column.
        mme_col: Name of the MME column.
        adf_share_col: Name of the ADF share column.
        validate: If True, check for valid ranges.

    Returns:
        Tuple of (rx_prescriptions, rx_patients, rx_mme, adf_share) TimeSeries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Rx inputs file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Check required columns
    required_cols = [
        year_col,
        prescriptions_col,
        patients_col,
        mme_col,
        adf_share_col,
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {filepath}. "
                f"Available columns: {list(df.columns)}"
            )

    prescriptions: Dict[int, float] = {}
    patients: Dict[int, float] = {}
    mme: Dict[int, float] = {}
    adf_share: Dict[int, float] = {}

    for _, row in df.iterrows():
        year = int(row[year_col])

        presc_val = float(row[prescriptions_col])
        pat_val = float(row[patients_col])
        mme_val = float(row[mme_col])
        adf_val = float(row[adf_share_col])

        if validate:
            if year < 1900 or year > 2100:
                raise ValueError(f"Suspicious year value: {year}")
            if presc_val < 0:
                raise ValueError(
                    f"Prescriptions must be non-negative, got {presc_val} "
                    f"in year {year}"
                )
            if pat_val < 0:
                raise ValueError(
                    f"Patients must be non-negative, got {pat_val} "
                    f"in year {year}"
                )
            if mme_val < 0:
                raise ValueError(
                    f"MME must be non-negative, got {mme_val} in year {year}"
                )
            if not 0 <= adf_val <= 1:
                raise ValueError(
                    f"ADF share must be in [0, 1], got {adf_val} in year {year}"
                )

        prescriptions[year] = presc_val
        patients[year] = pat_val
        mme[year] = mme_val
        adf_share[year] = adf_val

    return (
        TimeSeries(prescriptions, name="rx_prescriptions"),
        TimeSeries(patients, name="rx_patients"),
        TimeSeries(mme, name="rx_mme"),
        TimeSeries(adf_share, name="adf_share"),
    )


def load_rx_price_csv(
    filepath: Union[str, Path],
    year_col: str = "year",
    price_col: str = "rx_price_usd",
    validate: bool = True,
) -> TimeSeries:
    """
    Load prescription opioid street price data from a CSV file.

    This is optional data; the model can run without Rx price inputs.

    Expected CSV format:
        year,rx_price_usd
        2010,1.50
        2011,1.75
        ...

    Args:
        filepath: Path to the CSV file.
        year_col: Name of the year column.
        price_col: Name of the price column.
        validate: If True, check for positive prices.

    Returns:
        TimeSeries of Rx opioid street prices by year.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Rx price file not found: {filepath}")

    df = pd.read_csv(filepath)

    for col in [year_col, price_col]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {filepath}. "
                f"Available columns: {list(df.columns)}"
            )

    data: Dict[int, float] = {}
    for _, row in df.iterrows():
        year = int(row[year_col])
        price = float(row[price_col])

        if validate:
            if price <= 0:
                raise ValueError(
                    f"Invalid Rx price {price} for year {year}. "
                    "Price must be positive."
                )
            if year < 1900 or year > 2100:
                raise ValueError(f"Suspicious year value: {year}")

        data[year] = price

    return TimeSeries(data, name="rx_price")


def save_results_csv(
    filepath: Union[str, Path],
    years: List[int],
    columns: Dict[str, List[float]],
) -> None:
    """
    Save results to a CSV file.

    Args:
        filepath: Output file path.
        years: List of years.
        columns: Dictionary mapping column names to lists of values.

    Raises:
        ValueError: If column lengths don't match years.
    """
    for name, values in columns.items():
        if len(values) != len(years):
            raise ValueError(
                f"Column '{name}' has {len(values)} values but "
                f"there are {len(years)} years"
            )

    df = pd.DataFrame({"year": years, **columns})
    df.to_csv(filepath, index=False)


def validate_year_alignment(
    *time_series: TimeSeries,
    names: Optional[List[str]] = None,
) -> List[int]:
    """
    Validate that multiple TimeSeries cover the same years.

    Args:
        *time_series: Variable number of TimeSeries to check.
        names: Optional names for error messages.

    Returns:
        Sorted list of common years.

    Raises:
        ValueError: If year sets differ between series.
    """
    if len(time_series) < 2:
        if time_series:
            return time_series[0].years
        return []

    ref_years = set(time_series[0].years)
    ref_name = names[0] if names else "series_0"

    for i, ts in enumerate(time_series[1:], start=1):
        ts_name = names[i] if names and len(names) > i else f"series_{i}"
        if set(ts.years) != ref_years:
            raise ValueError(
                f"Year mismatch: {ref_name} has years {sorted(ref_years)}, "
                f"but {ts_name} has years {ts.years}"
            )

    return sorted(ref_years)
