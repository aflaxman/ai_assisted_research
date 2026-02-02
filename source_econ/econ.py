"""
Core economic input layer for FDA SOURCE opioid model replication.

This module implements the economic/drug-market drivers as time series inputs
and derived indices. It provides a minimal, readable replication of:
  - Heroin street price and availability index
  - Prescription opioid availability index (composite of multiple components)
  - Relative attractiveness indices for Rx vs heroin

Documentation notes:
  - Published SOURCE documentation describes heroin street price sourced from
    UNODC datasets (World Drug Report).
  - Prescription opioid availability is estimated from IQVIA audits (NPA/TPT/NSP)
    including annual prescriptions, patients with Rx opioids, total MME, and
    abuse-deterrent formulation (ADF) share.
  - In this minimal replication, we treat Rx components as user-supplied inputs
    since IQVIA raw data is proprietary.

References:
  - FDA SOURCE Model: Simulation of Opioid Use, Response, Consequences, Effects
  - Pitt AL, et al. (2018) "Modeling Health Benefits and Harms of Public Policy
    Responses to the US Opioid Epidemic"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TimeSeries:
    """
    A time series of annual values indexed by year.

    Attributes:
        data: Dictionary mapping year (int) to value (float).
        name: Optional descriptive name for the series.

    Examples:
        >>> ts = TimeSeries({2010: 100.0, 2011: 110.0, 2012: 105.0})
        >>> ts[2011]
        110.0
        >>> ts.years
        [2010, 2011, 2012]
    """

    data: Dict[int, float]
    name: str = ""

    def __post_init__(self) -> None:
        """Validate that data is non-empty and years are integers."""
        if not self.data:
            raise ValueError("TimeSeries data cannot be empty")
        for year in self.data:
            if not isinstance(year, (int, np.integer)):
                raise TypeError(f"Year must be int, got {type(year)}: {year}")

    def __getitem__(self, year: int) -> float:
        """Get value for a specific year."""
        return self.data[year]

    def __contains__(self, year: int) -> bool:
        """Check if year is in the series."""
        return year in self.data

    @property
    def years(self) -> List[int]:
        """Return sorted list of years."""
        return sorted(self.data.keys())

    @property
    def values(self) -> List[float]:
        """Return values in year-sorted order."""
        return [self.data[y] for y in self.years]

    @property
    def min_year(self) -> int:
        """Return the earliest year."""
        return min(self.data.keys())

    @property
    def max_year(self) -> int:
        """Return the latest year."""
        return max(self.data.keys())

    def is_monotonic_years(self) -> bool:
        """Check if years form a contiguous sequence (no gaps)."""
        years = self.years
        if len(years) < 2:
            return True
        return years[-1] - years[0] == len(years) - 1

    def check_monotonic_years(self) -> None:
        """Raise ValueError if years are not contiguous."""
        if not self.is_monotonic_years():
            missing = set(range(self.min_year, self.max_year + 1)) - set(
                self.years
            )
            raise ValueError(f"Non-contiguous years. Missing: {sorted(missing)}")

    def interpolate_to_annual(self) -> "TimeSeries":
        """
        Return a new TimeSeries with linear interpolation for missing years.

        Fills gaps between min_year and max_year using linear interpolation.

        Returns:
            New TimeSeries with contiguous annual values.
        """
        if self.is_monotonic_years():
            return TimeSeries(self.data.copy(), name=self.name)

        years = np.array(self.years)
        vals = np.array(self.values)
        all_years = np.arange(self.min_year, self.max_year + 1)
        interp_vals = np.interp(all_years, years, vals)

        return TimeSeries(
            {int(y): float(v) for y, v in zip(all_years, interp_vals)},
            name=self.name,
        )

    def normalize(
        self, method: str = "first", ref_value: Optional[float] = None
    ) -> "TimeSeries":
        """
        Normalize the series to a reference value.

        Args:
            method: 'first' (default) normalizes to first year = 1.0,
                    'mean' normalizes to mean = 1.0,
                    'max' normalizes to max = 1.0,
                    'ref' uses ref_value as reference.
            ref_value: Reference value when method='ref'.

        Returns:
            New TimeSeries with normalized values.
        """
        vals = np.array(self.values)

        if method == "first":
            ref = vals[0]
        elif method == "mean":
            ref = np.mean(vals)
        elif method == "max":
            ref = np.max(vals)
        elif method == "ref":
            if ref_value is None:
                raise ValueError("ref_value required when method='ref'")
            ref = ref_value
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if ref == 0:
            raise ValueError("Cannot normalize: reference value is zero")

        normalized = vals / ref
        return TimeSeries(
            {y: float(v) for y, v in zip(self.years, normalized)},
            name=f"{self.name}_normalized",
        )

    def rescale(self, factor: float) -> "TimeSeries":
        """Return a new TimeSeries with all values multiplied by factor."""
        return TimeSeries(
            {y: v * factor for y, v in self.data.items()},
            name=self.name,
        )

    def to_dict(self) -> Dict[int, float]:
        """Return a copy of the underlying data dictionary."""
        return self.data.copy()


def heroin_availability(
    price_ts: TimeSeries,
    elasticity: float = 1.0,
    price_ref: Optional[float] = None,
) -> TimeSeries:
    """
    Calculate heroin availability index from street price.

    Availability is modeled as inversely proportional to price:
        availability_year = (price_ref / price_year) ** elasticity

    This serves as a proxy for supply-side availability—when prices drop,
    it indicates increased supply/availability in the market.

    Args:
        price_ts: TimeSeries of heroin street prices (e.g., USD per gram).
        elasticity: Price elasticity parameter. Default 1.0 gives simple
            inverse relationship. Values > 1 amplify price effects.
        price_ref: Reference price for normalization. Defaults to first
            year's price, making first year availability = 1.0.

    Returns:
        TimeSeries of availability index values.

    Raises:
        ValueError: If any price is zero or negative.

    Examples:
        >>> prices = TimeSeries({2010: 400, 2011: 350, 2012: 300})
        >>> avail = heroin_availability(prices)
        >>> avail[2010]  # Reference year = 1.0
        1.0
        >>> avail[2012] > avail[2010]  # Lower price -> higher availability
        True
    """
    for year, price in price_ts.data.items():
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price} in {year}")

    if price_ref is None:
        price_ref = price_ts[price_ts.min_year]

    availability = {}
    for year in price_ts.years:
        price = price_ts[year]
        availability[year] = (price_ref / price) ** elasticity

    return TimeSeries(availability, name="heroin_availability")


@dataclass
class RxAvailabilityWeights:
    """
    Weights for combining Rx availability components.

    Default weights based on SOURCE model documentation:
    - prescriptions: 0.3 (number of annual prescriptions)
    - patients: 0.3 (number of patients receiving Rx opioids)
    - mme: 0.3 (total morphine milligram equivalents dispensed)
    - adf_share: 0.1 (share of abuse-deterrent formulations; higher = less
        divertible supply, so this enters negatively)
    """

    prescriptions: float = 0.3
    patients: float = 0.3
    mme: float = 0.3
    adf_share: float = 0.1

    def __post_init__(self) -> None:
        """Validate weights are non-negative."""
        for name in ["prescriptions", "patients", "mme", "adf_share"]:
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"Weight {name} must be non-negative: {val}")

    def total(self) -> float:
        """Return sum of all weights."""
        return (
            self.prescriptions + self.patients + self.mme + self.adf_share
        )

    def normalized(self) -> "RxAvailabilityWeights":
        """Return weights normalized to sum to 1.0."""
        total = self.total()
        if total == 0:
            raise ValueError("Cannot normalize: all weights are zero")
        return RxAvailabilityWeights(
            prescriptions=self.prescriptions / total,
            patients=self.patients / total,
            mme=self.mme / total,
            adf_share=self.adf_share / total,
        )


def rx_availability_index(
    rx_prescriptions: TimeSeries,
    rx_patients: TimeSeries,
    rx_mme: TimeSeries,
    adf_share: TimeSeries,
    weights: Optional[RxAvailabilityWeights] = None,
    normalize: bool = True,
) -> TimeSeries:
    """
    Calculate prescription opioid availability index from multiple components.

    Combines four input series into a single availability index:
    - rx_prescriptions: Number of annual opioid prescriptions (millions)
    - rx_patients: Number of patients receiving Rx opioids (millions)
    - rx_mme: Total morphine milligram equivalents dispensed (billions)
    - adf_share: Share of prescriptions that are abuse-deterrent (0-1)

    ADF share acts as a multiplicative discount: higher ADF penetration
    reduces divertible supply. The formula is:
        index = weighted_sum(prescriptions, patients, mme) * (1 - adf_share)

    This ensures the index stays non-negative and reflects that ADF products
    are harder to abuse/divert.

    Args:
        rx_prescriptions: TimeSeries of annual prescription counts.
        rx_patients: TimeSeries of patients receiving Rx opioids.
        rx_mme: TimeSeries of total MME dispensed.
        adf_share: TimeSeries of ADF share (fraction, 0-1).
        weights: RxAvailabilityWeights for combining components.
            Defaults to (0.3, 0.3, 0.3, 0.1). The adf_share weight controls
            how strongly ADF affects the discount (0.1 means 10% weight).
        normalize: If True, normalize each component before combining.

    Returns:
        TimeSeries of combined availability index.

    Raises:
        ValueError: If time series have mismatched years or invalid values.
    """
    if weights is None:
        weights = RxAvailabilityWeights()

    # Verify all series cover the same years
    years_set = set(rx_prescriptions.years)
    for ts, name in [
        (rx_patients, "rx_patients"),
        (rx_mme, "rx_mme"),
        (adf_share, "adf_share"),
    ]:
        if set(ts.years) != years_set:
            raise ValueError(
                f"Year mismatch: rx_prescriptions has {sorted(years_set)}, "
                f"{name} has {ts.years}"
            )

    # Validate ADF share is in [0, 1]
    for year in adf_share.years:
        val = adf_share[year]
        if not 0 <= val <= 1:
            raise ValueError(f"adf_share must be in [0,1], got {val} in {year}")

    # Normalize each component if requested (except ADF which is a fraction)
    if normalize:
        norm_prescriptions = rx_prescriptions.normalize(method="first")
        norm_patients = rx_patients.normalize(method="first")
        norm_mme = rx_mme.normalize(method="first")
    else:
        norm_prescriptions = rx_prescriptions
        norm_patients = rx_patients
        norm_mme = rx_mme

    # Combine with weights
    # First, compute weighted sum of positive components (normalized to sum=1)
    w = weights
    pos_total = w.prescriptions + w.patients + w.mme
    if pos_total == 0:
        raise ValueError("At least one positive weight must be non-zero")

    combined = {}
    for year in rx_prescriptions.years:
        # Weighted average of positive contributors
        positive = (
            w.prescriptions * norm_prescriptions[year]
            + w.patients * norm_patients[year]
            + w.mme * norm_mme[year]
        ) / pos_total

        # ADF acts as multiplicative discount on divertible supply
        # adf_weight controls the strength of the discount
        # discount = 1 - (adf_weight * adf_share)
        # With default adf_weight=0.1 and adf_share=0.5, discount = 0.95
        adf_discount = 1.0 - (w.adf_share * adf_share[year])
        combined[year] = positive * max(0.0, adf_discount)

    return TimeSeries(combined, name="rx_availability_index")


def relative_attractiveness(
    rx_avail: TimeSeries,
    heroin_avail: TimeSeries,
    rx_price: Optional[TimeSeries] = None,
    heroin_price: Optional[TimeSeries] = None,
    price_elasticity: float = 0.0,
) -> Tuple[TimeSeries, TimeSeries]:
    """
    Calculate relative attractiveness indices for Rx opioids and heroin.

    Attractiveness combines availability with optional price effects:
        attractiveness = availability * (1 / price) ** price_elasticity

    When price_elasticity = 0 (default), attractiveness equals availability.
    When price_elasticity > 0, lower prices increase attractiveness.

    The returned indices are relative to each other within each year,
    normalized so they sum to 1.0 (representing market share of attention).

    Args:
        rx_avail: TimeSeries of Rx opioid availability index.
        heroin_avail: TimeSeries of heroin availability index.
        rx_price: Optional TimeSeries of Rx opioid street prices.
        heroin_price: Optional TimeSeries of heroin street prices.
        price_elasticity: How much price affects attractiveness. Default 0
            means only availability matters.

    Returns:
        Tuple of (rx_attractiveness, heroin_attractiveness) TimeSeries.
        Both are normalized to sum to 1.0 in each year.

    Raises:
        ValueError: If time series have mismatched years.
    """
    # Verify matching years
    if set(rx_avail.years) != set(heroin_avail.years):
        raise ValueError(
            f"Year mismatch: rx_avail={rx_avail.years}, "
            f"heroin_avail={heroin_avail.years}"
        )

    years = rx_avail.years

    # If price elasticity is used, validate price series
    if price_elasticity != 0:
        if rx_price is not None and set(rx_price.years) != set(years):
            raise ValueError("rx_price years must match availability years")
        if heroin_price is not None and set(heroin_price.years) != set(years):
            raise ValueError("heroin_price years must match availability years")

    rx_attr = {}
    heroin_attr = {}

    for year in years:
        # Base attractiveness = availability
        rx_raw = rx_avail[year]
        heroin_raw = heroin_avail[year]

        # Apply price effects if price_elasticity != 0
        if price_elasticity != 0:
            if rx_price is not None:
                rx_raw *= (1.0 / rx_price[year]) ** price_elasticity
            if heroin_price is not None:
                heroin_raw *= (1.0 / heroin_price[year]) ** price_elasticity

        # Normalize to sum to 1.0
        total = rx_raw + heroin_raw
        if total == 0:
            # Edge case: both zero, split evenly
            rx_attr[year] = 0.5
            heroin_attr[year] = 0.5
        else:
            rx_attr[year] = rx_raw / total
            heroin_attr[year] = heroin_raw / total

    return (
        TimeSeries(rx_attr, name="rx_attractiveness"),
        TimeSeries(heroin_attr, name="heroin_attractiveness"),
    )
