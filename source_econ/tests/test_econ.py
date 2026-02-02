"""
Unit tests for SOURCE economic input layer.

Tests cover:
  - TimeSeries dataclass functionality
  - heroin_availability calculations
  - rx_availability_index calculations
  - relative_attractiveness calculations
  - Error handling for invalid inputs
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from source_econ.econ import (
    TimeSeries,
    RxAvailabilityWeights,
    heroin_availability,
    rx_availability_index,
    relative_attractiveness,
)
from source_econ.io import (
    load_heroin_price_csv,
    load_rx_inputs_csv,
    save_results_csv,
    validate_year_alignment,
)


class TestTimeSeries:
    """Tests for TimeSeries dataclass."""

    def test_basic_creation(self):
        """TimeSeries can be created with year->value dict."""
        ts = TimeSeries({2010: 100.0, 2011: 110.0, 2012: 105.0})
        assert ts[2010] == 100.0
        assert ts[2011] == 110.0
        assert 2012 in ts
        assert 2009 not in ts

    def test_years_sorted(self):
        """Years property returns sorted list."""
        ts = TimeSeries({2012: 1.0, 2010: 2.0, 2011: 3.0})
        assert ts.years == [2010, 2011, 2012]

    def test_values_year_order(self):
        """Values property returns values in year-sorted order."""
        ts = TimeSeries({2012: 3.0, 2010: 1.0, 2011: 2.0})
        assert ts.values == [1.0, 2.0, 3.0]

    def test_min_max_year(self):
        """Min and max year properties work correctly."""
        ts = TimeSeries({2015: 1.0, 2010: 2.0, 2020: 3.0})
        assert ts.min_year == 2010
        assert ts.max_year == 2020

    def test_empty_data_raises(self):
        """Empty data dictionary raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TimeSeries({})

    def test_non_int_year_raises(self):
        """Non-integer year raises TypeError."""
        with pytest.raises(TypeError, match="Year must be int"):
            TimeSeries({"2010": 100.0})

    def test_is_monotonic_contiguous(self):
        """is_monotonic_years returns True for contiguous years."""
        ts = TimeSeries({2010: 1.0, 2011: 2.0, 2012: 3.0})
        assert ts.is_monotonic_years() is True

    def test_is_monotonic_with_gaps(self):
        """is_monotonic_years returns False when years have gaps."""
        ts = TimeSeries({2010: 1.0, 2012: 3.0})  # Missing 2011
        assert ts.is_monotonic_years() is False

    def test_check_monotonic_raises_on_gaps(self):
        """check_monotonic_years raises ValueError when gaps exist."""
        ts = TimeSeries({2010: 1.0, 2013: 4.0})  # Missing 2011, 2012
        with pytest.raises(ValueError, match="Missing.*2011.*2012"):
            ts.check_monotonic_years()

    def test_interpolate_fills_gaps(self):
        """interpolate_to_annual fills missing years linearly."""
        ts = TimeSeries({2010: 100.0, 2012: 200.0})
        interp = ts.interpolate_to_annual()
        assert interp.years == [2010, 2011, 2012]
        assert interp[2011] == 150.0  # Linear interpolation

    def test_interpolate_no_change_if_contiguous(self):
        """interpolate_to_annual returns copy if already contiguous."""
        ts = TimeSeries({2010: 1.0, 2011: 2.0, 2012: 3.0})
        interp = ts.interpolate_to_annual()
        assert interp.years == ts.years
        assert interp.values == ts.values

    def test_normalize_first(self):
        """normalize(method='first') makes first year = 1.0."""
        ts = TimeSeries({2010: 200.0, 2011: 300.0, 2012: 250.0})
        norm = ts.normalize(method="first")
        assert norm[2010] == 1.0
        assert norm[2011] == 1.5
        assert norm[2012] == 1.25

    def test_normalize_mean(self):
        """normalize(method='mean') makes mean = 1.0."""
        ts = TimeSeries({2010: 100.0, 2011: 200.0, 2012: 300.0})
        norm = ts.normalize(method="mean")
        assert np.isclose(np.mean(norm.values), 1.0)

    def test_normalize_max(self):
        """normalize(method='max') makes max = 1.0."""
        ts = TimeSeries({2010: 50.0, 2011: 200.0, 2012: 100.0})
        norm = ts.normalize(method="max")
        assert norm[2011] == 1.0
        assert norm[2010] == 0.25

    def test_normalize_ref(self):
        """normalize(method='ref') uses provided reference value."""
        ts = TimeSeries({2010: 100.0, 2011: 200.0})
        norm = ts.normalize(method="ref", ref_value=50.0)
        assert norm[2010] == 2.0
        assert norm[2011] == 4.0

    def test_normalize_zero_ref_raises(self):
        """normalize raises if reference value is zero."""
        ts = TimeSeries({2010: 0.0, 2011: 100.0})
        with pytest.raises(ValueError, match="reference value is zero"):
            ts.normalize(method="first")

    def test_rescale(self):
        """rescale multiplies all values by factor."""
        ts = TimeSeries({2010: 10.0, 2011: 20.0})
        scaled = ts.rescale(2.5)
        assert scaled[2010] == 25.0
        assert scaled[2011] == 50.0


class TestHeroinAvailability:
    """Tests for heroin_availability function."""

    def test_basic_inverse_relationship(self):
        """Availability rises when price falls."""
        prices = TimeSeries({2010: 400, 2011: 350, 2012: 300})
        avail = heroin_availability(prices)

        # First year is baseline = 1.0
        assert avail[2010] == 1.0
        # Lower prices -> higher availability
        assert avail[2011] > avail[2010]
        assert avail[2012] > avail[2011]

    def test_elasticity_effect(self):
        """Higher elasticity amplifies price effects."""
        prices = TimeSeries({2010: 400, 2011: 200})

        avail_e1 = heroin_availability(prices, elasticity=1.0)
        avail_e2 = heroin_availability(prices, elasticity=2.0)

        # With elasticity=1, 50% price drop -> 2x availability
        assert avail_e1[2011] == 2.0
        # With elasticity=2, 50% price drop -> 4x availability
        assert avail_e2[2011] == 4.0

    def test_custom_price_ref(self):
        """Custom price_ref changes baseline."""
        prices = TimeSeries({2010: 400, 2011: 200})
        avail = heroin_availability(prices, price_ref=200)

        # 2010: 200/400 = 0.5
        # 2011: 200/200 = 1.0
        assert avail[2010] == 0.5
        assert avail[2011] == 1.0

    def test_zero_price_raises(self):
        """Zero price raises ValueError."""
        prices = TimeSeries({2010: 400, 2011: 0})
        with pytest.raises(ValueError, match="must be positive"):
            heroin_availability(prices)

    def test_negative_price_raises(self):
        """Negative price raises ValueError."""
        prices = TimeSeries({2010: -100})
        with pytest.raises(ValueError, match="must be positive"):
            heroin_availability(prices)


class TestRxAvailabilityWeights:
    """Tests for RxAvailabilityWeights dataclass."""

    def test_default_weights(self):
        """Default weights are as documented."""
        w = RxAvailabilityWeights()
        assert w.prescriptions == 0.3
        assert w.patients == 0.3
        assert w.mme == 0.3
        assert w.adf_share == 0.1

    def test_total(self):
        """total() returns sum of all weights."""
        w = RxAvailabilityWeights(0.2, 0.3, 0.4, 0.1)
        assert w.total() == 1.0

    def test_normalized(self):
        """normalized() returns weights summing to 1.0."""
        w = RxAvailabilityWeights(2, 3, 4, 1)  # Sum = 10
        n = w.normalized()
        assert n.prescriptions == 0.2
        assert n.patients == 0.3
        assert n.mme == 0.4
        assert n.adf_share == 0.1
        assert n.total() == 1.0

    def test_negative_weight_raises(self):
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            RxAvailabilityWeights(prescriptions=-0.1)


class TestRxAvailabilityIndex:
    """Tests for rx_availability_index function."""

    def test_basic_calculation(self):
        """Basic index calculation produces expected results."""
        years = [2010, 2011, 2012]
        prescriptions = TimeSeries(dict(zip(years, [100, 90, 80])))
        patients = TimeSeries(dict(zip(years, [50, 45, 40])))
        mme = TimeSeries(dict(zip(years, [200, 180, 160])))
        adf_share = TimeSeries(dict(zip(years, [0.1, 0.2, 0.3])))

        rx_avail = rx_availability_index(
            prescriptions, patients, mme, adf_share
        )

        # First year should be close to baseline (but not exactly 1.0
        # due to ADF effect)
        assert 2010 in rx_avail
        # Availability should decrease as inputs decrease and ADF rises
        assert rx_avail[2012] < rx_avail[2010]

    def test_year_mismatch_raises(self):
        """Mismatched years between inputs raises ValueError."""
        prescriptions = TimeSeries({2010: 100, 2011: 90})
        patients = TimeSeries({2010: 50, 2012: 40})  # Wrong year!
        mme = TimeSeries({2010: 200, 2011: 180})
        adf_share = TimeSeries({2010: 0.1, 2011: 0.2})

        with pytest.raises(ValueError, match="Year mismatch"):
            rx_availability_index(prescriptions, patients, mme, adf_share)

    def test_invalid_adf_share_raises(self):
        """ADF share outside [0,1] raises ValueError."""
        years = [2010, 2011]
        prescriptions = TimeSeries(dict(zip(years, [100, 90])))
        patients = TimeSeries(dict(zip(years, [50, 45])))
        mme = TimeSeries(dict(zip(years, [200, 180])))
        adf_share = TimeSeries(dict(zip(years, [0.1, 1.5])))  # Invalid!

        with pytest.raises(ValueError, match="adf_share must be in"):
            rx_availability_index(prescriptions, patients, mme, adf_share)


class TestRelativeAttractiveness:
    """Tests for relative_attractiveness function."""

    def test_sums_to_one(self):
        """Attractiveness values sum to 1.0 in each year."""
        rx_avail = TimeSeries({2010: 1.0, 2011: 0.8, 2012: 0.6})
        heroin_avail = TimeSeries({2010: 1.0, 2011: 1.2, 2012: 1.4})

        rx_attr, heroin_attr = relative_attractiveness(rx_avail, heroin_avail)

        for year in rx_attr.years:
            total = rx_attr[year] + heroin_attr[year]
            assert np.isclose(total, 1.0), f"Year {year}: sum = {total}"

    def test_higher_availability_means_higher_attr(self):
        """Higher availability leads to higher attractiveness share."""
        rx_avail = TimeSeries({2010: 2.0, 2011: 1.0})
        heroin_avail = TimeSeries({2010: 1.0, 2011: 1.0})

        rx_attr, heroin_attr = relative_attractiveness(rx_avail, heroin_avail)

        # 2010: Rx has higher availability -> higher attractiveness
        assert rx_attr[2010] > heroin_attr[2010]
        # 2011: Equal availability -> equal attractiveness
        assert rx_attr[2011] == 0.5
        assert heroin_attr[2011] == 0.5

    def test_price_elasticity_zero(self):
        """With price_elasticity=0, prices are ignored."""
        rx_avail = TimeSeries({2010: 1.0})
        heroin_avail = TimeSeries({2010: 1.0})
        rx_price = TimeSeries({2010: 10.0})
        heroin_price = TimeSeries({2010: 100.0})

        rx_attr, heroin_attr = relative_attractiveness(
            rx_avail, heroin_avail,
            rx_price=rx_price,
            heroin_price=heroin_price,
            price_elasticity=0.0,
        )

        # Prices should have no effect
        assert rx_attr[2010] == 0.5
        assert heroin_attr[2010] == 0.5

    def test_price_elasticity_positive(self):
        """With price_elasticity>0, lower prices increase attractiveness."""
        rx_avail = TimeSeries({2010: 1.0})
        heroin_avail = TimeSeries({2010: 1.0})
        rx_price = TimeSeries({2010: 10.0})
        heroin_price = TimeSeries({2010: 100.0})

        rx_attr, heroin_attr = relative_attractiveness(
            rx_avail, heroin_avail,
            rx_price=rx_price,
            heroin_price=heroin_price,
            price_elasticity=1.0,
        )

        # Rx has lower price -> higher attractiveness
        assert rx_attr[2010] > heroin_attr[2010]

    def test_year_mismatch_raises(self):
        """Mismatched years raises ValueError."""
        rx_avail = TimeSeries({2010: 1.0, 2011: 0.9})
        heroin_avail = TimeSeries({2010: 1.0, 2012: 1.1})  # Wrong year!

        with pytest.raises(ValueError, match="Year mismatch"):
            relative_attractiveness(rx_avail, heroin_avail)


class TestIO:
    """Tests for I/O functions."""

    def test_load_heroin_price_csv(self, tmp_path):
        """load_heroin_price_csv loads valid CSV correctly."""
        csv_content = "year,heroin_price_usd\n2010,400\n2011,350\n"
        csv_path = tmp_path / "heroin.csv"
        csv_path.write_text(csv_content)

        ts = load_heroin_price_csv(csv_path)
        assert ts[2010] == 400.0
        assert ts[2011] == 350.0

    def test_load_heroin_price_missing_column(self, tmp_path):
        """Missing column raises ValueError."""
        csv_content = "year,price\n2010,400\n"  # Wrong column name
        csv_path = tmp_path / "heroin.csv"
        csv_path.write_text(csv_content)

        with pytest.raises(ValueError, match="Missing required column"):
            load_heroin_price_csv(csv_path)

    def test_load_heroin_price_invalid_price(self, tmp_path):
        """Negative price raises ValueError."""
        csv_content = "year,heroin_price_usd\n2010,-100\n"
        csv_path = tmp_path / "heroin.csv"
        csv_path.write_text(csv_content)

        with pytest.raises(ValueError, match="must be positive"):
            load_heroin_price_csv(csv_path)

    def test_load_rx_inputs_csv(self, tmp_path):
        """load_rx_inputs_csv loads valid CSV correctly."""
        csv_content = (
            "year,prescriptions,patients,mme,adf_share\n"
            "2010,250,45,100,0.05\n"
            "2011,245,44,98,0.08\n"
        )
        csv_path = tmp_path / "rx.csv"
        csv_path.write_text(csv_content)

        presc, pat, mme, adf = load_rx_inputs_csv(csv_path)
        assert presc[2010] == 250.0
        assert pat[2011] == 44.0
        assert mme[2010] == 100.0
        assert adf[2011] == 0.08

    def test_load_rx_inputs_invalid_adf(self, tmp_path):
        """ADF share > 1 raises ValueError."""
        csv_content = (
            "year,prescriptions,patients,mme,adf_share\n"
            "2010,250,45,100,1.5\n"
        )
        csv_path = tmp_path / "rx.csv"
        csv_path.write_text(csv_content)

        with pytest.raises(ValueError, match="ADF share must be in"):
            load_rx_inputs_csv(csv_path)

    def test_save_results_csv(self, tmp_path):
        """save_results_csv creates valid CSV."""
        output_path = tmp_path / "output.csv"
        save_results_csv(
            output_path,
            years=[2010, 2011],
            columns={"col_a": [1.0, 2.0], "col_b": [3.0, 4.0]},
        )

        content = output_path.read_text()
        assert "year" in content
        assert "col_a" in content
        assert "col_b" in content

    def test_validate_year_alignment(self):
        """validate_year_alignment detects mismatches."""
        ts1 = TimeSeries({2010: 1.0, 2011: 2.0})
        ts2 = TimeSeries({2010: 1.0, 2011: 2.0})
        ts3 = TimeSeries({2010: 1.0, 2012: 2.0})  # Mismatch!

        # Should succeed with matching years
        years = validate_year_alignment(ts1, ts2)
        assert years == [2010, 2011]

        # Should fail with mismatched years
        with pytest.raises(ValueError, match="Year mismatch"):
            validate_year_alignment(ts1, ts3)


class TestInterpolationHandling:
    """Tests for handling missing years via interpolation."""

    def test_heroin_availability_with_gaps(self):
        """heroin_availability works with interpolated prices."""
        # Prices with gap in 2011
        prices_sparse = TimeSeries({2010: 400, 2012: 300})
        prices_interp = prices_sparse.interpolate_to_annual()

        avail = heroin_availability(prices_interp)

        assert avail[2011] == pytest.approx(400 / 350)  # Interpolated price
        assert avail.years == [2010, 2011, 2012]

    def test_interpolation_for_rx_inputs(self):
        """Interpolation fills gaps in Rx input series."""
        # Sparse data with gap
        sparse = TimeSeries({2010: 100, 2012: 200, 2014: 300})
        interp = sparse.interpolate_to_annual()

        assert interp.years == [2010, 2011, 2012, 2013, 2014]
        assert interp[2011] == 150.0  # Linear interpolation
        assert interp[2013] == 250.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_year_timeseries(self):
        """Single-year TimeSeries is valid."""
        ts = TimeSeries({2010: 100.0})
        assert ts.is_monotonic_years()
        assert ts.years == [2010]

    def test_both_availabilities_zero(self):
        """Zero availability for both gives 50/50 split."""
        rx_avail = TimeSeries({2010: 0.0})
        heroin_avail = TimeSeries({2010: 0.0})

        rx_attr, heroin_attr = relative_attractiveness(rx_avail, heroin_avail)
        assert rx_attr[2010] == 0.5
        assert heroin_attr[2010] == 0.5

    def test_numpy_integer_years(self):
        """NumPy integer types work as years."""
        ts = TimeSeries({np.int64(2010): 100.0, np.int32(2011): 200.0})
        assert 2010 in ts
        assert ts[2011] == 200.0
