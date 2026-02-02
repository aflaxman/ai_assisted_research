#!/usr/bin/env python
"""
Example end-to-end run of the SOURCE economic input layer.

This script demonstrates:
  1. Creating toy input CSV files with plausible placeholder data
  2. Loading the data using the I/O utilities
  3. Calculating heroin availability, Rx availability, and attractiveness
  4. Printing results and saving to output.csv

Run from the source_econ directory:
    python example_run.py

Or from the parent directory:
    python -m source_econ.example_run
"""

from __future__ import annotations

import os
from pathlib import Path

# Local imports (works whether run as script or module)
try:
    from source_econ import (
        load_heroin_price_csv,
        load_rx_inputs_csv,
        heroin_availability,
        rx_availability_index,
        relative_attractiveness,
        save_results_csv,
    )
except ImportError:
    # Running as script from source_econ directory
    from . import (
        load_heroin_price_csv,
        load_rx_inputs_csv,
        heroin_availability,
        rx_availability_index,
        relative_attractiveness,
        save_results_csv,
    )


def create_toy_heroin_price_csv(filepath: Path) -> None:
    """
    Create a toy heroin price CSV with plausible placeholder data.

    NOTE: These values are PLACEHOLDERS for demonstration only.
    Real data would come from UNODC World Drug Reports or similar sources.

    The pattern shows: prices declining 2010-2016 (increased supply),
    then rising 2017-2020 (fentanyl disruption, enforcement).
    """
    # Placeholder data - prices in USD per pure gram (street level)
    # Real UNODC data would show regional variation; this is illustrative
    data = """year,heroin_price_usd
2010,400
2011,380
2012,350
2013,320
2014,300
2015,280
2016,260
2017,290
2018,320
2019,340
2020,360"""
    filepath.write_text(data)
    print(f"Created toy heroin price data: {filepath}")


def create_toy_rx_inputs_csv(filepath: Path) -> None:
    """
    Create a toy Rx inputs CSV with plausible placeholder data.

    NOTE: These values are PLACEHOLDERS for demonstration only.
    Real data would come from IQVIA audits (NPA/TPT/NSP), which are proprietary.

    The pattern shows:
    - Prescriptions/patients/MME: rising 2010-2012, then declining
    - ADF share: gradually increasing as reformulated products entered market
    """
    # Placeholder data
    # - prescriptions: millions of annual Rx
    # - patients: millions of unique patients
    # - mme: billions of morphine milligram equivalents
    # - adf_share: fraction that are abuse-deterrent formulations
    data = """year,prescriptions,patients,mme,adf_share
2010,250,40,85,0.02
2011,260,42,90,0.03
2012,255,43,92,0.05
2013,245,41,88,0.08
2014,235,39,82,0.12
2015,220,37,75,0.18
2016,205,35,68,0.25
2017,190,33,60,0.32
2018,175,31,52,0.38
2019,160,29,45,0.42
2020,150,28,40,0.45"""
    filepath.write_text(data)
    print(f"Created toy Rx inputs data: {filepath}")


def print_results_table(
    years: list,
    heroin_price: list,
    heroin_avail: list,
    rx_avail: list,
    rx_attr: list,
    heroin_attr: list,
) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 75)
    print("SOURCE Economic Input Layer - Results")
    print("=" * 75)
    print(
        f"{'Year':>6}  {'H_Price':>8}  {'H_Avail':>8}  {'Rx_Avail':>8}  "
        f"{'Rx_Attr':>8}  {'H_Attr':>8}"
    )
    print("-" * 75)

    for i, year in enumerate(years):
        print(
            f"{year:>6}  {heroin_price[i]:>8.1f}  {heroin_avail[i]:>8.3f}  "
            f"{rx_avail[i]:>8.3f}  {rx_attr[i]:>8.3f}  {heroin_attr[i]:>8.3f}"
        )

    print("-" * 75)
    print("Legend:")
    print("  H_Price   = Heroin street price (USD/gram, placeholder)")
    print("  H_Avail   = Heroin availability index (inverse of price)")
    print("  Rx_Avail  = Rx opioid availability index (composite)")
    print("  Rx_Attr   = Rx opioid relative attractiveness (normalized)")
    print("  H_Attr    = Heroin relative attractiveness (normalized)")
    print("=" * 75 + "\n")


def main() -> None:
    """Run the example end-to-end."""
    # Determine output directory (same as this script)
    script_dir = Path(__file__).parent
    data_dir = script_dir / "example_data"
    data_dir.mkdir(exist_ok=True)

    # Step 1: Create toy input CSV files
    print("Step 1: Creating toy input data files...")
    heroin_csv = data_dir / "heroin_price.csv"
    rx_csv = data_dir / "rx_inputs.csv"

    create_toy_heroin_price_csv(heroin_csv)
    create_toy_rx_inputs_csv(rx_csv)

    # Step 2: Load the data
    print("\nStep 2: Loading input data...")
    heroin_price_ts = load_heroin_price_csv(heroin_csv)
    rx_prescriptions, rx_patients, rx_mme, adf_share = load_rx_inputs_csv(
        rx_csv
    )
    print(f"  Loaded heroin prices: {heroin_price_ts.years[0]}-"
          f"{heroin_price_ts.years[-1]}")
    print(f"  Loaded Rx inputs: {rx_prescriptions.years[0]}-"
          f"{rx_prescriptions.years[-1]}")

    # Step 3: Calculate heroin availability
    print("\nStep 3: Calculating heroin availability index...")
    heroin_avail_ts = heroin_availability(heroin_price_ts, elasticity=1.0)
    print(f"  First year availability (baseline): {heroin_avail_ts[2010]:.3f}")
    print(f"  Peak availability year 2016: {heroin_avail_ts[2016]:.3f}")

    # Step 4: Calculate Rx availability index
    print("\nStep 4: Calculating Rx opioid availability index...")
    rx_avail_ts = rx_availability_index(
        rx_prescriptions=rx_prescriptions,
        rx_patients=rx_patients,
        rx_mme=rx_mme,
        adf_share=adf_share,
        normalize=True,  # Normalize components before combining
    )
    print(f"  First year Rx availability: {rx_avail_ts[2010]:.3f}")
    print(f"  Last year Rx availability: {rx_avail_ts[2020]:.3f}")

    # Step 5: Calculate relative attractiveness
    print("\nStep 5: Calculating relative attractiveness indices...")
    rx_attr_ts, heroin_attr_ts = relative_attractiveness(
        rx_avail=rx_avail_ts,
        heroin_avail=heroin_avail_ts,
        price_elasticity=0.0,  # Only availability matters (default)
    )
    print(f"  2010 Rx attractiveness: {rx_attr_ts[2010]:.3f}")
    print(f"  2010 Heroin attractiveness: {heroin_attr_ts[2010]:.3f}")
    print(f"  2020 Rx attractiveness: {rx_attr_ts[2020]:.3f}")
    print(f"  2020 Heroin attractiveness: {heroin_attr_ts[2020]:.3f}")

    # Step 6: Compile results and print table
    years = heroin_price_ts.years
    heroin_prices = heroin_price_ts.values
    heroin_avails = heroin_avail_ts.values
    rx_avails = rx_avail_ts.values
    rx_attrs = rx_attr_ts.values
    heroin_attrs = heroin_attr_ts.values

    print_results_table(
        years, heroin_prices, heroin_avails, rx_avails, rx_attrs, heroin_attrs
    )

    # Step 7: Save results to CSV
    output_path = data_dir / "output.csv"
    save_results_csv(
        output_path,
        years=years,
        columns={
            "heroin_price": heroin_prices,
            "heroin_availability": heroin_avails,
            "rx_availability": rx_avails,
            "rx_attractiveness": rx_attrs,
            "heroin_attractiveness": heroin_attrs,
        },
    )
    print(f"Results saved to: {output_path}")

    # Interpretation notes
    print("\n" + "=" * 75)
    print("Interpretation Notes:")
    print("=" * 75)
    print("""
These results show how economic inputs translate to availability and
attractiveness indices that could feed into an opioid use simulation:

1. HEROIN AVAILABILITY rises as price falls (2010-2016), indicating
   increased market supply. After 2016, rising prices (possibly due to
   fentanyl market disruption or enforcement) reduce availability.

2. RX AVAILABILITY declines steadily as prescribing falls and ADF share
   rises, reflecting policy interventions (PDMP programs, CDC guidelines,
   abuse-deterrent reformulations).

3. RELATIVE ATTRACTIVENESS shows the market shift: in 2010, Rx opioids
   had higher attractiveness; by 2020, heroin's relative attractiveness
   increased as Rx became harder to obtain.

NOTE: All input data are PLACEHOLDERS. Real analysis would require:
  - UNODC data for heroin street prices
  - IQVIA proprietary data for Rx components
  - Calibration against observed transition rates
""")


if __name__ == "__main__":
    main()
