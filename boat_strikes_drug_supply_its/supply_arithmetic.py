"""How large an effect on cocaine deaths should the campaign produce at all?

A null result only means something once you know what size of effect the design
was looking for. This module works forward from tonnage to an expected change
in cocaine-involved deaths, keeping every assumption explicit and every
uncertain input a range rather than a point.

The chain is:

    vessels destroyed x payload        -> tonnes removed by strikes
    + change in interdiction seizures  -> total change in tonnes removed
    / tonnes departing toward the US   -> fractional supply reduction
    x elasticity of deaths to supply   -> expected change in deaths

Sources for the inputs are listed in SOURCES and repeated in the README. The
elasticity is the weakest link and is deliberately swept over a wide range.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
LB_PER_TONNE = 2204.62

SOURCES = {
    "vessels_destroyed": "strike-level table compiled from NYT and ABC trackers "
                         "(see fetch_strikes.py), 70 vessels Sep 2025-Aug 2026",
    "payload_per_vessel": "go-fast boats typically carry ~1-2 t; low-profile "
                          "vessels 2-4 t. Swept 1.0-2.5 t.",
    "uscg_fy25_seizures": "USCG: ~510,000 lb cocaine seized in FY2025, a record",
    "uscg_baseline": "USCG: ~167,000 lb/yr is the long-run average",
    "us_bound_flow": "US-bound departures are not measured directly. Swept "
                     "300-900 t/yr, bracketing RAND/ONDCP consumption of "
                     "~145-250 t plus interdiction and in-transit losses. "
                     "Scenarios leaving less than consumption to arrive are "
                     "discarded as internally inconsistent.",
    "global_production": "UNODC World Drug Report: 3,708 t (2023), ~4,100 t (2024)",
    "elasticity": "No credible estimate exists for deaths per unit cocaine "
                  "supply. Swept 0.1-0.7; values near 1 are implausible given "
                  "inventories, inelastic demand, and that most "
                  "cocaine-involved deaths also involve fentanyl.",
}

# Ranges, not points: low / central / high.
VESSELS = 70
PAYLOAD_T = (1.0, 1.5, 2.5)
USCG_FY25_T = 510_000 / LB_PER_TONNE
USCG_BASELINE_T = 167_000 / LB_PER_TONNE
US_BOUND_FLOW_T = (300.0, 500.0, 700.0, 900.0)
ELASTICITY = (0.1, 0.3, 0.7)

# RAND/ONDCP put US consumption of pure cocaine near 145 t in 2010; use has
# risen since. A scenario is only internally consistent if what survives
# interdiction is at least roughly what Americans actually consume.
CONSUMPTION_T = (145.0, 250.0)

# Only about a quarter of the vessels struck were in the Caribbean; the rest
# were in the Eastern Pacific, whose cargo mostly moves on to Mexico and then
# overland. Both feed the US market, so the base case counts all of them, and
# the Caribbean-only variant is reported as a lower bound.
CARIBBEAN_VESSELS = 18


def tonnes_removed(vessels: int, payload_t: float,
                   include_seizure_surge: bool = True) -> dict:
    struck = vessels * payload_t
    surge = (USCG_FY25_T - USCG_BASELINE_T) if include_seizure_surge else 0.0
    return {
        "destroyed_by_strikes_t": struck,
        "interdiction_surge_t": surge,
        "total_removed_t": struck + surge,
    }


def expected_death_change(removed_t: float, flow_t: float,
                          elasticity: float) -> dict:
    """Fractional supply reduction and the death change it implies."""
    supply_cut = min(removed_t / flow_t, 0.95)
    return {
        "supply_cut_pct": 100.0 * supply_cut,
        "expected_death_change_pct": -100.0 * elasticity * supply_cut,
    }


def grid(vessels: int = VESSELS, include_seizure_surge: bool = True) -> pd.DataFrame:
    rows = []
    for payload, flow, elast in itertools.product(PAYLOAD_T, US_BOUND_FLOW_T,
                                                  ELASTICITY):
        rem = tonnes_removed(vessels, payload, include_seizure_surge)
        eff = expected_death_change(rem["total_removed_t"], flow, elast)
        arrivals = flow - rem["total_removed_t"]
        rows.append({
            "payload_t_per_vessel": payload,
            "us_bound_flow_t": flow,
            "elasticity": elast,
            "tonnes_removed": round(rem["total_removed_t"], 1),
            "implied_arrivals_t": round(arrivals, 1),
            "consistent": arrivals >= CONSUMPTION_T[0],
            "supply_cut_pct": round(eff["supply_cut_pct"], 1),
            "expected_death_change_pct": round(eff["expected_death_change_pct"], 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 78)
    print("SUPPLY ARITHMETIC: what size of effect is even on the table?")
    print("=" * 78)
    print(f"vessels destroyed by strikes           {VESSELS} "
          f"({CARIBBEAN_VESSELS} Caribbean, {VESSELS - CARIBBEAN_VESSELS} E. Pacific)")
    print(f"USCG cocaine seizures FY2025           {USCG_FY25_T:,.0f} t "
          f"(baseline {USCG_BASELINE_T:,.0f} t, surge "
          f"+{USCG_FY25_T - USCG_BASELINE_T:,.0f} t)")
    for payload in PAYLOAD_T:
        rem = tonnes_removed(VESSELS, payload)
        print(f"  at {payload:.1f} t/vessel: strikes destroy "
              f"{rem['destroyed_by_strikes_t']:,.0f} t, "
              f"total removed {rem['total_removed_t']:,.0f} t")

    full = grid()
    print("\nfractional supply reduction (independent of elasticity):")
    cut = (full.groupby(["payload_t_per_vessel", "us_bound_flow_t"])
           ["supply_cut_pct"].first().unstack())
    print(cut.round(1).to_string())

    print("\nexpected change in cocaine-involved deaths, %:")
    pivot = full.pivot_table(index=["payload_t_per_vessel", "us_bound_flow_t"],
                             columns="elasticity",
                             values="expected_death_change_pct")
    print(pivot.round(1).to_string())

    ok = full[full["consistent"]]
    dropped = full[~full["consistent"]]["us_bound_flow_t"].unique()
    print(f"\ninternal consistency: scenarios leaving less than "
          f"{CONSUMPTION_T[0]:.0f} t to reach US users are discarded "
          f"({len(dropped)} flow value(s) affected: "
          f"{', '.join(f'{v:.0f} t' for v in sorted(dropped))})")
    print(f"consistent scenarios: {len(ok)} of {len(full)}")
    print(f"expected change in cocaine deaths across consistent scenarios: "
          f"{ok['expected_death_change_pct'].min():.1f}% to "
          f"{ok['expected_death_change_pct'].max():.1f}%  "
          f"(median {ok['expected_death_change_pct'].median():.1f}%)")

    strikes_only = grid(include_seizure_surge=False)
    print(f"\nstrikes alone, excluding the interdiction surge: "
          f"{strikes_only['expected_death_change_pct'].min():.1f}% to "
          f"{strikes_only['expected_death_change_pct'].max():.1f}%")
    caribbean_only = grid(vessels=CARIBBEAN_VESSELS, include_seizure_surge=False)
    print(f"Caribbean strikes alone: "
          f"{caribbean_only['expected_death_change_pct'].min():.1f}% to "
          f"{caribbean_only['expected_death_change_pct'].max():.1f}%")

    full.to_csv(HERE / "outputs" / "supply_arithmetic.csv", index=False)
    print("\nSOURCES")
    for key, val in SOURCES.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
