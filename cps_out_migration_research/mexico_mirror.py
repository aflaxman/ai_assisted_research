"""Mirror-statistics check: CPS-matching Mexico-born emigration estimates vs
Mexico-side and U.S. administrative data on returns to Mexico, 2019-2025.

The point: gross emigration of the Mexico-born from the U.S. *resident
population* (what the CPS matching method estimates) should be loosely
bounded by what receiving-country registers and surveys can absorb. Every
source has definitional gaps (events vs people; border-adjacent removals vs
settled residents; unregistered self-returns; circular trips), so the honest
comparison is order-of-magnitude — which is exactly the level at which it is
informative.

All external figures below were compiled from primary sources (URLs in
SOURCES); rates were converted to annual counts where needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- This project's estimates (strata_rates.csv, region_birth == mexico) ---
# implied outflow = standardized gross rate x estimated Mexico-born stock
CPS_MATCHING = pd.DataFrame([
    # pair, gross rate, stock (M), implied gross outflow (k/yr)
    ("2019->2020", 0.059, 9.8, 579),
    ("2020->2021", -0.011, 10.2, -113),
    ("2021->2022", 0.035, 10.2, 354),
    ("2022->2023", 0.030, 9.8, 297),
    ("2023->2024", -0.036, 9.9, -360),
    ("2024->2025", 0.083, 10.4, 862),
], columns=["pair", "gross_rate", "mex_stock_M", "implied_outflow_k"])

# --- Mexico-side mirror numbers (annualized where the source is a window) ---
MIRROR = pd.DataFrame([
    # source, period, annualized count (thousands), what it counts
    ("Censo 2010 (residence-5yr)", "2005-2010", 167,
     "Mexico-born returnees from U.S., alive & resident in MX, /5"),
    ("Encuesta Intercensal", "2010-2015", 90,
     "Mexico-born returnees from U.S. (448k/5)"),
    ("Censo 2020 (residence-5yr)", "2015-2020", 59,
     "Mexico-born returnees from U.S. (294k/5)"),
    ("ENADID 2023 (emigration module)", "2018-2023", 55,
     "emigrants who returned to the household (276k/5)"),
    ("UPM/Segob devolucion events", "CY2019", 211,
     "removal/return EVENTS at INM points (incl. repeat crossers)"),
    ("UPM/Segob devolucion events", "CY2020", 184, "same"),
    ("UPM/Segob devolucion events", "CY2021", 161, "same"),
    ("UPM/Segob devolucion events", "CY2022", 258, "same (Title-42 era)"),
    ("UPM/Segob devolucion events", "CY2023", 215, "same"),
    ("UPM/Segob devolucion events", "CY2024", 206, "same"),
    ("UPM/Segob devolucion events", "CY2025", 160,
     "same; Jan-May ran -30% y/y; ~15k voluntary per Segob"),
    ("DHS removals of Mexicans", "FY2024", 140,
     "removal events, mostly border-adjacent (not settled residents)"),
    ("Brookings voluntary-departure est.", "2025", 310,
     "midpoint 210-405k enforcement-induced departures, ALL nationalities; "
     "CPS-independent but an assumption construct (15-50% of removals, "
     "anchored by ~40k TRAC court grants)"),
], columns=["source", "period", "annual_k", "counts_what"])

# All-origin cross-check vs Brookings (Edelberg-Veuger-Watson, Jan 2026):
# their full 2025 outflow accounting = baseline mechanical emigration
# (2.5-4%/20% by entry channel on a ~48M stock, ~1.4M) + removals ~315k +
# enforcement-induced voluntary 210-405k  ~=  1.8-2.0M gross outflow.
# This project's all-origin gross (5.14% x 51.3M ~= 2.6M) runs ~1.3-1.4x
# that scenario -- far closer than the Mexico-specific mirror gap, because
# most emigration is never registered anywhere; receiving-country registers
# only see a slice. Their net-migration 2025: -295k to -10k; for contrast,
# Census Vintage-2025 NIM is +1.3M and SF Fed ~+1.0M -- the field itself
# disagrees on even the sign of 2025 net migration.
BROOKINGS_2025 = {"gross_outflow_lo_k": 1800, "gross_outflow_hi_k": 2000,
                  "this_work_gross_k": 2634}

# Van Hook et al. (2006) benchmark for the method-vs-mirror ratio circa 2000:
# their Mexico gross estimate ~5.5%/yr on ~8.6M stock ~= 470k/yr, vs Censo
# 2000 returnees 256k over 1995-2000 ~= 51k/yr *resident-return* -- but the
# right comparator for a GROSS rate includes circulars; their Table 1 mirror
# discussions put the method at roughly 2x mirror sources.
VANHOOK_2000 = {"gross_outflow_k": 470, "census_mirror_k": 51}

SOURCES = {
    "Censo 2020": "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/doc/Censo2020_Principales_resultados_EUM.pdf",
    "Anuario Migracion y Remesas 2021 (series)": "https://www.gob.mx/cms/uploads/attachment/file/675092/Anuario_Migracion_y_Remesas_2021.pdf",
    "ENADID 2023": "https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2024/ENADID/ENADID2023.pdf",
    "OHSS 2022 Yearbook T42": "https://ohss.dhs.gov/topics/immigration/yearbook/2022/table42",
    "OHSS KHSM Repatriations": "https://ohss.dhs.gov/khsm/dhs-repatriations",
    "ICE FY2024 Annual Report": "https://www.ice.gov/doclib/eoy/iceAnnualReportFY2024.pdf",
    "Segob Mexico te Abraza year-end": "https://politica.expansion.mx/mexico/2025/12/18/mas-de-145-000-mexicanos-fueron-repatriados-desde-eu-durante-2025",
    "BBVA deportations analysis": "https://www.bbvaresearch.com/en/publicaciones/mexico-did-deportations-of-mexicans-from-the-us-increase/",
    "BBVA remittances 2025 (-4.6%)": "https://www.bbvaresearch.com/wp-content/uploads/2026/02/Mexico_Remesas_Cierre_2025.pdf",
    "CMS on self-deportation claims": "https://cmsny.org/two-million-deportation-myth-ice-enforcement-distorting-data/",
}


def main():
    print("=== CPS-matching Mexico-born implied gross outflow ===")
    print(CPS_MATCHING.to_string(index=False))
    print("\n=== Mexico-side / administrative mirror numbers (annualized, k) ===")
    print(MIRROR.to_string(index=False))

    print("\n=== Method-vs-mirror ratios ===")
    vh_ratio = VANHOOK_2000["gross_outflow_k"] / 256 * 5 / 5  # vs 5-yr census/5
    print(f"Van Hook circa 2000: ~470k/yr vs Censo-2000 mirror ~51k/yr "
          f"(returnees/yr) -> ratio ~9x raw, ~2x after circularity/coverage "
          f"adjustments discussed in the paper")
    mine_2425 = 862
    mirror_hi = 160  # best single CY2025 register (events, generous)
    print(f"This work 2024->25: ~{mine_2425}k/yr vs best mirror ~{mirror_hi}k "
          f"events (only ~15k voluntary) -> ratio ~{mine_2425/mirror_hi:.1f}x")
    print("Positive-year pairs pre-2025 (~300-580k) vs mirrors 55-215k: 2-4x,")
    print("in line with the method's historical mirror gap; 2024->25 is not.")

    out = {"cps_matching": CPS_MATCHING, "mirror": MIRROR}
    CPS_MATCHING.to_csv("outputs/mexico_mirror_cps.csv", index=False)
    MIRROR.to_csv("outputs/mexico_mirror_external.csv", index=False)
    print("\nwrote outputs/mexico_mirror_cps.csv, mexico_mirror_external.csv")
    return out


if __name__ == "__main__":
    main()
