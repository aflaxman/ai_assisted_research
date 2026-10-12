"""Load GBD 2023 and UNAIDS targets for the transmission fits.

Returns per-country targets (all-ages combined rates) and the HIV/ART inputs.
GBD rates are per 100,000; LTBI and HIV prevalences are converted to fractions.
These are model outputs used as fitting targets, never as validation (PLAN.md).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GBD = Path(__file__).resolve().parent.parent / "data" / "gbd"

COUNTRIES = ["Viet Nam", "South Africa", "India", "Kenya", "Netherlands"]


def _series(fname: str) -> pd.DataFrame:
    return pd.read_csv(GBD / fname)


def _rate_at(df: pd.DataFrame, country: str, year: int) -> dict:
    r = df[(df["location_name"] == country) & (df["year"] == year)]
    if r.empty:
        return {}
    r = r.iloc[0]
    return dict(rate=float(r["rate_combined"]),
                lo=float(r["rate_lower_combined"]), hi=float(r["rate_upper_combined"]))


def _active_prevalence(country: str, year: int) -> dict:
    """Active-TB prevalence per 100k.

    The GBD parent cause 'Tuberculosis' (297) folds LATENT infection into its
    prevalence, so parent prevalence is ~40% and not active disease.  Prefer the
    clean sum of active sub-causes (active_tb_prevalence.csv); otherwise fall
    back to parent-minus-LTBI, which matches it to within rounding.
    """
    f = GBD / "active_tb_prevalence.csv"
    if f.exists():
        return _rate_at(pd.read_csv(f), country, year)
    parent = _rate_at(_series("tuberculosis_prevalence.csv"), country, year)
    ltbi = _rate_at(_series("latent_tuberculosis_infection_prevalence.csv"), country, year)
    if not parent or not ltbi:
        return {}
    return dict(rate=parent["rate"] - ltbi["rate"],
                lo=max(parent["lo"] - ltbi["hi"], 0.0), hi=parent["hi"] - ltbi["lo"])


def targets(country: str, year: int = 2019) -> dict:
    """TB incidence, active prevalence, deaths (per 100k) and LTBI fraction."""
    inc = _rate_at(_series("tuberculosis_incidence.csv"), country, year)
    prev = _active_prevalence(country, year)
    dth = _rate_at(_series("tuberculosis_deaths.csv"), country, year)
    ltbi = _rate_at(_series("latent_tuberculosis_infection_prevalence.csv"), country, year)
    return dict(
        country=country, year=year,
        incidence=inc, prevalence=prev, deaths=dth,
        ltbi_fraction={k: v / 1e5 for k, v in ltbi.items()} if ltbi else {},
    )


def haq(country: str, year: int = 2019) -> float:
    h = _series("haq_index.csv")
    r = h[(h["location_name"] == country) & (h["year"] == year)]
    return float(r["value_combined"].iloc[0]) if not r.empty else float("nan")


def hiv_inputs(country: str):
    """Callables hiv_prevalence(t) and art_coverage(t) over calendar year t."""
    hiv = _series("hiv-aids_prevalence.csv")
    h = hiv[hiv["location_name"] == country][["year", "rate_combined"]].copy()
    h["frac"] = h["rate_combined"] / 1e5
    hdict = dict(zip(h["year"], h["frac"]))

    art = _series("art_coverage_worldbank.csv")
    a = art[art["location"] == country][["year", "art_coverage_pct_plhiv"]]
    adict = dict(zip(a["year"], a["art_coverage_pct_plhiv"] / 100.0))

    def hiv_prev(t):
        yr = int(round(t))
        yr = min(max(yr, min(hdict)), max(hdict))
        return hdict.get(yr, 0.0)

    def art_cov(t):
        if not adict:
            return 0.0
        yr = int(round(t))
        yr = min(max(yr, min(adict)), max(adict))
        return adict.get(yr, list(adict.values())[0])

    return hiv_prev, art_cov
