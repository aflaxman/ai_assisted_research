"""Prepare measles incidence data for the four Bjornstad Fig 1.4 cities.

Sources (fetched by fetch_sources.sh into sources/):
- twentycities.rda -- He, Ionides & King (2010) UK registry data:
  weekly measles notifications + annual births and population for
  20 English & Welsh towns, 1944-1964.  London and Liverpool here.
  https://kingaa.github.io/pomp/vignettes/twentycities.rda
- dalziel.rda (from the epimdr2 R package, CRAN) -- Dalziel et al. (2016)
  biweekly measles incidence + population + susceptible recruits for
  40 pre-vaccination US cities, 1906-1948.  New York and Baltimore here.

Outputs (data/):
- <city>_cases.tsv        -- time (days), cases at the native cadence
                             (weekly UK, biweekly US)
- <city>_covariates.tsv   -- t (days), pop, birthrate (per capita per year,
                             school-entry lagged) on a ~monthly grid
- fig14_biweekly.tsv      -- biweekly incidence for all four cities
                             (UK weekly series aggregated to fortnights),
                             for the Figure 1.4 replica
- city_meta.tsv           -- per-city N0 (pop at t=0), window, cadence
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rdata

HERE = Path(__file__).parent
SRC = HERE / "sources"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

DAYS_PER_YEAR = 365.25

# Analysis windows (decimal years).  The book's Fig 1.4 spans 1944-1958.
# The UK registry covers that whole window; the Dalziel US series ends
# mid-1948, so the US window takes the last 14.5 pre-vaccination years.
UK_WINDOW = (1944.0, 1959.0)
US_WINDOW = (1934.0, 1948.5)


def load_uk():
    d = rdata.read_rda(SRC / "twentycities.rda")
    return d["measles"], d["demog"]


def load_us():
    d = rdata.read_rda(SRC / "dalziel.rda")
    return d["dalziel"]


def uk_city(measles, demog, town, out_name):
    """Weekly cases + monthly covariates for a UK town, 1944-1958."""
    cases = measles[measles["town"] == town].copy()
    # `date` is days since 1970-01-01 (R Date); convert to decimal years.
    cases["decyear"] = 1970.0 + cases["date"].astype(float) / DAYS_PER_YEAR
    cases = cases[(cases["decyear"] >= UK_WINDOW[0]) & (cases["decyear"] < UK_WINDOW[1])]
    cases = cases.sort_values("date").reset_index(drop=True)

    # t = 0 one reporting interval before the first observation, so the
    # first case count lands at t = 7 (the camdl emit_schedule convention).
    date0 = cases["date"].iloc[0] - 7.0
    cases["time"] = cases["date"] - date0
    cases_out = pd.DataFrame(
        {"time": cases["time"].astype(int), "weekly_cases": cases["cases"].astype(int)}
    )
    cases_out.to_csv(OUT / f"{out_name}_cases.tsv", sep="\t", index=False)

    # Covariates: annual pop (interpolated) and per-capita birthrate with
    # the 4-year school-entry lag of He et al. (2010).
    dem = demog[demog["town"] == town].sort_values("year")
    years = dem["year"].to_numpy(float)
    pop = dem["pop"].to_numpy(float)
    births = dem["births"].to_numpy(float)

    # Monthly grid covering the simulation window with margin.
    t_grid = np.arange(-60.0, cases["time"].max() + 120.0, 30.4375)
    decyear_grid = 1970.0 + (t_grid + date0) / DAYS_PER_YEAR
    pop_i = np.interp(decyear_grid, years + 0.5, pop)
    births_lagged = np.interp(decyear_grid - 4.0, years + 0.5, births)
    birthrate = births_lagged / pop_i  # per capita per year

    cov = pd.DataFrame({"t": t_grid, "pop": pop_i, "birthrate": birthrate})
    cov.to_csv(OUT / f"{out_name}_covariates.tsv", sep="\t", index=False)

    n0 = float(np.interp(1970.0 + date0 / DAYS_PER_YEAR, years + 0.5, pop))
    biweekly = biweekly_from_weekly(cases)
    return n0, date0, biweekly


def biweekly_from_weekly(cases):
    """Sum consecutive weekly counts into fortnights (Fig 1.4 cadence)."""
    c = cases["cases"].to_numpy(float)
    y = cases["decyear"].to_numpy(float)
    n = len(c) // 2 * 2
    return pd.DataFrame(
        {"decyear": y[:n:2], "cases": c[:n:2] + c[1 : n + 1 : 2]}
    )


def us_city(dalziel, loc, out_name):
    """Biweekly cases + monthly covariates for a Dalziel US city."""
    d = dalziel[dalziel["loc"] == loc].copy()
    d = d[(d["decimalYear"] >= US_WINDOW[0]) & (d["decimalYear"] < US_WINDOW[1])]
    d = d.sort_values("decimalYear").reset_index(drop=True)
    d = d[d["cases"].notna()].reset_index(drop=True)

    year0 = d["decimalYear"].iloc[0] - 14.0 / DAYS_PER_YEAR
    d["time"] = np.rint((d["decimalYear"] - year0) * DAYS_PER_YEAR).astype(int)
    cases_out = pd.DataFrame(
        {"time": d["time"], "biweekly_cases": d["cases"].astype(float).round().astype(int)}
    )
    cases_out.to_csv(OUT / f"{out_name}_cases.tsv", sep="\t", index=False)

    # Covariates: biweekly pop is given; Dalziel's `rec` are susceptible
    # recruits per biweek (births already lagged to school entry by the
    # source's reconstruction), so the per-capita per-year birthrate for
    # the model's birth stream is rec * 26 / pop.
    t_grid = np.arange(-60.0, d["time"].max() + 120.0, 30.4375)
    decyear_grid = year0 + t_grid / DAYS_PER_YEAR
    pop_i = np.interp(decyear_grid, d["decimalYear"], d["pop"].astype(float))
    rec = d["rec"].astype(float).rolling(26, center=True, min_periods=5).mean()
    rec_i = np.interp(decyear_grid, d["decimalYear"], rec)
    birthrate = rec_i * 26.0 / pop_i

    cov = pd.DataFrame({"t": t_grid, "pop": pop_i, "birthrate": birthrate})
    cov.to_csv(OUT / f"{out_name}_covariates.tsv", sep="\t", index=False)

    n0 = float(np.interp(year0, d["decimalYear"], d["pop"].astype(float)))
    biweekly = pd.DataFrame(
        {"decyear": d["decimalYear"], "cases": d["cases"].astype(float)}
    )
    return n0, year0, biweekly


def main():
    measles, demog = load_uk()
    dalziel = load_us()

    meta = []
    fig = {}

    for town, name in [("London", "london"), ("Liverpool", "liverpool")]:
        n0, date0, biweekly = uk_city(measles, demog, town, name)
        meta.append((name, n0, "weekly", 1970.0 + date0 / DAYS_PER_YEAR))
        fig[name] = biweekly

    for loc, name in [("NEW YORK", "newyork"), ("BALTIMORE", "baltimore")]:
        n0, year0, biweekly = us_city(dalziel, loc, name)
        meta.append((name, n0, "biweekly", year0))
        fig[name] = biweekly

    rows = []
    for name, bw in fig.items():
        bw = bw.copy()
        bw["city"] = name
        rows.append(bw)
    pd.concat(rows).to_csv(OUT / "fig14_biweekly.tsv", sep="\t", index=False)

    pd.DataFrame(meta, columns=["city", "N0", "cadence", "t0_decyear"]).to_csv(
        OUT / "city_meta.tsv", sep="\t", index=False
    )

    for name, n0, cadence, t0 in meta:
        print(f"{name}: N0={n0:,.0f}  cadence={cadence}  t0={t0:.3f}")


if __name__ == "__main__":
    main()
