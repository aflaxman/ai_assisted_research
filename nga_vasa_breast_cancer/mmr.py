"""DHS-style direct estimation of maternal mortality from household deaths.

Reproduces the method in Chapter 7 of the 2024 Nigeria VASA report:

* Woman-years of exposure by 5-year age group over the 60 months before the
  household interview, from the NDHS household roster (de jure women) plus the
  deceased women themselves up to their month of death.
* Deaths by age group from the VASA file (NDHS-frame deaths only, since the
  screened households have no exposure denominators).
* Age-specific rates standardised to the age distribution of interviewed women
  (removes truncation bias at ages 15 and 49).
* General fertility rate from NDHS birth histories over the same window.
* MMR = standardised maternal mortality rate / standardised GFR x 100,000.
* Jackknife (leave-one-cluster-out) confidence intervals.

All NDHS inputs are read from IHME's J: drive and never modified.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

NDHS_DIR = Path(os.environ.get("NDHS_DIR", "/home/j/DATA/DHS_PROG_DHS/NGA/2023_2024"))
PR_FILE = "NGA_DHS8_2023_2024_HHM_NGPR8AFL_Y2025M10D30.DTA"  # household members
IR_FILE = "NGA_DHS8_2023_2024_WN_NGIR8AFL_Y2025M10D30.DTA"  # interviewed women

AGE_GROUPS = [15, 20, 25, 30, 35, 40, 45]  # lower bounds, 15-19 ... 45-49
AGE_LABELS = [f"{a}-{a + 4}" for a in AGE_GROUPS]
WINDOW_MONTHS = 60  # months 1..60 before the interview month


def _read(path, usecols):
    try:
        df, _ = pyreadstat.read_dta(str(path), usecols=usecols, apply_value_formats=False)
    except UnicodeDecodeError:
        df, _ = pyreadstat.read_dta(
            str(path), usecols=usecols, apply_value_formats=False, encoding="LATIN1"
        )
    return df


def _age_group(age_exact: np.ndarray) -> np.ndarray:
    """Map exact ages to 5-year group lower bounds; NaN outside 15-49."""
    g = np.floor(age_exact / 5) * 5
    g = np.where((age_exact >= 15) & (age_exact < 50), g, np.nan)
    return g


def _accumulate(df_long: pd.DataFrame, value: str) -> pd.DataFrame:
    """Sum `value` by cluster x age group -> wide table (clusters x AGE_GROUPS)."""
    t = df_long.pivot_table(index="cluster", columns="age_group", values=value, aggfunc="sum", fill_value=0.0)
    return t.reindex(columns=AGE_GROUPS, fill_value=0.0)


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

def living_women_exposure(pr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Woman-years by cluster x age group contributed by women alive at interview.

    De jure female household members aged 15-54 at interview. For each of the
    60 months before the interview month, age is taken as completed age + 0.5
    minus elapsed months (birthdays assumed uniformly spread within the year).
    Weighted by the household weight hv005.
    """
    if pr is None:
        pr = _read(NDHS_DIR / PR_FILE, ["hv001", "hv005", "hv008", "hv102", "hv104", "hv105"])
    w = pr[(pr.hv104 == 2) & (pr.hv102 == 1) & pr.hv105.between(15, 54)].copy()
    wt = (w.hv005 / 1e6).to_numpy()
    m = np.arange(1, WINDOW_MONTHS + 1)
    age = w.hv105.to_numpy()[:, None] + 0.5 - m[None, :] / 12
    grp = _age_group(age)
    rows = []
    for a in AGE_GROUPS:
        months = (grp == a).sum(axis=1)
        rows.append(pd.DataFrame({"cluster": w.hv001.to_numpy(), "age_group": a, "years": wt * months / 12}))
    return _accumulate(pd.concat(rows), "years")


def deceased_women_exposure(deaths: pd.DataFrame, weight: str = "va005w") -> pd.DataFrame:
    """Woman-years contributed by women who died during the window, from the
    month after the window opened up to their month of death.

    `deaths` needs columns cluster, age_at_death, months_since_death, and the
    weight column.
    """
    d = deaths.dropna(subset=["age_at_death", "months_since_death"]).copy()
    d = d[d.months_since_death.between(0, WINDOW_MONTHS - 1)]
    m = np.arange(1, WINDOW_MONTHS + 1)
    s = d.months_since_death.to_numpy()[:, None]
    alive = m[None, :] > s  # months before interview that precede the death
    age = d.age_at_death.to_numpy()[:, None] + 0.5 - (m[None, :] - s) / 12
    grp = np.where(alive, _age_group(age), np.nan)
    wt = d[weight].to_numpy()
    rows = []
    for a in AGE_GROUPS:
        months = (grp == a).sum(axis=1)
        rows.append(pd.DataFrame({"cluster": d.cluster.to_numpy(), "age_group": a, "years": wt * months / 12}))
    return _accumulate(pd.concat(rows), "years")


# ---------------------------------------------------------------------------
# Fertility
# ---------------------------------------------------------------------------

IR_FERTILITY_COLS = ["v001", "v005", "v008", "v011", "v012"] + [f"b3_{i:02d}" for i in range(1, 21)]
SIBLING_ITEMS = (1, 2, 4, 8, 9, 12, 16)
IR_SIBLING_COLS = [f"mm{k}_{i:02d}" for i in range(1, 21) for k in SIBLING_ITEMS]


IR_FILE_2018 = Path(os.environ.get("NDHS_DIR_2018", "/home/j/DATA/DHS_PROG_DHS/NGA/2018")) / \
    "NGA_DHS7_2018_WN_NGIR7AFL_Y2019M11D05.DTA"  # used to validate the sibling method


def read_ir(sibling: bool = False, path: str | Path | None = None) -> pd.DataFrame:
    """Women's recode with the columns needed for fertility (and, optionally,
    the sibling module). Defaults to the 2024 NDHS; pass `path` for another
    round with standard DHS variable names (e.g. IR_FILE_2018)."""
    cols = IR_FERTILITY_COLS + (IR_SIBLING_COLS if sibling else [])
    return _read(Path(path) if path else NDHS_DIR / IR_FILE, cols)


def fertility(ir: pd.DataFrame | None = None, window_months: int = WINDOW_MONTHS):
    """Births and woman-years by cluster x age group from interviewed women's
    birth histories over the `window_months` before interview, plus the
    weighted age distribution of respondents used for standardisation.

    Returns (births, exposure, standard_age_distribution).
    """
    if ir is None:
        ir = read_ir()
    b3 = [c for c in ir.columns if c.startswith("b3_")]
    wt = (ir.v005 / 1e6).to_numpy()
    m = np.arange(1, window_months + 1)
    # exposure: exact age at each month before interview from CMC of birth
    age = (ir.v008.to_numpy()[:, None] - m[None, :] - ir.v011.to_numpy()[:, None]) / 12
    grp = _age_group(age)
    rows = []
    for a in AGE_GROUPS:
        months = (grp == a).sum(axis=1)
        rows.append(pd.DataFrame({"cluster": ir.v001.to_numpy(), "age_group": a, "years": wt * months / 12}))
    exposure = _accumulate(pd.concat(rows), "years")
    # births in the window by mother's age at birth
    bl = ir[["v001", "v005", "v008", "v011"] + b3].melt(
        id_vars=["v001", "v005", "v008", "v011"], value_name="b3"
    ).dropna(subset=["b3"])
    months_before = bl.v008 - bl.b3
    bl = bl[(months_before >= 1) & (months_before <= window_months)].copy()
    bl["age_group"] = _age_group(((bl.b3 - bl.v011) / 12).to_numpy())
    bl = bl.dropna(subset=["age_group"])
    bl["n"] = bl.v005 / 1e6
    births = _accumulate(bl.rename(columns={"v001": "cluster"}), "n")
    # standard population: respondents 15-49 at interview
    resp = ir[ir.v012.between(15, 49)]
    std = resp.groupby(np.floor(resp.v012 / 5) * 5)["v005"].sum()
    std = (std / std.sum()).reindex(AGE_GROUPS, fill_value=0.0)
    return births, exposure, std


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def standardised_rate(num: pd.Series, den: pd.Series, std: pd.Series) -> float:
    """Sum over age groups of std weight x (numerator / denominator)."""
    return float((std * (num / den)).sum())


def estimate(deaths_by_age: dict[str, pd.DataFrame], exposure: pd.DataFrame, births: pd.DataFrame,
             fert_exposure: pd.DataFrame, std: pd.Series, per: int = 100_000) -> pd.Series:
    """Point estimates from cluster x age tables. `deaths_by_age` maps a label
    (e.g. 'maternal') to a cluster x age table of weighted deaths."""
    E = exposure.sum()
    F = fert_exposure.sum()
    B = births.sum()
    gfr = standardised_rate(B, F, std)  # births per woman-year
    out = {"exposure_years": E.sum(), "gfr_per_1000": gfr * 1000,
           "tfr": float((B / F).sum() * 5)}
    for k, D in deaths_by_age.items():
        rate = standardised_rate(D.sum(), E, std)
        out[f"{k}_deaths"] = D.sum().sum()
        out[f"{k}_rate_per_1000"] = rate * 1000
        out[f"{k}_ratio_per_{per}"] = rate / gfr * per
    return pd.Series(out)


def jackknife(deaths_by_age: dict[str, pd.DataFrame], exposure: pd.DataFrame, births: pd.DataFrame,
              fert_exposure: pd.DataFrame, std: pd.Series, keys=("maternal",), per: int = 100_000,
              z: float = 1.96) -> pd.DataFrame:
    """Leave-one-cluster-out jackknife CI for the ratio(s) named in `keys`."""
    clusters = sorted(set(exposure.index) | set(births.index) | set(fert_exposure.index)
                      | set().union(*[set(d.index) for d in deaths_by_age.values()]))
    E = exposure.reindex(clusters, fill_value=0.0)
    B = births.reindex(clusters, fill_value=0.0)
    F = fert_exposure.reindex(clusters, fill_value=0.0)
    Ds = {k: d.reindex(clusters, fill_value=0.0) for k, d in deaths_by_age.items()}
    Et, Bt, Ft = E.sum().to_numpy(), B.sum().to_numpy(), F.sum().to_numpy()
    Dt = {k: d.sum().to_numpy() for k, d in Ds.items()}
    s = std.to_numpy()
    n = len(clusters)
    res = {}
    for k in keys:
        full = (s * (Dt[k] / Et)).sum() / (s * (Bt / Ft)).sum() * per
        loo = np.empty(n)
        for i in range(n):
            e = Et - E.iloc[i].to_numpy(); b = Bt - B.iloc[i].to_numpy(); f = Ft - F.iloc[i].to_numpy()
            d = Dt[k] - Ds[k].iloc[i].to_numpy()
            loo[i] = (s * (d / e)).sum() / (s * (b / f)).sum() * per
        se = np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
        res[k] = {"estimate": full, "se": se, "lower": full - z * se, "upper": full + z * se, "clusters": n}
    return pd.DataFrame(res).T


# ---------------------------------------------------------------------------
# Sibling survival (direct sisterhood) method
# ---------------------------------------------------------------------------

PREG_RELATED_CODES = (2, 3, 4, 5, 6)  # mm9: while pregnant, during delivery, within 2 months
PREG_42DAY_CODES = (2, 3, 4, 5)  # drops code 6 = "2 months after delivery" (43-60 days)


def sibling_tables(ir: pd.DataFrame, window_months: int = 84):
    """Sister exposure and deaths by cluster x age group from the NDHS maternal
    mortality (sibling history) module, DHS-style.

    Exposure: every reported sister contributes woman-years at ages 15-49 for
    each of the `window_months` before the respondent's interview, up to the
    month before her death if she died. Weighted by the respondent's v005.

    Deaths: sisters who died in the window aged 15-49. Flags follow the DHS
    Guide to Statistics: pregnancy-related = mm9 in 2-6 (during pregnancy,
    delivery, or within two months); a 42-day variant drops code 6; the
    "maternal" variant used in the 2018 NDHS excludes deaths from violence or
    accidents (mm16 in 1, 2).

    Returns (exposure, deaths_dict, sisters_long_dataframe_of_deaths).
    """
    stubs = [f"mm{k}_" for k in SIBLING_ITEMS]
    keep = ["v001", "v005", "v008", "v011", "v012"] + [c for c in ir.columns if c.startswith("mm")]
    long = pd.wide_to_long(ir[keep].reset_index(), stubnames=stubs, i="index", j="sib", suffix=r"\d+").reset_index()
    long.columns = [c.rstrip("_") for c in long.columns]
    sis = long[(long.mm1 == 2) & long.mm4.notna() & long.mm2.isin([0, 1])].copy()
    sis = sis.rename(columns={"v001": "cluster"})
    wt = (sis.v005 / 1e6).to_numpy()
    m = np.arange(1, window_months + 1)
    cmc = sis.v008.to_numpy()[:, None] - m[None, :]
    age = (cmc - sis.mm4.to_numpy()[:, None]) / 12
    death_cmc = np.where(sis.mm2.to_numpy() == 0, sis.mm8.to_numpy(), np.inf)
    alive = cmc < death_cmc[:, None]
    grp = np.where(alive, _age_group(age), np.nan)
    rows = []
    for a in AGE_GROUPS:
        months = (grp == a).sum(axis=1)
        rows.append(pd.DataFrame({"cluster": sis.cluster.to_numpy(), "age_group": a, "years": wt * months / 12}))
    exposure = _accumulate(pd.concat(rows), "years")

    d = sis[sis.mm2 == 0].copy()
    d["months_since_death"] = d.v008 - d.mm8
    d["age_at_death"] = (d.mm8 - d.mm4) / 12
    d = d[d.months_since_death.between(1, window_months) & (d.age_at_death >= 15) & (d.age_at_death < 50)].copy()
    d["w"] = d.v005 / 1e6
    d["preg_related"] = d.mm9.isin(PREG_RELATED_CODES)
    d["preg_42d"] = d.mm9.isin(PREG_42DAY_CODES)
    d["maternal_excl_ext"] = d.preg_related & ~d.mm16.isin([1, 2])
    # NDHS 2018 "maternal death": within 42 days AND not due to violence/accident
    d["maternal_42d_excl_ext"] = d.preg_42d & ~d.mm16.isin([1, 2])
    deaths = {"all": deaths_table(d, pd.Series(True, index=d.index), "w")}
    for k in ["preg_related", "preg_42d", "maternal_excl_ext", "maternal_42d_excl_ext"]:
        deaths[k] = deaths_table(d, d[k], "w")
    return exposure, deaths, d


def deaths_table(deaths: pd.DataFrame, flag: pd.Series, weight: str = "va005w") -> pd.DataFrame:
    """Cluster x age table of weighted deaths where `flag` is True."""
    d = deaths[flag].copy()
    d["age_group"] = _age_group(d.age_at_death.to_numpy() + 0.0)
    d = d.dropna(subset=["age_group"])
    return _accumulate(d.assign(n=d[weight]), "n")
