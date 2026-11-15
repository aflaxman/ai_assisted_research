"""Inputs that do not come from GBD: published tables and other agencies' data.

Each function reads one small file under ``data/`` and returns it in the shape
the loader wants.  The files and where they came from:

``data/literature/sbp_relative_risk_per_10mmhg.csv``
    Relative risk per 10 mmHg of systolic blood pressure by age, from eTable 4
    of the web appendix to Marklund et al. (BMJ 2020;369:m824), the model this
    paper adapts.  The paper cites "GBD 2019 relative risks", which sit behind
    a login; these are the same log-linear, age-specific risks as an earlier
    GBD round published them.
``data/literature/ckd_cvd_mortality_hazard_ratio_by_age.csv``
    Hazard ratio of cardiovascular death with chronic kidney disease, by age
    (eTable 1 of the same appendix, interpolated there from Hallan et al. 2012).
``data/literature/serum_potassium_cvd_mortality_hazard_ratio.csv``
    Hazard ratio of cardiovascular death by serum potassium against 4.2 mmol/L,
    digitised from Figure 3C of Kovesdy et al. (Eur Heart J 2018;39:1535).  The
    paper had the numbers from the consortium; the one value in print, 1.22
    (1.04 to 1.42) at 5.5 mmol/L, matches the digitised curve.
``data/literature/bhat_2020_discretionary_salt_by_country.csv``
    Reported share of salt from discretionary sources for 24 countries, from
    Supplemental Table 3 of Bhat et al. (Adv Nutr 2020;11:677).
``data/ncdrisc/ncdrisc_hypertension_2019_age_specific.csv``
    NCD-RisC (Lancet 2021;398:957) hypertension prevalence, diagnosis, and
    treatment in 2019 for ages 30 to 79.
``data/worldbank/gdp_per_capita_2021.csv``
    World Bank GDP per capita, with IMF figures where the Bank has none.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import special

from vivarium_potassium_salt.constants import data_values, metadata, paths
from vivarium_potassium_salt.data import gbd_extract

# Which column of the relative-risk table each modelled cause reads.
RELATIVE_RISK_GROUP = {
    "ischemic_heart_disease": "ischemic_heart_disease",
    "ischemic_stroke": "ischemic_stroke",
    "intracerebral_hemorrhage": "hemorrhagic_stroke",
    "subarachnoid_hemorrhage": "hemorrhagic_stroke",
    "hypertensive_heart_disease": "hypertensive_heart_disease",
    "aortic_aneurysm": "aortic_aneurysm",
    "rheumatic_heart_disease": "rheumatic_heart_disease",
    "endocarditis": "endocarditis",
    "non_rheumatic_calcific_aortic_valve_disease": "other_cardiovascular_diseases",
    "other_cardiomyopathy": "other_cardiovascular_diseases",
    "atrial_fibrillation_and_flutter": "other_cardiovascular_diseases",
    "lower_extremity_peripheral_arterial_disease": "other_cardiovascular_diseases",
}
# Marklund et al. (2020), Table 1: one value for all ages.
CKD_RELATIVE_RISK = (1.28, 1.18, 1.39)

# Names in Bhat et al. that differ from GBD location names.
_BHAT_TO_GBD = {
    "Brunei": "Brunei Darussalam", "South Korea": "Republic of Korea", "Turkey": "Türkiye",
    "UK": "United Kingdom", "USA": "United States of America",
}


def _expand_terminal_age_band(table: pd.DataFrame, column: str = "age_band") -> pd.DataFrame:
    """Repeat the '80 plus' rows for each five-year band from 80 upward."""
    terminal = [name for name in metadata.AGE_BAND_NAMES if int(name[:2]) >= 80]
    rest = table[table[column] != "80 plus"]
    last = table[table[column] == "80 plus"]
    repeated = [last.assign(**{column: name}) for name in terminal]
    return pd.concat([rest] + repeated, ignore_index=True)


@lru_cache
def sbp_relative_risks() -> pd.DataFrame:
    """Relative risk per 10 mmHg for every modelled cause and age band.

    Columns ``cause, age_group, mean, lower, upper``.  The source table stops
    at "80 and over", and its last value is carried to every older band.
    """
    table = pd.read_csv(paths.LITERATURE_DIR / "sbp_relative_risk_per_10mmhg.csv")
    table = _expand_terminal_age_band(table).rename(columns={"age_band": "age_group"})
    parts = [
        table[table["outcome_group"] == group].assign(cause=cause)
        for cause, group in RELATIVE_RISK_GROUP.items()
    ]
    mean, lower, upper = CKD_RELATIVE_RISK
    parts.append(pd.DataFrame({
        "age_group": metadata.AGE_BAND_NAMES, "mean": mean, "lower": lower, "upper": upper,
        "cause": metadata.CKD_CAUSE,
    }))
    risks = pd.concat(parts, ignore_index=True)[["cause", "age_group", "mean", "lower", "upper"]]
    assert set(risks["cause"]) == set(metadata.CAUSES)
    assert len(risks) == len(metadata.CAUSES) * len(metadata.AGE_BANDS)
    return risks


@lru_cache
def _gdp_table() -> pd.DataFrame:
    return pd.read_csv(paths.WORLD_BANK_DIR / "gdp_per_capita_2021.csv").set_index("gbd_name")


def gdp_per_capita_2010_usd(location: str) -> float:
    """2021 GDP per capita, deflated to 2010 dollars the way the paper does."""
    current = float(_gdp_table().loc[location, "gdp_per_capita_current_usd"])
    return current / data_values.GDP_DEFLATOR_2021_TO_2010


def is_high_income(location: str) -> bool:
    current = float(_gdp_table().loc[location, "gdp_per_capita_current_usd"])
    return current > data_values.HIGH_INCOME_THRESHOLD_USD


@lru_cache
def reported_discretionary_shares() -> dict[str, float]:
    """Reported discretionary share (0 to 1) for the 24 countries with a survey."""
    table = pd.read_csv(paths.LITERATURE_DIR / "bhat_2020_discretionary_salt_by_country.csv")
    names = table["country"].map(lambda name: _BHAT_TO_GBD.get(name, name))
    known = set(gbd_extract.locations()["gbd_name"])
    assert set(names) <= known, sorted(set(names) - known)
    return dict(zip(names, table["discretionary_percent"] / 100))


@lru_cache
def hypertension(location: str) -> pd.DataFrame:
    """Hypertension prevalence, and the shares diagnosed and treated, by sex and age.

    NCD-RisC covers ages 30 to 79.  As in the paper, ages 25 to 29 take the
    values for 30 to 34, and ages 80 and over take those for 75 to 79.

    Returns
    -------
        Columns ``sex, age_group`` and, for each of ``prevalence``,
        ``diagnosed``, and ``treated``, the estimate with ``_lower`` and
        ``_upper``.  ``diagnosed`` and ``treated`` are proportions of people
        with hypertension.
    """
    table = pd.read_csv(paths.NCDRISC_DIR / "ncdrisc_hypertension_2019_age_specific.csv")
    iso3 = gbd_extract.locations().set_index("gbd_name").loc[location, "iso3"]
    table = table[table["iso3"] == iso3]
    if table.empty:
        raise KeyError(f"NCD-RisC has no hypertension estimates for {location!r}")
    table = table.rename(columns={
        "htn_prev": "prevalence", "htn_prev_lower": "prevalence_lower",
        "htn_prev_upper": "prevalence_upper",
    })
    table["age_group"] = table["age"].str.replace("-", " to ")
    columns = [f"{name}{suffix}" for name in ("prevalence", "diagnosed", "treated")
               for suffix in ("", "_lower", "_upper")]
    table = table[["sex", "age_group"] + columns]

    def borrowed(source: str, targets: list[str]) -> list[pd.DataFrame]:
        rows = table[table["age_group"] == source]
        return [rows.assign(age_group=target) for target in targets]

    oldest = [name for name in metadata.AGE_BAND_NAMES if int(name[:2]) >= 80]
    table = pd.concat(
        [table] + borrowed("30 to 34", ["25 to 29"]) + borrowed("75 to 79", oldest),
        ignore_index=True,
    )
    assert len(table) == len(metadata.SEXES) * len(metadata.AGE_BANDS)
    return table.sort_values(["sex", "age_group"]).reset_index(drop=True)


@lru_cache
def ckd_cvd_hazard_ratio_by_age() -> pd.DataFrame:
    """Hazard ratio of cardiovascular death with chronic kidney disease, by age band."""
    table = pd.read_csv(paths.LITERATURE_DIR / "ckd_cvd_mortality_hazard_ratio_by_age.csv")
    table = _expand_terminal_age_band(table).rename(columns={"age_band": "age_group"})
    under_55 = table["age_group"].str[:2].astype(int) < 55
    table["log_sd"] = np.where(
        under_55, data_values.CKD_HAZARD_RATIO_LOG_SD["under_55"],
        data_values.CKD_HAZARD_RATIO_LOG_SD["55_plus"],
    )
    return table


@lru_cache
def serum_potassium_hazard_ratio() -> pd.DataFrame:
    """Hazard ratio of cardiovascular death by serum potassium, 4.2 to 6.5 mmol/L."""
    table = pd.read_csv(paths.LITERATURE_DIR / "serum_potassium_cvd_mortality_hazard_ratio.csv")
    table = table[table["serum_potassium_mmol_per_l"] >= 4.2].reset_index(drop=True)
    return table


def hyperkalaemia_relative_risk(serum_potassium, hazard_z=0.0):
    """Relative risk of cardiovascular death counted as due to hyperkalaemia.

    One below the paper's threshold of 5.5 mmol/L; above it, the consortium's
    hazard ratio against 4.2 mmol/L, held at its last value beyond 6.5 mmol/L.
    ``hazard_z`` moves the whole curve along its confidence band, for draws; it
    may be an array that broadcasts against ``serum_potassium``.
    """
    curve = serum_potassium_hazard_ratio()
    levels = curve["serum_potassium_mmol_per_l"].to_numpy()
    log_mean = np.log(curve["hazard_ratio"].to_numpy())
    log_sd = (np.log(curve["upper"].to_numpy()) - np.log(curve["lower"].to_numpy())) / (2 * 1.96)
    serum_potassium = np.asarray(serum_potassium, dtype=float)
    risk = np.exp(
        np.interp(serum_potassium, levels, log_mean)
        + hazard_z * np.interp(serum_potassium, levels, log_sd)
    )
    return np.where(serum_potassium >= data_values.HYPERKALAEMIA_THRESHOLD_MMOL_PER_L,
                    np.maximum(risk, 1.0), 1.0)


def expected_hyperkalaemia_relative_risk(mean, sd, hazard_z=0.0):
    """Mean of ``hyperkalaemia_relative_risk`` over a normal distribution of serum potassium.

    The log hazard ratio is linear between its tabulated points, so on each
    interval ``[u, v]`` between them, with ``log HR = a + b k``,

        integral of exp(a + b k) dN(k; m, s) =
            exp(a + b m + b^2 s^2 / 2) [Phi((v - m - b s^2) / s) - Phi((u - m - b s^2) / s)],

    and the mean is a sum of such terms.  This is exact, which matters here:
    the risk jumps from 1 to about 1.2 at 5.5 mmol/L, and at stage G3a only one
    person in two hundred is above it, so averaging over a few hundred sampled
    or evenly spaced potassium levels gets that tail badly wrong.

    The arguments broadcast against each other.  Where ``sd`` is zero the
    result is the risk at ``mean``.
    """
    mean, sd, hazard_z = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(sd, dtype=float),
        np.asarray(hazard_z, dtype=float))
    curve = serum_potassium_hazard_ratio()
    threshold = data_values.HYPERKALAEMIA_THRESHOLD_MMOL_PER_L
    keep = curve["serum_potassium_mmol_per_l"].to_numpy() >= threshold - 1e-9
    levels = curve["serum_potassium_mmol_per_l"].to_numpy()[keep]
    assert abs(levels[0] - threshold) < 1e-9, "the threshold must be a tabulated level"
    log_mean = np.log(curve["hazard_ratio"].to_numpy())[keep]
    log_sd = ((np.log(curve["upper"].to_numpy()) - np.log(curve["lower"].to_numpy()))
              / (2 * 1.96))[keep]

    spread = sd > 0
    safe_sd = np.where(spread, sd, 1.0)[..., None]
    centre = mean[..., None]
    # Log hazard ratio at each tabulated level, for each element's place in the band.
    nodes = log_mean + hazard_z[..., None] * log_sd
    left, right = levels[:-1], levels[1:]
    slope = (nodes[..., 1:] - nodes[..., :-1]) / (right - left)
    intercept = nodes[..., :-1] - slope * left
    # The part of each interval where the hazard ratio is above one (all of it,
    # unless a draw from the bottom of the confidence band dips below).
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = np.where(slope != 0, -intercept / slope, np.inf)
    crossing = np.clip(crossing, left, right)
    start = np.where(slope > 0, crossing, left)
    stop = np.where(slope < 0, crossing, right)
    flat_below = (slope == 0) & (intercept <= 0)
    stop = np.where(flat_below, start, stop)

    def mass(lower, upper, shift=0.0):
        return (special.ndtr((upper - centre - shift) / safe_sd)
                - special.ndtr((lower - centre - shift) / safe_sd))

    tilt = slope * safe_sd**2
    excess = (np.exp(intercept + slope * centre + 0.5 * slope * tilt) * mass(start, stop, tilt)
              - mass(start, stop)).sum(axis=-1)
    beyond = np.maximum(np.exp(nodes[..., -1]) - 1.0, 0.0) * special.ndtr(
        (mean - levels[-1]) / safe_sd[..., 0])
    return np.where(spread, 1.0 + excess + beyond, hyperkalaemia_relative_risk(mean, hazard_z))
