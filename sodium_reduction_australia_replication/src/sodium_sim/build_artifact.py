"""Build the vivarium artifact for Australia from the GBD 2023 CSV extracts.

Run as ``python -m sodium_sim.build_artifact``. Reads ``data/gbd/*.csv`` (pulled
from the IHME GBD 2023 results via the IHME_GBD MCP server), derives the
quantities the vivarium components expect, and writes ``artifacts/australia.hdf``.

Derivations, all by sex and GBD age band:

- excess mortality rate = cause-specific mortality rate / prevalence
- disability weight = YLD rate / prevalence (the average weight of a prevalent case)
- incidence per susceptible person = incidence per capita / (1 - prevalence)
- background disability (all causes not modeled) = all-cause YLD rate - sum of
  the modeled causes' YLD rates, saved for post-processing rather than simulated
- relative risks per 10 mmHg SBP (per 1 g/day sodium for stomach cancer) are
  backed out from the GBD population attributable fraction and the exposure
  distribution, so that the model reproduces GBD's attributable burden by
  construction (see ``solve_rr_from_paf``)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from vivarium.artifact import Artifact

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "gbd"
ARTIFACT_PATH = ROOT / "artifacts" / "australia.hdf"

LOCATION = "Australia"
YEAR_START, YEAR_END = 2020, 2050
SEXES = ["Male", "Female"]

CAUSES = [
    "ischemic_heart_disease",
    "ischemic_stroke",
    "intracerebral_hemorrhage",
    "subarachnoid_hemorrhage",
    "hypertensive_heart_disease",
    "atrial_fibrillation_and_flutter",
    "peripheral_arterial_disease",
    "ckd_due_to_diabetes_type_2",
    "ckd_due_to_hypertension",
    "ckd_due_to_glomerulonephritis",
    "ckd_due_to_other_causes",
    "stomach_cancer",
]
# GBD 2023 estimates aortic aneurysm as a fatal-only cause (no prevalence or YLDs),
# so it enters the model through the mortality component's "unmodeled causes"
# pipeline, which the SBP risk effect modifies directly.
MORTALITY_ONLY_CAUSES = ["aortic_aneurysm"]
UNMODELED_TARGET = ("affected_unmodeled", "cause_specific_mortality_rate")
# GBD 2023 publishes no incidence for hypertensive heart disease; it is derived
# from the prevalence age pattern and excess mortality (see incidence_from_prevalence).
DERIVED_INCIDENCE_CAUSES = ["hypertensive_heart_disease"]
SBP_CAUSES = [c for c in CAUSES if c != "stomach_cancer"]

SBP = "high_systolic_blood_pressure"
SODIUM = "diet_high_in_sodium"
SBP_TMREL = (110.0, 115.0)
SBP_SCALE = 10.0
SODIUM_TMREL = (1.0, 5.0)
SODIUM_SCALE = 1.0
MAX_PAF = 0.9
MAX_LOG_RR_PER_UNIT = {SBP: np.log(4.0), SODIUM: np.log(1.6)}

# Assumed within-age-band standard deviations (GBD publishes only the mean here).
SBP_SD_BY_AGE = {25: 11.0, 35: 12.0, 45: 14.0, 55: 16.0, 65: 18.0, 75: 20.0}
SODIUM_CV = 0.30

AGE_BAND_EDGES = {
    "Early Neonatal": (0.0, 7 / 365.25),
    "Late Neonatal": (7 / 365.25, 28 / 365.25),
    "1-5 months": (28 / 365.25, 0.5),
    "6-11 months": (0.5, 1.0),
    "12 to 23 months": (1.0, 2.0),
    "2 to 4": (2.0, 5.0),
    "95 plus": (95.0, 125.0),
}


def age_edges(name: str) -> tuple[float, float]:
    if name in AGE_BAND_EDGES:
        return AGE_BAND_EDGES[name]
    lo, hi = name.split(" to ")
    return float(lo), float(hi) + 1


def age_bins() -> pd.DataFrame:
    pop = pd.read_csv(DATA / "population_by_age.csv")
    rows = [(*age_edges(n), n) for n in pop["age_group"]]
    return pd.DataFrame(rows, columns=["age_start", "age_end", "age_group_name"])


def long_by_sex(df: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Reshape a wide GBD table (``<stem>_male``, ``<stem>_female``) to long form by age and sex."""
    out = []
    for sex in SEXES:
        part = pd.DataFrame(
            {
                "age_group": df["age_group"],
                "sex": sex,
                "value": df[f"{stem}_{sex.lower()}"].to_numpy(dtype=float),
            }
        )
        out.append(part)
    long = pd.concat(out, ignore_index=True)
    edges = long["age_group"].map(age_edges)
    long["age_start"] = edges.str[0]
    long["age_end"] = edges.str[1]
    long["year_start"] = YEAR_START
    long["year_end"] = YEAR_END
    return long[["sex", "age_start", "age_end", "year_start", "year_end", "value"]]


def full_grid(long: pd.DataFrame, fill: float = 0.0) -> pd.DataFrame:
    """Expand a long table to every age band, filling missing bands (e.g. under 25)."""
    bins = age_bins()
    grid = pd.MultiIndex.from_product(
        [SEXES, bins["age_start"]], names=["sex", "age_start"]
    ).to_frame(index=False)
    grid = grid.merge(bins[["age_start", "age_end"]], on="age_start")
    grid["year_start"] = YEAR_START
    grid["year_end"] = YEAR_END
    merged = grid.merge(
        long[["sex", "age_start", "value"]], on=["sex", "age_start"], how="left"
    )
    merged["value"] = merged["value"].fillna(fill)
    return merged[["sex", "age_start", "age_end", "year_start", "year_end", "value"]]


def rate_table(cause: str, measure: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{cause}_{measure}.csv")
    long = long_by_sex(df, "rate")
    long["value"] = long["value"] / 1e5
    return full_grid(long)


def incidence_from_prevalence(prev: pd.DataFrame, emr: pd.DataFrame) -> pd.DataFrame:
    """Derive incidence per susceptible from a prevalence age pattern with no remission.

    In a stationary population without remission, dP/da = I (1 - P) - P (EMR), so
    I = (dP/da + P EMR) / (1 - P). The age derivative is taken between band
    midpoints; negative values are floored at zero.
    """
    out = prev.copy()
    out["value"] = 0.0
    for sex in SEXES:
        m = (prev["sex"] == sex).to_numpy()
        p = prev.loc[m].sort_values("age_start")
        mid = (p["age_start"] + p["age_end"]).to_numpy() / 2
        pv = p["value"].to_numpy()
        dp = np.gradient(pv, mid)
        e = emr.loc[m].sort_values("age_start")["value"].to_numpy()
        inc = np.clip((dp + pv * e) / np.clip(1 - pv, 1e-9, None), 0, None)
        out.loc[p.index, "value"] = inc
    return out


def population_structure() -> pd.DataFrame:
    """Population counts by age and sex, repeated for each year (held at the 2023 structure)."""
    pop = pd.read_csv(DATA / "population_by_age.csv")
    long = long_by_sex(pop, "value")
    long["location"] = LOCATION
    yearly = []
    for year in range(YEAR_START, YEAR_END):
        part = long.copy()
        part["year_start"], part["year_end"] = year, year + 1
        yearly.append(part)
    return pd.concat(yearly, ignore_index=True)[
        ["age_start", "age_end", "sex", "year_start", "year_end", "location", "value"]]


def live_births() -> pd.DataFrame:
    births = pd.read_csv(DATA / "live_births.csv").iloc[0]
    rows = []
    for year in range(YEAR_START, YEAR_END):
        for sex in SEXES:
            rows.append((year, year + 1, sex, "mean_value", float(births[f"value_{sex.lower()}"])))
    return pd.DataFrame(rows, columns=["year_start", "year_end", "sex", "parameter", "value"])


def theoretical_minimum_risk_life_expectancy() -> pd.DataFrame:
    """A smooth stand-in for the GBD reference life table; only used for YLLs, which this analysis does not report."""
    ages = np.arange(0, 126, dtype=float)
    value = np.maximum(88.9 - ages, 1.5)
    return pd.DataFrame({"age_start": ages[:-1], "age_end": ages[1:], "value": value[:-1]})


def exposure_table(risk: str) -> pd.DataFrame:
    fname = {SBP: "exposure_sbp_by_age.csv", SODIUM: "exposure_sodium_by_age.csv"}[risk]
    df = pd.read_csv(DATA / fname)
    long = long_by_sex(df, "mean")
    fill = {SBP: 115.0, SODIUM: 3.0}[risk]
    return full_grid(long, fill=fill)


def exposure_sd_table(risk: str) -> pd.DataFrame:
    mean = exposure_table(risk)
    if risk == SBP:
        starts = np.array(sorted(SBP_SD_BY_AGE))
        idx = np.clip(np.searchsorted(starts, mean["age_start"], side="right") - 1, 0, None)
        sd = np.array([SBP_SD_BY_AGE[s] for s in starts])[idx]
        sd = np.where(mean["age_start"] < 25, 8.0, sd)
    else:
        sd = SODIUM_CV * mean["value"].to_numpy()
    out = mean.copy()
    out["value"] = sd
    return out


def expected_relative_risk(log_rr: float, mean: float, sd: float, tmrel: float, scale: float) -> float:
    """E[max(RR^((X - tmrel)/scale), 1)] for X ~ Normal(mean, sd)."""
    x = np.linspace(mean - 6 * sd, mean + 6 * sd, 4001)
    w = norm.pdf(x, mean, sd)
    w /= w.sum()
    rr = np.exp(log_rr * np.clip(x - tmrel, 0, None) / scale)
    return float((rr * w).sum())


def solve_rr_from_paf(paf: float, mean: float, sd: float, tmrel: float, scale: float, max_log_rr: float) -> float:
    """Return the per-unit log RR whose population-average RR gives this PAF, capped."""
    paf = min(max(paf, 0.0), MAX_PAF)
    if paf <= 0:
        return 0.0
    target = 1 / (1 - paf)
    f = lambda b: expected_relative_risk(b, mean, sd, tmrel, scale) - target
    if f(max_log_rr) < 0:
        return max_log_rr
    return brentq(f, 0.0, max_log_rr, xtol=1e-6)


def paf_table_from_gbd(cause: str, risk: str) -> pd.DataFrame:
    suffix = {SBP: "paf_sbp_deaths", SODIUM: "paf_sodium_deaths"}[risk]
    df = pd.read_csv(DATA / f"{cause}_{suffix}.csv")
    long = long_by_sex(df, "percent")
    long["value"] = long["value"] / 100
    return full_grid(long)


def risk_effect_tables(
    risk: str, causes: list[str], targets: dict[str, tuple[str, str]] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (relative_risk, paf, diagnostics) tables for a risk over its causes.

    ``targets`` maps a cause to the (affected_entity, affected_measure) pair the
    vivarium RiskEffect looks up; the default is the cause's incidence rate.
    """
    targets = targets or {}
    tmrel = 0.5 * sum({SBP: SBP_TMREL, SODIUM: SODIUM_TMREL}[risk])
    scale = {SBP: SBP_SCALE, SODIUM: SODIUM_SCALE}[risk]
    mean = exposure_table(risk).rename(columns={"value": "mean"})
    sd = exposure_sd_table(risk).rename(columns={"value": "sd"})
    dist = mean.merge(sd)
    rr_rows, paf_rows, diag_rows = [], [], []
    for cause in causes:
        gbd_paf = paf_table_from_gbd(cause, risk).rename(columns={"value": "gbd_paf"})
        d = dist.merge(gbd_paf)
        for _, r in d.iterrows():
            if r["age_start"] < 25:
                log_rr, paf = 0.0, 0.0
            else:
                log_rr = solve_rr_from_paf(
                    r["gbd_paf"], r["mean"], r["sd"], tmrel, scale, MAX_LOG_RR_PER_UNIT[risk]
                )
                paf = 1 - 1 / expected_relative_risk(log_rr, r["mean"], r["sd"], tmrel, scale)
            entity, measure = targets.get(cause, (cause, "incidence_rate"))
            base = dict(
                sex=r["sex"], age_start=r["age_start"], age_end=r["age_end"],
                year_start=YEAR_START, year_end=YEAR_END,
                affected_entity=entity, affected_measure=measure,
            )
            rr_rows.append({**base, "value": float(np.exp(log_rr))})
            paf_rows.append({**base, "value": float(paf)})
            diag_rows.append({**base, "gbd_paf": r["gbd_paf"], "model_paf": paf,
                              "rr_per_unit": float(np.exp(log_rr)), "exposure_mean": r["mean"]})
    return pd.DataFrame(rr_rows), pd.DataFrame(paf_rows), pd.DataFrame(diag_rows)


def background_disability() -> pd.DataFrame:
    """All-cause YLD rate minus the modeled causes' YLD rates, per person-year."""
    all_cause = rate_table("all_causes", "ylds").rename(columns={"value": "all_causes"})
    modeled = sum(rate_table(c, "ylds")["value"] for c in CAUSES)  # mortality-only causes have no YLDs
    out = all_cause.copy()
    out["modeled"] = modeled.to_numpy()
    out["value"] = (out["all_causes"] - out["modeled"]).clip(lower=0)
    return out


def write(art: Artifact, key: str, data) -> None:
    """Write to the artifact, indexing DataFrames by every non-value column.

    The simulation loads artifact tables with a draw filter that keeps only the
    ``value`` column, so demographic columns survive only if they are in the index.
    """
    if isinstance(data, pd.DataFrame):
        index_cols = [c for c in data.columns if c != "value"]
        data = data.set_index(index_cols)
    art.write(key, data)


def build(path: Path = ARTIFACT_PATH) -> Artifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    art = Artifact(str(path))

    bins = age_bins()
    structure = population_structure()
    write(art, "population.structure", structure)
    write(art, "population.age_bins", bins)
    write(art, "population.location", LOCATION)
    write(art, "population.demographic_dimensions", structure.drop(columns=["location", "value"]))
    write(art, "population.theoretical_minimum_risk_life_expectancy", theoretical_minimum_risk_life_expectancy())
    write(art, "covariate.live_births_by_sex.estimate", live_births())
    write(art, "cause.all_causes.cause_specific_mortality_rate", rate_table("all_causes", "deaths"))
    write(art, "cause.all_causes.yld_rate", rate_table("all_causes", "ylds"))
    write(art, "cause.background.disability_weight", background_disability()[
        ["sex", "age_start", "age_end", "year_start", "year_end", "value"]])

    summary = []
    for cause in CAUSES:
        prev = rate_table(cause, "prevalence")
        csmr = rate_table(cause, "deaths")
        ylds = rate_table(cause, "ylds")
        safe_prev = prev["value"].where(prev["value"] > 0, np.nan)
        emr = csmr.copy()
        emr["value"] = (csmr["value"] / safe_prev).fillna(0.0)
        if cause in DERIVED_INCIDENCE_CAUSES:
            inc = incidence_from_prevalence(prev, emr)
            inc["value"] = inc["value"] * (1 - prev["value"])
        else:
            inc = rate_table(cause, "incidence")
        dw = ylds.copy()
        dw["value"] = (ylds["value"] / safe_prev).fillna(0.0).clip(upper=1.0)
        inc_sus = inc.copy()
        inc_sus["value"] = inc["value"] / (1 - prev["value"])
        write(art, f"cause.{cause}.prevalence", prev)
        write(art, f"cause.{cause}.incidence_rate", inc_sus)
        write(art, f"cause.{cause}.cause_specific_mortality_rate", csmr)
        write(art, f"cause.{cause}.excess_mortality_rate", emr)
        write(art, f"cause.{cause}.disability_weight", dw)
        write(art, f"cause.{cause}.yld_rate", ylds)
        write(art, f"cause.{cause}.restrictions", {"yld_only": False})
        summary.append((cause, prev["value"].max(), inc["value"].max(), csmr["value"].max(),
                        emr["value"].max(), dw["value"].max()))

    for cause in MORTALITY_ONLY_CAUSES:
        write(art, f"cause.{cause}.cause_specific_mortality_rate", rate_table(cause, "deaths"))

    unmodeled_targets = {c: UNMODELED_TARGET for c in MORTALITY_ONLY_CAUSES}
    for risk, causes, tmrel in [
        (SBP, SBP_CAUSES + MORTALITY_ONLY_CAUSES, SBP_TMREL),
        (SODIUM, ["stomach_cancer"], SODIUM_TMREL),
    ]:
        rr, paf, diag = risk_effect_tables(risk, causes, unmodeled_targets)
        write(art, f"risk_factor.{risk}.distribution", "normal")
        write(art, f"risk_factor.{risk}.exposure", exposure_table(risk))
        write(art, f"risk_factor.{risk}.exposure_standard_deviation", exposure_sd_table(risk))
        write(art, f"risk_factor.{risk}.tmred", {"distribution": "uniform", "min": tmrel[0], "max": tmrel[1], "inverted": False})
        write(art, f"risk_factor.{risk}.relative_risk_scalar", {SBP: SBP_SCALE, SODIUM: SODIUM_SCALE}[risk])
        write(art, f"risk_factor.{risk}.relative_risk", rr)
        write(art, f"risk_factor.{risk}.population_attributable_fraction", paf)
        diag.to_csv(ROOT / "results" / f"risk_effect_diagnostics_{risk}.csv", index=False)

    print(pd.DataFrame(summary, columns=["cause", "max_prev", "max_inc", "max_csmr", "max_emr", "max_dw"]).to_string())
    print(f"wrote {path} with {len(art.keys)} keys")
    return art


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ARTIFACT_PATH)
