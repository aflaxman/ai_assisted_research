"""Build the second artifact: GBD 2023 via vivarium_inputs, exposures from the paper.

Run as ``python -m sodium_sim.build_artifact_v2``. Differences from the first
builder (``build_artifact.py``, which used the MCP server extracts):

- disease rates, excess mortality, disability weights and all-cause mortality
  come from ``vivarium_inputs`` pulls (GBD 2023, release 16), stored as draw
  means in ``data/gbd_vi/``;
- the SBP relative risks per 10 mmHg and the sodium relative risk per g/day
  are GBD's own, not backed out of PAFs;
- the sodium and SBP exposure distributions are the paper's Tables S1 and S2
  (SEIFA quintiles averaged), lognormal for sodium and normal for SBP;
- the population attributable fractions are recomputed from those relative
  risks and exposure distributions so the model is internally consistent.

Background disability still comes from the MCP all-cause YLD extract, since
``vivarium_inputs`` has no all-cause YLD measure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
from vivarium.artifact import Artifact

from sodium_sim.build_artifact import (
    ARTIFACT_PATH,
    CAUSES,
    DATA,
    MORTALITY_ONLY_CAUSES,
    ROOT,
    SBP,
    SBP_CAUSES,
    SEXES,
    SODIUM,
    UNMODELED_TARGET,
    YEAR_END,
    YEAR_START,
    age_bins,
    background_disability,
    full_grid,
    rate_table,
    theoretical_minimum_risk_life_expectancy,
    write,
)

DATA_VI = ROOT / "data" / "gbd_vi"
DATA_PAPER = ROOT / "data" / "paper"
ARTIFACT_V2_PATH = ROOT / "artifacts" / "australia_v2.hdf"

# vivarium_inputs cause names for the model's cause names.
VI_CAUSE = {
    "peripheral_arterial_disease": "lower_extremity_peripheral_arterial_disease",
    "ckd_due_to_diabetes_type_2": "chronic_kidney_disease_due_to_diabetes_mellitus_type_2",
    "ckd_due_to_hypertension": "chronic_kidney_disease_due_to_hypertension",
    "ckd_due_to_glomerulonephritis": "chronic_kidney_disease_due_to_glomerulonephritis",
    "ckd_due_to_other_causes": "chronic_kidney_disease_due_to_other_and_unspecified_causes",
}
SBP_TMRED = {"distribution": "uniform", "min": 105.0, "max": 115.0, "inverted": False}
SODIUM_TMRED = {"distribution": "uniform", "min": 1.0, "max": 5.0, "inverted": False}
SCALE = {SBP: 10.0, SODIUM: 1.0}
DISTRIBUTION = {SBP: "normal", SODIUM: "lognormal"}


def vi_name(cause: str) -> str:
    return VI_CAUSE.get(cause, cause)


def draw_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse draw columns to their mean in a ``value`` column."""
    draws = [c for c in df.columns if c.startswith("draw_")]
    if draws:
        df = df.drop(columns=draws).assign(value=df[draws].mean(axis=1))
    return df


def load_vi(key: str) -> pd.DataFrame:
    """Load a vivarium_inputs pull, averaged over draws, with the standard year span."""
    df = draw_mean(pd.read_csv(DATA_VI / f"{key}.csv"))
    df = df.drop(columns=[c for c in ["location"] if c in df.columns])
    df["year_start"], df["year_end"] = YEAR_START, YEAR_END
    return df


def vi_cause_table(cause: str, measure: str) -> pd.DataFrame:
    df = load_vi(f"cause.{vi_name(cause)}.{measure}")
    return df[["sex", "age_start", "age_end", "year_start", "year_end", "value"]]


def population_structure_vi() -> pd.DataFrame:
    pop = pd.read_csv(DATA_VI / "population.structure.csv")
    yearly = []
    for year in range(YEAR_START, YEAR_END):
        part = pop.copy()
        part["year_start"], part["year_end"] = year, year + 1
        yearly.append(part)
    return pd.concat(yearly, ignore_index=True)[
        ["age_start", "age_end", "sex", "year_start", "year_end", "location", "value"]]


def live_births_vi() -> pd.DataFrame:
    births = pd.read_csv(DATA_VI / "covariate.live_births_by_sex.estimate.csv")
    births = births[births["parameter"] == "mean_value"]
    rows = []
    for year in range(YEAR_START, YEAR_END):
        for sex in SEXES:
            value = float(births.loc[births["sex"] == sex, "value"].iloc[0])
            rows.append((year, year + 1, sex, "mean_value", value))
    return pd.DataFrame(rows, columns=["year_start", "year_end", "sex", "parameter", "value"])


def paper_exposure(risk: str) -> pd.DataFrame:
    """Mean and pooled SD by sex and GBD age band from the paper's Table S1 or S2.

    The five SEIFA quintiles are equal-size, so the population mean is the mean
    of the quintile means and the population variance is the mean within-quintile
    variance plus the variance of the quintile means. Bands above 85 to 89 take
    that row's values.
    """
    fname = {SBP: "table_s2_sbp.csv", SODIUM: "table_s1_sodium_intake.csv"}[risk]
    t = pd.read_csv(DATA_PAPER / fname)
    pooled = (
        t.groupby(["sex", "age_start"])
        .apply(lambda g: pd.Series({"mean": g["mean"].mean(),
                                    "sd": np.sqrt((g["sd"] ** 2).mean() + g["mean"].var(ddof=0))}),
               include_groups=False)
        .reset_index()
    )
    bins = age_bins()
    rows = []
    for sex in SEXES:
        p = pooled[pooled["sex"] == sex].sort_values("age_start")
        for _, b in bins.iterrows():
            band = p[p["age_start"] <= b["age_start"]].iloc[-1]
            rows.append((sex, b["age_start"], b["age_end"], YEAR_START, YEAR_END, band["mean"], band["sd"]))
    return pd.DataFrame(rows, columns=["sex", "age_start", "age_end", "year_start", "year_end", "mean", "sd"])


def exposure_grid(x: np.ndarray, dist: str, mean: float, sd: float) -> np.ndarray:
    if dist == "normal":
        return norm.pdf(x, mean, sd)
    s = np.sqrt(np.log(1 + (sd / mean) ** 2))
    scale = mean / np.exp(s ** 2 / 2)
    return lognorm.pdf(x, s, scale=scale)


def expected_rr(log_rr_per_unit: float, dist: str, mean: float, sd: float, tmrel: float, scale: float) -> float:
    """E[max(RR^((X - tmrel)/scale), 1)] under a normal or lognormal exposure."""
    lo = max(mean - 6 * sd, 1e-6) if dist == "lognormal" else mean - 6 * sd
    x = np.linspace(lo, mean + 8 * sd, 6001)
    w = exposure_grid(x, dist, mean, sd)
    w /= w.sum()
    rr = np.exp(log_rr_per_unit * np.clip(x - tmrel, 0, None) / scale)
    return float((rr * w).sum())


def gbd_relative_risk(risk: str) -> pd.DataFrame:
    """GBD relative risks per unit (draw mean) by sex, age, affected entity and measure."""
    rr = load_vi(f"risk.{risk}.relative_risk")
    keep = ["sex", "age_start", "age_end", "year_start", "year_end", "affected_entity", "affected_measure", "value"]
    if "parameter" in rr.columns:
        params = rr["parameter"].unique()
        if len(params) > 1:
            raise ValueError(f"Expected one RR parameter for {risk}, found {params}")
    return rr[keep]


def risk_effect_tables_v2(risk: str, causes: list[str], targets: dict[str, tuple[str, str]] | None = None):
    """Return (relative_risk, paf, diagnostics) for GBD RRs under the paper's exposure."""
    targets = targets or {}
    tmred = {SBP: SBP_TMRED, SODIUM: SODIUM_TMRED}[risk]
    tmrel = 0.5 * (tmred["min"] + tmred["max"])
    scale, dist = SCALE[risk], DISTRIBUTION[risk]
    exposure = paper_exposure(risk)
    rr_all = gbd_relative_risk(risk)
    rr_rows, paf_rows, diag_rows = [], [], []
    for cause in causes:
        rr_c = rr_all[(rr_all["affected_entity"] == vi_name(cause)) & (rr_all["affected_measure"] == "incidence_rate")]
        if rr_c.empty:
            available = sorted(rr_all["affected_entity"].unique())
            raise ValueError(f"No GBD RR for {risk} on {vi_name(cause)}; available: {available}")
        rr_c = rr_c.rename(columns={"value": "rr"})
        d = exposure.merge(rr_c[["sex", "age_start", "rr"]], on=["sex", "age_start"], how="left")
        d["rr"] = d["rr"].fillna(1.0)
        entity, measure = targets.get(cause, (cause, "incidence_rate"))
        for _, r in d.iterrows():
            rr_value = float(r["rr"]) if r["age_start"] >= 25 else 1.0
            paf = 0.0 if rr_value <= 1.0 else 1 - 1 / expected_rr(np.log(rr_value), dist, r["mean"], r["sd"], tmrel, scale)
            base = dict(sex=r["sex"], age_start=r["age_start"], age_end=r["age_end"],
                        year_start=YEAR_START, year_end=YEAR_END,
                        affected_entity=entity, affected_measure=measure)
            rr_rows.append({**base, "value": rr_value})
            paf_rows.append({**base, "value": float(paf)})
            diag_rows.append({**base, "rr_per_unit": rr_value, "model_paf": float(paf),
                              "exposure_mean": r["mean"], "exposure_sd": r["sd"]})
    return pd.DataFrame(rr_rows), pd.DataFrame(paf_rows), pd.DataFrame(diag_rows)


def build(path: Path = ARTIFACT_V2_PATH) -> Artifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    art = Artifact(str(path))

    structure = population_structure_vi()
    write(art, "population.structure", structure)
    write(art, "population.age_bins", age_bins())
    write(art, "population.location", "Australia")
    write(art, "population.demographic_dimensions", structure.drop(columns=["location", "value"]))
    write(art, "population.theoretical_minimum_risk_life_expectancy", theoretical_minimum_risk_life_expectancy())
    write(art, "covariate.live_births_by_sex.estimate", live_births_vi())
    write(art, "cause.all_causes.cause_specific_mortality_rate", vi_cause_table("all_causes", "cause_specific_mortality_rate"))
    write(art, "cause.all_causes.yld_rate", rate_table("all_causes", "ylds"))
    write(art, "cause.background.disability_weight", background_disability()[
        ["sex", "age_start", "age_end", "year_start", "year_end", "value"]])

    summary = []
    for cause in CAUSES:
        tables = {m: vi_cause_table(cause, m) for m in
                  ["prevalence", "incidence_rate", "cause_specific_mortality_rate",
                   "excess_mortality_rate", "disability_weight"]}
        for m, df in tables.items():
            write(art, f"cause.{cause}.{m}", df)
        write(art, f"cause.{cause}.restrictions", {"yld_only": False})
        summary.append((cause, *[tables[m]["value"].max() for m in tables]))
    for cause in MORTALITY_ONLY_CAUSES:
        write(art, f"cause.{cause}.cause_specific_mortality_rate", vi_cause_table(cause, "cause_specific_mortality_rate"))

    unmodeled_targets = {c: UNMODELED_TARGET for c in MORTALITY_ONLY_CAUSES}
    for risk, causes, tmred in [(SBP, SBP_CAUSES + MORTALITY_ONLY_CAUSES, SBP_TMRED),
                                (SODIUM, ["stomach_cancer"], SODIUM_TMRED)]:
        rr, paf, diag = risk_effect_tables_v2(risk, causes, unmodeled_targets)
        exposure = paper_exposure(risk)
        write(art, f"risk_factor.{risk}.distribution", DISTRIBUTION[risk])
        write(art, f"risk_factor.{risk}.exposure", exposure.drop(columns=["sd"]).rename(columns={"mean": "value"}))
        write(art, f"risk_factor.{risk}.exposure_standard_deviation", exposure.drop(columns=["mean"]).rename(columns={"sd": "value"}))
        write(art, f"risk_factor.{risk}.tmred", tmred)
        write(art, f"risk_factor.{risk}.relative_risk_scalar", SCALE[risk])
        write(art, f"risk_factor.{risk}.relative_risk", rr)
        write(art, f"risk_factor.{risk}.population_attributable_fraction", paf)
        diag.to_csv(ROOT / "results" / f"risk_effect_diagnostics_v2_{risk}.csv", index=False)

    cols = ["cause", "max_prev", "max_inc", "max_csmr", "max_emr", "max_dw"]
    print(pd.DataFrame(summary, columns=cols).to_string())
    print(f"wrote {path} with {len(art.keys)} keys")
    return art


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ARTIFACT_V2_PATH)
