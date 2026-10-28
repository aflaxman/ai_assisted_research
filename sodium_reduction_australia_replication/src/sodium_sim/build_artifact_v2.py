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
    MAX_PAF,
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
    paf_table_from_gbd,
    rate_table,
    theoretical_minimum_risk_life_expectancy,
    write,
)

DATA_VI = ROOT / "data" / "gbd_vi"
DATA_2019 = ROOT / "data" / "gbd2019"
DATA_PAPER = ROOT / "data" / "paper"
ARTIFACT_V2_PATH = ROOT / "artifacts" / "australia_v2.hdf"
ARTIFACT_2019_PATH = ROOT / "artifacts" / "australia_gbd2019.hdf"

# The paper's 18 sodium-related causes (Table S6); the last five are SBP outcomes in
# GBD 2019 only. Aortic aneurysm is fatal-only in both releases.
CAUSES_2019 = CAUSES + [
    "ckd_due_to_diabetes_type_1",
    "rheumatic_heart_disease",
    "endocarditis",
    "non_rheumatic_calcific_aortic_valve_disease",
    "other_cardiomyopathy",
]
GBD2019_CAUSE_IDS = {
    "ischemic_heart_disease": 493, "ischemic_stroke": 495, "intracerebral_hemorrhage": 496,
    "subarachnoid_hemorrhage": 497, "hypertensive_heart_disease": 498, "atrial_fibrillation_and_flutter": 500,
    "aortic_aneurysm": 501, "peripheral_arterial_disease": 502, "ckd_due_to_diabetes_type_1": 997,
    "ckd_due_to_diabetes_type_2": 998, "ckd_due_to_hypertension": 591, "ckd_due_to_glomerulonephritis": 592,
    "ckd_due_to_other_causes": 593, "stomach_cancer": 414, "rheumatic_heart_disease": 492, "endocarditis": 503,
    "non_rheumatic_calcific_aortic_valve_disease": 968, "other_cardiomyopathy": 944,
}
GBD_AGE_GROUP_START = {10: 25, 11: 30, 12: 35, 13: 40, 14: 45, 15: 50, 16: 55, 17: 60, 18: 65, 19: 70,
                       20: 75, 30: 80, 31: 85, 32: 90, 235: 95}
SODIUM_FIT_RANGE_G = (2.0, 6.0)

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


def gbd2019_table(cause: str, measure: str) -> pd.DataFrame:
    """A GBD 2019 (get_outputs, release 6) rate by sex and GBD 2023 age band.

    GBD 2019 uses 23 age bands; the two under-5 bands that GBD 2023 splits
    (post-neonatal, 1 to 4) are copied onto their sub-bands. A missing file
    (no nonfatal estimates) gives zeros.
    """
    f = DATA_2019 / f"cause.{cause}.{measure}.csv"
    bins = age_bins()
    rows = []
    if not f.exists():
        for sex in SEXES:
            for _, b in bins.iterrows():
                rows.append((sex, b["age_start"], b["age_end"], YEAR_START, YEAR_END, 0.0))
        return pd.DataFrame(rows, columns=["sex", "age_start", "age_end", "year_start", "year_end", "value"])
    t = pd.read_csv(f)
    for sex in SEXES:
        ts = t[t["sex"] == sex]
        for _, b in bins.iterrows():
            mid = 0.5 * (b["age_start"] + b["age_end"])
            m = ts[(ts["age_start"] <= mid) & (mid < ts["age_end"])]
            value = float(m["value"].iloc[0]) if len(m) else 0.0
            rows.append((sex, b["age_start"], b["age_end"], YEAR_START, YEAR_END, value))
    return pd.DataFrame(rows, columns=["sex", "age_start", "age_end", "year_start", "year_end", "value"])


def gbd2019_cause_tables(cause: str) -> dict[str, pd.DataFrame]:
    """Prevalence, incidence per susceptible, CSMR, EMR and DW for a cause from GBD 2019 outputs."""
    from sodium_sim.build_artifact import DERIVED_INCIDENCE_CAUSES, incidence_from_prevalence

    prev = gbd2019_table(cause, "prevalence")
    csmr = gbd2019_table(cause, "deaths")
    ylds = gbd2019_table(cause, "ylds")
    safe_prev = prev["value"].where(prev["value"] > 0, np.nan)
    emr = csmr.copy()
    emr["value"] = (csmr["value"] / safe_prev).fillna(0.0)
    dw = ylds.copy()
    dw["value"] = (ylds["value"] / safe_prev).fillna(0.0).clip(upper=1.0)
    if cause in DERIVED_INCIDENCE_CAUSES or not (DATA_2019 / f"cause.{cause}.incidence.csv").exists():
        inc = incidence_from_prevalence(prev, emr)
        inc["value"] = inc["value"] * (1 - prev["value"])
    else:
        inc = gbd2019_table(cause, "incidence")
    inc_sus = inc.copy()
    inc_sus["value"] = inc["value"] / (1 - prev["value"])
    return {"prevalence": prev, "incidence_rate": inc_sus, "cause_specific_mortality_rate": csmr,
            "excess_mortality_rate": emr, "disability_weight": dw, "yld_rate": ylds}


def gbd2019_relative_risk(risk: str) -> pd.DataFrame:
    """GBD 2019 (release 6) relative risks per unit by sex, age and cause slug.

    SBP comes as one RR per 10 mmHg by age; the sodium RR is a curve over
    exposure, to which a log-linear slope per g/day is fitted between 2 and
    6 g/day.
    """
    slug = {v: k for k, v in GBD2019_CAUSE_IDS.items()}
    if risk == SBP:
        rr = pd.read_csv(DATA_2019 / "sbp_rr_gbd2019.csv")
        rr = rr[rr["cause_id"].isin(slug)]
        return pd.DataFrame({
            "sex": rr["sex_id"].map({1: "Male", 2: "Female"}),
            "age_start": rr["age_group_id"].map(GBD_AGE_GROUP_START).astype(float),
            "affected_entity": rr["cause_id"].map(slug),
            "value": rr["value"],
        })
    rr = pd.read_csv(DATA_2019 / "sodium_rr_gbd2019.csv")
    lo, hi = SODIUM_FIT_RANGE_G
    rows = []
    for (sex_id, age_id, cause_id), g in rr.groupby(["sex_id", "age_group_id", "cause_id"]):
        g = g[(g["exposure"] >= lo) & (g["exposure"] <= hi) & (g["value"] > 0)]
        slope = np.polyfit(g["exposure"], np.log(g["value"]), 1)[0] if len(g) > 1 else 0.0
        rows.append((sex_id, age_id, cause_id, float(np.exp(slope))))
    out = pd.DataFrame(rows, columns=["sex_id", "age_group_id", "cause_id", "value"])
    out = out[out["cause_id"].isin(slug)]
    return pd.DataFrame({
        "sex": out["sex_id"].map({1: "Male", 2: "Female"}),
        "age_start": out["age_group_id"].map(GBD_AGE_GROUP_START).astype(float),
        "affected_entity": out["cause_id"].map(slug),
        "value": out["value"],
    })


def background_disability_2019(causes: list[str]) -> pd.DataFrame:
    all_cause = gbd2019_table("all_causes", "ylds")
    out = all_cause.copy()
    out["value"] = (all_cause["value"] - sum(gbd2019_table(c, "ylds")["value"] for c in causes)).clip(lower=0)
    return out


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
    """GBD relative risks per unit by sex, age and affected entity.

    These are log-linear slopes fitted to the GBD 2023 Burden-of-Proof curves by
    ``prepare_gbd_vi`` (RR per 10 mmHg for SBP, per g/day for sodium).
    """
    rr = pd.read_csv(DATA_VI / f"risk.{risk}.relative_risk_per_unit.csv")
    return rr[["sex", "age_start", "age_end", "affected_entity", "affected_measure", "value"]]


def gbd_paf(risk: str) -> pd.DataFrame:
    """GBD 2023 population attributable fractions (draw mean) by sex, age, entity and measure."""
    paf = pd.read_csv(DATA_VI / f"risk.{risk}.population_attributable_fraction.csv")
    return paf[["sex", "age_start", "affected_entity", "affected_measure", "value"]]


def solve_log_rr(paf: float, dist: str, mean: float, sd: float, tmrel: float, scale: float, max_log_rr: float) -> float:
    """Return the per-unit log RR whose population-average RR gives this PAF, capped."""
    from scipy.optimize import brentq

    paf = min(max(paf, 0.0), MAX_PAF)
    if paf <= 0:
        return 0.0
    f = lambda b: expected_rr(b, dist, mean, sd, tmrel, scale) - 1 / (1 - paf)
    if f(max_log_rr) < 0:
        return max_log_rr
    return brentq(f, 0.0, max_log_rr, xtol=1e-6)


def risk_effect_tables_v2(risk: str, causes: list[str], targets: dict[str, tuple[str, str]] | None = None):
    """Return (relative_risk, paf, diagnostics) for GBD RRs under the paper's exposure."""
    targets = targets or {}
    tmred = {SBP: SBP_TMRED, SODIUM: SODIUM_TMRED}[risk]
    tmrel = 0.5 * (tmred["min"] + tmred["max"])
    scale, dist = SCALE[risk], DISTRIBUTION[risk]
    exposure = paper_exposure(risk)
    if source == "gbd2019":
        rr_all = gbd2019_relative_risk(risk)
        paf_all = pd.DataFrame(columns=["sex", "age_start", "affected_entity", "affected_measure", "value"])
        entity_name = lambda c: c
    else:
        rr_all = gbd_relative_risk(risk)
        paf_all = gbd_paf(risk)
        entity_name = vi_name
    max_log_rr = {SBP: np.log(4.0), SODIUM: np.log(1.6)}[risk]
    rr_rows, paf_rows, diag_rows = [], [], []
    for cause in causes:
        name = entity_name(cause)
        rr_c = rr_all[(rr_all["affected_entity"] == name) & (rr_all["value"] > 1.0)]
        if not rr_c.empty:
            source_label = "gbd2019_rr" if source == "gbd2019" else "gbd_curve_fit"
            d = exposure.merge(rr_c.rename(columns={"value": "rr"})[["sex", "age_start", "rr"]],
                               on=["sex", "age_start"], how="left")
            d["rr"] = d["rr"].fillna(1.0)
        else:
            # No RR curve in GBD 2023 (CKD, aortic aneurysm, the PAF-of-one causes):
            # back the per-unit RR out of GBD's PAF under the paper's exposure.
            source_label = "paf_backed_out"
            measure_paf = "cause_specific_mortality_rate" if cause in MORTALITY_ONLY_CAUSES else "incidence_rate"
            p = paf_all[(paf_all["affected_entity"] == name) & (paf_all["affected_measure"] == measure_paf)]
            if p.empty:
                # vivarium_inputs drops YLL-only causes such as aortic aneurysm; use the MCP deaths PAF.
                source_label = "paf_backed_out_mcp"
                p = paf_table_from_gbd(cause, risk).rename(columns={"value": "gbd_paf"})
            else:
                p = p.rename(columns={"value": "gbd_paf"})
            d = exposure.merge(p[["sex", "age_start", "gbd_paf"]], on=["sex", "age_start"], how="left")
            d["gbd_paf"] = d["gbd_paf"].fillna(0.0)
            d["rr"] = [np.exp(solve_log_rr(r.gbd_paf, dist, r["mean"], r["sd"], tmrel, scale, max_log_rr))
                       for _, r in d.iterrows()]
        entity, measure = targets.get(cause, (cause, "incidence_rate"))
        for _, r in d.iterrows():
            rr_value = float(r["rr"]) if r["age_start"] >= 25 else 1.0
            paf = 0.0 if rr_value <= 1.0 else 1 - 1 / expected_rr(np.log(rr_value), dist, r["mean"], r["sd"], tmrel, scale)
            base = dict(sex=r["sex"], age_start=r["age_start"], age_end=r["age_end"],
                        year_start=YEAR_START, year_end=YEAR_END,
                        affected_entity=entity, affected_measure=measure)
            rr_rows.append({**base, "value": rr_value})
            paf_rows.append({**base, "value": float(paf)})
            diag_rows.append({**base, "cause": cause, "source": source_label, "rr_per_unit": rr_value,
                              "model_paf": float(paf), "gbd_paf": float(r.get("gbd_paf", np.nan)),
                              "exposure_mean": r["mean"], "exposure_sd": r["sd"]})
    return pd.DataFrame(rr_rows), pd.DataFrame(paf_rows), pd.DataFrame(diag_rows)


def expected_rr_curve(grid_x: np.ndarray, grid_rr: np.ndarray, dist: str, mean: float, sd: float) -> float:
    """E[max(RR(X), 1)] with RR interpolated on the GBD grid and held flat beyond it."""
    lo = max(mean - 6 * sd, 1e-6) if dist == "lognormal" else mean - 6 * sd
    x = np.linspace(lo, mean + 8 * sd, 6001)
    w = exposure_grid(x, dist, mean, sd)
    w /= w.sum()
    rr = np.clip(np.interp(x, grid_x, grid_rr), 1.0, None)
    return float((rr * w).sum())


def curve_tables(risk: str, causes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (relative_risk_curve, paf_curve, causes_with_curves) for NonLogLinearRiskEffect.

    The curve is GBD 2023's Burden-of-Proof relative risk on its exposure grid
    (thinned by ``prepare_gbd_vi``), and the PAF is computed under the paper's
    exposure distribution so the two are consistent. Causes whose curve is all
    zeros (hypertensive heart disease, a PAF-of-one cause) or absent are
    skipped and keep the log-linear effect.
    """
    dist = DISTRIBUTION[risk]
    curve = pd.read_csv(DATA_VI / f"risk.{risk}.relative_risk_curve.csv")
    exposure = paper_exposure(risk)
    rr_rows, paf_rows, with_curves = [], [], []
    for cause in causes:
        c = curve[(curve["affected_entity"] == vi_name(cause)) & (curve["affected_measure"] == "incidence_rate")]
        if c.empty or (c["value"] <= 0).all():
            continue
        with_curves.append(cause)
        for _, e in exposure.iterrows():
            g = c[(c["sex"] == e["sex"]) & (c["age_start"] == e["age_start"])].sort_values("parameter")
            base = dict(sex=e["sex"], age_start=e["age_start"], age_end=e["age_end"],
                        year_start=YEAR_START, year_end=YEAR_END,
                        affected_entity=cause, affected_measure="incidence_rate")
            if g.empty or e["age_start"] < 25 or (g["value"] <= 0).all():
                grid_x, grid_rr = np.array([0.0, 1000.0]), np.array([1.0, 1.0])
            else:
                grid_x, grid_rr = g["parameter"].to_numpy(), g["value"].to_numpy()
            for x, rr in zip(grid_x, grid_rr):
                rr_rows.append({**base, "parameter": float(x), "value": float(max(rr, 1.0))})
            paf = 1 - 1 / expected_rr_curve(grid_x, grid_rr, dist, e["mean"], e["sd"])
            paf_rows.append({**base, "value": float(paf)})
    return pd.DataFrame(rr_rows), pd.DataFrame(paf_rows), with_curves


def build(path: Path | None = None, source: str = "gbd2023") -> Artifact:
    """Build the artifact from GBD 2023 (``vivarium_inputs``) or GBD 2019 (``get_outputs``) inputs."""
    if source not in ("gbd2023", "gbd2019"):
        raise ValueError(f"Unknown source {source!r}")
    path = path or (ARTIFACT_V2_PATH if source == "gbd2023" else ARTIFACT_2019_PATH)
    causes = CAUSES if source == "gbd2023" else CAUSES_2019
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
    if source == "gbd2023":
        write(art, "cause.all_causes.cause_specific_mortality_rate", vi_cause_table("all_causes", "cause_specific_mortality_rate"))
        write(art, "cause.all_causes.yld_rate", rate_table("all_causes", "ylds"))
        background = background_disability()
    else:
        write(art, "cause.all_causes.cause_specific_mortality_rate", gbd2019_table("all_causes", "deaths"))
        write(art, "cause.all_causes.yld_rate", gbd2019_table("all_causes", "ylds"))
        background = background_disability_2019(causes)
    write(art, "cause.background.disability_weight", background[
        ["sex", "age_start", "age_end", "year_start", "year_end", "value"]])

    measures = ["prevalence", "incidence_rate", "cause_specific_mortality_rate", "excess_mortality_rate", "disability_weight"]
    summary = []
    for cause in causes:
        if source == "gbd2023":
            tables = {m: vi_cause_table(cause, m) for m in measures}
        else:
            tables = gbd2019_cause_tables(cause)
        for m in measures:
            write(art, f"cause.{cause}.{m}", tables[m])
        write(art, f"cause.{cause}.restrictions", {"yld_only": False})
        summary.append((cause, *[tables[m]["value"].max() for m in measures]))
    for cause in MORTALITY_ONLY_CAUSES:
        csmr = (vi_cause_table(cause, "cause_specific_mortality_rate") if source == "gbd2023"
                else gbd2019_table(cause, "deaths"))
        write(art, f"cause.{cause}.cause_specific_mortality_rate", csmr)

    unmodeled_targets = {c: UNMODELED_TARGET for c in MORTALITY_ONLY_CAUSES}
    sbp_causes = [c for c in causes if c != "stomach_cancer"] + MORTALITY_ONLY_CAUSES
    for risk, risk_causes, tmred in [(SBP, sbp_causes, SBP_TMRED), (SODIUM, ["stomach_cancer"], SODIUM_TMRED)]:
        rr, paf, diag = risk_effect_tables_v2(risk, risk_causes, unmodeled_targets, source=source)
        exposure = paper_exposure(risk)
        write(art, f"risk_factor.{risk}.distribution", DISTRIBUTION[risk])
        write(art, f"risk_factor.{risk}.exposure", exposure.drop(columns=["sd"]).rename(columns={"mean": "value"}))
        write(art, f"risk_factor.{risk}.exposure_standard_deviation", exposure.drop(columns=["mean"]).rename(columns={"sd": "value"}))
        write(art, f"risk_factor.{risk}.tmred", tmred)
        write(art, f"risk_factor.{risk}.relative_risk_scalar", SCALE[risk])
        write(art, f"risk_factor.{risk}.relative_risk", rr)
        write(art, f"risk_factor.{risk}.population_attributable_fraction", paf)
        diag.to_csv(ROOT / "results" / f"risk_effect_diagnostics_{source}_{risk}.csv", index=False)
        if source == "gbd2023":
            rr_curve, paf_curve, with_curves = curve_tables(risk, risk_causes)
            write(art, f"risk_factor.{risk}.relative_risk_curve", rr_curve)
            write(art, f"risk_factor.{risk}.population_attributable_fraction_curve", paf_curve)
            print(f"{risk}: non-log-linear curves for {with_curves}")

    cols = ["cause", "max_prev", "max_inc", "max_csmr", "max_emr", "max_dw"]
    print(pd.DataFrame(summary, columns=cols).to_string())
    print(f"wrote {path} with {len(art.keys)} keys")
    return art


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gbd2023", "gbd2019"], default="gbd2023")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    build(Path(args.path) if args.path else None, source=args.source)
