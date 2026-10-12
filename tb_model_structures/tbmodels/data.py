"""Loaders for every input, each returning a tidy pandas DataFrame with
provenance columns.  Raw files come from ``data/raw`` (fetched by
``fetch_data.py``) or ``data/vendored`` (redistributable).  Nothing here fits a
model; see ``tbmodels.likelihood`` and ``tbmodels.calibrate``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
VENDORED = ROOT / "data" / "vendored"


# --------------------------------------------------------------------------
# Spectrum transition cohorts (Richards 2023 / Horton 2023, identical file)
# --------------------------------------------------------------------------
def load_spectrum_transitions() -> pd.DataFrame:
    """22 untreated cohorts, 54 rows, of Min/Sub/Clin transition observations.

    Columns of interest: record.id (cohort), model.transition (Min-Sub, Min-Clin,
    Clin-Min, Min-Inf, Inf-Min), Transition (the LibBi observable), n.of.y
    (events), d.of.y (denominator), x.months (follow-up), repeats, data type.
    """
    p = RAW / "repos" / "richards" / "Calibration" / "Data" / "included_data.csv"
    df = pd.read_csv(p)
    df["years"] = df["x.months"] / 12.0
    df["source"] = "Richards 2023 (natural_history_tb_disease)"
    return df


# --------------------------------------------------------------------------
# Infection -> disease converter cohorts (Horton 2023 SI Table 3, adjusted)
# --------------------------------------------------------------------------
# Transcribed verbatim from pnas.2221186120.sapp.pdf, Supplemental Table 3
# ("Adjusted data").  "band" is the disease definition the cohort measured:
# Daniels/Madsen counted minimal disease (radiographic); the three NTI waves
# counted infectious (CXR+ and culture+) disease.
_CONVERTER_ROWS = [
    # study, band, year, events, at_risk
    ("Daniels 1944", "minimal", 1, 5, 248),
    ("Daniels 1944", "minimal", 2, 10, 212),
    ("Daniels 1944", "minimal", 3, 3, 128),
    ("Daniels 1944", "minimal", 4, 1, 58),
    ("Daniels 1944", "minimal", 5, 0, 16),
    ("Madsen 1942", "minimal", 1, 32, 208),
    ("Madsen 1942", "minimal", 2, 7, 171),
    ("Madsen 1942", "minimal", 3, 1, 111),
    ("Madsen 1942", "minimal", 4, 0, 64),
    ("Madsen 1942", "minimal", 5, 0, 26),
    ("Madsen 1942", "minimal", 6, 0, 16),
    ("NTI wave 1", "infectious", 1, 17, 674),
    ("NTI wave 1", "infectious", 2, 3, 657),
    ("NTI wave 1", "infectious", 3, 2, 654),
    ("NTI wave 1", "infectious", 4, 1, 652),
    ("NTI wave 1", "infectious", 5, 0, 651),
    ("NTI wave 2", "infectious", 1, 7, 451),
    ("NTI wave 2", "infectious", 2, 2, 444),
    ("NTI wave 2", "infectious", 3, 1, 442),
    ("NTI wave 2", "infectious", 4, 0, 441),
    ("NTI wave 3", "infectious", 1, 6, 413),
    ("NTI wave 3", "infectious", 2, 2, 407),
]


def load_converter_cohorts() -> pd.DataFrame:
    df = pd.DataFrame(_CONVERTER_ROWS, columns=["study", "band", "year", "events", "at_risk"])
    df["source"] = "Horton 2023 PNAS SI Table 3 (adjusted)"
    return df


# --------------------------------------------------------------------------
# Untreated disease survival (Tiemersma 2011 Table 3)
# --------------------------------------------------------------------------
# Transcribed verbatim from Tiemersma et al. 2011 PLoS ONE, Table 3.
# survival is a proportion; smear status open = smear-positive.
_TIEMERSMA_ROWS = [
    # study, smear, n, surv5, surv10 (None where not reported)
    ("Hartley", "positive", 3326, 0.58, None),
    ("Sinding-Larsen", "positive", 1114, 0.57, 0.47),
    ("Trail & Stockman", "positive", 2615, 0.50, 0.34),
    ("Backer", "positive", 2312, 0.35, 0.21),
    ("Furth/Krebs", "positive", 996, 0.30, 0.19),
    ("Tattersall", "positive", 1082, None, 0.23),
    ("Magnusson", "positive", 379, 0.37, 0.27),
    ("Rutledge & Crouch", "positive", 511, 0.39, None),
    ("Munchbach", "positive", 3966, 0.50, None),
    ("Braeuning & Neisen", "positive", 607, 0.25, 0.18),
    ("Griep", "positive", 975, 0.51, 0.34),
    ("Baart de la Faille", "positive", 534, 0.38, 0.29),
    ("Buhl & Nyboe", "positive", 314, 0.45, 0.34),
    ("Lindhardt", "positive", 11797, 0.43, None),
    ("Berg", "positive", 2042, 0.42, 0.29),
    ("Thompson", "positive", 406, 0.27, 0.14),
    ("Furth/Krebs", "negative", 469, 0.88, 0.78),
    ("Magnusson", "negative", 413, 0.92, 0.85),
    ("Rutledge & Crouch", "negative", 185, 0.86, None),
    ("Baart de la Faille (became neg)", "negative", 597, 0.85, 0.75),
    ("Baart de la Faille (consistently neg)", "negative", 2484, 0.90, None),
]


def load_tiemersma_survival() -> pd.DataFrame:
    df = pd.DataFrame(_TIEMERSMA_ROWS, columns=["study", "smear", "n", "surv5", "surv10"])
    df["source"] = "Tiemersma 2011 PLoS ONE Table 3"
    return df


# --------------------------------------------------------------------------
# Empirical cumulative-incidence curves (Menzies Fig 5; Dataverse)
# --------------------------------------------------------------------------
def load_ferebee() -> pd.DataFrame:
    """Ferebee 1970 isoniazid-trial placebo arm: at-risk and calculated case
    counts by year, for 5% and 10% progression scalings."""
    p = VENDORED / "menzies_dataverse" / "Ferebee_1970_data_rev3.tab"
    df = pd.read_csv(p)  # comma-separated, CR line endings
    df["source"] = "Ferebee 1970 (Menzies Dataverse)"
    return df


def load_sutherland() -> pd.DataFrame:
    """Sutherland 1968 ten-year incidence after conversion."""
    p = VENDORED / "menzies_dataverse" / "Sutherland_1968_data_rev.tab"
    df = pd.read_csv(p)  # comma-separated, CR line endings
    df["source"] = "Sutherland 1968 (Menzies Dataverse)"
    return df


# --------------------------------------------------------------------------
# Prevalence-survey subclinical fractions (epidemery)
# --------------------------------------------------------------------------
def load_prevalence_surveys() -> pd.DataFrame:
    """15 national prevalence surveys with subclinical/clinical x smear-neg/pos
    counts (sub_neg, sub_pos, clin_neg, clin_pos), number screened, symptom
    threshold."""
    p = VENDORED / "epidemery" / "prevalence_survey_data.csv"
    df = pd.read_csv(p)
    df["bact_pos_total"] = df[["sub_neg", "sub_pos", "clin_neg", "clin_pos"]].sum(axis=1)
    df["subclinical_total"] = df[["sub_neg", "sub_pos"]].sum(axis=1)
    df["subclinical_fraction"] = df["subclinical_total"] / df["bact_pos_total"]
    df["source"] = "Emery/Dodd subclinical_transmission (prevalence_survey_data.csv)"
    return df


def load_duration_targets() -> pd.DataFrame:
    """Subclinical and clinical disease durations (months) from epidemery."""
    p = VENDORED / "epidemery" / "duration_data.csv"
    df = pd.read_csv(p)
    df["source"] = "Emery/Dodd subclinical_transmission (duration_data.csv)"
    return df


# --------------------------------------------------------------------------
# Menzies review parameter values (literature priors)
# --------------------------------------------------------------------------
def load_menzies_review() -> pd.DataFrame:
    """680 model-stratum rows with each reviewed model's structure (A-L),
    stratum (HIV+, children, ...) and parameter values a,b,c,d,e,f,d1..d10,
    f1..f3,x1..x3.  Used only as literature priors, never as data."""
    p = VENDORED / "menzies_dataverse" / "LTBI_Review_Data_Clean_10-18-2017.csv"
    df = pd.read_csv(p)
    df["source"] = "Menzies 2018 Dataverse (doi:10.7910/DVN/STRA2P)"
    return df


# --------------------------------------------------------------------------
# Published posterior parameter medians (acceptance targets)
# --------------------------------------------------------------------------
def load_richards_param_table() -> pd.DataFrame:
    p = RAW / "repos" / "richards" / "Results" / "Calibration" / "latest_param_table.csv"
    return pd.read_csv(p)


def load_horton_traces() -> pd.DataFrame:
    """150,000 posterior draws of the Horton main-model parameters."""
    p = RAW / "horton_traces" / "param_traces.csv"
    return pd.read_csv(p)
