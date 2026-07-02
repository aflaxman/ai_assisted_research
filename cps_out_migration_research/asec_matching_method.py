"""Replicate Van Hook, Zhang, Bean & Passel (Demography 2006) "CPS matching
method" for foreign-born emigration, using modern ASEC pairs (2023->2024,
2024->2025).

Method (their Eq. 9): for each foreign-born adult i,

    e_i = [u_i^f - m_i^f + m_i^f d_i^f - d_i^f - u_i^s + m_i^s - m_i^s d_i^s + d_i^s]
          / (1 - m_i^f)

where u = P(non-follow-up March t -> March t+1), m = P(internal move) from the
"lived here 1 year ago" question in the t+1 file, d = P(die in year), and the
f/s superscripts are predictions for person i's covariates under foreign-born
vs second-generation coefficient vectors (weighted logits). Second-generation
adults (U.S.-born, >=1 foreign-born parent) serve as the control for residual
nonresponse; their emigration is assumed ~0. Children 0-14 inherit their
household's mean adult emigration probability. Gross emigration averages e_i
over all foreign-born; net = gross - return-immigration ratio.

Deviations from the paper (documented in the writeup):
  * Mortality d comes from an embedded 2022 U.S. abridged life table by
    age/sex (identical for both generations) instead of NHIS-NDI event-history
    models; a foreign-born mortality advantage (qx * 0.8) is a sensitivity.
    Setting d^f = d^s biases gross e DOWN slightly (~0.05-0.1 pp).
  * One pooled logit per generation (sex + Mexican origin as covariates)
    instead of 8 sex-x-Mexican-stratified models, for one-pair sample sizes.
    Overall gross e is robust to this; SEX- and MEXICAN-SPECIFIC rates are
    where fidelity is weakest -- quote those with caution.
  * Both sides of the match are restricted to the ASEC's March-basic
    subsample, with the TRUE month-in-sample taken from the March basic
    monthly files (the paper kept matchable oversample cases with footnote-7
    workarounds). Unbiased; costs precision on Hispanic cells.
  * Children 0-14 inherit the weighted household mean of foreign-born adults'
    e_i (overall FB-adult mean if no matched FB adult in the household),
    rather than a specific parent's probability.
  * The internal-migration sample is additionally restricted to true MIS 5-8
    in the t+1 file (precision cost only).
  * Match validation uses sex consistency and age change in [-1, +2]
    (paper's footnote 8 states only the upper bound by example; race is not
    required, matching the paper).
  * The return-immigrant "came more than two years before" cutoff is
    approximated with 2-year PEINUSYR entry cohorts; for the 2024->2025 pair
    the topcoded 2022-25 category cannot be split, understating the return
    ratio and biasing NET e up slightly (gross e unaffected).
  * Bootstrap SEs cluster on households with model coefficients held fixed
    (the paper also held coefficients fixed).
"""

from __future__ import annotations

import gzip
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

PP_COLS = ["PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "A_SEX", "PENATVTY",
           "PEFNTVTY", "PEMNTVTY", "PRCITSHP", "PEINUSYR", "MARSUPWT",
           "MIGSAME", "MIG_REG", "A_HGA", "A_HSCOL", "PRDTHSP"]
HH_COLS = ["H_SEQ", "H_IDNUM", "H_MIS", "GESTFIPS", "H_TENURE"]


def load_asec(data_dir: str, year: int) -> pd.DataFrame:
    yy = str(year)[2:]
    pp = pd.read_csv(f"{data_dir}/asec{yy}/pppub{yy}.csv", usecols=PP_COLS)
    hh = pd.read_csv(f"{data_dir}/asec{yy}/hhpub{yy}.csv", usecols=HH_COLS,
                     dtype={"H_IDNUM": str})
    df = pp.merge(hh, left_on="PH_SEQ", right_on="H_SEQ", how="left")
    df["hhkey"] = df["H_IDNUM"].str.strip()
    df["MARSUPWT"] = df["MARSUPWT"] / 100.0  # 2 implied decimals
    df["foreign_born"] = df["PRCITSHP"].isin([4, 5])
    # Second generation: U.S.-born (incl. island areas / born abroad to
    # American parents) with at least one foreign-born parent (country >= 100).
    df["second_gen"] = (~df["foreign_born"]) & (
        (df["PEFNTVTY"] >= 100) | (df["PEMNTVTY"] >= 100))
    df["mexican"] = (df["PRDTHSP"] == 1)
    df["male"] = (df["A_SEX"] == 1)
    df["owner"] = (df["H_TENURE"] == 1)
    df["enrolled"] = df["A_HSCOL"].isin([1, 2])
    df["educ4"] = pd.cut(df["A_HGA"], [-1, 38, 39, 42, 99],
                         labels=["lt_hs", "hs", "some_col", "ba_plus"])
    df["agegrp"] = pd.cut(df["A_AGE"], [-1, 14, 24, 34, 44, 54, 64, 74, 200],
                          labels=["0-14", "15-24", "25-34", "35-44", "45-54",
                                  "55-64", "65-74", "75+"])
    return df


def load_march_true_mis(data_dir: str, year: int) -> dict:
    """hhkey (HRHHID+HRHHID2) -> true month-in-sample, from the March basic
    monthly file, for interviewed households.

    The basic file's HRMIS is authoritative. The ASEC's own H_MIS cannot be
    trusted for rotation logic: in the 2024 ASEC it is REVERSE-CODED
    (1<->5, 2<->6, 3<->7, 4<->8) for the entire March-basic subsample, while
    in the 2025 ASEC it is correct -- the modern incarnation of the mis-coding
    Van Hook et al. (2006, footnote 7) reported for their era's files.
    """
    yy = str(year)[2:]
    path = f"{data_dir}/mar{yy}pub.dat.gz"
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"PK":
        zf = zipfile.ZipFile(path)
        stream = zf.open(zf.namelist()[0])
    else:
        stream = gzip.open(path, "rb")
    mis = {}
    with stream:
        for raw in stream:
            line = raw.decode("ascii", "replace")
            if len(line) < 95:
                continue
            if line[56:58].strip() == "1":  # HRINTSTA == 1
                key = (line[0:15] + line[70:75]).strip()
                mis[key] = int(line[62:64])  # HRMIS
    return mis


# --------------------------------------------------------------------------
# Mortality: 2022 U.S. abridged period life table (annual death probability)
# --------------------------------------------------------------------------

QX_TABLE = {  # age-band lower bound -> (male qx, female qx)
    0: (0.0006, 0.0005), 15: (0.0009, 0.0004), 20: (0.0016, 0.0006),
    25: (0.0019, 0.0008), 30: (0.0023, 0.0011), 35: (0.0027, 0.0014),
    40: (0.0032, 0.0018), 45: (0.0041, 0.0025), 50: (0.0060, 0.0037),
    55: (0.0092, 0.0055), 60: (0.0136, 0.0082), 65: (0.0192, 0.0122),
    70: (0.0289, 0.0192), 75: (0.0442, 0.0314), 80: (0.0724, 0.0545),
    85: (0.1500, 0.1300),
}
_QX_AGES = sorted(QX_TABLE)


def qx(age: pd.Series, male: pd.Series, fb_factor: float = 1.0) -> pd.Series:
    band = pd.cut(age, _QX_AGES + [200], labels=_QX_AGES, right=False)
    band = band.astype(float)
    out = np.where(
        male,
        band.map({a: QX_TABLE[a][0] for a in _QX_AGES}),
        band.map({a: QX_TABLE[a][1] for a in _QX_AGES}),
    )
    return pd.Series(out * fb_factor, index=age.index, dtype=float)


# --------------------------------------------------------------------------
# Matching (March t MIS 1-4  ->  March t+1 MIS 5-8)
# --------------------------------------------------------------------------

AGE_LO, AGE_HI = -1, 2  # allowed A_AGE(t+1) - A_AGE(t)


def build_match(df_t: pd.DataFrame, df_t1: pd.DataFrame,
                mis_t: dict, mis_t1: dict) -> pd.DataFrame:
    """Return the year-t eligible persons with a `nonfollowup` flag.

    Eligibility and targeting use the TRUE month-in-sample from the March
    basic files (`mis_t`, `mis_t1` from load_march_true_mis), restricting both
    sides to the March-basic subsample of the ASEC (the oversample is not in
    scope: its next-year re-inclusion is not guaranteed by the rotation).
    """
    true_mis_t = df_t["hhkey"].map(mis_t)
    base = df_t[true_mis_t.isin([1, 2, 3, 4])].copy()
    true_mis_t1 = df_t1["hhkey"].map(mis_t1)
    tgt = df_t1[true_mis_t1.isin([5, 6, 7, 8])]
    tgt = (tgt[["hhkey", "A_LINENO", "A_SEX", "A_AGE"]]
           .drop_duplicates(["hhkey", "A_LINENO"])
           .rename(columns={"A_SEX": "A_SEX_1", "A_AGE": "A_AGE_1"}))
    m = base.merge(tgt, on=["hhkey", "A_LINENO"], how="left", indicator=True)
    dage = m["A_AGE_1"] - m["A_AGE"]
    ok = ((m["_merge"] == "both") & (m["A_SEX_1"] == m["A_SEX"])
          & dage.between(AGE_LO, AGE_HI))
    m["nonfollowup"] = ~ok
    return m.drop(columns=["_merge"])


# --------------------------------------------------------------------------
# Weighted logits and Eq. (9)
# --------------------------------------------------------------------------

U_FORMULA = ("nonfollowup ~ C(agegrp) + male + mexican + C(educ4) "
             "+ enrolled + owner")
M_FORMULA = "mover ~ C(agegrp) + male + mexican + enrolled + owner"


def _fit_predict(train: pd.DataFrame, formula: str, target: pd.DataFrame,
                 wcol: str = "MARSUPWT") -> pd.Series:
    """Weighted logit fit on `train`, predicted probabilities for `target`."""
    import statsmodels.formula.api as smf
    train = train.copy()
    yname = formula.split("~")[0].strip()
    train[yname] = train[yname].astype(float)
    w = train[wcol] / train[wcol].mean()
    model = smf.glm(formula, data=train, family=sm.families.Binomial(),
                    var_weights=w).fit()
    return model.predict(target)


def internal_migration_frame(df_t1: pd.DataFrame, mis_t1: dict) -> pd.DataFrame:
    """Adults 15+ in the t+1 returning half (true MIS 5-8) who lived in the
    U.S. a year ago. mover = moved within the U.S. (MIGSAME 2 vs 1); movers
    from abroad (MIGSAME 3) are excluded (not at risk of internal migration)."""
    true_mis = df_t1["hhkey"].map(mis_t1)
    d = df_t1[true_mis.isin([5, 6, 7, 8]) & (df_t1["A_AGE"] >= 15)
              & (df_t1["MIGSAME"].isin([1, 2]))].copy()
    d["mover"] = (d["MIGSAME"] == 2)
    return d


def matching_method(df_t: pd.DataFrame, df_t1: pd.DataFrame,
                    mis_t: dict, mis_t1: dict,
                    dur_0_4_cats: set, dur_5_9_cats: set,
                    ret_cutoff_cat: int,
                    fb_mort_factor: float = 1.0) -> dict:
    """Run the full CPS matching method for one ASEC pair.

    mis_t / mis_t1: hhkey -> true HRMIS maps from load_march_true_mis.
    dur_0_4_cats / dur_5_9_cats: year-t PEINUSYR categories approximating
    0-4 and 5-9 years in the U.S. (2-year entry cohorts; see driver).
    ret_cutoff_cat: max t+1 PEINUSYR category counted as "came to the U.S.
    more than two years before" for the return-immigrant definition.
    """
    matched = build_match(df_t, df_t1, mis_t, mis_t1)
    adults = matched[matched["A_AGE"] >= 15]
    fb_ad = adults[adults["foreign_born"]].copy()
    sg_ad = adults[adults["second_gen"]].copy()

    mig = internal_migration_frame(df_t1, mis_t1)
    mig_f, mig_s = mig[mig["foreign_born"]], mig[mig["second_gen"]]

    # --- raw (unadjusted) components ------------------------------------
    def wmean(d, col, wcol="MARSUPWT"):
        return float(np.average(d[col].astype(float), weights=d[wcol]))

    raw = {
        "u_f": wmean(fb_ad, "nonfollowup"), "u_s": wmean(sg_ad, "nonfollowup"),
        "m_f": wmean(mig_f, "mover"), "m_s": wmean(mig_s, "mover"),
        "n_fb_adults": len(fb_ad), "n_sg_adults": len(sg_ad),
        "n_mig_f": len(mig_f), "n_mig_s": len(mig_s),
    }

    # --- composition-adjusted per-person components on FB adults ---------
    uf = _fit_predict(fb_ad, U_FORMULA, fb_ad)
    us = _fit_predict(sg_ad, U_FORMULA, fb_ad)
    mf = _fit_predict(mig_f, M_FORMULA, fb_ad)
    ms = _fit_predict(mig_s, M_FORMULA, fb_ad)
    df_ = qx(fb_ad["A_AGE"], fb_ad["male"], fb_mort_factor)
    ds_ = qx(fb_ad["A_AGE"], fb_ad["male"], 1.0)

    e_i = (uf - mf + mf * df_ - df_ - us + ms - ms * ds_ + ds_) / (1.0 - mf)
    fb_ad = fb_ad.assign(e_i=e_i.values, u_f=uf.values, u_s=us.values,
                         m_f=mf.values, m_s=ms.values, d_f=df_.values,
                         d_s=ds_.values)

    # --- children 0-14: inherit household mean of FB-adult e_i -----------
    fb_kids = matched[(matched["A_AGE"] <= 14) & matched["foreign_born"]].copy()
    hh_e = fb_ad.groupby("hhkey").apply(
        lambda g: np.average(g["e_i"], weights=g["MARSUPWT"]),
        include_groups=False)
    overall_adult_e = float(np.average(fb_ad["e_i"], weights=fb_ad["MARSUPWT"]))
    fb_kids["e_i"] = fb_kids["hhkey"].map(hh_e).fillna(overall_adult_e)

    fb_all = pd.concat(
        [fb_ad[["hhkey", "A_AGE", "male", "GESTFIPS", "MARSUPWT", "PEINUSYR",
                "e_i"]],
         fb_kids[["hhkey", "A_AGE", "male", "GESTFIPS", "MARSUPWT",
                  "PEINUSYR", "e_i"]]],
        ignore_index=True)

    gross_e = float(np.average(fb_all["e_i"], weights=fb_all["MARSUPWT"]))
    d_bar = float(np.average(
        qx(fb_all["A_AGE"], fb_all["male"], fb_mort_factor),
        weights=fb_all["MARSUPWT"]))

    # --- return immigration ratio (from full t+1 file) --------------------
    fb1 = df_t1[df_t1["foreign_born"]]
    ret = fb1[(fb1["MIGSAME"] == 3) & (fb1["PEINUSYR"] > 0)
              & (fb1["PEINUSYR"] <= ret_cutoff_cat)]
    at_risk = fb1[fb1["MIGSAME"].isin([1, 2])]
    ret_ratio_raw = float(ret["MARSUPWT"].sum() / at_risk["MARSUPWT"].sum())
    ret_ratio = ret_ratio_raw * (1.0 - gross_e - d_bar)

    net_e = gross_e - ret_ratio

    # --- cluster bootstrap (households; coefficients fixed) --------------
    rng = np.random.default_rng(20260702)
    hh_ids = fb_all["hhkey"].unique()
    grouped = dict(tuple(fb_all.groupby("hhkey")))
    boots = []
    for _ in range(200):
        draw = rng.choice(hh_ids, size=len(hh_ids), replace=True)
        s = pd.concat([grouped[h] for h in draw], ignore_index=True)
        boots.append(np.average(s["e_i"], weights=s["MARSUPWT"]))
    gross_se = float(np.std(boots, ddof=1))

    # --- subgroup rates ----------------------------------------------------
    def sub(mask):
        d = fb_all[mask]
        if d["MARSUPWT"].sum() == 0:
            return np.nan, 0
        return float(np.average(d["e_i"], weights=d["MARSUPWT"])), len(d)

    min_5_9 = min(dur_5_9_cats)
    subgroups = {
        "male": sub(fb_all["male"]),
        "female": sub(~fb_all["male"]),
        "in_us_0_4": sub(fb_all["PEINUSYR"].isin(dur_0_4_cats)),
        "in_us_5_9": sub(fb_all["PEINUSYR"].isin(dur_5_9_cats)),
        "in_us_10plus": sub((fb_all["PEINUSYR"] > 0)
                            & (fb_all["PEINUSYR"] < min_5_9)),
        "washington": sub(fb_all["GESTFIPS"] == 53),
    }

    fb_pop = float(fb_all["MARSUPWT"].sum())
    wa_pop = float(fb_all.loc[fb_all["GESTFIPS"] == 53, "MARSUPWT"].sum())
    # Full foreign-born stock (all MIS/oversample) for scaling rates to counts.
    fb_stock = float(df_t.loc[df_t["foreign_born"], "MARSUPWT"].sum())
    wa_fb_stock = float(df_t.loc[df_t["foreign_born"]
                                 & (df_t["GESTFIPS"] == 53), "MARSUPWT"].sum())

    return {
        "raw": raw,
        "gross_e": gross_e, "gross_se": gross_se,
        "ret_ratio_raw": ret_ratio_raw, "ret_ratio": ret_ratio,
        "net_e": net_e, "d_bar": d_bar,
        "subgroups": subgroups,
        "fb_pop": fb_pop, "wa_fb_pop": wa_pop,
        "fb_stock": fb_stock, "wa_fb_stock": wa_fb_stock,
        "n_fb_all": len(fb_all), "n_return_unwt": len(ret),
        "fb_all": fb_all,
    }
