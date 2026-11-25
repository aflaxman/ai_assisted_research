"""First look: does the GBD Healthcare Access and Quality (HAQ) Index predict
the share of incident strokes that are admitted to hospital?

This is a diagnostic of a candidate estimation strategy, not an estimate.
Nothing this script writes is a GBD estimate or a recommended model input.

Run from the project folder:

    uv run --with-requirements analysis/requirements.txt python analysis/first_look_haq.py

Inputs
    data/study_level_admitted_share.csv        study-level table transcribed from the research notes
    data/haq_gbd2023_study_country_years.csv   HAQ Index (GBD 2023) as returned by the IHME GBD connector

Outputs
    outputs/first_look_summary.txt             every number reported
    outputs/primary_analysis_set.csv           the rows behind the primary fit, with HAQ attached
    figures/admitted_share_vs_haq.png
    figures/within_site_trajectories.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS, OUT = ROOT / "data", ROOT / "figures", ROOT / "outputs"

# ---------------------------------------------------------------------------
# Analysis choices that are not columns of the table. Each is listed in the
# summary file, and each has a sensitivity fit that undoes it.
# ---------------------------------------------------------------------------

# Direct-measure rows left out of the primary set because the same site and
# years are already represented by another row.
OVERLAP = {
    "HIC-03": "OXVASC 2002-14, all episodes: same register and years as HIC-02 (2002-17, incident strokes)",
    "HIC-05": "South London 1995-98: lies inside the 1995-2000 wave (HIC-06b)",
    "HIC-08b": "South London 2007-09: lies inside the 2007-12 wave (HIC-08a)",
    "HIC-10": "three southern England districts 1989-90: overlaps HIC-11 (two of the districts, 1989-91)",
    "LMIC-A4a": "Tianjin 1992-2008: a second cut of the surveillance in LMIC-A1 to A3",
    "LMIC-A4b": "Tianjin 2009-18: a second cut of the surveillance in LMIC-A1 to A3",
    "LMIC-A6": "Tianjin 2010-20, 1 January to 20 April only: overlaps LMIC-A3",
    "XTR-SL1": "South London 1995-97 (Addo 2011): overlaps the Bhalla 2016 waves",
    "XTR-SL2": "South London 1998-2000 (Addo 2011): overlaps the Bhalla 2016 waves",
    "XTR-SL3": "South London 2001-03 (Addo 2011): overlaps the Bhalla 2016 waves",
    "XTR-SL4": "South London 2004-06 (Addo 2011): overlaps the Bhalla 2016 waves",
}

# Where the notes give two values for one site-period, the primary set keeps
# the value from the source that reports the site's whole series, and the
# other value (the key below) is swapped in as a sensitivity.
ALTERNATE = {
    "HIC-06a": ("HIC-06b", "South London 1995-2000: 85.1% (McKevitt 2005); the primary set keeps 82% (Bhalla 2016)"),
    "HIC-43b": ("HIC-43a", "Tartu 1991-93: 82% (Korv 1999); the primary set keeps 72% (Vibo thesis)"),
    "XTR-AKL1": ("HIC-15", "Auckland 1981-82: 64.5% (secondary citation); the primary set keeps 62.5% (Feigin 2025)"),
    "XTR-AKL2": ("HIC-17", "Auckland 2002-03: 92.4% (secondary citation); the primary set keeps 90.9% (Feigin 2025)"),
}

# Direct-measure rows whose study years are not in the notes. They are out of
# every fit except one crude sensitivity, which places them at the year of
# publication minus 2. The publication years are stated in the notes.
PUB_YEAR = {"HIC-14": 2000, "HIC-29": 1998, "HIC-36": 2023, "HIC-45": 2001, "LMIC-B3": 2005, "XTR-TBILISI": 2004}
PUB_LAG = 2

# The IHME rule for injuries, as quoted in gbd_pipeline_evidence.md: 100% of
# injuries treated at HAQ 75 and above, and all other location-years "scaled
# ... between 10% and 100% treated based on their HAQ Index value". The form
# between the anchors is not published; a straight line is assumed here, with
# the 10% anchor drawn at HAQ 0, the bottom of the index's 0-100 scale.
RULE_LOW_PCT, RULE_TOP_HAQ = 10.0, 75.0

PRED_GRID = [20, 30, 40, 50, 60, 70, 80, 90]


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def injuries_rule(haq, low_anchor=0.0):
    """Assumed linear interpolation of the IHME injuries rule (percent treated).

    low_anchor is the HAQ value at which the rule gives 10%.
    """
    haq = np.asarray(haq, dtype=float)
    pct = RULE_LOW_PCT + (100.0 - RULE_LOW_PCT) * (haq - low_anchor) / (RULE_TOP_HAQ - low_anchor)
    return np.clip(pct, RULE_LOW_PCT, 100.0)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
tab = pd.read_csv(DATA / "study_level_admitted_share.csv", dtype={"pmid": "string", "doi": "string"})
haq = pd.read_csv(DATA / "haq_gbd2023_study_country_years.csv")
haq = haq.rename(columns={"location_name": "country", "year": "haq_year"})
haq["haq_year"] = haq["haq_year"].astype(float)

# HAQ year: the calendar year that holds the study's midpoint, and for a
# period with an even number of years the earlier of the two central years.
# National values are used throughout, also for city-level studies.
tab["haq_year"] = np.floor(tab["year_mid"])
tab["year_imputed"] = False
for rid, pub in PUB_YEAR.items():
    tab.loc[tab["row_id"] == rid, "haq_year"] = pub - PUB_LAG
    tab.loc[tab["row_id"] == rid, "year_imputed"] = True
tab = tab.merge(haq[["country", "haq_year", "haq", "haq_lower", "haq_upper"]], on=["country", "haq_year"], how="left")

# The logit needs a share strictly inside (0, 1). 0.5% and 99.5% are the
# limits; they bind only in the MONICA rows (four populations at 99.6% to 100%).
tab["p_clipped"] = (tab["pct_admitted"] / 100).clip(0.005, 0.995)
tab["logit_pct"] = logit(tab["p_clipped"])
tab["rural"] = (tab["setting"] == "rural").astype(float)
tab["clinic"] = tab["clinic_system_flag"].astype(float)
tab["year_c"] = tab["year_mid"] - 2000
# "National" is a site name in several countries, so sites are keyed by country as well.
tab["site_key"] = tab["country"] + " | " + tab["site"]
tab["site_label"] = tab["site"].where(tab["site"] != "National", tab["country"] + " (national)")

direct = tab[tab["measure_class"] == "direct_incident"]
usable = direct[direct["haq"].notna() & direct["pct_admitted"].notna() & ~direct["year_imputed"]]
primary = usable[~usable["row_id"].isin(OVERLAP) & ~usable["row_id"].isin(ALTERNATE)].copy()
imputed = direct[direct["year_imputed"] & direct["haq"].notna() & direct["pct_admitted"].notna()].copy()
monica = tab[(tab["measure_class"] == "monica_nonfatal") & tab["haq"].notna()].copy()
upper = tab[(tab["measure_class"] == "upper_bound_linked") & tab["haq"].notna()].copy()


# ---------------------------------------------------------------------------
# 2. Fitting helpers
# ---------------------------------------------------------------------------
def fit_ols(d, xcols=("haq",), ycol="logit_pct"):
    """OLS with standard errors clustered by site; t reference with (sites - 1) df."""
    d = d.reset_index(drop=True)
    X = sm.add_constant(d[list(xcols)].astype(float))
    groups = pd.factorize(d["site_key"])[0]
    return sm.OLS(d[ycol].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)


def describe(res, d):
    ci = res.conf_int()
    out = {
        "n_rows": len(d), "n_sites": d["site_key"].nunique(), "n_countries": d["country"].nunique(),
        "resid_sd": float(np.sqrt(res.ssr / res.df_resid)), "r2": float(res.rsquared),
        "df_t": float(res.df_resid_inference),
    }
    for name in res.params.index:
        out[name] = (float(res.params[name]), float(ci.loc[name, 0]), float(ci.loc[name, 1]), float(res.bse[name]))
    return out


def fmt_fit(label, d, xcols=("haq",), ycol="logit_pct"):
    """Fit, and return (result, description, text lines for the summary)."""
    lines = [label]
    res = fit_ols(d, xcols, ycol)
    s = describe(res, d)
    lines.append(f"    rows / sites / countries: {s['n_rows']} / {s['n_sites']} / {s['n_countries']}")
    b, lo, hi, se = s["haq"]
    lines.append(f"    slope per HAQ point: {b:.4f} (95% CI {lo:.4f} to {hi:.4f}; SE {se:.4f}, clustered by site, t with {s['df_t']:.0f} df)")
    if ycol == "logit_pct":
        lines.append(f"    odds ratio per 10 HAQ points: {np.exp(10 * b):.2f} ({np.exp(10 * lo):.2f} to {np.exp(10 * hi):.2f})")
    b0, lo0, hi0, _ = s["const"]
    lines.append(f"    intercept: {b0:.3f} (95% CI {lo0:.3f} to {hi0:.3f})")
    for name in xcols:
        if name != "haq":
            bx, lox, hix, sex = s[name]
            lines.append(f"    coefficient on {name}: {bx:.3f} (95% CI {lox:.3f} to {hix:.3f}; SE {sex:.3f})")
    unit = "logit units" if ycol == "logit_pct" else "percentage points"
    lines.append(f"    R-squared: {s['r2']:.3f}; residual SD: {s['resid_sd']:.3f} {unit}")
    return res, s, lines


L = []  # lines of the summary file


def say(*lines):
    L.extend(lines)


# ---------------------------------------------------------------------------
# 3. Row counts
# ---------------------------------------------------------------------------
say("FIRST LOOK: HAQ INDEX AGAINST THE SHARE OF INCIDENT STROKES ADMITTED TO HOSPITAL",
    "A diagnostic of a candidate estimation strategy. Nothing below is a GBD estimate or a recommended input.",
    "HAQ values are GBD 2023 national values from the IHME GBD connector, used as a regression covariate.",
    "Written by analysis/first_look_haq.py from data/study_level_admitted_share.csv and",
    "data/haq_gbd2023_study_country_years.csv.", "")

say("1. ROWS", f"  evidence table: {len(tab)} rows")
for cls, n in tab["measure_class"].value_counts().items():
    say(f"    {cls}: {n}")
say("  by notes file:")
for f, n in tab["notes_file"].value_counts().items():
    say(f"    {f}: {n}")
say("  by evidence grade (blank = the notes do not say):")
for g, n in tab["evidence_grade"].fillna("(blank)").value_counts().items():
    say(f"    {g}: {n}")
say(f"  HAQ file: {len(haq)} country-years. HAQ is attached at the calendar year holding the study midpoint (floor of year_mid),",
    "  and national values are used for every site, city studies included.", "")

no_pct = direct[direct["pct_admitted"].isna()]
no_haq = direct[direct["pct_admitted"].notna() & direct["haq"].isna()]
say("  direct_incident rows and how they reach the primary set:",
    f"    all direct_incident rows: {len(direct)}",
    f"    no pooled percentage in the notes: {len(no_pct)} ({', '.join(no_pct['row_id'])})",
    f"    no HAQ year or location: {len(no_haq) + len(imputed)}")
for _, r in pd.concat([no_haq, imputed]).sort_values("row_id").iterrows():
    if r["year_imputed"]:
        why = "study years not in the notes"
    elif r["year_mid"] < 1980:
        why = "midpoint before 1980"
    elif pd.isna(r["year_mid"]):
        why = "location and years not in the notes"
    else:
        why = "not a GBD location"
    say(f"      {r['row_id']}: {r['site_label']}, {r['pct_admitted']:g}% ({why})")
say(f"    with percentage and HAQ: {len(usable)}",
    f"    left out as overlapping another row: {usable['row_id'].isin(OVERLAP).sum()}")
for rid, why in OVERLAP.items():
    say(f"      {rid}: {why}")
say(f"    left out as the second of two conflicting values: {usable['row_id'].isin(ALTERNATE).sum()}")
for rid, (_, why) in ALTERNATE.items():
    say(f"      {rid}: {why}")
say(f"    PRIMARY SET: {len(primary)} rows, {primary['site_key'].nunique()} sites, {primary['country'].nunique()} countries",
    f"    by setting: " + ", ".join(f"{k} {v}" for k, v in primary["setting"].value_counts().items())
    + f"; clinic_system_flag = 1: {int(primary['clinic'].sum())}",
    f"    by evidence grade: " + ", ".join(f"{k} {v}" for k, v in primary["evidence_grade"].fillna("(blank)").value_counts().items()),
    f"    HAQ range: {primary['haq'].min():.1f} to {primary['haq'].max():.1f}; "
    f"rows below HAQ 40: {(primary['haq'] < 40).sum()}; rows below HAQ 50: {(primary['haq'] < 50).sum()}",
    f"    percentage admitted: {primary['pct_admitted'].min():g} to {primary['pct_admitted'].max():g}", "")

# ---------------------------------------------------------------------------
# 4. Primary fit
# ---------------------------------------------------------------------------
say("2. PRIMARY FIT",
    "  Ordinary least squares of logit(percent admitted / 100) on HAQ, unweighted, standard errors clustered by site.",
    "  Unweighted because sample sizes are missing for some rows and between-study spread dwarfs sampling error",
    "  (see the dispersion estimate under the binomial cross-check).")
res0, s0, lines = fmt_fit("  primary", primary)
say(*lines, "")
B0, B1 = s0["const"][0], s0["haq"][0]
SD0 = s0["resid_sd"]
V0 = res0.cov_params().to_numpy()
TCRIT = stats.t.ppf(0.975, s0["df_t"])
primary["fitted_pct"] = 100 * expit(B0 + B1 * primary["haq"])
primary["resid_logit"] = primary["logit_pct"] - (B0 + B1 * primary["haq"])
say("  largest residuals on the logit scale (observed against fitted percent):")
for _, r in primary.reindex(primary["resid_logit"].abs().sort_values(ascending=False).index).head(8).iterrows():
    say(f"    {r['row_id']:<10} {r['site_label'][:34]:<34} {r['year_mid']:>7g}  HAQ {r['haq']:.1f}  observed {r['pct_admitted']:g}%  "
        f"fitted {r['fitted_pct']:.0f}%  residual {r['resid_logit']:+.2f}")
say("")

# ---------------------------------------------------------------------------
# 5. Sensitivity fits
# ---------------------------------------------------------------------------
say("3. SENSITIVITY FITS (same model unless stated)")
grade = primary["evidence_grade"].fillna("")
_, sa, lines = fmt_fit("  (a) excluding secondary and derived-grade rows", primary[~grade.isin(["secondary", "derived"])])
say(*lines, f"    dropped: {', '.join(primary.loc[grade.isin(['secondary', 'derived']), 'row_id'])}")
_, sa2, lines = fmt_fit("  (a2, extra) excluding secondary-grade rows only", primary[grade != "secondary"])
say(*lines, f"    dropped: {', '.join(primary.loc[grade == 'secondary', 'row_id'])}")
_, sb, lines = fmt_fit("  (b) excluding clinic_system_flag == 1 rows", primary[primary["clinic"] == 0])
say(*lines, f"    dropped: {', '.join(primary.loc[primary['clinic'] == 1, 'row_id'])}")
_, sc, lines = fmt_fit("  (c) adding a rural indicator and the clinic flag", primary, ("haq", "rural", "clinic"))
say(*lines, f"    rural rows: {', '.join(primary.loc[primary['rural'] == 1, 'row_id'])}. They are one site, so the rural coefficient is a",
    "    Tianjin effect and its clustered SE is not trustworthy. Clinic rows: " + ", ".join(primary.loc[primary["clinic"] == 1, "row_id"]))
_, sd_, lines = fmt_fit("  (d) adding year_mid (centred at 2000) alongside HAQ", primary, ("haq", "year_c"))
say(*lines, f"    correlation of HAQ with year_mid in the primary set: {primary['haq'].corr(primary['year_mid']):.2f}")
is_hic = primary["notes_file"].str.startswith("hic_europe")
_, se_h, lines = fmt_fit("  (e) high-income and Europe rows only (hic_europe_population_studies.md)", primary[is_hic])
say(*lines)
_, se_l, lines = fmt_fit("  (e) LMIC rows only (lmic_population_studies.md, plus the Sichuan row)", primary[~is_hic])
say(*lines)

loso = []
for key, d_site in primary.groupby("site_key"):
    loso.append((d_site["site_label"].iloc[0], float(fit_ols(primary[primary["site_key"] != key]).params["haq"])))
loso = pd.DataFrame(loso, columns=["site", "slope"]).sort_values("slope")
say("  (f) leave-one-site-out slope",
    f"    range: {loso['slope'].min():.4f} (without {loso.iloc[0]['site']}) to {loso['slope'].max():.4f} (without {loso.iloc[-1]['site']})",
    "    all sites, lowest slope first:")
for _, r in loso.iterrows():
    say(f"      without {r['site'][:44]:<44} {r['slope']:.4f}")

swap = pd.concat([primary[~primary["row_id"].isin([v[0] for v in ALTERNATE.values()])],
                  usable[usable["row_id"].isin(ALTERNATE)]])
_, sg, lines = fmt_fit("  (g, extra) the second value of each conflicting pair swapped in", swap)
say(*lines)
_, sh, lines = fmt_fit("  (h, extra) every direct_incident row with HAQ, overlapping and conflicting rows included", usable)
say(*lines)
_, si, lines = fmt_fit(f"  (i, extra) primary set plus rows with no study year, placed at publication year minus {PUB_LAG}",
                       pd.concat([primary, imputed]))
say(*lines)
for _, r in imputed.iterrows():
    say(f"    added: {r['row_id']} {r['site_label']}, {r['pct_admitted']:g}%, placed at {r['country']} {r['haq_year']:.0f} (HAQ {r['haq']:.1f})")

# Binomial cross-check on rows with a denominator. The quasi-binomial step is
# written out: fit the binomial GLM, estimate the dispersion as Pearson
# chi-square over residual degrees of freedom, and scale the SEs by its root.
g = primary[primary["n_strokes"].notna()].copy()
g["k"] = g["n_admitted"].where(g["n_admitted"].notna(), (g["pct_admitted"] / 100 * g["n_strokes"]).round())


def fit_quasibinomial(d):
    endog = np.column_stack([d["k"], d["n_strokes"] - d["k"]])
    res = sm.GLM(endog, sm.add_constant(d[["haq"]].astype(float)), family=sm.families.Binomial()).fit()
    dispersion = res.pearson_chi2 / res.df_resid
    se = res.bse * np.sqrt(dispersion)
    t = stats.t.ppf(0.975, res.df_resid)
    return res.params, se, res.params - t * se, res.params + t * se, dispersion


say("  Binomial GLM cross-check (logit link, weights from n_strokes; quasi-binomial SEs = binomial SEs x sqrt(dispersion))",
    f"    rows with n_strokes: {len(g)} of {len(primary)}; total strokes {g['n_strokes'].sum():.0f}")
for label, d in [("all rows with n_strokes", g), ("without the Netherlands national row (HIC-34)", g[g["row_id"] != "HIC-34"])]:
    b, se, lo, hi, disp = fit_quasibinomial(d)
    say(f"    {label}: {len(d)} rows; the largest row holds {d['n_strokes'].max() / d['n_strokes'].sum():.0%} of all strokes",
        f"      slope per HAQ point {b['haq']:.4f} (95% CI {lo['haq']:.4f} to {hi['haq']:.4f}; quasi-binomial SE {se['haq']:.4f}); "
        f"intercept {b['const']:.3f}",
        f"      dispersion (Pearson chi-square / df): {disp:.0f}, that is, {disp:.0f} times binomial variance")
say("")

# ---------------------------------------------------------------------------
# 6. Within-site time trends
# ---------------------------------------------------------------------------
say("4. WITHIN-SITE TIME TRENDS",
    "  Within-site slope = least-squares slope of logit(percent admitted) on HAQ across a site's waves",
    "  (with two waves, the change in logit divided by the change in HAQ).",
    f"  Cross-sectional slope from the primary fit: {B1:.4f} per HAQ point.")
multi = [k for k, n in primary["site_key"].value_counts().items() if n >= 2]
traj = {}
for key in multi:
    d = primary[primary["site_key"] == key].sort_values("year_mid")
    traj[d["site_label"].iloc[0]] = d
by_id = usable.set_index("row_id")
ALT_SL = "South London (Addo 2011, 3-year periods; alternative series)"
ALT_TARTU = "Tartu (82% in 1991-93, Korv 1999; alternative value)"
traj[ALT_SL] = by_id.loc[["XTR-SL1", "XTR-SL2", "XTR-SL3", "XTR-SL4", "HIC-08b"]].reset_index()
traj[ALT_TARTU] = by_id.loc[["HIC-43b", "HIC-44"]].reset_index()
within_rows = []
for name, d in traj.items():
    slope = float(np.polyfit(d["haq"], d["logit_pct"], 1)[0])
    per_year = float(np.polyfit(d["year_mid"], d["logit_pct"], 1)[0])
    first, last = d.iloc[0], d.iloc[-1]
    within_rows.append({
        "site": name, "in_primary": name not in (ALT_SL, ALT_TARTU), "country": first["country"], "waves": len(d),
        "years": f"{first['year_start']:.0f}-{first['year_end']:.0f} to {last['year_start']:.0f}-{last['year_end']:.0f}",
        "pct": " > ".join(f"{v:g}" for v in d["pct_admitted"]),
        "haq_first": float(first["haq"]), "haq_first_year": int(first["haq_year"]),
        "haq_last": float(last["haq"]), "haq_last_year": int(last["haq_year"]),
        "slope": slope, "ratio": slope / B1, "per_year": per_year,
    })
within = pd.DataFrame(within_rows).sort_values(["in_primary", "slope"], ascending=[False, False]).reset_index(drop=True)
say("  site | waves and periods | percent admitted by wave | HAQ at first and last wave (country, year) |",
    "  within-site slope per HAQ point | ratio to cross-sectional slope | within-site slope per calendar year")
for _, r in within.iterrows():
    say(f"    {r['site']}",
        f"      {r['waves']} waves, {r['years']}; admitted {r['pct']}%; "
        f"HAQ {r['haq_first']:.1f} ({r['country']} {r['haq_first_year']}) and {r['haq_last']:.1f} ({r['haq_last_year']}); "
        f"slope {r['slope']:.4f} per HAQ point; ratio {r['ratio']:.1f}; {r['per_year']:.3f} logit units per calendar year")
main_within = within[within["in_primary"]]
say(f"  across the {len(main_within)} multi-wave sites in the primary set: within-site slopes run from "
    f"{main_within['slope'].min():.4f} to {main_within['slope'].max():.4f}; median {main_within['slope'].median():.4f}",
    f"  per calendar year the same sites run from {main_within['per_year'].min():.3f} to {main_within['per_year'].max():.3f} logit units; "
    f"the cross-sectional year coefficient in sensitivity (d) is {sd_['year_c'][0]:.3f}")

# Pooled within-site slope: site fixed effects on the multi-wave sites
mw = primary[primary["site_key"].isin(multi)].reset_index(drop=True)
X = pd.concat([mw[["haq"]].astype(float), pd.get_dummies(mw["site_key"], dtype=float)], axis=1)
fe = sm.OLS(mw["logit_pct"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(mw["site_key"])[0]}, use_t=True)
fe_ci = fe.conf_int().loc["haq"]
say(f"  pooled within-site slope (site fixed effects; {len(mw)} rows, {mw['site_key'].nunique()} sites): {fe.params['haq']:.4f} "
    f"(95% CI {fe_ci[0]:.4f} to {fe_ci[1]:.4f}; clustered by site with {fe.df_resid_inference:.0f} df, so the interval is rough)")
# Between-site slope: one point per site (means over its waves)
site_means = primary.groupby("site_key")[["haq", "logit_pct"]].mean()
bt = sm.OLS(site_means["logit_pct"], sm.add_constant(site_means[["haq"]])).fit()
bt_ci = bt.conf_int().loc["haq"]
say(f"  between-site slope (site means, {len(site_means)} sites, classical SE): {bt.params['haq']:.4f} (95% CI {bt_ci[0]:.4f} to {bt_ci[1]:.4f})",
    f"  ratio of pooled within-site slope to between-site slope: {fe.params['haq'] / bt.params['haq']:.0f}", "")

# ---------------------------------------------------------------------------
# 7. The IHME injuries rule
# ---------------------------------------------------------------------------
primary["rule_pct"] = injuries_rule(primary["haq"])
diff = primary["pct_admitted"] - primary["rule_pct"]
top = primary[primary["haq"] >= RULE_TOP_HAQ]
low = primary[primary["haq"] < RULE_TOP_HAQ]
above = primary[diff > 0].sort_values("haq")
say("5. COMPARISON WITH THE IHME INJURIES RULE",
    "  Rule as quoted in gbd_pipeline_evidence.md (GBD 2023 nonfatal appendix, injuries, p.724): 100% treated at HAQ 75 and above;",
    "  other location-years 'scaled ... between 10% and 100% treated based on their HAQ Index value'.",
    "  The form between the anchors is not published. ASSUMED HERE: a straight line from 10% at HAQ 0 to 100% at HAQ 75.",
    f"  primary rows above the rule / below the rule: {(diff > 0).sum()} / {(diff < 0).sum()} of {len(primary)}",
    "  rows above the rule: " + "; ".join(f"{r['site_label']} {r['pct_admitted']:g}% (rule {r['rule_pct']:.0f}%)" for _, r in above.iterrows()),
    f"  mean (observed minus rule): {diff.mean():+.1f} percentage points; median {diff.median():+.1f}",
    f"  root mean square difference from the rule: {np.sqrt((diff ** 2).mean()):.1f} points; "
    f"from the fitted curve: {np.sqrt(((primary['pct_admitted'] - primary['fitted_pct']) ** 2).mean()):.1f} points",
    f"  HAQ 75 and above (rule says 100%): {len(top)} rows, observed {top['pct_admitted'].min():g}% to {top['pct_admitted'].max():g}%, "
    f"median {top['pct_admitted'].median():.1f}%; rows under 90%: {(top['pct_admitted'] < 90).sum()} "
    f"({', '.join(top.loc[top['pct_admitted'] < 90, 'site_label'] + ' ' + top.loc[top['pct_admitted'] < 90, 'pct_admitted'].map('{:g}%'.format))})",
    f"  below HAQ 75: {len(low)} rows; observed minus rule runs from {(low['pct_admitted'] - low['rule_pct']).min():+.1f} to "
    f"{(low['pct_admitted'] - low['rule_pct']).max():+.1f} points, mean {(low['pct_admitted'] - low['rule_pct']).mean():+.1f}",
    "  Where the 10% anchor sits is not stated. Three hypothetical positions (not GBD values) show how much it matters below HAQ 75:",
    "    10% anchor at HAQ | rule at HAQ 30 / 50 / 70 | primary rows below the rule | mean (observed minus rule), rows under HAQ 75")
for a in (0, 10, 20):
    rule_a = injuries_rule(primary["haq"], a)
    d_a = primary["pct_admitted"] - rule_a
    say(f"    {a:>3} | {float(injuries_rule(30, a)):.0f} / {float(injuries_rule(50, a)):.0f} / {float(injuries_rule(70, a)):.0f} | "
        f"{(d_a < 0).sum()} of {len(primary)} | {d_a[primary['haq'] < RULE_TOP_HAQ].mean():+.1f}")
say("  rule (10% anchor at HAQ 0) against the fitted curve, percent:")
for h in PRED_GRID:
    say(f"    HAQ {h}: rule {float(injuries_rule(h)):.0f}, fitted {100 * expit(B0 + B1 * h):.0f}")
say("")

# ---------------------------------------------------------------------------
# 8. Illustrative predictions
# ---------------------------------------------------------------------------


def predict(h):
    """Fitted share, 95% interval for the fitted mean, and 95% prediction interval, in percent."""
    h = np.atleast_1d(np.asarray(h, dtype=float))
    X0 = np.column_stack([np.ones_like(h), h])
    eta = X0 @ np.array([B0, B1])
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", X0, V0, X0))
    se_pred = np.sqrt(SD0 ** 2 + se_mean ** 2)
    return (100 * expit(eta), 100 * expit(eta - TCRIT * se_mean), 100 * expit(eta + TCRIT * se_mean),
            100 * expit(eta - TCRIT * se_pred), 100 * expit(eta + TCRIT * se_pred))


say("6. ILLUSTRATIVE PREDICTIONS FROM THE PRIMARY FIT (not estimates, not for use as inputs)",
    f"  Prediction interval for a new study-period: fitted logit +/- t({s0['df_t']:.0f}) x sqrt(residual SD^2 + SE of fit^2), back-transformed.",
    "  HAQ | fitted % admitted | 95% CI of the fitted mean | 95% prediction interval")
fit_, lo_m, hi_m, lo_p, hi_p = predict(PRED_GRID)
for i, h in enumerate(PRED_GRID):
    flag = "" if primary["haq"].min() <= h <= primary["haq"].max() else "  (outside the HAQ range of the data)"
    say(f"    {h:>3} | {fit_[i]:5.1f} | {lo_m[i]:5.1f} to {hi_m[i]:5.1f} | {lo_p[i]:5.1f} to {hi_p[i]:5.1f}{flag}")
say("")

# ---------------------------------------------------------------------------
# 9. MONICA (a different measure) and the linked-data upper bound
# ---------------------------------------------------------------------------
say("7. WHO MONICA ROWS (non-fatal events at ages 25-64 managed in hospital; not pooled with the primary set)")
_, sm1, lines = fmt_fit("  logit scale (shares above 99.5% set to 99.5%)", monica)
say(*lines, f"    populations at the 99.5% limit: {(monica['pct_admitted'] > 99.5).sum()}")
_, sm2, lines = fmt_fit("  percentage-point scale (no transformation)", monica, ycol="pct_admitted")
say(*lines, f"    HAQ range {monica['haq'].min():.1f} to {monica['haq'].max():.1f}; percent in hospital {monica['pct_admitted'].min():g} to {monica['pct_admitted'].max():g}", "")
say("8. LINKED-DATA UPPER BOUND (shown in the figure, not in any fit)")
for _, r in upper.iterrows():
    say(f"  {r['row_id']}: {r['country']} {r['year_start']:.0f}-{r['year_end']:.0f}, {r['pct_admitted']:g}% of registered incident strokes hospitalised; "
        f"HAQ {r['haq']:.1f} ({r['haq_year']:.0f}); primary fit at that HAQ {float(predict(r['haq'])[0][0]):.0f}%")
say("")

# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})
LEADER = {"arrowstyle": "-", "color": "#9a9a9a", "linewidth": 0.8, "shrinkA": 2, "shrinkB": 5}


def setting_group(s):
    return s if s in ("urban", "rural") else "other"


SETTING_COLOUR = {"urban": BLUE, "rural": ORANGE, "other": AQUA}
primary["colour"] = primary["setting"].map(setting_group).map(SETTING_COLOUR)
imputed["colour"] = imputed["setting"].map(setting_group).map(SETTING_COLOUR)

# --- Figure 1: scatter -------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.6, 7.0), dpi=200)
fig.subplots_adjust(left=0.07, right=0.70, top=0.86, bottom=0.11)
hh = np.linspace(20, 95, 301)
f_, _, _, lo_, hi_ = predict(hh)
ax.fill_between(hh, lo_, hi_, color=INK, alpha=0.07, linewidth=0, zorder=1)
ax.plot(hh, f_, color=INK, linewidth=2, zorder=3)
ax.plot(hh, injuries_rule(hh), color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
ax.scatter(monica["haq"], monica["pct_admitted"], marker="x", s=26, color="#8f8f8f", linewidths=1.2, zorder=4)
ax.scatter(upper["haq"], upper["pct_admitted"], marker="v", s=64, facecolor="white", edgecolor=INK2, linewidths=1.6, zorder=5)
ax.scatter(imputed["haq"], imputed["pct_admitted"], marker="o", s=62, facecolor="white", edgecolor=imputed["colour"], linewidths=1.8, zorder=5)
for clinic, marker, size in [(0, "o", 78), (1, "D", 62)]:
    d = primary[primary["clinic"] == clinic]
    ax.scatter(d["haq"], d["pct_admitted"], marker=marker, s=size, color=d["colour"], edgecolor="white", linewidths=1.0, zorder=6)

# direct labels: row_id -> (text, offset in points, horizontal alignment)
LABELS = {
    "LMIC-A1": ("Rural Tianjin 1992-2000", (9, -3), "left"),
    "LMIC-A2": ("Rural Tianjin 2001-09", (9, -3), "left"),
    "LMIC-A3": ("Rural Tianjin 2010-18", (-9, -1), "right"),
    "LMIC-B1": ("Mumbai 2005-06", (9, -9), "left"),
    "LMIC-B2": ("Kolkata 2003-05", (9, 6), "left"),
    "HIC-48": ("Krasnodar 1998", (-9, -3), "right"),
    "HIC-47": ("Varna 2000-01", (9, -5), "left"),
    "HIC-01": ("Oxfordshire 1981-86", (0, 12), "center"),
    "HIC-02": ("Oxfordshire 2002-17", (9, -3), "left"),
    "HIC-33": ("Porto 2009-11", (-9, 8), "right"),
    "HIC-34": ("Netherlands 2000", (9, -3), "left"),
    "HIC-15": ("Auckland 1981-82", (9, -2), "left"),
    "HIC-19": ("Auckland 2021-22", (6, 10), "left"),
    "HIC-30": ("Dijon 2013-20", (9, -5), "left"),
    "LMIC-D4": ("Matão 2003-04", (-9, 3), "right"),
    "LMIC-B3": ("Rural West Bengal (year assumed)", (9, -3), "left"),
}
lab_src = pd.concat([primary, imputed]).set_index("row_id")
for rid, (text, off, ha) in LABELS.items():
    r = lab_src.loc[rid]
    ax.annotate(text, (r["haq"], r["pct_admitted"]), xytext=off, textcoords="offset points",
                ha=ha, va="center", fontsize=8.5, color=INK, zorder=7)
pair = imputed[imputed["row_id"].isin(["HIC-45", "XTR-TBILISI"])]
ax.annotate("Tbilisi and Uzhgorod\n(years assumed)", (pair["haq"].mean(), pair["pct_admitted"].mean()), xytext=(-46, 40),
            textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK, zorder=7, arrowprops=LEADER)
for _, r in upper.iterrows():
    ax.annotate("Norway 2001-21, linked\nregisters (upper bound)", (r["haq"], r["pct_admitted"]), xytext=(-38, -50),
                textcoords="offset points", ha="left", va="center", fontsize=8.5, color=INK, zorder=7, arrowprops=LEADER)
ax.annotate("IHME injuries rule\n(assumed straight line)", (25.5, float(injuries_rule(25.5))), xytext=(14, -27),
            textcoords="offset points", ha="left", va="top", fontsize=8.5, color=INK2, arrowprops=LEADER)

ax.set_xlim(20, 100)
ax.set_ylim(0, 104)
ax.set_xticks(range(20, 101, 10))
ax.set_yticks(range(0, 101, 10))
ax.grid(True)
ax.set_xlabel("Healthcare Access and Quality Index of the country at the study's mid-year (GBD 2023, scale 0 to 100)")
ax.set_ylabel("Incident strokes admitted to hospital (%)")
fig.text(0.07, 0.955, "National HAQ Index is a loose guide to the share of strokes admitted to hospital", fontsize=13.5, color=INK, ha="left", va="center")
fig.text(0.07, 0.915, f"{len(primary)} study-periods from {primary['site_key'].nunique()} sites in {primary['country'].nunique()} countries. "
         f"Fitted curve: logit(share) = {B0:.2f} + {B1:.3f} x HAQ, residual SD {SD0:.2f} on the logit scale. A diagnostic, not an estimate.",
         fontsize=9, color=INK2, ha="left", va="center")

LEG = {"loc": "upper left", "fontsize": 9, "title_fontsize": 9, "alignment": "left"}
fig.legend(handles=[
    Line2D([], [], marker="o", linestyle="", markersize=8.5, color=BLUE, markeredgecolor="white", label="urban"),
    Line2D([], [], marker="o", linestyle="", markersize=8.5, color=ORANGE, markeredgecolor="white", label="rural"),
    Line2D([], [], marker="o", linestyle="", markersize=8.5, color=AQUA, markeredgecolor="white", label="mixed, national or not stated"),
], title="Study site (colour)", bbox_to_anchor=(0.715, 0.86), **LEG)
fig.legend(handles=[
    Line2D([], [], marker="D", linestyle="", markersize=7.5, color=INK2, markeredgecolor="white",
           label="non-admitted strokes mostly seen in a\nclinic or emergency department"),
    Line2D([], [], marker="o", linestyle="", markersize=8.5, color=INK2, markeredgecolor="white", label="no such pathway noted"),
], title="Care system (shape)", bbox_to_anchor=(0.715, 0.70), **LEG)
fig.legend(handles=[
    Line2D([], [], color=INK, linewidth=2, label="fitted curve (primary set)"),
    Patch(facecolor=INK, alpha=0.07, label="95% prediction band for a\nnew study-period"),
    Line2D([], [], color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), label="IHME injuries rule; exact form not\npublished, so a straight line is assumed\nfrom 10% at HAQ 0 to 100% at HAQ 75+"),
], title="Lines", bbox_to_anchor=(0.715, 0.545), **LEG)
fig.legend(handles=[
    Line2D([], [], marker="o", linestyle="", markersize=7.5, markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=1.8,
           label=f"study year not in the notes; placed\nat publication year minus {PUB_LAG}"),
    Line2D([], [], marker="x", linestyle="", markersize=5.5, color="#8f8f8f", markeredgewidth=1.2,
           label="WHO MONICA: non-fatal events at\nages 25-64 managed in hospital"),
    Line2D([], [], marker="v", linestyle="", markersize=7.5, markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=1.6,
           label="linked registers: an upper bound"),
], title="Not in the fit", bbox_to_anchor=(0.715, 0.325), **LEG)
fig.text(0.07, 0.022, "HAQ values are national, also for city studies. Sources: study percentages from the research notes; HAQ from IHME, GBD 2023.",
         fontsize=8, color=INK2, ha="left", va="center")
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "admitted_share_vs_haq.png")
plt.close(fig)

# --- Figure 2: within-site trajectories, one panel per site, logit axis ------
# Every panel shows all sites (grey) and the cross-sectional fit; the panel's
# own site is drawn in blue with its wave years.
SHORT = {"Yangjinzhuang township, Tianjin": "Rural Tianjin"}
WAVE_OFFSET = {  # row_id -> (offset in points, alignment) of the wave label
    "HIC-15": ((-7, 0), "right"), "HIC-16": ((-7, 3), "right"), "HIC-17": ((-3, 9), "right"), "HIC-18": ((8, -1), "left"), "HIC-19": ((-8, 0), "right"),
    "HIC-06b": ((6, -9), "left"), "HIC-07": ((8, 0), "left"), "HIC-08a": ((-7, 4), "right"),
    "HIC-43a": ((7, -8), "left"), "HIC-44": ((8, 0), "left"),
    "LMIC-A1": ((7, -6), "left"), "LMIC-A2": ((7, -6), "left"), "LMIC-A3": ((-8, 4), "right"),
    "HIC-01": ((0, -11), "center"), "HIC-02": ((0, -11), "center"),
    "HIC-28": ((0, 10), "center"), "HIC-30": ((-4, 10), "center"),
}
panels = list(main_within["site"])
ncol = 3
nrow = int(np.ceil(len(panels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(12.0, 3.55 * nrow + 1.1), dpi=200, sharex=True, sharey=True, squeeze=False)
fig.subplots_adjust(left=0.065, right=0.985, top=1 - 1.35 / (3.55 * nrow + 1.1), bottom=0.085, wspace=0.07, hspace=0.36)
hh = np.linspace(36, 95, 50)
pct_ticks = [5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 99]
for i, ax in enumerate(axes.flat):
    if i >= len(panels):
        ax.set_visible(False)
        continue
    site = panels[i]
    w = main_within[main_within["site"] == site].iloc[0]
    for other in panels:
        if other != site:
            ax.plot(traj[other]["haq"], traj[other]["logit_pct"], color="#d6d6d6", linewidth=1.3, zorder=1)
    ax.plot(hh, B0 + B1 * hh, color=INK2, linewidth=1.6, zorder=2)
    d = traj[site]
    if site == "Tartu":  # the conflicting 1991-93 value
        alt = traj[ALT_TARTU]
        ax.plot(alt["haq"], alt["logit_pct"], color=BLUE, linewidth=1.3, linestyle=(0, (2, 2)), zorder=3)
        ax.scatter(alt["haq"].iloc[:1], alt["logit_pct"].iloc[:1], s=44, facecolor="white", edgecolor=BLUE, linewidths=1.6, zorder=4)
        ax.annotate("82% in a\nsecond source", (alt["haq"].iloc[0], alt["logit_pct"].iloc[0]), xytext=(-8, 3), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK2)
    ax.plot(d["haq"], d["logit_pct"], color=BLUE, linewidth=2, zorder=4, solid_capstyle="round")
    ax.scatter(d["haq"], d["logit_pct"], s=54, color=BLUE, edgecolor="white", linewidths=1.0, zorder=5)
    for _, r in d.iterrows():
        off, ha = WAVE_OFFSET.get(r["row_id"], ((7, -7), "left"))
        y0, y1 = int(r["year_start"]), int(r["year_end"])
        years = f"{y0}-{y1}" if y0 // 100 != y1 // 100 else f"{y0}-{str(y1)[2:]}"
        ax.annotate(years, (r["haq"], r["logit_pct"]), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=8, color=INK, zorder=6)
    ax.set_title(SHORT.get(site, site), loc="left", fontsize=10.5, color=INK, pad=19)
    ax.text(0, 1.03, f"slope {w['slope']:.2f} per HAQ point, {w['ratio']:.1f} x the cross-sectional slope",
            transform=ax.transAxes, fontsize=8.5, color=INK2, ha="left", va="bottom")
    ax.set_yticks([logit(p / 100) for p in pct_ticks])
    ax.set_yticklabels([str(p) for p in pct_ticks])
    ax.set_ylim(logit(0.045), logit(0.9925))
    ax.set_xlim(36, 95)
    ax.grid(True)
    ax.tick_params(labelbottom=True)
    if i == 0:
        ax.annotate(f"cross-sectional fit\n({B1:.3f} per HAQ point)", (46, B0 + B1 * 46), xytext=(4, -30), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK2, arrowprops=LEADER)
fig.supxlabel("Healthcare Access and Quality Index of the country at each wave's mid-year (GBD 2023, scale 0 to 100)", fontsize=10, color=INK2, y=0.012)
fig.supylabel("Incident strokes admitted to hospital (%, logit scale)", fontsize=10, color=INK2, x=0.008)
fig_h = 3.55 * nrow + 1.1
fig.text(0.065, 1 - 0.33 / fig_h, "Within a site, admission rises much faster per HAQ point than the cross-sectional fit, or no faster at all",
         fontsize=13, color=INK, ha="left", va="center")
fig.text(0.065, 1 - 0.65 / fig_h, f"Blue: the site's waves, labelled with their years. Light grey: the other {len(panels) - 1} sites. Dark grey: "
         "cross-sectional fit to the primary set. Slopes are in logit units. A diagnostic, not an estimate.", fontsize=9, color=INK2, ha="left", va="center")
fig.savefig(FIGS / "within_site_trajectories.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Write outputs
# ---------------------------------------------------------------------------
OUT.mkdir(exist_ok=True)
keep = ["row_id", "study_label", "country", "site", "setting", "year_start", "year_end", "year_mid", "haq_year", "haq",
        "pct_admitted", "logit_pct", "fitted_pct", "resid_logit", "rule_pct", "n_strokes", "clinic_system_flag", "evidence_grade", "notes_file"]
primary[keep].sort_values(["country", "site", "year_mid"]).to_csv(OUT / "primary_analysis_set.csv", index=False, float_format="%.6g")
say("9. FILES",
    "  outputs/primary_analysis_set.csv: the rows behind the primary fit",
    "  figures/admitted_share_vs_haq.png, figures/within_site_trajectories.png")
(OUT / "first_look_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
