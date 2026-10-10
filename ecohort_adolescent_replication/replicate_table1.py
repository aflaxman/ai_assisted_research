"""Partial replication of Amboko et al. (2026) Table 1 and the first-ANC rows of Table 2
using the open eCOANC1 analytic file (doi:10.7910/DVN/SSVXKY), restricted to
Ethiopia, Kenya and South Africa.

Adolescents = enrolment age 15-19; adults = 20+ (same cut as the paper).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).parent / "data"
df = pd.read_stata(DATA / "eCOANC1.dta", convert_categoricals=False)
df = df[df.country.isin(["Ethiopia", "Kenya", "South Africa"])].copy()
df["adol"] = (df.enrollage <= 19).astype(int)

# ---- derived variables, following Amboko_analysis_2025.do where the inputs exist ----
df["rural"] = df.site.isin([0, 2, 6]).astype(float)          # site codes: 0 Rural-ETH, 2 Rural-KEN, 6 Rural-ZAF
df["married"] = df.marriedp
df["secondary_plus"] = df.second
df["poorest"] = (df.tertile == 1).astype(float).where(df.tertile.notna())
df["middle"] = (df.tertile == 2).astype(float).where(df.tertile.notna())
df["richest"] = (df.tertile == 3).astype(float).where(df.tertile.notna())
df["healthlit"] = df.healthlit_corr
df["tri1"] = (df.trimester == 1).astype(float).where(df.trimester.notna())
df["tri2"] = (df.trimester == 2).astype(float).where(df.trimester.notna())
df["tri3"] = (df.trimester == 3).astype(float).where(df.trimester.notna())
df["intended"] = df.preg_intent.where(df.preg_intent.isin([0, 1]))
df["underweight"] = df.maln_underw
df["dangersign"] = df.m1_dangersigns
# The paper's do-file uses `ANCdepression` (PHQ-9 mild-to-severe, = `depress` here) for Ethiopia and
# Kenya but `anc1depression` for South Africa; following that reproduces Table 1 exactly.
df["depression"] = np.where(df.country == "South Africa", df.anc1depression, df.depress)
df["poorfair_health"] = df.poorhealth  # already a 0/1 flag for fair/poor in eCOANC1
df["primigravida"] = df.primipara

# First-ANC completeness index (screening/treatments). The paper's do-file averages 13 items:
# BP, weight, height, MUAC, HIV test, syphilis test, blood sugar, urine, Hb, ultrasound, IFA, calcium, TT.
# eCOANC1 collapses the blood tests into one `anc1blood` and `ultrasound`/`calcium`/`tt` are mostly
# missing, so this 7-item version is only an approximation and is NOT expected to match.
screen_items = ["anc1bp", "anc1weight", "anc1height", "anc1muac", "anc1blood", "anc1urine", "anc1ifa"]
df["anc1_screen_idx"] = 100 * df[screen_items].mean(axis=1, skipna=True)
# Counselling index: paper averages nutrition, exercise, danger signs (716e), birth plan (809), 724a.
# The m1_counsel_* items exist only for Ethiopia in eCOANC1, so Kenya/South Africa are blank.
counsel_items = ["m1_counsel_nutri", "m1_counsel_exer", "m1_counsel_complic", "m1_counsel_birthplan",
                 "m1_counsel_comeback"]
df["anc1_counsel_idx"] = 100 * df[counsel_items].mean(axis=1, skipna=False)

# ---- paper values (Table 1 / Table 2), adolescents vs adults, per country and total ----
P = {
 # row: {country: (adol, adult)}  strings as printed in the paper
 "Age, mean (SD)": {"Ethiopia": ("18.2 (0.9)", "26.1 (4.5)"), "Kenya": ("18.0 (1.0)", "28.0 (5.7)"),
                    "South Africa": ("18.0 (0.97)", "28.4 (5.71)"), "Total": ("18.1 (0.9)", "27.5 (5.4)")},
 "Married or partnered": {"Ethiopia": ("82 (95.4)", "890 (97.5)"), "Kenya": ("57 (47.5)", "730 (82.5)"),
                          "South Africa": ("2 (1.2)", "124 (14.3)"), "Total": ("141 (37.2)", "1744 (65.4)")},
 "Completed secondary or higher": {"Ethiopia": ("1 (1.2)", "219 (24.0)"), "Kenya": ("42 (34.7)", "546 (61.5)"),
                          "South Africa": ("83 (48.0)", "652 (75.2)"), "Total": ("126 (33.2)", "1417 (53.1)")},
 "Wealth: poorest tertile": {"Ethiopia": ("32 (37.6)", "166 (18.5)"), "Kenya": ("48 (39.7)", "154 (17.5)"),
                          "South Africa": ("40 (23.1)", "224 (25.8)"), "Total": ("120 (31.7)", "544 (20.6)")},
 "Wealth: middle tertile": {"Ethiopia": ("48 (56.5)", "166 (18.5)"), "Kenya": ("63 (52.1)", "545 (61.8)"),
                          "South Africa": ("72 (41.6)", "338 (38.9)"), "Total": ("183 (48.3)", "1423 (53.8)")},
 "Wealth: richest tertile": {"Ethiopia": ("5 (5.9)", "540 (60.3)"), "Kenya": ("10 (8.3)", "183 (20.8)"),
                          "South Africa": ("61 (35.3)", "306 (35.3)"), "Total": ("76 (20.1)", "679 (25.7)")},
 "Rural": {"Ethiopia": ("60 (69.8)", "448 (49.0)"), "Kenya": ("89 (73.6)", "418 (47.0)"),
           "South Africa": ("115 (66.5)", "400 (46.0)"), "Total": ("264 (69.5)", "1265 (47.4)")},
 "Very good health literacy": {"Ethiopia": ("5 (5.8)", "261 (28.6)"), "Kenya": ("35 (28.9)", "416 (46.8)"),
           "South Africa": ("35 (20.2)", "310 (35.7)"), "Total": ("75 (19.7)", "987 (37.0)")},
 "First ANC in 1st trimester": {"Ethiopia": ("18 (21.4)", "286 (31.9)"), "Kenya": ("9 (7.4)", "147 (16.6)"),
           "South Africa": ("59 (34.3)", "366 (42.2)"), "Total": ("86 (22.8)", "799 (30.1)")},
 "First ANC in 2nd trimester": {"Ethiopia": ("60 (71.4)", "555 (61.9)"), "Kenya": ("74 (61.2)", "564 (63.5)"),
           "South Africa": ("89 (51.7)", "418 (48.2)"), "Total": ("223 (59.2)", "1537 (58.0)")},
 "First ANC in 3rd trimester": {"Ethiopia": ("6 (7.1)", "55 (6.1)"), "Kenya": ("38 (31.4)", "177 (19.9)"),
           "South Africa": ("24 (14.0)", "84 (9.7)"), "Total": ("68 (18.0)", "316 (11.9)")},
 "Pregnancy was intended": {"Ethiopia": ("61 (70.9)", "667 (73.1)"), "Kenya": ("48 (40.0)", "567 (64.7)"),
           "South Africa": ("9 (5.2)", "172 (19.8)"), "Total": ("118 (31.2)", "1405 (52.9)")},
 "Underweight (MUAC<23 or BMI<18.5)": {"Ethiopia": ("33 (38.4)", "239 (26.2)"), "Kenya": ("9 (7.4)", "37 (4.2)"),
           "South Africa": ("5 (2.9)", "12 (1.4)"), "Total": ("47 (12.4)", "288 (10.8)")},
 "At least one danger sign": {"Ethiopia": ("21 (24.4)", "221 (24.2)"), "Kenya": ("25 (20.7)", "208 (23.4)"),
           "South Africa": ("82 (47.7)", "400 (46.4)"), "Total": ("128 (33.8)", "829 (31.1)")},
 "ANC depression (PHQ9>5)": {"Ethiopia": ("16 (18.6)", "231 (25.3)"), "Kenya": ("21 (17.4)", "179 (20.2)"),
           "South Africa": ("34 (20.0)", "116 (13.4)"), "Total": ("71 (18.8)", "526 (19.7)")},
 "Rates own health poor/fair": {"Ethiopia": ("10 (11.6)", "160 (17.5)"), "Kenya": ("7 (5.8)", "80 (9.0)"),
           "South Africa": ("18 (10.4)", "64 (7.4)"), "Total": ("35 (9.2)", "304 (11.4)")},
 "T2: First ANC completeness index, screening (mean (SD))": {"Ethiopia": ("49.4 (14.9)", "50.9 (15.5)"),
           "Kenya": ("70.9 (15.1)", "69.3 (14.5)"), "South Africa": ("85.4 (7.9)", "84.6 (8.7)"),
           "Total": ("72.6 (18.6)", "68.0 (19.2)")},
 "T2: First ANC completeness index, counselling (mean (SD))": {"Ethiopia": ("31.3 (18.9)", "33.7 (20.7)"),
           "Kenya": ("50.3 (30.0)", "60.6 (29.6)"), "South Africa": ("56.7 (27.6)", "57.8 (27.8)"),
           "Total": ("48.9 (28.5)", "50.5 (28.9)")},
}

ROWS = [  # (label, variable, kind)
 ("Age, mean (SD)", "enrollage", "mean"),
 ("Married or partnered", "married", "pct"),
 ("Completed secondary or higher", "secondary_plus", "pct"),
 ("Wealth: poorest tertile", "poorest", "pct"),
 ("Wealth: middle tertile", "middle", "pct"),
 ("Wealth: richest tertile", "richest", "pct"),
 ("Rural", "rural", "pct"),
 ("Very good health literacy", "healthlit", "pct"),
 ("First ANC in 1st trimester", "tri1", "pct"),
 ("First ANC in 2nd trimester", "tri2", "pct"),
 ("First ANC in 3rd trimester", "tri3", "pct"),
 ("Pregnancy was intended", "intended", "pct"),
 ("Underweight (MUAC<23 or BMI<18.5)", "underweight", "pct"),
 ("At least one danger sign", "dangersign", "pct"),
 ("ANC depression (PHQ9>5)", "depression", "pct"),
 ("Rates own health poor/fair", "poorfair_health", "pct"),
 ("T2: First ANC completeness index, screening (mean (SD))", "anc1_screen_idx", "mean"),  # approx. only
 ("T2: First ANC completeness index, counselling (mean (SD))", "anc1_counsel_idx", "mean"),
]

def cell(s, kind):
    s = s.dropna()
    if kind == "mean":
        return f"{s.mean():.1f} ({s.std():.1f})"
    return f"{int(s.sum())} ({100*s.mean():.1f})"

def pval(a, b, kind):
    a, b = a.dropna(), b.dropna()
    if kind == "mean":
        return stats.ttest_ind(a, b, equal_var=True).pvalue
    tab = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    if (tab.sum(axis=0) == 0).any() or (tab.sum(axis=1) == 0).any():
        return np.nan
    return stats.chi2_contingency(tab, correction=False)[1]

groups = {c: df[df.country == c] for c in ["Ethiopia", "Kenya", "South Africa"]}
groups["Total"] = df

out = []
for c, g in groups.items():
    a, b = g[g.adol == 1], g[g.adol == 0]
    out.append({"country": c, "row": "N", "open: adolescents": len(a), "open: adults": len(b),
                "paper: adolescents": {"Ethiopia": 86, "Kenya": 121, "South Africa": 173, "Total": 380}[c],
                "paper: adults": {"Ethiopia": 914, "Kenya": 888, "South Africa": 869, "Total": 2671}[c], "open p": ""})
    for label, var, kind in ROWS:
        out.append({"country": c, "row": label,
                    "open: adolescents": cell(a[var], kind), "open: adults": cell(b[var], kind),
                    "paper: adolescents": P[label][c][0], "paper: adults": P[label][c][1],
                    "open p": f"{pval(a[var], b[var], kind):.3f}"})
res = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 60)
print(res.to_string(index=False))
res.to_csv(Path(__file__).parent / "table1_comparison.csv", index=False)
with open(Path(__file__).parent / "table1_comparison.md", "w") as f:
    f.write(res.to_markdown(index=False))
