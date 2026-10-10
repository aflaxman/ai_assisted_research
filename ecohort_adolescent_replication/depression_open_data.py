"""What the open eCohort files say about depression during pregnancy.

1. eCOANC1 (4 countries, first ANC visit): PHQ-9 screen-positive prevalence, whether the
   provider asked about / referred for mental health, and simple correlates.
2. econut (Ethiopia, 811 women): PHQ-9 items repeated at each monthly phone follow-up,
   giving within-pregnancy trajectories.
"""
from pathlib import Path
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- eCOANC1
d = pd.read_stata(DATA / "eCOANC1.dta", convert_categoricals=False)
d["adol"] = (d.enrollage <= 19)
yn = lambda s: s.where(s.isin([0, 1]))
d["asked_anx_dep"] = yn(d.m1_716c)        # 716c discussed anxiety/depression
d["discussed_mh"] = yn(d.m1_721)          # 721 discussed mental health (asked of those with prior dx?)
d["referred_mh"] = yn(d.m1_724d)          # 724d told to see a mental health provider
d["prior_mh_dx"] = yn(d.m1_202d)          # knew of mental health disorder before pregnancy
d["mh_drug"] = yn(d.anc1mental_health_drug)
d["hiv_known"] = yn(d.m1_202e)
d["intended"] = yn(d.preg_intent)

print("=" * 90); print("eCOANC1: PHQ-9 mild-or-worse (`depress`) vs `anc1depression`, by country")
print(d.groupby("country").agg(n=("depress", "size"), phq9_mild_plus=("depress", "mean"),
                               anc1depression=("anc1depression", "mean"),
                               asked_anx_dep=("asked_anx_dep", "mean"), discussed_mh=("discussed_mh", "mean"),
                               referred_mh=("referred_mh", "mean"), prior_mh_dx=("prior_mh_dx", "mean"),
                               mh_drug=("mh_drug", "mean")).round(3))
print("\nagreement between depress and anc1depression (all countries):")
print(pd.crosstab(d.depress, d.anc1depression, margins=True))

print("\nPHQ-9 mild+ by country x adolescent:")
print(d.groupby(["country", "adol"]).depress.agg(["size", "mean"]).round(3).unstack())

print("\nDetection gap: among PHQ-9 screen-positive women, share asked about anxiety/depression and referred:")
print(d.groupby(["country", "depress"]).agg(n=("asked_anx_dep", "size"), asked=("asked_anx_dep", "mean"),
                                           referred=("referred_mh", "mean")).round(3))

print("\nCorrelates of PHQ-9 mild+ (row %), pooled 4 countries:")
for v in ["adol", "marriedp", "intended", "tertile", "healthlit_corr", "m1_dangersigns", "hiv_known",
          "prior_mh_dx", "trimester", "primipara", "poorhealth", "anemic", "maln_underw"]:
    t = d.groupby(v).depress.agg(["size", "mean"]).round(3)
    print(f"  {v}: " + ", ".join(f"{k}: {r['mean']:.2f} (n={int(r['size'])})" for k, r in t.iterrows()))

# ---------------------------------------------------------------- econut (Ethiopia, longitudinal PHQ-9)
e = pd.read_stata(DATA / "econut_analyticdata.dta", convert_categoricals=False)
print("\n" + "=" * 90); print("econut (Ethiopia): PHQ-9 items at follow-up rounds")
items = "abcdefghi"
rounds = range(1, 9)
print("coding of m2_205a_r1:", e.m2_205a_r1.value_counts(dropna=False).to_dict())
rows = []
for r in rounds:
    cols = [f"m2_205{i}_r{r}" for i in items]
    sc = e[cols].where(e[cols].le(3))        # treat 98/99 as missing
    tot = sc.sum(axis=1, min_count=9)
    e[f"phq9_r{r}"] = tot
    rows.append(dict(round=r, interviewed=e[f"m2_date_r{r}"].notna().sum(), phq9_complete=tot.notna().sum(),
                     mean_ga_wk=e[f"m2_ga_r{r}"].mean(), mean_phq9=tot.mean(),
                     pct_ge5=(tot >= 5).sum() / tot.notna().sum() if tot.notna().sum() else np.nan,
                     pct_ge10=(tot >= 10).sum() / tot.notna().sum() if tot.notna().sum() else np.nan,
                     pct_item9_any=(sc[f"m2_205i_r{r}"] >= 1).sum() / sc[f"m2_205i_r{r}"].notna().sum()
                     if sc[f"m2_205i_r{r}"].notna().sum() else np.nan))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nbaseline `depress` (ANC1 PHQ-9 mild+):", e.depress.value_counts(dropna=False).to_dict())
e["adol"] = e.m1_enrollage <= 19
print("baseline depress by adolescent:", e.groupby("adol").depress.agg(["size", "mean"]).round(3).to_dict())
# trajectory: baseline status vs any follow-up PHQ-9>=5 and vs mean follow-up score
fu = e[[f"phq9_r{r}" for r in rounds]]
e["fu_any_ge5"] = (fu >= 5).any(axis=1).where(fu.notna().any(axis=1))
e["fu_mean"] = fu.mean(axis=1)
e["fu_n"] = fu.notna().sum(axis=1)
print("\nfollow-up PHQ-9 by baseline status:")
print(e.groupby("depress").agg(n=("fu_n", "size"), fu_rounds_mean=("fu_n", "mean"), fu_mean_score=("fu_mean", "mean"),
                               any_fu_ge5=("fu_any_ge5", "mean")).round(3))
print("\nfollow-up PHQ-9 by adolescent:")
print(e.groupby("adol").agg(n=("fu_n", "size"), fu_mean_score=("fu_mean", "mean"), any_fu_ge5=("fu_any_ge5", "mean")).round(3))
print("\nwithin-woman: share of women with >=3 scored rounds whose scores cross 5 in both directions:")
m = e[e.fu_n >= 3]
print("  n=", len(m), " ever>=5:", round((m[[f'phq9_r{r}' for r in rounds]] >= 5).any(axis=1).mean(), 3),
      " always>=5:", round(((m[[f'phq9_r{r}' for r in rounds]] >= 5) | m[[f'phq9_r{r}' for r in rounds]].isna()).all(axis=1).mean(), 3))
