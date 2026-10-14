"""Aggregate the replication runs (results/study*_reps/seed*/summary.json) into tables.

Writes results/replications.json and prints Markdown tables for the README.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def load_all(pattern):
    runs = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            runs.append(json.load(fh))
    return runs


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def study1(runs):
    models = list(runs[0]["models"])
    out = {}
    for m in models:
        rmse = [r["models"][m]["theta"]["rmse"] for r in runs]
        terc = np.array([r["models"][m]["theta"]["rmse_by_tercile"] for r in runs])
        row = dict(n=len(runs), rmse=mean_se(rmse),
                   rmse_by_tercile=[mean_se(terc[:, k]) for k in range(3)])
        if "ktau" in runs[0]["models"][m]:
            kt = [r["models"][m]["ktau"]["mean"] for r in runs]
            cover = [r["models"][m]["ktau"]["q05"] <= r["truth"]["ktau"] <= r["models"][m]["ktau"]["q95"] for r in runs]
            row["ktau"] = mean_se(kt)
            row["ktau_90pct_coverage"] = float(np.mean(cover))
            pv = [r["models"][m]["ppc_tail_concentration"]["p_value_upper10_minus_lower10"] for r in runs]
            row["ppc_pvalue_mean"] = float(np.mean(pv))
            row["ppc_pvalue_extreme_share"] = float(np.mean([(p < 0.05) or (p > 0.95) for p in pv]))
        out[m] = row
    comps = {}
    for key in runs[0]["waic_comparisons"]:
        d = [r["waic_comparisons"][key]["elpd_diff"] for r in runs]
        comps[key] = dict(mean=mean_se(d), share_positive=float(np.mean(np.array(d) > 0)))
    return dict(models=out, waic_comparisons=comps)


def study2(runs):
    fams = list(runs[0]["models"])
    out = {}
    for f in fams:
        row = dict(n=len(runs))
        row["ktau"] = mean_se([r["models"][f]["ktau"]["mean"] for r in runs])
        for key in ("auc", "log_loss", "brier"):
            row[key] = mean_se([r["models"][f]["test"][key] for r in runs])
        for top in ("top1pct", "top5pct", "top20pct"):
            row[f"{top}_observed"] = mean_se([r["models"][f]["test"][top]["observed_rate"] for r in runs])
            row[f"{top}_predicted"] = mean_se([r["models"][f]["test"][top]["predicted_rate"] for r in runs])
        row["joint_tail_q99"] = mean_se([r["models"][f]["joint_tail_p_eta_gt_q99_and_Y"] for r in runs])
        out[f] = row
    oracle = dict(log_loss=mean_se([r["truth"]["oracle_from_indicators"]["log_loss"] for r in runs]),
                  auc=mean_se([r["truth"]["oracle_from_indicators"]["auc"] for r in runs]),
                  top1pct_observed=mean_se([r["truth"]["oracle_from_indicators"]["top1pct"]["observed_rate"] for r in runs]),
                  top1pct_predicted=mean_se([r["truth"]["oracle_from_indicators"]["top1pct"]["predicted_rate"] for r in runs]),
                  joint_tail_q99=runs[0]["truth"]["joint_tail_p_eta_gt_q99_and_Y"])
    comps = {}
    for key in runs[0]["waic_comparisons"]:
        d = [r["waic_comparisons"][key]["elpd_diff"] for r in runs]
        comps[key] = dict(mean=mean_se(d), share_positive=float(np.mean(np.array(d) > 0)))
    return dict(models=out, oracle=oracle, waic_comparisons=comps)


def fmt(ms, digits=3):
    m, se = ms
    return f"{m:.{digits}f} ± {se:.{digits}f}"


def main():
    res = {}
    s1 = load_all("results/study1_reps/seed*/summary.json")
    s2 = load_all("results/study2_reps/seed*/summary.json")
    if s1:
        res["study1"] = study1(s1)
        a = res["study1"]
        print(f"\n### Study 1, {len(s1)} replications (mean ± SE across data sets)\n")
        print("| model | RMSE of ability | lowest third | middle third | highest third | Kendall's tau (true 0.5) | 90% interval covers truth |")
        print("|---|---|---|---|---|---|---|")
        for m, row in a["models"].items():
            t = row["rmse_by_tercile"]
            kt = fmt(row["ktau"]) if "ktau" in row else "-"
            cov = f"{row['ktau_90pct_coverage']:.0%}" if "ktau" in row else "-"
            print(f"| {m} | {fmt(row['rmse'])} | {fmt(t[0])} | {fmt(t[1])} | {fmt(t[2])} | {kt} | {cov} |")
        print("\n| WAIC comparison (elpd difference) | mean ± SE | share of data sets with first model better |")
        print("|---|---|---|")
        for k, c in a["waic_comparisons"].items():
            print(f"| {k.replace('_minus_', ' minus ')} | {fmt(c['mean'], 1)} | {c['share_positive']:.0%} |")
    if s2:
        res["study2"] = study2(s2)
        a = res["study2"]
        print(f"\n### Study 2, {len(s2)} replications (mean ± SE across data sets)\n")
        print("| model | Kendall's tau (true 0.5) | AUC | log-loss | top 1% predicted | top 1% observed | top 20% predicted | top 20% observed |")
        print("|---|---|---|---|---|---|---|---|")
        for f, row in a["models"].items():
            print(f"| {f} | {fmt(row['ktau'])} | {fmt(row['auc'])} | {fmt(row['log_loss'], 4)} | {fmt(row['top1pct_predicted'])} | {fmt(row['top1pct_observed'])} | {fmt(row['top20pct_predicted'])} | {fmt(row['top20pct_observed'])} |")
        o = a["oracle"]
        print(f"| oracle (true model, true parameters) | 0.5 | {fmt(o['auc'])} | {fmt(o['log_loss'], 4)} | {fmt(o['top1pct_predicted'])} | {fmt(o['top1pct_observed'])} | | |")
        print("\n| WAIC comparison (elpd difference) | mean ± SE | share of data sets with first model better |")
        print("|---|---|---|")
        for k, c in a["waic_comparisons"].items():
            print(f"| {k.replace('_minus_', ' minus ')} | {fmt(c['mean'], 1)} | {c['share_positive']:.0%} |")
    os.makedirs("results", exist_ok=True)
    with open("results/replications.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
