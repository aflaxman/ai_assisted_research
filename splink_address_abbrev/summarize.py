"""Aggregate experiment outputs into the tables quoted in README.md."""

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
RESULTS = Path(__file__).parent / "results"


def load_all(stem):
    frames = []
    for suffix in ["", "_address"]:
        path = RESULTS / f"{stem}{suffix}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "model" not in df.columns:
                df.insert(0, "model", "full")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all("linkage_results")
    g = (
        df.groupby(["model", "noise", "regime", "treatment"])
        .agg(
            best_f1_mean=("best_f1", "mean"),
            best_f1_sd=("best_f1", "std"),
            ap_mean=("avg_precision", "mean"),
            precision_mean=("best_f1_precision", "mean"),
            recall_mean=("best_f1_recall", "mean"),
        )
        .reindex(["none", "abbreviate", "expand"], level="treatment")
        .round(4)
    )
    print("=== linkage quality (mean of 3 replicates) ===")
    print(g.to_string())

    print("\n=== paired per-replicate comparisons ===")
    for metric in ["best_f1", "avg_precision"]:
        p = df.pivot_table(
            index=["model", "noise", "regime", "rep"], columns="treatment", values=metric
        )
        ae, na = p.abbreviate - p.expand, p.none - p.abbreviate
        print(
            f"{metric}: abbreviate > expand in {(ae > 0).sum()}/{len(p)} "
            f"(mean diff {ae.mean():+.5f}); none > abbreviate in "
            f"{(na > 0).sum()}/{len(p)} (mean diff {na.mean():+.5f})"
        )

    params = load_all("street_params")
    params["match_weight"] = np.log2(params["m"] / params["u"])
    pw = (
        params[params.model == "full"]
        .groupby(["noise", "regime", "treatment", "level"])["match_weight"]
        .mean()
        .unstack("level")
        .round(2)
    )
    print("\n=== trained street_name match weights, log2(m/u), full model ===")
    print(pw.to_string())

    phase2 = RESULTS / "linkage_results_phase2.csv"
    if phase2.exists():
        p2 = pd.read_csv(phase2)
        g2 = (
            p2.groupby(["comparator", "noise", "model", "treatment"])
            .agg(best_f1_mean=("best_f1", "mean"), ap_mean=("avg_precision", "mean"))
            .reindex(["none", "abbreviate", "expand"], level="treatment")
            .round(4)
        )
        print("\n=== phase 2 (split conventions): comparator and noise sensitivity ===")
        print(g2.to_string())
        for metric in ["best_f1", "avg_precision"]:
            p = p2.pivot_table(
                index=["comparator", "noise", "model", "rep"],
                columns="treatment",
                values=metric,
            )
            ae, na = p.abbreviate - p.expand, p.none - p.abbreviate
            print(
                f"{metric}: abbreviate > expand in {(ae > 0).sum()}/{len(p)} "
                f"(mean {ae.mean():+.5f}); none > abbreviate in "
                f"{(na > 0).sum()}/{len(p)} (mean {na.mean():+.5f})"
            )


if __name__ == "__main__":
    main()
