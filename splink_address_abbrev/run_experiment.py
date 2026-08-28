"""Full splink linkage experiment.

Factorial design: 2 noise levels x 2 convention regimes x 3 street_name
treatments x 3 replicates. The splink model specification, blocking rules,
and training procedure are identical in every cell — only the content of
the street_name column differs. Blocking never uses street_name, so every
cell scores the same candidate pairs.

Outputs:
  results/linkage_results.csv   one row per (cell, replicate) with PR metrics
  results/street_params.csv     trained m/u/match-weights for street_name levels
  results/pr_curves.parquet     precision-recall curves for plotting
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from generate_data import NOISE_CONFIGS, apply_regime, load_pair
from suffix_maps import TREATMENTS

RESULTS = Path(__file__).parent / "results"

BLOCKING_RULES = [
    block_on("last_name"),
    block_on("first_name", "date_of_birth"),
    block_on("street_number", "first_name"),
]


def make_settings(model="full"):
    """'full' uses all five fields; 'address_heavy' drops last_name and
    date_of_birth, mimicking datasets where addresses must carry the
    discriminating weight."""
    comparisons = [
        cl.NameComparison("first_name"),
        cl.LevenshteinAtThresholds("street_number", 1),
        cl.JaroWinklerAtThresholds("street_name", [0.92, 0.7]).configure(
            term_frequency_adjustments=True
        ),
    ]
    if model == "full":
        comparisons[1:1] = [
            cl.NameComparison("last_name"),
            cl.DamerauLevenshteinAtThresholds("date_of_birth", [1, 2]),
        ]
    return SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=BLOCKING_RULES,
        comparisons=comparisons,
        retain_intermediate_calculation_columns=False,
    )


def prep(df, treat_fn):
    df = df.copy()
    df["street_name"] = df["street_name"].map(treat_fn, na_action="ignore")
    df.loc[df["date_of_birth"] == "nan", "date_of_birth"] = None
    df["unique_id"] = np.arange(len(df))
    return df


def run_one(df_a, df_b, treat_fn, model="full"):
    df_a, df_b = prep(df_a, treat_fn), prep(df_b, treat_fn)
    linker = Linker(
        [df_a, df_b],
        make_settings(model),
        db_api=DuckDBAPI(),
        input_table_aliases=["a", "b"],
    )
    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "last_name", "date_of_birth")], recall=0.8
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name", "last_name")
    )
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("date_of_birth")
    )
    preds = linker.inference.predict(threshold_match_probability=0.001)
    scored = preds.as_pandas_dataframe()[
        ["unique_id_l", "unique_id_r", "match_probability", "match_weight"]
    ]

    # ground truth
    id_a = df_a.set_index("unique_id")["simulant_id"]
    id_b = df_b.set_index("unique_id")["simulant_id"]
    scored["is_match"] = (
        scored["unique_id_l"].map(id_a).to_numpy()
        == scored["unique_id_r"].map(id_b).to_numpy()
    )
    n_true = len(set(id_a) & set(id_b))

    # trained street_name parameters
    settings = linker.misc.save_model_to_json()
    street = [
        c for c in settings["comparisons"] if c["output_column_name"] == "street_name"
    ][0]
    params = [
        {
            "level": lv["label_for_charts"],
            "m": lv.get("m_probability"),
            "u": lv.get("u_probability"),
        }
        for lv in street["comparison_levels"]
        if not lv.get("is_null_level")
    ]
    return scored, n_true, params


def pr_metrics(scored, n_true):
    """Precision/recall/F1 swept over match_probability thresholds."""
    s = scored.sort_values("match_probability", ascending=False)
    tp = s["is_match"].cumsum().to_numpy()
    n_pred = np.arange(1, len(s) + 1)
    precision = tp / n_pred
    recall = tp / n_true
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    probs = s["match_probability"].to_numpy()
    curve = pd.DataFrame(
        {"threshold": probs, "precision": precision, "recall": recall, "f1": f1}
    )
    # average precision: unscored true pairs contribute zero
    ap = (precision * s["is_match"].to_numpy()).sum() / n_true
    best = curve.iloc[f1.argmax()]
    return curve, best, ap


def main(model="full", suffix=""):
    RESULTS.mkdir(exist_ok=True)
    rows, param_rows, curves = [], [], []
    cells = list(
        itertools.product(NOISE_CONFIGS, ["consistent", "split"], TREATMENTS, range(3))
    )
    for i, (noise, regime, treatment, rep) in enumerate(cells):
        df_a, df_b = apply_regime(*load_pair(rep, noise), regime)
        scored, n_true, params = run_one(df_a, df_b, TREATMENTS[treatment], model)
        curve, best, ap = pr_metrics(scored, n_true)
        key = dict(model=model, noise=noise, regime=regime, treatment=treatment, rep=rep)
        rows.append(
            {
                **key,
                "n_true_pairs": n_true,
                "n_scored": len(scored),
                "avg_precision": ap,
                "best_f1": best["f1"],
                "best_f1_precision": best["precision"],
                "best_f1_recall": best["recall"],
                "best_f1_threshold": best["threshold"],
            }
        )
        for p in params:
            param_rows.append({**key, **p})
        # decimate curve for storage
        step = max(len(curve) // 2000, 1)
        curve = curve.iloc[::step]
        for k, v in key.items():
            curve[k] = v
        curves.append(curve)
        print(
            f"[{i + 1}/{len(cells)}] {noise}/{regime}/{treatment}/rep{rep}: "
            f"best F1={best['f1']:.4f} (P={best['precision']:.4f}, R={best['recall']:.4f})",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(RESULTS / f"linkage_results{suffix}.csv", index=False)
    pd.DataFrame(param_rows).to_csv(RESULTS / f"street_params{suffix}.csv", index=False)
    pd.concat(curves).to_parquet(RESULTS / f"pr_curves{suffix}.parquet")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "address_heavy":
        main("address_heavy", "_address")
    else:
        main()
