"""Phase 2: kick the tires on the phase-1 findings.

Grid 1 (comparator sensitivity, elevated noise, split conventions):
  - jw_no_tf: is "no cleaning wins best F1" an artifact of splink's TF
    adjustment applying only to the exact-match level?
  - lev_abs: does the result survive an absolute-edit-distance comparator,
    where "st" vs "street" is 4 edits and cannot ride the fuzzy levels?

Grid 2 (noise sensitivity, split conventions, baseline jw_tf comparator):
  - severe (~39% of street names corrupted) and garbled (~14% of cells hit
    but half the tokens mangled) noise levels.

Both grids run both linkage models and all three treatments, 3 replicates.
Writes results/linkage_results_phase2.csv and results/street_params_phase2.csv.
"""

import itertools

import pandas as pd

from generate_data import apply_regime, load_pair
from run_experiment import RESULTS, pr_metrics, run_one
from suffix_maps import TREATMENTS

GRID = [
    # (comparator, noise) pairs; regime is always split — the discussion's scenario
    ("jw_no_tf", "elevated"),
    ("lev_abs", "elevated"),
    ("jw_tf", "severe"),
    ("jw_tf", "garbled"),
]


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, param_rows = [], []
    cells = [
        (comparator, noise, model, treatment, rep)
        for (comparator, noise), model, treatment, rep in itertools.product(
            GRID, ["full", "address_heavy"], TREATMENTS, range(3)
        )
    ]
    for i, (comparator, noise, model, treatment, rep) in enumerate(cells):
        df_a, df_b = apply_regime(*load_pair(rep, noise), "split")
        scored, n_true, params = run_one(
            df_a, df_b, TREATMENTS[treatment], model, comparator
        )
        curve, best, ap = pr_metrics(scored, n_true)
        key = dict(
            comparator=comparator,
            model=model,
            noise=noise,
            regime="split",
            treatment=treatment,
            rep=rep,
        )
        rows.append(
            {
                **key,
                "n_true_pairs": n_true,
                "n_scored": len(scored),
                "avg_precision": ap,
                "best_f1": best["f1"],
                "best_f1_precision": best["precision"],
                "best_f1_recall": best["recall"],
            }
        )
        for p in params:
            param_rows.append({**key, **p})
        print(
            f"[{i + 1}/{len(cells)}] {comparator}/{noise}/{model}/{treatment}/rep{rep}: "
            f"best F1={best['f1']:.4f} AP={ap:.4f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(RESULTS / "linkage_results_phase2.csv", index=False)
    pd.DataFrame(param_rows).to_csv(RESULTS / "street_params_phase2.csv", index=False)


if __name__ == "__main__":
    main()
