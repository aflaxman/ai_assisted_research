"""How well does Jaro-Winkler on street_name separate matches from
non-matches under each standardization treatment?

For each (noise level, regime, treatment) cell, pool replicates and compute
JW similarity for true pairs (same simulant, both street names present) and
for random non-match pairs. Report ROC AUC of the JW score as a
one-number summary of separability, plus similarity distributions.
"""

import itertools

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from generate_data import NOISE_CONFIGS, apply_regime, load_pair
from suffix_maps import TREATMENTS

RNG = np.random.default_rng(42)
N_REPS = 3
N_NONMATCH = 50_000


def pairs_for_cell(noise_level, regime):
    """Return (true_a, true_b, non_a, non_b) street-name string arrays."""
    trues_a, trues_b = [], []
    for rep in range(N_REPS):
        df_a, df_b = apply_regime(*load_pair(rep, noise_level), regime)
        m = df_a.merge(df_b, on="simulant_id", suffixes=("_a", "_b"))
        m = m.dropna(subset=["street_name_a", "street_name_b"])
        trues_a.append(m["street_name_a"].to_numpy())
        trues_b.append(m["street_name_b"].to_numpy())
    true_a = np.concatenate(trues_a)
    true_b = np.concatenate(trues_b)

    # non-matches: random cross pairs from replicate 0, different simulants
    df_a, df_b = apply_regime(*load_pair(0, noise_level), regime)
    df_a = df_a.dropna(subset=["street_name"])
    df_b = df_b.dropna(subset=["street_name"])
    ia = RNG.integers(0, len(df_a), N_NONMATCH)
    ib = RNG.integers(0, len(df_b), N_NONMATCH)
    keep = df_a["simulant_id"].to_numpy()[ia] != df_b["simulant_id"].to_numpy()[ib]
    non_a = df_a["street_name"].to_numpy()[ia][keep]
    non_b = df_b["street_name"].to_numpy()[ib][keep]
    return true_a, true_b, non_a, non_b


def jw(a_arr, b_arr, treat_fn):
    return np.array(
        [JaroWinkler.similarity(treat_fn(a), treat_fn(b)) for a, b in zip(a_arr, b_arr)]
    )


def auc(pos, neg):
    """ROC AUC via Mann-Whitney U."""
    scores = np.concatenate([pos, neg])
    ranks = pd.Series(scores).rank().to_numpy()
    n_pos, n_neg = len(pos), len(neg)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    rows = []
    sims = {}
    for noise_level, regime in itertools.product(
        NOISE_CONFIGS, ["consistent", "split", "mixed"]
    ):
        true_a, true_b, non_a, non_b = pairs_for_cell(noise_level, regime)
        for treatment, fn in TREATMENTS.items():
            pos = jw(true_a, true_b, fn)
            neg = jw(non_a, non_b, fn)
            sims[(noise_level, regime, treatment)] = (pos, neg)
            rows.append(
                {
                    "noise": noise_level,
                    "regime": regime,
                    "treatment": treatment,
                    "n_true": len(pos),
                    "auc": auc(pos, neg),
                    "true_jw_mean": pos.mean(),
                    "true_pct_exact": (pos == 1.0).mean(),
                    "true_pct_ge_.92": (pos >= 0.92).mean(),
                    "nonmatch_jw_mean": neg.mean(),
                    "nonmatch_pct_ge_.92": (neg >= 0.92).mean(),
                    "nonmatch_pct_ge_.7": (neg >= 0.7).mean(),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv("results/microbench.csv", index=False)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    np.savez_compressed(
        "results/microbench_sims.npz",
        **{"|".join(k) + "|pos": v[0] for k, v in sims.items()},
        **{"|".join(k) + "|neg": v[1] for k, v in sims.items()},
    )


if __name__ == "__main__":
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)
    main()
