"""Does the abbreviate-vs-expand result depend on the string comparator?

Same design as microbench.py, but crossed with four similarity functions:
Jaro-Winkler (splink's usual choice for names/addresses), plain Jaro
(no prefix bonus), and normalized Levenshtein / Damerau-Levenshtein.
Reports ROC AUC of each score separating true from non-matching pairs.
"""

import itertools

import numpy as np
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein, Jaro, JaroWinkler, Levenshtein

from generate_data import NOISE_CONFIGS
from microbench import auc, pairs_for_cell
from suffix_maps import TREATMENTS

METRICS = {
    "jaro_winkler": JaroWinkler.similarity,
    "jaro": Jaro.similarity,
    "levenshtein_norm": Levenshtein.normalized_similarity,
    "damerau_lev_norm": DamerauLevenshtein.normalized_similarity,
}

REGIMES = ["consistent", "split", "mixed", "dropsuffix"]


def main():
    rows = []
    for noise_level, regime in itertools.product(NOISE_CONFIGS, REGIMES):
        true_a, true_b, non_a, non_b = pairs_for_cell(noise_level, regime)
        for treatment, fn in TREATMENTS.items():
            ta = [fn(s) for s in true_a]
            tb = [fn(s) for s in true_b]
            na = [fn(s) for s in non_a]
            nb = [fn(s) for s in non_b]
            for metric, sim in METRICS.items():
                pos = np.array([sim(a, b) for a, b in zip(ta, tb)])
                neg = np.array([sim(a, b) for a, b in zip(na, nb)])
                rows.append(
                    {
                        "metric": metric,
                        "noise": noise_level,
                        "regime": regime,
                        "treatment": treatment,
                        "auc": auc(pos, neg),
                        "true_mean": pos.mean(),
                        "nonmatch_mean": neg.mean(),
                    }
                )
        print(f"{noise_level}/{regime} done", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv("results/microbench_metrics.csv", index=False)
    piv = results.pivot_table(
        index=["noise", "regime", "treatment"], columns="metric", values="auc"
    ).round(5)
    print(piv.to_string())


if __name__ == "__main__":
    main()
