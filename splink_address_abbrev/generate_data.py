"""Generate pairs of noised pseudopeople census extracts for linkage.

Each replicate draws the same simulated 2020 census twice with different
noise seeds, so every simulant appears in both extracts with independently
corrupted fields. Ground truth for linkage is simulant_id.

Convention regimes applied to street_name after noise:
  consistent — leave the generated mixture of forms alone (the same
               household keeps the same form in both extracts, up to noise)
  split      — extract A's pipeline abbreviates suffixes, extract B's
               spells them out (the scenario in splink discussion #3250)

Cached to parquet under data/ so replicates are reproducible and fast.
"""

from pathlib import Path

import pandas as pd
import pseudopeople as psp

from suffix_maps import abbreviate, expand

DATA_DIR = Path(__file__).parent / "data"

COLS = [
    "simulant_id",
    "first_name",
    "last_name",
    "date_of_birth",
    "street_number",
    "street_name",
    "city",
    "zipcode",
]

# elevated noise: messy admin-data street names (~15% of cells corrupted)
ELEVATED_STREET_NOISE = {
    "decennial_census": {
        "column_noise": {
            "street_name": {
                "make_typos": {"cell_probability": 0.05, "token_probability": 0.1},
                "make_ocr_errors": {"cell_probability": 0.05, "token_probability": 0.1},
                "make_phonetic_errors": {"cell_probability": 0.05, "token_probability": 0.1},
            }
        }
    }
}

NOISE_CONFIGS = {"default": None, "elevated": ELEVATED_STREET_NOISE}


def _one_extract(seed, noise_level):
    config = NOISE_CONFIGS[noise_level]
    df = psp.generate_decennial_census(year=2020, seed=seed, config=config)
    df = df[COLS].copy()
    # a handful of simulants are duplicated (guardian duplication noise);
    # keep one row each so ground truth is one-to-one
    df = df.drop_duplicates(subset="simulant_id", keep="first")
    df["date_of_birth"] = df["date_of_birth"].astype(str)
    return df.reset_index(drop=True)


def load_pair(replicate, noise_level):
    """Return (df_a, df_b) for one replicate, generating and caching if needed."""
    DATA_DIR.mkdir(exist_ok=True)
    dfs = []
    for side, seed in [("a", 2 * replicate), ("b", 2 * replicate + 1)]:
        path = DATA_DIR / f"census_{noise_level}_rep{replicate}_{side}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = _one_extract(seed, noise_level)
            df.to_parquet(path)
        dfs.append(df)
    return dfs


def apply_regime(df_a, df_b, regime):
    """Apply the convention regime, returning copies."""
    df_a, df_b = df_a.copy(), df_b.copy()
    if regime == "split":
        df_a["street_name"] = df_a["street_name"].map(abbreviate, na_action="ignore")
        df_b["street_name"] = df_b["street_name"].map(expand, na_action="ignore")
    elif regime != "consistent":
        raise ValueError(regime)
    return df_a, df_b


if __name__ == "__main__":
    for noise_level in NOISE_CONFIGS:
        for rep in range(3):
            a, b = load_pair(rep, noise_level)
            print(f"{noise_level} rep{rep}: {len(a)} x {len(b)} records cached")
