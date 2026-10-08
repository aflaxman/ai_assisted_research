"""Load generations, judgments, and text metrics into one table, and summarize them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import text_metrics
from tasks import ORDER as TORDER
from tasks import TASKS
from variants import ORDER as VORDER
from variants import VARIANTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

JUDGE_COLS = ["clarity", "accuracy", "completeness", "actionability", "facts_recall"]
STYLE_COLS = [
    "words", "mean_sentence_words", "pct_sentences_over_20", "fk_grade",
    "semicolons", "contractions", "latin_abbreviations", "passive_per_100_sentences", "hedges",
]
COST_COLS = ["cost_usd", "output_tokens", "thinking_tokens", "wall_s"]

PRETTY = {
    "clarity": "Clarity (1-5)",
    "accuracy": "Accuracy (1-5)",
    "completeness": "Completeness (1-5)",
    "actionability": "Actionability (1-5)",
    "facts_recall": "Facts recalled (share)",
    "words": "Words",
    "mean_sentence_words": "Words per sentence",
    "pct_sentences_over_20": "Sentences over 20 words (%)",
    "fk_grade": "Flesch-Kincaid grade",
    "semicolons": "Semicolons",
    "contractions": "Contractions",
    "latin_abbreviations": "Latin abbreviations",
    "passive_per_100_sentences": "Passive hits per 100 sentences",
    "hedges": "Hedge words",
    "cost_usd": "Cost per run (USD)",
    "output_tokens": "Output tokens",
    "thinking_tokens": "Thinking tokens",
    "wall_s": "Wall time (s)",
    "traps_hit_n": "Traps stated as fact",
    "unsupported_claims": "Unsupported claims",
}


def group_of(variant: str) -> str:
    src = VARIANTS[variant]["source"]
    if src.startswith("PR"):
        return "PR #304"
    if src.startswith("Karpathy"):
        return "Karpathy"
    return "control"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_all() -> pd.DataFrame:
    gens = pd.DataFrame(read_jsonl(RESULTS / "generations.jsonl"))
    if gens.empty:
        raise FileNotFoundError("no generations yet; run run_variants.py")
    m = pd.DataFrame([text_metrics(o) for o in gens["output"]])
    df = pd.concat([gens.reset_index(drop=True), m], axis=1)
    judg = read_jsonl(RESULTS / "judgments.jsonl")
    if judg:
        j = pd.DataFrame(judg).drop(columns=["n_facts"], errors="ignore")
        df = df.merge(j, on=["variant", "task", "rep"], how="left")
        df["traps_hit_n"] = df["traps_hit"].map(lambda x: len(x) if isinstance(x, list) else np.nan)
    df["variant_label"] = df["variant"].map(lambda v: VARIANTS[v]["label"])
    df["group"] = df["variant"].map(group_of)
    df["task_title"] = df["task"].map(lambda t: TASKS[t]["title"])
    df["variant"] = pd.Categorical(df["variant"], [v for v in VORDER if v in set(df["variant"])])
    df["task"] = pd.Categorical(df["task"], [t for t in TORDER if t in set(df["task"])])
    return df.sort_values(["variant", "task", "rep"]).reset_index(drop=True)


def bootstrap_ci(x, n_boot: int = 4000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (x[0], x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(df: pd.DataFrame, cols: list[str], by: str = "variant") -> pd.DataFrame:
    """Long table: one row per (by, metric) with mean, 95% bootstrap CI, and n."""
    rows = []
    for key, sub in df.groupby(by, observed=True):
        for c in cols:
            if c not in sub:
                continue
            x = sub[c].dropna()
            lo, hi = bootstrap_ci(x)
            rows.append({by: key, "metric": c, "mean": x.mean(), "lo": lo, "hi": hi, "n": len(x)})
    out = pd.DataFrame(rows)
    if by == "variant":
        out["label"] = out["variant"].map(lambda v: VARIANTS[v]["label"])
        out["group"] = out["variant"].map(group_of)
    return out


def wide_table(df: pd.DataFrame, cols: list[str], digits: int = 2, by: str = "variant") -> pd.DataFrame:
    """Readable table: rows per variant, 'mean [lo, hi]' per metric."""
    s = summarize(df, cols, by=by)
    fmt = lambda r: f"{r['mean']:.{digits}f} [{r['lo']:.{digits}f}, {r['hi']:.{digits}f}]"  # noqa: E731
    s["cell"] = s.apply(fmt, axis=1)
    idx = "label" if by == "variant" else by
    w = s.pivot(index=idx, columns="metric", values="cell")
    w = w[[c for c in cols if c in w.columns]]
    w.columns = [PRETTY.get(c, c) for c in w.columns]
    if by == "variant":
        order = [VARIANTS[v]["label"] for v in VORDER if VARIANTS[v]["label"] in w.index]
        w = w.loc[order]
    w.index.name = None
    return w


def delta_vs(df: pd.DataFrame, cols: list[str], ref: str = "none") -> pd.DataFrame:
    """Mean difference of each variant from the reference variant, with a paired-by-(task, rep) bootstrap CI."""
    rows = []
    base = df[df["variant"] == ref].set_index(["task", "rep"])
    for v, sub in df.groupby("variant", observed=True):
        if v == ref:
            continue
        sub = sub.set_index(["task", "rep"])
        common = sub.index.intersection(base.index)
        for c in cols:
            if c not in sub:
                continue
            d = (sub.loc[common, c].astype(float) - base.loc[common, c].astype(float)).dropna()
            lo, hi = bootstrap_ci(d)
            rows.append({"variant": v, "label": VARIANTS[v]["label"], "metric": c, "delta": d.mean(), "lo": lo, "hi": hi, "n": len(d)})
    return pd.DataFrame(rows)
