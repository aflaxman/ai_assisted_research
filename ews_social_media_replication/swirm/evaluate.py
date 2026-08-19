"""Tie the pieces together: simulate an ensemble, locate the changepoint,
measure the delay past the bifurcation, and score early-warning signals.

The heavy simulation happens once per scenario; everything downstream is scalar
per realization, so results fit comfortably in a tidy table that the notebook
caches and reloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Params, STATE_NAMES
from .simulate import simulate_ensemble
from .changepoint import amoc_meanshift
from .ews import ews_trends

VAR_INDEX = {"I": 2, "W": 1, "M": 4}


@dataclass
class Scenario:
    noise_type: str
    sigma: float = 0.0
    label: str = ""

    def name(self) -> str:
        return self.label or f"{self.noise_type}:{self.sigma:g}"


def run_scenario(
    p: Params,
    scenario: Scenario,
    variables=("I", "W", "M"),
    n_real: int = 300,
    t_max: float = 150.0,
    dt: float = 0.005,
    seed: int = 0,
) -> pd.DataFrame:
    """Run one noise scenario and return a per-(realization, variable) table.

    Columns: changepoint time, delay past the bifurcation, a delay/advance flag,
    and the Kendall-tau of variance and autocorrelation on the two comparison
    segments (leading up to the bifurcation point, and up to the changepoint).
    """
    times, traj = simulate_ensemble(
        p, scenario.noise_type, scenario.sigma, n_real=n_real,
        t_max=t_max, dt=dt, seed=seed,
    )
    bp_time = p.bifurcation_time()
    bp_idx = int(round(bp_time / dt))  # segment length = points up to BP

    rows = []
    for var in variables:
        series_all = traj[:, :, VAR_INDEX[var]]
        for j in range(series_all.shape[0]):
            series = series_all[j]
            cp = amoc_meanshift(series)
            cp_time = np.nan if cp is None else times[cp]
            is_delay = cp is not None and cp > bp_idx

            row = {
                "scenario": scenario.name(),
                "noise_type": scenario.noise_type,
                "sigma": scenario.sigma,
                "variable": var,
                "realization": j,
                "cp_time": cp_time,
                "delay": np.nan if cp is None else cp_time - bp_time,
                "is_delay": is_delay,
            }
            # EWS are only meaningful for the delayed series the paper compares.
            if is_delay and cp - bp_idx >= 0:
                bp_seg = series[:bp_idx]
                cp_seg = series[cp - bp_idx:cp]
                bp = ews_trends(bp_seg)
                cp_t = ews_trends(cp_seg)
                row.update(
                    tau_var_bp=bp["tau_variance"], tau_ac_bp=bp["tau_autocorr"],
                    tau_var_cp=cp_t["tau_variance"], tau_ac_cp=cp_t["tau_autocorr"],
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUC via the Mann-Whitney U interpretation (rank of positives)."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if scores_pos.size == 0 or scores_neg.size == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([scores_pos, scores_neg])).rank().to_numpy()
    r_pos = ranks[: scores_pos.size].sum()
    return (r_pos - scores_pos.size * (scores_pos.size + 1) / 2) / (
        scores_pos.size * scores_neg.size
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(scenario, variable) EWS performance summary.

    Reproduces the two headline tables: the percent of positive Kendall taus at
    the bifurcation vs changepoint (Table 2), and the AUC of variance and
    autocorrelation for discriminating changepoint segments from bifurcation
    segments (Figs 7-10).
    """
    out = []
    keys = ["scenario", "noise_type", "sigma", "variable"]
    for key, g in df.groupby(keys, sort=False):
        d = g.dropna(subset=["tau_var_cp"])
        if d.empty:
            continue
        rec = dict(zip(keys, key))
        rec["n_delay"] = len(d)
        # Table 2: percent positive taus.
        rec["pct_pos_var_bp"] = 100 * (d["tau_var_bp"] > 0).mean()
        rec["pct_pos_var_cp"] = 100 * (d["tau_var_cp"] > 0).mean()
        rec["pct_pos_ac_bp"] = 100 * (d["tau_ac_bp"] > 0).mean()
        rec["pct_pos_ac_cp"] = 100 * (d["tau_ac_cp"] > 0).mean()
        # AUC: CP segments are positives, BP segments are negatives.
        rec["auc_variance"] = _auc(d["tau_var_cp"], d["tau_var_bp"])
        rec["auc_autocorr"] = _auc(d["tau_ac_cp"], d["tau_ac_bp"])
        out.append(rec)
    return pd.DataFrame(out)


def roc_curve(scores_pos: np.ndarray, scores_neg: np.ndarray):
    """False-positive and true-positive rates for an ROC plot."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    thresholds = np.sort(np.unique(np.concatenate([scores_pos, scores_neg])))[::-1]
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])
    tpr = [np.mean(scores_pos >= t) for t in thresholds]
    fpr = [np.mean(scores_neg >= t) for t in thresholds]
    return np.array(fpr), np.array(tpr)
