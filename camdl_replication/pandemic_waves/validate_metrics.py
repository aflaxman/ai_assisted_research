"""Validate the metric pipeline against the authors' own particles.

Runs the same simulate/metric code as analyze.py on the 100 lowest-error
particles from the paper's published posterior files ("Paper Particles/"
in the authors' repository), so the abstract's headline numbers can be
reproduced independently of any fitting done here. Any gap between these
numbers and the paper's is metric-implementation error; any gap between
camdl's numbers and these is inference difference.

Usage: uv run python validate_metrics.py /path/to/authors-repo-clone
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import aicc_terms, r0_stat, simulate
from prep_data import COUNTRIES

HERE = Path(__file__).parent


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/paper_code")
    rows = []
    for name, stem, pop, cutoff, _, start in COUNTRIES:
        cases = pd.read_csv(HERE / f"data/{name}_cases.tsv", sep="\t")["cases"].values / pop
        osi = pd.read_csv(HERE / f"data/{name}_osi.tsv", sep="\t")["osi"].values / 1000.0
        n_all = len(cases)
        pred = slice(cutoff, n_all)
        y, m, d = map(int, start.split("-"))
        dec31 = (date(2020, 12, 31) - date(y, m, d)).days + 1
        peak2_win = slice(cutoff, min(dec31, n_all))
        for model, fname, err_col in [
            ("sir", f"{stem}Avg_sir_latest.csv", "Error"),
            ("sirx", f"{stem}_final_par.csv", "Error_I"),
        ]:
            f = src / "Paper Particles" / fname
            if not f.exists():
                print(f"skip {name} {model}: {f.name} missing", file=sys.stderr)
                continue
            par = pd.read_csv(f, index_col=0).nsmallest(100, err_col)
            par = par.rename(columns={"eta_0": "eta0", "epsilon": "eps",
                                      "lambda": "lam", "I0": "i0"})
            reps, xs, r0s, reffs = [], [], [], []
            for _, r in par.iterrows():
                p = r.to_dict()
                rep, x = simulate(model, p, n_all)
                if rep is None:
                    continue
                reps.append(rep[:n_all])
                xs.append(x[:n_all])
                # Table S3's caption says "after 200 days", but its SIR
                # values only reproduce when R0(t) is pooled over the full
                # year (median seasonal factor ~ 1): full-axis pooling
                tt = np.arange(1.0, n_all + 1, 1.0)
                season = 1.0 + p["b"] * np.cos((tt - p["phi"]) * 2 * np.pi / 365)
                r0s.append(p["beta"] * season / p["gamma"])
                if model == "sirx":
                    reffs.append(r0s[-1] * (1.0 - p["eps"] * x[:n_all]))
            reps = np.array(reps)
            avg = reps.mean(axis=0)
            k_fit = 7 if model == "sir" else 12
            n_pred = n_all - cutoff
            rss_inc = float(np.mean((cases[pred] - avg[pred]) ** 2))
            row = {
                "country": name, "model": model, "n_ok": len(reps),
                "peak2_mag": float(avg[peak2_win].max()),
                "peak2_day": peak2_win.start + int(avg[peak2_win].argmax()) + 1,
                "data_peak2_mag": float(cases[peak2_win].max()),
                "data_peak2_day": cutoff + int(cases[peak2_win].argmax()) + 1,
                "area_pred": float(np.trapezoid(np.abs(cases[pred] - avg[pred]))),
                "aicc_inc": aicc_terms([(rss_inc, n_pred)], k_fit),
                "r0_med": r0_stat(np.array(r0s), cutoff)[0],
            }
            if model == "sirx":
                x_avg = np.array(xs).mean(axis=0)
                rss_osi = float(np.mean((osi[pred] - x_avg[pred]) ** 2))
                row["aicc_inc_osi"] = aicc_terms(
                    [(rss_inc, n_pred), (rss_osi, n_pred)], k_fit)
                row["reff_med"] = r0_stat(np.array(reffs), cutoff)[0]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "paper_particle_metrics.tsv", sep="\t", index=False)
    for model in ("sir", "sirx"):
        sub = df[df.model == model]
        print(f"\n== paper particles, {model} ({len(sub)} countries) ==")
        for c in ("aicc_inc", "area_pred", "peak2_mag", "peak2_day"):
            print(f"{c:12s} {sub[c].mean():10.4f} +/- {sub[c].std():.4f}")
        if model == "sirx" and "aicc_inc_osi" in sub:
            print(f"{'aicc_inc_osi':12s} {sub.aicc_inc_osi.mean():10.4f} +/- {sub.aicc_inc_osi.std():.4f}")


if __name__ == "__main__":
    main()
