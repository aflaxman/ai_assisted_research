"""Post-fit analysis: trajectories, peaks, AICc, area, R0/Reff tables.

Reads the pooled posterior draws camdl's MH stage wrote for each
(country, model) fit, re-integrates the deterministic ODE with scipy
(validated against camdl's ODE backend to <0.2% on the reported-incidence
series), and computes the paper's comparison metrics:

- posterior trajectory bands for reported incidence and the mitigator
  proportion x (results/trajectories/*.tsv)
- first/second-wave peak magnitude and day, vs the data (Fig 3 / S3)
- area between the average simulation and the data over the prediction
  window (Fig 5b)
- AICc on the prediction window, SIR vs SIRx-incidence vs
  SIRx-incidence+stringency (Fig 5a; supplement eq. 18)
- R0 and Reff medians with 95% CrI over the prediction window (Table S3)

The paper's metrics use "the average simulation from 100 selected
particles"; here the average simulation is the mean over N_DRAWS
posterior draws, and credible intervals are quantiles over draws.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from prep_data import COUNTRIES

HERE = Path(__file__).parent
RESULTS = HERE / "results"
N_DRAWS = 200
RNG = np.random.default_rng(20260819)

# Table S3 of the paper, for side-by-side comparison:
# country -> (SIRX R0, SIR R0, Reff), each (median, lo, hi)
PAPER_TABLE_S3 = {
    "austria":     ((1.333, 0.786, 1.566), (2.905, 0.243, 3.598), (1.321, 0.678, 1.545)),
    "belarus":     ((1.409, 0.835, 1.796), (1.631, 1.160, 1.760), (1.405, 0.835, 1.788)),
    "belgium":     ((1.430, 0.938, 1.786), (2.159, 0.837, 2.699), (1.313, 0.541, 1.626)),
    "denmark":     ((1.638, 0.958, 2.361), (1.536, 1.173, 1.664), (1.553, 0.614, 2.159)),
    "finland":     ((1.492, 0.493, 1.819), (1.611, 1.057, 1.766), (1.461, 0.493, 1.783)),
    "france":      ((2.020, 0.702, 2.996), (1.515, 1.157, 1.600), (1.908, 0.454, 2.558)),
    "germany":     ((2.331, 0.145, 2.964), (2.264, 0.684, 2.562), (2.306, 0.067, 2.931)),
    "ireland":     ((2.027, 0.141, 2.476), (3.584, 0.024, 4.348), (1.991, 0.131, 2.424)),
    "netherlands": ((1.823, 1.127, 2.636), (2.424, 0.644, 2.931), (1.512, 0.426, 2.106)),
    "norway":      ((1.288, 0.705, 1.483), (2.481, 0.418, 3.160), (1.260, 0.632, 1.438)),
    "portugal":    ((1.882, 0.621, 2.805), (1.418, 1.217, 1.455), (1.809, 0.339, 2.703)),
    "uk":          ((1.548, 0.769, 1.998), (1.534, 1.152, 1.654), (1.537, 0.677, 1.972)),
    "switzerland": ((2.148, 0.141, 2.624), (2.759, 0.344, 3.372), (2.067, 0.107, 2.537)),
}

FREQ = 2 * np.pi / 365.0


def find_draws(label: str) -> pd.DataFrame:
    hits = sorted((RESULTS / "fits").glob(f"{label}-*/02-posterior-*/seed_*/draws.tsv"))
    if not hits:
        raise FileNotFoundError(f"no draws.tsv for {label}")
    return pd.read_csv(hits[-1], sep="\t")


def simulate(model: str, p: dict, t_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the paper's ODE in proportions.

    Returns (reported incidence for days 1..t_end, x for days 1..t_end).
    """
    if model == "sirx":
        def fun(t, y):
            season = 1.0 + p["b"] * np.cos((t - p["phi"]) * FREQ)
            e1 = p["beta"] * season * (1.0 - y[2] * p["eps"]) * y[0] * y[1]
            e2 = p["gamma"] * y[1]
            e3 = (p["kappa"] * y[2] * (1.0 - y[2])
                  * (y[1] / p["eta0"] * np.exp(-p["lam"] * t) - p["c"]))
            return [-e1, e1 - e2, e3, e1]
        ic = [1.0 - p["i0"], p["i0"], p["x0"], 0.0]
    else:
        def fun(t, y):
            season = 1.0 + p["b"] * np.cos((t - p["phi"]) * FREQ)
            e1 = p["beta"] * season * 0.8 * y[0] * y[1]
            e2 = p["gamma"] * y[1]
            return [-e1, e1 - e2, 0.0, e1]
        ic = [1.0 - p["i0"], p["i0"], 0.0, 0.0]
    sol = solve_ivp(fun, (0.0, t_end + 1), ic, method="RK45",
                    t_eval=np.arange(0, t_end + 1, 1.0), rtol=1e-8, atol=1e-10)
    if sol.y.shape[1] != t_end + 1:
        return None, None   # diverged; the paper's code rejects these too
    rep = np.diff(sol.y[3]) / p["eta0"]
    return rep, sol.y[2][1:]


def r0_stat(curves: np.ndarray, cutoff: int) -> tuple[float, float, float]:
    """Table S3's statistic (from the authors' Particle_analysis.ipynb):
    max over the TRAINING window of the across-particle mean R0(t) curve,
    with a "CI" of (min lower band, max upper band) over the same window
    — despite the table caption saying "after 200 days"."""
    mean = curves.mean(axis=0)[:cutoff]
    lo = np.quantile(curves, 0.025, axis=0)[:cutoff]
    hi = np.quantile(curves, 0.975, axis=0)[:cutoff]
    return float(mean.max()), float(lo.min()), float(hi.max())


def aicc_terms(rss_over_n: list[tuple[float, int]], k: int) -> float:
    """Supplement eq. 18: sum of n_i ln(RSS_i/n_i) plus the shared
    small-sample penalty 2*K*N*n_t/(n_t - K*N - N)."""
    n_sets = len(rss_over_n)
    n_t = sum(n for _, n in rss_over_n)
    gof = sum(n * np.log(rss / n) for rss, n in rss_over_n)
    denom = n_t - k * n_sets - n_sets
    return gof + 2 * k * n_sets * (n_t / denom)


def main() -> None:
    traj_dir = RESULTS / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    rows, r0_rows = [], []
    for name, _, pop, cutoff, _, start in COUNTRIES:
        cases = pd.read_csv(HERE / f"data/{name}_cases.tsv", sep="\t")["cases"].values / pop
        osi = pd.read_csv(HERE / f"data/{name}_osi.tsv", sep="\t")["osi"].values / 1000.0
        n_all = len(cases)
        t_end = n_all
        pred = slice(cutoff, n_all)   # 0-based data rows in the prediction window
        n_pred = n_all - cutoff
        # the paper's "second peak" is the Sep-Dec maximum (Fig 1), so the
        # peak2 window is capped at Dec 31 even though prediction runs to
        # late January
        from datetime import date
        y, m, d = map(int, start.split("-"))
        dec31 = (date(2020, 12, 31) - date(y, m, d)).days + 1
        peak2_win = slice(cutoff, min(dec31, n_all))
        country_out = {}
        # "sirx" = the code's lam prior [0, 0.2]; "sirx_s2" = the
        # supplement's [0, 0.03], which reproduces the paper's estimates
        for model in ("sir", "sirx", "sirx_s2"):
            label = f"{name}_{model}"
            try:
                draws = find_draws(label)
            except FileNotFoundError as e:
                print(f"skip {label}: {e}", file=sys.stderr)
                continue
            mkind = "sir" if model == "sir" else "sirx"
            k_fit = 7 if model == "sir" else 12
            pick = draws.iloc[RNG.choice(len(draws), size=min(N_DRAWS, len(draws)),
                                         replace=False)]
            reps, xs, r0s, reffs = [], [], [], []
            for _, d in pick.iterrows():
                p = d.to_dict()
                if mkind == "sirx":
                    p.setdefault("sigma_x", 300.0)
                rep, x = simulate(mkind, p, t_end)
                if rep is None:
                    continue
                reps.append(rep[:n_all])
                xs.append(x[:n_all])
                # Table S3's caption says "after 200 days", but its SIR
                # values only reproduce when R0(t) is pooled over the full
                # year (median seasonal factor ~ 1): full-axis pooling
                tt = np.arange(1.0, n_all + 1, 1.0)
                season = 1.0 + p["b"] * np.cos((tt - p["phi"]) * FREQ)
                r0_t = p["beta"] * season / p["gamma"]
                r0s.append(r0_t)
                if mkind == "sirx":
                    reffs.append(r0_t * (1.0 - p["eps"] * x[:n_all]))
            reps, xs = np.array(reps), np.array(xs)
            # the paper's metrics use "the average simulation from 100
            # selected particles" (lowest-error); mirror that selection to
            # keep chains stuck in poor local optima from polluting the
            # average (the MH chains do not always agree)
            train_mse = np.mean((reps[:, :cutoff] - cases[:cutoff]) ** 2, axis=1)
            best = np.argsort(train_mse)[:100]
            reps, xs = reps[best], xs[best]
            r0s = [r0s[i] for i in best]
            if reffs:
                reffs = [reffs[i] for i in best]
            avg = reps.mean(axis=0)
            q = np.quantile(reps, [0.025, 0.5, 0.975], axis=0)
            traj = pd.DataFrame({
                "time": np.arange(1, n_all + 1), "data": cases,
                "mean": avg, "q025": q[0], "med": q[1], "q975": q[2],
            })
            if mkind == "sirx":
                xq = np.quantile(xs, [0.025, 0.5, 0.975], axis=0)
                traj["x_mean"] = xs.mean(axis=0)
                traj["x_q025"], traj["x_med"], traj["x_q975"] = xq
                traj["osi"] = osi
            traj.to_csv(traj_dir / f"{label}.tsv", sep="\t", index=False)

            # peaks: of the average simulation (the paper's convention),
            # CrI from per-draw peaks
            metrics = {"country": name, "model": model}
            for wname, sl in [("peak1", slice(0, cutoff)), ("peak2", peak2_win)]:
                seg = avg[sl]
                metrics[f"{wname}_day"] = sl.start + int(seg.argmax()) + 1
                metrics[f"{wname}_mag"] = float(seg.max())
                per_draw_day = sl.start + reps[:, sl].argmax(axis=1) + 1
                per_draw_mag = reps[:, sl].max(axis=1)
                metrics[f"{wname}_day_lo"], metrics[f"{wname}_day_hi"] = \
                    np.quantile(per_draw_day, [0.025, 0.975])
                metrics[f"{wname}_mag_lo"], metrics[f"{wname}_mag_hi"] = \
                    np.quantile(per_draw_mag, [0.025, 0.975])
            metrics["data_peak1_day"] = int(cases[:cutoff].argmax()) + 1
            metrics["data_peak1_mag"] = float(cases[:cutoff].max())
            metrics["data_peak2_day"] = cutoff + int(cases[peak2_win].argmax()) + 1
            metrics["data_peak2_mag"] = float(cases[peak2_win].max())

            # area between average simulation and data, prediction window
            metrics["area_pred"] = float(np.trapezoid(np.abs(cases[pred] - avg[pred])))

            # AICc on the prediction window (paper Fig 5a). The coupled
            # model is penalised with its full parameter count both ways.
            rss_inc = float(np.mean((cases[pred] - avg[pred]) ** 2))
            metrics["aicc_inc"] = aicc_terms([(rss_inc, n_pred)], k_fit)
            if mkind == "sirx":
                x_avg = xs.mean(axis=0)
                rss_osi = float(np.mean((osi[pred] - x_avg[pred]) ** 2))
                metrics["aicc_inc_osi"] = aicc_terms(
                    [(rss_inc, n_pred), (rss_osi, n_pred)], k_fit)
                metrics["osi_rms_train"] = float(np.sqrt(np.mean(
                    (osi[:cutoff] - x_avg[:cutoff]) ** 2)))
                # does predicted NPI support rise during the second wave?
                metrics["x_rebound"] = float(x_avg[n_all - 1] - x_avg[pred].min())

            # training-window fit quality (Table S4 analogue)
            rss_train = float(np.mean((cases[:cutoff] - avg[:cutoff]) ** 2))
            sst = float(np.mean((cases[:cutoff] - cases[:cutoff].mean()) ** 2))
            metrics["r2adj_train"] = 1.0 - (rss_train / (cutoff - k_fit - 1)) / \
                (sst / (cutoff - 1))

            # R0 / Reff, the paper's Table S3 statistic
            metrics["r0_med"], metrics["r0_lo"], metrics["r0_hi"] = \
                r0_stat(np.array(r0s), cutoff)
            if mkind == "sirx":
                metrics["reff_med"], metrics["reff_lo"], metrics["reff_hi"] = \
                    r0_stat(np.array(reffs), cutoff)
            rows.append(metrics)
            country_out[model] = metrics

        if {"sir", "sirx_s2"} <= country_out.keys():
            ps = PAPER_TABLE_S3[name]
            m = country_out
            r0_rows.append({
                "country": name,
                "camdl_sirx_r0": m["sirx_s2"]["r0_med"], "paper_sirx_r0": ps[0][0],
                "camdl_sir_r0": m["sir"]["r0_med"], "paper_sir_r0": ps[1][0],
                "camdl_reff": m["sirx_s2"]["reff_med"], "paper_reff": ps[2][0],
            })

    pd.DataFrame(rows).to_csv(RESULTS / "summary_stats.tsv", sep="\t", index=False)
    pd.DataFrame(r0_rows).to_csv(RESULTS / "table_s3_comparison.tsv", sep="\t",
                                 index=False)
    df = pd.DataFrame(rows)
    for model in ("sir", "sirx", "sirx_s2"):
        sub = df[df.model == model]
        if len(sub) == 0:
            continue
        print(f"\n== {model} across {len(sub)} countries ==")
        print(f"AICc (incidence):    {sub.aicc_inc.mean():8.0f} +/- {sub.aicc_inc.std():.0f}")
        if model.startswith("sirx") and "aicc_inc_osi" in sub:
            print(f"AICc (inc+osi):      {sub.aicc_inc_osi.mean():8.0f} +/- {sub.aicc_inc_osi.std():.0f}")
        print(f"area (pred):         {sub.area_pred.mean():8.4f} +/- {sub.area_pred.std():.4f}")
        print(f"peak2 mag:           {sub.peak2_mag.mean():8.4f} +/- {sub.peak2_mag.std():.4f}")
        print(f"peak2 day:           {sub.peak2_day.mean():8.1f} +/- {sub.peak2_day.std():.1f}")
        print(f"data peak2 mag:      {sub.data_peak2_mag.mean():8.4f} +/- {sub.data_peak2_mag.std():.4f}")
        print(f"data peak2 day:      {sub.data_peak2_day.mean():8.1f} +/- {sub.data_peak2_day.std():.1f}")


if __name__ == "__main__":
    main()
