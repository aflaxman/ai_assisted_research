"""A retrospective "digital twin" of London's measles system, 1946-58.

Replays the historical record as if it were arriving live. Every 4 weeks
the twin:

1. ASSIMILATES everything reported so far -- a bootstrap particle filter
   (`camdl pfilter`) over the case series up to "today", saving the
   filtered particle cloud p(state | data so far);
2. NOWCASTS the hidden state -- quantiles of the susceptible/exposed/
   infectious compartments nobody can observe directly;
3. FORECASTS 8 weeks ahead -- `camdl simulate --init-state` restarts the
   model from the filtered cloud (`gh#641` forecast workflow) and emits a
   200-member predictive fan;
4. SCORES itself once the "future" arrives -- sample CRPS, interval
   coverage, and CRPS of two point baselines (persistence and
   seasonal-naive) for comparison.

Everything runs at the fitted scout MLE (results/london_mle_params.toml);
parameters are held fixed, so this demo exercises the state-updating loop
of a twin, not the parameter-drift loop (see README).

Run (about 10-20 minutes):  uv run python digital_twin_replay.py
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "results" / "twin"
OUT.mkdir(parents=True, exist_ok=True)

CITY = "london"
PARAMS = HERE / "results" / f"{CITY}_mle_params.toml"
CASES = HERE / "data" / f"{CITY}_cases.tsv"
COVARIATES = HERE / "data" / f"{CITY}_covariates.tsv"
MODEL = f"{CITY}_seir.camdl"

PARTICLES = 2000
FORECAST_MEMBERS = 200          # particle subsample restarted as forecast fan
HORIZON_DAYS = 56               # 8 weeks
CADENCE_DAYS = 28               # assimilate/forecast every 4 weeks
WARMUP_DAYS = 728               # first 2 years: assimilate only, no forecast
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def crps_sample(samples, y):
    """Sample-based CRPS estimator: E|X-y| - 0.5 E|X-X'|."""
    x = np.asarray(samples, float)
    return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :])))


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=HERE, check=True, capture_output=True, text=True, **kw)


def main():
    cases = pd.read_csv(CASES, sep="\t")
    obs = dict(zip(cases["time"], cases["weekly_cases"]))
    t_end = int(cases["time"].max())
    cov = pd.read_csv(COVARIATES, sep="\t")

    issue_times = list(range(WARMUP_DAYS, t_end - HORIZON_DAYS + 1, CADENCE_DAYS))
    print(f"{CITY}: {len(issue_times)} assimilation dates, "
          f"{PARTICLES} particles, {FORECAST_MEMBERS}-member fans")

    nowcast_rows, fq_rows, score_rows, assim_rows = [], [], [], []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, t_now in enumerate(issue_times):
            # 1. assimilate: filter the series up to t_now
            upto = tmp / f"upto_{t_now}.tsv"
            cases[cases["time"] <= t_now].to_csv(upto, sep="\t", index=False)
            state = tmp / f"state_{t_now}.tsv"
            r = run(["camdl", "pfilter", MODEL, "--params", str(PARAMS),
                     "--data", str(upto), "--particles", str(PARTICLES),
                     "--seed", str(1000 + k), "--save-final-state", str(state),
                     "--no-progress"])
            loglik = float(r.stdout.strip().splitlines()[0])
            assim_rows.append({"t": t_now, "loglik": loglik})

            # 2. nowcast: hidden-state quantiles from the filtered cloud
            cloud = pd.read_csv(state, sep="\t", comment="#")
            pop_now = float(np.interp(t_now, cov["t"], cov["pop"]))
            row = {"t": t_now, "pop": pop_now}
            for comp in ["S", "E", "I"]:
                for q in QUANTILES:
                    row[f"{comp}_q{int(q * 100):02d}"] = float(cloud[comp].quantile(q))
                row[f"{comp}_mean"] = float(cloud[comp].mean())
            nowcast_rows.append(row)

            # 3. forecast: restart the model from a particle subsample
            stride = PARTICLES // FORECAST_MEMBERS
            sub = cloud.iloc[::stride].head(FORECAST_MEMBERS)
            substate = tmp / f"substate_{t_now}.tsv"
            with open(state) as f:
                header = f.readline()  # "# camdl-final-state v1  t=..."
            with open(substate, "w") as f:
                f.write(header)
                sub.to_csv(f, sep="\t", index=False)
            fc = tmp / f"fc_{t_now}.tsv"
            run(["camdl", "simulate", MODEL, "--params", str(PARAMS),
                 "--init-state", str(substate),
                 "--replicates", str(len(sub)),
                 "--to", str(t_now + HORIZON_DAYS),
                 "--backend", "chain_binomial", "--dt", "1.0",
                 "--seed", str(2000 + k), "--obs-only", str(fc),
                 "--no-progress"])
            fan = pd.read_csv(fc, sep="\t")
            fan = fan[fan["time"] > t_now]

            # 4. score each horizon against what actually arrived
            for h_wk in range(1, HORIZON_DAYS // 7 + 1):
                t_target = t_now + 7 * h_wk
                samples = fan.loc[fan["time"] == t_target, "weekly_cases"].to_numpy()
                if len(samples) == 0 or t_target not in obs:
                    continue
                y = float(obs[t_target])
                qs = np.quantile(samples, QUANTILES)
                fq_rows.append({"t_issue": t_now, "t_target": t_target, "h_weeks": h_wk,
                                **{f"q{int(q * 100):02d}": v for q, v in zip(QUANTILES, qs)},
                                "mean": float(samples.mean()), "observed": y})
                y_persist = float(obs[t_now])
                y_seasonal = float(obs.get(t_target - 364, np.nan))
                score_rows.append({
                    "t_issue": t_now, "t_target": t_target, "h_weeks": h_wk,
                    "observed": y,
                    "crps_twin": crps_sample(samples, y),
                    "crps_persistence": abs(y - y_persist),
                    "crps_seasonal": abs(y - y_seasonal),
                    "in50": int(qs[1] <= y <= qs[3]),
                    "in90": int(qs[0] <= y <= qs[4]),
                })

            if k % 10 == 0:
                print(f"  t={t_now:5d} ({1943.996 + t_now / 365.25:.2f}) "
                      f"loglik={loglik:9.1f} S_med={row['S_q50']:,.0f}")

    pd.DataFrame(nowcast_rows).to_csv(OUT / f"{CITY}_nowcast.tsv", sep="\t", index=False)
    pd.DataFrame(fq_rows).to_csv(OUT / f"{CITY}_forecast_quantiles.tsv", sep="\t", index=False)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT / f"{CITY}_scores.tsv", sep="\t", index=False)
    pd.DataFrame(assim_rows).to_csv(OUT / f"{CITY}_assimilation.tsv", sep="\t", index=False)

    # 5. camdl's own one-step-ahead scorecard over the full series
    run(["camdl", "pfilter", MODEL, "--params", str(PARAMS),
         "--data", str(CASES), "--particles", str(PARTICLES), "--seed", "77",
         "--save-prequential", str(OUT / f"{CITY}_prequential"),
         "--no-save-samples", "--no-progress"])

    summary = scores.groupby("h_weeks")[
        ["crps_twin", "crps_persistence", "crps_seasonal", "in50", "in90"]
    ].mean()
    summary.to_csv(OUT / f"{CITY}_score_summary.tsv", sep="\t")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
