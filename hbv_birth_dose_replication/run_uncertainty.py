"""Probabilistic uncertainty analysis, following the authors' ``uncertainity.py``.

Every global parameter is drawn uniformly between its lower and upper bound
(Table 1), every regional parameter uniformly between the ``regions_lb`` /
``regions_ub`` sheets (plus or minus 5% where no interval was available), and
HBsAg prevalence from a normal distribution truncated to its 95% interval.
The paper reports the median and IQR of 1000 such draws.

The cohort model has no transmission inside the cohort, so every outcome is
proportional to the number of perinatal infections. One camdl run per
(draw, region) with ``prot_fac = prot_com = 0`` therefore gives the outcomes
per infection for that draw; each scenario's outcomes follow by multiplying
by its number of infections, which is the Figure 1 arithmetic evaluated in
Python with the same drawn parameters. ``check_linearity`` verifies the
shortcut against direct camdl runs.

As in the authors' code, the two optimised strategies keep coverage, facility
births, professional attendance and vaccine prices at their point values (the
optimisation is done once, deterministically) and only the epidemiological
parameters vary.

Usage: ``uv run python run_uncertainty.py [N_DRAWS] [--check]``
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

import hbv_model as m
from run_replication import YLD_FACTOR  # noqa: E402  (also re-runs the deterministic layer; cheap)

RESULTS = m.ROOT / "results"
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
CHECK = "--check" in sys.argv
SEED = 1234

inp = m.load_inputs("regions")
rng = np.random.default_rng(SEED)
weight = float(inp.glob.loc["weight", "Value"])
settings = inp.settings
det = pd.read_csv(RESULTS / "scenarios_per_1000.tsv", sep="\t")

# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------

GLOBAL_SYMBOLS = list(m.GLOBAL_RATE_ROWS) + m.VE_ROWS


def draw_global() -> tuple[dict[str, float], np.ndarray]:
    """One draw of the Table 1 parameters: camdl dict and the five VE values."""
    g = inp.glob
    out = {}
    for sym, camdl in m.GLOBAL_RATE_ROWS.items():
        out[camdl] = rng.uniform(g.loc[sym, "Lower bound"], g.loc[sym, "Upper bound"])
    out["disc"] = float(g.loc["discount", "Value"])
    ve = np.array([rng.uniform(g.loc[r, "Lower bound"], g.loc[r, "Upper bound"]) for r in m.VE_ROWS])
    return out, ve


REGIONAL_UNIFORM = ["birth_dose", "HBV3", "SBA", "Facility"] + m.ACM_COLS + list(m.COST_COLS.values()) \
    + m.VACCINE_COST_COLS + ["SBA_nfac", "hbe_prev", "hbe_risk", "n_hbe_risk"]


def draw_regional(i: int) -> pd.Series:
    """One draw of a setting's row: uniform between the bound sheets, prevalence truncated normal."""
    pt, lb, ub = inp.regions.iloc[i], inp.regions_lb.iloc[i], inp.regions_ub.iloc[i]
    row = pt.copy()
    for c in REGIONAL_UNIFORM:
        row[c] = rng.uniform(float(lb[c]), float(ub[c]))
    mu, sd = float(pt["prev"]), float(pt["sigma"])
    a, b = (float(pt["prev_lb"]) - mu) / sd, (float(pt["prev_ub"]) - mu) / sd
    row["prev"] = stats.truncnorm.rvs(a, b, loc=mu, scale=sd, random_state=rng)
    return row


def infections_per_1000(row: pd.Series, prot_fac: float, prot_com: float, p_fac: float | None = None) -> float:
    """The Figure 1 arithmetic (the camdl ``init`` block), per 1000 births."""
    p_fac = float(row["Facility"]) if p_fac is None else p_fac
    risk = float(row["prev"]) * (float(row["hbe_prev"]) * float(row["hbe_risk"])
                                 + (1 - float(row["hbe_prev"])) * float(row["n_hbe_risk"]))
    return m.PER * risk * (p_fac * (1 - prot_fac) + (1 - p_fac) * (1 - prot_com))


# ---------------------------------------------------------------------------
# Optional check that scaling per-infection outcomes reproduces direct runs
# ---------------------------------------------------------------------------

if CHECK:
    row = inp.regions.iloc[6]
    ve0 = m.vaccine_effectiveness(inp)
    fac, com = m.weighted_baseline_coverage(float(row["birth_dose"]), float(row["Facility"]), weight)
    pf, pc = m.protection(m.Coverage(fac, 0, com, 0, 0), ve0, float(row["SBA_nfac"]))
    direct = m.outcomes_per_1000(m.simulate(m.cohort_params(inp, row, pf, pc)))
    unit = m.outcomes_per_1000(m.simulate(m.cohort_params(inp, row, 0.0, 0.0)))
    n = infections_per_1000(row, pf, pc)
    print("linearity check (direct vs scaled):")
    for k in ["hbv_deaths", "yll", "yld", "disease_cost"]:
        scaled = unit[k] / unit["infections"] * n
        print(f"  {k:14s} {direct[k]:12.5f} {scaled:12.5f}  rel diff {abs(direct[k]-scaled)/direct[k]:.2e}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print(f"drawing {N} parameter sets x {len(settings)} settings ...")
draws = []       # (draw, region) -> drawn row, global dict, ve
param_sets = []
for d in range(N):
    glob, ve = draw_global()
    for i, setting in enumerate(settings):
        row = draw_regional(i)
        draws.append((d, i, row, glob, ve))
        param_sets.append(m.cohort_params(inp, row, 0.0, 0.0, glob=glob, reg=m.regional_point(row)))

print(f"running {len(param_sets)} camdl cohorts ...")
unit_outcomes = m.simulate_many(param_sets, workers=4)

records = []
for (d, i, row, glob, ve), q in zip(draws, unit_outcomes):
    setting = settings[i]
    pt = inp.regions.iloc[i]
    per_inf = {k: v / q["infections"] for k, v in q.items() if k != "infections"}

    # Scenarios 0-4: everything drawn.
    p_fac, com_sba = float(row["Facility"]), float(row["SBA_nfac"])
    unit_cost = np.array([float(row[c]) for c in m.VACCINE_COST_COLS])
    fac_cov, com_cov = m.weighted_baseline_coverage(float(row["birth_dose"]), p_fac, weight)
    sat, sat_costing = m.saturation_levels(fac_cov, com_cov, com_sba)
    plans = []
    for scen in range(5):
        cov = m.scenario_coverage(scen, fac_cov, com_cov, sat)
        vcost = m.vaccine_costs(cov, unit_cost, p_fac, sat, sat_costing)[5]
        pf, pc = m.protection(cov, ve, com_sba, clamp=True)
        plans.append((m.SCENARIOS[scen], cov.total(p_fac), vcost, infections_per_1000(row, pf, pc)))

    # Optimised strategies: point coverage, prices and facility share; drawn epidemiology.
    p_fac0, com_sba0 = float(pt["Facility"]), float(pt["SBA_nfac"])
    for name in ["optimised_baseline", "optimised_90"]:
        r = det[(det.scenario == name) & (det.region == setting)].iloc[0]
        cov = m.Coverage(r.fac_cc, r.fac_ctc, r.com_cc, r.com_ctc, r.com_cpad)
        pf, pc = m.protection(cov, ve, com_sba0, clamp=True)
        plans.append((name, r.coverage, r.vaccine_cost, infections_per_1000(row, pf, pc, p_fac0)))

    for name, coverage, vcost, n_inf in plans:
        rec = dict(draw=d, region=setting, scenario=name, coverage=coverage, vaccine_cost=vcost,
                   infections=n_inf)
        for k, v in per_inf.items():
            rec[k] = v * n_inf
        rec["yld_as_published"] = rec["yld"] * YLD_FACTOR
        rec["dalys"] = rec["yll"] + rec["yld"]
        rec["dalys_as_published"] = rec["yll"] + rec["yld_as_published"]
        rec["total_cost"] = vcost + rec["disease_cost"]
        records.append(rec)

psa = pd.DataFrame(records)
base = psa[psa.scenario == "baseline"].set_index(["draw", "region"])
idx = pd.MultiIndex.from_frame(psa[["draw", "region"]])
psa["dalys_averted"] = base["dalys"].reindex(idx).to_numpy() - psa["dalys"].to_numpy()
psa["dalys_averted_as_published"] = base["dalys_as_published"].reindex(idx).to_numpy() - psa["dalys_as_published"].to_numpy()
psa["additional_cost"] = psa["total_cost"].to_numpy() - base["total_cost"].reindex(idx).to_numpy()
psa["icer_as_published"] = psa["additional_cost"] / psa["dalys_averted_as_published"]
psa.to_csv(RESULTS / "uncertainty_draws.tsv.gz", sep="\t", index=False, float_format="%.6g")

# Population-scale totals per draw (the paper's 36.3 million DALYs averted).
births = inp.regions.set_index("Setting")["2017_births"]
psa["dalys_averted_as_published_total"] = psa["dalys_averted_as_published"] * psa["region"].map(births) / m.PER
psa["dalys_averted_total"] = psa["dalys_averted"] * psa["region"].map(births) / m.PER
psa["additional_cost_total"] = psa["additional_cost"] * psa["region"].map(births) / m.PER


def q(s: pd.Series) -> pd.Series:
    return pd.Series({"median": s.median(), "q25": s.quantile(0.25), "q75": s.quantile(0.75)})


summary_cols = ["coverage", "vaccine_cost", "infections", "hbv_deaths", "dalys", "dalys_as_published",
                "disease_cost", "total_cost", "dalys_averted", "dalys_averted_as_published",
                "additional_cost", "icer_as_published", "dalys_averted_as_published_total",
                "dalys_averted_total", "additional_cost_total"]
summary = (psa.groupby(["region", "scenario"])[summary_cols]
           .agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]))
summary.columns = [f"{c}_{n}" for c, n in zip(summary.columns.get_level_values(0),
                                              np.tile(["median", "q25", "q75"], len(summary_cols)))]
summary = summary.reset_index()
summary["n_draws"] = N
summary["share_cost_saving"] = psa.groupby(["region", "scenario"])["additional_cost"] \
    .apply(lambda s: (s < 0).mean()).to_numpy()
summary.to_csv(RESULTS / "uncertainty_summary.tsv", sep="\t", index=False, float_format="%.6g")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
g = summary[summary.region == "All GBD Regions"].set_index("scenario")
print("\n=== All GBD regions, per 1000 births, median (IQR) over", N, "draws ===")
for s in g.index:
    print(f"{s:20s} cov {g.loc[s,'coverage_median']:.3f}  vacc ${g.loc[s,'vaccine_cost_median']:7.0f} "
          f"({g.loc[s,'vaccine_cost_q25']:.0f}-{g.loc[s,'vaccine_cost_q75']:.0f})  "
          f"DALYs(pub) {g.loc[s,'dalys_as_published_median']:5.0f} ({g.loc[s,'dalys_as_published_q25']:.0f}-{g.loc[s,'dalys_as_published_q75']:.0f})  "
          f"disease ${g.loc[s,'disease_cost_median']:6.0f} ({g.loc[s,'disease_cost_q25']:.0f}-{g.loc[s,'disease_cost_q75']:.0f})  "
          f"averted(pub) {g.loc[s,'dalys_averted_as_published_median']:4.0f} ({g.loc[s,'dalys_averted_as_published_q25']:.0f}-{g.loc[s,'dalys_averted_as_published_q75']:.0f})  "
          f"net cost ${g.loc[s,'additional_cost_median']:6.0f} ({g.loc[s,'additional_cost_q25']:.0f}-{g.loc[s,'additional_cost_q75']:.0f})")

# Paper: 36.3 million (IQR 30.1-43.4) DALYs averted summed over regions for the 90% optimum.
tot = (psa[(psa.scenario == "optimised_90") & (psa.region != "All GBD Regions")]
       .groupby("draw")[["dalys_averted_as_published_total", "dalys_averted_total", "additional_cost_total"]].sum())
print("\n=== Six-region sum, optimised 90% coverage, per cohort ===")
for c in tot.columns:
    print(f"  {c:36s} median {tot[c].median()/1e6:8.2f}M  IQR {tot[c].quantile(.25)/1e6:8.2f}M - {tot[c].quantile(.75)/1e6:8.2f}M")
ssa = psa[(psa.scenario == "optimised_90") & (psa.region == "Sub-Saharan Africa")].set_index("draw")
print(f"  Sub-Saharan Africa share of averted DALYs: {(ssa['dalys_averted_as_published_total'] / tot['dalys_averted_as_published_total']).median():.2f}")
tot.to_csv(RESULTS / "uncertainty_optimised90_totals.tsv", sep="\t", float_format="%.6g")

print("\n=== ICER (US$ per DALY averted, as-published DALYs), regions where not always cost saving ===")
for (reg, sc), grp in psa.groupby(["region", "scenario"]):
    if sc in ("optimised_90", "scenario4") and (grp.additional_cost > 0).any():
        print(f"  {reg:45s} {sc:14s} share cost-saving {(grp.additional_cost<0).mean():.2f}  "
              f"ICER median {grp.icer_as_published.median():7.2f}  q75 {grp.icer_as_published.quantile(.75):7.2f}  max {grp.icer_as_published.max():7.2f}")
