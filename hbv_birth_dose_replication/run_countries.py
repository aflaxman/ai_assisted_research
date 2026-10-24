"""Country-level analysis: the 46 LMICs reporting birth-dose coverage below 90%.

Mirrors the ``region == 0`` branch of the authors' optimisation script and
their Figure 4: for every country below 90% coverage, find the cheapest
allocation across the five delivery strategies that reaches 90%, simulate the
baseline and optimised cohorts in camdl, and report the incremental cost,
DALYs averted, ICER and the share of optimised expenditure that goes to CPAD.

Writes ``results/countries_90pct.tsv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import hbv_model as m
from run_replication import YLD_FACTOR  # noqa: E402

RESULTS = m.ROOT / "results"
inp = m.load_inputs("countries")
ve = m.vaccine_effectiveness(inp)
weight = float(inp.glob.loc["weight", "Value"])

below = inp.regions[inp.regions["birth_dose"] < 0.9].reset_index(drop=True)
print(f"{len(below)} countries below 90% birth-dose coverage")

param_sets, meta = [], []
for _, row in below.iterrows():
    p_fac, com_sba = float(row["Facility"]), float(row["SBA_nfac"])
    unit = np.array([float(row[c]) for c in m.VACCINE_COST_COLS])
    fac_cov, com_cov = m.weighted_baseline_coverage(float(row["birth_dose"]), p_fac, weight)
    sat, sat_costing = m.saturation_levels(fac_cov, com_cov, com_sba)
    curve = m.CostCoverage(unit, sat, p_fac)

    cov0 = m.scenario_coverage(0, fac_cov, com_cov, sat)
    vcost0 = m.vaccine_costs(cov0, unit, p_fac, sat, sat_costing)
    x_90 = curve.optimise_target(0.9)
    cov90 = curve.coverage(x_90)
    for name, cov, vcost in [("baseline", cov0, vcost0), ("optimised_90", cov90, np.append(x_90, x_90.sum()))]:
        pf, pc = m.protection(cov, ve, com_sba)
        param_sets.append(m.cohort_params(inp, row, pf, pc))
        meta.append(dict(
            scenario=name, country=row["Setting"], births_2017=float(row["2017_births"]),
            baseline_birth_dose=float(row["birth_dose"]), p_facility=p_fac,
            coverage=cov.total(p_fac), fac_cc=cov.fac_cc, fac_ctc=cov.fac_ctc, com_cc=cov.com_cc,
            com_ctc=cov.com_ctc, com_cpad=cov.com_cpad,
            vaccine_cost=vcost[5], vc_fac_cc=vcost[0], vc_fac_ctc=vcost[1], vc_com_cc=vcost[2],
            vc_com_ctc=vcost[3], vc_com_cpad=vcost[4], cpad_share=vcost[4] / vcost[5] if vcost[5] else 0.0,
        ))

print(f"running {len(param_sets)} camdl cohorts ...")
for info, q in zip(meta, m.simulate_many(param_sets)):
    o = m.outcomes_per_1000(q)
    info.update(o)
    info["dalys"] = o["yll"] + o["yld"]
    info["dalys_as_published"] = o["yll"] + o["yld"] * YLD_FACTOR
    info["total_cost"] = info["vaccine_cost"] + o["disease_cost"]

df = pd.DataFrame(meta)
base = df[df.scenario == "baseline"].set_index("country")
opt = df[df.scenario == "optimised_90"].set_index("country")
out = pd.DataFrame({
    "baseline_birth_dose": base["baseline_birth_dose"],
    "p_facility": base["p_facility"],
    "births_2017": base["births_2017"],
    "baseline_vaccine_cost": base["vaccine_cost"],
    "opt90_vaccine_cost": opt["vaccine_cost"],
    "opt90_vc_fac_cc": opt["vc_fac_cc"], "opt90_vc_fac_ctc": opt["vc_fac_ctc"],
    "opt90_vc_com_cc": opt["vc_com_cc"], "opt90_vc_com_ctc": opt["vc_com_ctc"],
    "opt90_vc_com_cpad": opt["vc_com_cpad"],
    "opt90_cpad_share": opt["cpad_share"],
    "opt90_com_cpad_coverage": opt["com_cpad"],
    "annual_additional_vaccine_cost_total": (opt["vaccine_cost"] - base["vaccine_cost"]) * base["births_2017"] / m.PER,
    "dalys_averted_as_published": base["dalys_as_published"] - opt["dalys_as_published"],
    "dalys_averted": base["dalys"] - opt["dalys"],
    "additional_cost": opt["total_cost"] - base["total_cost"],
})
out["icer_as_published"] = out["additional_cost"] / out["dalys_averted_as_published"]
out["cost_saving"] = out["additional_cost"] < 0
out = out.sort_values("opt90_vaccine_cost", ascending=False)
out.to_csv(RESULTS / "countries_90pct.tsv", sep="\t", float_format="%.6g")
df.to_csv(RESULTS / "countries_scenarios_per_1000.tsv", sep="\t", index=False, float_format="%.6g")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(out[["baseline_birth_dose", "opt90_vaccine_cost", "opt90_cpad_share", "dalys_averted_as_published",
           "additional_cost", "icer_as_published"]].round(3).to_string())
print(f"\ncost saving in {int(out.cost_saving.sum())} of {len(out)} countries")
print("not cost saving:", ", ".join(f"{c} (ICER ${out.loc[c,'icer_as_published']:.2f})" for c in out.index[~out.cost_saving]))
print("no CPAD needed:", ", ".join(out.index[out.opt90_vc_com_cpad < 1e-6]))
dep = out[out.opt90_vc_com_cpad >= 1e-6].opt90_cpad_share
print(f"CPAD share of optimised expenditure elsewhere: {dep.min():.4%} ({dep.idxmin()}) to {dep.max():.1%} ({dep.idxmax()})")
