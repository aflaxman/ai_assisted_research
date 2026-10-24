"""Deterministic replication of Seaman et al. (2020) for the six GBD regions.

For every setting (six GBD regions plus their aggregate) this runs the
baseline, the four scenarios and the two optimisations (reallocating the
baseline budget; the cheapest route to 90% coverage), simulating each cohort
in camdl. It writes

* ``results/scenarios_per_1000.tsv`` -- coverage, vaccine cost, infections,
  deaths, YLL, YLD, DALYs and disease cost per 1000 births;
* ``results/population_totals.tsv`` -- the same scaled to each region's 2017
  births (Figure 2 / the $687.5 million headline);
* ``results/comparison_published_code.tsv`` -- camdl against the authors'
  own code run on the same inputs (``data/published_code_reference.tsv``).

DALYs are reported two ways. ``dalys_as_published`` multiplies the YLD
integral by 2997/500/0.2 = 29.97, the factor the authors' code applies to
disability-weighted prevalence per time step; ``dalys`` is YLL + YLD with
the YLD integral taken at face value (disability weight x person-years).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import hbv_model as m

RESULTS = m.ROOT / "results"
RESULTS.mkdir(exist_ok=True)

#: The authors' per-step YLD multiplier (2997 / len(t_steps)) divided by the
#: step length dt they integrate with: how far their YLD sits above the
#: integral of disability-weighted prevalence.
YLD_FACTOR = (2997 / 500) / 0.2

inp = m.load_inputs("regions")
ve = m.vaccine_effectiveness(inp)
weight = float(inp.glob.loc["weight", "Value"])

rows = []
param_sets = []
meta = []

for _, row in inp.regions.iterrows():
    setting = row["Setting"]
    p_fac, com_sba = float(row["Facility"]), float(row["SBA_nfac"])
    unit = np.array([float(row[c]) for c in m.VACCINE_COST_COLS])
    fac_cov, com_cov = m.weighted_baseline_coverage(float(row["birth_dose"]), p_fac, weight)
    sat, sat_costing = m.saturation_levels(fac_cov, com_cov, com_sba)
    curve = m.CostCoverage(unit, sat, p_fac)

    plans: list[tuple[str, m.Coverage, np.ndarray]] = []
    for scen in range(5):
        cov = m.scenario_coverage(scen, fac_cov, com_cov, sat)
        plans.append((m.SCENARIOS[scen], cov, m.vaccine_costs(cov, unit, p_fac, sat, sat_costing)))
    baseline_budget = plans[0][2][5]
    x_budget = curve.optimise_budget(baseline_budget)
    plans.append(("optimised_baseline", curve.coverage(x_budget), np.append(x_budget, x_budget.sum())))
    x_90 = curve.optimise_target(0.9)
    plans.append(("optimised_90", curve.coverage(x_90), np.append(x_90, x_90.sum())))

    for name, cov, vcost in plans:
        prot_fac, prot_com = m.protection(cov, ve, com_sba)
        param_sets.append(m.cohort_params(inp, row, prot_fac, prot_com))
        meta.append(dict(
            scenario=name, region=setting, births_2017=float(row["2017_births"]),
            coverage=cov.total(p_fac), fac_cc=cov.fac_cc, fac_ctc=cov.fac_ctc,
            com_cc=cov.com_cc, com_ctc=cov.com_ctc, com_cpad=cov.com_cpad,
            prot_fac=prot_fac, prot_com=prot_com,
            vaccine_cost=vcost[5], vc_fac_cc=vcost[0], vc_fac_ctc=vcost[1],
            vc_com_cc=vcost[2], vc_com_ctc=vcost[3], vc_com_cpad=vcost[4],
        ))

print(f"running {len(param_sets)} camdl cohorts ...")
quantities = m.simulate_many(param_sets)

for info, q in zip(meta, quantities):
    o = m.outcomes_per_1000(q)
    info.update(o)
    info["yld_as_published"] = o["yld"] * YLD_FACTOR
    info["dalys"] = o["yll"] + o["yld"]
    info["dalys_as_published"] = o["yll"] + info["yld_as_published"]
    info["total_cost"] = info["vaccine_cost"] + o["disease_cost"]
    rows.append(info)

df = pd.DataFrame(rows)

# Incremental results against the baseline of the same region.
base = df[df.scenario == "baseline"].set_index("region")
df["dalys_averted"] = df.apply(lambda r: base.loc[r.region, "dalys"] - r["dalys"], axis=1)
df["dalys_averted_as_published"] = df.apply(
    lambda r: base.loc[r.region, "dalys_as_published"] - r["dalys_as_published"], axis=1)
df["additional_cost"] = df.apply(lambda r: r["total_cost"] - base.loc[r.region, "total_cost"], axis=1)
df["icer_as_published"] = df["additional_cost"] / df["dalys_averted_as_published"]
df["icer"] = df["additional_cost"] / df["dalys_averted"]
df.to_csv(RESULTS / "scenarios_per_1000.tsv", sep="\t", index=False, float_format="%.6g")

# Population scale (Figure 2, the $687.5 million).
pop = df.copy()
scale = pop["births_2017"] / m.PER
for col in ["vaccine_cost", "vc_fac_cc", "vc_fac_ctc", "vc_com_cc", "vc_com_ctc", "vc_com_cpad",
            "disease_cost", "total_cost", "dalys", "dalys_as_published", "dalys_averted",
            "dalys_averted_as_published", "additional_cost", "infections", "hbv_deaths"]:
    pop[col] = pop[col] * scale
pop = pop[["scenario", "region", "births_2017", "coverage", "vaccine_cost", "vc_fac_cc", "vc_fac_ctc",
           "vc_com_cc", "vc_com_ctc", "vc_com_cpad", "infections", "hbv_deaths", "disease_cost",
           "total_cost", "dalys", "dalys_as_published", "dalys_averted", "dalys_averted_as_published",
           "additional_cost"]]
pop.to_csv(RESULTS / "population_totals.tsv", sep="\t", index=False, float_format="%.6g")

# Against the authors' code on identical inputs.
ref = pd.read_csv(m.ROOT / "data" / "published_code_reference.tsv", sep="\t")
cmp = df.merge(ref, on=["scenario", "region"], suffixes=("_camdl", "_code"))
cmp_out = pd.DataFrame({
    "scenario": cmp.scenario, "region": cmp.region,
    "coverage_camdl": cmp.coverage_camdl, "coverage_code": cmp.coverage_code,
    "vaccine_cost_camdl": cmp.vaccine_cost_camdl, "vaccine_cost_code": cmp.vaccine_cost_code,
    "infections_camdl": cmp.infections, "infections_code": cmp.latent0,
    "hbv_deaths_camdl": cmp.hbv_deaths_camdl, "hbv_deaths_code": cmp.hbv_deaths_code,
    "yll_camdl": cmp.yll_camdl, "yll_code": cmp.yll_code,
    "yld_camdl_as_published": cmp.yld_as_published, "yld_code": cmp.yld_code,
    "yld_camdl_integral": cmp.yld_camdl,
    "dalys_camdl_as_published": cmp.dalys_as_published, "dalys_code": cmp.dalys_code,
    "disease_cost_camdl": cmp.disease_cost_camdl, "disease_cost_code": cmp.disease_cost_code,
})
cmp_out.to_csv(RESULTS / "comparison_published_code.tsv", sep="\t", index=False, float_format="%.6g")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
show = ["scenario", "region", "coverage", "vaccine_cost", "infections", "hbv_deaths", "yll", "yld",
        "dalys", "dalys_as_published", "disease_cost", "dalys_averted_as_published", "additional_cost"]
print(df[show].round(3).to_string())

print("\n=== All GBD regions, population scale (US$ million / million DALYs) ===")
g = pop[pop.region == "All GBD Regions"].set_index("scenario")
for s in g.index:
    print(f"{s:20s} coverage {g.loc[s,'coverage']:.3f}  vaccine ${g.loc[s,'vaccine_cost']/1e6:8.1f}M  "
          f"DALYs(as published) {g.loc[s,'dalys_as_published']/1e6:6.2f}M  "
          f"averted {g.loc[s,'dalys_averted_as_published']/1e6:6.2f}M  "
          f"extra cost ${g.loc[s,'additional_cost']/1e6:8.1f}M")

print("\n=== camdl vs authors' code: ratio camdl/code, All GBD regions, baseline ===")
r = cmp[(cmp.region == "All GBD Regions") & (cmp.scenario == "baseline")].iloc[0]
for a, b in [("hbv_deaths_camdl", "hbv_deaths_code"), ("yll_camdl", "yll_code"),
             ("yld_as_published", "yld_code"), ("disease_cost_camdl", "disease_cost_code")]:
    print(f"  {a:22s} {r[a]:10.3f}  vs {r[b]:10.3f}   ratio {r[a]/r[b]:.3f}")
