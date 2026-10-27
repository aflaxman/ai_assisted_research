"""Run the authors' published Python model verbatim, as the reference point.

Seaman et al. (2020) released their code at
https://github.com/ChrisSeaman-Burnet/Seaman-et-al.-HBV-Birth-Dose-Model .
This driver imports those functions unmodified (pass the clone's ``code/``
directory as the first argument), reproduces ``mainscript.py`` for the GBD
regions with no uncertainty runs, and writes the per-1000-births outcomes
for every scenario to ``data/published_code_reference.tsv``.

It exists so that the camdl model can be compared against what the authors'
code actually computes, separately from what the paper prints.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
XLSX = ROOT / "data" / "hbv_model_inputs_revision.xlsx"

code_dir = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(code_dir))

from model import (  # noqa: E402
    coverage_saturation, hbv_model, model_initializer, model_scenarios, vaccine_costs,
)
from model_setup import facility_weight, time_steps  # noqa: E402

# --- mainscript.py, region = 1 (GBD regions), runs = 1 -----------------------
xls = pd.ExcelFile(XLSX)
globz = pd.read_excel(xls, "global", index_col=None).iloc[0:30, 1:3]
hbv_pars = pd.DataFrame(pd.read_excel(xls, "regions"))
runs, ve_warm, region = 1, 1, 1

Tin = globz.iloc[1, 1]
t_steps, dt = time_steps(Tin, 5)
if len(t_steps) < 400:
    globz.iloc[14, 1:] = 1 / dt
    globz.iloc[15, 1:] = 1 / dt
    globz.iloc[17, 1:] = 1 / dt
elif len(t_steps) >= 400 or len(t_steps) < 2400:
    globz.iloc[14, 1:] = 1 / dt
print(f"t_steps={len(t_steps)} dt={dt} r_LA={globz.iloc[14,1]} r_AC={globz.iloc[15,1]} r_AZ={globz.iloc[17,1]}")

P = globz.iloc[0, 1]
weight = globz.iloc[2, 1]
discount = globz.iloc[3, 1]
regions_perturbed = globz_perturbed = 0
fac_cov_weighted, com_cov_weighted = facility_weight(runs, weight, hbv_pars, regions_perturbed)
settings = list(hbv_pars["Setting"])
saturation, sat_costing = coverage_saturation(fac_cov_weighted, com_cov_weighted, hbv_pars, regions_perturbed, runs)

SCEN_NAMES = ["baseline", "scenario1", "scenario2", "scenario3", "scenario4"]
rows = []


# Age-dependent chronicity exactly as hbv_model() builds it, so that the number
# of chronic infections the Euler scheme produces can be read off a trajectory.
p_AC = np.array([globz.iloc[16, 1] if y < 0.5 else np.exp(-0.645 * y ** 0.455) for y in t_steps])
r_AC = globz.iloc[15, 1]


def record(name, cov, vcost, traj, fac_cc, fac_ctc, com_cc, com_ctc, com_cpad):
    for reg, setting in enumerate(settings):
        last = traj[reg, -1, :]
        chronic = float(np.sum(dt * p_AC[:-1] * r_AC * traj[reg, :-1, 3]))
        rows.append(dict(
            scenario=name, region=setting, coverage=cov[reg],
            fac_cc=fac_cc[reg], fac_ctc=fac_ctc[reg], com_cc=com_cc[reg],
            com_ctc=com_ctc[reg], com_cpad=com_cpad[reg],
            vaccine_cost=vcost[reg, 5], vc_fac_cc=vcost[reg, 0], vc_fac_ctc=vcost[reg, 1],
            vc_com_cc=vcost[reg, 2], vc_com_ctc=vcost[reg, 3], vc_com_cpad=vcost[reg, 4],
            chronic_cases=chronic, immune0=traj[reg, 0, 0], susceptible0=traj[reg, 0, 1], latent0=traj[reg, 0, 2],
            hbv_deaths=last[8], yll=last[9], yld=last[10], disease_cost=last[11],
            total_cost=last[12], dalys=last[13],
        ))


scen_out = {}
for scen in range(5):
    fac_cc, fac_ctc, com_cc, com_ctc, com_cpad = model_scenarios(
        runs, scen, hbv_pars, fac_cov_weighted, com_cov_weighted, saturation)
    model_init = model_initializer(runs, globz, hbv_pars, globz_perturbed, regions_perturbed,
                                   fac_cc, fac_ctc, com_cc, com_ctc, com_cpad, t_steps, ve_warm)
    vcost, cov = vaccine_costs(runs, hbv_pars, fac_cc, fac_ctc, com_cc, com_ctc, com_cpad,
                               sat_costing, saturation, regions_perturbed)
    for reg in range(len(hbv_pars)):
        model_init[reg, 0, -2] = vcost[reg, 5]
    traj = hbv_model(runs, t_steps, dt, model_init, hbv_pars, regions_perturbed, globz,
                     globz_perturbed, discount)
    scen_out[scen] = (vcost, cov, traj)
    record(SCEN_NAMES[scen], cov, vcost, traj, fac_cc, fac_ctc, com_cc, com_ctc, com_cpad)
    if scen == 0:
        baseline_vcost, baseline_coverage, baseline = vcost, cov, traj

# --- costcoverage.py + optimisation_model.py (region == 1, runs == 1) --------
# These scripts are written to be exec'd into mainscript's namespace.
exec(open(code_dir / "costcoverage.py").read())
exec(open(code_dir / "optimisation_model.py").read())

record("optimised_baseline", np.array([-objective_coverage(e[:5]) / 1000 for e in baseline_optimised_expenditure])
       if False else np.array([(baseline_optimised_coverage[r, 0] + baseline_optimised_coverage[r, 2]
                                + baseline_optimised_coverage[r, 4] + baseline_optimised_coverage[r, 6]
                                + baseline_optimised_coverage[r, 8]) / 1000 for r in range(len(hbv_pars))]),
       np.column_stack([baseline_optimised_expenditure[:, :5], baseline_optimised_expenditure[:, 5]]),
       baseline_optimised,
       baseline_optimised_coverage[:, 1], baseline_optimised_coverage[:, 3],
       baseline_optimised_coverage[:, 5], baseline_optimised_coverage[:, 7],
       baseline_optimised_coverage[:, 9])
record("optimised_90",
       np.array([(optimised_90_coverage[r, 0] + optimised_90_coverage[r, 2] + optimised_90_coverage[r, 4]
                  + optimised_90_coverage[r, 6] + optimised_90_coverage[r, 8]) / 1000
                 for r in range(len(hbv_pars))]),
       np.column_stack([optimised_90_expenditure[:, :5], optimised_90_expenditure[:, 5]]),
       optimised_90,
       optimised_90_coverage[:, 1], optimised_90_coverage[:, 3],
       optimised_90_coverage[:, 5], optimised_90_coverage[:, 7], optimised_90_coverage[:, 9])

out = pd.DataFrame(rows)
out.to_csv(ROOT / "data" / "published_code_reference.tsv", sep="\t", index=False, float_format="%.6g")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(out[["scenario", "region", "coverage", "vaccine_cost", "latent0", "chronic_cases", "hbv_deaths", "yll",
           "yld", "dalys", "disease_cost"]].to_string())
