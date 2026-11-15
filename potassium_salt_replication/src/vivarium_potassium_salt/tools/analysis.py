"""Turn the simulation's summed rates into burden averted.

The observers record, for each stratum of simulants, the sum of every
outcome's rate.  The baseline and a scenario share their simulants, so

    potential impact fraction = 1 - (scenario sum) / (baseline sum),

and the burden averted is that fraction of the stratum's GBD burden.  The
kidney-disease quantities come from the same table, which is also split by
kidney disease stage.  The functions here return tables in the same layout as
``analytic.stratum_table`` and ``analytic.summarise``, so the simulation and
the closed form can be laid side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vivarium_potassium_salt import analytic
from vivarium_potassium_salt.components import burden, intervention, kidney, observers
from vivarium_potassium_salt.constants import metadata

_STRATUM = ["sex", "age_group"]


def _labels_to_names(table: pd.DataFrame) -> pd.DataFrame:
    """'25_to_29' -> '25 to 29', so strata match the input tables."""
    return table.assign(age_group=table["age_group"].astype(str).str.replace("_", " "))


def simulation_impact(burden_results: pd.DataFrame, policy_results: pd.DataFrame,
                      inputs: analytic.LocationInputs, draw: int = 0,
                      stage_deaths: str = "among_ckd") -> dict[str, dict]:
    """Per-scenario results from one location's simulations at one draw.

    Returns the same structure as ``analytic.evaluate``, with a single draw:
    ``pif`` is [cause, stratum, 1] and the other entries are [stratum, 1].
    """
    burden_results = _labels_to_names(burden_results[burden_results["draw"] == draw])
    policy_results = _labels_to_names(policy_results[policy_results["draw"] == draw])
    order = pd.MultiIndex.from_frame(inputs.strata[_STRATUM])
    measures = list(inputs.counts)
    rate_columns = {
        (measure, cause): burden.rate_pipeline(cause, measure)
        for measure in measures for cause in metadata.CAUSES_BY_MEASURE[measure]
    }
    cvd = [c in metadata.CVD_CAUSES for c in metadata.CAUSES]
    death_columns = [rate_columns[("deaths", c)] for c in metadata.CVD_CAUSES]
    parameters = inputs.parameters
    stage_hazard = {
        stage: analytic._by_draw(parameters.ckd_stage_hazard_ratio[stage])[..., draw:draw + 1]
        for stage in metadata.CKD_STAGES
    }
    hazard_by_age = inputs.ckd_hazard_ratio[:, draw]
    cvd_deaths = inputs.counts["deaths"][cvd][:, :, draw]

    def sums(scenario: str, by_stage: bool = False) -> pd.DataFrame:
        table = burden_results[burden_results["scenario"] == scenario]
        keys = _STRATUM + ([kidney.STAGE] if by_stage else [])
        return table.groupby(keys, observed=True).sum(numeric_only=True)

    baseline, baseline_by_stage = sums("baseline"), sums("baseline", by_stage=True)

    def stage_slice(table: pd.DataFrame, stages: list[str], columns) -> np.ndarray:
        selected = table[table.index.get_level_values(kidney.STAGE).isin(stages)]
        return selected.groupby(_STRATUM).sum().reindex(order)[columns].fillna(0.0).to_numpy()

    # How the stratum's cardiovascular deaths divide among kidney disease
    # stages, from the simulated prevalence of each stage.
    people = baseline[observers.PERSON_COUNT].reindex(order).to_numpy()
    simulated_prevalence = {
        stage: stage_slice(baseline_by_stage, [stage], [observers.PERSON_COUNT]) / people[:, None]
        for stage in metadata.CKD_STAGES
    }
    ckd_total = sum(simulated_prevalence.values())[:, 0]
    ckd_share = ckd_total * hazard_by_age / (ckd_total * hazard_by_age + 1 - ckd_total)
    with np.errstate(invalid="ignore", divide="ignore"):
        stage_shares = analytic.stage_shares_of_cvd_deaths(
            simulated_prevalence, stage_hazard, ckd_share[:, None], stage_deaths)
    stage_shares = {stage: np.nan_to_num(share) for stage, share in stage_shares.items()}

    results = {}
    for scenario in metadata.SCENARIOS:
        if scenario.name not in set(burden_results["scenario"]):
            continue
        scenario_sums, by_stage = sums(scenario.name), sums(scenario.name, by_stage=True)
        pif = np.zeros((len(metadata.CAUSES), len(order), 1))
        for (measure, cause), column in rate_columns.items():
            if measure != "deaths":
                continue  # one fraction per cause; every measure shares it
            before = baseline[column].reindex(order).to_numpy()
            after = scenario_sums[column].reindex(order).to_numpy()
            # A sampled death rate can be zero; nothing is averted where nobody dies.
            ratio = np.divide(after, before, out=np.ones_like(before), where=before > 0)
            pif[metadata.CAUSES.index(cause), :, 0] = 1 - ratio

        # Cardiovascular deaths averted among people with stage 3-5 disease.
        with_ckd = list(metadata.CKD_STAGES)
        base_ckd = stage_slice(baseline_by_stage, with_ckd, death_columns)
        base_rest = stage_slice(baseline_by_stage, [metadata.NO_ADVANCED_CKD], death_columns)
        scen_ckd = stage_slice(by_stage, with_ckd, death_columns)
        hazard = hazard_by_age[:, None]
        everyone = hazard * base_ckd + base_rest
        share_prevented = np.divide(hazard * (base_ckd - scen_ckd), everyone,
                                    out=np.zeros_like(everyone), where=everyone > 0)
        prevented = (cvd_deaths.T * share_prevented).sum(axis=1)

        # Extra deaths from hyperkalaemia, stage by stage.
        extra = np.zeros(len(order))
        if scenario.hyperkalaemia_modelled:
            risk = kidney.HYPERKALAEMIA_RELATIVE_RISK
            for stage in metadata.CKD_STAGES:
                before = stage_slice(baseline_by_stage, [stage], [risk])[:, 0]
                after = stage_slice(by_stage, [stage], [risk])[:, 0]
                change = np.divide(after, before, out=np.ones_like(after), where=before > 0) - 1
                extra += cvd_deaths.sum(axis=0) * stage_shares[stage][:, 0] * change

        policy = policy_results[policy_results["scenario"] == scenario.name]
        policy = policy.groupby(_STRATUM).sum(numeric_only=True).reindex(order)
        people = policy[observers.PERSON_COUNT].to_numpy()
        results[scenario.name] = {
            "pif": pif,
            "users": (policy[intervention.USES_SUBSTITUTE].to_numpy() / people
                      * inputs.strata["population"].to_numpy())[:, None],
            "sbp_reduction": (policy[intervention.SBP_REDUCTION].to_numpy() / people)[:, None],
            "prevented_cvd_deaths_in_ckd": prevented[:, None],
            "hyperkalaemia_deaths": extra[:, None],
        }
    return results


def one_draw(inputs: analytic.LocationInputs, draw: int) -> analytic.LocationInputs:
    """The inputs with only one draw kept, as column 0."""
    keep = slice(draw, draw + 1)
    return analytic.LocationInputs(
        location=inputs.location, strata=inputs.strata,
        sodium=inputs.sodium[:, keep], sbp_mean=inputs.sbp_mean[:, keep],
        hypertension=inputs.hypertension[:, keep], diagnosed=inputs.diagnosed[:, keep],
        treated=inputs.treated[:, keep], relative_risk=inputs.relative_risk[:, :, keep],
        counts={m: c[:, :, keep] for m, c in inputs.counts.items()},
        ckd_prevalence=inputs.ckd_prevalence, ckd_hazard_ratio=inputs.ckd_hazard_ratio[:, keep],
        gdp_per_capita=inputs.gdp_per_capita, high_income=inputs.high_income,
        reported_discretionary_share=inputs.reported_discretionary_share,
        parameters=inputs.parameters,
    )


def summarise_simulation(burden_results: pd.DataFrame, policy_results: pd.DataFrame,
                         stage_deaths: str = "among_ckd") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Burden averted for every location and draw in a set of simulation results.

    Returns ``(strata, totals)``: the first has a row per location, draw,
    scenario, measure, cause, sex, and age band; the second a row per location,
    draw, scenario, quantity, and sex, as ``analytic.summarise`` does.
    """
    strata, totals = [], []
    for location, location_burden in burden_results.groupby("location"):
        location_policy = policy_results[policy_results["location"] == location]
        draws = sorted(location_burden["draw"].unique())
        inputs = analytic.load_inputs(location, n_sampled=max(draws))
        for draw in draws:
            results = simulation_impact(location_burden, location_policy, inputs, draw, stage_deaths)
            single = one_draw(inputs, draw)
            strata.append(analytic.stratum_table(single, results).assign(draw=draw))
            summary = analytic.summarise(single, results)["draw_0"].rename("value").reset_index()
            totals.append(summary.assign(draw=draw))
    return pd.concat(strata, ignore_index=True), pd.concat(totals, ignore_index=True)
