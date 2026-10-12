"""Compare MYRIAD inputs/outputs with IHME GBD 2023 estimates for the United States.

GBD values were retrieved with the IHME GBD 2023 MCP tools (get_cause_burden,
get_cause_trend, get_risk_attributable_burden, get_risk_exposure,
get_mortality_probability, get_population) and are transcribed here with
their 95% uncertainty intervals. Citation: Institute for Health Metrics and
Evaluation (IHME). Global Burden of Disease Study 2023.

Model values come from the base-case replication (results/annual_base_mean.csv)
and from the paper's inputs (myriad/parameters.py). The comparison year is
2023, the last GBD year and a burn-in year of the model.

    uv run python gbd_comparison.py   # writes results/gbd_comparison.md and a figure
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from myriad import parameters as P

RESULTS = pathlib.Path(__file__).parent / "results"
YEAR = 2023

# ----------------------------------------------------------------------------
# GBD 2023, United States of America (value, lower, upper)
# ----------------------------------------------------------------------------
GBD = {
    # Opioid use disorders (cause 562), all ages, both sexes
    "oud_prevalence_2018": (5_893_492, 5_205_231, 6_611_306),
    "oud_prevalence_2023": (6_990_364, 6_058_906, 7_956_232),
    "oud_deaths_2015": (36_687, 29_807, 45_540),
    "oud_deaths_2018": (47_530, 38_875, 57_962),
    "oud_deaths_2019": (51_796, 41_083, 63_008),
    "oud_deaths_2023": (72_697, 57_750, 89_669),
    # Drug use disorders (cause 561): all drugs, overdose deaths sit here
    "dud_prevalence_2023": (12_796_839, 11_594_263, 14_140_490),
    "dud_deaths_2018": (62_894, 51_426, 76_441),
    "dud_deaths_2023": (95_333, 76_054, 115_274),
    # Injected drug use exposure (rei 138), prevalence proportion, all ages, 2022
    "idu_prev_male_2022": (0.010645, 0.009009, 0.012521),
    "idu_prev_female_2022": (0.004813, 0.004054, 0.005678),
    "pop_male_2022": 165_315_306,  # GBD population, all ages
    "pop_female_2022": 168_405_554,
    # HIV/AIDS (cause 298)
    "hiv_prevalence_2023": (1_799_956, 1_079_296, 2_552_080),
    "hiv_incidence_2023": (58_512, 29_026, 90_645),
    "hiv_deaths_2023": (5_676, 5_351, 5_993),
    "hiv_deaths_drug_use_2023": (376, 310, 474),  # attributable to drug use
    "hiv_deaths_drug_use_paf_2023": (6.6, 5.5, 8.3),  # percent
    "hiv_ylds_drug_use_paf_2023": (6.9, 5.7, 8.5),  # percent
    # Hepatitis C
    "hcv_chronic_prevalence_2023": (4_615_743, 3_766_638, 5_624_341),  # cause 523
    "hcv_acute_incidence_2023": (210_180, 179_849, 244_237),  # cause 403
    "hcv_total_deaths_2023": (37_964, 33_634, 42_777),  # cause 1027
    "hcv_chronic_deaths_2023": (27_512, 23_212, 31_584),  # cause 523
    "hcv_liver_cancer_deaths_2023": (10_446, 9_116, 11_879),  # cause 419
    "hcv_total_deaths_drug_use_2023": (33_769, 28_456, 38_580),
    "hcv_total_deaths_drug_use_paf_2023": (88.9, 78.1, 95.2),  # percent
    # Endocarditis (cause), bacterial skin diseases (cause)
    "endocarditis_incidence_2023": (67_728, 59_676, 77_844),
    "endocarditis_deaths_2023": (9_450, 8_160, 10_666),
    "bacterial_skin_incidence_2023": (11_418_188, 11_125_111, 11_729_716),
}
# GBD 2023 life table q(x) per 1,000, USA 2023 (male, female)
GBD_QX = {
    "20-24": (6.9, 2.6), "25-29": (9.0, 3.5), "30-34": (11.5, 5.1), "35-39": (13.8, 6.8),
    "40-44": (17.1, 9.3), "45-49": (21.4, 12.3), "50-54": (29.5, 17.8), "55-59": (43.8, 26.7),
    "60-64": (64.3, 40.5), "65-69": (89.3, 57.6),
}


def fmt(x: float) -> str:
    return f"{x:,.0f}"


def ui(key: str) -> str:
    v, lo, hi = GBD[key]
    return f"{fmt(v)} ({fmt(lo)}–{fmt(hi)})"


def model_qx() -> dict[str, tuple[float, float]]:
    """5-year probability of death from the model's cause-deleted monthly table (S-B27)."""
    bg = P.BACKGROUND_MORTALITY
    out = {}
    for lo in range(20, 70, 5):
        q = []
        for sex in ["male", "female"]:
            surv = np.prod([(1 - bg.loc[a, sex]) ** 12 for a in range(lo, lo + 5)])
            q.append((1 - surv) * 1000)
        out[f"{lo}-{lo + 4}"] = tuple(q)
    return out


def build() -> tuple[str, pd.DataFrame]:
    a = pd.read_csv(RESULTS / "annual_base_mean.csv", index_col=0)
    m = a.loc[YEAR]
    idu_count = (GBD["idu_prev_male_2022"][0] * GBD["pop_male_2022"]
                 + GBD["idu_prev_female_2022"][0] * GBD["pop_female_2022"])

    rows = [
        # (quantity, model (PWID cohort), GBD (whole US population), note)
        ("People who inject drugs",
         f"{fmt(P.N_PWID_2018)} (2018 input, Bradley et al.); {fmt(m.alive_end_jan)} alive in model, Jan {YEAR}",
         f"Injected drug use exposure, 2022: {GBD['idu_prev_male_2022'][0]*100:.2f}% of males, "
         f"{GBD['idu_prev_female_2022'][0]*100:.2f}% of females (all ages) ≈ {fmt(idu_count)} people",
         "GBD's exposure series is the input to its drug-use risk model; the count is my arithmetic (proportion x GBD population)."),
        ("Opioid use disorder, people",
         f"{fmt(m.state_high_jan)} in the high-risk (OUD) state, Jan {YEAR} "
         f"(input: 45.1% of PWID in 2018 = {fmt(0.451 * P.N_PWID_2018)})",
         f"Prevalence 2023: {ui('oud_prevalence_2023')}; 2018: {ui('oud_prevalence_2018')}",
         "The model covers only the injecting subset; GBD counts all OUD. The 2023 model paper's own calibration denominator was 22.8M people misusing opioids (2015), 35% with OUD."),
        ("Drug use disorders (all drugs), people",
         f"{fmt(m.state_high_jan + m.state_medium_jan)} in high- plus medium-risk states (medium = illicit non-OUD use)",
         f"Prevalence 2023: {ui('dud_prevalence_2023')}",
         "Medium-risk also includes non-opioid injection (e.g. methamphetamine), which GBD counts under amphetamine/cocaine use disorders."),
        ("Overdose deaths per year",
         f"{fmt(m.deaths_overdose)} among PWID in {YEAR} (paper S-B3 estimate: 46,902 = 45.1% of all overdose deaths)",
         f"Drug use disorders deaths 2023: {ui('dud_deaths_2023')}; opioid use disorders: {ui('oud_deaths_2023')}",
         "GBD codes US overdose deaths to drug use disorders. Model PWID deaths / GBD all-drug deaths = "
         f"{m.deaths_overdose / GBD['dud_deaths_2023'][0]:.2f}, close to the paper's 45.1% injection multiplier."),
        ("Overdose deaths per year, 2018",
         f"{fmt(a.loc[2018, 'deaths_overdose'])} (paper target 30,226)",
         f"Drug use disorders deaths 2018: {ui('dud_deaths_2018')}; opioid use disorders: {ui('oud_deaths_2018')}",
         ""),
        ("Annual overdose death probability, people with OUD",
         f"{P.SCENARIOS['base'][2025][0]:.2f}% in 2025 (input, high-risk state); 1.66% in 2023 (S-B3)",
         f"OUD deaths / OUD prevalence 2023 = {fmt(GBD['oud_deaths_2023'][0])} / {fmt(GBD['oud_prevalence_2023'][0])} "
         f"= {GBD['oud_deaths_2023'][0] / GBD['oud_prevalence_2023'][0] * 100:.2f}% (my ratio of two GBD estimates)",
         "The model's rate is for injecting people with OUD, a higher-risk subset of GBD's OUD population."),
        ("People with HIV",
         f"{fmt(m.hiv_prevalent_jan)} among PWID, Jan {YEAR} (input: 5.12% of PWID in 2018)",
         f"HIV prevalence 2023: {ui('hiv_prevalence_2023')}; share of HIV YLDs attributable to drug use: "
         f"{GBD['hiv_ylds_drug_use_paf_2023'][0]}% ({GBD['hiv_ylds_drug_use_paf_2023'][1]}–{GBD['hiv_ylds_drug_use_paf_2023'][2]}%)",
         "The paper's introduction cites 'nearly 100,000 PWID live with HIV', half its own input (5.12% x 3.69M = 189,000). GBD's drug-use PAF applied to its prevalence would give about "
         f"{fmt(GBD['hiv_prevalence_2023'][0] * GBD['hiv_ylds_drug_use_paf_2023'][0] / 100)} (my arithmetic)."),
        ("New HIV infections per year",
         f"{fmt(m.new_hiv)} among PWID in {YEAR} (input: 0.112% annual incidence, CDC IDU + MSM/IDU categories)",
         f"HIV incidence 2023, all transmission: {ui('hiv_incidence_2023')}",
         "GBD publishes no drug-use-attributable incidence. CDC's surveillance estimate for all new infections (~32,000) sits at the low end of GBD's interval."),
        ("HIV deaths per year",
         f"{fmt(m.deaths_hiv)} among PWID in {YEAR}",
         f"HIV deaths 2023: {ui('hiv_deaths_2023')}; attributable to drug use: {ui('hiv_deaths_drug_use_2023')} "
         f"(PAF {GBD['hiv_deaths_drug_use_paf_2023'][0]}%)",
         "The model's HIV deaths among current PWID are about 3x GBD's drug-use-attributable HIV deaths and a fifth of all US HIV deaths."),
        ("People with chronic HCV",
         f"{fmt(m.hcv_prevalent_jan)} among PWID, Jan {YEAR} (input: 43.5% of PWID in 2018)",
         f"Chronic hepatitis C including cirrhosis, prevalence 2023: {ui('hcv_chronic_prevalence_2023')}",
         "CDC's viremic estimate (about 2.4M, 2017-2020) is below GBD's interval; the model's PWID carry a third of GBD's count."),
        ("New HCV infections per year",
         f"{fmt(m.new_hcv)} among PWID in {YEAR} (input: 1.28% annual incidence among susceptible)",
         f"Acute hepatitis C incidence 2023: {ui('hcv_acute_incidence_2023')}",
         "GBD's acute HCV incidence is far above CDC's estimated acute infections (about 67,000 in 2022); the model's 23,000 is current PWID only."),
        ("HCV deaths per year",
         f"{fmt(m.deaths_hcv)} among PWID in {YEAR}, rising to {fmt(a.loc[2034, 'deaths_hcv'])} by 2034",
         f"Total burden related to hepatitis C, deaths 2023: {ui('hcv_total_deaths_2023')} "
         f"(chronic HCV incl. cirrhosis {fmt(GBD['hcv_chronic_deaths_2023'][0])}, liver cancer due to HCV {fmt(GBD['hcv_liver_cancer_deaths_2023'][0])}); "
         f"attributable to drug use: {ui('hcv_total_deaths_drug_use_2023')} (PAF {GBD['hcv_total_deaths_drug_use_paf_2023'][0]}%)",
         "GBD attributes nearly all US HCV deaths to drug use (ever injected), but most occur in older former PWID outside the model's active-IDU cohort: the model's current-PWID HCV deaths are 15% of GBD's total."),
        ("Endocarditis per year",
         f"{fmt(m.new_sbi_endocarditis)} hospitalisations among PWID in {YEAR} (4.8% of severe infections); "
         f"about {fmt(m.new_sbi_endocarditis * 0.058)} deaths at the 5.8% in-hospital fatality input",
         f"Endocarditis incidence 2023: {ui('endocarditis_incidence_2023')}; deaths: {ui('endocarditis_deaths_2023')}",
         "PWID account for about a tenth of GBD's US endocarditis incidence in this comparison, in the range of published injection-associated shares."),
        ("Skin and soft-tissue infections per year",
         f"{fmt(m.new_sbi_ssti)} hospitalisations among PWID in {YEAR}",
         f"Bacterial skin diseases (cellulitis, pyoderma) incidence 2023: {ui('bacterial_skin_incidence_2023')} episodes, all severities",
         "Not comparable: GBD counts all episodes, the model only hospitalisations. Osteomyelitis and CNS abscess have no GBD cause."),
    ]
    df = pd.DataFrame(rows, columns=["Quantity", "MYRIAD (PWID cohort)", "GBD 2023 (United States)", "Note"])

    lines = ["| Quantity | MYRIAD replication (PWID cohort) | GBD 2023, United States (95% UI) | Note |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    table1 = "\n".join(lines)

    qx = model_qx()
    lines = ["| Age | Model background q(x), male | GBD all-cause q(x), male | Model, female | GBD, female |", "|---|---:|---:|---:|---:|"]
    for band, (gm, gf) in GBD_QX.items():
        mm, mf = qx[band]
        lines.append(f"| {band} | {mm:.1f} | {gm:.1f} | {mf:.1f} | {gf:.1f} |")
    table2 = "\n".join(lines)

    crude = {y: a.loc[y, "deaths_total"] / a.loc[y, "person_months"] * 12 * 1000 for y in [2018, 2023, 2025, 2034]}
    other = {y: a.loc[y, "deaths_other"] / a.loc[y, "person_months"] * 12 * 1000 for y in [2018, 2023, 2025, 2034]}
    lines = ["| Year | Model cohort all-cause deaths per 1,000 person-years | of which other causes | Approximate mean age (40 at entry) |", "|---|---:|---:|---:|"]
    for y in crude:
        lines.append(f"| {y} | {crude[y]:.1f} | {other[y]:.1f} | {40 + (y - 2018):.0f} |")
    table3 = "\n".join(lines)

    text = (f"## Model vs GBD 2023 (United States), comparison year {YEAR}\n\n{table1}\n\n"
            "### Background mortality: model input vs GBD life table\n\n"
            "Five-year probabilities of death per 1,000 (model: cumulated from the supplement's monthly, "
            "cause-deleted table; GBD: 2023 abridged life table, all causes).\n\n" + table2 +
            "\n\n### All-cause mortality in the simulated cohort\n\n" + table3 + "\n")
    return text, df


def figure(df: pd.DataFrame, out: pathlib.Path) -> None:
    a = pd.read_csv(RESULTS / "annual_base_mean.csv", index_col=0)
    m = a.loc[YEAR]
    pairs = [
        ("Overdose deaths", m.deaths_overdose, GBD["dud_deaths_2023"], "drug use disorders"),
        ("People with OUD", m.state_high_jan, GBD["oud_prevalence_2023"], "OUD prevalence"),
        ("People with HIV", m.hiv_prevalent_jan, GBD["hiv_prevalence_2023"], "HIV prevalence"),
        ("HIV deaths", m.deaths_hiv, GBD["hiv_deaths_2023"], "HIV deaths"),
        ("HIV deaths, drug-use attrib.", m.deaths_hiv, GBD["hiv_deaths_drug_use_2023"], "attributable to drug use"),
        ("People with chronic HCV", m.hcv_prevalent_jan, GBD["hcv_chronic_prevalence_2023"], "chronic HCV prevalence"),
        ("New HCV infections", m.new_hcv, GBD["hcv_acute_incidence_2023"], "acute HCV incidence"),
        ("HCV deaths", m.deaths_hcv, GBD["hcv_total_deaths_2023"], "total HCV-related deaths"),
        ("HCV deaths, drug-use attrib.", m.deaths_hcv, GBD["hcv_total_deaths_drug_use_2023"], "attributable to drug use"),
        ("Endocarditis cases", m.new_sbi_endocarditis, GBD["endocarditis_incidence_2023"], "endocarditis incidence"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(pairs))
    model = [p[1] for p in pairs]
    gbd = [p[2][0] for p in pairs]
    err = np.array([[p[2][0] - p[2][1], p[2][2] - p[2][0]] for p in pairs]).T
    ax.barh(y + 0.2, gbd, height=0.38, color="#9e9e9e", label="GBD 2023, whole US population (95% UI)")
    ax.errorbar(gbd, y + 0.2, xerr=err, fmt="none", ecolor="#424242", capsize=3, lw=1)
    ax.barh(y - 0.2, model, height=0.38, color="#d62728", label=f"MYRIAD replication, PWID cohort, {YEAR}")
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("People or events per year (log scale)")
    ax.set_title("MYRIAD PWID cohort vs GBD 2023 estimates, whole United States")
    ax.grid(alpha=0.3, axis="x", which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    text, df = build()
    (RESULTS / "gbd_comparison.md").write_text(text)
    df.to_csv(RESULTS / "gbd_comparison.csv", index=False)
    figure(df, RESULTS / "fig_gbd_comparison.png")
    print(text)


if __name__ == "__main__":
    main()
