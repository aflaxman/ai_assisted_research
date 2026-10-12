"""vivarium components implementing the MYRIAD state machines.

Each condition is one component that owns its own state columns and updates
them once per monthly time step. Components contribute cause-specific monthly
death probabilities to a shared value pipeline; ``Mortality`` combines them
(rates summed, cause chosen in proportion to its rate, as in the paper) and
records the cause of death. ``OutcomeObserver`` tallies events, person-time
and discounted costs by calendar year.

Time-step order (all at the default priority, in registration order):
Population (ageing) -> DrugUse -> HIV -> HCV -> SevereBacterialInfection ->
Mortality (priority 8). Observations are collected after the step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

from . import parameters as P

CAUSES = ["overdose", "sbi", "hiv", "hcv"]
DEATH_PIPELINE = "cause_specific_monthly_death_probability"


def _to_rate(p: pd.Series | np.ndarray) -> np.ndarray:
    return -np.log1p(-np.asarray(p, dtype=float))


def _transition_matrix(
    states: list[str], transitions: dict[tuple[str, str], float]
) -> pd.DataFrame:
    """Row-stochastic monthly transition matrix with the stay-probability on the diagonal."""
    m = pd.DataFrame(0.0, index=states, columns=states)
    for (a, b), p in transitions.items():
        m.loc[a, b] += p
    for s in states:
        m.loc[s, s] = 1.0 - m.loc[s].sum()
        assert m.loc[s, s] >= 0, f"transitions out of {s} exceed 1"
    return m


def _infection_risk_is_idu(drug_state: pd.Series) -> pd.Series:
    """Who gets the PWID (vs non-IDU) infection inputs; see parameters.py."""
    if P.REMISSION_KEEPS_IDU_INFECTION_RISK:
        return pd.Series(True, index=drug_state.index)
    return drug_state.isin(P.ACTIVE_IDU_STATES)


# --------------------------------------------------------------------------


class Calendar(Component):
    """Tracks the model month and calendar year (one time step = one month)."""

    def __init__(self, start_year: int = P.COHORT_START_YEAR):
        super().__init__()
        self.start_year = start_year
        self.step = -1  # becomes 0 on the first time step

    @property
    def time_step_prepare_priority(self) -> int:
        return 0

    @property
    def year(self) -> int:
        return self.start_year + max(self.step, 0) // 12

    def on_time_step_prepare(self, event: Event) -> None:
        self.step += 1


class Population(Component):
    """Initial cohort: age ~ N(40, 10) truncated to [18, 65], 29.7% female."""

    def __init__(self, calendar: Calendar):
        super().__init__()
        self.calendar = calendar

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("population")
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants, columns=["age", "sex"]
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        a = (P.AGE_MIN - P.AGE_MEAN) / P.AGE_SD
        b = (P.AGE_MAX - P.AGE_MEAN) / P.AGE_SD
        age = stats.truncnorm.ppf(
            self.randomness.get_draw(pop_data.index, "age"), a, b, loc=P.AGE_MEAN, scale=P.AGE_SD
        )
        female = self.randomness.get_draw(pop_data.index, "sex") < P.PROPORTION_FEMALE
        self.population_view.initialize(
            pd.DataFrame(
                {"age": age, "sex": np.where(female, "female", "male")}, index=pop_data.index
            )
        )

    def on_time_step(self, event: Event) -> None:
        alive = self.population_view.get(event.index, ["alive"]).query("alive").index
        self.population_view.update("age", lambda age: age.loc[alive] + 1.0 / 12.0)


class DrugUse(Component):
    """Drug-use behaviour states (low / medium / high risk, remission) and overdose death."""

    def __init__(self, calendar: Calendar, scenario: str = "base"):
        super().__init__()
        self.calendar = calendar
        self.scenario = scenario
        self.matrix = _transition_matrix(P.DRUG_STATES, P.DRUG_TRANSITIONS)
        self.overdose = P.overdose_schedule(scenario)

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("drug_use")
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants, columns=["drug_state"]
        )
        builder.value.register_value_modifier(DEATH_PIPELINE, modifier=self.add_overdose_death)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        states = list(P.INITIAL_DRUG_STATE)
        p = np.array([P.INITIAL_DRUG_STATE[s] for s in states])
        self.population_view.initialize(
            pd.Series(
                self.randomness.choice(pop_data.index, states, p=p / p.sum(), additional_key="init"),
                index=pop_data.index, name="drug_state",
            )
        )

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index, ["alive", "drug_state"]).query("alive")
        probs = self.matrix.loc[pop["drug_state"]].to_numpy()
        new = self.randomness.choice(pop.index, P.DRUG_STATES, p=probs, additional_key="transition")
        self.population_view.update("drug_state", lambda _: pd.Series(new, index=pop.index))

    def add_overdose_death(self, index: pd.Index, value: pd.DataFrame) -> pd.DataFrame:
        state = self.population_view.get(index, "drug_state")
        sched = P.schedule_lookup(self.overdose, self.calendar.year)
        value.loc[index, "overdose"] += state.map(sched).to_numpy()
        return value


class HIV(Component):
    """HIV infection, diagnosis, ART, CD4 progression and HIV death."""

    COLUMNS = ["hiv", "hiv_diagnosed", "on_art", "cd4", "new_hiv"]

    def __init__(self, calendar: Calendar):
        super().__init__()
        self.calendar = calendar
        # CD4 transition matrices stacked as [regimen, from_cd4, to_cd4]
        self.regimens = ["off_art", "on_art_idu", "on_art_non_idu"]
        self.cd4_matrix = np.stack(
            [_transition_matrix(P.CD4_STATES, P.CD4_TRANSITIONS[k]).to_numpy() for k in self.regimens]
        )
        # ART probabilities as [group (idu, non_idu), cd4] arrays
        self.art = {
            which: np.array(
                [[P.HIV_ART[g][which][c] for c in P.CD4_STATES] for g in ("idu", "non_idu")]
            )
            for which in ("start_at_diagnosis", "start_monthly", "stop_monthly")
        }
        self.death = np.array(
            [[P.HIV_DEATH_P[a][c] for c in P.CD4_STATES] for a in ("off_art", "on_art")]
        )

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("hiv")
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants, columns=self.COLUMNS
        )
        builder.value.register_value_modifier(DEATH_PIPELINE, modifier=self.add_hiv_death)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        idx = pop_data.index
        hiv = self.randomness.get_draw(idx, "hiv") < P.HIV_PREVALENCE_2018
        diagnosed = hiv & (self.randomness.get_draw(idx, "dx") < P.HIV_INITIAL_DIAGNOSED)
        on_art = diagnosed & (
            self.randomness.get_draw(idx, "art") < P.HIV_INITIAL_ON_ART_GIVEN_DIAGNOSED
        )
        p = np.where(
            on_art.to_numpy()[:, None], P.HIV_INITIAL_CD4["on_art"], P.HIV_INITIAL_CD4["off_art"]
        )
        cd4 = self.randomness.choice(idx, P.CD4_STATES, p=p, additional_key="cd4")
        cd4 = pd.Series(np.where(hiv, cd4, "none"), index=idx)
        self.population_view.initialize(
            pd.DataFrame(
                {"hiv": hiv, "hiv_diagnosed": diagnosed, "on_art": on_art, "cd4": cd4,
                 "new_hiv": False},
                index=idx,
            )
        )

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(
            event.index, ["alive", "drug_state"] + self.COLUMNS
        ).query("alive")
        year = self.calendar.year
        idu = _infection_risk_is_idu(pop["drug_state"])
        out = pop[self.COLUMNS].copy()
        out["new_hiv"] = False

        # --- new infections among HIV-negative -------------------------------
        neg = pop.index[~pop["hiv"]]
        p_inc = np.where(
            idu.loc[neg],
            P.annual_to_monthly(P.schedule_lookup(P.HIV_INCIDENCE_ANNUAL_IDU, year)),
            P.annual_to_monthly(P.schedule_lookup(P.HIV_INCIDENCE_ANNUAL_NON_IDU, year)),
        )
        infected = self.randomness.filter_for_probability(neg, p_inc, additional_key="incidence")
        out.loc[infected, "hiv"] = True
        out.loc[infected, "new_hiv"] = True
        out.loc[infected, "cd4"] = self.randomness.choice(
            infected, P.CD4_STATES, p=list(P.HIV_INITIAL_CD4["new_infection"]), additional_key="cd4_new"
        )

        # --- diagnosis of prevalent undiagnosed (not those infected this month) ---
        undx = pop.index[pop["hiv"] & ~pop["hiv_diagnosed"]]
        p_test = np.where(
            idu.loc[undx],
            P.annual_to_monthly(P.HIV_TESTING_ANNUAL["idu"]),
            P.annual_to_monthly(P.HIV_TESTING_ANNUAL["non_idu"]),
        )
        newly_dx = self.randomness.filter_for_probability(undx, p_test, additional_key="test")
        out.loc[newly_dx, "hiv_diagnosed"] = True

        # --- ART start / stop ---------------------------------------------------
        cd4_code = pd.Series(pd.Categorical(pop["cd4"], categories=P.CD4_STATES).codes, index=pop.index)
        grp = pd.Series(np.where(idu, 0, 1), index=pop.index)  # 0 = idu, 1 = non_idu

        def art_p(which: str, idx: pd.Index) -> np.ndarray:
            return self.art[which][grp.loc[idx].to_numpy(), cd4_code.loc[idx].to_numpy()]

        # newly diagnosed: start within 30 days of diagnosis
        started = self.randomness.filter_for_probability(
            newly_dx, art_p("start_at_diagnosis", newly_dx), additional_key="art_dx"
        )
        out.loc[started, "on_art"] = True
        # diagnosed earlier, off ART: monthly start probability
        off = pop.index[pop["hiv"] & pop["hiv_diagnosed"] & ~pop["on_art"]]
        started = self.randomness.filter_for_probability(
            off, art_p("start_monthly", off), additional_key="art_start"
        )
        out.loc[started, "on_art"] = True
        on = pop.index[pop["on_art"]]
        stopped = self.randomness.filter_for_probability(
            on, art_p("stop_monthly", on), additional_key="art_stop"
        )
        out.loc[stopped, "on_art"] = False

        # --- CD4 progression for prevalent infections (ART status at start of month) ---
        prev = pop.index[pop["hiv"]]
        if len(prev):
            regimen = np.where(pop.loc[prev, "on_art"], np.where(idu.loc[prev], 1, 2), 0)
            probs = self.cd4_matrix[regimen, cd4_code.loc[prev].to_numpy(), :]
            out.loc[prev, "cd4"] = self.randomness.choice(
                prev, P.CD4_STATES, p=probs, additional_key="cd4_progress"
            )

        self.population_view.update(self.COLUMNS, lambda _: out)

    def add_hiv_death(self, index: pd.Index, value: pd.DataFrame) -> pd.DataFrame:
        pop = self.population_view.get(index, ["hiv", "on_art", "cd4"])
        pos = pop.index[pop["hiv"]]
        if len(pos):
            cd4_code = pd.Categorical(pop.loc[pos, "cd4"], categories=P.CD4_STATES).codes
            value.loc[pos, "hiv"] += self.death[pop.loc[pos, "on_art"].to_numpy().astype(int), cd4_code]
        return value


class HCV(Component):
    """HCV infection, spontaneous clearance, fibrosis progression and liver death.

    ``fibrosis`` is kept after clearance (progression from F4 continues at the
    "without HCV" rates); people never infected are F0.
    """

    COLUMNS = ["hcv", "fibrosis", "hcv_months", "dc_months", "new_hcv"]

    def __init__(self, calendar: Calendar):
        super().__init__()
        self.calendar = calendar
        self.p_prog = {k: P.annual_to_monthly(v) for k, v in P.HCV_PROGRESSION_ANNUAL.items()}
        self.p_f4 = {
            g: {k: P.annual_to_monthly(v) for k, v in d.items()}
            for g, d in P.HCV_F4_PROGRESSION_ANNUAL.items()
        }
        self.p_death = {k: P.annual_to_monthly(v) for k, v in P.HCV_DEATH_ANNUAL.items()}

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("hcv")
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants, columns=self.COLUMNS
        )
        builder.value.register_value_modifier(DEATH_PIPELINE, modifier=self.add_hcv_death)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        idx = pop_data.index
        hcv = self.randomness.get_draw(idx, "hcv") < P.HCV_PREVALENCE_2018
        fib = self.randomness.choice(
            idx, P.FIBROSIS_STATES, p=[P.HCV_INITIAL_FIBROSIS[s] for s in P.FIBROSIS_STATES],
            additional_key="fibrosis",
        )
        self.population_view.initialize(
            pd.DataFrame(
                {
                    "hcv": hcv,
                    "fibrosis": np.where(hcv, fib, "F0"),
                    # prevalent infections are chronic: past the acute window
                    "hcv_months": np.where(hcv, P.HCV_ACUTE_MONTHS + 1, 0),
                    "dc_months": np.where(fib == "DC", 13, 0),
                    "new_hcv": False,
                },
                index=idx,
            )
        )

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(
            event.index, ["alive", "drug_state", "hiv", "on_art"] + self.COLUMNS
        ).query("alive")
        year = self.calendar.year
        idu = _infection_risk_is_idu(pop["drug_state"])
        out = pop[self.COLUMNS].copy()
        out["new_hcv"] = False

        # --- new infections / reinfections among susceptible -------------------
        sus = pop.index[~pop["hcv"]]
        p_inc = np.where(
            idu.loc[sus],
            P.annual_to_monthly(P.schedule_lookup(P.HCV_INCIDENCE_ANNUAL_IDU, year)),
            P.annual_to_monthly(P.HCV_INCIDENCE_ANNUAL_NON_IDU),
        )
        infected = self.randomness.filter_for_probability(sus, p_inc, additional_key="incidence")
        out.loc[infected, ["hcv", "new_hcv"]] = True
        out.loc[infected, "hcv_months"] = 0

        # --- spontaneous clearance during the acute phase -----------------------
        acute = pop.index[pop["hcv"] & (pop["hcv_months"] < P.HCV_ACUTE_MONTHS)]
        cleared = self.randomness.filter_for_probability(
            acute, P.HCV_SPONTANEOUS_CLEARANCE_MONTHLY, additional_key="clearance"
        )
        out.loc[cleared, "hcv"] = False
        prev = pop.index[pop["hcv"]]
        out.loc[prev, "hcv_months"] = pop.loc[prev, "hcv_months"] + 1

        # --- fibrosis progression (state at start of month) ---------------------
        fib = pop["fibrosis"]
        rr = np.where(
            pop["hiv"],
            np.where(pop["on_art"], P.HCV_FIBROSIS_RR_HIV["on_art"], P.HCV_FIBROSIS_RR_HIV["off_art"]),
            1.0,
        )
        rr = pd.Series(rr, index=pop.index)
        for a, b in [("F0", "F1"), ("F1", "F2"), ("F2", "F3"), ("F3", "F4")]:
            at_risk = pop.index[pop["hcv"] & (fib == a)]  # only active infection progresses
            p = np.minimum(1.0, self.p_prog[(a, b)] * rr.loc[at_risk].to_numpy())
            moved = self.randomness.filter_for_probability(at_risk, p, additional_key=f"{a}{b}")
            out.loc[moved, "fibrosis"] = b
        f4 = pop.index[fib == "F4"]
        if len(f4):
            with_hcv = pop.loc[f4, "hcv"].to_numpy()
            p_dc = np.where(with_hcv, self.p_f4["with_hcv"][("F4", "DC")], self.p_f4["without_hcv"][("F4", "DC")])
            p_hcc = np.where(with_hcv, self.p_f4["with_hcv"][("F4", "HCC")], self.p_f4["without_hcv"][("F4", "HCC")])
            p_dc = np.minimum(1.0, p_dc * rr.loc[f4].to_numpy())
            p_hcc = np.minimum(1.0, p_hcc * rr.loc[f4].to_numpy())
            probs = np.column_stack([1 - p_dc - p_hcc, p_dc, p_hcc])
            dest = self.randomness.choice(f4, ["F4", "DC", "HCC"], p=probs, additional_key="f4")
            out.loc[f4, "fibrosis"] = dest
            out.loc[f4[dest == "DC"], "dc_months"] = 0
        dc = pop.index[fib == "DC"]
        if len(dc):
            moved = self.randomness.filter_for_probability(dc, self.p_prog[("DC", "HCC")], additional_key="dchcc")
            out.loc[moved, "fibrosis"] = "HCC"
            out.loc[dc, "dc_months"] = pop.loc[dc, "dc_months"] + 1

        self.population_view.update(self.COLUMNS, lambda _: out)

    def add_hcv_death(self, index: pd.Index, value: pd.DataFrame) -> pd.DataFrame:
        pop = self.population_view.get(index, ["fibrosis", "dc_months"])
        dc = pop.index[pop["fibrosis"] == "DC"]
        value.loc[dc, "hcv"] += np.where(
            pop.loc[dc, "dc_months"] < 12, self.p_death["DC_first_year"], self.p_death["DC_after"]
        )
        value.loc[pop.index[pop["fibrosis"] == "HCC"], "hcv"] += self.p_death["HCC"]
        return value


class SevereBacterialInfection(Component):
    """Hospitalisation for SSTI, endocarditis, osteomyelitis or CNS infection."""

    COLUMNS = ["sbi_active", "new_sbi"]

    def __init__(self, calendar: Calendar):
        super().__init__()
        self.calendar = calendar

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("sbi")
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants, columns=self.COLUMNS
        )
        builder.value.register_value_modifier(DEATH_PIPELINE, modifier=self.add_sbi_death)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.initialize(
            pd.DataFrame({"sbi_active": False, "new_sbi": "none"}, index=pop_data.index)
        )

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(
            event.index, ["alive", "drug_state", "hiv", "cd4"] + self.COLUMNS
        ).query("alive")
        year = self.calendar.year
        idu = _infection_risk_is_idu(pop["drug_state"])
        out = pop[self.COLUMNS].copy()
        out["new_sbi"] = "none"

        # --- cure of previously uncured infections -------------------------------
        oud = pop["drug_state"] == "high"
        p_cure = np.where(oud, 1 - P.SBI_SELF_DISCHARGE_P["oud"], 1 - P.SBI_SELF_DISCHARGE_P["non_oud"])
        p_cure = pd.Series(p_cure, index=pop.index)
        active = pop.index[pop["sbi_active"]]
        cured = self.randomness.filter_for_probability(active, p_cure.loc[active], additional_key="cure")
        out.loc[cured, "sbi_active"] = False

        # --- new hospitalisations among those without an active infection --------
        at_risk = pop.index[~pop["sbi_active"]]
        p_base = np.where(
            idu.loc[at_risk],
            P.annual_to_monthly(P.sbi_annual_probability_idu(year)),
            P.annual_to_monthly(P.SBI_ANNUAL_PROBABILITY_NON_IDU),
        )
        mult = pop.loc[at_risk, "cd4"].map(P.SBI_HIV_MULTIPLIER).fillna(1.0).to_numpy()
        p = 1 - np.exp(-_to_rate(p_base) * mult)
        new = self.randomness.filter_for_probability(at_risk, p, additional_key="incidence")
        kind = self.randomness.choice(new, P.SBI_TYPES, p=P.SBI_TYPE_P, additional_key="type")
        out.loc[new, "new_sbi"] = kind
        # survivors of the hospitalisation are cured unless self-directed discharge
        uncured = self.randomness.filter_for_probability(
            new, 1 - p_cure.loc[new], additional_key="discharge"
        )
        out.loc[uncured, "sbi_active"] = True

        self.population_view.update(self.COLUMNS, lambda _: out)

    def add_sbi_death(self, index: pd.Index, value: pd.DataFrame) -> pd.DataFrame:
        kind = self.population_view.get(index, "new_sbi")
        value.loc[index, "sbi"] += kind.map(P.SBI_DEATH_P).fillna(0.0).to_numpy()
        return value


class Mortality(Component):
    """Background + cause-specific mortality with proportional cause attribution."""

    COLUMNS = ["alive", "cause_of_death", "death_step"]

    def __init__(self, calendar: Calendar):
        super().__init__()
        self.calendar = calendar

    @property
    def time_step_priority(self) -> int:
        return 8  # after every disease component has updated its state

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream("mortality")
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants, columns=self.COLUMNS
        )
        self.cause_specific = builder.value.register_value_producer(
            DEATH_PIPELINE, source=lambda index: pd.DataFrame(0.0, index=index, columns=CAUSES)
        )
        bg = P.BACKGROUND_MORTALITY
        self.bg_male = bg["male"].to_numpy()
        self.bg_female = bg["female"].to_numpy()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.initialize(
            pd.DataFrame(
                {"alive": True, "cause_of_death": "none", "death_step": -1}, index=pop_data.index
            )
        )

    def background(self, pop: pd.DataFrame) -> np.ndarray:
        age = np.clip(np.floor(pop["age"].to_numpy()).astype(int), 0, len(self.bg_male) - 1)
        return np.where(pop["sex"] == "female", self.bg_female[age], self.bg_male[age])

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index, ["alive", "age", "sex"]).query("alive")
        rates = pd.DataFrame(index=pop.index)
        rates["other"] = _to_rate(self.background(pop))
        cs = self.cause_specific(pop.index)
        for c in CAUSES:
            rates[c] = _to_rate(cs[c].to_numpy())
        total = rates.sum(axis=1)
        p_death = 1 - np.exp(-total)
        dying = self.randomness.filter_for_probability(pop.index, p_death, additional_key="death")
        if len(dying):
            share = rates.loc[dying].div(total.loc[dying], axis=0)
            cause = self.randomness.choice(dying, list(rates.columns), p=share.to_numpy(), additional_key="cause")
            update = pd.DataFrame(
                {"alive": False, "cause_of_death": cause, "death_step": self.calendar.step},
                index=dying,
            )
            self.population_view.update(self.COLUMNS, lambda _: update)


class OutcomeObserver(Component):
    """Counts events, deaths by cause, person-years and costs per calendar year."""

    def __init__(self, calendar: Calendar):
        super().__init__()
        self.calendar = calendar
        self.records: list[dict[str, Any]] = []

    def on_collect_metrics(self, event: Event) -> None:
        pop = self.population_view.get(
            event.index,
            ["alive", "death_step", "cause_of_death", "drug_state", "hiv", "cd4", "on_art",
             "hcv", "fibrosis", "new_sbi", "new_hiv", "new_hcv"],
        )
        step, year = self.calendar.step, self.calendar.year
        died = pop["death_step"] == step
        in_month = pop["alive"] | died  # alive at the start of this month
        m = pop[in_month]
        rec: dict[str, Any] = {"year": year, "month": step % 12, "person_months": int(in_month.sum()),
                               "alive_end": int(pop["alive"].sum())}
        for cause in CAUSES + ["other"]:
            rec[f"deaths_{cause}"] = int((died & (pop["cause_of_death"] == cause)).sum())
        rec["new_sbi"] = int((m["new_sbi"] != "none").sum())
        for t in P.SBI_TYPES:
            rec[f"new_sbi_{t}"] = int((m["new_sbi"] == t).sum())
        rec["new_hiv"] = int(m["new_hiv"].sum())
        rec["new_hcv"] = int(m["new_hcv"].sum())
        for s in P.DRUG_STATES:
            rec[f"state_{s}"] = int((m["drug_state"] == s).sum())
        rec["hiv_prevalent"] = int(m["hiv"].sum())
        rec["hiv_on_art"] = int(m["on_art"].sum())
        rec["hiv_cd4_low"] = int((m["cd4"] == "low").sum())
        rec["hcv_prevalent"] = int(m["hcv"].sum())
        # costs (undiscounted, 2025 USD) for people alive at the start of the month
        drug = m["drug_state"].map(P.COST_DRUG_USE_MONTHLY).sum()
        drug += P.COST_IDU_MONTHLY * m["drug_state"].isin(P.ACTIVE_IDU_STATES).sum()
        drug += P.COST_OVERDOSE_DEATH * rec["deaths_overdose"]
        hiv = m.loc[m["hiv"], "cd4"].map(P.COST_HIV_MONTHLY_BY_CD4).sum() + P.COST_ART_MONTHLY * m["on_art"].sum()
        hcv = m.loc[m["hcv"], "fibrosis"].map(P.COST_HCV_MONTHLY).sum()
        sbi = m["new_sbi"].map(P.COST_SBI_HOSPITALISATION).fillna(0.0).sum()
        rec.update(cost_drug_use=drug, cost_hiv=hiv, cost_hcv=hcv, cost_sbi=sbi)
        self.records.append(rec)

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
