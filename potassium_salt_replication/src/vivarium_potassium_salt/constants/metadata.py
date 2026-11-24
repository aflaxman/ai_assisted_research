"""Names shared across the model: scenarios, age groups, causes, and measures."""
from __future__ import annotations

from typing import NamedTuple

PAPER = (
    "Huang L, Xu X, Matsushita K, et al. Global health outcomes from replacing dietary "
    "sodium chloride with potassium-enriched salt. Nat. Health (2026). "
    "https://doi.org/10.1038/s44360-026-00216-9"
)

# The paper's inputs are for 2021; the cached GBD 2023 extract is for that year too.
REFERENCE_YEAR = 2021

SEXES = ("Female", "Male")

#############
# Age bands #
#############


class AgeBand(NamedTuple):
    name: str  # GBD age group name, as the IHME connector spells it
    start: float
    end: float

    @property
    def label(self) -> str:
        """Name in the form the results stratifier uses, e.g. '25_to_29'."""
        return self.name.replace(" ", "_")

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.start + self.end)


# Adults aged 25 and over in five-year bands, as in the paper.  The open-ended
# band is closed at 100 so that its midpoint (97.5) is a usable age.
AGE_BANDS = tuple(
    [AgeBand(f"{start} to {start + 4}", float(start), float(start + 5)) for start in range(25, 95, 5)]
    + [AgeBand("95 plus", 95.0, 100.0)]
)
AGE_BAND_NAMES = tuple(band.name for band in AGE_BANDS)
MIN_AGE, MAX_AGE = AGE_BANDS[0].start, AGE_BANDS[-1].end

##########
# Causes #
##########

# The outcomes with a GBD 2019 relative risk for systolic blood pressure, named
# as in vivarium.gbd_mapping.  The five aetiologies of chronic kidney disease
# share one relative risk, so they are modelled as their parent cause.
CVD_CAUSES = (
    "ischemic_heart_disease",
    "ischemic_stroke",
    "intracerebral_hemorrhage",
    "subarachnoid_hemorrhage",
    "hypertensive_heart_disease",
    "rheumatic_heart_disease",
    "non_rheumatic_calcific_aortic_valve_disease",
    "other_cardiomyopathy",
    "atrial_fibrillation_and_flutter",
    "aortic_aneurysm",
    "lower_extremity_peripheral_arterial_disease",
    "endocarditis",
)
CKD_CAUSE = "chronic_kidney_disease"
CAUSES = CVD_CAUSES + (CKD_CAUSE,)

# The paper keeps the "inflammatory heart diseases" in its death counts but
# drops them from new cases and DALYs, because their incidence does not depend
# much on blood pressure.
INFLAMMATORY_HEART_DISEASES = (
    "rheumatic_heart_disease",
    "endocarditis",
    "other_cardiomyopathy",
)
# GBD publishes no incidence for these.
NO_INCIDENCE = ("hypertensive_heart_disease", "aortic_aneurysm", "other_cardiomyopathy")

MEASURES = ("deaths", "incident_cases", "dalys")
CAUSES_BY_MEASURE = {
    "deaths": CAUSES,
    "incident_cases": tuple(
        c for c in CAUSES if c not in INFLAMMATORY_HEART_DISEASES + NO_INCIDENCE
    ),
    "dalys": tuple(c for c in CAUSES if c not in INFLAMMATORY_HEART_DISEASES),
}

# How the IHME connector spells each cause.
GBD_CAUSE_NAMES = {
    "ischemic_heart_disease": "Ischemic heart disease",
    "ischemic_stroke": "Ischemic stroke",
    "intracerebral_hemorrhage": "Intracerebral hemorrhage",
    "subarachnoid_hemorrhage": "Subarachnoid hemorrhage",
    "hypertensive_heart_disease": "Hypertensive heart disease",
    "rheumatic_heart_disease": "Rheumatic heart disease",
    "non_rheumatic_calcific_aortic_valve_disease": "Non-rheumatic calcific aortic valve disease",
    "other_cardiomyopathy": "Other cardiomyopathy",
    "atrial_fibrillation_and_flutter": "Atrial fibrillation and flutter",
    "aortic_aneurysm": "Aortic aneurysm",
    "lower_extremity_peripheral_arterial_disease": "Lower extremity peripheral arterial disease",
    "endocarditis": "Endocarditis",
    "chronic_kidney_disease": "Chronic kidney disease",
}

#############
# Scenarios #
#############


class Scenario(NamedTuple):
    """One salt-replacement policy.

    ``sources`` says which salt is replaced: ``all``, ``discretionary`` (added
    at the table or in cooking), or ``non_discretionary`` (processed and
    restaurant food).  ``eligible`` says who receives it, and ``uptake`` is the
    share of the eligible who use it.
    """

    name: str
    sources: str = "all"
    eligible: str = "everyone"
    uptake: float = 1.0
    hyperkalaemia_modelled: bool = True
    label: str = ""


BASELINE = Scenario("baseline", sources="none", label="No replacement")
SCENARIOS = (
    Scenario("all_salt", "all", label="(1) All salt"),
    Scenario("discretionary", "discretionary", label="(2) Discretionary salt"),
    Scenario("non_discretionary", "non_discretionary", label="(3) Non-discretionary salt"),
    # In the two targeted scenarios the paper assumes that a fifth of people
    # with hypertension are advised against the salt because of kidney disease,
    # and it models no hyperkalaemia harm among the rest.
    Scenario("discretionary_diagnosed_hypertension", "discretionary", "diagnosed", 0.8, False,
             "(4) Discretionary salt, diagnosed hypertension"),
    Scenario("discretionary_treated_hypertension", "discretionary", "treated", 0.8, False,
             "(5) Discretionary salt, treated hypertension"),
)
SCENARIOS_BY_NAME = {s.name: s for s in (BASELINE,) + SCENARIOS}

#######################
# Hypertension status #
#######################

NORMOTENSIVE = "normotensive"
UNDIAGNOSED = "undiagnosed_hypertension"
DIAGNOSED_UNTREATED = "diagnosed_untreated_hypertension"
TREATED = "treated_hypertension"
HYPERTENSION_STATES = (NORMOTENSIVE, UNDIAGNOSED, DIAGNOSED_UNTREATED, TREATED)

##########################
# Chronic kidney disease #
##########################

NO_ADVANCED_CKD = "none"
CKD_STAGES = ("G3a", "G3b", "G4", "G5")
CKD_STATES = (NO_ADVANCED_CKD,) + CKD_STAGES
