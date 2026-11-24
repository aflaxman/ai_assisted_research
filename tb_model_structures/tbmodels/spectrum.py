"""Spectrum TB models: bidirectional minimal <-> subclinical <-> clinical.

Following Horton et al. 2023 (PNAS 120:e2221186120) and Richards et al. 2023
(Lancet Glob Health).  States:

* Inf  -- Mtb infection, no disease (upstream of the spectrum)
* Min  -- minimal disease: pathology, not yet bacteriologically positive,
          non-infectious, asymptomatic
* Sub  -- subclinical disease: bacteriologically positive, infectious,
          asymptomatic
* Clin -- clinical disease: bacteriologically positive, infectious, symptomatic
* Clr  -- infection cleared without disease
* Rec  -- recovered after disease
* Dead -- TB death

Three variants (decision, PLAN.md section 2):

* ``SP-full``        -- Horton main model (Inf -> Clr/Min/Sub; Min -> Rec; the
                        full bidirectional spectrum; Clin -> Dead).
* ``SP-noclear``     -- same, minus self-clearance of infection (no Inf -> Clr).
* ``SP-nomin``       -- Horton without the minimal state (Inf -> Clr/Sub;
                        Sub -> Rec for disease recovery; Sub <-> Clin).
* ``SP-nomin-noclear`` -- both removed; the closest spectrum analogue of the
                        single-pathway classic structure A.

Posterior medians from the two source repositories are reproduced in the
calibration milestone; the parameter names here match Horton's LibBi model so
their 150,000-draw trace can be used as an acceptance test.
"""
from __future__ import annotations

from .core import Compartment, Structure, Transition, const

# relative infectiousness of subclinical disease (transmission-mode parameter,
# not a cohort-calibration parameter); clinical disease has rel_inf 1.0.
REL_INF_SUB = "rel_inf_sub"


def _spectrum(name, *, include_min: bool, include_clear: bool) -> Structure:
    comps = [Compartment("Inf", "latent")]
    trans = []
    params = []

    if include_clear:
        comps.append(Compartment("Clr", "cleared"))
        trans.append(Transition("Inf", "Clr", const("infclr"), "self-clearance"))
        params.append("infclr")

    sub = Compartment("Sub", "disease", band="subclinical", infectious=True,
                      rel_inf=REL_INF_SUB, symptomatic=False, bact_positive=True)
    clin = Compartment("Clin", "disease", band="clinical", infectious=True,
                       rel_inf=1.0, symptomatic=True, bact_positive=True)
    comps += [sub, clin, Compartment("Rec", "recovered"), Compartment("Dead", "dead")]

    if include_min:
        comps.insert(1, Compartment("Min", "disease", band="minimal",
                                    infectious=False, symptomatic=False, bact_positive=False))
        trans += [
            Transition("Inf", "Min", const("infmin"), "infection->minimal"),
            Transition("Inf", "Sub", const("infsub"), "infection->subclinical"),
            Transition("Min", "Rec", const("minrec"), "minimal recovery"),
            Transition("Min", "Sub", const("min_sub"), "minimal->subclinical"),
            Transition("Sub", "Min", const("sub_min"), "subclinical->minimal"),
        ]
        params += ["infmin", "infsub", "minrec", "min_sub", "sub_min"]
        entry = lambda p: {"Inf": 1.0}
    else:
        # no minimal state: infection enters subclinical directly; disease
        # recovery is from subclinical (sub_rec)
        trans += [
            Transition("Inf", "Sub", const("infsub"), "infection->subclinical"),
            Transition("Sub", "Rec", const("sub_rec"), "subclinical recovery"),
        ]
        params += ["infsub", "sub_rec"]
        entry = lambda p: {"Inf": 1.0}

    trans += [
        Transition("Sub", "Clin", const("sub_clin"), "subclinical->clinical"),
        Transition("Clin", "Sub", const("clin_sub"), "clinical->subclinical"),
        Transition("Clin", "Dead", const("mort"), "TB death"),
    ]
    params += ["sub_clin", "clin_sub", "mort"]
    # phi: share of a just-infectious cohort that is subclinical (vs clinical) at
    # t=0.  An observation-model nuisance for the transition data only; it does
    # not appear in any transition rate, so it is inert in the ODE.
    params.append("phi")

    return Structure(
        name=name, family="spectrum", compartments=comps, transitions=trans,
        param_names=params, entry=entry,
        disease_states=("Sub", "Clin"),  # bacteriologically positive = "infectious TB"
        notes=f"Spectrum model (min={include_min}, clear={include_clear}).",
    )


def spectrum_full() -> Structure:
    return _spectrum("SP-full", include_min=True, include_clear=True)


def spectrum_noclear() -> Structure:
    return _spectrum("SP-noclear", include_min=True, include_clear=False)


def spectrum_nomin() -> Structure:
    return _spectrum("SP-nomin", include_min=False, include_clear=True)


def spectrum_nomin_noclear() -> Structure:
    return _spectrum("SP-nomin-noclear", include_min=False, include_clear=False)


SPECTRUM_BUILDERS = {
    "SP-full": spectrum_full,
    "SP-noclear": spectrum_noclear,
    "SP-nomin": spectrum_nomin,
    "SP-nomin-noclear": spectrum_nomin_noclear,
}


def all_spectrum() -> dict[str, Structure]:
    return {k: v() for k, v in SPECTRUM_BUILDERS.items()}
