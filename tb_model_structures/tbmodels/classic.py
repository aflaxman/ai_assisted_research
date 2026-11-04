"""The 12 Menzies (2018) progression structures A-L as declarative Structures.

Transcribed from Figure 1 of Menzies et al., Lancet Infect Dis 2018;18:e228
(PMC6070419).  Notation: S susceptible; L latent; Ls slow latent; Lf fast
latent; I active disease; lambda force of infection; c progression from L/Ls;
d progression from Lf; e transfer Lf->Ls; f transfer Ls->Lf; a probability of
immediate progression on infection; b probability of entering Lf on infection.

Every structure gets a shared disease-exit block (decision 1, PLAN.md section 7):
the single active-disease compartment I loses mass to self-cure (I->Rec) and TB
death (I->Dead).  The entry rule places a newly infected cohort into the
post-infection compartments; the S->... force-of-infection step only matters in
transmission mode, where S is included.

Chains B and J, and the time-since-infection hazards K and L, are individual-
based in their source papers; here they are rendered as ODE compartment chains
(fixed length ``N_TUNNEL`` / ``N_CHAIN``) and as single-latent states with
time-varying progression.  The cohort cumulative-incidence curve is what the
Menzies review compares, and these renderings reproduce its qualitative shape
(B: early high then declining risk; J: rising risk; K/L: declining risk).
"""
from __future__ import annotations

import numpy as np

from .core import Compartment, Structure, Transition, const, foi

# structural constants for the chain structures
N_TUNNEL = 3  # fast-latent tunnel states in structure B
N_CHAIN = 4   # sequential latent states in structure J

# shared exit-block parameters
EXIT_PARAMS = ("self_cure", "tb_death")


def _active_I() -> Compartment:
    return Compartment("I", "disease", band="clinical", infectious=True,
                       rel_inf=1.0, symptomatic=True, bact_positive=True)


def _with_exit(name, family, comps, trans, params, entry, notes=""):
    """Append Rec/Dead compartments and the I->Rec, I->Dead exit transitions."""
    comps = list(comps) + [
        Compartment("Rec", "recovered"),
        Compartment("Dead", "dead"),
    ]
    trans = list(trans) + [
        Transition("I", "Rec", const("self_cure"), "self-cure"),
        Transition("I", "Dead", const("tb_death"), "TB death"),
    ]
    params = list(params) + list(EXIT_PARAMS)
    return Structure(
        name=name, family=family, compartments=comps, transitions=trans,
        param_names=params, entry=entry, disease_states=("I",), notes=notes,
    )


def structure_A() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("L", "latent"), _active_I()]
    trans = [
        Transition("S", "L", foi(), "infection"),
        Transition("L", "I", const("c"), "progression"),
    ]
    return _with_exit("A", "classic", comps, trans, ["c"], lambda p: {"L": 1.0},
                      notes="Constant progression hazard; special case of E with a=0.")


def structure_B() -> Structure:
    comps = [Compartment("S", "susceptible")]
    comps += [Compartment(f"Lf{i}", "latent") for i in range(1, N_TUNNEL + 1)]
    comps += [Compartment("Ls", "latent"), _active_I()]
    trans = [Transition("S", "Lf1", foi(), "infection")]
    # tunnel transfer at rate g; each tunnel state progresses to I at rate d
    for i in range(1, N_TUNNEL):
        trans.append(Transition(f"Lf{i}", f"Lf{i+1}", const("g"), "tunnel"))
    trans.append(Transition(f"Lf{N_TUNNEL}", "Ls", const("g"), "tunnel->slow"))
    for i in range(1, N_TUNNEL + 1):
        trans.append(Transition(f"Lf{i}", "I", const("d"), "fast progression"))
    trans.append(Transition("Ls", "I", const("c"), "slow progression"))
    return _with_exit("B", "classic", comps, trans, ["d", "c", "g"],
                      lambda p: {"Lf1": 1.0},
                      notes=f"Fast-latent tunnel ({N_TUNNEL} states) then slow latent.")


def structure_C() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("Lf", "latent"),
             Compartment("Ls", "latent"), _active_I()]
    trans = [
        Transition("S", "Lf", foi(), "infection"),
        Transition("Lf", "I", const("d"), "fast progression"),
        Transition("Lf", "Ls", const("e"), "fast->slow"),
        Transition("Ls", "I", const("c"), "slow progression"),
    ]
    return _with_exit("C", "classic", comps, trans, ["d", "e", "c"],
                      lambda p: {"Lf": 1.0},
                      notes="Fast/slow latent in series; special case of G with a=0.")


def structure_D() -> Structure:
    comps = [Compartment("S", "susceptible"), _active_I()]
    trans = [Transition("S", "I", foi(), "immediate progression")]
    return _with_exit("D", "classic", comps, trans, [], lambda p: {"I": 1.0},
                      notes="Immediate progression for all; used only for advanced HIV.")


def structure_E() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("L", "latent"), _active_I()]
    trans = [
        Transition("S", "I", lambda p, lam, t: lam * p["a"], "fast infection"),
        Transition("S", "L", lambda p, lam, t: lam * (1 - p["a"]), "slow infection"),
        Transition("L", "I", const("c"), "reactivation"),
    ]
    return _with_exit("E", "classic", comps, trans, ["a", "c"],
                      lambda p: {"I": p["a"], "L": 1 - p["a"]},
                      notes="The standard Blower-type fast/slow split (49% of models).")


def structure_F() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("Ls", "latent"),
             Compartment("Lf", "latent"), _active_I()]
    trans = [
        Transition("S", "Ls", lambda p, lam, t: lam * (1 - p["b"]), "slow infection"),
        Transition("S", "Lf", lambda p, lam, t: lam * p["b"], "fast infection"),
        Transition("Ls", "I", const("c"), "slow progression"),
        Transition("Lf", "I", const("d"), "fast progression"),
    ]
    return _with_exit("F", "classic", comps, trans, ["b", "c", "d"],
                      lambda p: {"Ls": 1 - p["b"], "Lf": p["b"]},
                      notes="Parallel fast/slow latent compartments, no transfer.")


def structure_G() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("Lf", "latent"),
             Compartment("Ls", "latent"), _active_I()]
    trans = [
        Transition("S", "I", lambda p, lam, t: lam * p["a"], "immediate progression"),
        Transition("S", "Lf", lambda p, lam, t: lam * (1 - p["a"]), "infection"),
        Transition("Lf", "I", const("d"), "fast progression"),
        Transition("Lf", "Ls", const("e"), "fast->slow"),
        Transition("Ls", "I", const("c"), "slow progression"),
    ]
    return _with_exit("G", "classic", comps, trans, ["a", "d", "e", "c"],
                      lambda p: {"I": p["a"], "Lf": 1 - p["a"]},
                      notes="Structure C plus immediate progression.")


def structure_H() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("Ls", "latent"),
             Compartment("Lf", "latent"), _active_I()]
    trans = [
        Transition("S", "Ls", lambda p, lam, t: lam * (1 - p["b"]), "slow infection"),
        Transition("S", "Lf", lambda p, lam, t: lam * p["b"], "fast infection"),
        Transition("Ls", "Lf", const("f"), "slow->fast"),
        Transition("Lf", "I", const("d"), "progression"),
    ]
    return _with_exit("H", "classic", comps, trans, ["b", "f", "d"],
                      lambda p: {"Ls": 1 - p["b"], "Lf": p["b"]},
                      notes="Progression only from Lf; Ls feeds Lf.")


def structure_I() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("Lf", "latent"),
             Compartment("Ls", "latent"), _active_I()]
    trans = [
        Transition("S", "Lf", foi(), "infection"),
        Transition("Lf", "I", const("d"), "progression"),
        Transition("Lf", "Ls", const("e"), "fast->slow"),
        Transition("Ls", "Lf", const("f"), "slow->fast"),
    ]
    return _with_exit("I", "classic", comps, trans, ["d", "e", "f"],
                      lambda p: {"Lf": 1.0},
                      notes="Progression only from Lf; reversible Lf<->Ls.")


def structure_J() -> Structure:
    comps = [Compartment("S", "susceptible")]
    comps += [Compartment(f"L{i}", "latent") for i in range(1, N_CHAIN + 1)]
    comps += [_active_I()]
    trans = [Transition("S", "L1", foi(), "infection")]
    for i in range(1, N_CHAIN):
        trans.append(Transition(f"L{i}", f"L{i+1}", const("f"), "ageing"))
    trans.append(Transition(f"L{N_CHAIN}", "I", const("c"), "progression"))
    return _with_exit("J", "classic", comps, trans, ["f", "c"],
                      lambda p: {"L1": 1.0},
                      notes=f"Sequential latent chain ({N_CHAIN}); risk rises with time since infection.")


def structure_K() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("L", "latent"), _active_I()]
    # c_t = x1 * max(1, t)^x2
    rate_k = lambda p, lam, t: p["x1"] * max(1.0, t) ** p["x2"]
    trans = [
        Transition("S", "L", foi(), "infection"),
        Transition("L", "I", rate_k, "time-varying progression"),
    ]
    return _with_exit("K", "classic", comps, trans, ["x1", "x2"],
                      lambda p: {"L": 1.0},
                      notes="Single latent; hazard x1*max(1,t)^x2 (Blaser 2016).")


def structure_L() -> Structure:
    comps = [Compartment("S", "susceptible"), Compartment("L", "latent"), _active_I()]
    # c_t = x1 * (x2 + exp(-x3 * t)); x2 >= 0 keeps the hazard non-negative, and
    # the max(0, .) guards against an out-of-range draw during calibration.
    rate_l = lambda p, lam, t: max(0.0, p["x1"] * (p["x2"] + np.exp(-p["x3"] * t)))
    trans = [
        Transition("S", "L", foi(), "infection"),
        Transition("L", "I", rate_l, "time-varying progression"),
    ]
    return _with_exit("L", "classic", comps, trans, ["x1", "x2", "x3"],
                      lambda p: {"L": 1.0},
                      notes="Single latent; hazard x1*(x2+exp(-x3 t)).")


CLASSIC_BUILDERS = {
    "A": structure_A, "B": structure_B, "C": structure_C, "D": structure_D,
    "E": structure_E, "F": structure_F, "G": structure_G, "H": structure_H,
    "I": structure_I, "J": structure_J, "K": structure_K, "L": structure_L,
}


def all_classic() -> dict[str, Structure]:
    return {k: v() for k, v in CLASSIC_BUILDERS.items()}
