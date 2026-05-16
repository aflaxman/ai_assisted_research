"""Lens #2: Decision Tree (static cost-effectiveness analysis).

Strengths:
- Transparent: every probability and payoff is visible in one diagram.
- Fast: arithmetic, not simulation.
- The right tool for one-shot policy choices when timing is not the question.

Weaknesses:
- Static: no time, no feedback, no dynamics.
- Probabilities come from somewhere - usually from another model. Garbage in,
  garbage out: a decision tree is only as good as the numbers feeding it.
- Awkward for path-dependent or stochastic effects beyond simple chance nodes.

This file wires the tree to the ODE final-size for the "outbreak happens" branch
and a stochastic-extinction probability for the "no outbreak" branch, which is
something the ODE alone cannot tell us.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from scenario import Scenario, DEFAULT
from ode_sir import final_size


@dataclass
class Node:
    """A single node in the decision tree.

    - kind = 'decision': children are choices; the analyst picks the best.
    - kind = 'chance':   children carry probabilities; we take the expectation.
    - kind = 'terminal': leaf with a deterministic payoff.
    """
    name: str
    kind: str  # 'decision' | 'chance' | 'terminal'
    children: List["Node"] = field(default_factory=list)
    prob: float = 1.0       # only meaningful under a chance parent
    payoff: float = 0.0     # only meaningful at terminals (cost; lower is better)


def evaluate(node: Node) -> tuple[float, Optional[str]]:
    """Return (expected_cost, best_choice_label)."""
    if node.kind == "terminal":
        return node.payoff, None
    if node.kind == "chance":
        total = sum(c.prob * evaluate(c)[0] for c in node.children)
        return total, None
    if node.kind == "decision":
        options = [(c.name, evaluate(c)[0]) for c in node.children]
        best_name, best_cost = min(options, key=lambda x: x[1])
        return best_cost, best_name
    raise ValueError(f"unknown node kind: {node.kind}")


def outbreak_cost(scn: Scenario, vaccinated: bool) -> float:
    """Total cost given an outbreak takes off, using the ODE final-size eqn."""
    s0 = scn.effective_susceptible_fraction(vaccinated)
    attack_rate = final_size(scn.R0, s0)
    cases = attack_rate * scn.N * s0
    hosp = cases * scn.hospitalization_rate
    mild = cases - hosp
    treatment_cost = mild * scn.cost_mild_case + hosp * scn.cost_hospitalization
    vaccine_cost = scn.cost_vaccine_dose * scn.achievable_coverage * scn.N if vaccinated else 0.0
    return treatment_cost + vaccine_cost


def build_tree(
    scn: Scenario = DEFAULT,
    extinction_prob_no_vax: float = 0.0,
    extinction_prob_vax: float = 0.0,
) -> Node:
    """Build the policy decision tree.

    extinction_prob_* let us inject stochastic extinction estimates from the ABM
    - the deterministic ODE cannot tell us how often the outbreak fizzles.
    """
    vaccine_only_cost = scn.cost_vaccine_dose * scn.achievable_coverage * scn.N

    no_vax = Node(
        name="Do nothing",
        kind="chance",
        children=[
            Node("Outbreak takes off", kind="terminal",
                 prob=1 - extinction_prob_no_vax,
                 payoff=outbreak_cost(scn, vaccinated=False)),
            Node("Outbreak fizzles", kind="terminal",
                 prob=extinction_prob_no_vax,
                 payoff=0.0),
        ],
    )
    vax = Node(
        name="Mass-vaccinate",
        kind="chance",
        children=[
            Node("Outbreak takes off", kind="terminal",
                 prob=1 - extinction_prob_vax,
                 payoff=outbreak_cost(scn, vaccinated=True)),
            Node("Outbreak fizzles", kind="terminal",
                 prob=extinction_prob_vax,
                 payoff=vaccine_only_cost),
        ],
    )
    return Node(name="Policy choice", kind="decision", children=[no_vax, vax])


def summarize(tree: Node) -> dict:
    """Walk the tree once and return a flat summary for plotting/reporting."""
    out = {"options": [], "best": None}
    for option in tree.children:
        ev, _ = evaluate(option)
        leaves = [{"label": c.name, "prob": c.prob, "cost": c.payoff} for c in option.children]
        out["options"].append({"label": option.name, "expected_cost": ev, "leaves": leaves})
    _, best = evaluate(tree)
    out["best"] = best
    return out
