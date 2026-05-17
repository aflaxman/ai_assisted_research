"""Lens #4: Agent-Based Model on a small-world contact network.

Strengths:
- Individuals matter: heterogeneity (age, contacts), behavior, networks.
- Honest stochasticity: outbreaks fizzle, take off, or peak late, with
  realistic run-to-run variance.
- Easy to add structure (households, schools, vaccinated subgroups) where ODEs
  would require ever more compartments.

Weaknesses:
- Slow: hundreds of runs to map the distribution.
- Many small choices (network, transmission rule) shape the answer.
- Harder to communicate "what assumption drove the result" than with an ODE.

This ABM runs the same disease, same town, same vaccination on a small-world
graph. The interesting outputs are the *distribution* of outcomes and the
fraction of runs in which the outbreak goes extinct - which the ODE cannot see.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from scenario import Scenario, DEFAULT


SUSCEPTIBLE, INFECTED, REMOVED = 0, 1, 2


@dataclass
class ABMRun:
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    total_infected: int
    peak_prevalence: int
    peak_day: int
    extinct: bool


def _build_network(N: int, mean_degree: int, rewire: float, seed: int) -> nx.Graph:
    g = nx.watts_strogatz_graph(N, mean_degree, rewire, seed=seed)
    return g


def simulate_one(
    scn: Scenario = DEFAULT,
    vaccinated: bool = False,
    mean_degree: int = 10,
    rewire: float = 0.1,
    seed: int = 0,
) -> ABMRun:
    rng = np.random.default_rng(seed)
    g = _build_network(scn.N, mean_degree, rewire, seed)
    # Per-edge daily transmission probability calibrated so that
    # expected secondary cases per infected ~ R0.
    p_trans = scn.R0 / (mean_degree * scn.infectious_days)
    p_recover = 1.0 - np.exp(-scn.gamma)

    state = np.zeros(scn.N, dtype=np.int8)

    # Vaccination: a random fraction is moved straight to REMOVED.
    if vaccinated:
        n_immune = int(scn.achievable_coverage * scn.vaccine_effectiveness * scn.N)
        immune_idx = rng.choice(scn.N, size=n_immune, replace=False)
        state[immune_idx] = REMOVED

    # Seed initial infections in still-susceptible agents
    susceptible = np.where(state == SUSCEPTIBLE)[0]
    seeds = rng.choice(susceptible, size=scn.initial_infected, replace=False)
    state[seeds] = INFECTED

    adj = [np.array(list(g.neighbors(i)), dtype=np.int64) for i in range(scn.N)]

    S_t, I_t, R_t = [], [], []
    for _day in range(scn.horizon_days + 1):
        S_t.append(int(np.sum(state == SUSCEPTIBLE)))
        I_t.append(int(np.sum(state == INFECTED)))
        R_t.append(int(np.sum(state == REMOVED)))

        if I_t[-1] == 0:
            # Pad remaining days
            remaining = scn.horizon_days - _day
            S_t.extend([S_t[-1]] * remaining)
            I_t.extend([0] * remaining)
            R_t.extend([R_t[-1]] * remaining)
            break

        infected_idx = np.where(state == INFECTED)[0]
        new_infections = []
        for i in infected_idx:
            neighbors = adj[i]
            if neighbors.size == 0:
                continue
            sus_mask = state[neighbors] == SUSCEPTIBLE
            sus_neighbors = neighbors[sus_mask]
            if sus_neighbors.size == 0:
                continue
            hits = rng.random(sus_neighbors.size) < p_trans
            new_infections.append(sus_neighbors[hits])
        recoveries = infected_idx[rng.random(infected_idx.size) < p_recover]

        if new_infections:
            new_idx = np.unique(np.concatenate(new_infections))
            state[new_idx] = INFECTED
        state[recoveries] = REMOVED

    S = np.array(S_t[: scn.horizon_days + 1])
    I = np.array(I_t[: scn.horizon_days + 1])
    R = np.array(R_t[: scn.horizon_days + 1])

    # Total infected during this run (excluding vaccinated-at-start).
    n_immune_start = int(R[0])
    total_inf = int(R[-1] - n_immune_start) + int(I[-1])
    extinct = total_inf < 0.05 * (scn.N - n_immune_start)
    return ABMRun(
        S=S, I=I, R=R,
        total_infected=total_inf,
        peak_prevalence=int(I.max()),
        peak_day=int(I.argmax()),
        extinct=extinct,
    )


def simulate_many(
    scn: Scenario = DEFAULT,
    vaccinated: bool = False,
    n_runs: int = 200,
    progress: bool = False,
    base_seed: int = 1,
    **kwargs,
) -> List[ABMRun]:
    it = range(n_runs)
    if progress:
        it = tqdm(it, desc=f"ABM ({'vax' if vaccinated else 'no-vax'})")
    return [simulate_one(scn, vaccinated=vaccinated, seed=base_seed + i, **kwargs) for i in it]


def extinction_probability(runs: List[ABMRun]) -> float:
    return sum(r.extinct for r in runs) / max(len(runs), 1)
