"""Vectorized replication of the probabilistic cellular automaton in:

Duffus ALJ, Standridge JP, Bartlett PL, George JC (2026).
"Using Probabilistic Cellular Automata to Model the Transmission of an
Emerging Infectious Disease of Amphibians -- A Preliminary Model."
Pathogens 15(8):827. https://doi.org/10.3390/pathogens15080827

The authors' original code (Supplementary File S1, mirrored in
``original/Ranavirus(Jul-13-26).py``) simulates one pond at a time with
nested Python loops.  This module reproduces the same stochastic process
with NumPy, batching all ponds at once, so a 10,000-pond experiment takes
seconds instead of many minutes.

Model semantics preserved from the original code
------------------------------------------------
* 10x10 grid, Moore (8-cell) neighborhood, hard boundaries (no wrap).
* States: S(0) susceptible, U(1) ulcerative, H(2) hemorrhagic,
  C(3) combined, D(4) dead.
* Initial condition: all S except one C frog at grid position
  (wid-2, lnt-2) = (8, 8).
* Synchronous update: a frog infected during pass t cannot transmit
  until pass t+1 (transmission tests read neighbors' *current* state;
  the focal frog accumulates changes in its *future* state).
* Within one pass a frog's future state evolves across its (ordered)
  neighbor checks, so S -> U/H -> C is possible within a single pass.
* Each infectious neighbor triggers an independent Bernoulli test per
  strain it carries (ph for hemorrhagic, pu for ulcerative).
* Quirk faithfully reproduced: in ``test_infect`` the strain checks both
  read the future state captured at the *start* of the call, so a
  susceptible frog that catches BOTH strains from a single C neighbor in
  one call ends the call in state U (the ulcerative branch overwrites
  the hemorrhagic one), not C.
* After the infection sweep, every infected frog (including those
  infected this very pass) dies with probability pd.
* Dead frogs never transmit, never recover, and are never replaced.

Summary statistics are computed exactly as the original code computes
them, including its convention of dividing the "all infected by"
iteration sum by *all* ponds (ponds that never reach 100% infection
contribute zero, biasing the mean downward at high mortality).
"""

import numpy as np

# Neighbor offsets in the exact order used by the original find_nbrs()
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

S, U, H, C, D = 0, 1, 2, 3, 4
GRID = 10  # both dimensions


def _shift(grid, dr, dc, fill=D):
    """Return array whose (i,j) entry is grid[i+dr, j+dc], out-of-bounds -> fill."""
    out = np.full_like(grid, fill)
    src_r = slice(max(dr, 0), GRID + min(dr, 0))
    dst_r = slice(max(-dr, 0), GRID + min(-dr, 0))
    src_c = slice(max(dc, 0), GRID + min(dc, 0))
    dst_c = slice(max(-dc, 0), GRID + min(-dc, 0))
    out[:, dst_r, dst_c] = grid[:, src_r, src_c]
    return out


def simulate(pu=0.65, ph=0.75, pd=0.15, n_ponds=10_000, iters=120, seed=0):
    """Simulate n_ponds independent ponds for `iters` passes.

    Returns a dict with per-iteration mean/min/max counts for each state
    (length iters+1, index 0 = initial condition) and the paper's summary
    statistics computed with the original code's conventions.
    """
    rng = np.random.default_rng(seed)
    shape = (n_ponds, GRID, GRID)

    cur = np.zeros(shape, dtype=np.int8)
    cur[:, GRID - 2, GRID - 2] = C  # Pond[wid-2][lnt-2].set_state("C")

    n_states = 5
    counts = np.zeros((iters + 1, n_states, n_ponds), dtype=np.int32)
    for k in range(n_states):
        counts[0, k] = (cur == k).sum(axis=(1, 2))

    first_all_infected = np.zeros(n_ponds, dtype=np.int32)  # 0 = never
    first_all_dead = np.zeros(n_ponds, dtype=np.int32)      # 0 = never

    total_cells = GRID * GRID
    for t in range(1, iters + 1):
        fut = cur.copy()
        for dr, dc in OFFSETS:
            nb = _shift(cur, dr, dc)
            carries_h = (nb == H) | (nb == C)
            carries_u = (nb == U) | (nb == C)
            fut0 = fut.copy()  # state captured at start of test_infect call
            # Hemorrhagic check: applies when captured state is S or U
            hs = carries_h & (rng.random(shape) <= ph)
            fut[(fut0 == U) & hs] = C
            fut[(fut0 == S) & hs] = H
            # Ulcerative check: condition uses the *captured* state, so a
            # frog that just caught H in this same call is still treated
            # as S here and, on success, is overwritten to U (original quirk)
            us = carries_u & (rng.random(shape) <= pu)
            fut[(fut0 == H) & us] = C
            fut[(fut0 == S) & us] = U
        cur = fut
        # Death phase: every currently infected frog (incl. newly infected)
        infected = (cur == U) | (cur == H) | (cur == C)
        cur[infected & (rng.random(shape) <= pd)] = D

        for k in range(n_states):
            counts[t, k] = (cur == k).sum(axis=(1, 2))
        newly_all_inf = (counts[t, S] == 0) & (first_all_infected == 0)
        first_all_infected[newly_all_inf] = t
        newly_all_dead = (counts[t, D] == total_cells) & (first_all_dead == 0)
        first_all_dead[newly_all_dead] = t

    dead_ponds = first_all_dead > 0
    results = {
        "mean": counts.mean(axis=2),          # (iters+1, 5)
        "min": counts.min(axis=2),
        "max": counts.max(axis=2),
        # As computed by the original code (divides by ALL ponds):
        "all_infected_mean": first_all_infected.sum() / n_ponds,
        "all_dead_mean": (first_all_dead[dead_ponds].mean() if dead_ponds.any() else np.nan),
        "dead_pond_count": int(dead_ponds.sum()),
        # Extra diagnostics not in the paper:
        "all_infected_pond_count": int((first_all_infected > 0).sum()),
        "all_infected_conditional_mean": (
            first_all_infected[first_all_infected > 0].mean()
            if (first_all_infected > 0).any() else np.nan
        ),
        "params": dict(pu=pu, ph=ph, pd=pd, n_ponds=n_ponds, iters=iters, seed=seed),
    }
    return results
