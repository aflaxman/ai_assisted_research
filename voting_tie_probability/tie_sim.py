#!/usr/bin/env python3
"""
Monte Carlo simulation: probability of a tie for first place in multi-candidate voting.

Context: The Douglass-Truth Library naming vote (Seattle, 1975) produced a
first-place tie between Frederick Douglass and Sojourner Truth among ~2,000
voters choosing from 10 candidates. How unlikely was that?

Usage:
    python tie_sim.py [--trials N] [--voters N] [--plot]
"""

import argparse
import numpy as np
from typing import NamedTuple


class ScenarioResult(NamedTuple):
    """Results from simulating a voting scenario."""
    name: str
    p_tie: float  # P(D and T tie for first place)
    se: float  # standard error


def simulate_scenario(
    probs: np.ndarray,
    n_voters: int,
    n_trials: int,
    idx_d: int = 0,  # index of Douglass in probs array
    idx_t: int = 1,  # index of Truth in probs array
    rng: np.random.Generator = None,
) -> tuple[float, float]:
    """
    Simulate votes and compute probability of a two-way tie for first place.

    Args:
        probs: Array of vote share probabilities for each candidate (must sum to 1)
        n_voters: Number of voters in each trial
        n_trials: Number of Monte Carlo trials
        idx_d, idx_t: Indices for the two candidates of interest
        rng: Random number generator

    Returns:
        (p_tie, standard_error)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw all trials at once: shape (n_trials, n_candidates)
    votes = rng.multinomial(n_voters, probs, size=n_trials)

    # Extract counts for Douglass and Truth
    x_d = votes[:, idx_d]
    x_t = votes[:, idx_t]

    # Two-way tie for first place:
    # Both must equal the max, and exactly 2 candidates share that max
    max_votes = votes.max(axis=1)
    d_at_max = (x_d == max_votes)
    t_at_max = (x_t == max_votes)
    at_max_count = (votes == max_votes[:, np.newaxis]).sum(axis=1)

    first_place_tie = d_at_max & t_at_max & (at_max_count == 2)

    # Compute probability and standard error
    p_tie = first_place_tie.mean()
    se = np.sqrt(p_tie * (1 - p_tie) / n_trials)

    return p_tie, se


def binomial_tie_probability(n: int, p: float = 0.5) -> tuple[float, float]:
    """
    Exact probability of a tie in a 2-candidate race with n voters.

    For fair coin (p=0.5): P(tie) = C(n, n/2) * (0.5)^n

    Uses Stirling approximation for large n:
    P(tie) ≈ sqrt(2 / (pi * n))
    """
    from math import sqrt, pi, lgamma, log, exp

    if n % 2 == 1:
        return 0.0, 0.0  # Odd voters can't tie

    # Use log-space calculation to avoid overflow
    k = n // 2
    log_comb = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    log_prob = log_comb + k * log(p) + (n - k) * log(1 - p)
    exact = exp(log_prob)

    # Stirling approximation (for p=0.5)
    approx = sqrt(2 / (pi * n))

    return exact, approx


def create_scenarios(n_candidates: int = 10) -> list[tuple[str, np.ndarray]]:
    """
    Create the voting scenarios to simulate.

    Candidates are ordered: [Douglass, Truth, others...]
    """
    scenarios = []

    # A) Equal support: everyone at 10%
    probs_equal = np.ones(n_candidates) / n_candidates
    scenarios.append(("A: Equal (all 10%)", probs_equal))

    # B) Two front-runners equal: D=T=20%, others split 60%
    probs_front = np.array([0.20, 0.20] + [0.60 / 8] * 8)
    scenarios.append(("B: Front-runners (20%/20%)", probs_front))

    # C) Slightly unequal front-runners: 21% vs 19%
    probs_unequal = np.array([0.21, 0.19] + [0.60 / 8] * 8)
    scenarios.append(("C: Unequal (21%/19%)", probs_unequal))

    # D1) Front-runners at 25%/25%
    probs_strong = np.array([0.25, 0.25] + [0.50 / 8] * 8)
    scenarios.append(("D1: Strong leads (25%/25%)", probs_strong))

    # D2) Front-runners at 15%/15%
    probs_weak = np.array([0.15, 0.15] + [0.70 / 8] * 8)
    scenarios.append(("D2: Weak leads (15%/15%)", probs_weak))

    return scenarios


def run_voter_sweep(
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """
    Sweep number of voters for different scenarios.

    Returns voter counts and dict of scenario -> probabilities.
    """
    voter_counts = np.array([500, 1000, 1500, 2000, 2500, 3000, 4000, 5000])

    scenarios = {
        "Equal chance for 10 candidates": np.ones(10) / 10,
        "Two front-runners tied": np.array([0.20, 0.20] + [0.60 / 8] * 8),
        "One candidate has 10-point lead": np.array([0.25, 0.15] + [0.60 / 8] * 8),
    }

    results = {name: [] for name in scenarios}

    for n_voters in voter_counts:
        for name, probs in scenarios.items():
            p_tie, _ = simulate_scenario(probs, n_voters, n_trials, rng=rng)
            results[name].append(p_tie)

    return voter_counts, {k: np.array(v) for k, v in results.items()}


def create_plot(voter_counts, results, output_path="tie_sensitivity.png"):
    """Create a plot showing how tie probability varies with number of voters."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']

    for (name, probs), color, marker in zip(results.items(), colors, markers):
        ax.plot(voter_counts, probs * 100, f'-{marker}',
                color=color, markersize=6, linewidth=2, label=name)

    ax.set_xlabel('Number of Votes', fontsize=12)
    ax.set_ylabel('Chance of Tie (%)', fontsize=12)
    ax.set_title('Probability of a First-Place Tie\n(10 candidates)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5500)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate first-place tie probabilities in multi-candidate voting"
    )
    parser.add_argument(
        "--trials", type=int, default=200_000,
        help="Number of Monte Carlo trials per scenario (default: 200000)"
    )
    parser.add_argument(
        "--voters", type=int, default=2000,
        help="Number of voters (default: 2000)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate sensitivity plot (requires matplotlib)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("  FIRST-PLACE TIE SIMULATION")
    print("  Douglass-Truth Library Naming Vote (Seattle, 1975)")
    print("=" * 65)
    print(f"\nSimulating {args.voters:,} voters, {args.trials:,} trials per scenario")
    print(f"Random seed: {args.seed}")

    # Sanity check: binomial approximation
    print("\n" + "-" * 65)
    print("SANITY CHECK: Two-candidate race (p = 0.5 each)")
    print("-" * 65)
    exact, approx = binomial_tie_probability(args.voters)
    print(f"  N = {args.voters}")
    print(f"  Exact P(tie)    = {exact:.4f}  ({exact*100:.2f}%)")
    print(f"  Stirling approx = {approx:.4f}  ({approx*100:.2f}%)")
    print(f"  Key insight: P(tie) ~ 1/sqrt(N), not 1/N")

    # Run scenarios
    print("\n" + "-" * 65)
    print("RESULTS: 10-candidate race")
    print("-" * 65)

    scenarios = create_scenarios()
    results = []

    for name, probs in scenarios:
        p_tie, se = simulate_scenario(probs, args.voters, args.trials, rng=rng)
        results.append(ScenarioResult(name, p_tie, se))

    # Print results table
    print(f"\n{'Scenario':<32} {'P(tie)':<14} {'Odds':<15}")
    print("-" * 65)
    for r in results:
        odds = f"1 in {1/r.p_tie:,.0f}" if r.p_tie > 0 else "N/A"
        print(f"{r.name:<32} {r.p_tie*100:>5.2f}% ± {r.se*100:.2f}%   {odds}")

    # Key interpretation
    print("\n" + "-" * 65)
    print("INTERPRETATION")
    print("-" * 65)

    scen_b = results[1]  # Front-runners at 20%/20%
    scen_c = results[2]  # 21%/19%

    print(f"""
Scenario B (two front-runners at 20% each) is plausible for the
Douglass-Truth vote. Result: a tie for first happens about
{scen_b.p_tie*100:.1f}% of the time, or roughly 1 in {1/scen_b.p_tie:.0f} elections.

Even a small gap matters: at 21%/19% (Scenario C), the probability
drops to {scen_c.p_tie*100:.2f}%, or about 1 in {1/scen_c.p_tie:.0f}.
""")

    # Voter count sweep plot
    if args.plot:
        print("-" * 65)
        print("GENERATING PLOT: Tie probability vs number of voters...")
        print("-" * 65)
        voter_counts, sweep_results = run_voter_sweep(args.trials // 2, rng)
        create_plot(voter_counts, sweep_results)

    print("=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    print(f"""
A first-place tie in a 2,000-voter, 10-candidate race is unlikely
but not miraculous. With two equally popular front-runners, it
happens roughly once every {1/scen_b.p_tie:.0f} elections.

The Douglass-Truth tie was rare—but democracy occasionally
delivers such beautiful coincidences.
""")


if __name__ == "__main__":
    main()
