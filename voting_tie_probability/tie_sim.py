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


def run_sensitivity_sweep(
    n_voters: int,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep front-runner gap from 0% to 10% difference.

    Base: p_D + p_T = 0.40 total, others split 0.60
    Vary: p_D from 0.20 to 0.30, p_T = 0.40 - p_D
    """
    gaps = np.linspace(0, 0.10, 21)  # 0% to 10% gap in 0.5% increments
    p_ties = []

    for gap in gaps:
        p_d = 0.20 + gap / 2
        p_t = 0.20 - gap / 2
        probs = np.array([p_d, p_t] + [0.60 / 8] * 8)

        p_tie, _ = simulate_scenario(probs, n_voters, n_trials, rng=rng)
        p_ties.append(p_tie)

    return gaps, np.array(p_ties)


def create_plot(gaps, p_ties, n_voters, output_path="tie_sensitivity.png"):
    """Create a plot showing how tie probability varies with front-runner gap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(gaps * 100, p_ties * 100, 'b-o', markersize=5, linewidth=2)

    ax.set_xlabel('Gap between front-runners (percentage points)', fontsize=12)
    ax.set_ylabel('Probability of first-place tie (%)', fontsize=12)
    ax.set_title(
        f'How Likely Is a Tie for First Place?\n'
        f'(N = {n_voters:,} voters, 10 candidates, two front-runners)',
        fontsize=13
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(bottom=0)

    # Add annotation at zero gap
    ax.annotate(
        f'Equal support:\n~{p_ties[0]*100:.1f}% chance\n(1 in {1/p_ties[0]:.0f})',
        xy=(0, p_ties[0] * 100),
        xytext=(2.5, p_ties[0] * 100 * 0.85),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
    )

    # Add annotation at 2% gap
    idx_2pct = 4  # 2% gap
    ax.annotate(
        f'2-point gap:\n~{p_ties[idx_2pct]*100:.2f}%\n(1 in {1/p_ties[idx_2pct]:.0f})',
        xy=(2, p_ties[idx_2pct] * 100),
        xytext=(4.5, p_ties[idx_2pct] * 100 + 0.3),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
    )

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

    # Sensitivity sweep
    if args.plot:
        print("-" * 65)
        print("GENERATING SENSITIVITY PLOT...")
        print("-" * 65)
        gaps, p_ties = run_sensitivity_sweep(
            args.voters, args.trials // 2, rng
        )
        create_plot(gaps, p_ties, args.voters)

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
