#!/usr/bin/env python3
"""
Monte Carlo simulation of tie probabilities in multi-candidate voting.

Context: The Douglass-Truth Library naming vote (Seattle, 1975) produced a
tie between Frederick Douglass and Sojourner Truth among ~2,000 voters
choosing from 10 candidates. How unlikely was that?

Usage:
    python tie_sim.py [--trials N] [--voters N] [--plot]
"""

import argparse
import numpy as np
from typing import NamedTuple


class ScenarioResult(NamedTuple):
    """Results from simulating a voting scenario."""
    name: str
    p_any_tie: float  # P(X_D == X_T), regardless of winning
    p_first_tie: float  # P(X_D == X_T == max and only those two share max)
    se_any: float  # standard error for p_any_tie
    se_first: float  # standard error for p_first_tie


def simulate_scenario(
    probs: np.ndarray,
    n_voters: int,
    n_trials: int,
    idx_d: int = 0,  # index of Douglass in probs array
    idx_t: int = 1,  # index of Truth in probs array
    rng: np.random.Generator = None,
) -> tuple[float, float, float, float]:
    """
    Simulate votes and compute tie probabilities.

    Args:
        probs: Array of vote share probabilities for each candidate (must sum to 1)
        n_voters: Number of voters in each trial
        n_trials: Number of Monte Carlo trials
        idx_d, idx_t: Indices for the two candidates of interest
        rng: Random number generator

    Returns:
        (p_any_tie, p_first_tie, se_any, se_first)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw all trials at once: shape (n_trials, n_candidates)
    votes = rng.multinomial(n_voters, probs, size=n_trials)

    # Extract counts for Douglass and Truth
    x_d = votes[:, idx_d]
    x_t = votes[:, idx_t]

    # Condition 1: Any tie between D and T
    any_tie = (x_d == x_t)

    # Condition 2: Two-way tie for first place
    # Both must equal the max, and no other candidate shares that max
    max_votes = votes.max(axis=1)
    d_at_max = (x_d == max_votes)
    t_at_max = (x_t == max_votes)

    # Count how many candidates are at the max
    at_max_count = (votes == max_votes[:, np.newaxis]).sum(axis=1)

    # Two-way tie for first: D and T both at max, exactly 2 candidates at max
    first_tie = d_at_max & t_at_max & (at_max_count == 2)

    # Compute probabilities and standard errors
    p_any = any_tie.mean()
    p_first = first_tie.mean()

    # Standard error: sqrt(p(1-p)/n)
    se_any = np.sqrt(p_any * (1 - p_any) / n_trials)
    se_first = np.sqrt(p_first * (1 - p_first) / n_trials)

    return p_any, p_first, se_any, se_first


def binomial_tie_probability(n: int, p: float = 0.5) -> tuple[float, float]:
    """
    Exact probability of a tie in a 2-candidate race with n voters.

    For fair coin (p=0.5): P(tie) = C(n, n/2) * (0.5)^n

    Uses Stirling approximation for large n:
    P(tie) ≈ 1 / sqrt(pi * n / 2) ≈ sqrt(2 / (pi * n))
    """
    from math import sqrt, pi, lgamma, log, exp

    if n % 2 == 1:
        return 0.0, 0.0  # Odd voters can't tie

    # Use log-space calculation to avoid overflow
    # log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
    # Using lgamma: log(n!) = lgamma(n+1)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep front-runner gap from 0% to 10% difference.

    Base: p_D + p_T = 0.40 total, others split 0.60
    Vary: p_D from 0.20 to 0.30, p_T = 0.40 - p_D
    """
    gaps = np.linspace(0, 0.10, 21)  # 0% to 10% gap in 0.5% increments
    p_any_ties = []
    p_first_ties = []

    for gap in gaps:
        p_d = 0.20 + gap / 2
        p_t = 0.20 - gap / 2
        probs = np.array([p_d, p_t] + [0.60 / 8] * 8)

        p_any, p_first, _, _ = simulate_scenario(
            probs, n_voters, n_trials, rng=rng
        )
        p_any_ties.append(p_any)
        p_first_ties.append(p_first)

    return gaps, np.array(p_any_ties), np.array(p_first_ties)


def create_plot(gaps, p_any, p_first, n_voters, output_path="tie_sensitivity.png"):
    """Create a plot showing how tie probability varies with front-runner gap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(gaps * 100, p_any * 100, 'b-o', markersize=4, label='Any tie (D = T)')
    ax.plot(gaps * 100, p_first * 100, 'r-s', markersize=4, label='Tie for 1st place')

    ax.set_xlabel('Gap between front-runners (percentage points)', fontsize=11)
    ax.set_ylabel('Probability of tie (%)', fontsize=11)
    ax.set_title(f'Tie Probability vs. Front-Runner Gap\n(N = {n_voters:,} voters, 10 candidates)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(bottom=0)

    # Add annotation
    ax.annotate(
        f'At 0% gap:\nAny tie ≈ {p_any[0]*100:.2f}%\n1st place tie ≈ {p_first[0]*100:.2f}%',
        xy=(0, p_any[0] * 100),
        xytext=(2, p_any[0] * 100 * 0.7),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate tie probabilities in multi-candidate voting"
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

    print("=" * 70)
    print("TIE PROBABILITY SIMULATION: Douglass-Truth Library Naming Vote")
    print("=" * 70)
    print(f"\nParameters: {args.voters:,} voters, {args.trials:,} trials per scenario")
    print(f"Random seed: {args.seed}")

    # Sanity check: binomial approximation
    print("\n" + "-" * 70)
    print("SANITY CHECK: Two-candidate binomial model (p = 0.5)")
    print("-" * 70)
    exact, approx = binomial_tie_probability(args.voters)
    print(f"N = {args.voters}")
    print(f"Exact P(tie)   = {exact:.6f} ({exact*100:.4f}%)")
    print(f"Stirling approx = {approx:.6f} ({approx*100:.4f}%)")
    print(f"Order of magnitude: O(1/sqrt(N)) ≈ 1/sqrt({args.voters}) ≈ {1/np.sqrt(args.voters):.4f}")

    # Run scenarios
    print("\n" + "-" * 70)
    print("SCENARIO RESULTS: 10-candidate multinomial model")
    print("-" * 70)

    scenarios = create_scenarios()
    results = []

    for name, probs in scenarios:
        p_any, p_first, se_any, se_first = simulate_scenario(
            probs, args.voters, args.trials, rng=rng
        )
        results.append(ScenarioResult(name, p_any, p_first, se_any, se_first))

    # Print results table
    print(f"\n{'Scenario':<30} {'P(D=T)':<12} {'P(1st tie)':<12} {'1 in X':<10}")
    print("-" * 70)
    for r in results:
        odds_any = f"1 in {1/r.p_any_tie:,.0f}" if r.p_any_tie > 0 else "N/A"
        odds_first = f"1 in {1/r.p_first_tie:,.0f}" if r.p_first_tie > 0 else "N/A"
        print(f"{r.name:<30} {r.p_any_tie*100:>6.3f}% ± {r.se_any*100:.3f}  "
              f"{r.p_first_tie*100:>6.4f}% ± {r.se_first*100:.4f}")

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    # Find scenario B for interpretation
    scen_b = results[1]  # Front-runners at 20%/20%
    print(f"\nWith two front-runners at 20% each (Scenario B):")
    print(f"  - Any tie between D & T:     {scen_b.p_any_tie*100:.3f}% (about 1 in {1/scen_b.p_any_tie:,.0f})")
    print(f"  - Tie for first place:       {scen_b.p_first_tie*100:.4f}% (about 1 in {1/scen_b.p_first_tie:,.0f})")

    scen_c = results[2]  # 21%/19%
    print(f"\nWith slight imbalance 21%/19% (Scenario C):")
    print(f"  - Any tie between D & T:     {scen_c.p_any_tie*100:.3f}% (about 1 in {1/scen_c.p_any_tie:,.0f})")
    print(f"  - Tie for first place:       {scen_c.p_first_tie*100:.4f}% (about 1 in {1/scen_c.p_first_tie:,.0f})")

    # Sensitivity sweep
    if args.plot:
        print("\n" + "-" * 70)
        print("GENERATING SENSITIVITY PLOT...")
        print("-" * 70)
        gaps, p_any_sweep, p_first_sweep = run_sensitivity_sweep(
            args.voters, args.trials // 2, rng  # Fewer trials for sweep
        )
        create_plot(gaps, p_any_sweep, p_first_sweep, args.voters)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
A tie in a 2000-voter, 10-candidate race is unlikely but far from impossible.
With equal front-runners at ~20% each, a tie for first occurs roughly once
every {1/scen_b.p_first_tie:,.0f} elections. Not astronomical odds at all.

The Douglass-Truth tie was genuinely rare—but democracy occasionally
delivers such serendipitous outcomes.
""")


if __name__ == "__main__":
    main()
