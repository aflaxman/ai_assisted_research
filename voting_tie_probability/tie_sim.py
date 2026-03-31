#!/usr/bin/env python3
"""
Monte Carlo simulation: probability of a tie for first place in multi-candidate voting.

Based on: https://gist.github.com/aflaxman/65659878cdac12cb3991fc91b686671d

Context: The Douglass-Truth Library naming vote (Seattle, 1975) produced a
first-place tie between Frederick Douglass and Sojourner Truth among ~2,000
voters choosing from 10 candidates. How unlikely was that?

Usage:
    python tie_sim.py [--trials N] [--plot]
"""

import argparse
import numpy as np
import pandas as pd


def simulate_election(n_votes, p):
    """
    Simulate an election and report whether there is a tie for first place.

    Parameters
    ----------
    n_votes : int
        Number of independent votes to draw.
    p : array-like of float
        Voting probabilities for each candidate. Must be 1-D, nonnegative,
        sum to 1, and have length >= 2.

    Returns
    -------
    bool
        True if at least two candidates are tied for the highest vote total;
        False otherwise.
    """
    votes = np.random.choice(range(len(p)), p=p, size=n_votes)
    vote_tallys = pd.Series(votes).value_counts()

    if vote_tallys.iloc[0] == vote_tallys.iloc[1]:
        return True
    else:
        return False


def run_scenario(n_votes, probabilities, n_replications):
    """Run many elections and return the fraction that resulted in a tie."""
    n_ties = 0
    for _ in range(n_replications):
        n_ties += simulate_election(n_votes, probabilities)
    return n_ties / n_replications


def main():
    parser = argparse.ArgumentParser(
        description="Simulate first-place tie probabilities in multi-candidate voting"
    )
    parser.add_argument(
        "--trials", type=int, default=100_000,
        help="Number of Monte Carlo trials per scenario (default: 100000)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plot (requires matplotlib)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 65)
    print("  FIRST-PLACE TIE SIMULATION")
    print("  Douglass-Truth Library Naming Vote (Seattle, 1975)")
    print("=" * 65)
    print(f"\nRunning {args.trials:,} replications per scenario")

    # Define scenarios
    probability_scenarios = {
        'Equal chance for 10 candidates':
            [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
        'Two front-runners tied (20%/20%)':
            [.20, .20, .075, .075, .075, .075, .075, .075, .075, .075],
        'Two main candidates tied (49%/49%)':
            [.49, .49, .02],
        'One candidate has a 10-point lead':
            [.55, .45],
    }

    # Run single-N scenarios at N=2000
    print("\n" + "-" * 65)
    print("RESULTS at N = 2,000 voters")
    print("-" * 65)

    n_votes = 2000
    print(f"\n{'Scenario':<40} {'P(tie)':<12} {'Odds'}")
    print("-" * 65)

    for name, probs in probability_scenarios.items():
        assert np.allclose(sum(probs), 1), f"Probabilities must sum to 1: {probs}"
        p_tie = run_scenario(n_votes, probs, args.trials)
        odds = f"1 in {1/p_tie:,.0f}" if p_tie > 0 else "N/A"
        print(f"{name:<40} {p_tie*100:>5.2f}%      {odds}")

    # Voter count sweep for plot
    if args.plot:
        import matplotlib.pyplot as plt

        print("\n" + "-" * 65)
        print("GENERATING PLOT: Tie probability vs number of voters...")
        print("-" * 65)

        vote_sizes = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

        plot_scenarios = {
            'Equal chance for 10 candidates':
                [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
            'Two front-runners tied':
                [.20, .20, .075, .075, .075, .075, .075, .075, .075, .075],
            'One candidate has 10-point lead':
                [.25, .15, .075, .075, .075, .075, .075, .075, .075, .075],
        }

        results = {}
        for scenario, probs in plot_scenarios.items():
            print(f"  Running: {scenario}...")
            pr_tie = {}
            for n_votes in vote_sizes:
                pr_tie[n_votes] = run_scenario(n_votes, probs, args.trials // 2)
            results[scenario] = pd.Series(pr_tie)

        results_df = pd.DataFrame(results)

        # Create plot
        fig, ax = plt.subplots(figsize=(9, 5))
        (100 * results_df).plot(marker='s', ax=ax, linewidth=2, markersize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Chance of Tie (%)', fontsize=12)
        ax.set_xlabel('Number of Votes', fontsize=12)
        ax.set_title('Probability of a First-Place Tie\n(10 candidates)', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(0, 5500)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig('tie_sensitivity.png', dpi=150)
        print(f"\nPlot saved to: tie_sensitivity.png")
        plt.close()

    print("\n" + "=" * 65)
    print("CONCLUSION")
    print("=" * 65)
    print("""
A first-place tie in a 2,000-voter, 10-candidate race is unlikely
but not miraculous. With two equally popular front-runners at ~20%
each, it happens roughly once every 50-100 elections.

The Douglass-Truth tie was rare—but democracy occasionally
delivers such beautiful coincidences.
""")


if __name__ == "__main__":
    main()
