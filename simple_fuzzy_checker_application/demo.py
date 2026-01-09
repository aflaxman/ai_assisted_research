#!/usr/bin/env python
"""
Interactive Demo: Visualize the Bug
====================================

This script runs side-by-side comparisons of the correct and buggy versions,
showing you the directional bias visually.

Usage:
    python demo.py --size 11 --runs 100

What to look for:
    - Correct version: Roughly equal moves in all directions
    - Buggy version: Lots of "up" moves, ZERO "down" moves
"""

import argparse
from collections import Counter
import random
from random_walk import Grid, fill_grid, fill_grid_buggy


def visualize_direction_distribution(counts, total, title):
    """Create a simple ASCII bar chart of direction proportions."""
    print(f"\n{title}")
    print("=" * 60)

    for direction in ["left", "right", "up", "down"]:
        count = counts.get(direction, 0)
        proportion = count / total if total > 0 else 0
        bar_length = int(proportion * 50)  # Scale to 50 chars max
        bar = "█" * bar_length

        print(f"{direction:>6}: {bar} {proportion:.1%} ({count:,} moves)")

    print(f"Total moves: {total:,}\n")


def run_comparison(size, num_runs, seed_start=1000):
    """Run both versions and compare their directional distributions."""

    print(f"\nRunning {num_runs} random walks on a {size}x{size} grid...\n")

    # Track correct version
    correct_counts = Counter()
    grid = Grid(size)

    for i in range(num_runs):
        random.seed(seed_start + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # Correct

        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            if move == [-1, 0]:
                correct_counts["left"] += 1
            elif move == [1, 0]:
                correct_counts["right"] += 1
            elif move == [0, -1]:
                correct_counts["up"] += 1
            elif move == [0, 1]:
                correct_counts["down"] += 1

    # Track buggy version
    buggy_counts = Counter()

    for i in range(num_runs):
        random.seed(seed_start + i)  # Same seeds for fair comparison
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # BUGGY!

        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            if move == [-1, 0]:
                buggy_counts["left"] += 1
            elif move == [1, 0]:
                buggy_counts["right"] += 1
            elif move == [0, -1]:
                buggy_counts["up"] += 1

    # Display results
    correct_total = sum(correct_counts.values())
    buggy_total = sum(buggy_counts.values())

    visualize_direction_distribution(
        correct_counts, correct_total, "✓ CORRECT VERSION - Unbiased Random Walk"
    )

    visualize_direction_distribution(
        buggy_counts, buggy_total, "✗ BUGGY VERSION - Directional Bias!"
    )

    # Highlight the smoking gun
    print("🔍 THE SMOKING GUN:")
    print("=" * 60)
    print(f"Down moves in correct version: {correct_counts['down']:,}")
    print(f"Down moves in buggy version:   {buggy_counts.get('down', 0):,}")
    print(f"\nUp moves in correct version:   {correct_counts['up']:,}")
    print(f"Up moves in buggy version:     {buggy_counts['up']:,}")
    print("\nThe bug: walker can move up but NEVER down! 🐛")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the directional bias bug in the random walk simulation"
    )
    parser.add_argument(
        "--size", type=int, default=11, help="Grid size (default: 11)"
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of walks to simulate (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="Starting random seed (default: 1000)"
    )

    args = parser.parse_args()

    run_comparison(args.size, args.runs, args.seed)

    print("\n💡 TIP: Run the fuzzy checker tests to see how Bayesian statistics")
    print("   can catch this bug automatically:")
    print("   pytest test_random_walk.py -v\n")


if __name__ == "__main__":
    main()
