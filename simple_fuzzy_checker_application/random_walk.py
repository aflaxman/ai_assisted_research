"""
Random Walk Simulation on a 2D Grid
====================================

This module demonstrates a classic spatial simulation: a random walker starting
at the center of a grid and moving randomly until it reaches an edge.

Following the pattern from Greg Wilson's blog post, we use a Grid class and
a fill_grid function that takes moves as a parameter, making it easy to test
with both correct and buggy move sets.
"""

import argparse
import csv
import io
import random
from collections import Counter


class Grid:
    """Store a grid of numbers representing visit counts."""

    def __init__(self, size):
        """Construct empty grid of given size."""
        assert size > 0, f"Grid size must be positive not {size}"
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]

    def __getitem__(self, key):
        """Get grid element at position (x, y)."""
        x, y = key
        return self.grid[x][y]

    def __setitem__(self, key, value):
        """Set grid element at position (x, y)."""
        x, y = key
        self.grid[x][y] = value

    def __str__(self):
        """Convert grid to CSV string for easy visualization."""
        output = io.StringIO()
        csv.writer(output).writerows(self.grid)
        return output.getvalue()


def fill_grid(grid, moves):
    """
    Fill grid with a random walk starting from center.

    This function follows Greg Wilson's pattern - it takes the moves list
    as a parameter so we can test with different move sets (correct or buggy).

    Args:
        grid: Grid object to fill
        moves: List of [dx, dy] moves, e.g., [[-1, 0], [1, 0], [0, -1], [0, 1]]

    Returns:
        tuple: (num_steps, direction_counts) where:
            - num_steps: Number of steps taken before reaching boundary
            - direction_counts: Counter with keys 'left', 'right', 'up', 'down'
    """
    center = grid.size // 2
    size_1 = grid.size - 1
    x, y = center, center
    num = 0

    # Track which direction each move went
    direction_counts = Counter()

    # Walk until we hit an edge
    while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
        grid[x, y] += 1
        num += 1

        # Pick random direction
        m = random.choice(moves)
        x += m[0]
        y += m[1]

        # OBSERVE: Track which direction we moved
        # This is the key observation code that tests will use!
        if m == [-1, 0]:
            direction_counts["left"] += 1
        elif m == [1, 0]:
            direction_counts["right"] += 1
        elif m == [0, -1]:
            direction_counts["up"] += 1
        elif m == [0, 1]:
            direction_counts["down"] += 1

    return num, direction_counts


# Standard move sets for testing
CORRECT_MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down
BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]]   # left, right, up, up (!)


# ==============================================================================
# Demo Visualization
# ==============================================================================


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


def run_demo(size, num_runs, seed_start=1000):
    """
    Run both correct and buggy versions and compare their directional distributions.

    This demonstrates the bug visually by running many simulations with each
    move set and displaying the results as ASCII bar charts.
    """
    print(f"\nRunning {num_runs} random walks on a {size}x{size} grid...\n")

    # Track correct version
    correct_total = Counter()
    grid = Grid(size)

    for i in range(num_runs):
        random.seed(seed_start + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, direction_counts = fill_grid(grid, CORRECT_MOVES)
        correct_total.update(direction_counts)

    # Track buggy version
    buggy_total = Counter()

    for i in range(num_runs):
        random.seed(seed_start + i)  # Same seeds for fair comparison
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, direction_counts = fill_grid(grid, BUGGY_MOVES)
        buggy_total.update(direction_counts)

    # Display results
    visualize_direction_distribution(
        correct_total, sum(correct_total.values()),
        "✓ CORRECT VERSION - Unbiased Random Walk"
    )

    visualize_direction_distribution(
        buggy_total, sum(buggy_total.values()),
        "✗ BUGGY VERSION - Directional Bias!"
    )

    # Highlight the smoking gun
    print("🔍 THE SMOKING GUN:")
    print("=" * 60)
    print(f"Down moves in correct version: {correct_total['down']:,}")
    print(f"Down moves in buggy version:   {buggy_total.get('down', 0):,}")
    print(f"\nUp moves in correct version:   {correct_total['up']:,}")
    print(f"Up moves in buggy version:     {buggy_total['up']:,}")
    print("\nThe bug: walker can move up but NEVER down! 🐛")
    print("=" * 60)


# ==============================================================================
# Command Line Interface
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Random walk simulation and visualization"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run a single simulation
    run_parser = subparsers.add_parser("run", help="Run a single simulation")
    run_parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    run_parser.add_argument("--size", type=int, required=True, help="grid size")
    run_parser.add_argument(
        "--buggy",
        action="store_true",
        help="Use buggy version with directional bias",
    )

    # Visualize the bug
    demo_parser = subparsers.add_parser("demo", help="Visualize the bug")
    demo_parser.add_argument(
        "--size", type=int, default=11, help="Grid size (default: 11)"
    )
    demo_parser.add_argument(
        "--runs", type=int, default=100, help="Number of walks (default: 100)"
    )
    demo_parser.add_argument(
        "--seed", type=int, default=1000, help="Starting random seed (default: 1000)"
    )

    args = parser.parse_args()

    if args.command == "run":
        random.seed(args.seed)
        grid = Grid(args.size)

        moves = BUGGY_MOVES if args.buggy else CORRECT_MOVES
        steps, direction_counts = fill_grid(grid, moves)

        version = "BUGGY" if args.buggy else "CORRECT"
        print(f"{version} VERSION: Took {steps} steps")
        print(f"Direction counts: {dict(direction_counts)}")
        print(grid)

    elif args.command == "demo":
        run_demo(args.size, args.runs, args.seed)
        print("\n💡 TIP: Run the fuzzy checker tests to see how Bayesian statistics")
        print("   can catch this bug automatically:")
        print("   pytest test_random_walk.py -v\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
