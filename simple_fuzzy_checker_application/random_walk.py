"""
Random Walk Simulation on a 2D Grid
====================================

This module demonstrates a classic spatial simulation: a random walker starting
at the center of a grid and moving randomly until it reaches an edge.

The Challenge
-------------
How do you test stochastic code? Traditional assertions fail because:
1. Random walks produce different output each time
2. Statistical properties only emerge in aggregate
3. Bugs often create subtle biases rather than obvious failures

The Bug
-------
The original implementation has a directional bias bug (see fill_grid_buggy):
    moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # OOPS! Two "up" moves

This should be:
    moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]   # left, right, up, down

The walker can move left, right, and up (twice as likely), but never down.
Traditional unit tests miss this because the code "works"—it runs without errors
and produces plausible-looking output. Only statistical analysis reveals the bias.

Fuzzy Checking to the Rescue
-----------------------------
See test_random_walk.py for how we use Bayesian hypothesis testing to catch
this bug without arbitrary thresholds or p-value fishing.
"""

import argparse
import csv
import io
import random


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


def cmdline_args():
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    parser.add_argument("--size", type=int, required=True, help="grid size")
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Use buggy version with directional bias",
    )
    return parser.parse_args()


def fill_grid_buggy(grid):
    """
    Fill grid with a random walk starting from center (BUGGY VERSION).

    This version has a subtle bug: the moves list contains [0, -1] twice,
    meaning the walker can move up twice as often but never down. This creates
    a directional bias that's hard to spot by eye but shows up in statistics.

    Returns:
        int: Number of steps taken before reaching boundary
    """
    # THE BUG: [0, -1] appears twice, [0, 1] is missing
    moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # left, right, up, up (!)

    center = grid.size // 2
    size_1 = grid.size - 1
    x, y = center, center
    num = 0

    # Walk until we hit an edge (not including edges in the walk)
    while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
        grid[x, y] += 1  # Mark this cell as visited
        num += 1
        m = random.choice(moves)  # Pick random direction
        x += m[0]
        y += m[1]

    return num


def fill_grid(grid):
    """
    Fill grid with a random walk starting from center (CORRECT VERSION).

    The walker starts at the center and takes random steps in one of four
    directions (left, right, up, down) with equal probability. The walk
    continues until it reaches a boundary cell.

    This is an unbiased random walk—each direction has exactly 25% probability.

    Returns:
        int: Number of steps taken before reaching boundary
    """
    # CORRECT: All four orthogonal directions with equal probability
    moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down

    center = grid.size // 2
    size_1 = grid.size - 1
    x, y = center, center
    num = 0

    # Walk until we hit an edge (not including edges in the walk)
    while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
        grid[x, y] += 1  # Mark this cell as visited
        num += 1
        m = random.choice(moves)  # Pick random direction
        x += m[0]
        y += m[1]

    return num


def run_simulation(size, seed, buggy=False):
    """
    Run a single random walk simulation.

    Args:
        size: Grid size (size x size)
        seed: Random seed for reproducibility
        buggy: If True, use buggy version with directional bias

    Returns:
        tuple: (grid, num_steps) where grid is the Grid object and
               num_steps is the walk length
    """
    random.seed(seed)
    grid = Grid(size)

    if buggy:
        num_steps = fill_grid_buggy(grid)
    else:
        num_steps = fill_grid(grid)

    return grid, num_steps


if __name__ == "__main__":
    # Allow standalone execution for manual exploration
    args = cmdline_args()
    random.seed(args.seed)
    grid = Grid(args.size)

    if args.buggy:
        steps = fill_grid_buggy(grid)
        print(f"BUGGY VERSION: Took {steps} steps")
    else:
        steps = fill_grid(grid)
        print(f"CORRECT VERSION: Took {steps} steps")

    print(grid)
