"""
Random Walk Simulation on a 2D Grid
====================================

A random walker starts at the center of a grid and moves randomly until
it reaches an edge.

The fill_grid function takes moves as a parameter, making it easy to test
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

    Args:
        grid: Grid object to fill
        moves: List of [dx, dy] moves, e.g., [[-1, 0], [1, 0], [0, -1], [0, 1]]

    Returns:
        tuple: (num_steps, final_x, final_y) where:
            - num_steps: Number of steps taken before reaching boundary
            - final_x, final_y: Final position on the edge
    """
    center = grid.size // 2
    size_1 = grid.size - 1
    x, y = center, center
    num_steps = 0

    while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
        grid[x, y] += 1
        num_steps += 1
        m = random.choice(moves)
        x += m[0]
        y += m[1]

    return num_steps, x, y


# Standard move sets for testing
CORRECT_MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down
BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]]   # left, right, up, up (!)


# ==============================================================================
# Command Line Interface
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Random walk simulation"
    )
    parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    parser.add_argument("--size", type=int, required=True, help="Grid size")
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Use buggy version with directional bias",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    grid = Grid(args.size)

    moves = BUGGY_MOVES if args.buggy else CORRECT_MOVES
    steps, final_x, final_y = fill_grid(grid, moves)

    version = "BUGGY" if args.buggy else "CORRECT"
    print(f"{version} VERSION: Took {steps} steps")
    print(f"Final position: ({final_x}, {final_y})")


if __name__ == "__main__":
    main()
