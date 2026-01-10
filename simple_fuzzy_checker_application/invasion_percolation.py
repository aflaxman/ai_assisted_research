"""
Invasion Percolation Simulation on a 2D Grid
=============================================

A filled region starts at the center of a grid and grows by selecting
the neighboring cell with minimum random value to fill next, until it
reaches an edge.

Based on Greg Wilson's testing challenge:
https://third-bit.com/2025/04/20/a-testing-question/#invasion-percolation

The bug: when there are ties (multiple cells with same minimum value),
the buggy version picks the first one encountered in the scan (x=0 to size,
y=0 to size), creating a bias toward (0,0).
"""

import argparse
import csv
import io
import random


class Grid:
    """Store a grid of numbers representing fill status."""

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


def fill_grid_percolation(grid, buggy=False):
    """
    Fill grid using invasion percolation starting from center.

    Algorithm:
    1. Assign random value to each cell
    2. Mark center cell as filled
    3. Find unfilled neighbors of filled cells
    4. Select neighbor with minimum random value
    5. Mark it as filled
    6. Repeat until boundary is reached

    The bug: when multiple neighbors have the same minimum value,
    buggy version picks first one in scan order (bias toward 0,0),
    correct version randomly picks among ties.

    Args:
        grid: Grid object to fill
        buggy: If True, use buggy tie-breaking (bias toward 0,0)

    Returns:
        tuple: (num_filled, final_x, final_y) where:
            - num_filled: Number of cells filled
            - final_x, final_y: Position of the boundary cell that was filled
    """
    # Assign random values to all cells
    values = [[random.random() for _ in range(grid.size)] for _ in range(grid.size)]

    center = grid.size // 2
    size_1 = grid.size - 1

    # Start by marking the center as filled
    grid[center, center] = 1
    num_filled = 1

    # Keep filling until we hit a boundary
    while True:
        # Find all unfilled cells adjacent to filled cells
        candidates = []
        min_val = float('inf')

        for x in range(grid.size):
            for y in range(grid.size):
                # Skip if already filled
                if grid[x, y] == 1:
                    continue

                # Check if adjacent to a filled cell
                is_neighbor = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid.size and 0 <= ny < grid.size:
                        if grid[nx, ny] == 1:
                            is_neighbor = True
                            break

                if is_neighbor:
                    val = values[x][y]
                    if buggy:
                        # Buggy: keep first encountered minimum (bias toward 0,0)
                        if val < min_val:
                            min_val = val
                            candidates = [(x, y)]
                        # Note: when val == min_val, we DON'T add it (keeps first)
                    else:
                        # Correct: collect all ties
                        if val < min_val:
                            min_val = val
                            candidates = [(x, y)]
                        elif val == min_val:
                            candidates.append((x, y))

        if not candidates:
            raise RuntimeError("No candidates found - percolation failed")

        # Select next cell
        if buggy:
            # Buggy version: candidates already contains only first encountered
            next_x, next_y = candidates[0]
        else:
            # Correct version: randomly select among tied candidates
            next_x, next_y = random.choice(candidates)

        # Fill the selected cell
        grid[next_x, next_y] = 1
        num_filled += 1

        # Check if we hit the boundary
        if next_x == 0 or next_y == 0 or next_x == size_1 or next_y == size_1:
            return num_filled, next_x, next_y


# ==============================================================================
# Command Line Interface
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Invasion percolation simulation"
    )
    parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    parser.add_argument("--size", type=int, required=True, help="Grid size")
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Use buggy tie-breaking (bias toward 0,0)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    grid = Grid(args.size)

    num_filled, final_x, final_y = fill_grid_percolation(grid, buggy=args.buggy)

    version = "BUGGY" if args.buggy else "CORRECT"
    print(f"INVASION PERCOLATION ({version}): Filled {num_filled} cells")
    print(f"Final position: ({final_x}, {final_y})")

    # Determine which edge was hit
    size_1 = args.size - 1
    if final_x == 0:
        edge = "left edge (x=0)"
    elif final_x == size_1:
        edge = f"right edge (x={size_1})"
    elif final_y == 0:
        edge = "top edge (y=0)"
    else:
        edge = f"bottom edge (y={size_1})"
    print(f"Reached boundary at: {edge}")
    print(grid)


if __name__ == "__main__":
    main()
