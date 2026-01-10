"""
Invasion Percolation Simulation on a 2D Grid
=============================================

A filled region starts at the center of a grid and grows by randomly
selecting neighboring cells to fill next, until it reaches an edge.

Based on Greg Wilson's testing challenge:
https://third-bit.com/2025/04/20/a-testing-question/#invasion-percolation
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


def fill_grid_percolation(grid):
    """
    Fill grid using invasion percolation starting from center.

    The algorithm:
    1. Start with center cell filled
    2. Find all neighbors of filled cells (the perimeter)
    3. Randomly select one neighbor to fill
    4. Repeat until a boundary cell is filled

    Returns:
        tuple: (num_filled, final_x, final_y) where:
            - num_filled: Number of cells filled
            - final_x, final_y: Position of the boundary cell that was filled
    """
    center = grid.size // 2
    size_1 = grid.size - 1

    # Start by filling the center
    grid[center, center] = 1
    filled = {(center, center)}
    num_filled = 1

    # Track the perimeter (unfilled neighbors of filled cells)
    perimeter = set()

    def add_neighbors(x, y):
        """Add unfilled neighbors of (x, y) to perimeter."""
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.size and 0 <= ny < grid.size:
                if (nx, ny) not in filled and (nx, ny) not in perimeter:
                    perimeter.add((nx, ny))

    # Initialize perimeter with neighbors of center
    add_neighbors(center, center)

    # Keep filling until we hit a boundary
    while perimeter:
        # Randomly select a cell from the perimeter
        next_cell = random.choice(list(perimeter))
        perimeter.remove(next_cell)

        x, y = next_cell
        grid[x, y] = 1
        filled.add(next_cell)
        num_filled += 1

        # Check if we hit the boundary
        if x == 0 or y == 0 or x == size_1 or y == size_1:
            return num_filled, x, y

        # Add neighbors of newly filled cell to perimeter
        add_neighbors(x, y)

    # Should never reach here if grid is bounded
    raise RuntimeError("Percolation failed to reach boundary")


# ==============================================================================
# Command Line Interface
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Invasion percolation simulation"
    )
    parser.add_argument("--seed", type=int, required=True, help="RNG seed")
    parser.add_argument("--size", type=int, required=True, help="Grid size")

    args = parser.parse_args()

    random.seed(args.seed)
    grid = Grid(args.size)

    num_filled, final_x, final_y = fill_grid_percolation(grid)

    print(f"INVASION PERCOLATION: Filled {num_filled} cells")
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
