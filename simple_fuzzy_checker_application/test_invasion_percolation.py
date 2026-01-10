"""
Testing Invasion Percolation with Fuzzy Checking
=================================================

This test suite demonstrates fuzzy checking applied to invasion percolation.

Question: For an unbiased invasion percolation process, should we expect
exits at each edge with equal probability (25% each)?

Let's find out!
"""

import random
from collections import Counter

import pytest

from vivarium_testing_utils import FuzzyChecker
from invasion_percolation import Grid, fill_grid_percolation


# =============================================================================
# Test Fixtures (Pytest Setup)
# =============================================================================


@pytest.fixture(scope="session")
def output_directory(tmp_path_factory):
    """Create a temporary directory for diagnostic output."""
    return tmp_path_factory.mktemp("fuzzy_checker_diagnostics")


@pytest.fixture(scope="session")
def fuzzy_checker(output_directory):
    """
    Create a FuzzyChecker instance that persists across all tests.

    The session scope means this fixture is created once and shared by all
    tests. At the end, it automatically saves diagnostics to CSV.
    """
    checker = FuzzyChecker()
    yield checker
    checker.save_diagnostic_output(output_directory)
    print(f"\nFuzzyChecker diagnostics saved to: {output_directory}")


# =============================================================================
# Tests for Invasion Percolation
# =============================================================================


def test_percolation_exit_edges(fuzzy_checker):
    """
    Test if invasion percolation exits at each edge with equal probability.

    For an unbiased invasion percolation, we might expect about 25% of
    simulations to exit at each edge:
    - x == 0 (left edge)
    - x == size-1 (right edge)
    - y == 0 (top edge)
    - y == size-1 (bottom edge)

    But is this actually true? Let's test and see!
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(1000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid_percolation(grid)

        # Classify which edge the percolation reached
        if final_x == 0:
            edge_counts["left"] += 1
        elif final_x == size_1:
            edge_counts["right"] += 1
        elif final_y == 0:
            edge_counts["top"] += 1
        else:  # final_y == size_1
            edge_counts["bottom"] += 1

    print(f"\nInvasion percolation - Exit edge counts: {dict(edge_counts)}")
    print(f"Proportions: {[(k, f'{v/num_runs:.3f}') for k, v in edge_counts.items()]}")

    # Test hypothesis: each edge should be ~25%
    for edge in ["left", "right", "top", "bottom"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=edge_counts[edge],
            observed_denominator=num_runs,
            target_proportion=(0.23, 0.27),  # 25% ± 2%
            name=f"percolation_{edge}_exit_proportion",
        )


def test_percolation_horizontal_symmetry(fuzzy_checker):
    """
    Test if left/right exits are balanced.

    Among processes that exit horizontally (left or right edge),
    do we expect about 50% on each side?
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(2000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid_percolation(grid)

        if final_x == 0:
            edge_counts["left"] += 1
        elif final_x == size_1:
            edge_counts["right"] += 1

    horizontal_exits = edge_counts["left"] + edge_counts["right"]

    print(f"\nInvasion percolation - Horizontal exits: {horizontal_exits}")
    print(f"Left: {edge_counts['left']} ({edge_counts['left']/horizontal_exits:.3f})")
    print(f"Right: {edge_counts['right']} ({edge_counts['right']/horizontal_exits:.3f})")

    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=edge_counts["left"],
        observed_denominator=horizontal_exits,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="percolation_left_right_symmetry",
    )


def test_percolation_vertical_symmetry(fuzzy_checker):
    """
    Test if top/bottom exits are balanced.

    Among processes that exit vertically (top or bottom edge),
    do we expect about 50% on each side?
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(3000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid_percolation(grid)

        if final_y == 0:
            edge_counts["top"] += 1
        elif final_y == size_1:
            edge_counts["bottom"] += 1

    vertical_exits = edge_counts["top"] + edge_counts["bottom"]

    print(f"\nInvasion percolation - Vertical exits: {vertical_exits}")
    print(f"Top: {edge_counts['top']} ({edge_counts['top']/vertical_exits:.3f})")
    print(f"Bottom: {edge_counts['bottom']} ({edge_counts['bottom']/vertical_exits:.3f})")

    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=edge_counts["top"],
        observed_denominator=vertical_exits,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="percolation_top_bottom_symmetry",
    )


# =============================================================================
# Running the Tests
# =============================================================================

if __name__ == "__main__":
    """
    Run with pytest to execute the test suite:
        pytest test_invasion_percolation.py -v

    This will reveal whether invasion percolation actually has uniform
    exit probabilities at each edge!
    """
    print("Run with pytest to execute the test suite:")
    print("  pytest test_invasion_percolation.py -v")
