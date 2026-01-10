"""
Fuzzy Checker Tutorial: Testing Stochastic Simulations
=======================================================

This test suite demonstrates Bayesian hypothesis testing (via FuzzyChecker)
to validate statistical properties of random simulations.

The Pattern
-----------
1. Run many simulations using fill_grid(grid, moves)
2. Aggregate the direction_counts returned from each run
3. Use fuzzy_assert_proportion() to validate statistical properties

No p-hacking. No arbitrary thresholds. Rigorous Bayesian reasoning.
"""

import random
from collections import Counter

import pytest

from vivarium_testing_utils import FuzzyChecker
from random_walk import Grid, fill_grid, CORRECT_MOVES, BUGGY_MOVES


# =============================================================================
# Test Fixtures (Pytest Setup)
# =============================================================================


@pytest.fixture(scope="session")
def fuzzy_checker():
    """
    Create a FuzzyChecker instance that persists across all tests.

    The session scope means this fixture is created once and shared by all tests.
    """
    return FuzzyChecker()


# =============================================================================
# Tests for CORRECT Version
# =============================================================================


def test_correct_version_exit_edges(fuzzy_checker):
    """
    Validate that walks exit at each edge with equal probability.

    For an unbiased random walk, we expect about 25% of walks to exit at each edge:
    - x == 0 (left edge)
    - x == size-1 (right edge)
    - y == 0 (top edge)
    - y == size-1 (bottom edge)
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(2000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid(grid, CORRECT_MOVES)

        # Classify which edge the walk exited at
        if final_x == 0:
            edge_counts["left"] += 1
        elif final_x == size_1:
            edge_counts["right"] += 1
        elif final_y == 0:
            edge_counts["top"] += 1
        else:  # final_y == size_1
            edge_counts["bottom"] += 1

    print(f"\nCorrect version - Exit edge counts: {dict(edge_counts)}")
    print(f"Proportions: {[(k, f'{v/num_runs:.3f}') for k, v in edge_counts.items()]}")

    # Validate each edge is ~25%
    for edge in ["left", "right", "top", "bottom"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=edge_counts[edge],
            observed_denominator=num_runs,
            target_proportion=(0.23, 0.27),  # 25% ± 2%
            name=f"correct_{edge}_exit_proportion",
        )


def test_correct_version_horizontal_symmetry(fuzzy_checker):
    """
    Validate that left/right exits are balanced.

    Among walks that exit horizontally (left or right edge), expect about 50% on each side.
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(3000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid(grid, CORRECT_MOVES)

        if final_x == 0:
            edge_counts["left"] += 1
        elif final_x == size_1:
            edge_counts["right"] += 1

    horizontal_exits = edge_counts["left"] + edge_counts["right"]

    print(f"\nCorrect version - Horizontal exits: {horizontal_exits}")
    print(f"Left: {edge_counts['left']} ({edge_counts['left']/horizontal_exits:.3f})")
    print(f"Right: {edge_counts['right']} ({edge_counts['right']/horizontal_exits:.3f})")

    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=edge_counts["left"],
        observed_denominator=horizontal_exits,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="correct_left_right_symmetry",
    )


def test_correct_version_vertical_symmetry(fuzzy_checker):
    """
    Validate that top/bottom exits are balanced.

    Among walks that exit vertically (top or bottom edge), expect about 50% on each side.
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(4000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid(grid, CORRECT_MOVES)

        if final_y == 0:
            edge_counts["top"] += 1
        elif final_y == size_1:
            edge_counts["bottom"] += 1

    vertical_exits = edge_counts["top"] + edge_counts["bottom"]

    print(f"\nCorrect version - Vertical exits: {vertical_exits}")
    print(f"Top: {edge_counts['top']} ({edge_counts['top']/vertical_exits:.3f})")
    print(f"Bottom: {edge_counts['bottom']} ({edge_counts['bottom']/vertical_exits:.3f})")

    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=edge_counts["top"],
        observed_denominator=vertical_exits,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="correct_top_bottom_symmetry",
    )


# =============================================================================
# Tests for BUGGY Version (This should FAIL!)
# =============================================================================


def test_buggy_version_catches_exit_bias(fuzzy_checker):
    """
    Demonstrate fuzzy checking catching the bug.

    BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]] can move up twice but never down.
    Expected exit proportions:
    - Left edge:   ~25% ✓
    - Right edge:  ~25% ✓
    - Top edge:    ~50% ✗ (should be ~25%)
    - Bottom edge:  ~0% ✗ (should be ~25%)

    Expected: Test fails on top edge with large Bayes factor
    """
    grid = Grid(size=11)
    num_runs = 1000
    size_1 = grid.size - 1
    edge_counts = Counter()

    for i in range(num_runs):
        random.seed(5000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, final_x, final_y = fill_grid(grid, BUGGY_MOVES)

        if final_x == 0:
            edge_counts["left"] += 1
        elif final_x == size_1:
            edge_counts["right"] += 1
        elif final_y == 0:
            edge_counts["top"] += 1
        else:  # final_y == size_1
            edge_counts["bottom"] += 1

    print(f"\nBuggy version - Exit edge counts: {dict(edge_counts)}")
    print(f"Proportions: {[(k, f'{v/num_runs:.3f}') for k, v in edge_counts.items()]}")

    # Check if x == 0 happens the expected 25% of the time
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=edge_counts["left"],
        observed_denominator=num_runs,
        target_proportion=(0.23, 0.27),
        name="buggy_left_exit_proportion",
    )


# =============================================================================
# Running the Tests
# =============================================================================

if __name__ == "__main__":
    """
    Run with pytest to execute the test suite:
        pytest test_random_walk.py -v

    Try these experiments:
    1. Run all tests and see which pass/fail
    2. Modify BUGGY_MOVES to create subtler bugs
    3. Check the CSV diagnostics in the output directory
    """
    print("Run with pytest to execute the test suite:")
    print("  pytest test_random_walk.py -v")
    print("\nTo see fuzzy checker catch the bug:")
    print("  pytest test_random_walk.py::test_buggy_version_catches_exit_bias -v")
