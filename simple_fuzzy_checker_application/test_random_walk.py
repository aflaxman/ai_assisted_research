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
def output_directory(tmp_path_factory):
    """
    Create a temporary directory for diagnostic output.

    FuzzyChecker saves detailed diagnostics about each test to CSV files,
    which is helpful for understanding results and tuning validation strategy.
    """
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
# Tests for CORRECT Version
# =============================================================================


def test_correct_version_directional_balance(fuzzy_checker):
    """
    Validate that all four directions occur with equal probability.

    Pattern:
    1. Run fill_grid() many times with CORRECT_MOVES
    2. Aggregate the direction counts
    3. Validate with fuzzy_assert_proportion()

    The observation code is in random_walk.py (lines 80-89).
    """
    grid = Grid(size=11)
    num_runs = 1000
    total_counts = Counter()

    for i in range(num_runs):
        random.seed(2000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, direction_counts = fill_grid(grid, CORRECT_MOVES)
        total_counts.update(direction_counts)

    total_moves = sum(total_counts.values())

    print(f"\nCorrect version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(total_counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in total_counts.items()]}")

    # Validate each direction is close to 0.25
    for direction in ["left", "right", "up", "down"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=total_counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),  # 25% ± 2%
            name=f"correct_{direction}_moves_proportion",
        )


def test_correct_version_horizontal_symmetry(fuzzy_checker):
    """
    Validate that left/right moves are balanced.

    Among horizontal moves (left + right), expect about 50% left, 50% right.
    """
    grid = Grid(size=11)
    num_runs = 1000
    total_counts = Counter()

    for i in range(num_runs):
        random.seed(3000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, direction_counts = fill_grid(grid, CORRECT_MOVES)
        total_counts.update(direction_counts)

    # Focus on horizontal moves only
    horizontal_moves = total_counts["left"] + total_counts["right"]

    print(f"\nCorrect version - Horizontal moves: {horizontal_moves}")
    print(f"Left: {total_counts['left']} ({total_counts['left']/horizontal_moves:.3f})")
    print(f"Right: {total_counts['right']} ({total_counts['right']/horizontal_moves:.3f})")

    # Among horizontal moves, left should be ~50%
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_counts["left"],
        observed_denominator=horizontal_moves,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="correct_left_right_symmetry",
    )


def test_correct_version_vertical_symmetry(fuzzy_checker):
    """
    Validate that up/down moves are balanced.
    """
    grid = Grid(size=11)
    num_runs = 1000
    total_counts = Counter()

    for i in range(num_runs):
        random.seed(4000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, direction_counts = fill_grid(grid, CORRECT_MOVES)
        total_counts.update(direction_counts)

    # Focus on vertical moves only
    vertical_moves = total_counts["up"] + total_counts["down"]

    print(f"\nCorrect version - Vertical moves: {vertical_moves}")
    print(f"Up: {total_counts['up']} ({total_counts['up']/vertical_moves:.3f})")
    print(f"Down: {total_counts['down']} ({total_counts['down']/vertical_moves:.3f})")

    # Among vertical moves, up should be ~50%
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_counts["up"],
        observed_denominator=vertical_moves,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="correct_up_down_symmetry",
    )


# =============================================================================
# Tests for BUGGY Version (This should FAIL!)
# =============================================================================


def test_buggy_version_catches_directional_bias(fuzzy_checker):
    """
    Demonstrate fuzzy checking catching the bug.

    BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]] has:
    - Left:  25% ✓
    - Right: 25% ✓
    - Up:    50% ✗
    - Down:   0% ✗

    Expected output:
        AssertionError: buggy_up_moves_proportion value 0.504 is significantly
        greater than expected, bayes factor = 6.90011e+87
    """
    grid = Grid(size=11)
    num_runs = 1000
    total_counts = Counter()

    for i in range(num_runs):
        random.seed(5000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        _, direction_counts = fill_grid(grid, BUGGY_MOVES)
        total_counts.update(direction_counts)

    total_moves = sum(total_counts.values())

    print(f"\nBuggy version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(total_counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in total_counts.items()]}")

    # Left and right should still pass (they're each ~25%)
    for direction in ["left", "right"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=total_counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),
            name=f"buggy_{direction}_moves_proportion",
        )

    # Up should FAIL (it's ~50%, we expect ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_counts["up"],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_up_moves_proportion",
    )

    # Down should FAIL SPECTACULARLY (it's 0%, we expect ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_counts.get("down", 0),
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_down_moves_proportion",
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
    2. Look at random_walk.py lines 80-89 to see the observation code
    3. Modify BUGGY_MOVES to create subtler bugs
    4. Check the CSV diagnostics in the output directory
    """
    print("Run with pytest to execute the test suite:")
    print("  pytest test_random_walk.py -v")
    print("\nTo see fuzzy checker catch the bug:")
    print("  pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v")
