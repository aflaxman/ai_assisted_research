"""
Fuzzy Checker Tutorial: Testing Stochastic Simulations
=======================================================

This test suite demonstrates how to use Bayesian hypothesis testing (via FuzzyChecker)
to validate statistical properties of random simulations. We'll catch a subtle
directional bias bug that traditional testing misses!

Key Concepts
------------

1. **The Problem**: Random walks produce different output every time. How do you
   write assertions for code that's deliberately non-deterministic?

2. **The Solution**: Instead of testing exact values, we test *statistical properties*:
   - "About 25% of moves should go left" (within a reasonable range)
   - "Steps should scale roughly with grid size"

3. **The Innovation**: FuzzyChecker uses **Bayes factors** instead of arbitrary
   thresholds. We're not asking "is this close enough?" but rather "what's the
   evidence ratio for bug vs. no-bug?"

   A Bayes factor > 100 is "decisive" evidence of a bug.
   A Bayes factor < 0.1 is "substantial" evidence of no bug.

The Fuzzy Checking Pattern
---------------------------

Every fuzzy check follows the same simple pattern:

    1. Run many simulations to gather statistics
    2. Count events (numerator) and opportunities (denominator)
    3. Call fuzzy_assert_proportion() with your expectations
    4. Let Bayesian inference decide: bug or no bug?

Example:
    # We ran 1000 simulations and saw 245 left moves out of 980 total moves
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=245,          # Count of "successes"
        observed_denominator=980,        # Total opportunities
        target_proportion=(0.23, 0.27),  # Expected range (23%-27%)
        name="left_moves_proportion"     # For diagnostics
    )

The target can be:
- A single value: 0.25 (exact expectation)
- A range (tuple): (0.23, 0.27) (95% uncertainty interval)

That's it! No manual threshold tweaking. No p-hacking. Just rigorous Bayesian reasoning.
"""

import random
from collections import Counter

import pytest

from vivarium_testing_utils import FuzzyChecker
from random_walk import Grid


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
    PATTERN EXAMPLE: Validate that all four directions occur with equal probability.

    This test demonstrates the complete fuzzy checking pattern with the
    observation code fully exposed.

    Step 1: Gather statistics by running many simulations
    ------------------------------------------------------
    We run 1000 random walks and track which direction each move went.
    This gives us a large sample to analyze statistically.
    """
    grid = Grid(size=11)
    num_runs = 1000
    direction_counts = Counter()  # Track counts for each direction

    # Run many simulations and observe what happens
    for i in range(num_runs):
        random.seed(2000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        # Start at center
        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        # The CORRECT moves list
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down

        # Walk until we hit an edge
        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1

            # Pick a random move
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            # OBSERVE: Track which direction we moved
            if move == [-1, 0]:
                direction_counts["left"] += 1
            elif move == [1, 0]:
                direction_counts["right"] += 1
            elif move == [0, -1]:
                direction_counts["up"] += 1
            elif move == [0, 1]:
                direction_counts["down"] += 1

    total_moves = sum(direction_counts.values())

    print(f"\nCorrect version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(direction_counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in direction_counts.items()]}")

    """
    Step 2: Apply the fuzzy checking pattern
    -----------------------------------------
    For each direction, we use fuzzy_assert_proportion() to check:
        - Numerator: How many moves went in this direction?
        - Denominator: Total moves across all directions
        - Target: We expect 25% (with 95% UI of 23%-27%)

    Why 23%-27%?
    ------------
    With 1000 runs producing ~6000 total moves, pure random variation might
    give us anywhere from 23% to 27% for each direction. This range represents
    our 95% uncertainty interval.
    """
    for direction in ["left", "right", "up", "down"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=direction_counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),  # 25% ± 2%
            name=f"correct_{direction}_moves_proportion",
        )

    # Step 3: If we get here, all directions passed! The walk is unbiased. ✓


def test_correct_version_horizontal_symmetry(fuzzy_checker):
    """
    Validate that left/right moves are balanced.

    This checks a different property: among horizontal moves (left + right),
    we expect about 50% to go left and 50% to go right.

    Key insight: Multiple properties for robustness
    -----------------------------------------------
    We could have a bug that biases horizontal movement but still passes the
    overall directional test. Checking multiple properties catches more bugs!
    """
    grid = Grid(size=11)
    num_runs = 1000
    direction_counts = Counter()

    # Observation code - same pattern as before
    for i in range(num_runs):
        random.seed(3000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            if move == [-1, 0]:
                direction_counts["left"] += 1
            elif move == [1, 0]:
                direction_counts["right"] += 1

    horizontal_moves = direction_counts["left"] + direction_counts["right"]

    print(f"\nCorrect version - Horizontal moves: {horizontal_moves}")
    print(f"Left: {direction_counts['left']} ({direction_counts['left']/horizontal_moves:.3f})")
    print(f"Right: {direction_counts['right']} ({direction_counts['right']/horizontal_moves:.3f})")

    # Among horizontal moves, left should be ~50%
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=direction_counts["left"],
        observed_denominator=horizontal_moves,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="correct_left_right_symmetry",
    )


def test_correct_version_vertical_symmetry(fuzzy_checker):
    """
    Validate that up/down moves are balanced.

    Same pattern, checking vertical symmetry: among vertical moves (up + down),
    we expect ~50% in each direction.
    """
    grid = Grid(size=11)
    num_runs = 1000
    direction_counts = Counter()

    # Observation code
    for i in range(num_runs):
        random.seed(4000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            if move == [0, -1]:
                direction_counts["up"] += 1
            elif move == [0, 1]:
                direction_counts["down"] += 1

    vertical_moves = direction_counts["up"] + direction_counts["down"]

    print(f"\nCorrect version - Vertical moves: {vertical_moves}")
    print(f"Up: {direction_counts['up']} ({direction_counts['up']/vertical_moves:.3f})")
    print(f"Down: {direction_counts['down']} ({direction_counts['down']/vertical_moves:.3f})")

    # Among vertical moves, up should be ~50%
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=direction_counts["up"],
        observed_denominator=vertical_moves,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name="correct_up_down_symmetry",
    )


# =============================================================================
# Tests for BUGGY Version (This should FAIL!)
# =============================================================================


def test_buggy_version_catches_directional_bias(fuzzy_checker):
    """
    Demonstrate that fuzzy checking CATCHES the bug!

    The bug: moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]
    ------------------------------------------------------
    - Left:  25% (1 out of 4 options) ✓
    - Right: 25% (1 out of 4 options) ✓
    - Up:    50% (2 out of 4 options) ✗ TWICE AS LIKELY!
    - Down:   0% (not in the list)    ✗ NEVER HAPPENS!

    Expected outcome: SPECTACULAR FAILURE
    --------------------------------------
    The Bayes factor will be astronomical (>> 100), providing decisive evidence
    that something is wrong.

    Try it yourself:
        pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v

    You'll see something like:
        AssertionError: buggy_up_moves_proportion value 0.504 is significantly
        greater than expected, bayes factor = 6.90011e+87
    """
    grid = Grid(size=11)
    num_runs = 1000
    direction_counts = Counter()

    # Observation code with BUGGY moves
    for i in range(num_runs):
        random.seed(5000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        # THE BUG: [0, -1] twice, missing [0, 1]
        moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # left, right, up, up (!)

        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            # OBSERVE: Track which direction we moved
            if move == [-1, 0]:
                direction_counts["left"] += 1
            elif move == [1, 0]:
                direction_counts["right"] += 1
            elif move == [0, -1]:
                direction_counts["up"] += 1
            # Note: [0, 1] (down) will never occur in buggy version

    total_moves = sum(direction_counts.values())

    print(f"\nBuggy version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(direction_counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in direction_counts.items()]}")

    # Left and right should still pass (they're each ~25%)
    for direction in ["left", "right"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=direction_counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),
            name=f"buggy_{direction}_moves_proportion",
        )

    # Up should FAIL (it's ~50%, we expect ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=direction_counts["up"],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_up_moves_proportion",
    )

    # Down should FAIL SPECTACULARLY (it's 0%, we expect ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=direction_counts.get("down", 0),
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
    2. Adjust the uncertainty intervals and see how it affects sensitivity
    3. Check the CSV diagnostics in the output directory
    4. Modify the buggy moves list to create subtler bugs
    """
    print("Run with pytest to execute the test suite:")
    print("  pytest test_random_walk.py -v")
    print("\nTo see fuzzy checker catch the bug:")
    print("  pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v")
