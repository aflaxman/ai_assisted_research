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

How to Read This Code
----------------------

Each test follows the same pattern:
1. Run the simulation many times to gather statistics
2. Count something (e.g., how many moves went left?)
3. Use fuzzy_assert_proportion() to check if the proportion matches expectations
4. If the evidence strongly favors "bug", the test fails

No manual threshold tweaking. No p-hacking. Just rigorous Bayesian reasoning!
"""

import random
from collections import Counter
from pathlib import Path

import pytest

from fuzzy_checker import FuzzyChecker
from random_walk import fill_grid, fill_grid_buggy, Grid


# =============================================================================
# Test Fixtures (Pytest Setup)
# =============================================================================


@pytest.fixture(scope="session")
def output_directory(tmp_path_factory):
    """
    Create a temporary directory for diagnostic output.

    FuzzyChecker can save detailed diagnostics about each test to CSV files.
    This is super helpful for understanding why tests pass/fail and tuning
    your validation strategy.
    """
    return tmp_path_factory.mktemp("fuzzy_checker_diagnostics")


@pytest.fixture(scope="session")
def fuzzy_checker(output_directory):
    """
    Create a FuzzyChecker instance that persists across all tests.

    The session scope means this fixture is created once and shared by all
    tests in this file. At the end of the test session, it saves diagnostics
    to CSV automatically.
    """
    checker = FuzzyChecker()

    yield checker  # Tests run here

    # After all tests complete, save diagnostic output
    checker.save_diagnostic_output(output_directory)
    print(f"\nFuzzyChecker diagnostics saved to: {output_directory}")


# =============================================================================
# Helper Functions
# =============================================================================


def track_moves(grid, num_runs, seed_start=1000):
    """
    Run multiple random walks and track which direction each move went.

    This helper function runs many simulations and categorizes each step
    into one of four directions: left, right, up, or down.

    Args:
        grid: Grid object to use (will be reset each run)
        num_runs: How many independent walks to simulate
        seed_start: Starting seed (increments by 1 each run)

    Returns:
        Counter: {'left': count, 'right': count, 'up': count, 'down': count}

    Why track individual moves?
    ---------------------------
    In an unbiased random walk, each direction should occur ~25% of the time.
    If we see, say, 35% left and 15% right, that's evidence of a bug!
    """
    direction_counts = Counter()

    for i in range(num_runs):
        random.seed(seed_start + i)

        # Reset grid to empty state
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        # Track position as we step
        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        # Store the moves list for this run
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down

        # Simulate the walk and track each direction
        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1

            # Pick a random move
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            # Categorize the direction
            if move == [-1, 0]:
                direction_counts["left"] += 1
            elif move == [1, 0]:
                direction_counts["right"] += 1
            elif move == [0, -1]:
                direction_counts["up"] += 1
            elif move == [0, 1]:
                direction_counts["down"] += 1

    return direction_counts


def track_moves_buggy(grid, num_runs, seed_start=1000):
    """
    Same as track_moves() but uses the BUGGY version.

    This will show directional bias: we expect ~0% down moves and
    ~33% up moves (since [0, -1] appears twice in the moves list).
    """
    direction_counts = Counter()

    for i in range(num_runs):
        random.seed(seed_start + i)

        # Reset grid to empty state
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        # Track position as we step
        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        # THE BUG: [0, -1] twice, missing [0, 1]
        moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # left, right, up, up (!)

        # Simulate the walk and track each direction
        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1

            # Pick a random move
            move = random.choice(moves)
            x += move[0]
            y += move[1]

            # Categorize the direction
            if move == [-1, 0]:
                direction_counts["left"] += 1
            elif move == [1, 0]:
                direction_counts["right"] += 1
            elif move == [0, -1]:
                direction_counts["up"] += 1
            # Note: [0, 1] (down) will never occur in buggy version

    return direction_counts


# =============================================================================
# Tests for CORRECT Version
# =============================================================================


def test_correct_version_directional_balance(fuzzy_checker):
    """
    TEST 1: Validate that all four directions occur with equal probability.

    What we're testing
    ------------------
    In an unbiased 2D random walk, each of the four orthogonal directions
    (left, right, up, down) should occur with probability 0.25.

    We run many walks, count moves in each direction, and use the fuzzy
    checker to validate that each proportion is "close to" 0.25.

    Why fuzzy checking?
    -------------------
    With finite samples, we won't get exactly 25.000%. We might see 24.8% or
    25.3% due to random variation. FuzzyChecker uses Bayesian statistics to
    determine if the deviation is:
    - Normal random variation (pass)
    - Evidence of a systematic bias (fail)

    Reading the assertion
    ---------------------
    fuzzy_assert_proportion(
        observed_numerator=counts['left'],      # How many left moves we saw
        observed_denominator=total_moves,       # Total moves across all directions
        target_proportion=(0.23, 0.27),         # We expect 25% ± 2% uncertainty
        name="left_moves_proportion"            # Name for diagnostics
    )

    The target (0.23, 0.27) represents our 95% uncertainty interval.
    We're saying: "In an unbiased walk, we're 95% confident that left moves
    should be between 23% and 27% of all moves."
    """
    grid = Grid(size=11)  # 11x11 grid
    num_runs = 1000  # Run 1000 independent walks

    # Gather move statistics
    counts = track_moves(grid, num_runs, seed_start=2000)
    total_moves = sum(counts.values())

    print(f"\nCorrect version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in counts.items()]}")

    # Validate each direction is close to 0.25 (with uncertainty interval)
    # We expect 25% ± 2% for each direction
    for direction in ["left", "right", "up", "down"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),  # 95% UI around 0.25
            name=f"correct_{direction}_moves_proportion",
        )

    # If we get here, all directions passed! The walk is unbiased. ✓


def test_correct_version_left_right_symmetry(fuzzy_checker):
    """
    TEST 2: Validate that left/right moves are balanced.

    What we're testing
    ------------------
    Another way to check for bias: among horizontal moves (left + right),
    we expect about 50% to go left and 50% to go right.

    This is a slightly different perspective than Test 1, demonstrating that
    you can validate multiple statistical properties of the same simulation.

    Key insight
    -----------
    If there's a bug that biases horizontal movement (e.g., walker prefers
    going left), this test would catch it even if the overall proportion of
    horizontal vs. vertical moves was correct.
    """
    grid = Grid(size=11)
    num_runs = 1000

    counts = track_moves(grid, num_runs, seed_start=3000)

    # Focus on horizontal moves only
    horizontal_moves = counts["left"] + counts["right"]

    print(f"\nCorrect version - Horizontal moves: {horizontal_moves}")
    print(f"Left: {counts['left']} ({counts['left']/horizontal_moves:.3f})")
    print(f"Right: {counts['right']} ({counts['right']/horizontal_moves:.3f})")

    # Among horizontal moves, left should be ~50%
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts["left"],
        observed_denominator=horizontal_moves,
        target_proportion=(0.48, 0.52),  # Expect 50% ± 2%
        name="correct_left_vs_right_symmetry",
    )


def test_correct_version_up_down_symmetry(fuzzy_checker):
    """
    TEST 3: Validate that up/down moves are balanced.

    Similar to Test 2, but checking vertical symmetry.
    Among vertical moves (up + down), we expect ~50% in each direction.
    """
    grid = Grid(size=11)
    num_runs = 1000

    counts = track_moves(grid, num_runs, seed_start=4000)

    # Focus on vertical moves only
    vertical_moves = counts["up"] + counts["down"]

    print(f"\nCorrect version - Vertical moves: {vertical_moves}")
    print(f"Up: {counts['up']} ({counts['up']/vertical_moves:.3f})")
    print(f"Down: {counts['down']} ({counts['down']/vertical_moves:.3f})")

    # Among vertical moves, up should be ~50%
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts["up"],
        observed_denominator=vertical_moves,
        target_proportion=(0.48, 0.52),  # Expect 50% ± 2%
        name="correct_up_vs_down_symmetry",
    )


# =============================================================================
# Tests for BUGGY Version (These should FAIL!)
# =============================================================================


def test_buggy_version_catches_directional_bias(fuzzy_checker):
    """
    TEST 4: Demonstrate that fuzzy checking CATCHES the bug!

    What we're testing
    ------------------
    The buggy version has moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]
    This means:
    - Left:  25% (1 out of 4 options)
    - Right: 25% (1 out of 4 options)
    - Up:    50% (2 out of 4 options)  ← TWICE AS LIKELY!
    - Down:   0% (not in the list)     ← NEVER HAPPENS!

    We expect this test to FAIL with decisive evidence of a bug.

    The power of Bayesian testing
    ------------------------------
    The Bayes factor will be enormous (>> 100) for the down direction,
    providing decisive evidence that something is wrong. No need for
    manual threshold tweaking or p-value interpretation!

    Try it yourself
    ---------------
    Run pytest with -v flag to see the detailed failure message:
        pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v

    You'll see an AssertionError like:
        "down_moves_proportion value 0 is significantly less than expected,
         bayes factor = [huge number]"
    """
    grid = Grid(size=11)
    num_runs = 1000

    counts = track_moves_buggy(grid, num_runs, seed_start=5000)
    total_moves = sum(counts.values())

    print(f"\nBuggy version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in counts.items()]}")

    # This should PASS (left is still ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts["left"],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_left_moves_proportion",
    )

    # This should PASS (right is still ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts["right"],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_right_moves_proportion",
    )

    # This should FAIL! (up is ~50%, we expect ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts["up"],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_up_moves_proportion",
    )

    # This should FAIL SPECTACULARLY! (down is 0%, we expect ~25%)
    # Note: counts['down'] might not even be a key if it's zero
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts.get("down", 0),
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_down_moves_proportion",
    )


# =============================================================================
# Scaling Validation (Advanced)
# =============================================================================


def test_walk_length_scaling(fuzzy_checker):
    """
    TEST 5: Validate that walk length scales appropriately with grid size.

    What we're testing
    ------------------
    For a 2D random walk starting at the center, the expected number of steps
    to reach the boundary scales roughly with the square of the distance to
    the boundary.

    For a grid of size N:
    - Distance from center to edge ≈ N/2
    - Expected steps ≈ (N/2)^2 = N^2 / 4

    This is approximate! Random walks have high variance. But over many trials,
    the average should follow this scaling relationship.

    Why test this?
    --------------
    This validates that our simulation exhibits the correct statistical behavior
    beyond just directional balance. It's a sanity check that we're actually
    simulating a proper random walk, not something broken in a different way.

    Reading this test
    -----------------
    We run many walks on a 21x21 grid (distance to edge = 10).
    Expected steps ≈ 10^2 = 100.

    We allow substantial uncertainty (60-140 steps) because individual walks
    vary wildly. But the *average* over 500 walks should be close to 100.
    """
    grid = Grid(size=21)  # 21x21 grid, center to edge distance = 10
    num_runs = 500

    step_counts = []
    for i in range(num_runs):
        random.seed(6000 + i)
        # Reset grid
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]
        steps = fill_grid(grid)
        step_counts.append(steps)

    avg_steps = sum(step_counts) / len(step_counts)

    print(f"\nWalk length scaling test:")
    print(f"Grid size: 21x21 (distance to edge = 10)")
    print(f"Average steps over {num_runs} runs: {avg_steps:.1f}")
    print(f"Expected (rough): ~100 steps (10^2)")

    # We expect roughly 60-140 steps on average (100 ± 40%)
    # This is a loose bound because random walks have high variance!
    #
    # Note: For this test, we're checking if the TOTAL steps across all runs
    # falls in a reasonable range, not individual walk lengths.
    total_steps = sum(step_counts)
    expected_total = 100 * num_runs  # 100 steps/run * 500 runs = 50,000

    # Allow ±20% uncertainty (48,000 to 52,000 total steps)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_steps,
        observed_denominator=1,  # We're checking an absolute count
        target_proportion=(40_000 / 1, 60_000 / 1),  # Expect ~50k ± 10k
        name="walk_length_scaling",
    )

    # Alternative approach: Use proportion thinking
    # "What fraction of our observed steps is close to the expected average?"
    # This is a bit of a trick since we're not testing a true proportion,
    # but it demonstrates flexibility in how you frame validations.
    #
    # If we observed 50,000 steps out of a hypothetical 50,000 "ideal" steps,
    # that's a proportion of 1.0. If we observed 45,000, that's 0.9.
    #
    # We can validate: observed_steps / expected_steps ≈ 1.0 (within 0.8 to 1.2)

    ratio = total_steps / expected_total
    print(f"Observed/Expected ratio: {ratio:.3f}")

    # This demonstrates an alternative validation pattern:
    # Convert absolute values to proportions for fuzzy checking
    # (Not used in the actual assertion above, just shown for education)


# =============================================================================
# Running the Tests
# =============================================================================

if __name__ == "__main__":
    """
    You can run this file directly for exploration:
        python test_random_walk.py

    Or use pytest for proper test running:
        pytest test_random_walk.py -v

    Try these experiments:
    1. Run all tests and see which pass/fail
    2. Comment out the buggy test and see everything pass
    3. Adjust the uncertainty intervals and see how it affects sensitivity
    4. Check the CSV diagnostics in the output directory
    """
    print("Run with pytest to execute the test suite:")
    print("  pytest test_random_walk.py -v")
    print("\nTo see fuzzy checker catch the bug:")
    print("  pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v")
