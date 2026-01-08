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
from typing import Literal

import pytest

from vivarium_testing_utils import FuzzyChecker
from random_walk import fill_grid, Grid


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
# Helper Functions - The DRY Approach
# =============================================================================


# Move definitions for both versions
CORRECT_MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down
BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]]   # left, right, up, up (!)


def categorize_move(move: list[int]) -> str:
    """
    Convert a move vector into a direction name.

    This eliminates the repetitive if/elif chains for classifying moves.
    """
    move_map = {
        (-1, 0): "left",
        (1, 0): "right",
        (0, -1): "up",
        (0, 1): "down",
    }
    return move_map.get(tuple(move), "unknown")


def track_moves(grid, moves_list, num_runs, seed_start=1000):
    """
    Run multiple random walks and track which direction each move went.

    This is our main data-gathering helper. It runs many independent
    simulations and counts how often each direction was chosen.

    Args:
        grid: Grid object to use (will be reset each run)
        moves_list: Which move set to use (CORRECT_MOVES or BUGGY_MOVES)
        num_runs: How many independent walks to simulate
        seed_start: Starting seed (increments by 1 each run)

    Returns:
        Counter: {'left': count, 'right': count, 'up': count, 'down': count}

    Why track individual moves?
    ---------------------------
    In an unbiased random walk, each direction should occur ~25% of the time.
    If we see, say, 35% left and 15% right, that's evidence of a bug!

    The DRY improvement:
    --------------------
    Instead of having separate track_moves() and track_moves_buggy() functions
    that are 95% identical, we pass the moves_list as a parameter. This
    eliminates duplication while keeping the logic clear.
    """
    direction_counts = Counter()

    for i in range(num_runs):
        random.seed(seed_start + i)

        # Reset grid to empty state
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        # Track position as we walk
        center = grid.size // 2
        size_1 = grid.size - 1
        x, y = center, center

        # Simulate the walk and track each direction
        while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
            grid[x, y] += 1

            # Pick a random move and categorize it
            move = random.choice(moves_list)
            direction_counts[categorize_move(move)] += 1

            # Update position
            x += move[0]
            y += move[1]

    return direction_counts


def check_symmetry(
    fuzzy_checker,
    counts: Counter,
    axis: Literal["horizontal", "vertical"],
    test_name: str,
    verbose: bool = True,
):
    """
    Check if moves are balanced along horizontal or vertical axis.

    This consolidates the left/right and up/down symmetry tests into one
    reusable helper. Instead of duplicating similar test code, we parameterize
    the axis being checked.

    Args:
        fuzzy_checker: The FuzzyChecker instance
        counts: Direction counts from track_moves()
        axis: "horizontal" (check left vs right) or "vertical" (check up vs down)
        test_name: Name for diagnostics
        verbose: Whether to print detailed info

    Example:
        counts = track_moves(grid, CORRECT_MOVES, 1000)
        check_symmetry(checker, counts, "horizontal", "left_right_balance")
        # Validates: left ≈ 50% of (left + right)
    """
    if axis == "horizontal":
        direction_a, direction_b = "left", "right"
    else:  # vertical
        direction_a, direction_b = "up", "down"

    total = counts[direction_a] + counts[direction_b]
    proportion_a = counts[direction_a] / total if total > 0 else 0

    if verbose:
        print(f"\n{axis.capitalize()} symmetry check:")
        print(f"  {direction_a}: {counts[direction_a]} ({proportion_a:.3f})")
        print(f"  {direction_b}: {counts[direction_b]} ({1-proportion_a:.3f})")

    # In a symmetric walk, each direction should be ~50% of its axis
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts[direction_a],
        observed_denominator=total,
        target_proportion=(0.48, 0.52),  # 50% ± 2%
        name=test_name,
    )


# =============================================================================
# Tests for CORRECT Version
# =============================================================================


def test_correct_version_directional_balance(fuzzy_checker):
    """
    PATTERN EXAMPLE: Validate that all four directions occur with equal probability.

    This test demonstrates the complete fuzzy checking pattern:

    Step 1: Gather statistics by running many simulations
    ------------------------------------------------------
    We run 1000 random walks and track which direction each move went.
    This gives us a large sample to analyze statistically.

    Step 2: Apply the fuzzy checking pattern
    -----------------------------------------
    For each direction, we use fuzzy_assert_proportion() to check:
        - Numerator: How many moves went in this direction?
        - Denominator: Total moves across all directions
        - Target: We expect 25% (with 95% UI of 23%-27%)

    Step 3: Interpret the Bayes factor
    -----------------------------------
    If the observed proportion is far from 25%, the Bayes factor will be high,
    indicating strong evidence of a bug. If it's close to 25%, Bayes factor
    will be low, indicating the simulation is working correctly.

    Why 23%-27%?
    ------------
    With 1000 runs producing ~6000 total moves, pure random variation might
    give us anywhere from 23% to 27% for each direction. This range represents
    our 95% uncertainty interval - we're saying "if there's no bug, we expect
    to see proportions in this range 95% of the time."

    The remaining tests follow this same pattern - just applied to different
    statistical properties!
    """
    grid = Grid(size=11)
    num_runs = 1000

    # STEP 1: Gather statistics
    counts = track_moves(grid, CORRECT_MOVES, num_runs, seed_start=2000)
    total_moves = sum(counts.values())

    print(f"\nCorrect version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in counts.items()]}")

    # STEP 2: Check each direction using the fuzzy checking pattern
    for direction in ["left", "right", "up", "down"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),  # 25% ± 2%
            name=f"correct_{direction}_moves_proportion",
        )

    # STEP 3: If we get here, all directions passed! The walk is unbiased. ✓


def test_correct_version_left_right_symmetry(fuzzy_checker):
    """
    Validate that left/right moves are balanced.

    This checks a different property: among horizontal moves (left + right),
    we expect about 50% to go left and 50% to go right.

    Key insight: Multiple properties for robustness
    -----------------------------------------------
    We could have a bug that biases horizontal movement (e.g., walker prefers
    going left) but still passes the overall directional test if vertical
    moves compensate. Checking multiple properties catches more bugs!

    Note the DRY improvement:
    -------------------------
    We use check_symmetry() instead of duplicating the symmetry logic.
    """
    grid = Grid(size=11)
    counts = track_moves(grid, CORRECT_MOVES, num_runs=1000, seed_start=3000)

    check_symmetry(
        fuzzy_checker,
        counts,
        axis="horizontal",
        test_name="correct_left_right_symmetry",
    )


def test_correct_version_up_down_symmetry(fuzzy_checker):
    """
    Validate that up/down moves are balanced.

    Same pattern as the horizontal symmetry test, just checking the vertical axis.
    Among vertical moves (up + down), we expect ~50% in each direction.
    """
    grid = Grid(size=11)
    counts = track_moves(grid, CORRECT_MOVES, num_runs=1000, seed_start=4000)

    check_symmetry(
        fuzzy_checker,
        counts,
        axis="vertical",
        test_name="correct_up_down_symmetry",
    )


# =============================================================================
# Tests for BUGGY Version (These should FAIL!)
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
    The Bayes factor for "down moves" will be astronomical (>> 100), providing
    decisive evidence that something is wrong. For the "up moves" test, we'll
    also see a very high Bayes factor since 50% is far from our expected 25%.

    Try it yourself:
        pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v

    You'll see something like:
        AssertionError: buggy_up_moves_proportion value 0.504 is significantly
        greater than expected, bayes factor = 6.90011e+87

    That's not just "significantly different" - it's "the sun is more likely
    to explode tomorrow" level of certainty! 🌟
    """
    grid = Grid(size=11)
    counts = track_moves(grid, BUGGY_MOVES, num_runs=1000, seed_start=5000)
    total_moves = sum(counts.values())

    print(f"\nBuggy version - Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(counts)}")
    print(f"Proportions: {[f'{k}: {v/total_moves:.3f}' for k, v in counts.items()]}")

    # Left and right should still pass (they're each ~25%)
    for direction in ["left", "right"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),
            name=f"buggy_{direction}_moves_proportion",
        )

    # Up should FAIL (it's ~50%, we expect ~25%)
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=counts["up"],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),
        name="buggy_up_moves_proportion",
    )

    # Down should FAIL SPECTACULARLY (it's 0%, we expect ~25%)
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
    Validate that walk length scales appropriately with grid size.

    Advanced statistical property: Quadratic scaling
    -------------------------------------------------
    For a 2D random walk starting at the center, the expected number of steps
    to reach the boundary scales roughly as (distance to edge)².

    For a 21x21 grid:
    - Distance from center to edge = 10
    - Expected steps ≈ 10² = 100

    Why test this?
    --------------
    This validates that our simulation exhibits correct statistical behavior
    beyond just directional balance. It's a sanity check that we're simulating
    a proper random walk, not something broken in a different way.

    Note on variance:
    -----------------
    Individual walks vary wildly (some might be 30 steps, others 300!), but
    the *average* over many walks should be close to the theoretical expectation.
    We allow substantial uncertainty (±20%) because of this natural variation.
    """
    grid = Grid(size=21)  # 21x21 grid, center to edge distance = 10
    num_runs = 500

    # Gather walk lengths
    step_counts = []
    for i in range(num_runs):
        random.seed(6000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]
        steps = fill_grid(grid)
        step_counts.append(steps)

    avg_steps = sum(step_counts) / len(step_counts)
    total_steps = sum(step_counts)

    print(f"\nWalk length scaling test:")
    print(f"Grid size: 21x21 (distance to edge = 10)")
    print(f"Average steps over {num_runs} runs: {avg_steps:.1f}")
    print(f"Expected (rough): ~100 steps (10²)")

    # Check if total steps falls in expected range
    # Expected: 100 steps/run × 500 runs = 50,000 total
    # We allow ±20% due to high variance in individual walks
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_steps,
        observed_denominator=1,
        target_proportion=(40_000 / 1, 60_000 / 1),
        name="walk_length_scaling",
    )

    ratio = total_steps / (100 * num_runs)
    print(f"Observed/Expected ratio: {ratio:.3f}")


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
