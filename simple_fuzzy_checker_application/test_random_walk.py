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
"""

import random
from collections import Counter

import pytest

from vivarium_testing_utils import FuzzyChecker
from random_walk import Grid, fill_grid, CORRECT_MOVES, BUGGY_MOVES


def test_correct_version_exit_edges():
    """
    Validate that walks exit at each edge with equal probability.

    For an unbiased random walk, we expect 25% of walks to exit at each edge
    """
    num_runs = 1000
    num_left_exits = 0

    for i in range(num_runs):
        random.seed(2000 + i) # set a random seed for reproducibility, but make sure it is different for each replicate of the simulation
        grid = Grid(size=11)

        num_steps, final_x, final_y = fill_grid(grid, CORRECT_MOVES)

        if final_x == 0:
           num_left_exits += 1

    FuzzyChecker().fuzzy_assert_proportion(
        observed_numerator=num_left_exits,
        observed_denominator=num_runs,
        target_proportion=0.25,
    )


# This next test should FAIL!
def test_buggy_version_catches_exit_bias():
    """
    Demonstrate fuzzy checking catching the bug.

    BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]] can move up twice but never down.
    """
    grid = Grid(size=11)
    num_runs = 1000
    num_left_exits = 0

    for i in range(num_runs):
        random.seed(5000 + i)
        grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

        num_steps, final_x, final_y = fill_grid(grid, BUGGY_MOVES)

        if final_x == 0:
            num_left_exits += 1

    FuzzyChecker().fuzzy_assert_proportion(
        observed_numerator=num_left_exits,
        observed_denominator=num_runs,
        target_proportion=0.25,
    )

