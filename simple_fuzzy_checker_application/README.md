# Simple Fuzzy Checker Application

## The Challenge

This directory contains a case study inspired by [Greg Wilson's testing question](https://third-bit.com/2025/04/20/a-testing-question/). The challenge: how do you test stochastic simulations when:

1. **Traditional assertions fail** - A random walk won't produce the same output twice
2. **Statistical properties are subtle** - Bugs create biases that only appear in aggregate
3. **Arbitrary thresholds are brittle** - "Is 20% error acceptable?" becomes p-hacking

## The Answer: Bayesian Fuzzy Checking

Zeb's `FuzzyChecker` from [vivarium_testing_utils](https://github.com/ihmeuw/vivarium_testing_utils) uses **Bayesian hypothesis testing** to rigorously validate proportions without arbitrary cutoffs.

Instead of: "Is this close enough?" ❌
We ask: "What's the evidence ratio for bug vs. no-bug?" ✅

## Files

- `random_walk.py` - The spatial simulation (buggy version included!)
- `test_random_walk.py` - Demonstrates fuzzy checker validation (DRY refactored!)
- `fuzzy_checker.py` - Standalone copy of the checker for this tutorial
- `demo.py` - Interactive visualization showing the bug

## The Bug

In `fill_grid()`, the moves list should be:
```python
moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down
```

But the buggy version has:
```python
moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # left, right, up, up (!)
```

This creates directional bias—the walker can't move down. Traditional tests miss this, but fuzzy checking catches it!

## Running the Tests

```bash
# Install dependencies
uv pip install scipy pytest

# Run tests (watch the fuzzy checker catch the bug!)
pytest test_random_walk.py -v
```

## Key Insights

1. **Fuzzy checking validates scaling relationships** - "Steps should grow ~quadratically with grid size"
2. **Bayes factors quantify evidence** - No arbitrary "close enough" thresholds
3. **Uncertainty intervals are natural** - Express "between 23% and 27%" directly
