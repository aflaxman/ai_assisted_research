# Fuzzy Checker Tutorial Results

## What We Built

This directory demonstrates how to use **Bayesian fuzzy checking** to validate statistical properties of stochastic simulations. We tested a random walk simulation with a subtle directional bias bug.

## The Bug

The buggy version has this move list:
```python
moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # BUG: [0, -1] twice!
```

This should be:
```python
moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]   # left, right, up, down
```

**Impact**: The walker can move left (25%), right (25%), up (50%), but NEVER down (0%).

## Demo Output

Running `python demo.py --size 11 --runs 200` shows the bias visually:

### Correct Version (Unbiased)
```
  left: 25.6% ████████████
 right: 24.6% ████████████
    up: 24.6% ████████████
  down: 25.2% ████████████
```

All four directions cluster around 25% ✓

### Buggy Version (Biased)
```
  left: 25.9% ████████████
 right: 25.0% ████████████
    up: 49.0% ████████████████████████
  down:  0.0% [empty]
```

Up is ~50%, down is 0% ✗

## Test Results

### Correct Version Tests: ALL PASS ✓

```bash
$ python -m pytest test_random_walk.py -k "correct_version" -v

test_random_walk.py::test_correct_version_directional_balance PASSED
test_random_walk.py::test_correct_version_left_right_symmetry PASSED
test_random_walk.py::test_correct_version_up_down_symmetry PASSED

3 passed
```

Each test validates a different statistical property:
- **Directional balance**: All four directions ≈ 25%
- **Horizontal symmetry**: Left ≈ 50% of horizontal moves
- **Vertical symmetry**: Up ≈ 50% of vertical moves

### Buggy Version Test: FAILS AS EXPECTED ✗

```bash
$ python -m pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v

FAILED test_random_walk.py::test_buggy_version_catches_directional_bias

AssertionError: buggy_up_moves_proportion value 0.504 is significantly
greater than expected, bayes factor = 6.90011e+87
```

**Key observation**: The Bayes factor is 6.9 × 10^87 - that's not just "significantly different", it's **astronomically decisive evidence of a bug**!

For context:
- Bayes factor > 100 = "decisive" evidence of bug
- Bayes factor < 0.1 = "substantial" evidence of no bug
- Our result: 6.9 × 10^87 = "the sun is more likely to explode tomorrow" level of certainty 😄

## Why Fuzzy Checking Wins

### Traditional Testing Problems

1. **Exact assertions fail**: Random walks produce different output each time
2. **Arbitrary thresholds are brittle**: "Is 20% error acceptable?" → p-hacking
3. **Simple properties miss bias**: Checking "grid sum == walk length" passes even with the bug
4. **Visual inspection doesn't scale**: You can't eyeball 1000 grids

### Fuzzy Checking Solution

1. **Tests statistical proportions**: "25% of moves should go left"
2. **Expresses uncertainty naturally**: Target can be a range (23%, 27%)
3. **Uses rigorous Bayesian inference**: No arbitrary cutoffs, just evidence ratios
4. **Provides diagnostics**: Saves detailed CSV files for analysis

## Key Concepts for Beginners

### What is a Bayes Factor?

The Bayes factor is the **evidence ratio** for two competing hypotheses:
- **H₀ (no bug)**: The observed proportion came from our target distribution
- **H₁ (bug)**: The observed proportion came from some other distribution

```
Bayes Factor = P(data | bug exists) / P(data | no bug)
```

- BF > 100: Bug hypothesis is 100× more likely → FAIL the test
- BF < 0.1: No-bug hypothesis is 10× more likely → PASS the test
- BF between 0.1 and 100: Inconclusive → WARNING (need more data)

### Why Not Just Use P-Values?

P-values answer: "If there's no bug, how likely is this extreme result?"

Problems:
- Requires setting significance level (α = 0.05?) before the test
- Doesn't quantify evidence strength
- Encourages p-hacking (try different thresholds until it passes)

Bayes factors answer: "What's the evidence ratio for bug vs. no-bug?"

Benefits:
- No pre-commitment to thresholds (though cutoffs are still used)
- Directly quantifies evidence strength
- Can express uncertainty about target values naturally

## Practical Takeaways

1. **Use fuzzy checking for any stochastic validation**
   - Random walks, Monte Carlo simulations, agent-based models, etc.

2. **Express targets as uncertainty intervals when appropriate**
   ```python
   target_proportion=(0.23, 0.27)  # "25% ± 2%"
   ```

3. **Run many trials for statistical power**
   - More samples → more decisive Bayes factors
   - We used 1000 runs per test

4. **Check multiple properties for robustness**
   - Don't just test "sum of moves"
   - Also test directional balance, symmetries, scaling relationships

5. **Save diagnostics for investigation**
   - The FuzzyChecker saves CSV files with all test details
   - Helpful for tuning validation strategies

## Going Further

### Exercises

1. **Adjust sensitivity**: Change the target interval from (0.23, 0.27) to (0.20, 0.30). Do the tests still catch the bug?

2. **Sample size exploration**: Reduce `num_runs` from 1000 to 100. What happens to the Bayes factors?

3. **Create a subtle bug**: Make moves = `[[-1, 0], [1, 0], [1, 0], [0, -1], [0, 1]]` (two right moves). Does fuzzy checking catch it?

4. **Validate new properties**: Test that the center cell is visited most often, or that the walk forms a roughly circular distribution.

### Read More

- [Vivarium Fuzzy Checking Documentation](https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#fuzzy-checking)
- [Greg Wilson's Original Testing Question](https://third-bit.com/2025/04/20/a-testing-question/)
- [Bayesian Hypothesis Testing Primer](https://en.wikipedia.org/wiki/Bayes_factor)

## Files in This Directory

- `README.md` - Overview and setup instructions
- `RESULTS.md` - This file! Summary of findings
- `random_walk.py` - Simulation code (both buggy and correct versions)
- `test_random_walk.py` - Comprehensive test suite with fuzzy checking
- `demo.py` - Interactive visualization of the bug
- `requirements.txt` - Python dependencies (uses vivarium_testing_utils directly)
- `REFACTORING.md` - Documentation of DRY improvements

## Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Visualize the bug
python demo.py --size 11 --runs 200

# Run all tests
python -m pytest test_random_walk.py -v

# Run just the correct version tests
python -m pytest test_random_walk.py -k "correct" -v

# See the bug caught in action
python -m pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v
```

---

**Happy Fuzzy Checking!** 🎲📊✨
