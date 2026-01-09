# Testing Stochastic Simulations: Bayesian Fuzzy Checking in Python

*A hands-on guide to catching simulations bugs with automated tests*

## The Problem: How Do You Test Randomized Algorithms?

Imagine you're using computer simulation in your research, like an agent-based model or a Monte Carlo simulation. You run your code and it produces output. Then you run it again and get *different* output. That's expected! It's random. But here's the challenge:

**How do you know your "random" behavior is actually correct?**

Traditional unit tests fail:
```python
# This doesn't work - random walks vary!
assert result == 42  # ❌

# Neither does this - what threshold?
assert 40 <= result <= 44  # ❌ Arbitrary!
```

This tutorial, inspired by [Greg Wilson's testing challenge](https://third-bit.com/2025/04/20/a-testing-question/), demonstrates a rigorous solution: **Bayesian fuzzy checking** using the [`FuzzyChecker`](https://github.com/ihmeuw/vivarium_testing_utils/blob/main/src/vivarium_testing_utils/fuzzy_checker.py) from [vivarium_testing_utils](https://github.com/ihmeuw/vivarium_testing_utils).

## An Answer: Bayesian Hypothesis Testing

Instead of asking "is this close enough?" (with arbitrary thresholds), we ask:

**"What's the evidence ratio for bug vs. no-bug?"**

This is done through **Bayes factors**:
- Bayes factor > 100 = "decisive" evidence of a bug → Test FAILS
- Bayes factor < 0.1 = "substantial" evidence of correctness → Test PASSES
- Bayes factor between 0.1 and 100 = "inconclusive" → WARNING (need more data)


---

## The Bug We'll Catch

We'll test a simple 2D random walk simulation. The walker starts at the center of a grid and takes random steps (left, right, up, down) until it reaches an edge.

The **correct** implementation:
```python
moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # left, right, up, down
```

The **buggy** implementation has a subtle typo:
```python
moves = [[-1, 0], [1, 0], [0, -1], [0, -1]]  # left, right, up, up (!)
```

Can you spot it? `[0, -1]` appears twice (up), and `[0, 1]` (down) is missing!

**Impact**: The walker can move left (25%), right (25%), up (50%), but **never down (0%)**. This creates a directional bias that's hard to spot by eye but shows up dramatically in statistical tests.

---

## Setting Up Your Environment

We'll use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

### Step 1: Install uv (if needed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Create and Activate Environment

```bash
# Clone or navigate to the tutorial directory
cd simple_fuzzy_checker_application

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (vivarium_testing_utils + pytest)
uv pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test that everything works
python -c "from vivarium_testing_utils import FuzzyChecker; print('✓ FuzzyChecker imported successfully')"

# Run the demo visualization
python demo.py --size 11 --runs 200
```

You should see output like:
```
✓ CORRECT VERSION - Unbiased Random Walk
  left: 25.6% ████████████
 right: 24.6% ████████████
    up: 24.6% ████████████
  down: 25.2% ████████████

✗ BUGGY VERSION - Directional Bias!
  left: 25.9% ████████████
 right: 25.0% ████████████
    up: 49.0% ████████████████████████
  down:  0.0% [empty]

🔍 THE SMOKING GUN:
Down moves in correct version: 1,482
Down moves in buggy version:   0
```

---

## The Fuzzy Checking Pattern

Every statistical validation follows the same simple pattern:

### 1. Run Many Simulations
Gather statistics by running your simulation hundreds or thousands of times:
```python
for i in range(1000):
    random.seed(seed_start + i)
    result = run_simulation()
    # Track what happened
```

### 2. Count Events
Identify the numerator (successes) and denominator (opportunities):
```python
left_moves = 245        # How many moves went left
total_moves = 980       # Total moves in all directions
```

### 3. Assert with Bayes Factors
Use `fuzzy_assert_proportion()` to validate:
```python
fuzzy_checker.fuzzy_assert_proportion(
    observed_numerator=left_moves,       # Count of events
    observed_denominator=total_moves,    # Total opportunities
    target_proportion=(0.23, 0.27),      # Expected range: 25% ± 2%
    name="left_moves_proportion"         # For diagnostics
)
```

### 4. Let Bayesian Inference Decide
- If the observed proportion matches the target → Bayes factor is low → Test PASSES
- If there's a systematic bias → Bayes factor is high → Test FAILS with evidence quantification

**That's it!** No manual threshold tweaking. No p-value interpretation. Just rigorous reasoning.

---

## The Core Method: `fuzzy_assert_proportion()`

The heart of fuzzy checking is the [`fuzzy_assert_proportion()`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L272-L277) method. Here's how it works:

### Basic Usage

```python
from vivarium_testing_utils import FuzzyChecker

checker = FuzzyChecker()

# Example: Validate that ~25% of moves go left
checker.fuzzy_assert_proportion(
    observed_numerator=1506,      # We saw 1506 left moves
    observed_denominator=5892,    # Out of 5892 total moves
    target_proportion=(0.23, 0.27),  # We expect 23%-27%
    name="left_moves_proportion"
)
# If this passes, there's substantial evidence of no bug ✓
```

### Target Proportion: Two Forms

**Exact value** (no uncertainty):
```python
target_proportion=0.25  # Expect exactly 25%
```

**Uncertainty interval** (95% confidence range):
```python
target_proportion=(0.23, 0.27)  # Expect 25% ± 2%
```

The interval form is more realistic for most simulations since theoretical values often have inherent uncertainty.

### What Happens Under the Hood

`fuzzy_assert_proportion()` performs Bayesian hypothesis testing:

1. **Defines two distributions**:
   - H₀ (no bug): Beta-binomial based on your target proportion
   - H₁ (bug): Beta-binomial with broad prior (Jeffreys prior by default)

2. **Calculates Bayes factor**:
   ```
   BF = P(observed data | bug exists) / P(observed data | no bug)
   ```

3. **Decides**:
   - BF > 100 → Decisive evidence of bug → `AssertionError` raised
   - BF < 0.1 → Substantial evidence of no bug → Test passes silently
   - 0.1 ≤ BF ≤ 100 → Inconclusive → Warning logged (need more data)

### Example Output When Bug is Caught

From our [buggy test](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L333-L391):

```python
# Up moves: we observed ~50%, expected ~25%
fuzzy_checker.fuzzy_assert_proportion(
    observed_numerator=4884,  # 50.4% of moves
    observed_denominator=9696,
    target_proportion=(0.23, 0.27),
    name="buggy_up_moves_proportion",
)
# Result: AssertionError with Bayes factor = 6.9 × 10⁸⁷
```

The output:
```
AssertionError: buggy_up_moves_proportion value 0.504 is significantly
greater than expected, bayes factor = 6.90011e+87
```

That's not just "statistically significant"—it's **astronomically decisive** evidence of a bug!

---

## A Complete Example: Directional Balance Test

Let's walk through a complete test from [`test_random_walk.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L225-L279):

```python
def test_correct_version_directional_balance(fuzzy_checker):
    """
    Validate that all four directions occur with equal probability.

    In an unbiased 2D random walk, each direction should occur ~25% of the time.
    """
    grid = Grid(size=11)
    num_runs = 1000

    # STEP 1: Gather statistics by running many simulations
    counts = track_moves(grid, CORRECT_MOVES, num_runs, seed_start=2000)
    total_moves = sum(counts.values())

    print(f"Total moves: {total_moves}")
    print(f"Direction breakdown: {dict(counts)}")
    # Output: {'left': 1506, 'right': 1452, 'up': 1452, 'down': 1482}

    # STEP 2: Check each direction using fuzzy checking
    for direction in ["left", "right", "up", "down"]:
        fuzzy_checker.fuzzy_assert_proportion(
            observed_numerator=counts[direction],
            observed_denominator=total_moves,
            target_proportion=(0.23, 0.27),  # 25% ± 2%
            name=f"correct_{direction}_moves_proportion",
        )

    # STEP 3: If we get here, all directions passed! The walk is unbiased. ✓
```

See the [full implementation](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L117-L169) of `track_moves()` for details on gathering statistics.

---

## Running the Tests

### Run All Tests

```bash
pytest test_random_walk.py -v
```

Expected output:
```
test_correct_version_directional_balance PASSED
test_correct_version_left_right_symmetry PASSED
test_correct_version_up_down_symmetry PASSED
test_buggy_version_catches_directional_bias FAILED  # ✓ Catches the bug!
test_walk_length_scaling PASSED
```

### Run Just Correct Version Tests

```bash
pytest test_random_walk.py -k "correct" -v
```

All three should pass, demonstrating that the unbiased random walk passes statistical validation.

### See the Bug Get Caught

```bash
pytest test_random_walk.py::test_buggy_version_catches_directional_bias -v
```

Watch the Bayes factor explode to 10⁸⁷ when checking the "down moves" proportion!

---

## What Makes This Approach Powerful

### 1. No Arbitrary Thresholds
Traditional approach:
```python
assert 0.20 <= proportion <= 0.30  # ❌ Why 0.20? Why 0.30?
```

Fuzzy checking:
```python
target_proportion=(0.23, 0.27)  # ✓ Explicit uncertainty interval
# Bayes factor quantifies evidence automatically
```

### 2. Catches Subtle Bugs
The directional bias bug is hard to spot:
- Code runs without errors ✓
- Output looks reasonable ✓
- Individual runs seem fine ✓
- But aggregate behavior is wrong! ✗

Fuzzy checking catches it decisively with Bayes factor > 10⁸⁷.

### 3. Multiple Properties for Robustness
We test several statistical properties (see [`test_random_walk.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py)):

- **Directional balance**: Each direction ≈ 25% ([lines 225-279](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L225-L279))
- **Horizontal symmetry**: Left ≈ 50% of horizontal moves ([lines 282-307](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L282-L307))
- **Vertical symmetry**: Up ≈ 50% of vertical moves ([lines 310-325](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L310-L325))
- **Scaling relationship**: Steps ∝ (grid size)² ([lines 399-454](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L399-L454))

Different bugs break different properties. Testing multiple properties catches more bugs.

### 4. Diagnostic Output
FuzzyChecker saves detailed CSV diagnostics:
```python
checker.save_diagnostic_output(output_directory)
```

Each test records:
- Observed proportion
- Target bounds
- Bayes factor
- Pass/fail decision

Great for investigating warnings or tuning validation strategies.

---

## Key Files in This Tutorial

### [`random_walk.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/random_walk.py)
The simulation code with both correct and buggy implementations:
- `fill_grid(grid)` - Correct unbiased random walk
- `fill_grid_buggy(grid)` - Buggy version with directional bias
- `Grid` class - Simple 2D grid for tracking visits

### [`test_random_walk.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py)
Comprehensive test suite demonstrating fuzzy checking patterns:
- DRY helpers: `track_moves()`, `check_symmetry()`, `categorize_move()`
- Five test functions showing different statistical validations
- Extensive educational comments explaining each step
- **Start here to understand the pattern!**

### [`demo.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/demo.py)
Interactive visualization showing the bug:
```bash
python demo.py --size 11 --runs 200
```

Generates ASCII bar charts comparing correct vs buggy versions. Great for visual learners!

### [`RESULTS.md`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/RESULTS.md)
Detailed findings, test outputs, and key takeaways from running the tutorial.

### [`REFACTORING.md`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/REFACTORING.md)
Documents the DRY improvements made to eliminate code duplication while maintaining educational clarity. Useful for tutorial authors.

---

## Adapting This for Your Own Work

Ready to use fuzzy checking in your own spatial simulations? Here's how:

### Step 1: Install the Package

```bash
pip install vivarium_testing_utils pytest
```

### Step 2: Identify Statistical Properties

What should your simulation do in aggregate?
- **Agent-based models**: "30% of agents should be in state A"
- **Monte Carlo**: "Average outcome should be between X and Y"
- **Spatial sims**: "Density should be uniform across regions"
- **Random walks**: "Mean squared displacement ∝ time"

### Step 3: Write Fuzzy Tests

```python
import pytest
from vivarium_testing_utils import FuzzyChecker

@pytest.fixture(scope="session")
def fuzzy_checker():
    checker = FuzzyChecker()
    yield checker
    checker.save_diagnostic_output("./diagnostics")

def test_my_property(fuzzy_checker):
    # Run your simulation many times
    successes = 0
    total = 0

    for seed in range(1000):
        result = my_simulation(seed)
        if condition_met(result):
            successes += 1
        total += 1

    # Validate with Bayesian inference
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=successes,
        observed_denominator=total,
        target_proportion=(0.28, 0.32),  # Your expected range
        name="my_property_validation"
    )
```

### Step 4: Tune Sample Sizes

- **Small samples** (n < 100): Tests might be inconclusive
- **Medium samples** (n = 100-1000): Good for most properties
- **Large samples** (n > 1000): High power to detect subtle bugs

If you get "inconclusive" warnings, increase your number of simulation runs.

### Step 5: Express Uncertainty Appropriately

- **Known exact values**: `target_proportion=0.5`
- **Theoretical with uncertainty**: `target_proportion=(0.48, 0.52)`
- **Empirical estimates**: Use wider intervals `(0.45, 0.55)`

The uncertainty interval represents your 95% confidence about the true value if there's no bug.

---

## Advanced Topics

### Custom Bug Priors

By default, `fuzzy_assert_proportion()` uses a Jeffreys prior for the "bug" hypothesis. You can customize this:

```python
fuzzy_checker.fuzzy_assert_proportion(
    observed_numerator=count,
    observed_denominator=total,
    target_proportion=(0.23, 0.27),
    bug_issue_beta_distribution_parameters=(1.0, 1.0),  # Uniform prior
    name="my_test"
)
```

### Custom Bayes Factor Cutoffs

Adjust sensitivity vs specificity:

```python
fuzzy_checker.fuzzy_assert_proportion(
    observed_numerator=count,
    observed_denominator=total,
    target_proportion=(0.23, 0.27),
    fail_bayes_factor_cutoff=50.0,        # Lower = more sensitive
    inconclusive_bayes_factor_cutoff=0.2,  # Higher = fewer warnings
    name="my_test"
)
```

See the [Vivarium documentation](https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#sensitivity-and-specificity) for guidance on choosing cutoffs.

### Testing Other Quantities

While this tutorial focuses on proportions, you can validate other quantities:
- **Means**: Transform to proportion of "above threshold" events
- **Scaling relationships**: Check if total/expected falls in (0.8, 1.2)
- **Distributions**: Use multiple proportion tests for different bins

---

## Why Bayesian, Not Frequentist?

You might wonder: "Why not just use a χ² test or t-test?"

### Problems with p-values:
1. **Arbitrary α**: Is 0.05 the right threshold? Why not 0.01 or 0.10?
2. **No evidence quantification**: p = 0.03 vs p = 0.0001 are both "significant" but very different
3. **Encourages p-hacking**: Try different thresholds until tests pass
4. **Can't express uncertainty in targets**: "Expected value is between 23% and 27%" is awkward

### Benefits of Bayes factors:
1. **No pre-commitment needed**: Cutoffs are conventional (100 = decisive) not arbitrary
2. **Quantifies evidence**: BF = 1000 vs BF = 10⁶ tells you how strong the evidence is
3. **Natural uncertainty**: Target intervals map directly to beta distributions
4. **Interpretable**: "Evidence ratio for bug vs no-bug" is intuitive

See the [Vivarium research docs](https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#fuzzy-checking) for more on the statistical methodology.

---

## Further Reading

### Primary Resources
- [Vivarium Testing Utils (GitHub)](https://github.com/ihmeuw/vivarium_testing_utils) - The source package
- [Vivarium Fuzzy Checking Docs](https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html#fuzzy-checking) - Detailed methodology
- [Greg Wilson's Testing Question](https://third-bit.com/2025/04/20/a-testing-question/) - The original challenge

### Statistical Background
- [Bayes Factors (Wikipedia)](https://en.wikipedia.org/wiki/Bayes_factor) - Introduction to Bayesian hypothesis testing
- [Beta-Binomial Distribution](https://en.wikipedia.org/wiki/Beta-binomial_distribution) - The core distribution used

### Related Techniques
- [Property-Based Testing](https://hypothesis.works/) - Complementary approach for finding edge cases
- [Monte Carlo Methods](https://en.wikipedia.org/wiki/Monte_Carlo_method) - Context for stochastic validation

---

## Conclusion

Testing stochastic simulations doesn't have to rely on arbitrary thresholds or manual inspection. **Bayesian fuzzy checking** provides a rigorous, principled approach:

✅ Quantifies evidence with Bayes factors
✅ Expresses uncertainty naturally
✅ Catches subtle bugs that traditional tests miss
✅ Provides diagnostic output for investigation
✅ Works with any proportion-based statistical property

The `vivarium_testing_utils` package makes this approach accessible with a simple, clean API. Whether you're testing random walks, agent-based models, or Monte Carlo simulations, fuzzy checking helps you validate statistical properties with confidence.

**Try it on your own simulations and discover bugs you didn't know existed!**

---

## Questions or Issues?

- **Tutorial questions**: Open an issue in [this repository](https://github.com/aflaxman/ai_assisted_research/issues)
- **Package bugs**: Report to [vivarium_testing_utils](https://github.com/ihmeuw/vivarium_testing_utils/issues)
- **Statistical methodology**: See the [Vivarium research documentation](https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html)

---

*This tutorial was created to demonstrate practical statistical validation for spatial simulations. The fuzzy checking methodology was developed by the [Vivarium](https://vivarium-research.readthedocs.io/) team at IHME for validation and verification of complex health simulations.*
