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

# Run a single simulation
python random_walk.py --seed 42 --size 11
```

You should see output showing the step count and direction counts for a single random walk.

---

## The Fuzzy Checking Pattern

We separate simulation code (`random_walk.py`) from test code (`test_random_walk.py`). Observation code lives in the implementation; tests aggregate and validate.

### 1. Write Your Simulation (in `random_walk.py`)

The `fill_grid(grid, moves)` function simulates and observes:

```python
def fill_grid(grid, moves):
    """Fill grid with random walk, tracking direction counts."""
    center = grid.size // 2
    size_1 = grid.size - 1
    x, y = center, center
    num = 0
    direction_counts = Counter()

    while (x != 0) and (y != 0) and (x != size_1) and (y != size_1):
        grid[x, y] += 1
        num += 1

        m = random.choice(moves)
        x += m[0]
        y += m[1]

        # Track which direction we moved
        if m == [-1, 0]:
            direction_counts["left"] += 1
        elif m == [1, 0]:
            direction_counts["right"] += 1
        elif m == [0, -1]:
            direction_counts["up"] += 1
        elif m == [0, 1]:
            direction_counts["down"] += 1

    return num, direction_counts
```

See [`random_walk.py` lines 46-91](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/random_walk.py#L46-L91) for the complete implementation.

### 2. Test by Calling Your Implementation

```python
from random_walk import Grid, fill_grid, CORRECT_MOVES

grid = Grid(size=11)
num_runs = 1000
total_counts = Counter()

for i in range(num_runs):
    random.seed(2000 + i)
    grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

    _, direction_counts = fill_grid(grid, CORRECT_MOVES)
    total_counts.update(direction_counts)

total_moves = sum(total_counts.values())
```

### 3. Assert with Bayes Factors

```python
fuzzy_checker.fuzzy_assert_proportion(
    observed_numerator=direction_counts["left"],
    observed_denominator=total_moves,
    target_proportion=(0.23, 0.27),  # 25% ± 2%
    name="left_moves_proportion"
)
```

### 4. Bayesian Inference Decides

- Observed matches target → Low Bayes factor → PASS
- Systematic bias → High Bayes factor → FAIL with evidence quantification

No manual threshold tweaking. No p-value interpretation.

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

The interval form is more common in my simulations where it is harder to derive expected values precisely.

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

From our [buggy test](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L279-L362):

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

Let's walk through a complete test from [`test_random_walk.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L64-L107).

**Step 1: Run many simulations** (lines 77-90)
```python
from random_walk import Grid, fill_grid, CORRECT_MOVES

grid = Grid(size=11)
total_counts = Counter()

for i in range(num_runs):
    random.seed(2000 + i)
    grid.grid = [[0 for _ in range(grid.size)] for _ in range(grid.size)]

    _, direction_counts = fill_grid(grid, CORRECT_MOVES)
    total_counts.update(direction_counts)
```

The observation code is in [`random_walk.py` lines 80-89](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/random_walk.py#L80-L89).

**Step 2: Check each direction with fuzzy assertion** (lines 99-105)
```python
for direction in ["left", "right", "up", "down"]:
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=total_counts[direction],
        observed_denominator=total_moves,
        target_proportion=(0.23, 0.27),  # 25% ± 2%
        name=f"correct_{direction}_moves_proportion",
    )
```

**Step 3: If we get here, all passed!** The walk is unbiased. ✓

See the [complete test code](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L64-L107).

---

## Running the Tests

### Run All Tests

```bash
pytest test_random_walk.py -v
```

Expected output:
```
test_correct_version_directional_balance PASSED
test_correct_version_horizontal_symmetry PASSED
test_correct_version_vertical_symmetry PASSED
test_buggy_version_catches_directional_bias FAILED  # ✓ Catches the bug!

3 passed, 1 failed
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

- **Directional balance**: Each direction ≈ 25% ([lines 96-172](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L96-L172))
- **Horizontal symmetry**: Left ≈ 50% of horizontal moves ([lines 175-224](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L175-L224))
- **Vertical symmetry**: Up ≈ 50% of vertical moves ([lines 227-271](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py#L227-L271))

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
The simulation implementation:
- `Grid` class - Simple 2D grid for tracking visits
- `fill_grid(grid, moves)` - Random walk that returns (steps, direction_counts)
- `CORRECT_MOVES` and `BUGGY_MOVES` constants
- Command-line interface for running single simulations

Run a simulation:
```bash
python random_walk.py --seed 42 --size 11
```

### [`test_random_walk.py`](https://github.com/aflaxman/ai_assisted_research/blob/main/simple_fuzzy_checker_application/test_random_walk.py)
Comprehensive test suite with:
- Four test functions validating different statistical properties
- Examples of using `fuzzy_assert_proportion()`
- Comments explaining each step

The tests demonstrate: observation → aggregation → fuzzy assertion.

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

## A Challenge: Can You Find the Bug Without Observing Directions?

The tests above track individual direction counts. But what if we didn't instrument our code that way? **Can you detect the directional bias using only the grid visit counts?**

This is Greg Wilson's original challenge: find a statistical property of the grid itself that differs between correct and buggy versions.

Some ideas to explore:
- Does the distribution of visits differ between quadrants?
- Are edge cells visited at different rates?
- Does the center-to-edge gradient change?
- What about the variance in visit counts?

Try implementing a test that catches the bug without looking at `direction_counts`. It's harder than it seems!

---

## Exercises: Deepen Your Understanding

Ready to experiment? Try these exercises to build intuition about fuzzy checking:

### 1. Adjust Sensitivity
Change the target interval in `test_random_walk.py` from `(0.23, 0.27)` to `(0.20, 0.30)`.
- Do the tests still catch the bug?
- What happens to the Bayes factors?
- What does this teach you about uncertainty intervals?

### 2. Sample Size Exploration
Reduce `num_runs` from 1000 to 100 in the directional balance test.
- What happens to the Bayes factors?
- Do tests become inconclusive?
- How many runs do you need for decisive evidence?

### 3. Create a Subtle Bug
Modify the moves list to `[[-1, 0], [1, 0], [1, 0], [0, -1], [0, 1]]` (two right moves instead of two up moves).
- Does fuzzy checking catch this subtler 33% vs 25% bias?
- How does the Bayes factor compare to the up/down bug?
- What does this reveal about detection power?

### 4. Validate New Properties
Write a new test that validates:
- The center cell is visited most often
- The walk forms a roughly circular distribution
- The total path length scales as (grid size)²

**Hint**: For the center cell test, compare `grid[center, center]` to the average of edge cells using `fuzzy_assert_proportion()`.

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

---

## What About More Complex Simulations?

This tutorial used a simple random walk where tracking direction counts was straightforward. But what about more complex spatial processes?

Greg Wilson's blog post includes another example: **[invasion percolation](https://third-bit.com/2025/04/20/a-testing-question/#invasion-percolation)**, where a "filled" region grows by randomly selecting neighboring cells to fill next. The grid patterns are much more complex than a random walk.

**How would you test that?** What statistical properties would you validate? How would you instrument the code to observe the right quantities?

These are open questions. If you have ideas or try implementing fuzzy tests for invasion percolation, I'd love to hear about it! Open an issue or discussion in [this repository](https://github.com/aflaxman/ai_assisted_research/issues).

---

## Questions or Issues?

- **Tutorial questions**: Open an issue in [this repository](https://github.com/aflaxman/ai_assisted_research/issues)
- **Package bugs**: Report to [vivarium_testing_utils](https://github.com/ihmeuw/vivarium_testing_utils/issues)
- **Statistical methodology**: See the [Vivarium research documentation](https://vivarium-research.readthedocs.io/en/latest/model_design/vivarium_features/automated_v_and_v/index.html)

---

*This tutorial was created to demonstrate practical statistical validation for spatial simulations. The fuzzy checking methodology was developed by the [Vivarium](https://vivarium-research.readthedocs.io/) team at IHME for validation and verification of complex health simulations.*
