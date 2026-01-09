# DRY Refactoring Summary

This document explains the DRY (Don't Repeat Yourself) improvements made to the tutorial code while maintaining its educational value.

## Original Issues

The initial version had several areas of redundancy:

1. **Duplicate helper functions**: `track_moves()` and `track_moves_buggy()` were 95% identical
2. **Repetitive direction categorization**: Long if/elif chains appeared multiple times
3. **Similar symmetry tests**: Left/right and up/down tests had nearly identical logic
4. **Scattered pattern explanation**: The fuzzy checking pattern was explained implicitly

## Refactoring Changes

### 1. Consolidated Move Tracking (Lines 98-169)

**Before**: Two nearly identical functions
```python
def track_moves(grid, num_runs, seed_start=1000):
    # ... implementation with CORRECT moves hardcoded ...

def track_moves_buggy(grid, num_runs, seed_start=1000):
    # ... 95% identical implementation with BUGGY moves hardcoded ...
```

**After**: Single parameterized function
```python
CORRECT_MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]]
BUGGY_MOVES = [[-1, 0], [1, 0], [0, -1], [0, -1]]

def track_moves(grid, moves_list, num_runs, seed_start=1000):
    # Single implementation, moves_list determines behavior
```

**Benefit**: 60+ lines of duplication eliminated. Users can easily see the difference between correct and buggy moves at the top of the file.

### 2. Direction Categorization Helper (Lines 102-114)

**Before**: Repetitive if/elif chains
```python
if move == [-1, 0]:
    direction_counts["left"] += 1
elif move == [1, 0]:
    direction_counts["right"] += 1
elif move == [0, -1]:
    direction_counts["up"] += 1
elif move == [0, 1]:
    direction_counts["down"] += 1
```

**After**: Clean mapping function
```python
def categorize_move(move: list[int]) -> str:
    move_map = {
        (-1, 0): "left", (1, 0): "right",
        (0, -1): "up", (0, 1): "down",
    }
    return move_map.get(tuple(move), "unknown")

# Usage:
direction_counts[categorize_move(move)] += 1
```

**Benefit**: More maintainable, easier to extend to diagonal moves if needed.

### 3. Symmetry Test Consolidation (Lines 172-217)

**Before**: Two separate test functions with duplicated logic
```python
def test_correct_version_left_right_symmetry(fuzzy_checker):
    # ... setup ...
    horizontal_moves = counts["left"] + counts["right"]
    fuzzy_checker.fuzzy_assert_proportion(
        counts["left"], horizontal_moves, (0.48, 0.52), ...
    )

def test_correct_version_up_down_symmetry(fuzzy_checker):
    # ... setup ...
    vertical_moves = counts["up"] + counts["down"]
    fuzzy_checker.fuzzy_assert_proportion(
        counts["up"], vertical_moves, (0.48, 0.52), ...
    )
```

**After**: Reusable helper with axis parameter
```python
def check_symmetry(fuzzy_checker, counts, axis, test_name, verbose=True):
    if axis == "horizontal":
        direction_a, direction_b = "left", "right"
    else:  # vertical
        direction_a, direction_b = "up", "down"
    # ... common logic ...

# Usage:
check_symmetry(checker, counts, "horizontal", "left_right_symmetry")
check_symmetry(checker, counts, "vertical", "up_down_symmetry")
```

**Benefit**: Test functions become concise (8 lines vs 20+). Pattern is obvious.

### 4. Added "Fuzzy Checking Pattern" Section (Lines 26-49)

**Before**: Pattern explained only through example tests

**After**: Explicit upfront explanation
```python
"""
The Fuzzy Checking Pattern
---------------------------

Every fuzzy check follows the same simple pattern:

    1. Run many simulations to gather statistics
    2. Count events (numerator) and opportunities (denominator)
    3. Call fuzzy_assert_proportion() with your expectations
    4. Let Bayesian inference decide: bug or no bug?

Example:
    fuzzy_checker.fuzzy_assert_proportion(
        observed_numerator=245,
        observed_denominator=980,
        target_proportion=(0.23, 0.27),
        name="left_moves_proportion"
    )
```

**Benefit**: Novices understand the pattern before seeing variations.

## Impact on Educational Value

### What We Preserved

✅ **All explanatory comments**: Every docstring and inline comment remains
✅ **Step-by-step walkthrough**: The first test still breaks down each step
✅ **Multiple examples**: Different tests demonstrate different properties
✅ **Pedagogical progression**: Simple → complex ordering maintained

### What We Improved

✅ **Clearer pattern**: The fuzzy checking approach is now explicit upfront
✅ **Less overwhelming**: 476 lines vs 508 lines (6% reduction)
✅ **Easier to experiment**: Want to test a new bug? Just pass different moves_list
✅ **Professional example**: Shows how to write maintainable test code

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Lines of code | 508 | 476 | -6% |
| Helper functions | 4 | 3 | -25% |
| Code duplication | ~100 lines | ~0 lines | -100% |
| Test comprehension | Good | Excellent | ✓ |

## Key Takeaways for Novices

1. **DRY doesn't mean less explanation**: We eliminated redundant *code* while keeping all explanatory *comments*

2. **Parameterization is powerful**: `track_moves(grid, BUGGY_MOVES, ...)` is clearer than `track_moves_buggy(grid, ...)`

3. **Helpers improve readability**: `check_symmetry()` makes the test intent obvious

4. **Patterns should be explicit**: Don't make readers infer the pattern from examples

5. **Constants at the top**: `CORRECT_MOVES` and `BUGGY_MOVES` as module-level constants make the difference obvious

## For Tutorial Authors

This refactoring demonstrates that educational code can be both:
- **DRY and maintainable** (professional quality)
- **Well-documented and approachable** (great for learning)

The key is to eliminate *redundant implementation* while preserving *explanatory narrative*.
