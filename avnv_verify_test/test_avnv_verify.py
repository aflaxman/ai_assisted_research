"""Exercise the vivarium_testing_utils avnv verify code.

This script confirms the automated V&V verify pipeline:
1. Fails (reject_null=True / verified=False) when test data clearly disagrees
   with the reference target.
2. Succeeds (reject_null=False / verified=True) when test data is consistent
   with the reference target.

We exercise three layers in increasing order of integration:

  A. FuzzyChecker.test_proportion           -- the scalar statistical engine
  B. FuzzyChecker.test_proportion_vectorized -- vectorized engine called by verify
  C. FuzzyComparison.verify                  -- the comparison-level verify that
                                                ValidationContext.verify calls

Layer C uses lightweight duck-typed bundles to bypass the heavy DataLoader
machinery (sim output dir + artifact files) while still going through the
real verify code path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Shim: vivarium_testing_utils 0.5.4 imports VIVARIUM_COLUMNS from a newer
# vivarium_inputs that isn't on public PyPI. We only need the symbol for
# import-time, not for the verify code path we exercise here.
import vivarium_inputs.globals as _vi_globals  # noqa: E402

if not hasattr(_vi_globals, "VIVARIUM_COLUMNS"):
    _vi_globals.VIVARIUM_COLUMNS = [
        "location",
        "sex",
        "age_start",
        "age_end",
        "year_start",
        "year_end",
    ]

from vivarium_testing_utils.automated_validation.comparison import FuzzyComparison  # noqa: E402
from vivarium_testing_utils.automated_validation.constants import DataSource  # noqa: E402
from vivarium_testing_utils.fuzzy_checker import FuzzyChecker  # noqa: E402


# --------------------------------------------------------------------------- #
# Tiny test harness
# --------------------------------------------------------------------------- #

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    tag = PASS if condition else FAIL
    print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# Layer A: FuzzyChecker.test_proportion (scalar)
# --------------------------------------------------------------------------- #

def test_scalar_proportion() -> None:
    print("\n[A] FuzzyChecker.test_proportion (scalar)")
    fc = FuzzyChecker()

    # Case 1: observed proportion matches the target almost exactly.
    # 10,000 trials with target 0.10 → expect ~1000 events.
    good = fc.test_proportion(
        name="matches_target",
        target_proportion=0.10,
        observed_numerator=1000,
        observed_denominator=10_000,
    )
    check(
        "matches_target: does not reject null",
        not good.reject_null,
        f"bayes_factor={good.bayes_factor:.3g}",
    )

    # Case 2: observed proportion is far from the target.
    # Target 0.10 but we observed 0.50 in 10,000 trials → should be rejected.
    bad_high = fc.test_proportion(
        name="far_above_target",
        target_proportion=0.10,
        observed_numerator=5000,
        observed_denominator=10_000,
    )
    check(
        "far_above_target: rejects null",
        bad_high.reject_null,
        f"bayes_factor={bad_high.bayes_factor:.3g}",
    )
    check(
        "far_above_target: classified as Overestimated",
        bad_high.comparison_to_target == "Overestimated",
        bad_high.comparison_to_target,
    )

    # Case 3: observed proportion is far below the target.
    bad_low = fc.test_proportion(
        name="far_below_target",
        target_proportion=0.50,
        observed_numerator=100,
        observed_denominator=10_000,
    )
    check(
        "far_below_target: rejects null",
        bad_low.reject_null,
        f"bayes_factor={bad_low.bayes_factor:.3g}",
    )
    check(
        "far_below_target: classified as Underestimated",
        bad_low.comparison_to_target == "Underestimated",
        bad_low.comparison_to_target,
    )

    # Case 4: small sample size — should be inconclusive, not a hard fail.
    tiny = fc.test_proportion(
        name="tiny_sample",
        target_proportion=0.10,
        observed_numerator=2,
        observed_denominator=10,
    )
    check(
        "tiny_sample: does not reject null (n=10 too small)",
        not tiny.reject_null,
        f"bayes_factor={tiny.bayes_factor:.3g}, confidence={tiny.confidence}",
    )


# --------------------------------------------------------------------------- #
# Layer B: FuzzyChecker.test_proportion_vectorized
# --------------------------------------------------------------------------- #

def _value_df(values: list[float], index: pd.MultiIndex) -> pd.DataFrame:
    return pd.DataFrame({"value": values}, index=index)


def test_vectorized_proportion() -> None:
    print("\n[B] FuzzyChecker.test_proportion_vectorized (DataFrames)")

    # Build a 4-group stratified dataset. Two groups match target, two do not.
    idx = pd.MultiIndex.from_tuples(
        [("F", "young"), ("F", "old"), ("M", "young"), ("M", "old")],
        names=["sex", "age_group"],
    )

    # Reference (target) proportion: 0.10 in every group.
    target = _value_df([0.10] * 4, idx)

    # Denominators: 10,000 per group.
    denominator = _value_df([10_000.0] * 4, idx)

    # Numerators:
    #   F-young, F-old:   ~target * denom → should PASS
    #   M-young:          way above target → should FAIL
    #   M-old:            way below target → should FAIL
    numerator = _value_df([1000.0, 1010.0, 5000.0, 50.0], idx)

    fc = FuzzyChecker()
    fc.test_proportion_vectorized(
        name="vector_test",
        observed_numerator=numerator,
        observed_denominator=denominator,
        target_proportion=target,
    )

    # Group the per-row diagnostics by (sex, age_group) for the most granular tests.
    by_idx: dict[tuple[Any, ...], Any] = {}
    overall = None
    for r in fc.proportion_test_diagnostics:
        if r.name_additional == "overall":
            overall = r
        elif r.index_info and set(r.index_info.keys()) == {"sex", "age_group"}:
            key = (r.index_info["sex"], r.index_info["age_group"])
            by_idx[key] = r

    check("vectorized: produced overall result", overall is not None)
    check("vectorized: produced all 4 leaf results", len(by_idx) == 4)

    check(
        "F-young (matches target): does not reject",
        not by_idx[("F", "young")].reject_null,
    )
    check(
        "F-old (matches target): does not reject",
        not by_idx[("F", "old")].reject_null,
    )
    check(
        "M-young (5x target): rejects null",
        by_idx[("M", "young")].reject_null,
    )
    check(
        "M-old (0.05x target): rejects null",
        by_idx[("M", "old")].reject_null,
    )

    # Overall pools all four groups; with two strong outliers the pooled
    # proportion (6060/40000 ≈ 0.15) should still differ from target 0.10.
    check(
        "overall (pooled): rejects null due to outlier groups",
        overall is not None and overall.reject_null,
        f"bayes_factor={overall.bayes_factor:.3g}",
    )


def test_vectorized_all_pass() -> None:
    print("\n[B'] test_proportion_vectorized — all groups consistent with target")
    idx = pd.MultiIndex.from_tuples(
        [("F", "young"), ("F", "old"), ("M", "young"), ("M", "old")],
        names=["sex", "age_group"],
    )
    target = _value_df([0.10] * 4, idx)
    denominator = _value_df([10_000.0] * 4, idx)
    # All numerators within statistical noise of target * denominator.
    rng = np.random.default_rng(0)
    nums = rng.binomial(10_000, 0.10, size=4).astype(float).tolist()
    numerator = _value_df(nums, idx)

    fc = FuzzyChecker()
    fc.test_proportion_vectorized(
        name="vector_all_pass",
        observed_numerator=numerator,
        observed_denominator=denominator,
        target_proportion=target,
    )
    any_rejected = any(r.reject_null for r in fc.proportion_test_diagnostics)
    check(
        "all groups + overall: no rejections",
        not any_rejected,
        f"n_tests={len(fc.proportion_test_diagnostics)}",
    )


# --------------------------------------------------------------------------- #
# Layer C: FuzzyComparison.verify  (the function ValidationContext.verify
# delegates to). We duck-type the bundles to bypass DataLoader.
# --------------------------------------------------------------------------- #

@dataclass
class FakeMeasure:
    measure_key: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeMeasure) and self.measure_key == other.measure_key

    def __hash__(self) -> int:
        return hash(self.measure_key)


class FakeBundle:
    """Minimal stand-in for RatioMeasureDataBundle.

    FuzzyComparison.verify uses only:  measure, source, datasets, index_names.
    """

    def __init__(
        self,
        measure: FakeMeasure,
        source: DataSource,
        datasets: dict[str, pd.DataFrame],
    ) -> None:
        self.measure = measure
        self.source = source
        self.datasets = datasets

    @property
    def index_names(self) -> set[str]:
        return {n for df in self.datasets.values() for n in df.index.names}


def _make_comparison(
    num_values: list[float],
    denom_values: list[float],
    target_values: list[float],
    measure_key: str = "cause.test_disease.incidence_rate",
) -> FuzzyComparison:
    idx = pd.MultiIndex.from_tuples(
        [("F", "young"), ("F", "old"), ("M", "young"), ("M", "old")],
        names=["sex", "age_group"],
    )
    measure = FakeMeasure(measure_key)
    test_bundle = FakeBundle(
        measure=measure,
        source=DataSource.SIM,
        datasets={
            "numerator_data": _value_df(num_values, idx),
            "denominator_data": _value_df(denom_values, idx),
        },
    )
    ref_bundle = FakeBundle(
        measure=measure,
        source=DataSource.ARTIFACT,
        datasets={"data": _value_df(target_values, idx)},
    )
    return FuzzyComparison(test_bundle=test_bundle, reference_bundle=ref_bundle)


def test_comparison_verify_passes() -> None:
    print("\n[C] FuzzyComparison.verify — passing case")
    # Target = 0.10, observed ~ binomial(10000, 0.10) per group.
    rng = np.random.default_rng(42)
    nums = rng.binomial(10_000, 0.10, size=4).astype(float).tolist()
    cmp = _make_comparison(
        num_values=nums,
        denom_values=[10_000.0] * 4,
        target_values=[0.10] * 4,
    )

    # Initial state before verify().
    check("verified is None before verify()", cmp.verified is None)

    # step_size=None so target is used as-is (no rate scaling).
    cmp.verify(step_size=None, stratifications="all")

    check(
        "verified is True after passing verify()",
        cmp.verified is True,
        f"overall reject_null={cmp.proportion_test_results['overall'].reject_null}",
    )


def test_comparison_verify_fails() -> None:
    print("\n[C] FuzzyComparison.verify — failing case")
    # Target says 0.10 everywhere; sim says 0.50 everywhere. Should fail loudly.
    cmp = _make_comparison(
        num_values=[5_000.0, 5_000.0, 5_000.0, 5_000.0],
        denom_values=[10_000.0, 10_000.0, 10_000.0, 10_000.0],
        target_values=[0.10] * 4,
    )
    cmp.verify(step_size=None, stratifications="all")

    overall = cmp.proportion_test_results["overall"]
    check(
        "verified is False after failing verify()",
        cmp.verified is False,
        f"overall bayes_factor={overall.bayes_factor:.3g}",
    )
    check(
        "overall failure classified as Overestimated",
        overall.comparison_to_target == "Overestimated",
        overall.comparison_to_target,
    )

    # Every per-group leaf should also flag.
    stratified = cmp.proportion_test_results["stratified"]
    leaf_keys = [k for k in stratified if set(k) == {"sex", "age_group"}]
    assert leaf_keys, "expected a (sex, age_group) stratified entry"
    leaves = stratified[leaf_keys[0]]
    check(
        "all 4 leaf groups reject null",
        all(tr.reject_null for tr in leaves.values()),
        f"{sum(tr.reject_null for tr in leaves.values())}/{len(leaves)} rejected",
    )


def test_comparison_verify_partial_failure() -> None:
    print("\n[C] FuzzyComparison.verify — one bad group fails the comparison")
    rng = np.random.default_rng(7)
    good = rng.binomial(10_000, 0.10, size=3).astype(float).tolist()
    # 3 good + 1 obviously bad (5x target).
    cmp = _make_comparison(
        num_values=good + [5_000.0],
        denom_values=[10_000.0] * 4,
        target_values=[0.10] * 4,
    )
    cmp.verify(step_size=None, stratifications="all")
    check(
        "single bad group flips verified to False",
        cmp.verified is False,
    )


def test_comparison_verify_rejects_bad_sources() -> None:
    print("\n[C] FuzzyComparison.verify — guards against unsupported sources")
    measure = FakeMeasure("cause.test_disease.incidence_rate")
    idx = pd.MultiIndex.from_tuples([("F", "young")], names=["sex", "age_group"])
    # Test bundle deliberately marked as ARTIFACT (not SIM) — verify must refuse.
    bad_test = FakeBundle(
        measure=measure,
        source=DataSource.ARTIFACT,
        datasets={
            "numerator_data": _value_df([10.0], idx),
            "denominator_data": _value_df([100.0], idx),
        },
    )
    ref = FakeBundle(
        measure=measure,
        source=DataSource.ARTIFACT,
        datasets={"data": _value_df([0.1], idx)},
    )
    cmp = FuzzyComparison(test_bundle=bad_test, reference_bundle=ref)
    try:
        cmp.verify(step_size=None, stratifications="all")
    except NotImplementedError as e:
        check(
            "non-SIM test source raises NotImplementedError",
            "SIM" in str(e),
            str(e),
        )
    else:
        check("non-SIM test source raises NotImplementedError", False, "no error raised")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    test_scalar_proportion()
    test_vectorized_proportion()
    test_vectorized_all_pass()
    test_comparison_verify_passes()
    test_comparison_verify_fails()
    test_comparison_verify_partial_failure()
    test_comparison_verify_rejects_bad_sources()

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"  Summary: {n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        print("  Failing checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"    - {name} ({detail})")
        return 1
    print("  All avnv verify checks behave as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
