# avnv verify smoke test

A self-contained script that exercises the automated V&V (`avnv`) `verify`
code in `vivarium_testing_utils` and confirms it succeeds when it should and
fails when it should.

## What it tests

The script drives three layers of the verify pipeline:

| Layer | Function under test | Purpose |
|-------|---------------------|---------|
| A | `FuzzyChecker.test_proportion` | Scalar Bayesian proportion test |
| B | `FuzzyChecker.test_proportion_vectorized` | DataFrame-level test used by `verify` |
| C | `FuzzyComparison.verify` | The comparison-level entry point that `ValidationContext.verify` delegates to |

Layer C uses tiny duck-typed `FakeBundle` objects so we exercise the real
`verify` code path without standing up a full simulation output directory and
artifact.

## Scenarios covered

For each layer the script checks both directions:

- Observed proportion matches target ⇒ `reject_null = False`, `verified = True`
- Observed proportion 5× target ⇒ `reject_null = True`, classified `Overestimated`
- Observed proportion 0.05× target ⇒ `reject_null = True`, classified `Underestimated`
- Tiny sample (n=10) ⇒ does not spuriously reject, reports `Inconclusive`
- Mixed stratifications: one bad group flips the comparison's `verified` to `False`
- Non-SIM test source ⇒ `verify` raises `NotImplementedError`

## Quickstart

```bash
uv sync
uv run python test_avnv_verify.py
```

Expected final line:

```
Summary: 21/21 checks passed
All avnv verify checks behave as expected.
```

## Notes

`vivarium_testing_utils==0.5.4` imports `VIVARIUM_COLUMNS` from a newer
`vivarium_inputs` than is published on public PyPI. The script injects a
fallback `VIVARIUM_COLUMNS` into `vivarium_inputs.globals` before importing
the package; the symbol is unused by the verify code path under test.
