# pseudopeople noise questions

Dan Weinberg, at the Census Bureau's Center for Statistical Research and Methodology, sent me seven questions about how pseudopeople generates noise and duplicate records in its decennial census dataset. The documentation answers some of them and is ambiguous on others, so `noise_questions.ipynb` answers each one twice: once by reading the code, and once by running pseudopeople 1.2.8 on the 10,000-record sample census and counting what comes out. The notebook is committed with its outputs, so the answers can be read without running anything, and each section links to the code that implements the behavior, pinned to the [v1.2.8 tag](https://github.com/ihmeuw/pseudopeople/tree/v1.2.8) so that line numbers stay put.

## The answers in brief

1. **Several noise types can hit one first name.** The six types run one after another in the order fixed in [noise_entities.py](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_entities.py#L13-L114), each choosing its cells independently, so a name can become a nickname and then acquire a typo; in the sample, 671 first names were selected by one type and 10 by two. Blanking is the exception, since it runs first and blank cells are [excluded from every later type](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/utilities.py#L79-L83).
2. **A nickname is a uniform draw** ([use_nicknames](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_functions.py#L555-L582)) among the nicknames listed for that exact name in the packaged table of 1,080 names. Names without an entry (about 40% of first names) are never nicknamed, and the selection rate is [scaled up by one over the eligible share](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_scaling.py#L26-L34) so that a 1% `cell_probability` changes 1% of all first names.
3. **A fake name is a uniform draw** ([use_fake_names](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_functions.py#L585-L626)) from a fixed list of 90 fake first names (87 fake last names) transcribed from NORC's 2011 assessment of the Census Bureau's Person Identification Validation System.
4. **The OCR link is dead** because it points at a `develop` branch that no longer exists ([column_noise.rst, line 345](https://github.com/ihmeuw/pseudopeople/blob/main/docs/source/noise/column_noise.rst#L345)). The file is at `https://github.com/ihmeuw/pseudopeople/blob/main/src/pseudopeople/data/ocr_errors.csv`, the phonetic-variations link ([line 317](https://github.com/ihmeuw/pseudopeople/blob/main/docs/source/noise/column_noise.rst#L317)) has the same problem, and both files ship inside the installed package.
5. **More than one ZIP digit can be wrong.** [write_wrong_zipcode_digits](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_functions.py#L407-L459) corrupts each of the five positions independently, at 4%, 4%, 20%, 36% and 36% by default, and a wrong digit is uniform over the nine other digits. About 30% of selected ZIP codes come through unchanged, so the realized rate at the default settings is about 0.7% rather than 1%.
6. **The 2% in duplicate-with-guardian is a per-eligible-child probability.** [duplicate_with_guardian](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_functions.py#L161-L303) draws 2% of all household children with a guardian recorded and keeps only the drawn children whose guardian lives elsewhere, so of Dan's 100 such children, 2 duplicates are expected, with binomial spread and no guarantee. Setting the probability to 1.0 duplicates every eligible child exactly once (155 of 155 in the sample). The documentation's ["maximum probability"](https://github.com/ihmeuw/pseudopeople/blob/main/docs/source/noise/row_noise.rst#L30-L36) and the [warning](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/configuration/validator.py#L260-L339) that enforces it describe a scaling over the eligible subset, as is done for nicknames, that the duplication code never performs, so the documentation and its warning should be reworded, or the code changed to match, which would multiply the default number of duplicates by about seven.
7. **There is no other duplication in the package.** Generic `duplicate_row` is a commented-out stub ([noise_entities.py, line 37](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_entities.py#L37); [noise_functions.py, lines 145 to 158](https://github.com/ihmeuw/pseudopeople/blob/v1.2.8/src/pseudopeople/noise_functions.py#L145-L158)). Dan's two-seed workaround is sound if the injected records are sampled from simulants present in both runs (nonresponse differs by seed) and if one accepts that about half of the injected duplicates are exact copies at the default noise levels. It cannot produce what guardian duplication produces, a second record at a different address.

## Running the notebook

```bash
cd pseudopeople_noise_questions
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace noise_questions.ipynb
```

The whole notebook runs in about 45 seconds on the sample data. pseudopeople 1.2.8 fails with pandas 3 (`TypeError: Cannot change data-type for string array` in the token-corruption code), so `pyproject.toml` pins `pandas<3`; version 1.2.9 on the `main` branch fixes this.

## Files

- `noise_questions.ipynb`: the questions, the code-level answers, and the runs that confirm them, with outputs.
- `pyproject.toml`, `uv.lock`: the environment.
