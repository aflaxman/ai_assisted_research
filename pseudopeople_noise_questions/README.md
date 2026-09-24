# pseudopeople noise questions

Dan Weinberg, at the Census Bureau's Center for Statistical Research and Methodology, sent me seven questions about how pseudopeople generates noise and duplicate records in its decennial census dataset. The documentation answers some of them and is ambiguous on others, so `noise_questions.ipynb` answers each one twice: once by reading the code, and once by running pseudopeople 1.2.8 on the 10,000-record sample census and counting what comes out. The notebook is committed with its outputs, so the answers can be read without running anything.

## The answers in brief

1. **Several noise types can hit one first name.** The six types run one after another, each choosing its cells independently, so a name can become a nickname and then acquire a typo; in the sample, 671 first names were selected by one type and 10 by two. Blanking is the exception, since it runs first and blank cells are excluded from every later type.
2. **A nickname is a uniform draw** among the nicknames listed for that exact name in the packaged table of 1,080 names. Names without an entry (about 40% of first names) are never nicknamed, and the selection rate is scaled up by one over the eligible share so that a 1% `cell_probability` changes 1% of all first names.
3. **A fake name is a uniform draw** from a fixed list of 90 fake first names (87 fake last names) transcribed from NORC's 2011 assessment of the Census Bureau's Person Identification Validation System.
4. **The OCR link is dead** because it points at a `develop` branch that no longer exists. The file is at `https://github.com/ihmeuw/pseudopeople/blob/main/src/pseudopeople/data/ocr_errors.csv`, the phonetic-variations link has the same problem, and both files ship inside the installed package.
5. **More than one ZIP digit can be wrong.** Each of the five positions is corrupted independently, at 4%, 4%, 20%, 36% and 36% by default, and a wrong digit is uniform over the nine other digits. About 30% of selected ZIP codes come through unchanged, so the realized rate at the default settings is about 0.7% rather than 1%.
6. **The 2% in duplicate-with-guardian is a per-eligible-child probability.** Of Dan's 100 children with a guardian at another address, 2 duplicates are expected, with binomial spread and no guarantee. Setting the probability to 1.0 duplicates every eligible child exactly once (155 of 155 in the sample). The documentation's "maximum probability" describes a scaling over the eligible subset, as is done for nicknames, that the duplication code never performs, so the documentation and its warning should be reworded, or the code changed to match, which would multiply the default number of duplicates by about seven.
7. **There is no other duplication in the package.** Generic `duplicate_row` is a commented-out stub. Dan's two-seed workaround is sound if the injected records are sampled from simulants present in both runs (nonresponse differs by seed) and if one accepts that about half of the injected duplicates are exact copies at the default noise levels. It cannot produce what guardian duplication produces, a second record at a different address.

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
