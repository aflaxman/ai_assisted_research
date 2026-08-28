# St or Street? An experiment on address standardization direction for record linkage

```python
>>> from rapidfuzz.distance import JaroWinkler
>>> JaroWinkler.similarity("10 main st", "10 main street")     # same address
0.94
>>> JaroWinkler.similarity("maple street", "marple street")    # different streets
0.97
```

Two street names that differ only in suffix convention score *lower* than two
different streets that share a long suffix. That asymmetry is the heart of a
[debate in splink discussion #3250](https://github.com/moj-analytical-services/splink/discussions/3250):
before linking records, should you standardize street suffixes to full words
(St → Street), as the splink docs recommend, or to abbreviations
(Street → St), as the US healthcare interoperability standard
[Project US@](https://oncprojectracking.healthit.gov/wiki/display/TechLabSC/Project+US%40+Home)
specifies? This experiment uses [pseudopeople](https://pseudopeople.readthedocs.io/)
— simulated census-style data with known ground truth and realistic noise — to
measure what the choice actually does to [splink](https://moj-analytical-services.github.io/splink/)
linkage quality.

*(Results summary figure and tables below — see [Results](#results).)*

## TL;DR

- **Standardize; the direction is second-order.** When the two datasets follow
  different conventions, either mapping restores the exact-match rate on true
  pairs (13% → 86% in the messy-noise condition). Which direction you map to
  matters far less than mapping at all.
- **Abbreviation is the safer direction for precision.** Expanding to full
  words roughly doubles the share of *non-matching* address pairs that clear a
  Jaro–Winkler 0.7 threshold (1.5% → 3.0%), because long shared suffixes
  ("...street") pad the similarity of unrelated streets. Splink's trained
  u-probabilities absorb most of this, but the raw comparator separates
  matches from non-matches slightly better on abbreviated strings.
- **Expansion's classic failure is real but contained.** The naive token map
  corrupts Saint-style names ("st clair ave" → "street clair avenue") — 6 of
  4,066 distinct street names here — but it corrupts them *consistently*, so
  linkage barely notices. The collision argument against abbreviation
  (St = Street = Saint) also failed to materialize: both mappings merged
  exactly the same 44 form-pairs in this vocabulary.
- **With a rich model, none of it matters much.** When names and date of birth
  also enter the model, best-case F1 moves by less than replicate noise
  (~0.001) across all treatments — splink's fuzzy comparison levels and
  term-frequency adjustments are doing their job.

## The problem

The splink data-cleaning guidance says: "Replace abbreviations with full words
(e.g., standardize 'St.' and 'Street' to 'Street')." In [discussion
#3250](https://github.com/moj-analytical-services/splink/discussions/3250),
a user asked why — their pipeline abbreviates instead. One reply defended
expansion: abbreviations are many-to-one ("St" conflates Street and Saint),
and fuzzy string metrics disagree on several characters when one side
abbreviates. [Another reply](https://github.com/moj-analytical-services/splink/discussions/3250#discussioncomment-18163210)
pushed back: Project US@, the US standard for patient addressing, standardizes
*to* USPS abbreviations ("ST"), so for US health data the splink guidance is
arguably backwards.

Both sides agree consistency is what matters most. But when you control the
cleaning for both datasets, you still have to pick a direction, and the two
camps predict opposite failure modes:

- **Pro-expansion:** abbreviating collides distinct words; short strings give
  fuzzy metrics less to work with, so typos hurt more.
- **Pro-abbreviation:** long shared suffixes inflate the similarity of
  *different* addresses ("maple street" vs "marple street"), costing
  precision; the discriminating tokens should dominate the comparison.

Simulation can arbitrate, because with simulated data we know the truth.

## The experiment

[pseudopeople](https://pseudopeople.readthedocs.io/) generates census-style
datasets for a simulated US population, with configurable, realistic noise
(typos, OCR errors, phonetic errors, missingness). Its street names arrive
with naturally mixed conventions — the sample data contains "st" (1,209
records), "street" (515), "ave" (864), "avenue" (455), and typo-corrupted
forms like "stree" and "svv" that no cleaner's dictionary will recognize.

**Design.** Each replicate draws the same simulated 2020 census twice with
different noise seeds, giving two extracts of ~10,070 records where every
person appears in both and `simulant_id` provides ground truth. We cross four
factors:

| Factor | Levels |
|---|---|
| Street-name treatment | `none`, `abbreviate` (Street→St), `expand` (St→Street) |
| Convention regime | `consistent` (as generated), `split` (extract A's pipeline abbreviates, extract B's expands — the scenario from the discussion) |
| Street-name noise | `default` (~5% of cells corrupted), `elevated` (~15%) |
| Linkage model | `full` (first, last, DOB, street number, street name), `address_heavy` (first name + street number + street name only) |

with 3 replicates per cell. Both treatments use the same USPS suffix
dictionary ([suffix_maps.py](suffix_maps.py)), applied token-wise in either
direction, so neither direction gets a vocabulary advantage. Treatments and
regimes touch only the `street_name` column; the splink model specification,
blocking rules (which never use `street_name`), and training procedure
(random-sampling u estimation, two EM sessions) are identical in every cell.

**Evaluation.**

1. *Comparator level* ([microbench.py](microbench.py)): Jaro–Winkler
   similarity distributions for ~29,600 true pairs and ~50,000 non-matching
   pairs per cell.
2. *Pipeline level* ([run_experiment.py](run_experiment.py)): full splink
   linkage — estimate parameters, predict, sweep the match-probability
   threshold against ground truth, report best-F1 and average precision, and
   capture the trained m/u parameters of the street-name comparison.

## Results

### The mechanism: what the comparator sees

![Jaro-Winkler survival curves under three treatments](figures/jw_survival_split_elevated.png)

With split conventions and no cleaning (blue), only 13% of true pairs match
exactly, and the whole true-pair curve is shifted left — though note how much
of the damage Jaro–Winkler absorbs on its own: 90% of true pairs still clear
0.92. Either standardization (orange, green) restores the exact-match share
to 86% and the two directions are nearly indistinguishable on true pairs.

The right panel is where the directions separate: among *non-matching* pairs,
expansion (green) roughly doubles the share clearing any threshold between
0.6 and 0.9 relative to abbreviation (orange). Long shared suffixes pad the
similarity of unrelated streets, exactly as the pro-abbreviation camp
predicts. In ROC-AUC terms the comparator differences are tiny (fourth
decimal place), with abbreviation narrowly ahead of expansion in the split
regime.

### The collisions: rarer and tamer than argued

- The Saint problem is real: the naive expander turns "st clair ave" into
  "street clair avenue" (6 of 4,066 distinct street names). But it does so
  *consistently on both sides*, so the corrupted form still matches itself.
- The many-to-one problem did not bite either direction: both mappings merge
  exactly the same 44 pairs of distinct raw forms (e.g., "10th st" ↔
  "10th street"), which is the intended unification, and splink's
  term-frequency adjustment compensates for the merged forms' higher
  frequency.

### The outcome: what splink reports

RESULTS_PLACEHOLDER

## Reproducing

```bash
cd splink_address_abbrev
uv sync
uv run python generate_data.py    # ~3 min: builds data/ from pseudopeople sample population
uv run python microbench.py       # comparator-level analysis -> results/microbench.csv
uv run python run_experiment.py   # 36 splink runs, ~30 min -> results/linkage_results.csv
uv run python run_experiment.py address_heavy   # 36 more -> results/linkage_results_address.csv
uv run python summarize.py        # tables
uv run python make_figures.py     # figures/
uv run pytest                     # tests for the suffix maps
```

## Caveats

- Pseudopeople's sample population is one synthetic metro area (~10k people);
  its street-name vocabulary, typo processes, and the 2.3:1 abbreviated-to-full
  suffix ratio may not match your data. The generator does not model
  *systematic* convention differences between sources, so the split regime is
  imposed by construction.
- Ground-truth addresses are consistent per household before noise. Real
  cross-source address discrepancies (moves, PO boxes, unit-number chaos) are
  harsher than anything simulated here.
- Only token-wise dictionary mapping was tested. A parsing standardizer
  (e.g., libpostal, usaddress) that knows "St" at the start of a name means
  Saint would change the collision story.

## Challenges for the reader

1. Add a `parsed` treatment using libpostal and see whether smart expansion
   beats naive abbreviation.
2. Make the suffix the *only* difference: compare "main st" vs "main street"
   linkage with street-suffix tokens split into their own comparison column.
3. Re-run with pseudopeople's large-scale population (requires access to the
   full simulated US) where street-name term frequencies are far more skewed.

## Further reading

- [Splink discussion #3250](https://github.com/moj-analytical-services/splink/discussions/3250)
- [Project US@ Technical Specification](https://oncprojectracking.healthit.gov/wiki/display/TechLabSC/Project+US%40+Home) — the US healthcare address standard (Appendix B: USPS abbreviations)
- [USPS Publication 28](https://pe.usps.com/text/pub28/welcome.htm) — street suffix abbreviations
- [pseudopeople documentation](https://pseudopeople.readthedocs.io/)
- [Splink documentation](https://moj-analytical-services.github.io/splink/)
