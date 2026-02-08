# Data Note Outline: Florida-Scale Pseudopeople Linkage Benchmark

## 1. Working title
**Data Note:** A Florida-scale synthetic registry and finder-file benchmark for probabilistic record linkage using pseudopeople (2020 baseline; GBD 2023 prevalence weighting)

## 2. Abstract (structured, 200–300 words)
- **Background:** Synthetic population subset designed to prototype cancer registry linkage workflows.
- **Methods:** Stratified, prevalence-weighted sampling from Florida pseudopeople 2020 census; creation of three datasets with controlled overlap; controlled missingness and error in SSN and address fields.
- **Data records:** Three CSV/Parquet-style datasets (df_2, df_1a, df_1b) plus changelog and provenance notes.
- **Validation:** Checks for target sizes and overlaps; missingness rates; distributional checks of age and sex.
- **Usage:** Intended for linkage benchmarking and privacy-preserving record linkage exercises.

## 3. Introduction (IJPDS-aligned narrative)
- **Motivation:** Benchmark linkage accuracy and scalability for registry-scale linkages (5M registry file; 100k/500k finder files; ~20% overlap).
- **Applied setting:** Florida Cancer Data System (FCDS) linkage workflows and tool landscape (Match*Pro, fastLink, Splink).
- **Gap:** Realistic, reproducible, shareable benchmarks with **truth labels** are rare at this scale.
- **Contribution:** Reproducible benchmark dataset + scenario matrix + evaluation harness + governance/ethics framing.
- **Context:** Derived from pseudopeople simulated population (USA, 2.0.0) with Florida 2020 census records.
- **Key design choices:**
  - Prevalence-weighted sampling to enrich for cancer cases.
  - Three datasets with partial overlap to simulate registry–finder linkage scenarios.
  - Field-level perturbations (missingness, SSN transposition) to emulate data quality issues.

## 4. Data products
### 4.1 Files / tables
- **Registry file (df_2 / “FCDS-side”):** ~5,000,000 records, Florida pseudopeople subset (seeded/reproducible).
- **Finder files (df_1a / df_1b / “client-side”):** 500,000 and 100,000 records, with controlled overlap.
- **Truth labels:** mapping from finder records to registry records via `simulant_id` (TP/FP/FN/TN definable).
- **Scenario configs:** address missingness + SSN missingness/transposition regimes (finder vs registry).

### 4.2 Key variables
- Identifiers: first/last name, sex, DOB (or birth year), address components, SSN (for SSN-enabled variants).
- **Ground-truth fields:**
  - `simulant_id` (stable synthetic person identifier used to define matches).
  - `in_overlap` flag (convenience indicator for constructed overlap between finder and registry).
- Linkage keys are synthetic but realistic; dataset includes pseudopeople’s modeled duplication and inherent missingness.

## 5. Methods (core technical content)
### 5.1 Source data and environment
- Source: pseudopeople simulated population USA 2.0.0 (Florida, 2020 census, plus SSN observer table).
- Tooling: DuckDB for filtering and weighted sampling; pandas/numpy for post-processing.
- Reproducibility seeds and configuration.

### 5.2 Deduplication and Florida subset
- Build deduplicated views of decennial census and SSN observer tables.
- Filter to state = FL.
- Record-count checks on extracted Florida subset.

### 5.3 Cancer prevalence weighting
- Use age/sex-specific prevalence rates (GBD 2023, neoplasms, Florida).
- Collapse granular infant age groups to 0–1 bucket.
- Convert prevalence percent to probability weights for sampling.

### 5.4 Dataset design and target sizes
- **df_2 (Cancer cohort):** 5,000,000 records, prevalence-weighted sampling.
- **df_1a (Finder A):** 500,000 records, includes overlap with df_2.
- **df_1b (Finder B):** 100,000 records, includes overlap with df_2.
- Overlaps: 100,000 (df_1a–df_2) and 20,000 (df_1b–df_2).
- Oversampling factor (1.5x) to account for downstream non-response.

### 5.5 Default pseudopeople noise and modeled artifacts
- Document pseudopeople’s built-in behaviors that affect linkage realism:
  - Default row/column noise types and their independence across columns.
  - Modeled duplication (e.g., duplicate-with-guardian behavior) if present in the generated dataset.
  - Inherent missingness due to inapplicable fields (e.g., address components like unit numbers).
- Clearly distinguish:
  - **(A) pseudopeople defaults** (present even with `config=None`),
  - **(B) user-applied config noise** (address blanks), and
  - **(C) user post-processing noise** (SSN missingness targets + transpositions).

### 5.6 Address missingness configuration
- Address missingness applied via `config` in `generate_decennial_census`.
- Registry vs finder asymmetry (1% vs 5%) to mirror FCDS benchmarking regimes.
- Document which address components were subjected to missingness and how empirical missingness was verified.

### 5.7 SSN attachment and noise
- SSNs are joined from the SSN observer parquet(s) by `simulant_id`.
- Noise is applied in two steps:
  1. **Target-based missingness** (hit a specified missingness rate accounting for inherent missingness).
  2. **Digit transpositions** (adjacent swap) at a specified rate among non-missing SSNs.
- Document at outline level:
  - Deduplication approach for multiple SSNs per `simulant_id`.
  - Missingness targets for registry vs finder files.
  - Transposition targets for registry vs finder files.
  - Planned verification by storing an `orig_ssn` column (or hash) to compute empirical transposition rates.

### 5.8 Post-processing and exact overlap enforcement
- Identify survivor sets after noise generation.
- Enforce exact overlap counts via downsampling with fixed RNG seed.
- Final verification of sizes and overlap counts.

## 6. Data records (IJPDS format)
- Directory structure and files (three dataset folders + changelog).
- Data schema highlights: `simulant_id`, demographics, address fields, SSN field.
- Record counts per dataset and overlap counts.

## 7. Validation and verification (V&V)
### 7.1 Verification (data construction targets)
- Reproducibility via seeds: stable counts, hashes, overlap sizes.
- Empirical missingness checks:
  - Address component missingness matches configured `addr_missing`.
  - SSN missingness hits target accounting for inherent missingness.
- **Typographic error rate checks:**
  - SSN transposition rate among non-missing SSNs matches target.
  - Optional: character edit rates in address fields if pseudopeople column noise is enabled.
- Consistency checks:
  - Unique `simulant_id` within each dataset.
  - Overlap sizes between registry and finder files match targets.
  - Duplicate-with-guardian and other row-noise artifacts documented if present.

### 7.2 Validation (fit-for-purpose realism)
- Age/sex distribution for registry-like subset vs target (GBD prevalence rationale).
- Identifier “difficulty” diagnostics:
  - Name frequency distributions (blocking difficulty).
  - Address completeness and ZIP stability.
  - Blocking candidate counts under typical linkage strategies.
- Linkage difficulty checks under strict matching (ceiling) and relaxed matching (expected degradation).

## 8. Scenario matrix and benchmarking tasks
- Overlap: 20% overlap between finder and registry; justify realism and stability.
- Tasks:
  - 5M × 500k main benchmark.
  - 5M × 100k sensitivity benchmarks (two variants).
- Record linkage tools: Match*Pro, fastLink, Splink.

## 9. Usage notes
- Suggested use cases: linkage benchmarking, algorithm evaluation, privacy-preserving linkage.
- Known limitations: synthetic population assumptions; prevalence weighting based on GBD 2023 estimates.

## 10. Ethical considerations and governance
- Synthetic data (no real individuals), but still treat as sensitive for linkage methods.
- Discussion of responsible use and limitations for inference.
- Redaction/remediation pathway to remove or perturb specific records and re-issue versions.
- Release strategy decision (open vs gated; possible tiered release).
- Institutional consultation (planned): privacy/compliance review.

## 11. Reflections and lessons learned
- Operational overhead (permissions, downloads, compute) shaped the project timeline.
- DuckDB integration materially improved feasibility; share scripts and timings.
- Score disagreement across tools highlights need for standardized evaluation endpoints.

## 12. Code availability and reproducibility
- Notebook provenance and dependency list.
- Notes on parameters and seeds.

## 13. Limitations and future work
- Explore alternate noise models.
- Expand to multi-state or multi-year coverage.
- Validate against additional linkage quality metrics.
- **Future work:** contribute a pseudopeople patch that exposes SSN transpositions as a configurable noise type.

## 14. Comparison to common data note styles
- **IJPDS Data Note:** Emphasize data source, processing pipeline, validation, and usage/limitations sections.
- **Gates Open Research-style (verbal autopsy narratives):** Clear narrative with reproducibility and ethical considerations; apply concise storytelling in Introduction and Usage Notes.

## 15. Proposed next steps for drafting
- Confirm final dataset counts and location on storage.
- Verify GBD prevalence summary table for Florida 2023.
- Add data dictionary and schema appendix.
