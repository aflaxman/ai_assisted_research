# Twin-2K-500 Mega-Study: what the repo holds, and what it would take to point it at IHME instruments

Notes from reading [TianyiPeng/Twin-2K-500-Mega-Study](https://github.com/TianyiPeng/Twin-2K-500-Mega-Study)
(commit `afe2bb9`, 2026-06-08) on 2026-09-21, asking: **could this digital-twin
testbed help design better surveys for disability weights (DW) and verbal and
social autopsy (VA/SA)?** The ideas are in [MUSINGS.md](MUSINGS.md). This file
records what the repo and its two papers actually contain, so the musings rest
on verified facts.

## TL;DR

- **Twin-2K-500** is a panel of 2,058 US adults who answered ~500 questions over
  four waves (demographics, 19 personality scales, 11 cognitive measures, economic
  preferences, 16 heuristics-and-biases experiments, a pricing study). A "digital
  twin" is an LLM prompted with one person's full answer history (~128K
  characters) and asked to answer new questions as that person.
- **The mega-study** ran 19 pre-registered sub-studies (164 outcomes) on both the
  humans and their twins. Headline: twin-human correlation at the individual
  level averages **r = 0.20**; twins are only modestly better than the same LLM
  with no persona (r = 0.08) or with demographics only (r = 0.15); twin response
  distributions are **under-dispersed in 94% of outcomes**; twins reproduced about
  **half** of the experimental treatment effects. The final version of the paper
  is titled *"Digital Twins as Funhouse Mirrors: Five Key Distortions"*
  (insufficient individuation, stereotyping, representation bias, ideological
  bias, hyper-rationality).
- **The repo is a testbed, not a turnkey twin generator.** It ships the 19
  sub-studies' Qualtrics surveys and human responses, twin outputs for ~440
  model/prompt specifications, and the evaluation code. The `text_simulation/`
  module that turns personas into prompts and calls the LLM is referenced
  everywhere but **is not in the repository** (0 tracked files).
- **The pipeline is Qualtrics-only** (QSF in, CSV out). I confirmed the QSF parser
  runs on the bundled surveys with three pip dependencies. A WHO VA instrument
  (XLSForm/ODK) would need an adapter.
- **Nothing in the persona battery is about health status** beyond depression and
  anxiety scales. A DW module could be added to the panel; a VA twin would need a
  different kind of persona entirely (see MUSINGS).

## The panel and the twins

| Item | Value | Source |
|---|---|---|
| Participants completing all four waves | 2,058 (2,509 started wave 1) | dataset paper |
| Questions in waves 1–3 | ~500 (256 unique question IDs) | HF dataset card |
| Persona battery | demographics 14; personality 279 q / 19 scales / 26 constructs; cognitive 85 q / 11 measures; economic preferences 34 q / 10 measures; heuristics & biases 48 q / 16 experiments; pricing 40 q | mega-study paper |
| Time and pay per participant | ~145 min total, $37 | dataset paper |
| Wave 4 (two weeks after wave 3) | repeats 88 hold-out questions across 17 tasks (the between- and within-subject experiments and the pricing study) | dataset paper |
| Human test-retest accuracy (wave 4 vs. waves 1–3) | 81.7% | dataset paper |
| Twin accuracy on the same hold-outs (GPT-4.1-mini, full text persona) | 71.7%, i.e. 87.7% of the human test-retest ceiling; random guessing 59.2% | dataset paper |
| Persona formats | full text (~128K chars), JSON, LLM-written summary (~13K chars), demographics-only, empty | repo configs |
| License | CC BY 4.0 on Hugging Face (`LLM-Digital-Twin/Twin-2K-500`) | HF dataset card |

The dataset paper already reported what it called a **hyper-accuracy distortion**:
twins answered knowledge and normative questions "correctly" where humans did
not. Examples it gives: 98.8% of twins knew the number of African UN member
states (eliminating an anchoring effect); 4% of twins refused a vaccine in an
omission-bias task versus about 45% of humans.

## The mega-study in numbers

19 sub-studies proposed by outside scholars (consumer minimalism, default
effects, hiring algorithms, junk fees, misinformation sharing, privacy, story
beliefs, ...), fielded April–June 2025 to 1,784 of the panelists (13,506
participant-sessions), and to their twins. Twins were GPT-4.1 with the full text
persona at temperature 0.7 unless noted.

| Metric (mean over 164 outcomes) | Full persona | Demographics only | Empty persona | Random |
|---|---|---|---|---|
| Individual-level accuracy, 1 − \|twin − human\| / range | 0.748 | 0.746 | 0.734 | 0.629 |
| Correlation, twin vs. own human | 0.197 | 0.145 | 0.080 | 0.001 |
| SD ratio, twin / human | ~0.63 | ~0.57 | ~0.45 | 1.14 |

Other results worth keeping in mind:

- Best individual correlation among ~25 specifications: temperature 0
  (r ≈ 0.23). Fine-tuning GPT-4.1 on the panel did not help (r ≈ 0.18–0.19).
  Gemini, GPT-5, DeepSeek land in the same 0.19–0.21 band. Llama and Centaur
  personas were far worse (r ≈ 0.07).
- Twin means differ from human means by 0.35 SD on average; significantly in
  64% of outcomes.
- Full-persona twins are closer to demographics-only twins (MAD 0.13) and to
  empty-persona twins (0.18) than to their own humans (0.25). That is the
  paper's evidence for "stereotyping": the persona shifts answers toward what
  the model expects of the demographic, not toward the person.
- XGBoost trained on the same persona features matches the twins' correlation
  with ~180 human training cases and their accuracy with ~75. XGBoost with 650
  cases tops out below r = 0.29. So a twin is "worth" on the order of 100
  humans of information for predicting a new outcome, and the ceiling from
  these features is low for everyone.
- Twins were more accurate for higher-education, higher-income, moderate,
  religious-attending respondents (representation bias), tilted
  pro-technology and pro-trust (ideological bias), and chose the normative
  option in bounded-rationality tasks (hyper-rationality). They reproduced a
  classic default-effect paradigm but not a novel one, which the authors read
  as training-data leakage.

The repo's own `relevance_analysis/` is a nice touch: an LLM pre-scored every
outcome for how simulable it should be (0–1, with reasoning). Image-based items
and exogenous facts about the respondent (father's education) scored ~0.2;
text-driven attitude items scored 0.8–0.9.

## What is in the repository

| Path | Contents |
|---|---|
| `processing_qualtrics_qsf/` | QSF → structured JSON (blocks, randomizers, embedded data, branch logic; MC, matrix, slider, text, rank, constant-sum, side-by-side question types) |
| `processing_qualtrics_csv/` | Human CSV responses → one JSON per respondent, filled into the template |
| `configs/<study>/*.yaml` | 492 configs: 19 studies × ~26 specifications (model, temperature, persona variant, reasoning on/off) rendered from `configs/study_template.yaml.j2` |
| `.dat/<study>/raw_data/` | The 19 sub-studies' `survey.qsf` and anonymized human `response.csv` (16 MB total) |
| `results/<study>/<spec>_<date>/` | Twin outputs for 438 study × spec runs: `consolidated_llm_values.csv.gz` next to `consolidated_original_answers_values.csv.gz`, same columns, one row per twin/human (213 MB) |
| `mega_study_evaluation/` | Per-study meta-analysis scripts (converted from notebooks), combined tables in `meta_analysis_results/` |
| `post_metric_calculation/` | Seven human-vs-twin vector metrics per outcome (correlation, two accuracies, Wasserstein, SD ratio, mean difference, Cohen's d) plus a random benchmark |
| `prediction_comparison/`, `ml_prediction/` | XGBoost baselines and training-size curves |
| `fine_tuning/` | OpenAI fine-tuning job scripts (data-prep script is an empty file) |
| `relevance_analysis/` | LLM-judged simulability score per outcome |
| `Snakefile`, `scripts/setup_study.py` | Workflow: `process_qsf → process_csv → convert_questions → create_simulation_input → run_llm_simulation → convert_to_csv`; `setup_study.py <name> --qsf ... --csv ...` scaffolds a new study |
| `download_dataset.py` | Pulls personas, hold-out answer blocks, and raw wave CSVs from Hugging Face |

Twin prompts are documented in the configs. System instruction (paraphrased):
*answer the new survey question as the person described by their past survey
responses; stay consistent with their answers; account for human cognitive
limitations, uncertainty, and biases; return JSON in the given schema.* The user
prompt is `## Persona Profile` + full persona text + `## New Survey Question`.

## Gaps and gotchas

1. **`text_simulation/` is missing.** The Snakefile's `convert_questions`,
   `create_simulation_input`, and `run_llm_simulation` rules call
   `text_simulation/*.py`, and every config writes there, but the directory is
   not tracked. The human-side testbed and all evaluation code are reproducible;
   generating new twin responses is not, as cloned. Reimplementing it is
   feasible (the formats are documented in `docs/` and the configs), and worth
   an issue upstream.
2. **`parse_qsf.py` does not create its output directory.** Snakemake normally
   does. Run standalone it fails with `FileNotFoundError` until you `mkdir -p`
   the `wave_qsf_json/` folder. One-line fix.
3. **Qualtrics only.** WHO's 2022 VA instrument ships as an XLSForm for ODK; the
   GBD DW surveys were custom web and household instruments. An XLSForm → the
   repo's JSON template format adapter is the natural bridge (see MUSINGS #16).
4. **US, English, online panel.** Representation bias is documented within the
   US sample; extrapolating to rural LMIC VA respondents is a different problem
   again.
5. **Contamination.** GBD disability-weight tables, WHO VA instruments, the PHMRC
   dataset, InterVA probabilities, and the Tariff paper are all plausibly in
   training data. The mega-study's own default-effect result shows leakage
   inflates apparent fidelity on known paradigms.
6. **No health-status items in the persona.** Self-rated health, chronic
   conditions, disability, caregiving experience: none are asked. Depression
   (wave 2) and anxiety scales are the closest.

## Feasibility check I ran

From a clone with `data → .dat` symlinked:

```bash
mkdir -p data/targeting_fairness/wave_qsf_json
uv run --no-project --with beautifulsoup4 --with pyyaml --with lxml \
    python processing_qualtrics_qsf/parse_qsf.py \
    --config configs/targeting_fairness/targeting_fairness.yaml
```

Output: a JSON template with `Metadata` and an `Elements` tree (EmbeddedData →
intro Block → Randomizer over two scenario arms → Block with a 9-point MC
`fair1` item and a Timing item → conclusion Block with free text). A GBD-style
paired-comparison item ("Which person do you think is healthier?" with two
options) is an ordinary MC question in this schema, and randomized pair
assignment is an ordinary Randomizer. So a DW module authored in Qualtrics would
flow into this pipeline without parser changes.

## Sources

- Mega-study paper (final version): Peng, Toubia et al., *Digital Twins as
  Funhouse Mirrors: Five Key Distortions*, [arXiv:2509.19088](https://arxiv.org/abs/2509.19088)
  (earlier title: *A Mega-Study of Digital Twins Reveals Strengths, Weaknesses
  and Opportunities for Further Improvement*).
- Dataset paper: Toubia, Gui, Peng, Merlau, Li, Chen, *Twin-2K-500: A dataset
  for building digital twins of over 2,000 people based on their answers to
  over 500 questions*, [arXiv:2505.17479](https://arxiv.org/abs/2505.17479);
  *Marketing Science* 2025 database report.
- Dataset: [huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- Project page: [Columbia DAPLab digital twins](https://daplab.cs.columbia.edu/projects/digitaltwins/).
- Numbers in the tables above come from the repo files
  `post_metric_calculation/average_metrics_by_specification.csv`,
  `mega_study_evaluation/meta_analysis_results/summary_by_persona_specification_avg.csv`,
  and the paper text.
