# Twin-2K-500 Mega-Study: what the repo holds, where the rest lives, and what it would take to point it at IHME instruments

Notes from reading [TianyiPeng/Twin-2K-500-Mega-Study](https://github.com/TianyiPeng/Twin-2K-500-Mega-Study)
(commit `afe2bb9`, 2026-06-08) on 2026-09-21, then the paper's Supplementary
Materials and the two Hugging Face datasets on 2026-09-22, asking: **could this
digital-twin testbed help design better surveys for disability weights (DW) and
verbal and social autopsy (VA/SA)?** The ideas are in [MUSINGS.md](MUSINGS.md).
This file records what exists and where, so the musings rest on verified facts.

## TL;DR

- **Twin-2K-500** is a panel of 2,058 US adults who answered ~500 questions over
  four waves (demographics, 19 personality scales, 11 cognitive measures, economic
  preferences, 16 heuristics-and-biases experiments, a pricing study). A "digital
  twin" is an LLM prompted with one person's full answer history (~128K
  characters, ~30K tokens) and asked to answer new questions as that person.
- **The mega-study** ran 19 pre-registered sub-studies (164 outcomes) on both the
  humans and their twins. Headline: twin-human correlation at the individual
  level averages **r = 0.20**; twins are only modestly better than the same LLM
  with no persona (r = 0.08) or with demographics only (r = 0.15); twin response
  distributions are **under-dispersed in 94% of outcomes**; twins reproduced about
  **half** of the experimental treatment effects. Published as Peng et al.,
  *Digital twins are funhouse mirrors: Five systematic distortions*, Science
  Advances 12, eaeh8260 (2026), doi:10.1126/sciadv.aeh8260 (insufficient
  individuation, stereotyping, representation bias, ideological bias,
  hyper-rationality).
- **Everything needed to build and query the twins is public, but spread over
  four places** (see "Where each piece lives"): the mega-study GitHub repo, the
  older `tianyipeng-lab/Digital-Twin-Simulation` GitHub repo, and two Hugging
  Face datasets. The persona texts, persona summaries, per-study question
  prompts, human answers, and twin answers are all downloadable. The only
  code that is not published anywhere is the mega-study-era prompt formatter,
  the persona-summary generator, and the fine-tuning data prep. All three are
  re-derivable from published inputs and outputs.
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
| Questions in waves 1–3 | ~500 (256 unique Qualtrics QuestionIDs, 760 CSV columns) | HF question catalog |
| Persona battery | demographics 14; personality 279 q / 19 scales / 26 constructs; cognitive 85 q / 11 measures; economic preferences 34 q / 10 measures; heuristics & biases 48 q / 16 experiments; pricing 40 q | mega-study paper |
| Time and pay per participant | ~145 min total, $37 | dataset paper |
| Wave 4 (two weeks after wave 3) | repeats 88 hold-out questions across 17 tasks (the between- and within-subject experiments and the pricing study) | dataset paper |
| Human test-retest accuracy (wave 4 vs. waves 1–3) | 81.7% | dataset paper |
| Twin accuracy on the same hold-outs (GPT-4.1-mini, full text persona) | 71.7%, i.e. 87.7% of the human test-retest ceiling; random guessing 59.2% | dataset paper |
| Persona formats | full text (~128.7K chars), JSON (~167K chars), LLM-style summary (~13K chars), demographics-only (prefix of full), empty | HF parquet, verified |
| License | CC BY 4.0 (panel), Apache-2.0 (mega-study data) | HF dataset cards |

The dataset paper already reported what it called a **hyper-accuracy distortion**:
twins answered knowledge and normative questions "correctly" where humans did
not. Examples it gives: 98.8% of twins knew the number of African UN member
states (eliminating an anchoring effect); 4% of twins refused a vaccine in an
omission-bias task versus about 45% of humans. The mega-study supplement adds
sharper ones: twins identified 99.88% of junk-fee definitions correctly versus
51.75% for humans; 99.9% of twins chose the normative "guesstimate" option
versus 59.4% of humans; nearly 100% of twins reported that their father
completed high school.

## The mega-study in numbers

19 sub-studies proposed by outside scholars (consumer minimalism, default
effects, hiring algorithms, junk fees, misinformation sharing, privacy, story
beliefs, ...), fielded April–June 2025 to 1,784 of the panelists (13,506
participant-sessions), and to their twins. Twins were GPT-4.1 with the full text
persona at temperature 0.7 unless noted. Each human and their twin were assigned
the same experimental condition; image stimuli were replaced by text
descriptions for twins.

| Metric (mean over 164 outcomes) | Full persona | Demographics only | Empty persona | Random |
|---|---|---|---|---|
| Individual-level accuracy, 1 − \|twin − human\| / range | 0.748 | 0.746 | 0.734 | 0.629 |
| Correlation, twin vs. own human | 0.197 | 0.145 | 0.080 | 0.001 |
| SD ratio, twin / human | ~0.63 | ~0.57 | ~0.45 | 1.14 |

Other results worth keeping in mind:

- Best individual correlation among ~25 specifications: temperature 0
  (r ≈ 0.23). Fine-tuning GPT-4.1 on the panel did not help (r ≈ 0.18–0.19).
  Gemini, GPT-5, DeepSeek land in the same 0.19–0.21 band. Llama and Centaur
  personas were far worse (r ≈ 0.07). The persona summary performs about as
  well as the full persona at a tenth of the tokens.
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
- Aggregate rank orders often survive when individual correlation does not:
  privacy-violation ratings correlated ~0 person by person but the six
  advertising scenarios ranked the same (Spearman 0.88); targeting-fairness
  means matched but 63% of twins answered "7" versus 20% of humans.
- A survey of 68 academics, managers and students predicted the human
  treatment effects better than the twin ones (supplement S9).

The repo's own `relevance_analysis/` is a nice touch: an LLM pre-scored every
outcome for how simulable it should be (0–1, with reasoning). Image-based items
and exogenous facts about the respondent (father's education) scored ~0.2;
text-driven attitude items scored 0.8–0.9.

## Where each piece lives

Four public locations. GH = GitHub, HF = Hugging Face.

| Piece | Where | Status |
|---|---|---|
| Human panel answers, raw | HF `LLM-Digital-Twin/Twin-2K-500` → `raw_data/wave_{1..4}_{labels,numbers}_anonymized.csv` | Qualtrics exports, 6–14 MB each |
| Human panel answers, clean | same → `question_catalog_and_human_response_csv/wave1_3_response{,_label}.csv` (2,058 × 761) and `wave4_response{,_label}.csv` (2,058 × 127) | recommended by the authors for analysis |
| Question catalog | same → `question_catalog.json` (256 QuestionIDs: text, type, options, rows, columns, settings, block, CSV columns) plus a README reconciling 256 IDs ≈ 500 questions ≈ 760 columns | complete codebook |
| Scale scores | same → `raw_data/wave {1,2,3} scores.csv` (28, 18, 11 columns: `score_extraversion` … `score_dictator_sender`) | inputs to the persona summary |
| Panel questionnaires | same → `raw_data/questionnaire/Digital_Twins_-_Wave_{1..4} with flow.docx` | Word documents, **not QSF** |
| Full persona (text, JSON, summary) | same → `full_persona/chunks/persona_chunk_{001..007}.parquet`, 29 MB each, 294 rows each | `persona_text` mean 128,654 chars; `persona_summary` 13,019; `persona_json` 166,894; verified against the supplement's S3 examples |
| Train/test split for twin methods | same → `wave_split/chunks/*.parquet`: `wave1_3_persona_{text,json}` (13 blocks, ~95K chars) + `wave4_Q_wave1_3_A` and `wave4_Q_wave4_A` (18 hold-out blocks) | the benchmark the dataset paper used |
| Twin answers to the wave-4 hold-outs | same → `LLM_simulation_results/` (default GPT-4.1-mini run + 12 specification folders: text/JSON/summary/demographics personas, reasoning, repeating questions, fine-tuned, Gemini Flash 2.5), each with llm-vs-wave1-3-vs-wave4 CSVs and accuracy plots | ready-made benchmark results |
| The 19 sub-study instruments and human answers | GH mega-study repo `.dat/<study>/raw_data/{survey.qsf,response.csv}`; identical copy on HF `LLM-Digital-Twin/Twin-2K-500-Mega-Study/.dat/` | 16 MB |
| The exact question prompt each twin saw | HF Mega-Study → `data/<study>-00000-of-00001.parquet`, columns `PID`, `survey_json_with_human_response`, `survey_text` (13,299 rows) | `survey_text` is the rendered "Q1: … Answer: [Masked] … Format Instructions" block, without the persona; verified |
| Twin answers to the 19 sub-studies | GH mega-study repo `results/<study>/<spec>_<date>/` (438 folders, 213 MB); same content as HF Mega-Study `results.zip` (437 folders, 1.55 GB unzipped, plus `__MACOSX` junk) | `consolidated_llm_{values,labels}`, `consolidated_original_answers_values`, `consolidated_correlations`, `meta analysis{, individual level}.csv` |
| QSF → JSON parser, CSV → per-respondent JSON | GH mega-study repo `processing_qualtrics_qsf/`, `processing_qualtrics_csv/` | runs; see feasibility check |
| Persona → text, question → prompt, prompt assembly, LLM runner, post-processing | GH **`tianyipeng-lab/Digital-Twin-Simulation`** (2025-08-04) `text_simulation/`: `convert_persona_to_text.py`, `convert_question_json_to_text.py`, `create_text_simulation_input.py`, `llm_helper.py`, `run_LLM_simulations.py`, `postprocess_responses.py`, `batch_convert_personas.py` | the dataset-paper-era version; **absent from the mega-study repo** |
| Async batch LLM helper with caching and output verification | PyPI `llm-batch-helper` (0.4.0; mega-study pins 0.2.0), GH `TianyiPeng/LLM_batch_helper` | replaces `llm_helper.py` |
| Evaluation of wave-4 predictions | both GH repos `evaluation/` (`json2csv.py`, `mad_accuracy_evaluation.py`, …); mega-study adds `evaluation_engine.py`, `generate_full_report.py` | |
| Meta-analysis over the 19 studies, XGBoost baselines, training-size curves | GH mega-study repo `mega_study_evaluation/`, `ml_prediction/`, `prediction_comparison/`, `post_metric_calculation/` | reproducible from `results/` |
| Per-block codebook | GH `Digital-Twin-Simulation/docs/` (43 block pages by wave), rendered at digital-twin-simulation-version2.readthedocs.io | |
| Quick start without any pipeline | GH `Digital-Twin-Simulation/notebooks/demo_simple_simulation.ipynb`: loads `persona_summary` from HF, asks a new question with GPT-4.1-mini at temperature 0 | the fastest route to a DW-module pilot |
| Prompt templates and persona construction, in prose | Science Advances supplement S2 (full prompt template, example questions, format templates) and S3 (full, summary, demographics, empty personas); S7.2 (fine-tuning recipe) | |

## Persona and prompt formats (from the supplement, verified against the data)

**Full prompt** (S2.1). System instruction: answer the "New Survey Question" as
the person described in the "Persona Profile"; stay consistent with their past
answers; account for human cognitive limitations, uncertainty and biases; follow
formatting instructions. User message: `Persona Profile (This individual's past
survey responses):` + persona text, then `New Survey Question & Instructions
(Please respond as the persona described above):` + the questions, then
`Format Instructions:` with one JSON stub per question:

```
Q2:
How important is it to you that your employer actively invests in ...?
Question Type: Single Choice
Options:
  1 - Not at all important
  ...
  5 - Extremely important
Answer: [Masked]

"Q2": {"Question Type": "Single Choice",
       "Answers": {"SelectedByPosition": Masked,   // a number from 1 to 5
                   "SelectedText": "Masked"}}      // the option text
```

The `survey_text` column of the Mega-Study parquet is exactly this block for
each participant's randomized version of each sub-study.

**Full persona** (S3.1). One entry per question in survey order: question text,
`Question Type:` (Single Choice, Multiple Choice, Matrix, Text Entry, Slider),
`Options:` enumerated `1 - text` (Matrix columns as `1 = text`), then `Answer: 2 -
Female`; Matrix rows are listed `1. Is talkative` each with its own `Answer:`
line. Where a question was asked in both waves 1–3 and wave 4, the wave-4 answer
is used. One persona holds 221 question entries and 631 answer lines. The 14
demographic questions come first, so the demographics persona is a prefix.

**Persona summary** (S3.2). `The following is a description of a person.`, the 14
demographics as `Label: value` lines, then 35 short sections of scale scores with
percentile ranks and a one-sentence gloss of each scale (Big Five, need for
cognition, agency/communion, minimalism, empathy, GREEN, CRT, fluid and
crystallized intelligence, syllogisms, overconfidence, ultimatum game, mental
accounting, social desirability, anxiety, individualism/collectivism, financial
literacy, numeracy, deductive certainty, forward flow, discounting, risk and
loss aversion, trust game, regulatory focus, tightwad-spendthrift, depression,
need for uniqueness, self-monitoring, self-concept clarity, need for closure,
maximization, Wason, dictator game). Wave-4 experiments are excluded. About 3K
tokens.

**Empty persona**: the literal `[Empty Persona Profile]`.

**Fine-tuning** (S7.2): one training example per participant; user message is
the whole Twin-2K-500 questionnaire with every `Answer: [Masked]`, assistant
message is the same text with answers revealed; GPT-4.1, 3 epochs, batch 4,
learning-rate multiplier 2, 65,536-token limit; no sub-study data used.

## What is still missing

Nothing that blocks use, but four things exist only as outputs or prose:

1. **The mega-study-era prompt code.** The mega-study Snakefile calls
   `text_simulation/convert_personas_to_text.py`, `question_formatters/*.py`,
   `llm_batch_helper/*.py`, and `convert_responses_to_csv.py`. None are in that
   repo. The older `Digital-Twin-Simulation` module covers persona → text and
   prompt assembly, but its `convert_question_json_to_text.py` emits the older
   generic format instructions (with an optional `Reasoning` field), not the
   per-question JSON stubs of S2.3; it handles MC, Matrix, TE, Slider and DB
   only (the QSF parser also emits constant-sum, rank and side-by-side); and
   `create_text_simulation_input.py` hard-codes the header and separator that
   the mega-study configs make configurable (`persona_prompt_header`,
   `persona_question_prompt_separator`, `empty_persona_prompt`). Since the
   formatter's outputs for all 13,299 person-studies are on HF, a rewrite can be
   regression-tested exactly.
2. **The persona-summary generator.** Inputs (`scores.csv`, percentiles) and
   outputs (`persona_summary`) are public; the script is not. Needed only to
   summarize a new panel.
3. **Fine-tuning data prep.** `fine_tuning/prepare_finetuning_data.py` is an
   empty file; the recipe is in S7.2.
4. **Wave 1–4 QSF files.** HF ships Word questionnaires and Qualtrics CSV exports,
   and `persona_json` is the already-parsed product, so the panel instrument
   cannot be re-parsed. Not needed unless you want to change the parse.

Also not available to outsiders: the Prolific mapping (anonymized), so only the
Columbia team can re-field the panel; the expectations-survey data of S9; and
any health-status items, because none were asked.

## Gaps and gotchas

1. **`parse_qsf.py` does not create its output directory.** Snakemake normally
   does. Run standalone it fails with `FileNotFoundError` until you `mkdir -p`
   the `wave_qsf_json/` folder. One-line fix.
2. **Qualtrics only.** WHO's 2022 VA instrument ships as an XLSForm for ODK; the
   GBD DW surveys were custom web and household instruments. An XLSForm → the
   repo's JSON template format adapter is the natural bridge (see MUSINGS #17).
3. **US, English, online panel.** Representation bias is documented within the
   US sample; extrapolating to rural LMIC VA respondents is a different problem
   again.
4. **Contamination.** GBD disability-weight tables, WHO VA instruments, the PHMRC
   dataset, InterVA probabilities, and the Tariff paper are all plausibly in
   training data. The mega-study's own default-effect result shows leakage
   inflates apparent fidelity on known paradigms.
5. **No health-status items in the persona.** Self-rated health, chronic
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
flow into this pipeline without parser changes. For a pilot that skips the
pipeline entirely, the sibling repo's demo notebook prompts the HF persona
summaries directly.

## Sources

- Mega-study paper: Peng, Toubia et al., *Digital twins are funhouse mirrors:
  Five systematic distortions*, Science Advances 12, eaeh8260 (2026),
  doi:10.1126/sciadv.aeh8260; preprint [arXiv:2509.19088](https://arxiv.org/abs/2509.19088)
  (earlier titles: *A Mega-Study of Digital Twins Reveals Strengths, Weaknesses
  and Opportunities for Further Improvement*; *Digital Twins as Funhouse
  Mirrors: Five Key Distortions*). Supplementary Materials S1–S10.
- Dataset paper: Toubia, Gui, Peng, Merlau, Li, Chen, *Twin-2K-500: A dataset
  for building digital twins of over 2,000 people based on their answers to
  over 500 questions*, [arXiv:2505.17479](https://arxiv.org/abs/2505.17479);
  *Marketing Science* 44, 1446–1455 (2025).
- Code: [TianyiPeng/Twin-2K-500-Mega-Study](https://github.com/TianyiPeng/Twin-2K-500-Mega-Study),
  [tianyipeng-lab/Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation),
  [TianyiPeng/LLM_batch_helper](https://github.com/TianyiPeng/LLM_batch_helper).
- Data: [LLM-Digital-Twin/Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500),
  [LLM-Digital-Twin/Twin-2K-500-Mega-Study](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500-Mega-Study).
- Project page: [Columbia DAPLab digital twins](https://daplab.cs.columbia.edu/projects/digitaltwins/).
- Numbers in the tables above come from the repo files
  `post_metric_calculation/average_metrics_by_specification.csv`,
  `mega_study_evaluation/meta_analysis_results/summary_by_persona_specification_avg.csv`,
  the paper text and supplement, and my own reads of the HF parquet files.
