# Plain-English writing rules for Claude Code: how the variants pan out

Notes and an experiment on [ihmeuw/vivarium-suite PR #304](https://github.com/ihmeuw/vivarium-suite/pull/304),
"Add plain-English writing rules to agents and sessions", read against
[Karpathy's thread](https://twitter-thread.com/t/2105819303471976479) on getting readable output from LLMs.

The notebook with every output side by side is [`variants_notebook.ipynb`](variants_notebook.ipynb).

## What PR #304 does

The PR makes agent prose (review reports, PR text, tickets, commit messages, chat replies)
plain technical English, based on ASD-STE100 Simplified Technical English. It ships two things:

1. **A one-line rule in nine agent definitions** (the five `_review_*` agents and `_split_proposer`
   in `simsci`; `_vv_writer`, `_claim_auditor`, `_duplicate_finder` in `simsci-internal`):

   > Write prose for people in plain technical English. Use short sentences in the active voice and
   > simple tenses, and one name for each thing. Do not use semicolons, Latin abbreviations, or
   > metaphors. Put code identifiers and paths in backticks.

2. **A `SessionStart` hook** in `simsci-internal` that adds a 427-character text
   (`hooks/writing-style.txt`) to the start of each main session, again after `/clear`, compaction,
   resume, and fork. `SIMSCI_WRITING_STYLE=off` disables it.

   > Prose for people in this project is plain technical English. Sentences are short, in the active
   > voice, and in simple tenses. Each thing has one name. The text has no semicolons, contractions,
   > Latin abbreviations, or metaphors. Code identifiers and paths are in backticks. A guess or a
   > judgment is marked as one, and a reason that the source does not give is not stated as fact.
   > Code and quoted text keep their own conventions.

The PR went through several designs (a `writing-style` skill, a checker script `ste_check.py`, a
drift check) and removed them after evals showed the skill-plus-checker design added nothing over the
hook text and cost 35% more. Its eval results (255 runs, blind model judges, 1 to 5 scale):

| Subagent version | Clarity | Actionability | Recall | Cost per run |
|---|---|---|---|---|
| no rules | 3.70 | 4.27 | 1.00 | $0.024 |
| one-line copy in each agent (shipped) | 4.40 | 4.63 | 0.99 | $0.022 |
| `skills:` preload | 4.13 | 4.43 | 1.00 | $0.032 |
| Skill tool call | 3.87 | 4.13 | 0.94 | $0.040 |

For the main session, the hook text raised clarity by +0.73 and accuracy by +0.23 against no rules.
A longer 672-character text that dropped the clause "a guess or a judgment is marked as one" lost
accuracy (-0.53). And the finding most relevant to Karpathy's suggestion: **naming ASD-STE100 in the
text lowered clarity** (-0.37 in the main session, -0.23 for subagents, the latter interval crossing
zero). Judges mostly complained about missing content. So the shipped texts spell out the rules and
never say "ASD-STE100". The README, CHANGELOG, and a comment in the hook script name the standard
and record the reason.

## What Karpathy suggests

The thread (2 Oct) is about understanding LLM output as more work moves to oversight. Four tips,
each framed as "but even better":

1. **Writing.** "Ask your LLM to explain something in ASD-STE100 ... it comes with heavy constraints
   on clean writing style that I often find a lot more readable." Soften it with "80% of the way to
   ASD-STE100" because the spec is stringent.
2. **Diagrams.** Ask for a diagram instead of prose.
3. **Web pages.** Ask for output "in HTML".
4. **Explainer videos.** Custom video explainers with text-to-speech narration.

Tip 1 is the direct point of contact: Karpathy names the standard and lets the model's knowledge of
it do the work. The PR tried exactly that and rejected it. Tips 2 to 4 change the output medium, not
the prose, and are outside what a writing rule can do. Tip 2 is cheap enough to test as a prompt
variant, so it is included below. Tips 3 and 4 are not tested here.

## The experiment

Seven variants of the appended instruction, four writing tasks, five repetitions each. Every run is
one headless Claude Code call (`claude -p`, model `claude-opus-5-5`, tools disabled, a fresh
temporary working directory so no `CLAUDE.md` leaks in, no settings files). The variant text is
appended to the default Claude Code system prompt. A blind judge (also `claude-opus-5-5`, with its own
system prompt and a JSON schema) scores each text for clarity, accuracy, completeness, and
actionability (1 to 5), lists which planted facts the text states, and lists which traps it states as
fact. The judge never sees the variant.

| Variant | Source | Text appended |
|---|---|---|
| V0 no rules | control | (nothing) |
| V1 PR one-line agent rule | PR #304, shipped | the agent rule above, verbatim |
| V2 PR hook text | PR #304, shipped | the hook text above, verbatim |
| V3 PR hook text + names ASD-STE100 | PR #304, eval round 5, rejected | the hook text with "based on ASD-STE100 Simplified Technical English" in the first sentence |
| V4 "Write in ASD-STE100" | Karpathy | `Write in ASD-STE100 (Simplified Technical English).` |
| V5 "80% of the way" | Karpathy | `Write 80% of the way to ASD-STE100 (Simplified Technical English).` |
| V6 prefer a diagram | Karpathy | `Prefer a diagram over prose. When the content has a structure, a sequence, or dependencies, show it as a Mermaid or plain-text diagram, and keep the prose around it short.` |

The tasks mirror the PR's eval cases ([`tasks.py`](tasks.py)): a PR description for a retry-default
change in a small HTTP client, a Jira ticket for a config loader whose strict mode became the
default, the PR's exact chat prompt ("explain what this change does and what could break for people
who already use the config loader"), and a documentation review with six planted issues. Each task
carries six facts the source supports and three or four traps (claims it does not support, such as a
motive the ticket never gives).

## Results

### Opus 5.5 writer, blind Opus 5.5 judge (140 runs)

Judge scores, mean over 20 runs per variant, with 95% bootstrap intervals in the notebook:

| Variant | Clarity | Accuracy | Completeness | Actionability | Facts recalled |
|---|---|---|---|---|---|
| V0 no rules (control) | 4.80 | 4.35 | 4.90 | 4.95 | 0.97 |
| V1 PR one-line agent rule | 4.85 | 4.55 | 4.95 | 5.00 | 0.98 |
| V2 PR hook text | 4.90 | **4.70** | 5.00 | 5.00 | 1.00 |
| V3 PR hook text + names ASD-STE100 | 4.90 | **4.75** | 5.00 | 5.00 | 1.00 |
| V4 "Write in ASD-STE100" | 4.85 | 4.30 | 4.95 | 4.95 | 0.98 |
| V5 "80% of the way to ASD-STE100" | 4.90 | 4.35 | 4.85 | 4.95 | 0.97 |
| V6 prefer a diagram | 4.55 | 4.25 | 4.90 | 4.95 | 0.98 |

Paired difference from the control on accuracy, with 95% intervals: V1 +0.20 [-0.05, +0.45],
V2 +0.35 [0.00, +0.70], V3 +0.40 [+0.05, +0.75], V4 -0.05 [-0.40, +0.30], V5 0.00 [-0.35, +0.35],
V6 -0.10 [-0.40, +0.20]. Every clarity, completeness, and actionability difference has an interval
that includes zero.

Surface style of the prose (code removed), means over 20 runs:

| Variant | Words | Words per sentence | Sentences over 20 words | Passive hits per 100 sentences | Contractions | Hedge words |
|---|---|---|---|---|---|---|
| V0 no rules (control) | 310 | 9.6 | 6.5% | 9.9 | 2.5 | 0.9 |
| V1 PR one-line agent rule | 277 | 8.9 | 2.7% | 1.5 | 0.0 | 0.6 |
| V2 PR hook text | 340 | 9.5 | 4.1% | 2.6 | 0.0 | 2.5 |
| V3 PR hook text + names ASD-STE100 | 340 | 8.9 | 3.3% | 2.4 | 0.0 | 1.9 |
| V4 "Write in ASD-STE100" | 280 | 8.6 | 2.4% | 1.6 | 0.0 | 0.2 |
| V5 "80% of the way to ASD-STE100" | 330 | 8.6 | 3.5% | 3.8 | 0.0 | 0.3 |
| V6 prefer a diagram | 273 | 9.8 | 9.0% | 10.6 | 2.4 | 0.6 |

Semicolons and Latin abbreviations were near zero for every variant, including the control.
Runs that stated a trap as fact: control 5 of 20 (all on the Jira ticket), V1 3, V2 0, V3 0,
V4 1, V5 2, V6 4. Runs where every judgment was marked as one: control 60%, V1 70%, V2 85%,
V3 90%, V4 60%, V5 45%, V6 65%. The diagram variant drew a diagram in 65% of runs. Two of 20
replies under V1 and two under V4 arrived wrapped whole in a ```` ```markdown ```` fence.

<!-- SONNET_README -->

### Takeaways

<!-- TAKEAWAYS -->


## Files

| File | Purpose |
|---|---|
| [`variants.py`](variants.py) | The seven instruction texts (PR texts verbatim) |
| [`tasks.py`](tasks.py) | The four tasks with their facts and traps |
| [`claude_cli.py`](claude_cli.py) | Wrapper for headless `claude -p` |
| [`run_variants.py`](run_variants.py) | Generates the 140 outputs (resumable) |
| [`judge.py`](judge.py) | Blind judge with a JSON schema (resumable) |
| [`metrics.py`](metrics.py) | Deterministic style metrics (sentence length, semicolons, passive heuristic, ...) |
| [`analyze.py`](analyze.py), [`plots.py`](plots.py) | Tables, bootstrap intervals, charts |
| [`build_notebook.py`](build_notebook.py) | Builds and executes the notebook |
| `results/generations.jsonl`, `results/judgments.jsonl` | Raw outputs and judge records |

## Reproduce

```bash
cd writing_style_variants
uv sync
uv run python run_variants.py --reps 5 --workers 6
uv run python judge.py --workers 6
uv run python build_notebook.py
```

Both runners skip cells that are already in `results/`. The runs here cost about GEN_COST for
generation and JUDGE_COST for judging.
