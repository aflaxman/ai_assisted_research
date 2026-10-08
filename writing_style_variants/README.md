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

### Sonnet 5.5 writer, same Opus 5.5 judge (100 runs, five variants)

The PR's review agents declare `model: sonnet`, so the control, the two shipped PR texts, the rejected
named variant, and Karpathy's one-liner were rerun on `claude-sonnet-5-5`.

| Variant | Clarity | Accuracy | Completeness | Facts recalled | Words per sentence | Sentences over 20 words | Passive hits per 100 sentences | Cost per run |
|---|---|---|---|---|---|---|---|---|
| V0 no rules (control) | 4.95 | 4.55 | 4.50 | 0.92 | 11.5 | 12.0% | 14.8 | $0.018 |
| V1 PR one-line agent rule | 4.95 | 4.65 | **4.85** | 0.97 | 8.5 | 1.4% | 5.4 | $0.022 |
| V2 PR hook text | 4.90 | **4.90** | 4.80 | 0.96 | 8.6 | 1.9% | 4.1 | $0.025 |
| V3 PR hook text + names ASD-STE100 | 4.90 | **4.85** | 4.75 | 0.96 | 7.9 | 1.2% | 2.2 | $0.027 |
| V4 "Write in ASD-STE100" | 4.95 | 4.45 | 4.80 | 0.97 | 7.5 | 0.5% | 3.6 | $0.023 |

Paired differences from the Sonnet control: accuracy V1 +0.10 [-0.25, +0.45], V2 +0.35 [+0.10, +0.60],
V3 +0.30 [+0.05, +0.55], V4 -0.10 [-0.40, +0.20]; completeness V1 +0.35 [+0.05, +0.65],
V2 +0.30 [+0.10, +0.50], V3 +0.25 [-0.05, +0.55], V4 +0.30 [+0.05, +0.55]. No Sonnet reply was
wrapped in a code fence.

### Takeaways

**Clarity is at the ceiling on Opus 5.5, with or without rules.** The control scored 4.80 on clarity and 4.90 on completeness. Every rule variant landed between 4.85 and 4.90, and every paired difference from the control has an interval that includes zero. The +0.7 clarity gains the PR reports do not reproduce in this setup. The control already writes sentences of 9.6 words on average, with 6.5% over 20 words.

**The rules do change the surface of the text, and all of them do it about equally.** Passive constructions fall from about 10 per 100 sentences to 1.5 to 4. Contractions, semicolons, and Latin abbreviations go to zero. The Flesch-Kincaid grade drops from 5.1 to about 4. Karpathy's one-liner, "Write in ASD-STE100", does this as well as the PR's spelled-out rules, and gives the shortest sentences of all (8.6 words, 2.4% over 20 words). On surface style there is no measurable difference between naming the standard and spelling out its rules.

**Accuracy is the one score that separates the variants, and it tracks one clause.** The hook text says: "A guess or a judgment is marked as one, and a reason that the source does not give is not stated as fact." The two texts that carry this clause (V2, V3) scored 4.70 and 4.75 on accuracy against 4.35 for the control, a paired gain of +0.35 to +0.40 with intervals that touch zero (V2) or just clear it (V3). They stated zero traps as fact in 40 runs, where the control did so in 5 of 20 runs, all on the Jira ticket, and they marked their judgments in 85% to 90% of runs against 60%. The one-line agent rule, which has no such clause, gained +0.20 with an interval that crosses zero. Karpathy's two variants gained nothing (-0.05 and 0.00) and produced the most unsupported claims per text (1.0 and 1.2, control 0.85). In the chat task their texts state guesses about users as fact ("many users call `load(path)`", "few callers do this") and give the intent of the change as fact, where the diff gives none. The Jira ticket shows the effect most plainly: the control declared the new behavior a bug or an intended change as fact (accuracy 3.4), while the hook text marked it as the changelog's position and the team's call (4.6). This matches the PR's own eval, where a text without the clause lost 0.53 on accuracy.

**Naming ASD-STE100 did not hurt on Opus 5.5.** V3, the PR's rejected variant that names the standard, matched V2 on every score. V4, Karpathy's bare "Write in ASD-STE100", matched the one-line agent rule on clarity, completeness, and recall. Recall was 0.97 or better for every variant, so there was no missing content for a judge to complain about. Two explanations are open: the model (the PR's review agents declare `model: sonnet`, and its subagent evals ran there), or the tasks (small and single-draft). The Sonnet section above tests the first.

**On Sonnet 5.5 the picture is the same, and the surface effects are larger.** The PR's review agents declare `model: sonnet`, so five variants were rerun there (100 runs, same Opus judge). Sonnet's control writes longer sentences than Opus's (11.5 words, 12% over 20 words, 15 passive hits per 100 sentences), so every rule moves the surface further: "Write in ASD-STE100" gives 7.5-word sentences with 0.5% over 20 words, the named hook text 7.9, the one-line rule 8.5, the hook text 8.6, and passive hits fall to 2 to 5 per 100 sentences. Clarity is again at the ceiling for every variant (4.90 to 4.95). Accuracy again follows the clause about guesses: the hook text gains +0.35 [+0.10, +0.60] over the control and the named hook text +0.30 [+0.05, +0.55], with unsupported claims cut from 0.45 to 0.15 per text and judgments marked in 95% to 100% of runs. The one-line rule gains +0.10 and "Write in ASD-STE100" loses 0.10. The one new result is completeness. Sonnet's control drops facts that Opus's control kept (recall 0.92; it often leaves out that `strict=False` keeps the old behavior, and that the docstring has no `Raises` section), and every rule variant recovers them: completeness +0.25 to +0.35, recall 0.96 to 0.97. Naming ASD-STE100, alone or added to the hook text, caused no content loss on Sonnet 5.5 either. The PR's finding that naming the standard made output less clear and less complete did not reproduce in this setup on either model. The likeliest reason is the tasks. The PR's evals used review workflows on small repositories with tools, where the agent reads code before it writes, and a terse instruction may cost more there than on a single-draft writing task. One cost to note: on Sonnet, rules make the model think and write more. Output tokens, thinking included, rise from about 950 per run for the control to 1,400 to 1,500 for the one-line rule and "Write in ASD-STE100", and to 1,700 to 1,900 for the hook texts. Cost per run goes from $0.018 to between $0.022 and $0.027, and wall time from 11 s to between 14 and 17 s. On Opus the same rules added about 5%.

**Two side effects worth knowing.** First, 2 of 20 replies under the one-line agent rule and 2 of 20 under "Write in ASD-STE100" arrived wrapped whole in a ```` ```markdown ```` code fence, which a PR or ticket would show as raw text. The hook text never did this; its wording ("Prose for people in this project is ...") describes a state rather than ordering a format. Second, the hook text makes replies longer (340 words against 310) and adds hedge words (2.5 per text against 0.9). That is the cost of marked judgments, and the judge did not count it against clarity.

**"Prefer a diagram" is a different axis.** The diagram variant drew one in 65% of runs and left the prose metrics where the control had them (passive 10.6 per 100 sentences, contractions 2.4). Clarity fell by 0.25, with the Jira ticket worst at 4.0: a flow diagram in a bug report does not help the reader who needs steps, expected, and actual. In the chat task, where the change has a before and after, the diagram and the before-and-after table both read well. Diagrams, HTML pages, and videos are choices of medium. They sit beside a writing rule, not in place of it, and they are the right tool for an explanation and the wrong one for a ticket.

**Cost.** Every variant costs about $0.043 to $0.046 per run at this prompt size. The hook text adds about 5% more output and thinking tokens.

**What this suggests for PR #304.**

1. The clause about guesses and unstated reasons is the measurable win in the hook text. It is worth keeping exactly as written.
2. The one-line agent rule does not have that clause. Review findings are where an unsupported claim costs the most, so the agents might gain from a second sentence such as "Mark a guess or a judgment as one, and do not state a reason the source does not give." On Sonnet, where the agents run, the one-line rule already raised completeness by +0.35 and recall from 0.92 to 0.97, so the shipped rule does useful work beyond style there.
3. The decision not to name the standard cost nothing and bought nothing here, on Opus 5.5 or on Sonnet 5.5. The PR's evidence for it came from its own review-workflow evals, and this test does not contradict that evidence on its own ground. It does mean the rule need not be treated as general: in the main session, the hook text with the standard named (V3) performed the same as the shipped text (V2). Leaving the name out is harmless, so the choice can rest on the PR's evals.
4. Karpathy's one-liner is a good personal prompt for readability. It is not a substitute for the PR's text, because it does nothing for accuracy, and it sometimes returns the whole reply as a code block.

**Limits.** Five repetitions per cell, four synthetic tasks, a Sonnet rerun that covers five of the seven variants, one judge (Opus 5.5, the same family as the writer, with criteria close to the rules under test), rules placed in the system prompt where the real hook injects session context, tools disabled, and the default Claude Code system prompt present but without its tool sections. Differences whose interval crosses zero are noise at this sample size.


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

Both runners skip cells that are already in `results/`. The runs here cost $8.48 for generation ($6.17 on Opus 5.5, $2.31 on Sonnet 5.5) and $9.99 for
judging (240 judge calls on Opus 5.5).
