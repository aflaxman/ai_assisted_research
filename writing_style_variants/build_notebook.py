#!/usr/bin/env python
"""Build variants_notebook.ipynb from the results and execute it so the outputs are saved."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "variants_notebook.ipynb"


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


DISCUSSION_PATH = HERE / "DISCUSSION.md"
DISCUSSION = DISCUSSION_PATH.read_text() if DISCUSSION_PATH.exists() else "_Discussion to be written after the results are in._"

cells = [
    md("""
# Plain-English writing rules for Claude Code: how the variants pan out

[vivarium-suite PR #304](https://github.com/ihmeuw/vivarium-suite/pull/304) adds plain-English writing rules, based on ASD-STE100 Simplified Technical English, to the `simsci` plugins: a one-line rule in nine agent definitions and a 427-character `SessionStart` hook text for the main session. The PR's own evals found that *naming* the standard in the text made output less clear and less complete, so the shipped text spells out the rules and does not say "ASD-STE100".

[Karpathy's thread](https://twitter-thread.com/t/2105819303471976479) (2 Oct) suggests the opposite shortcut: just ask the model to "explain something in ASD-STE100", or soften it to "80% of the way to ASD-STE100". He also suggests asking for a diagram, an HTML page, or a video instead of prose.

This notebook runs both families of instruction, plus the PR's rejected "named" variant, through the same four writing tasks and compares the outputs: side by side, with surface style metrics, and with a blind LLM judge.

**Setup.** Each cell is one headless Claude Code call (`claude -p`, model `claude-opus-5-5`, tools disabled, fresh temporary working directory, no settings files). The variant text is appended to the default Claude Code system prompt. Seven variants x four tasks x five repetitions = 140 runs. A blind judge (also `claude-opus-5-5`, with its own system prompt and a JSON schema) scores each text on clarity, accuracy, completeness, and actionability (1 to 5), lists which planted facts the text states, and lists which traps it states as fact. The judge never sees the variant name.

**Read the numbers with care.** Five repetitions per cell give wide intervals. The tasks are small and synthetic. The judge is a model, and its criteria overlap with the rules under test (the PR notes the same limit). Treat differences whose interval crosses zero as noise.
"""),
    code("""
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 220)

from analyze import COST_COLS, JUDGE_COLS, PRETTY, STYLE_COLS, delta_vs, load_all, summarize, wide_table
from plots import bar_panels, dot_plot, heatmap
from tasks import ORDER as TORDER
from tasks import TASKS
from variants import ORDER as VORDER
from variants import VARIANTS

df = load_all()
print(f"{len(df)} runs: {df.variant.nunique()} variants x {df.task.nunique()} tasks x {df.rep.nunique()} reps, model {df.model.iloc[0]}")
print(f"generation cost ${df.cost_usd.sum():.2f}" + (f", judge cost ${df.judge_cost_usd.sum():.2f}" if "judge_cost_usd" in df else ""))
"""),
    md("## The variants\n\nV1 and V2 are copied verbatim from the PR. V3 is the PR's eval-round-5 text that names the standard. V4 to V6 follow Karpathy's thread."),
    code("""
pd.DataFrame(
    [{"variant": k, "label": v["label"], "source": v["source"], "chars": len(v["text"] or ""), "text appended to the system prompt": v["text"] or "(nothing)"} for k, v in VARIANTS.items()]
).set_index("variant")
"""),
    md("## The tasks\n\nThe tasks mirror the PR's eval cases: a PR description, a Jira ticket, the PR's exact chat prompt, and a documentation review with planted issues. Each task carries a list of facts the source supports and a list of traps (claims the source does not support). The writer sees only the prompt; the judge sees everything."),
    code("""
pd.DataFrame([{"task": k, "title": t["title"], "facts": len(t["facts"]), "traps": len(t["traps"])} for k, t in TASKS.items()]).set_index("task")
"""),
    md("""
## Blind judge scores by variant

Means over 4 tasks x 5 reps, with 95% bootstrap intervals. `Facts recalled` is the share of the task's planted facts that the judge found stated correctly in the text.
"""),
    code("""
wide_table(df, JUDGE_COLS)
"""),
    code("""
s = summarize(df, JUDGE_COLS)
fig = dot_plot(
    s, JUDGE_COLS,
    "Blind judge scores by variant (mean and 95% bootstrap CI over 4 tasks x 5 reps)",
    xlim={c: (0.8, 5.4) for c in JUDGE_COLS[:-1]} | {"facts_recall": (0, 1.1)},
)
"""),
    md("### Difference from the control (no rules), paired by task and rep\n\nA positive number means the variant scored higher than the control on the same task and repetition."),
    code("""
d = delta_vs(df, JUDGE_COLS, ref="none")
d["cell"] = d.apply(lambda r: f"{r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]", axis=1)
w = d.pivot(index="label", columns="metric", values="cell")[JUDGE_COLS]
w.columns = [PRETTY[c] for c in w.columns]
w.index.name = None
w.loc[[VARIANTS[v]["label"] for v in VORDER if v != "none"]]
"""),
    md("""
## Surface style metrics

Deterministic counts on the prose (code blocks and inline code removed). ASD-STE100 asks for sentences of at most 20 words in procedures (25 in descriptions), no semicolons, and the active voice. `Passive hits` is a regex heuristic (a form of *be* followed by a past participle), so read it as a rough rate, not a count of true passives. `Hedge words` counts words such as *might*, *likely*, *guess*, *judgment*; the PR hook text asks that guesses be marked, so a higher count is not bad by itself.
"""),
    code("""
wide_table(df, STYLE_COLS, digits=1)
"""),
    code("""
s2 = summarize(df, STYLE_COLS)
fig = bar_panels(
    s2, ["mean_sentence_words", "pct_sentences_over_20", "passive_per_100_sentences", "words"],
    "Surface style metrics by variant (mean and 95% CI)",
)
"""),
    md("## Recall of planted facts, by task\n\nThe PR's judges complained that texts which named ASD-STE100 were missing content. This grid shows where content goes missing: the share of each task's planted facts that the judge found in the text."),
    code("""
piv = df.pivot_table(index="variant_label", columns="task", values="facts_recall", aggfunc="mean", observed=True)
piv = piv.loc[[VARIANTS[v]["label"] for v in VORDER]]
piv.columns = [TASKS[t]["title"] for t in piv.columns]
fig = heatmap(piv, "Share of planted facts the text states, by task and variant")
"""),
    md("## Accuracy details: traps, unsupported claims, and marked judgments\n\n`Traps stated as fact` counts claims the source does not support that the text states anyway (for example a motive for a change that the ticket does not give). `Judgments marked` is the share of runs where every guess or judgment in the text was labelled as one. The last column counts a formatting quirk: the whole reply delivered inside a ```` ```markdown ```` code fence, which renders as raw text in a PR or ticket."),
    code("""
acc = wide_table(df, ["traps_hit_n", "unsupported_claims"], digits=2)
acc["Judgments marked (share of runs)"] = df.groupby("variant_label", observed=True)["judgments_marked"].mean().round(2).reindex(acc.index)
acc["Diagram present (share of runs)"] = df.groupby("variant_label", observed=True)["has_diagram"].mean().round(2).reindex(acc.index)
acc["Whole reply wrapped in a code fence (share of runs)"] = df.groupby("variant_label", observed=True)["wrapped_in_code_fence"].mean().round(2).reindex(acc.index)
acc
"""),
    md("## Cost, length, and time per run\n\nInput tokens are nearly constant (about 3.5K, mostly cached after the first call of each variant), so cost tracks output and thinking tokens."),
    code("""
wide_table(df, COST_COLS, digits=3)
"""),
    md("""
## Side-by-side outputs

Repetition 0 of every variant for each task, in variant order. Nothing is cherry-picked. The line under each heading gives the word count, the mean sentence length, and the blind judge's scores for that exact text.
"""),
    code("""
def show_task(task):
    t = TASKS[task]
    display(Markdown(f"## {t['title']}\\n\\n**Prompt given to the writer:**"))
    print(t["prompt"])
    display(Markdown("**Facts the judge checks:**\\n\\n" + "\\n".join(f"{i + 1}. {f}" for i, f in enumerate(t["facts"]))))
    sub = df[(df.task == task) & (df.rep == 0)]
    for _, r in sub.iterrows():
        line = f"{r.words} words, {r.mean_sentence_words:.0f} words per sentence, {r.output_tokens} output tokens, ${r.cost_usd:.3f}"
        if "clarity" in r.index and pd.notna(r.clarity):
            line += (f". Judge: clarity {int(r.clarity)}, accuracy {int(r.accuracy)}, completeness {int(r.completeness)}, "
                     f"actionability {int(r.actionability)}, facts {sorted(r.facts_covered)} of {len(t['facts'])}, traps {sorted(r.traps_hit)}")
        display(Markdown(f"### {r.variant_label}\\n\\n*{line}*"))
        display(Markdown(r.output))
        if "notes" in r.index and isinstance(r.notes, str):
            display(Markdown(f"> **Judge notes:** {r.notes}"))
        display(Markdown("---"))
"""),
    code("show_task('chat_reply')"),
    code("show_task('pr_description')"),
    code("show_task('jira_ticket')"),
    code("show_task('doc_review')"),
    md("## Discussion\n\n" + DISCUSSION),
    md("""
## Reproduce

```bash
cd writing_style_variants
uv sync
uv run python run_variants.py --reps 5 --workers 6   # 140 headless Claude Code calls
uv run python judge.py --workers 6                   # 140 blind judge calls
uv run python build_notebook.py                      # rebuilds this notebook with outputs
```

`run_variants.py` and `judge.py` are resumable: cells already in `results/*.jsonl` are skipped.
"""),
]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-errors", action="store_true", help="keep going after a cell error (for dry runs)")
    args = ap.parse_args()
    out = Path(args.out)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    client = NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}, allow_errors=args.allow_errors
    )
    client.execute()
    nbf.write(nb, out)
    n_err = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
        if errs:
            n_err += 1
            print(f"cell {i}: {errs[0]['ename']}: {errs[0]['evalue'][:200]}  <- {c.source.splitlines()[0][:80]}")
    print(f"wrote {out} ({len(nb.cells)} cells, {n_err} cells with errors)")


if __name__ == "__main__":
    main()
