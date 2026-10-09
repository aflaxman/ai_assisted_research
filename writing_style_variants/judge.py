#!/usr/bin/env python
"""Blind LLM judge for the generated texts.

The judge sees the task, the fact list, the trap list, and the text. It does
not see which variant produced the text. Scores follow the dimensions that
PR #304 reports (clarity, accuracy, actionability) plus completeness and a
per-fact coverage list. Output is constrained with a JSON schema.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from claude_cli import call_claude, summarize_usage
from tasks import TASKS

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()

JUDGE_SYSTEM = """You are a blind judge of technical prose written for software engineers. You will see the task a writer was given (which includes the source material), a numbered list of facts the source supports, a numbered list of traps (claims the source does not support), and the writer's text. You do not know who wrote the text or what instructions they had. Judge only the text.

Scores are integers from 1 (poor) to 5 (excellent):
- clarity: a busy engineer who has not read the source understands the text on one reading. Direct, unambiguous sentences and a structure that matches the task score high. Length by itself does not change the score: a short text that says what the reader needs is as clear as a long one.
- accuracy: every claim is supported by the source. A guess or a judgment that is labelled as one does not count against accuracy. A trap stated as fact, or an unsupported claim, lowers the score.
- completeness: the text gives the reader the listed facts that the task calls for, at the detail the task needs. For a review task this is the share of planted issues that the text reports.
- actionability: the reader knows what the change means for them or what to do next, without a follow-up question.

Also report:
- facts_covered: the numbers of the listed facts that the text states correctly (a paraphrase counts; a partial statement counts only if the key point is there).
- traps_hit: the numbers of the listed traps that the text states as fact.
- unsupported_claims: how many other claims in the text the source does not support (0 if none).
- judgments_marked: true if every guess or judgment in the text is labelled as one, or if the text has none.
- notes: one or two sentences on the main strength and the main weakness.
Return only the JSON object."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "actionability": {"type": "integer", "minimum": 1, "maximum": 5},
        "facts_covered": {"type": "array", "items": {"type": "integer"}},
        "traps_hit": {"type": "array", "items": {"type": "integer"}},
        "unsupported_claims": {"type": "integer", "minimum": 0},
        "judgments_marked": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "clarity", "accuracy", "completeness", "actionability",
        "facts_covered", "traps_hit", "unsupported_claims", "judgments_marked", "notes",
    ],
}


def judge_prompt(task: str, output: str) -> str:
    t = TASKS[task]
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(t["facts"]))
    traps = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(t["traps"]))
    return (
        "## Task given to the writer\n\n" + t["prompt"].strip() + "\n\n"
        "## Facts the source supports\n\n" + facts + "\n\n"
        "## Traps (claims the source does not support)\n\n" + traps + "\n\n"
        "## The writer's text\n\n<<<BEGIN TEXT>>>\n" + output.strip() + "\n<<<END TEXT>>>\n"
    )


def key_of(r: dict) -> str:
    return f"{r['variant']}|{r['task']}|{r['rep']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--inp", default=str(HERE / "results" / "generations.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results" / "judgments.jsonl"))
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    gens = [json.loads(l) for l in inp.read_text().splitlines() if l.strip()]
    done = set()
    if out.exists():
        done = {key_of(json.loads(l)) for l in out.read_text().splitlines() if l.strip()}
    todo = [g for g in gens if key_of(g) not in done]
    print(f"{len(done)} judged, {len(todo)} to judge", flush=True)

    total = 0.0
    t0 = time.time()

    def do(g):
        for attempt in range(3):
            try:
                data = call_claude(
                    judge_prompt(g["task"], g["output"]),
                    args.model,
                    system_replace=JUDGE_SYSTEM,
                    json_schema=JUDGE_SCHEMA,
                )
                so = data.get("structured_output")
                if not isinstance(so, dict):
                    so = json.loads(data["result"])
                n_facts = len(TASKS[g["task"]]["facts"])
                rec = {
                    "variant": g["variant"], "task": g["task"], "rep": g["rep"],
                    "judge_model": args.model,
                    **so,
                    "n_facts": n_facts,
                    "facts_recall": len({i for i in so["facts_covered"] if 1 <= i <= n_facts}) / n_facts,
                    "judge_" + "cost_usd": data.get("total_cost_usd"),
                    "judge_wall_s": data.get("_wall_s"),
                }
                with LOCK:
                    with out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1} for {key_of(g)}: {e}", flush=True)
                traceback.print_exc()
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"gave up on {key_of(g)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(do, g) for g in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            total += rec["judge_cost_usd"] or 0
            print(
                f"[{i}/{len(todo)}] {rec['variant']}/{rec['task']}/{rec['rep']} "
                f"clarity={rec['clarity']} acc={rec['accuracy']} compl={rec['completeness']} "
                f"recall={rec['facts_recall']:.2f} (${total:.2f}, {time.time() - t0:.0f}s)",
                flush=True,
            )
    print(f"done: ${total:.2f}", flush=True)


if __name__ == "__main__":
    main()
