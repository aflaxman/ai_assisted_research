#!/usr/bin/env python
"""Generate one output per (variant, task, rep) with headless Claude Code.

Resumable: cells already in the output file are skipped. The first call of each
variant runs alone so the later calls read its system prompt from the cache.
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
from variants import VARIANTS

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add(f"{r['variant']}|{r['task']}|{r['rep']}")
    return keys


def run_cell(variant: str, task: str, rep: int, model: str, out_path: Path) -> dict:
    v = VARIANTS[variant]
    t = TASKS[task]
    data = call_claude(t["prompt"], model, system_append=v["text"])
    rec = {
        "variant": variant,
        "task": task,
        "rep": rep,
        "model": model,
        "system_append": v["text"],
        "output": data.get("result", ""),
        **summarize_usage(data),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with LOCK:
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--out", default=str(HERE / "results" / "generations.jsonl"))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    cells = [
        (v, t, r)
        for v in args.variants
        for t in args.tasks
        for r in range(args.reps)
        if f"{v}|{t}|{r}" not in done
    ]
    print(f"{len(done)} cells done, {len(cells)} to run", flush=True)

    # Phase 1: warm the prompt cache with one call per variant (in parallel across variants).
    warm = {}
    for v, t, r in cells:
        warm.setdefault(v, (v, t, r))
    rest = [c for c in cells if c not in set(warm.values())]
    total_cost = 0.0
    t0 = time.time()

    def do(cell):
        v, t, r = cell
        for attempt in range(3):
            try:
                return run_cell(v, t, r, args.model, out_path)
            except Exception as e:  # noqa: BLE001
                print(f"  retry {attempt + 1} for {v}/{t}/{r}: {e}", flush=True)
                traceback.print_exc()
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"gave up on {cell}")

    for phase, batch in (("warm-up", list(warm.values())), ("main", rest)):
        if not batch:
            continue
        print(f"== {phase}: {len(batch)} calls", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(do, c): c for c in batch}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                total_cost += rec["cost_usd"] or 0
                print(
                    f"[{i}/{len(batch)}] {rec['variant']}/{rec['task']}/{rec['rep']} "
                    f"${rec['cost_usd']:.3f} out={rec['output_tokens']} think={rec['thinking_tokens']} "
                    f"wall={rec['wall_s']:.0f}s  (total ${total_cost:.2f}, {time.time() - t0:.0f}s)",
                    flush=True,
                )
    print(f"done: ${total_cost:.2f} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
