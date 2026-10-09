"""Thin wrapper around headless Claude Code (`claude -p`).

Every call runs in a fresh temporary directory so no CLAUDE.md or project
settings leak in, with tools disabled, so the model answers in one turn.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time


def call_claude(
    prompt: str,
    model: str,
    *,
    system_append: str | None = None,
    system_replace: str | None = None,
    json_schema: dict | None = None,
    max_budget_usd: float = 1.0,
    timeout_s: int = 900,
) -> dict:
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--tools", "",
        "--setting-sources", "",
        "--no-session-persistence",
        "--max-budget-usd", str(max_budget_usd),
    ]
    if system_append:
        cmd += ["--append-system-prompt", system_append]
    if system_replace:
        cmd += ["--system-prompt", system_replace]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]
    with tempfile.TemporaryDirectory(prefix="wsv-") as cwd:
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        wall = time.time() - t0
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"claude exited {proc.returncode} with no output: {proc.stderr[-1500:]}")
    data = json.loads(out)
    data["_wall_s"] = wall
    data["_returncode"] = proc.returncode
    return data


def summarize_usage(data: dict) -> dict:
    u = data.get("usage", {})
    return {
        "cost_usd": data.get("total_cost_usd"),
        "input_tokens": u.get("input_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
        "duration_api_ms": data.get("duration_api_ms"),
        "wall_s": data.get("_wall_s"),
        "num_turns": data.get("num_turns"),
        "stop_reason": data.get("stop_reason"),
        "is_error": data.get("is_error"),
    }
