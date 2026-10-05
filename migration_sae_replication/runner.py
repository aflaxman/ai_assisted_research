"""Chunked, resumable task runner shared by simulation.py and multinomial.py.

Tasks run in chunks, each chunk in a fresh process pool (so compiled JAX kernels and any
leaked JIT memory die with the workers), and each chunk's rows are written to
results/raw/<study>/partNNN.parquet before the next chunk starts. Re-running skips
chunks whose part file exists and finally concatenates all parts.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def run_tasks(tasks, out: Path, workers: int, chunk: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = out.with_suffix("")
    parts.mkdir(exist_ok=True)
    n_chunks = math.ceil(len(tasks) / chunk)
    t0 = time.time()
    for ci in range(n_chunks):
        part = parts / f"part{ci:03d}.parquet"
        if part.exists():
            print(f"chunk {ci + 1}/{n_chunks} already done, skipping", flush=True)
            continue
        sub = tasks[ci * chunk:(ci + 1) * chunk]
        frames, diags = [], []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(fn, *a) for fn, a in sub]
            for f in as_completed(futs):
                df, dg = f.result()
                frames.append(df)
                diags.extend(dg)
        pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
        pd.DataFrame(diags).to_csv(part.with_suffix(".diagnostics.csv"), index=False)
        done = min((ci + 1) * chunk, len(tasks))
        print(f"chunk {ci + 1}/{n_chunks} done ({done}/{len(tasks)} tasks), {time.time() - t0:.0f}s", flush=True)
    res = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part*.parquet"))], ignore_index=True)
    res.to_parquet(out, index=False)
    diag = pd.concat([pd.read_csv(p) for p in sorted(parts.glob("part*.diagnostics.csv"))], ignore_index=True)
    diag.to_csv(out.with_suffix(".diagnostics.csv"), index=False)
    print(f"wrote {out} ({len(res)} rows) in {time.time() - t0:.0f}s", flush=True)
    return res
