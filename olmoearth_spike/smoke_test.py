"""M1 smoke test: does OlmoEarth actually load and run on this box?

  A. BASE variant (89M params) — the variant a funded project would use. Load
     from Hugging Face, run one forward, pool to an embedding. Times the heavy
     path on CPU (a core datapoint for the go/no-go budget question).
  B. NANO variant (1.4M) — smallest custom call; embed twice to check
     determinism.

Logs wall-clock, peak RSS, thread count, and CPU-vs-GPU.

NOTE: inputs here are random noise — this checks plumbing, not signal.
"""

import resource
import time

import numpy as np
import torch

from olmoearth_pretrain_minimal import ModelID
from olmoe import embed_sample, load_encoder, make_s2_sample


def peak_rss_mb() -> float:
    """Process peak resident set size in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rand_chip(h, w, t, seed):
    return np.random.default_rng(seed).random((h, w, t, 12)).astype(np.float32)


def main():
    print("=" * 68)
    print("ENVIRONMENT")
    print("=" * 68)
    print(f"torch              {torch.__version__}")
    print(f"CUDA available     {torch.cuda.is_available()}")
    print(f"device             {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"torch num_threads  {torch.get_num_threads()}")
    print(f"peak RSS start     {peak_rss_mb():.0f} MB")

    # ---- Part A: BASE, the real-project variant ----
    print("\n" + "=" * 68)
    print("PART A - BASE (89M params), 128x128x12 chip")
    print("=" * 68)
    t0 = time.perf_counter()
    base = load_encoder(ModelID.OLMOEARTH_V1_BASE)
    print(f"load+download      {time.perf_counter() - t0:.1f} s")
    sample = make_s2_sample(rand_chip(128, 128, 12, seed=0))
    t1 = time.perf_counter()
    emb = embed_sample(base, sample)
    print(f"forward+pool       {time.perf_counter() - t1:.1f} s")
    print(f"embedding          shape={emb.shape} dtype={emb.dtype}")
    print(f"peak RSS after     {peak_rss_mb():.0f} MB")
    del base

    # ---- Part B: NANO, smallest call + determinism ----
    print("\n" + "=" * 68)
    print("PART B - NANO (1.4M params), 64x64x6 chip, determinism")
    print("=" * 68)
    t0 = time.perf_counter()
    nano = load_encoder(ModelID.OLMOEARTH_V1_NANO)
    print(f"load+download      {time.perf_counter() - t0:.1f} s")
    small = make_s2_sample(rand_chip(64, 64, 6, seed=42))
    t1 = time.perf_counter()
    e1 = embed_sample(nano, small)
    dt = time.perf_counter() - t1
    e2 = embed_sample(nano, small)  # identical input -> determinism
    max_abs_diff = float(np.abs(e1 - e2).max())
    print(f"forward+pool       {dt:.2f} s")
    print(f"embedding          shape={e1.shape} dtype={e1.dtype}")
    print(f"embedding[:5]      {np.round(e1[:5], 4).tolist()}")
    print(f"determinism max|d| {max_abs_diff:.2e} -> "
          f"{'DETERMINISTIC' if max_abs_diff == 0 else 'nondeterministic'}")
    print(f"peak RSS final     {peak_rss_mb():.0f} MB")
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
