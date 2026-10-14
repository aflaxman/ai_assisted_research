#!/bin/bash
# Reproduce every result in README.md: main runs (seed 0) and 10 replication seeds.
# Runs sequentially on purpose: two JAX processes on one machine oversubscribe the CPU.
set -e
cd "$(dirname "$0")"
mkdir -p logs
uv run python speed_accuracy.py --out results/study1 2>&1 | grep -v "^warning" | tee logs/study1.log
uv run python risk_sem.py --out results/study2 2>&1 | grep -v "^warning" | tee logs/study2.log
for seed in 1 2 3 4 5 6 7 8 9 10; do
  uv run python speed_accuracy.py --seed $seed --out results/study1_reps/seed$seed 2>&1 | grep -v "^warning" | tee logs/study1_seed$seed.log
  uv run python risk_sem.py --seed $seed --out results/study2_reps/seed$seed 2>&1 | grep -v "^warning" | tee logs/study2_seed$seed.log
done
uv run python aggregate_reps.py
uv run python make_figures.py
